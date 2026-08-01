"""CALCE integration with the existing scoring pipeline.

Runs CALCE cycling data through the same stages `main.py` uses — behaviour
flags, stress score, rolling and age features, risk, health index, RUL, Guardian
— by delegating to those modules unchanged. Nothing here reimplements a scoring
stage; a discrepancy between a CALCE run and a NASA run should be a property of
the data, not of two implementations.

The obstacle, and why it is not worked around
---------------------------------------------
`features.behavior_features.compute_behavior_flags` computes `high_temp_flag`
from `temperature_c` and `deep_discharge_flag` from `soc`.
`risk.stress_score` declares both in `REQUIRED_ROW_COLUMNS`.

**CS2 and CX2 Arbin exports contain neither.** The cells were cycled at room
temperature with no thermocouple, and Arbin reports accumulated charge rather
than state of charge.

There are three ways to handle that, and only one is honest.

1. Fill `temperature_c` with 23.0, the documented room temperature. The schema
   is satisfied, `high_temp_flag` is computed, and a thermal stress score is
   produced for a quantity nobody measured. This is the NaN-as-healthy defect
   with extra steps: a constant cannot raise a flag, so every CALCE cell would
   score as thermally unstressed regardless of what actually happened to it.

2. Derive `soc` by integrating current. Defensible for NASA, where the loader
   already does it against a known per-test capacity. For CALCE it would be
   circular here: the capacity being integrated toward is the fade target.

3. Refuse, and say which stages are unavailable and why.

This module does (3). `analyze_calce_cell` runs every stage the data supports
and reports the rest as unavailable with the missing channel named. A CS2 cell
therefore yields measured SOH and capacity fade — which is what CALCE is
actually good for — and no behavioural risk score, which it cannot support.

CX2_4 is the exception. With its thermocouple files joined it has a real
temperature channel, so the behavioural stages run. That asymmetry is the point:
the pipeline follows the instrumentation rather than the dataset label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.bms.io.load_calce_cycling import (
    CalceLoadReport,
    calce_capacity_loss,
    load_calce_cell,
    summarize_calce_cycles,
)

# Channels each downstream stage needs, and what it uses them for. Used to
# explain precisely which stage is blocked rather than emitting a bare KeyError.
STAGE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "behavior_flags": ("current_a", "temperature_c", "soc"),
    "risk_assessment": ("current_a", "temperature_c", "soc"),
    "health_index": ("current_a", "temperature_c", "soc"),
    "guardian": ("current_a", "temperature_c", "soc"),
}

CHANNEL_CONSUMERS: Mapping[str, str] = {
    "temperature_c": "high_temp_flag, temp_rolling_mean, thermal stress term",
    "soc": "deep_discharge_flag, high_soc_flag, depth-of-discharge term",
    "current_a": "fast_charge_flag, aggressive_discharge_event",
}


@dataclass(frozen=True)
class CalceAnalysis:
    """What a CALCE cell's analysis produced, and what it could not."""

    cell_id: str
    load_report: CalceLoadReport
    cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    guardian: pd.DataFrame = field(default_factory=pd.DataFrame)
    stages_completed: tuple[str, ...] = ()
    stages_unavailable: tuple[tuple[str, str], ...] = ()

    @property
    def has_measured_soh(self) -> bool:
        return not self.cycles.empty and "soh" in self.cycles.columns

    @property
    def scored(self) -> bool:
        return not self.guardian.empty

    @property
    def status(self) -> str:
        if self.scored:
            return "SCORED"
        return "MEASURED_ONLY" if self.has_measured_soh else "REFUSED"

    def render(self) -> str:
        lines = [f"{self.cell_id}: {self.status}"]
        lines.append("  " + self.load_report.render().replace("\n", "\n  "))
        lines.append(f"  stages completed: {list(self.stages_completed)}")

        if self.has_measured_soh:
            first = self.cycles.iloc[0]
            last = self.cycles.iloc[-1]
            lines.append(
                f"  measured SOH: {first['soh']:.1f}% at cycle {int(first['cycle'])} "
                f"-> {last['soh']:.1f}% at cycle {int(last['cycle'])}"
            )
            lines.append(
                f"  measured capacity fade: {last['capacity_loss']:.4f} Ah over "
                f"{len(self.cycles)} cycles"
            )

        # Stages blocked by the same missing channel share one explanation.
        # Repeating an identical paragraph per stage buries the one fact a
        # reader needs: which channel is absent.
        grouped: dict[str, list[str]] = {}
        for stage, reason in self.stages_unavailable:
            grouped.setdefault(reason, []).append(stage)
        for reason, stages in grouped.items():
            lines.append(f"  UNAVAILABLE: {', '.join(stages)}")
            lines.append(f"    {reason}")
        return "\n".join(lines)


def _missing_for(stage: str, telemetry: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column for column in STAGE_REQUIREMENTS[stage]
        if column not in telemetry.columns
    )


def _explain(missing: Sequence[str]) -> str:
    parts = [
        f"'{channel}' (needed by {CHANNEL_CONSUMERS.get(channel, 'this stage')})"
        for channel in missing
    ]
    return (
        f"CALCE does not record {', '.join(parts)}. Substituting a constant "
        f"would satisfy the schema and produce a score for a quantity nobody "
        f"measured, so the stage is skipped instead."
    )


def analyze_calce_cell(
    cell_dir: str | Path,
    cell_id: str | None = None,
    temperature_dir: str | Path | None = None,
) -> CalceAnalysis:
    """Load one CALCE cell and run every stage its instrumentation supports.

    Always produces measured SOH and capacity fade, which is what CS2/CX2 data
    is actually good for. Produces behavioural scores only when the cell has the
    channels those scores are computed from — in practice, only CX2_4 with its
    thermocouple files joined.
    """
    telemetry, load_report = load_calce_cell(
        cell_dir, cell_id=cell_id, temperature_dir=temperature_dir
    )
    resolved_id = load_report.cell_id

    stages: list[str] = ["load"]
    unavailable: list[tuple[str, str]] = []

    # Measured capacity fade. This needs only what every CALCE file carries.
    cycles = calce_capacity_loss(summarize_calce_cycles(telemetry))
    stages.append("capacity_fade")

    missing = _missing_for("behavior_flags", telemetry)
    if missing:
        reason = _explain(missing)
        for stage in ("behavior_flags", "risk_assessment", "health_index", "guardian"):
            unavailable.append((stage, reason))
        return CalceAnalysis(
            cell_id=resolved_id, load_report=load_report, cycles=cycles,
            stages_completed=tuple(stages), stages_unavailable=tuple(unavailable),
        )

    guardian = _score(telemetry, resolved_id)
    stages.extend(["behavior_flags", "risk_assessment", "health_index", "guardian"])

    return CalceAnalysis(
        cell_id=resolved_id, load_report=load_report, cycles=cycles,
        guardian=guardian, stages_completed=tuple(stages),
        stages_unavailable=tuple(unavailable),
    )


def _score(telemetry: pd.DataFrame, cell_id: str) -> pd.DataFrame:
    """Delegate to the scoring stages in the order `main.py` uses.

    The sequence, merge and dropped columns are deliberately identical to
    `run_pipeline`. Imported locally so the loader stays importable without the
    whole scoring stack.
    """
    from src.bms.features.behavior_features import (
        add_age_features,
        add_rolling_features,
        compute_behavior_flags,
        summarize_batteries,
    )
    from src.bms.guardian.guardian import generate_guardian_reports
    from src.bms.health.health_index import compute_health_index
    from src.bms.risk.stress_score import compute_risk_assessment, compute_stress_score
    from src.bms.rul.rul_estimation import compute_rul

    frame = telemetry.copy()
    frame["battery_id"] = cell_id

    flagged = compute_behavior_flags(frame)
    flagged["stress_score"] = compute_stress_score(flagged)
    featured = add_age_features(add_rolling_features(flagged))

    summary = summarize_batteries(featured)
    risk = compute_risk_assessment(summary)
    health = compute_health_index(summary)
    merged = risk.merge(
        health.drop(columns=[
            "avg_stress", "avg_temp", "deep_discharge_duration",
            "fast_charge_duration", "aggressive_discharge_count", "avg_soc",
        ]),
        on="battery_id",
    )
    return generate_guardian_reports(compute_rul(merged))


def measured_feasibility(
    source: pd.DataFrame,
    calce_cycles: pd.DataFrame,
    features: Sequence[str] = ("capacity_loss",),
    source_name: str = "nasa",
    target_name: str = "calce",
) -> "object":
    """Measure commensurability against real loaded CALCE data.

    `adaptive.dataset_specs.predict_transfer_feasibility` answers from published
    metadata, before a download. This answers from the files, which is the
    authoritative check and the one ADR 0006 says must be run before citing a
    transfer result.
    """
    from src.bms.adaptive.commensurability import assess_commensurability

    return assess_commensurability(
        source, calce_cycles, features, source_name, target_name
    )
