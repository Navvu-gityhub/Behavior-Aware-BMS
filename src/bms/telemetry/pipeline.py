"""End-to-end telemetry pipeline: CAN frames to Guardian output.

Wires the existing stages together for live and recorded telemetry:

    source -> decode (DBC) -> unified schema -> resample to cycles
           -> behaviour features -> risk -> health -> RUL -> Guardian -> twin

Nothing here reimplements a stage. Every computation is delegated to the module
that already owns it, so a telemetry run and a dataset run produce numbers by the
same code path. That matters more than convenience: if this module had its own
feature extraction, a discrepancy between live and batch results would be
untraceable.

Where this pipeline refuses
--------------------------
Three refusals, each for a reason established earlier in the project.

**Missing signal channels.** Checked before decoding. The example DBC has no
temperature signal, and `compute_behavior_flags` needs one for `high_temp_flag`.
The NaN-as-healthy fix established why this must refuse: a NumPy comparison
against NaN is False, so an absent temperature would have silently produced "not
hot" for every row and a healthy score for a pack nobody measured.

**No complete discharge cycle.** SOH is capacity relative to initial capacity,
and capacity is only comparable across equal depths of discharge. Real driving
produces mostly partial cycles, so a log can decode perfectly and still support
no SOH figure. `TelemetryResult.capacity_yield` reports how many usable points
were found.

**Fade prediction.** `AdaptiveCalibrator.score` refuses while nothing has passed
the promotion gate, which is the current state. This pipeline does not route
around that. It emits the rule-based severity index, which is labelled
throughout as triage rather than measurement, and does not present it as a
calibrated fade prediction.

The last one is the point of the whole exercise. Wiring a model the gate rejected
into a live dashboard, where it would look authoritative, is precisely the
failure this project was built to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from src.bms.telemetry.cycles import (
    CapacityYield,
    CycleMeasurement,
    capacity_yield,
    cycles_to_frame,
    measure_cycles,
)
from src.bms.telemetry.sources import (
    CanFrameSource,
    SignalCoverage,
    check_signal_coverage,
)
from src.bms.telemetry.twin_integration import (
    TwinHistory,
    TwinUpdate,
    evaluate_twin_from_guardian,
)

# Default mapping from the example DBC's signal names to unified schema
# channels. Deliberately incomplete, because the DBC is: there is no temperature
# signal to map. See `sources.REQUIRED_CHANNELS`.
TWIZY_SIGNAL_MAP: Mapping[str, str] = {
    "v_b_current": "current_a",
    "v_b_soc": "soc",
}


@dataclass(frozen=True)
class TelemetryResult:
    """Everything one telemetry run produced, including its refusals."""

    source: str
    n_frames: int
    n_decoded: int
    coverage: SignalCoverage
    telemetry: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    measurements: tuple[CycleMeasurement, ...] = ()
    guardian: pd.DataFrame = field(default_factory=pd.DataFrame)
    twin: TwinUpdate | None = None
    yield_summary: CapacityYield | None = None
    refusals: tuple[str, ...] = ()
    stages_completed: tuple[str, ...] = ()

    @property
    def scored(self) -> bool:
        return not self.guardian.empty

    @property
    def status(self) -> str:
        if self.refusals and not self.scored:
            return "REFUSED"
        return "SCORED_WITH_REFUSALS" if self.refusals else "SCORED"

    def render(self) -> str:
        lines = [f"{self.source}: {self.status}"]
        lines.append(f"  frames read: {self.n_frames}, decoded: {self.n_decoded}")
        lines.append(f"  stages completed: {list(self.stages_completed)}")
        if self.yield_summary is not None:
            lines.append(f"  {self.yield_summary.render()}")
        if not self.coverage.complete:
            lines.append(f"  signal coverage: {self.coverage.status}")
            for channel in self.coverage.missing_channels:
                lines.append(f"    missing: {channel}")
        if self.twin is not None:
            lines.append("  " + self.twin.render().replace("\n", "\n  "))
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        return "\n".join(lines)


def decode_frames(
    frames: Iterator[tuple[float, int, bytes]],
    dbc,
    signal_map: Mapping[str, str],
    cell_id: str = "VEHICLE_01",
) -> tuple[pd.DataFrame, int, int]:
    """Decode raw frames into a unified-schema telemetry frame.

    Returns (frame, n_read, n_decoded). Undecodable frames are counted rather
    than raised on: a real bus carries messages a given DBC does not define, and
    aborting on the first one would make the pipeline unusable on any real
    vehicle.
    """
    rows: list[dict[str, float]] = []
    n_read = 0
    n_decoded = 0

    for timestamp, arbitration_id, payload in frames:
        n_read += 1
        try:
            decoded = dbc.decode_message(arbitration_id, payload)
        except Exception:
            # Unknown id, or a payload that does not match the definition.
            continue
        n_decoded += 1

        row: dict[str, float] = {"test_time_s": float(timestamp)}
        for signal, value in decoded.items():
            channel = signal_map.get(signal)
            if channel is None:
                continue
            try:
                row[channel] = float(value)
            except (TypeError, ValueError):
                # Named-value signals decode to strings; they are not numeric
                # channels and are skipped rather than coerced to a number.
                continue
        if len(row) > 1:
            rows.append(row)

    if not rows:
        return pd.DataFrame(), n_read, n_decoded

    telemetry = pd.DataFrame(rows)
    telemetry["cell_id"] = cell_id

    # A CAN bus interleaves messages, so each frame carries a subset of
    # channels. Group by timestamp and take the first non-null per channel to
    # assemble complete rows. Forward-filling across timestamps is deliberately
    # avoided: it would invent measurements between frames.
    telemetry = (
        telemetry.groupby(["cell_id", "test_time_s"], as_index=False).first()
        .sort_values("test_time_s")
        .reset_index(drop=True)
    )
    return telemetry, n_read, n_decoded


def run_telemetry_pipeline(
    source: CanFrameSource,
    dbc,
    signal_map: Mapping[str, str] = TWIZY_SIGNAL_MAP,
    cell_id: str = "VEHICLE_01",
    dbc_path: str = "<dbc>",
    require_full_coverage: bool = True,
    twin_history: TwinHistory | None = None,
) -> TelemetryResult:
    """Run CAN telemetry through the existing scoring stages.

    Passing `twin_history` enables digital-twin transition detection across
    calls. Omitting it keeps this function a pure function of its inputs, which
    is what makes replay and live capture provably identical - see
    `twin_integration.py` for why the history is external rather than held here.

    `require_full_coverage=False` allows the run to proceed past a missing
    channel so a caller can inspect the decoded telemetry. It does not make the
    scoring stages run: those still refuse, because the refusal is about the
    data, not about permission.
    """
    coverage = check_signal_coverage(dbc, signal_map, dbc_path)
    refusals: list[str] = []
    stages: list[str] = []

    if not coverage.complete:
        refusals.append(
            f"DBC does not supply {list(coverage.missing_channels)}. "
            + coverage.render().split("\n", 1)[1].strip()
        )
        if require_full_coverage:
            return TelemetryResult(
                source=source.name, n_frames=0, n_decoded=0, coverage=coverage,
                refusals=tuple(refusals), stages_completed=(),
            )

    telemetry, n_frames, n_decoded = decode_frames(
        source.frames(), dbc, signal_map, cell_id
    )
    stages.append("decode")

    if telemetry.empty:
        refusals.append(
            f"No frames decoded into mapped channels. Read {n_frames} frame(s), "
            f"decoded {n_decoded}. Check that the DBC matches the bus and that "
            f"signal_map names signals this DBC defines."
        )
        return TelemetryResult(
            source=source.name, n_frames=n_frames, n_decoded=n_decoded,
            coverage=coverage, refusals=tuple(refusals),
            stages_completed=tuple(stages),
        )

    measurements = measure_cycles(telemetry, cell_id=cell_id)
    summary = capacity_yield(measurements)
    stages.append("segment_cycles")

    cycles = cycles_to_frame(measurements, complete_only=True)
    if cycles.empty:
        refusals.append(
            "No complete discharge cycle found, so no capacity measurement is "
            "available and SOH cannot be computed. "
            + summary.render()
            + " Capacity is only comparable across equal depths of discharge, "
            "so partial cycles are excluded rather than scaled."
        )
        return TelemetryResult(
            source=source.name, n_frames=n_frames, n_decoded=n_decoded,
            coverage=coverage, telemetry=telemetry,
            measurements=tuple(measurements), yield_summary=summary,
            refusals=tuple(refusals), stages_completed=tuple(stages),
        )

    guardian = pd.DataFrame()
    if coverage.complete:
        try:
            guardian = _score_cycles(telemetry, cycles, cell_id)
            stages.append("score")
        except ValueError as exc:
            # The feature and scoring layers raise ValueError deliberately when
            # data is absent or inconsistent, rather than treating missing as
            # safe. That is the explicit-failure path working.
            refusals.append(
                f"Scoring refused: {exc}. The feature layer raises on absent or "
                f"inconsistent data rather than treating it as healthy."
            )
        except (ImportError, AttributeError, KeyError) as exc:
            # These are wiring defects in this module, not properties of the
            # data. Labelling them as data refusals would hide real bugs, so
            # they are re-raised.
            raise RuntimeError(
                f"Telemetry pipeline is mis-wired against the scoring stages: "
                f"{type(exc).__name__}: {exc}. This is a defect in "
                f"telemetry/pipeline.py, not a property of the telemetry."
            ) from exc
    else:
        refusals.append(
            "Scoring skipped: the feature layer requires channels this DBC does "
            "not supply, and treating them as absent-equals-safe is the "
            "NaN-as-healthy defect this project already fixed."
        )

    twin_update = evaluate_twin_from_guardian(guardian, twin_history)
    if twin_update.evaluated:
        stages.append("twin")

    return TelemetryResult(
        source=source.name, n_frames=n_frames, n_decoded=n_decoded,
        coverage=coverage, telemetry=telemetry, cycles=cycles,
        measurements=tuple(measurements), guardian=guardian,
        twin=twin_update, yield_summary=summary, refusals=tuple(refusals),
        stages_completed=tuple(stages),
    )


def _score_cycles(
    telemetry: pd.DataFrame, cycles: pd.DataFrame, cell_id: str
) -> pd.DataFrame:
    """Delegate to the existing scoring stages, in the order `main.py` uses.

    The sequence, the merge and the column drops are deliberately identical to
    `run_pipeline`. Any divergence would mean a telemetry run and a dataset run
    could disagree for reasons no one could trace, which defeats the purpose of
    sharing the stages at all.

    Imported locally so `cycles.py` and `sources.py` stay importable without
    pulling in the whole scoring stack, which matters for the API's cold start.
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

    # Attach the cycle each telemetry row belongs to, so age and rolling
    # features see a real cycle index rather than a constant.
    enriched = _attach_cycle_index(telemetry, cycles)

    flagged = compute_behavior_flags(enriched)
    flagged["stress_score"] = compute_stress_score(flagged)
    featured = add_rolling_features(flagged)
    featured = add_age_features(featured)

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


def _attach_cycle_index(
    telemetry: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    """Label each telemetry row with the discharge cycle it falls inside.

    Rows outside any complete discharge — charge phases, rests, and excluded
    partial discharges — are dropped rather than assigned to a neighbouring
    cycle. Assigning them would fold charging temperatures into a discharge
    cycle's aggregate and shift every per-cycle feature.
    """
    enriched = telemetry.copy()
    enriched["cycle"] = pd.NA

    for _, cycle_row in cycles.iterrows():
        inside = (
            (enriched["test_time_s"] >= cycle_row["start_time_s"])
            & (enriched["test_time_s"] <= cycle_row["end_time_s"])
        )
        enriched.loc[inside, "cycle"] = int(cycle_row["cycle"])

    enriched = enriched[enriched["cycle"].notna()].copy()
    enriched["cycle"] = enriched["cycle"].astype(int)
    return enriched.sort_values(["cell_id", "cycle"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_log(
    path: Path | str,
    dbc,
    signal_map: Mapping[str, str] = TWIZY_SIGNAL_MAP,
    cell_id: str = "VEHICLE_01",
    dbc_path: str = "<dbc>",
    require_full_coverage: bool = True,
    twin_history: TwinHistory | None = None,
) -> TelemetryResult:
    """Replay a recorded CAN log through the same pipeline as live capture.

    Replay and live differ only in the source, which is the point: a result
    reproduced from a log is the same computation the vehicle produced, so a
    disagreement is a data difference rather than a code-path difference.
    """
    from src.bms.telemetry.sources import LogFileSource

    return run_telemetry_pipeline(
        LogFileSource(name=f"replay:{Path(path).name}", path=path),
        dbc=dbc, signal_map=signal_map, cell_id=cell_id, dbc_path=dbc_path,
        require_full_coverage=require_full_coverage, twin_history=twin_history,
    )
