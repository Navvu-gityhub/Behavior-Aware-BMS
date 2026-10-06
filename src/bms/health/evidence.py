"""What each result rests on, and whether there is enough of it.

Every figure BEACON reports answers "how healthy is it". This module answers
the questions a reviewer asks next: on what evidence, with what confidence,
and why is this other figure missing?

DATA SUFFICIENCY IS PER OUTPUT, NOT ONE SCORE
---------------------------------------------
A single "data sufficiency: 78%" would need weights saying how much a
temperature channel is worth against a complete discharge, and no measurement
supplies those weights - the number would be invented. What is decidable is,
for each output, whether its specific inputs exist. So each output is
AVAILABLE or NOT AVAILABLE, with the reason, and a confidence when available.

CONFIDENCE RULES
----------------
These are conventions stated in code, not fitted values. Each maps to the
validation that backs it:

  capacity health    HIGH    readings consistent (standard error <= 1 point),
                             fixed window, reference at beginning of life
                     MEDIUM  consistent, but a learned window (2 validation
                             cells) or a start-of-log reference
                     LOW     readings inconsistent: validation error was ~4x
                             larger in this state (1.3% vs 5.1% median)
                     The NUMBER of discharges is deliberately not a rule:
                     measured, error grows with age rather than shrinking
                     with count (reports/metrics/calce_sufficiency/).
  resistance health  HIGH    >= 30 discharges with a clean load step
                     MEDIUM  fewer
                     (tracked the cycler's resistance on 19 of 20 cells)
  remaining life     HIGH    predicted under 25 cycles (lasted at least that
                             long 95% of the time in validation)
                     MEDIUM  25-100 cycles
                     LOW     beyond 100 cycles (shown, with its wide band)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
SUFFICIENT_DISCHARGES = 30


@dataclass(frozen=True)
class OutputStatus:
    name: str
    available: bool
    confidence: str = ""
    reason: str = ""


@dataclass(frozen=True)
class Evidence:
    """The provenance block shown beside every result."""

    source: str
    samples: int
    duration_s: float
    channels: dict[str, bool]
    rated_capacity_known: bool
    discharges: int
    complete_discharges: int
    window_discharges: int
    reference: str
    outputs: tuple[OutputStatus, ...] = field(default_factory=tuple)

    def output(self, name: str) -> OutputStatus:
        for o in self.outputs:
            if o.name == name:
                return o
        raise KeyError(name)

    def render(self) -> list[str]:
        hours, rem = divmod(int(self.duration_s), 3600)
        tick = {True: "yes", False: "NO"}
        lines = [
            f"   Source: {self.source}",
            f"   Samples: {self.samples:,} over {hours} h {rem // 60} min",
            "   Channels: " + ", ".join(f"{k} {tick[v]}" for k, v in self.channels.items()),
            f"   Rated capacity declared: {tick[self.rated_capacity_known]}",
            f"   Discharges: {self.discharges} ({self.complete_discharges} complete, "
            f"{self.window_discharges} usable for capacity health)",
            f"   Reference: {self.reference}",
            "",
        ]
        for o in self.outputs:
            label = (f"{o.confidence} confidence" if o.confidence in (HIGH, MEDIUM, LOW)
                     else o.confidence)
            status = f"AVAILABLE ({label})" if o.available else "NOT AVAILABLE"
            lines.append(f"   {o.name:<22} {status}")
            if o.reason:
                lines.append(f"   {'':<22} {o.reason}")
        return lines


def evidence_from_result(result, rated_capacity_known: bool) -> Evidence:
    """Build the evidence block from a `TelemetryResult`."""
    tel = result.telemetry
    t = pd.to_numeric(tel["test_time_s"], errors="coerce") if "test_time_s" in tel else pd.Series(dtype=float)
    channels = {
        "voltage": "voltage_v" in tel.columns,
        "current": "current_a" in tel.columns,
        "temperature": "temperature_c" in tel.columns
        and pd.to_numeric(tel["temperature_c"], errors="coerce").notna().any(),
        "SOC": "soc" in tel.columns,
    }
    measurements = result.measurements
    n_complete = sum(1 for m in measurements if m.is_complete)
    soh = result.field_soh
    bol = result.soh_reference == "beginning_of_life"
    reference = ("this cell's first discharges, at beginning of life" if bol
                 else "this log's first discharges (the cell may not have been new)")

    outputs = [
        _capacity_status(soh, bol),
        _resistance_status(soh),
        _rul_status(result.rul_estimate),
        OutputStatus(
            "heat exposure", channels["temperature"],
            "MEASURED" if channels["temperature"] else "",
            "" if channels["temperature"] else "no temperature channel on this source"),
        OutputStatus(
            "behaviour risk score", bool(result.scored), "HEURISTIC" if result.scored else "",
            "hand-set rules, not validated against measured fade" if result.scored
            else "needs temperature, SOC and rated capacity"),
    ]
    return Evidence(
        source=result.source,
        samples=int(len(tel)),
        duration_s=float(t.max() - t.min()) if t.notna().sum() > 1 else 0.0,
        channels=channels,
        rated_capacity_known=rated_capacity_known,
        discharges=len(measurements),
        complete_discharges=n_complete,
        window_discharges=int(soh.n_accepted) if soh is not None else 0,
        reference=reference,
        outputs=tuple(outputs),
    )


def _capacity_status(soh, bol: bool) -> OutputStatus:
    name = "capacity health"
    if soh is None or not soh.available:
        return OutputStatus(name, False, reason=soh.refusal if soh is not None else "not computed")
    if not soh.consistent:
        u = soh.uncertainty
        detail = f"+/-{u * 100:.1f} points" if math.isfinite(u) else "too few recent readings to judge"
        return OutputStatus(name, True, LOW,
                            f"readings inconsistent ({detail}); in validation, error was "
                            f"about 4x larger in this state - keep observing")
    why = []
    if soh.window_mode != "fixed":
        why.append("window learned from this cell's usage (2 validation cells)")
    if not bol:
        why.append("reference may not be the cell as new")
    return OutputStatus(name, True, HIGH if not why else MEDIUM, "; ".join(why))


def _resistance_status(soh) -> OutputStatus:
    name = "resistance health"
    if soh is None or not math.isfinite(soh.resistance_growth):
        return OutputStatus(name, False, reason="no clean rest-to-load step was seen")
    if soh.n_accepted >= SUFFICIENT_DISCHARGES:
        return OutputStatus(name, True, HIGH)
    return OutputStatus(name, True, MEDIUM, f"{soh.n_accepted} discharges with a load step")


def _rul_status(rul) -> OutputStatus:
    name = "remaining life"
    if rul is None or not math.isfinite(rul.rul_cycles):
        return OutputStatus(name, False, reason=rul.refusal if rul is not None
                            else "no complete discharge to build a trend from")
    if rul.rul_cycles < 25:
        return OutputStatus(name, True, HIGH)
    if rul.rul_cycles < 100:
        return OutputStatus(name, True, MEDIUM)
    return OutputStatus(name, True, LOW, "far from end of life; the band is wide")


def resistance_health(soh) -> float:
    """Resistance at the reference over resistance now: 1.0 = as new.

    The inverse of growth, so it reads like capacity health - lower is worse.
    A cell whose resistance rose 19% reads 84%.
    """
    if soh is None or not math.isfinite(soh.resistance_growth) or soh.resistance_growth <= 0:
        return float("nan")
    return 1.0 / soh.resistance_growth
