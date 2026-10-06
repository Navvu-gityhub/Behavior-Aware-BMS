"""The battery health report card: what the project set out to give a user.

The original brief (June 2026) was a layer above the BMS that tells a driver
how their battery is ageing, how their usage affects it, and what to change.
Testing cut that down to what the data supports, and this is the result laid
out in that order:

    1. How healthy it is        measured SOH (voltage window), or why not
    2. How long it has left     fade-trend RUL, or why not
    3. What usage is doing      heat exposure - the one usage link that held
    4. What to do               actions that follow from 1-3
    5. What this cannot tell    every refusal the run produced

Every line says whether it is a measurement, an estimate, or general guidance.
Nothing is computed here: it reads a `TelemetryResult` and lays it out, so a
number on the card is the number the pipeline produced.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.bms.guardian.guardian import GENERAL_GUIDANCE, _heat_advice
from src.bms.health.error_bands import describe_rul_band, soh_band
from src.bms.health.evidence import evidence_from_result, resistance_health
from src.bms.health.field_soh import state_from_soh
from src.bms.rul.fade_extrapolation import DEFAULT_EOL_THRESHOLD

# Above this, time is counted as "hot". The V1 brief's own feature
# (`time_above_40C`); kept so the card reports what was originally planned.
HOT_C = 40.0

# Accuracy statements on the card come from health/error_bands.py, which is
# pinned to reports/metrics/error_bands.csv. RUL accuracy there is indexed by
# the PREDICTED value - the only one a user sees.


def render_health_card(result, battery_label: str | None = None,
                       rated_capacity_known: bool | None = None) -> str:
    """Plain-text report card for one battery from one pipeline run."""
    label = battery_label or (
        str(result.telemetry["cell_id"].iloc[0])
        if not result.telemetry.empty and "cell_id" in result.telemetry.columns
        else result.source)
    lines = [f"BATTERY HEALTH REPORT - {label}", "=" * 60, ""]

    if rated_capacity_known is None:
        rated_capacity_known = getattr(result, "rated_capacity_ah", None) is not None
    evidence = evidence_from_result(result, rated_capacity_known)
    lines.append("0. EVIDENCE - what these results rest on")
    lines.extend(evidence.render())
    lines.append("")

    soh = result.field_soh
    bol = result.soh_reference == "beginning_of_life"
    lines.append("1. HOW HEALTHY IT IS")
    if soh is not None and soh.available:
        what = "capacity health" if bol else "capacity relative to the start of this log"
        band = soh_band(soh.window_mode)
        cap = evidence.output("capacity health")
        lines.append(f"   {soh.soh:.1%} {what}  [MEASURED, {cap.confidence} confidence]")
        lines.append(
            f"   Validation error: 90% of readings within +/-{band * 100:.1f} "
            f"points of the lab's capacity test.")
        lines.append(
            f"   From charge delivered across the {soh.window} window, "
            f"{soh.n_accepted} of {soh.n_discharges} discharges usable, "
            f"latest at discharge {soh.at_cycle}.")
        if np.isfinite(soh.resistance_growth):
            res = evidence.output("resistance health")
            lines.append(
                f"   {resistance_health(soh):.1%} resistance health  [MEASURED, "
                f"{res.confidence} confidence] - resistance as new / resistance now")
            lines.append(
                f"   Internal resistance: {soh.resistance_reference_ohm * 1000:.0f} -> "
                f"{soh.resistance_now_ohm * 1000:.0f} mOhm "
                f"({(soh.resistance_growth - 1) * 100:+.0f}%)  [MEASURED, from the "
                f"voltage drop at each load step]. Rising resistance is power fade; "
                f"this tracked the lab cycler's own resistance on 19 of 20 cells.")
        if soh.window_mode == "learned":
            lines.append(
                "   This battery never discharges across the standard window, so "
                "the window was learned from its own early discharges. That "
                "mode measured 3.2-3.4% error on real top-of-charge partial "
                "cycling (2 CALCE cells) - fewer cells than the standard window.")
        if bol:
            lines.append(f"   State: {state_from_soh(soh.soh)}")
        else:
            lines.append(
                "   The log may not start when the cell was new, so this is fade "
                "since logging began, not state of health. No state is assigned.")
    else:
        reason = soh.refusal if soh is not None else "not computed"
        lines.append(f"   Not available: {reason}")
    lines.append("")

    lines.append("2. HOW LONG IT HAS LEFT")
    rul = result.rul_estimate
    if rul is not None and math.isfinite(rul.rul_cycles):
        if rul.rul_cycles == 0:
            lines.append(
                f"   Already below {DEFAULT_EOL_THRESHOLD:.0%} of its early capacity.  [MEASURED]")
        else:
            lines.append(
                f"   About {rul.rul_cycles:.0f} cycles until it falls to "
                f"{DEFAULT_EOL_THRESHOLD:.0%} of its early capacity.  [ESTIMATE]")
            lines.append(f"   {describe_rul_band(rul.rul_cycles)}")
    else:
        reason = rul.refusal if rul is not None else "no complete discharge to build a trend from"
        lines.append(f"   Not available: {reason}")
    lines.append("")

    lines.append("3. WHAT YOUR USAGE IS DOING TO IT")
    temps = (pd.to_numeric(result.telemetry["temperature_c"], errors="coerce").dropna()
             if "temperature_c" in result.telemetry.columns else pd.Series(dtype=float))
    if temps.empty:
        lines.append(f"   {_heat_advice(float('nan'))}")
    else:
        hot_share = float((temps > HOT_C).mean())
        lines.append(
            f"   Temperature: average {temps.mean():.1f} °C, max {temps.max():.1f} °C, "
            f"{hot_share:.0%} of samples above {HOT_C:.0f} °C.  [MEASURED]")
        lines.append(f"   {_heat_advice(float(temps.mean()))}")
    lines.append("")

    lines.append("4. WHAT TO DO")
    if soh is not None and soh.available and bol:
        from src.bms.guardian.guardian import _MEASURED_STATE_RECOMMENDATION
        lines.append(f"   {_MEASURED_STATE_RECOMMENDATION[state_from_soh(soh.soh)]}")
    else:
        lines.append("   No measured state, so no action is derived from one.")
    lines.append(f"   {GENERAL_GUIDANCE}")
    lines.append("")

    lines.append("5. WHAT THIS REPORT CANNOT TELL YOU")
    refusals = list(result.refusals)
    if not refusals:
        lines.append("   Nothing was refused on this run.")
    for refusal in refusals:
        parts = [part.strip() for part in refusal.splitlines() if part.strip()]
        body = ("\n" + " " * 5).join(parts)
        lines.append(f"   - {body}")
    lines.append(
        "   - Every method here was validated on single cells of one LCO family "
        "(CALCE, NASA). A pack, or another chemistry, is outside that evidence.")
    return "\n".join(lines)
