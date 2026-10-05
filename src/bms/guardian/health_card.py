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

import pandas as pd

from src.bms.guardian.guardian import GENERAL_GUIDANCE, _heat_advice
from src.bms.health.field_soh import state_from_soh
from src.bms.rul.fade_extrapolation import DEFAULT_EOL_THRESHOLD

# Above this, time is counted as "hot". The V1 brief's own feature
# (`time_above_40C`); kept so the card reports what was originally planned.
HOT_C = 40.0

# The RUL estimator's measured accuracy by distance to end of life, CALCE at
# threshold 0.90, with SOH referenced to each cell's FIRST cycles - the
# reference this pipeline uses and a BMS can actually hold. Quoted from
# reports/metrics/calce_rul_horizon_early_ref/, not recomputed.
#
# The often-quoted 88% is the same estimator against a lifetime-percentile
# reference that uses the cell's future; it is not what a deployed card gets.
_RUL_SCOPE = (
    "Accuracy depends on how close end of life is: within 25 cycles of it the "
    "estimate was within +/-20 cycles 73% of the time. Further out it runs "
    "early - by about 144 cycles when end of life is 200-400 cycles away - "
    "so a small number here is a prompt to keep measuring, not a verdict."
)


def render_health_card(result, battery_label: str | None = None) -> str:
    """Plain-text report card for one battery from one pipeline run."""
    label = battery_label or (
        str(result.telemetry["cell_id"].iloc[0])
        if not result.telemetry.empty and "cell_id" in result.telemetry.columns
        else result.source)
    lines = [f"BATTERY HEALTH REPORT - {label}", "=" * 60, ""]

    soh = result.field_soh
    bol = result.soh_reference == "beginning_of_life"
    lines.append("1. HOW HEALTHY IT IS")
    if soh is not None and soh.available:
        what = "state of health" if bol else "capacity relative to the start of this log"
        lines.append(f"   {soh.soh:.1%} {what}  [MEASURED]")
        lines.append(
            f"   From charge delivered across the {soh.window} window, "
            f"{soh.n_accepted} of {soh.n_discharges} discharges usable, "
            f"latest at discharge {soh.at_cycle}.")
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
            lines.append(f"   {_RUL_SCOPE}")
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
