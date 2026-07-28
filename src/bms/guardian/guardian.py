"""Battery Guardian AI: plain-language cause attribution and recommendations.

Ported from `notebooks/08_battery_guardian.ipynb`. The notebook built the
`guardian_report` string twice, with slightly different wording and column
order, and neither the causes-generator nor the recommendation function used
the risk assessment's own `risk_reason`/`recommended_action` from
n05_T1v1 — Guardian re-derives causes from raw thresholds instead of reusing
the risk module's reasoning. That inconsistency is preserved here (Guardian
intentionally explains *health/RUL* drivers, which are not identical inputs
to the *risk score*), but is worth knowing about if the two reports ever
need to agree with each other for the same battery.

Evidence labeling (added after docs/final_report.md's calibration work):
Guardian's causes were, until now, presented as equally certain -- "high
temperature exposure" and "frequent fast charging" read the same way, with
no indication that only one of them actually held up under testing. See
final_report.md Section 4: temperature exposure is a real, cohort-controlled,
statistically significant, transferable signal (coefficient 0.0038 Ah/°C,
95% CI 0.002-0.005, p<0.0001 -- but R²=0.015 in-sample, a real but modest
effect, and the full fitted model doesn't beat a naive baseline out-of-sample).
Current-based flags (fast charging, deep discharge) do NOT transfer --
significant and positive in room-temperature cohorts, significant and
*negative* in the 4°C cohort. `health_index` itself showed no significant
relationship with real fade rate at all (Section 4.1). None of that
distinction was visible in the diagnosis before -- it now is, via
`evidence_confidence`/`evidence_note`, without changing what causes are
detected or how severity is classified (that would be a bigger, separate
change -- this only changes what the system says about its own certainty).
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = (
    "battery_id",
    "battery_state",
    "health_index",
    "rul_cycles",
    "avg_temp",
    "fast_charge_duration",
    "deep_discharge_duration",
)

_SEVERITY_MESSAGE = {
    "CRITICAL": "High risk of battery failure",
    "DEGRADED": "Battery performance degradation detected",
    "WARNING": "Early degradation indicators detected",
    "HEALTHY": "Battery operating normally",
}

_STATE_RECOMMENDATION = {
    "CRITICAL": "Immediate inspection and battery replacement",
    "DEGRADED": "Reduce fast charging and monitor temperature",
    "WARNING": "Maintain SOC between 20 and 80 percent",
    "HEALTHY": "Continue normal operation",
}

# The one number in this module that's actually a fitted, cited result
# rather than a hand-picked threshold. Source: reports/metrics/
# health_model_v2_coefficients.txt, model `capacity_loss ~
# trailing_avg_temp + C(cohort)`, exactly the number final_report.md
# Section 4.3 reports. If that model is ever refit, update this constant
# from the same file -- don't let it silently drift out of sync.
TEMPERATURE_COEFFICIENT_AH_PER_C = 0.0038
TEMPERATURE_COEFFICIENT_CI = (0.002, 0.005)
TEMPERATURE_COEFFICIENT_R2 = 0.015

# Which causes are backed by the validation work in final_report.md
# Section 4, and which are the original hand-picked thresholds that
# were never confirmed (and in the current-based cases, were actively
# found NOT to transfer). "accelerated battery aging" is tied to
# health_index itself, which Section 4.1 found has no significant
# relationship with real fade rate at all.
_CAUSE_EVIDENCE = {
    "high temperature exposure": (
        "VALIDATED",
        f"Temperature exposure is a real, cohort-controlled, statistically "
        f"significant signal (coefficient {TEMPERATURE_COEFFICIENT_AH_PER_C} Ah/°C, "
        f"95% CI {TEMPERATURE_COEFFICIENT_CI[0]}-{TEMPERATURE_COEFFICIENT_CI[1]}, p<0.0001; "
        f"R²={TEMPERATURE_COEFFICIENT_R2} in-sample -- a real but modest effect, "
        f"see docs/final_report.md Section 4.3).",
    ),
    "frequent fast charging": (
        "HEURISTIC",
        "This threshold has not been confirmed against real degradation data. "
        "The related current-based signal was tested and did NOT transfer across "
        "conditions (sign flipped between room-temperature and cold cohorts) -- "
        "see docs/final_report.md Section 4.2.",
    ),
    "deep discharge events": (
        "HEURISTIC",
        "This threshold has not been confirmed against real degradation data -- "
        "see docs/final_report.md Section 4.2 for the current-based signals that "
        "were tested and did not transfer across conditions.",
    ),
    "accelerated battery aging": (
        "HEURISTIC",
        "health_index itself showed no statistically significant relationship "
        "with real measured fade rate in validation testing -- see "
        "docs/final_report.md Section 4.1.",
    ),
    "normal battery usage": ("N/A", "No causes were flagged for this battery."),
}


def _primary_causes(row: pd.Series) -> str:
    causes = []
    if row["avg_temp"] > 35:
        causes.append("high temperature exposure")
    if row["fast_charge_duration"] > 50:
        causes.append("frequent fast charging")
    if row["deep_discharge_duration"] > 50:
        causes.append("deep discharge events")
    if row["health_index"] > 60:
        causes.append("accelerated battery aging")
    if not causes:
        causes.append("normal battery usage")
    return ", ".join(causes)


def _evidence_confidence(causes_str: str) -> str:
    """Overall confidence label for a battery's diagnosis: VALIDATED if
    temperature (the one real signal) is driving it, HEURISTIC if only
    unvalidated thresholds fired, MIXED if both, N/A if no causes."""
    causes = [c.strip() for c in causes_str.split(",")]
    labels = {_CAUSE_EVIDENCE.get(c, ("HEURISTIC", ""))[0] for c in causes}
    if labels == {"N/A"}:
        return "N/A"
    if labels == {"VALIDATED"}:
        return "VALIDATED"
    if "VALIDATED" in labels:
        return "MIXED"
    return "HEURISTIC"


def _evidence_note(causes_str: str) -> str:
    """Human-readable explanation of which causes are backed by the
    calibration work and which aren't -- one sentence per distinct cause,
    deduplicated, in a stable order."""
    causes = [c.strip() for c in causes_str.split(",")]
    seen = []
    notes = []
    for c in causes:
        if c in seen:
            continue
        seen.append(c)
        _, note = _CAUSE_EVIDENCE.get(c, ("HEURISTIC", "This cause has not been independently validated."))
        notes.append(note)
    return " ".join(notes)


def generate_guardian_reports(battery: pd.DataFrame) -> pd.DataFrame:
    """Add primary_causes, recommendation, guardian_status, guardian_report,
    evidence_confidence, evidence_note.

    Expects the output of `rul.rul_estimation.compute_rul`.

    evidence_confidence/evidence_note are additive -- they change what the
    system SAYS about its own certainty, not what it detects or how it
    classifies severity. Existing callers reading `guardian_report`,
    `primary_causes`, `guardian_status`, or `recommendation` see identical
    values to before this was added.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery.columns]
    if missing:
        raise ValueError(f"generate_guardian_reports: missing required columns {missing}")

    out = battery.copy()
    out["primary_causes"] = out.apply(_primary_causes, axis=1)
    out["guardian_status"] = out["battery_state"].map(_SEVERITY_MESSAGE).fillna("Unknown state")
    out["recommendation"] = out["battery_state"].map(_STATE_RECOMMENDATION).fillna("No recommendation available")
    out["evidence_confidence"] = out["primary_causes"].apply(_evidence_confidence)
    out["evidence_note"] = out["primary_causes"].apply(_evidence_note)

    out["guardian_report"] = (
        "Battery " + out["battery_id"].astype(str)
        + " is in " + out["battery_state"]
        + " state with estimated remaining life of " + out["rul_cycles"].astype(int).astype(str)
        + " cycles. Primary degradation factors include " + out["primary_causes"]
        + ". Recommended action: " + out["recommendation"]
    )
    return out

