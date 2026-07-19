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


def generate_guardian_reports(battery: pd.DataFrame) -> pd.DataFrame:
    """Add primary_causes, recommendation, guardian_status, guardian_report.

    Expects the output of `rul.rul_estimation.compute_rul`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery.columns]
    if missing:
        raise ValueError(f"generate_guardian_reports: missing required columns {missing}")

    out = battery.copy()
    out["primary_causes"] = out.apply(_primary_causes, axis=1)
    out["guardian_status"] = out["battery_state"].map(_SEVERITY_MESSAGE).fillna("Unknown state")
    out["recommendation"] = out["battery_state"].map(_STATE_RECOMMENDATION).fillna("No recommendation available")

    out["guardian_report"] = (
        "Battery " + out["battery_id"].astype(str)
        + " is in " + out["battery_state"]
        + " state with estimated remaining life of " + out["rul_cycles"].astype(int).astype(str)
        + " cycles. Primary degradation factors include " + out["primary_causes"]
        + ". Recommended action: " + out["recommendation"]
    )
    return out
