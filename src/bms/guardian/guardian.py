"""Battery Guardian: plain-language cause attribution and recommendations.

WHAT CHANGED
--------------------
This module went through two iterations in parallel development:

**Iteration 1 (main branch, pre-this-commit):** Added evidence confidence labels
(VALIDATED/HEURISTIC/MIXED) to distinguish temperature (a real, tested signal) from
the hand-picked thresholds that were never confirmed. This was the right direction.

**Iteration 2 (this commit, feature branch):** Replaced the if/else threshold-based
cause attribution with exact Shapley decomposition of the rule scores themselves,
solving the deeper problem: Guardian was explaining scores using a rulebook that
existed nowhere else in the codebase. That allowed it to name causes contributing
nothing and omit terms that dominated.

**This merge:** Keeps the Shapley foundation (exact, consistent with the score)
and restores the evidence labels to track what was actually validated.
"""

from __future__ import annotations

import pandas as pd

from src.bms.explain.attribution import explain_scores
from src.bms.health.health_index import HEALTH_TERMS, health_penalty_from_terms
from src.bms.risk.stress_score import RISK_TERMS, risk_score_from_terms

REQUIRED_COLUMNS = (
    "battery_id",
    "battery_state",
    "health_index",
    "rul_cycles",
    "avg_temp",
    "fast_charge_duration",
    "deep_discharge_duration",
)

ATTRIBUTION_FEATURES = (
    "avg_stress",
    "avg_temp",
    "deep_discharge_duration",
    "fast_charge_duration",
    "aggressive_discharge_count",
    "avg_soc",
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
# found NOT to transfer).
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
    "sustained high stress": (
        "HEURISTIC",
        "stress-based signals were tested but showed no significant correlation "
        "with measured fade rate -- see docs/final_report.md Section 4.",
    ),
    "aggressive discharge": (
        "HEURISTIC",
        "This threshold has not been confirmed against real degradation data.",
    ),
    "SOC held at extremes": (
        "HEURISTIC",
        "SOC exposure thresholds have not been independently validated.",
    ),
    "normal usage": ("N/A", "No causes were flagged for this battery."),
}


def _evidence_confidence(dominant_cause: str) -> str:
    """Label the confidence level of a diagnosis.
    
    VALIDATED: temperature signal, the one real calibrated driver.
    HEURISTIC: hand-picked thresholds that were never confirmed.
    N/A: no causes flagged.
    """
    if dominant_cause == "normal usage":
        return "N/A"
    confidence, _ = _CAUSE_EVIDENCE.get(dominant_cause, ("HEURISTIC", ""))
    return confidence



def _causes_from_shap(shap_row: pd.Series, terms) -> str:
    """Extract top causes from a Shapley attribution row."""
    # Map HEALTH_TERMS to their labels
    term_labels = {t.name: t.label for t in terms}
    # Get the top contributions
    top = shap_row.nlargest(3)
    causes = []
    for name, contrib in top.items():
        # shap_row index is like "health_shap_stress" -> extract "stress"
        term_name = name.replace("health_shap_", "").replace("risk_shap_", "")
        label = term_labels.get(term_name)
        if label and contrib > 0.5:  # Only include meaningful contributions
            causes.append(label)
    return ", ".join(causes) if causes else "normal battery usage"


def _primary_causes(row: pd.Series) -> str:
    """Evidence-based cause naming (used when attribution features are absent)."""
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
    """Overall confidence label: VALIDATED if temperature, HEURISTIC if only unvalidated, MIXED if both, N/A if none."""
    if causes_str == "normal battery usage":
        return "N/A"
    causes = [c.strip() for c in causes_str.split(",")]
    labels = {_CAUSE_EVIDENCE.get(c, ("HEURISTIC", ""))[0] for c in causes}
    if "N/A" in labels:
        labels.discard("N/A")
    if not labels:
        return "N/A"
    if len(labels) == 1:
        return labels.pop()
    return "MIXED"


def _evidence_note(causes_str: str) -> str:
    """Concatenate the evidence notes for all causes in the string."""
    if causes_str == "normal battery usage":
        return _CAUSE_EVIDENCE.get("normal battery usage", ("N/A", ""))[1]
    causes = [c.strip() for c in causes_str.split(",") if c.strip()]
    return " ".join([_CAUSE_EVIDENCE.get(c, ("", ""))[1] for c in causes if c in _CAUSE_EVIDENCE])


def generate_guardian_reports(
    battery: pd.DataFrame,
    reference: str = "fleet",
    top_n: int = 3,
) -> pd.DataFrame:
    """Add attribution-backed causes, recommendations and a narrative report.

    Expects the output of `rul.rul_estimation.compute_rul`.

    `reference` selects the attribution baseline (`"fleet"` compares each
    battery against the rest of the fleet; `"ideal"` compares it against a
    no-penalty reference). Single-battery inputs are forced to `"ideal"`,
    since a fleet mean over one row would make every attribution zero.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery.columns]
    if missing:
        raise ValueError(f"generate_guardian_reports: missing required columns {missing}")

    out = battery.copy()

    # Check if we have attribution features for Shapley decomposition.
    # If not, fall back to evidence-based cause naming.
    has_attribution = all(c in out.columns for c in ATTRIBUTION_FEATURES)

    if has_attribution:
        if len(out) < 2 and reference == "fleet":
            reference = "ideal"

        features = out[list(ATTRIBUTION_FEATURES)]

        # Attribute the health index using exact Shapley.
        health_raw = pd.Series(health_penalty_from_terms(out), index=out.index)
        out["health_index_saturated"] = (health_raw.round(6) != out["health_index"].round(6))

        health_expl = explain_scores(
            features, HEALTH_TERMS, health_raw,
            reference=reference, top_n=top_n, prefix="health_",
        )

        # Attribute the risk score separately.
        risk_expl = None
        if "risk_score" in out.columns:
            risk_raw = pd.Series(risk_score_from_terms(out), index=out.index)
            out["risk_score_saturated"] = (risk_raw.round(6) != out["risk_score"].round(6))
            risk_expl = explain_scores(
                features, RISK_TERMS, risk_raw,
                reference=reference, top_n=top_n, prefix="risk_",
            )

        out = out.join(health_expl)
        if risk_expl is not None:
            out = out.join(risk_expl)

        out["primary_causes"] = out["health_top_causes"]
        out["dominant_cause"] = out["health_dominant_cause"]
    else:
        # Fallback: use evidence-based cause naming when attribution features absent
        out["primary_causes"] = out.apply(_primary_causes, axis=1)
        out["dominant_cause"] = out["primary_causes"]

    out["evidence_confidence"] = out["primary_causes"].apply(_evidence_confidence)
    out["evidence_note"] = out["primary_causes"].apply(_evidence_note)
    out["guardian_status"] = out["battery_state"].map(_SEVERITY_MESSAGE).fillna("Unknown state")
    out["recommendation"] = out["battery_state"].map(_STATE_RECOMMENDATION).fillna(
        "No recommendation available"
    )

    out["guardian_report"] = (
        "Battery " + out["battery_id"].astype(str)
        + " is in " + out["battery_state"]
        + " state with estimated remaining life of " + out["rul_cycles"].astype(int).astype(str)
        + " cycles. Primary degradation factors include " + out["primary_causes"]
        + ". Recommended action: " + out["recommendation"]
    )
    return out
