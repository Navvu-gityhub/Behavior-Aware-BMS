"""Battery Guardian: plain-language cause attribution and recommendations.

WHAT CHANGED AND WHY
--------------------
Originally ported from `notebooks/08_battery_guardian.ipynb`, this module
explained a battery's state by re-deriving causes from its OWN if/else
thresholds (`avg_temp > 35`, `fast_charge_duration > 50`,
`deep_discharge_duration > 50`). Those numbers appeared nowhere else in the
codebase. The risk score bands sit at 30/40, 20/100 and 20/100; the health
index bands at the same cut points with strict inequalities. Guardian was
therefore explaining a score using a different rulebook from the one that
produced it, which allowed two concrete failure modes:

  * **Naming a cause that contributed nothing.** A battery with avg_temp of
    36 was told "high temperature exposure" while the temperature term had
    in fact placed it in its *lowest* band, contributing the minimum penalty.
  * **Omitting the term that dominated.** `avg_stress` can contribute up to
    30 points and `aggressive_discharge_count` up to 15, and neither
    appeared in Guardian's cause list at all.

Causes are now derived from exact Shapley attribution of the actual scores
(`explain.attribution`), so an explanation is arithmetically guaranteed to
be consistent with the number it explains, and causes are ranked by how many
points each contributed rather than listed in arbitrary order.

WHAT GUARDIAN CAN AND CANNOT CLAIM
----------------------------------
This is a boundary worth stating precisely, because the phrase "Guardian AI"
invites over-reading.

Guardian explains **why this battery scored what it scored**. That is now
exact. Guardian does NOT explain **why this battery is degrading**, and the
distinction is not pedantic: `scripts/validate_health_index_versions.py`
shows the v1 health index correlates with measured NASA fade at
rho = -0.27 (n=33, p=0.12) -- non-significant and pointing the wrong way.
Attributing a score that does not track real degradation yields a faithful
account of the rules and no account of the physics.

Guardian's output is therefore worded as score attribution, and
`guardian_caveat` carries that limitation into the output table itself
rather than leaving it in a document nobody reads next to the dashboard.
`docs/adr/0003-explainability-layer.md` records the reasoning.
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

# Actionable advice keyed to the term that actually dominated the score, so
# the recommendation addresses the driver rather than restating the state.
_CAUSE_ACTION = {
    "high temperature exposure": "Reduce thermal load: avoid charging immediately after "
                                 "sustained high-power use, and park out of direct sun where possible.",
    "deep discharge events": "Recharge before dropping below 20% state of charge.",
    "aggressive discharge": "Moderate hard acceleration and sustained high-current draw.",
    "frequent fast charging": "Prefer AC or slower DC charging for routine top-ups; "
                              "reserve high-rate DC for long trips.",
    "sustained high stress": "Combined thermal and current loading is elevated; "
                             "reduce high-power use in warm conditions.",
    "SOC held at extremes": "Keep the pack between 20% and 80% for daily use.",
    "normal usage": "Continue normal operation.",
}

GUARDIAN_CAVEAT = (
    "Attribution explains the rule-based score, not measured degradation. The "
    "underlying v1 health index is not validated against real capacity fade "
    "(Spearman rho = -0.27, n=33, p=0.12); see docs/final_report.md Section 4."
)


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

    attribution_missing = [c for c in ATTRIBUTION_FEATURES if c not in battery.columns]
    if attribution_missing:
        raise ValueError(
            f"generate_guardian_reports: missing columns needed for attribution "
            f"{attribution_missing}. Guardian derives causes from the risk/health score "
            "decomposition, so it needs the same per-battery summary features the "
            "scores were computed from."
        )

    out = battery.copy()

    if len(out) < 2 and reference == "fleet":
        reference = "ideal"

    features = out[list(ATTRIBUTION_FEATURES)]

    # Attribute the UNCLIPPED penalty sum, not the displayed index.
    #
    # `compute_health_index` clips the aging budget to [0, 100], so a battery
    # whose penalties total more than 100 shows a health index of 100 while
    # its true additive penalty is higher. Clipping is not an additive
    # operation, so attributing the clipped value would violate the
    # efficiency identity and `explain_scores` would (correctly) raise.
    #
    # Attributing the unclipped sum keeps the decomposition exact and keeps
    # the relative ordering of causes truthful, which is what the explanation
    # is for. The discrepancy is not hidden: `health_index_saturated` marks
    # affected batteries so a reader knows the displayed 100 is a floor on
    # severity rather than a measured ceiling.
    health_raw = pd.Series(health_penalty_from_terms(out), index=out.index)
    out["health_index_saturated"] = (health_raw.round(6) != out["health_index"].round(6))

    health_expl = explain_scores(
        features, HEALTH_TERMS, health_raw,
        reference=reference, top_n=top_n, prefix="health_",
    )

    # Attribute the risk score separately. The two scores use the same
    # features but different inequalities and are deliberately not merged
    # (see stress_score.py), so they get independent attributions rather
    # than one being presented as a proxy for the other.
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
    out["primary_cause_contributions"] = out["health_top_cause_contributions"]
    out["dominant_cause"] = out["health_dominant_cause"]
    out["guardian_status"] = out["battery_state"].map(_SEVERITY_MESSAGE).fillna("Unknown state")
    out["recommendation"] = out["battery_state"].map(_STATE_RECOMMENDATION).fillna(
        "No recommendation available"
    )
    out["targeted_action"] = out["dominant_cause"].map(_CAUSE_ACTION).fillna(
        "Monitor usage patterns."
    )
    out["guardian_caveat"] = GUARDIAN_CAVEAT

    out["guardian_report"] = (
        "Battery " + out["battery_id"].astype(str)
        + " is in " + out["battery_state"].astype(str)
        + " state with an estimated remaining life of "
        + out["rul_cycles"].astype(int).astype(str)
        + " cycles. Its health index of " + out["health_index"].round(0).astype(int).astype(str)
        + " is driven mainly by " + out["dominant_cause"].astype(str)
        + " (" + out["health_dominant_contribution"].round(1).astype(str)
        + " points above the " + out["health_attribution_reference"].astype(str)
        + " reference). Full attribution: " + out["health_top_cause_contributions"].astype(str)
        + ". Recommended action: " + out["targeted_action"].astype(str)
    )
    return out
