"""Battery Health Index via an aging-budget model.

Ported from `notebooks/06_battery_health_index.ipynb`. That notebook actually
defined `battery_state()` twice with different thresholds (cell 17 used a
WARNING cutoff of 40; cells 21-23, which ran last and produced the saved
CSV/docs, used 30). The saved output and `docs/weighted_health_index.md` both
reflect the 30 cutoff, so that is treated as the intended, canonical
definition here; the 40-cutoff version was dead code and is dropped.

The notebook also computed two different health indices in sequence — a
normalized weighted sum (cell 16: 0.35*stress + 0.25*temp + 0.15*dd +
0.15*fc + 0.10*soc_abuse) and then immediately overwrote it with
`100 - aging_budget` (cell 20 onward). Only the aging-budget version was
ever saved to `data/features/battery_health_index_v1.csv` or documented in
`docs/weighted_health_index.md`, so it is the one implemented here. The
weighted-sum version is not reproduced — keeping two silently-conflicting
health index definitions in the same module would be worse than dropping
the one that was never actually used downstream.

Like the risk score, the aging-budget penalty schedule below has hand-chosen
cut points and point deductions. It is not fit to data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.bms.explain.attribution import ScoreTerm

REQUIRED_COLUMNS = (
    "avg_stress",
    "avg_temp",
    "deep_discharge_duration",
    "fast_charge_duration",
    "aggressive_discharge_count",
    "avg_soc",
)


# ---------------------------------------------------------------------------
# Term specification
#
# `aging_budget` is 100 minus a sum of six independent per-feature penalties,
# and `health_index` is `100 - aging_budget`. So absent clipping,
# `health_index` is exactly the sum of the penalties below — an additive
# score, which is what lets `explain.attribution` decompose it into exact
# Shapley values. See that module's docstring for the derivation.
#
# These are defined once and consumed by both `compute_health_index` (which
# sums them) and the explainer (which decomposes them), for the same
# anti-drift reason documented in `risk.stress_score.RISK_TERMS`.
#
# The cut points are hand-chosen, NOT fit to data — that is the whole reason
# `health_index_v2` exists as a fitted alternative. See
# docs/adr/0002-health-index-version.md for why v1 nonetheless remains the
# pipeline default.
#
# These bands use `>` where the structurally identical bands in
# `risk.stress_score.RISK_TERMS` use `>=`; see the note there.
# ---------------------------------------------------------------------------

HEALTH_TERMS: tuple[ScoreTerm, ...] = (
    ScoreTerm(
        name="stress",
        feature="avg_stress",
        label="sustained high stress",
        fn=lambda s: np.where(s > 70, 30, np.where(s > 50, 20, 10)),
    ),
    ScoreTerm(
        name="temperature",
        feature="avg_temp",
        label="high temperature exposure",
        fn=lambda s: np.where(s > 40, 25, np.where(s > 30, 15, 5)),
    ),
    ScoreTerm(
        name="deep_discharge",
        feature="deep_discharge_duration",
        label="deep discharge events",
        fn=lambda s: np.where(s > 100, 20, np.where(s > 20, 10, 5)),
    ),
    ScoreTerm(
        name="fast_charge",
        feature="fast_charge_duration",
        label="frequent fast charging",
        fn=lambda s: np.where(s > 100, 15, np.where(s > 20, 8, 2)),
    ),
    ScoreTerm(
        name="aggressive_discharge",
        feature="aggressive_discharge_count",
        label="aggressive discharge",
        fn=lambda s: np.where(s > 500, 15, np.where(s > 100, 8, 2)),
    ),
    ScoreTerm(
        name="soc_extremes",
        feature="avg_soc",
        label="SOC held at extremes",
        fn=lambda s: np.where((s > 80) | (s < 20), 10, 0),
    ),
)


def health_penalty_from_terms(battery_summary: pd.DataFrame) -> np.ndarray:
    """Sum `HEALTH_TERMS` into the total aging penalty deducted from the budget."""
    total = np.zeros(len(battery_summary), dtype=float)
    for term in HEALTH_TERMS:
        total += term.contribution(battery_summary[term.feature]).to_numpy(dtype=float)
    return total


def _battery_state(health_index: float) -> str:
    if health_index >= 80:
        return "CRITICAL"
    if health_index >= 60:
        return "DEGRADED"
    if health_index >= 30:
        return "WARNING"
    return "HEALTHY"


def compute_health_index(battery_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute aging_budget, health_index, remaining_health, consumed_life, battery_state.

    Expects the per-battery summary table from
    `features.behavior_features.summarize_batteries`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery_summary.columns]
    if missing:
        raise ValueError(f"compute_health_index: missing required columns {missing}")

    nulls = {c: int(battery_summary[c].isna().sum()) for c in REQUIRED_COLUMNS if battery_summary[c].isna().any()}
    if nulls:
        # See identical guard and rationale in risk.stress_score.compute_risk_assessment.
        raise ValueError(
            f"compute_health_index: NaN values in required columns {nulls} — "
            "would silently score as low-risk rather than reflect missing data. "
            "Impute or explicitly exclude these rows/columns before scoring."
        )

    out = battery_summary.copy()

    budget = 100.0 - health_penalty_from_terms(out)

    out["aging_budget"] = np.clip(budget, 0, 100)
    out["health_index"] = 100 - out["aging_budget"]
    out["remaining_health"] = out["aging_budget"]
    out["consumed_life"] = 100 - out["remaining_health"]
    out["battery_state"] = out["health_index"].apply(_battery_state)
    return out
