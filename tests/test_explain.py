"""Tests for the explainability layer.

The load-bearing test here is `test_efficiency_axiom_*`: Shapley values must
sum to `f(x) - E[f(X)]`. Because that identity is guaranteed by the maths for
an additive score, any failure means the term specification in
`RISK_TERMS`/`HEALTH_TERMS` has drifted from the scorer it claims to
decompose — which is exactly the class of bug that made the old Guardian
explain scores using thresholds that existed nowhere else.

`test_refactor_is_score_identical_on_real_nasa_data` is the other one worth
reading: it pins the refactored term-based scorers against the committed
pre-refactor NASA outputs, so "we extracted the terms without changing any
score" is enforced rather than asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.explain.attribution import (
    IDEAL_REFERENCE,
    ScoreTerm,
    additive_shapley_values,
    explain_scores,
    rank_attributions,
    verify_efficiency,
)
from src.bms.guardian.guardian import generate_guardian_reports
from src.bms.health.health_index import (
    HEALTH_TERMS,
    compute_health_index,
    health_penalty_from_terms,
)
from src.bms.risk.stress_score import (
    RISK_TERMS,
    compute_risk_assessment,
    risk_score_from_terms,
)

REPO = Path(__file__).resolve().parents[1]
SUMMARY_COLS = [
    "avg_stress", "avg_temp", "deep_discharge_duration",
    "fast_charge_duration", "aggressive_discharge_count", "avg_soc",
]


def _fleet(n: int = 8) -> pd.DataFrame:
    """A small synthetic fleet spanning every penalty band of every term."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "battery_id": [f"B{i:03d}" for i in range(n)],
        "avg_stress": rng.uniform(5, 90, n),
        "avg_temp": rng.uniform(15, 50, n),
        "deep_discharge_duration": rng.integers(0, 300, n).astype(float),
        "fast_charge_duration": rng.integers(0, 300, n).astype(float),
        "aggressive_discharge_count": rng.integers(0, 900, n).astype(float),
        "avg_soc": rng.uniform(10, 95, n),
    })


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------

def test_efficiency_axiom_holds_for_risk_score():
    fleet = _fleet()
    scores = pd.Series(risk_score_from_terms(fleet), index=fleet.index)
    verify_efficiency(fleet[SUMMARY_COLS], RISK_TERMS, scores, reference="fleet")


def test_efficiency_axiom_holds_for_health_index():
    fleet = _fleet()
    scores = pd.Series(health_penalty_from_terms(fleet), index=fleet.index)
    verify_efficiency(fleet[SUMMARY_COLS], HEALTH_TERMS, scores, reference="fleet")


def test_efficiency_axiom_holds_against_ideal_reference():
    fleet = _fleet()
    scores = pd.Series(health_penalty_from_terms(fleet), index=fleet.index)
    verify_efficiency(fleet[SUMMARY_COLS], HEALTH_TERMS, scores, reference="ideal")


def test_shapley_values_sum_to_deviation_from_fleet_mean():
    """The efficiency identity, checked numerically rather than via the helper."""
    fleet = _fleet(12)
    scores = pd.Series(health_penalty_from_terms(fleet), index=fleet.index)
    phi = additive_shapley_values(fleet[SUMMARY_COLS], HEALTH_TERMS, reference="fleet")
    np.testing.assert_allclose(
        phi.sum(axis=1).to_numpy(), (scores - scores.mean()).to_numpy(), atol=1e-9
    )


def test_efficiency_check_catches_term_drift():
    """A term whose definition no longer matches the scorer must be caught.

    This simulates the exact failure mode the shared term spec exists to
    prevent: someone edits a threshold in one place and not the other.
    """
    fleet = _fleet()
    # The moved cut point only matters for a battery sitting between the old
    # and new thresholds. Without one, the drifted spec scores identically and
    # the check has nothing to catch — so plant one explicitly rather than
    # relying on the random fleet to contain it.
    fleet.loc[0, "avg_temp"] = 37.0
    scores = pd.Series(risk_score_from_terms(fleet), index=fleet.index)

    drifted = list(RISK_TERMS)
    drifted[1] = ScoreTerm(
        name="temperature", feature="avg_temp", label="high temperature exposure",
        # Cut point moved 40 -> 35; the scorer still uses 40.
        fn=lambda s: np.where(s >= 35, 25, np.where(s >= 30, 15, 5)),
    )
    with pytest.raises(AssertionError, match="efficiency violated"):
        verify_efficiency(fleet[SUMMARY_COLS], tuple(drifted), scores)


def test_ideal_reference_gives_zero_attribution_to_a_perfect_battery():
    perfect = pd.DataFrame([IDEAL_REFERENCE] * 3)
    phi = additive_shapley_values(perfect, HEALTH_TERMS, reference="ideal")
    np.testing.assert_allclose(phi.to_numpy(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_fleet_reference_rejects_single_battery():
    """One row makes every fleet-relative attribution trivially zero."""
    with pytest.raises(ValueError, match="at least 2 batteries"):
        additive_shapley_values(_fleet(1)[SUMMARY_COLS], HEALTH_TERMS, reference="fleet")


def test_missing_feature_column_raises():
    fleet = _fleet().drop(columns=["avg_temp"])
    with pytest.raises(ValueError, match="missing feature columns"):
        additive_shapley_values(fleet, HEALTH_TERMS)


def test_unknown_reference_raises():
    with pytest.raises(ValueError, match="Unknown reference"):
        additive_shapley_values(_fleet()[SUMMARY_COLS], HEALTH_TERMS, reference="nonsense")


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def test_ranking_orders_causes_by_magnitude():
    fleet = _fleet(10)
    phi = additive_shapley_values(fleet[SUMMARY_COLS], HEALTH_TERMS)
    ranked = rank_attributions(phi, HEALTH_TERMS, top_n=3)

    for idx in phi.index:
        dominant = ranked.loc[idx, "dominant_cause"]
        if dominant == "normal usage":
            assert phi.loc[idx].max() <= 0.5
        else:
            labels = {t.label: t.name for t in HEALTH_TERMS}
            assert phi.loc[idx, labels[dominant]] == pytest.approx(phi.loc[idx].max())


def test_ranking_suppresses_noise_level_contributions():
    """A term contributing under the floor must not be called a primary cause."""
    phi = pd.DataFrame({"stress": [0.2], "temperature": [0.1], "deep_discharge": [0.05],
                        "fast_charge": [0.0], "aggressive_discharge": [0.0],
                        "soc_extremes": [0.0]})
    ranked = rank_attributions(phi, HEALTH_TERMS, min_magnitude=0.5)
    assert ranked.loc[0, "dominant_cause"] == "normal usage"
    assert ranked.loc[0, "n_significant_drivers"] == 0


# ---------------------------------------------------------------------------
# Refactor regression: the extraction must not have moved any score
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (REPO / "reports/metrics/calibration_merged.csv").exists(),
    reason="NASA calibration artifacts not present",
)
def test_refactor_is_score_identical_on_real_nasa_data():
    """Term extraction must reproduce the committed pre-refactor NASA scores.

    `calibration_merged.csv` was produced by the inlined arithmetic that the
    shared term specs replaced. Any divergence means the refactor silently
    changed a published number.
    """
    gold = pd.read_csv(REPO / "reports/metrics/calibration_merged.csv")
    summary = gold[["battery_id"] + SUMMARY_COLS].copy()

    risk = compute_risk_assessment(summary)
    health = compute_health_index(summary)

    np.testing.assert_allclose(risk["risk_score"].to_numpy(float),
                               gold["risk_score"].to_numpy(float))
    np.testing.assert_allclose(health["health_index"].to_numpy(float),
                               gold["health_index"].to_numpy(float))
    np.testing.assert_allclose(health["aging_budget"].to_numpy(float),
                               gold["aging_budget"].to_numpy(float))
    assert (risk["risk_level"].astype(str).to_numpy()
            == gold["risk_level"].astype(str).to_numpy()).all()
    assert (health["battery_state"].astype(str).to_numpy()
            == gold["battery_state"].astype(str).to_numpy()).all()


# ---------------------------------------------------------------------------
# Guardian integration
# ---------------------------------------------------------------------------

def _guardian_input(n: int = 6) -> pd.DataFrame:
    fleet = _fleet(n)
    risk = compute_risk_assessment(fleet)
    health = compute_health_index(fleet)
    merged = risk.merge(
        health[["battery_id", "aging_budget", "health_index",
                "remaining_health", "consumed_life", "battery_state"]],
        on="battery_id",
    )
    merged["rul_cycles"] = 500
    return merged


def test_guardian_causes_are_consistent_with_the_health_index():
    """Guardian's named cause must be the term that actually dominated.

    This is the specific defect the rewrite fixed: the old implementation
    could name a cause contributing nothing, because it used its own
    thresholds instead of the score's.
    """
    out = generate_guardian_reports(_guardian_input(10))
    shap_cols = [c for c in out.columns if c.startswith("health_shap_")]

    for _, row in out.iterrows():
        if row["dominant_cause"] == "normal usage":
            assert row[shap_cols].max() <= 0.5
            continue
        label_to_col = {t.label: f"health_shap_{t.name}" for t in HEALTH_TERMS}
        assert row[label_to_col[row["dominant_cause"]]] == pytest.approx(row[shap_cols].max())


def test_guardian_attribution_sums_to_unclipped_penalty_deviation():
    """Attribution decomposes the unclipped penalty, which is the additive quantity.

    The displayed `health_index` is clipped to [0, 100]; clipping is not
    additive, so the identity is asserted against the raw penalty sum. See
    the saturation comment in guardian.py.
    """
    out = generate_guardian_reports(_guardian_input(10))
    shap_cols = [c for c in out.columns if c.startswith("health_shap_")]
    raw = pd.Series(health_penalty_from_terms(out), index=out.index)
    np.testing.assert_allclose(
        out[shap_cols].sum(axis=1).to_numpy(),
        (raw - raw.mean()).to_numpy(),
        atol=1e-9,
    )


def test_guardian_flags_saturated_scores():
    """A battery whose penalties exceed 100 must be marked, not silently capped."""
    fleet = _fleet(4)
    # Force every term into its worst band: 30+25+20+15+15+10 = 115 > 100.
    fleet.loc[0, SUMMARY_COLS] = [95.0, 50.0, 400.0, 400.0, 900.0, 95.0]
    risk = compute_risk_assessment(fleet)
    health = compute_health_index(fleet)
    merged = risk.merge(
        health[["battery_id", "aging_budget", "health_index",
                "remaining_health", "consumed_life", "battery_state"]],
        on="battery_id",
    )
    merged["rul_cycles"] = 500

    out = generate_guardian_reports(merged)
    assert out.loc[0, "health_index"] == 100.0
    assert bool(out.loc[0, "health_index_saturated"]) is True
    assert not out.loc[1:, "health_index_saturated"].any()


def test_guardian_single_battery_falls_back_to_ideal_reference():
    out = generate_guardian_reports(_guardian_input(1))
    assert out.loc[0, "health_attribution_reference"] == "ideal"



