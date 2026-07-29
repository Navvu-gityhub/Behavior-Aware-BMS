"""Tests for the adaptive calibration foundation.

The load-bearing test in this file is
`test_gate_rejects_the_v2_model_on_real_nasa_data`. It runs the gate against
the actual shipped v2 specification on the real 2,682-observation NASA frame
and asserts it is REJECTED. That model is the one `docs/adr/0002` shows scores
rho=0.841 within a known protocol and rho=-0.295 on an unseen one, so if the
gate ever promotes it, the gate is broken and the adaptive system would be
automating protocol memorisation.

`test_gate_requires_loco_even_when_lobo_is_excellent` covers the same failure
from the other direction: a candidate with strong LOBO and no LOCO evidence
must not slip through.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.cohort import (
    DEFAULT_TOLERANCE,
    CohortRegistry,
    CohortSpec,
    DriftReport,
    InDistribution,
)
from src.bms.adaptive.validation import (
    CrossValidationResult,
    FoldResult,
    Validator,
    r2_against,
)

REPO = Path(__file__).resolve().parents[1]
NASA_TRAINING = REPO / "reports/metrics/continuous_model_training_data.csv"


def _synthetic(n_cohorts: int = 3, cells_per_cohort: int = 4, rows: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    frames = []
    for c in range(n_cohorts):
        base_temp = 10 + 15 * c
        for k in range(cells_per_cohort):
            cell = f"C{c}_{k}"
            temp = base_temp + rng.normal(0, 1.5, rows)
            frames.append(pd.DataFrame({
                "cell_id": cell,
                "cohort": f"COHORT_{c}",
                "cycle": np.arange(1, rows + 1),
                "avg_temp": temp,
                "max_temp": temp + rng.uniform(2, 5, rows),
                "avg_soc": rng.uniform(40, 60, rows),
                "aggressive_discharge_count": rng.integers(0, 50, rows).astype(float),
                "trailing_avg_temp": temp,
                # Target genuinely driven by temperature, identically across
                # cohorts, so a temperature model *should* generalise here.
                "capacity_loss": 0.004 * temp + rng.normal(0, 0.002, rows),
            }))
    return pd.concat(frames, ignore_index=True)


def _temperature_fit(train: pd.DataFrame):
    """A candidate with real cross-protocol signal: slope on temperature only."""
    slope, intercept = np.polyfit(train["trailing_avg_temp"], train["capacity_loss"], 1)
    return lambda test: intercept + slope * test["trailing_avg_temp"].to_numpy(float)


def _cohort_intercept_fit(train: pd.DataFrame):
    """A candidate that memorises cohorts and has nothing else.

    Predicts each cohort's training mean, and falls back to the global mean for
    a cohort it never saw. This is a deliberately faithful caricature of what
    ADR 0002 found the v2 model to be doing.
    """
    means = train.groupby("cohort")["capacity_loss"].mean().to_dict()
    global_mean = float(train["capacity_loss"].mean())
    return lambda test: test["cohort"].map(means).fillna(global_mean).to_numpy(float)


# ---------------------------------------------------------------------------
# Batch 1: cohort envelopes
# ---------------------------------------------------------------------------

def test_envelope_is_derived_from_observations_not_declared():
    data = _synthetic(n_cohorts=1)
    spec = CohortSpec.from_observations("COHORT_0", data)
    low, high = spec.bounds["avg_temp"]
    assert low == pytest.approx(data["avg_temp"].min())
    assert high == pytest.approx(data["avg_temp"].max())
    assert spec.n_cells == 4


def test_missing_envelope_feature_raises_rather_than_guessing():
    data = _synthetic(n_cohorts=1).drop(columns=["max_temp"])
    with pytest.raises(ValueError, match="missing envelope features"):
        CohortSpec.from_observations("COHORT_0", data)


def test_registry_identifies_a_familiar_operating_point():
    data = _synthetic()
    registry = CohortRegistry.from_training_data(data)
    assert len(registry) == 3

    warm = data[data["cohort"] == "COHORT_2"]
    observation = {f: float(warm[f].mean()) for f in
                   ("avg_temp", "max_temp", "avg_soc", "aggressive_discharge_count")}
    result = registry.identify(observation)
    assert isinstance(result, InDistribution)
    assert bool(result) is True
    assert result.cohort_id == "COHORT_2"


def test_registry_refuses_an_unseen_regime_and_says_why():
    """The whole point: novel conditions must not be silently scored."""
    registry = CohortRegistry.from_training_data(_synthetic())
    arctic = {"avg_temp": -25.0, "max_temp": -18.0,
              "avg_soc": 50.0, "aggressive_discharge_count": 10.0}

    result = registry.identify(arctic)
    assert isinstance(result, DriftReport)
    assert bool(result) is False
    assert "below observed min" in result.summary()
    assert result.nearest_cohort is not None


def test_zero_width_envelope_does_not_reject_everything():
    """`fast_charge_duration` is identically zero across all NASA data.

    A fractional tolerance on a zero-width range is zero, which would make any
    value at all a violation. The absolute fallback prevents that.
    """
    data = _synthetic(n_cohorts=1)
    data["aggressive_discharge_count"] = 0.0
    spec = CohortSpec.from_observations("FLAT", data)
    assert spec.bounds["aggressive_discharge_count"] == (0.0, 0.0)

    observation = {f: float(data[f].mean()) for f in
                   ("avg_temp", "max_temp", "avg_soc")}
    observation["aggressive_discharge_count"] = 0.0
    assert spec.contains(observation)


def test_screen_classifies_each_cell():
    data = _synthetic()
    registry = CohortRegistry.from_training_data(data)

    intruder = data[data["cell_id"] == "C0_0"].copy()
    intruder["cell_id"] = "ARCTIC"
    intruder["avg_temp"] -= 60
    intruder["max_temp"] -= 60

    report = registry.screen(pd.concat([data, intruder], ignore_index=True))
    arctic_row = report[report["cell_id"] == "ARCTIC"].iloc[0]
    assert arctic_row["status"] == "OUT_OF_DISTRIBUTION"
    assert (report[report["cell_id"] != "ARCTIC"]["status"] == "IN_DISTRIBUTION").all()


def test_registry_requires_cohort_labels():
    with pytest.raises(ValueError, match="no 'cohort' column"):
        CohortRegistry.from_training_data(_synthetic().drop(columns=["cohort"]))


# ---------------------------------------------------------------------------
# Batch 2: the gate
# ---------------------------------------------------------------------------

def test_r2_is_measured_against_the_supplied_baseline():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = y.copy()
    baseline = np.full_like(y, y.mean())
    assert r2_against(y, perfect, baseline) == pytest.approx(1.0)
    assert r2_against(y, baseline, baseline) == pytest.approx(0.0)


def test_r2_returns_nan_when_the_baseline_is_already_perfect():
    y = np.array([2.0, 2.0, 2.0])
    baseline = y.copy()
    assert np.isnan(r2_against(y, y, baseline))


def test_gate_promotes_a_candidate_with_real_cross_protocol_signal():
    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    lobo = validator.cross_validate(_temperature_fit, group_col="cell_id", split="LOBO")
    loco = validator.cross_validate(_temperature_fit, group_col="cohort", split="LOCO")

    verdict = validator.gate(lobo, loco)
    assert verdict.promote is True
    assert verdict.status == "PROMOTE"


def test_gate_rejects_a_cohort_memoriser():
    """A model that only knows per-cohort means must fail on an unseen cohort."""
    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    lobo = validator.cross_validate(_cohort_intercept_fit, group_col="cell_id", split="LOBO")
    loco = validator.cross_validate(_cohort_intercept_fit, group_col="cohort", split="LOCO")

    verdict = validator.gate(lobo, loco)
    assert verdict.promote is False
    assert any("unseen protocol" in r for r in verdict.reasons)
    # And LOBO should have looked fine, which is exactly why LOBO alone is unsafe.
    assert lobo.median_r2 > loco.median_r2


def test_gate_requires_loco_even_when_lobo_is_excellent():
    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    lobo = validator.cross_validate(_temperature_fit, group_col="cell_id", split="LOBO")
    assert lobo.median_r2 > 0  # genuinely good

    verdict = validator.gate(lobo, loco=None)
    assert verdict.promote is False
    assert any("LOCO is mandatory" in r for r in verdict.reasons)


def test_a_candidate_that_crashes_is_not_promoted():
    def broken(train):
        raise RuntimeError("fit failed")

    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    result = validator.cross_validate(broken, group_col="cohort", split="LOCO")
    assert all(f.error is not None for f in result.folds)
    assert not result.completed
    assert validator.gate(result, result).promote is False


def test_shape_mismatch_is_caught_rather_than_broadcast():
    """A predictor returning the wrong length must fail loudly."""
    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    result = validator.cross_validate(
        lambda train: (lambda test: np.zeros(3)), group_col="cohort", split="LOCO"
    )
    assert all("shape" in (f.error or "") for f in result.folds)


def test_folds_too_small_are_skipped_with_a_recorded_reason():
    data = _synthetic(n_cohorts=2, cells_per_cohort=2, rows=4)
    validator = Validator(data, target="capacity_loss")
    result = validator.cross_validate(_temperature_fit, group_col="cell_id", split="LOBO")
    assert any("below minimum" in (f.error or "") for f in result.folds)


def test_verdict_records_reasons_for_promotions_too():
    data = _synthetic()
    validator = Validator(data, target="capacity_loss")
    lobo = validator.cross_validate(_temperature_fit, group_col="cell_id", split="LOBO")
    loco = validator.cross_validate(_temperature_fit, group_col="cohort", split="LOCO")
    verdict = validator.gate(lobo, loco)
    assert verdict.reasons
    assert "PROMOTE" in verdict.render()


# ---------------------------------------------------------------------------
# The regression test that matters: real data, real model, known answer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NASA_TRAINING.exists(), reason="NASA training frame not present")
def test_gate_rejects_the_v2_model_on_real_nasa_data():
    """The shipped v2 specification must be REJECTED by the gate.

    ADR 0002: rho=0.841 holding out a cell, rho=-0.295 holding out a protocol.
    If this ever passes, the gate has stopped protecting against the exact
    failure the adaptive system exists to prevent.
    """
    smf = pytest.importorskip("statsmodels.formula.api")
    data = pd.read_csv(NASA_TRAINING)
    assert {"cell_id", "cohort", "capacity_loss", "trailing_avg_temp"} <= set(data.columns)

    def v2_fit(train: pd.DataFrame):
        model = smf.ols("capacity_loss ~ trailing_avg_temp + C(cohort)", data=train).fit()

        def predict(test: pd.DataFrame) -> np.ndarray:
            frame = test.copy()
            # A held-out cohort has no fitted intercept; substitute a seen one,
            # which is what the deployed module does for unknown cohorts.
            unseen = ~frame["cohort"].isin(train["cohort"].unique())
            if unseen.any():
                frame.loc[unseen, "cohort"] = train["cohort"].iloc[0]
            return model.predict(frame).to_numpy(dtype=float)

        return predict

    validator = Validator(data, target="capacity_loss")
    lobo = validator.cross_validate(v2_fit, group_col="cell_id", split="LOBO")
    loco = validator.cross_validate(v2_fit, group_col="cohort", split="LOCO")
    verdict = validator.gate(lobo, loco)

    assert verdict.promote is False, (
        f"Gate promoted the v2 model, which ADR 0002 shows collapses on unseen "
        f"protocols. Verdict:\n{verdict.render()}"
    )
    assert loco.median_r2 < lobo.median_r2
