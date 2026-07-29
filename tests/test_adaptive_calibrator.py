"""Tests for the calibration orchestrator.

The tests that matter most here are about **refusal**, not success:

`test_score_refuses_when_no_model_passed_the_gate` pins the behaviour in the
system's current and expected steady state. Nothing has been promoted, and the
correct response to "what is this battery's fade?" is a reasoned refusal, not
the most recent rejected candidate's guess.

`test_score_refuses_a_battery_outside_every_known_protocol` pins the other
refusal. ADR 0002 measured what the model does outside its training protocols
(rho=-0.295), so extrapolating there is worse than declining.

`test_a_missing_feature_refuses_rather_than_defaulting_to_zero` is the same
principle one level down, and the same bug the NaN-as-healthy fix addressed
elsewhere in this project: absent is not zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.calibrator import (
    AdaptiveCalibrator,
    CandidateSpec,
    ScoringRefusal,
    Scored,
    linear_candidate,
)
from src.bms.adaptive.datasets import CallableDatasetLoader, DatasetRegistry
from src.bms.adaptive.store import ModelStore
from src.bms.adaptive.validation import Verdict

REPO = Path(__file__).resolve().parents[1]
NASA_TRAINING = REPO / "reports/metrics/continuous_model_training_data.csv"

ENVELOPE = ("avg_temp", "max_temp", "avg_soc", "aggressive_discharge_count")


def _learnable(n_cohorts: int = 3, cells: int = 4, cycles: int = 40) -> pd.DataFrame:
    """A dataset where temperature genuinely drives fade in every cohort.

    Built so a temperature model *should* pass, which is what makes the
    rejection tests meaningful: the harness is capable of promoting.
    """
    rng = np.random.default_rng(31)
    frames = []
    for c in range(n_cohorts):
        base = 10 + 14 * c
        for k in range(cells):
            temp = base + rng.normal(0, 1.2, cycles)
            frames.append(pd.DataFrame({
                "cell_id": f"C{c}_{k}",
                "cohort": f"COHORT_{c}",
                "cycle": np.arange(1, cycles + 1),
                "capacity_ah": 2.0 - 0.003 * np.arange(1, cycles + 1),
                "avg_temp": temp,
                "max_temp": temp + rng.uniform(2, 4, cycles),
                "avg_soc": rng.uniform(40, 60, cycles),
                "aggressive_discharge_count": rng.integers(0, 40, cycles).astype(float),
                "trailing_avg_temp": temp,
                "capacity_loss": 0.004 * temp + rng.normal(0, 0.0015, cycles),
            }))
    return pd.concat(frames, ignore_index=True)


def _unusable() -> pd.DataFrame:
    """Single-cycle data: the CALCE shape."""
    return pd.DataFrame({
        "cell_id": [f"PL{i}" for i in range(50)],
        "cohort": "CALCE",
        "cycle": 1,
        "capacity_ah": np.linspace(1.0, 1.1, 50),
        "capacity_loss": np.linspace(0.0, 0.1, 50),
        "avg_temp": 25.0, "max_temp": 28.0, "avg_soc": 50.0,
        "aggressive_discharge_count": 5.0, "trailing_avg_temp": 25.0,
    })


@pytest.fixture()
def calibrator(tmp_path) -> AdaptiveCalibrator:
    datasets = DatasetRegistry()
    datasets.register(CallableDatasetLoader("learnable", _learnable))
    datasets.register(CallableDatasetLoader("calce", _unusable))
    return AdaptiveCalibrator(
        store=ModelStore(tmp_path / "store"), datasets=datasets
    )


def _observation(data: pd.DataFrame, cohort: str) -> dict[str, float]:
    subset = data[data["cohort"] == cohort]
    row = {f: float(subset[f].mean()) for f in ENVELOPE}
    row["trailing_avg_temp"] = float(subset["trailing_avg_temp"].mean())
    return row


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def test_calibrate_promotes_a_candidate_with_real_signal(calibrator):
    run = calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    assert run.aborted is None
    assert len(run.promoted) == 1
    assert calibrator.store.active_version("GLOBAL") == 1


def test_calibrate_aborts_on_an_unusable_dataset(calibrator):
    """Unusable data must not reach a model-fitting step at all."""
    run = calibrator.calibrate(
        "calce", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    assert run.aborted is not None
    assert run.outcomes == ()
    assert calibrator.store.active_version("GLOBAL") is None


def test_cohort_envelopes_are_learned_during_calibration(calibrator):
    run = calibrator.calibrate("learnable", [])
    assert len(run.cohorts_registered) == 3
    assert len(calibrator.cohorts) == 3


def test_every_candidate_is_evaluated_not_just_until_one_passes(calibrator):
    """A rejected alternative is evidence about the promoted one."""
    run = calibrator.calibrate("learnable", [
        linear_candidate("temp_only", ["trailing_avg_temp"]),
        linear_candidate("soc_only", ["avg_soc"]),
    ])
    assert len(run.outcomes) == 2
    assert {o.name for o in run.outcomes} == {"temp_only", "soc_only"}


def test_a_candidate_without_signal_is_rejected(calibrator):
    run = calibrator.calibrate("learnable", [linear_candidate("soc_only", ["avg_soc"])])
    outcome = run.outcomes[0]
    assert outcome.promoted is False
    assert outcome.verdict.promote is False


def test_a_crashing_candidate_is_recorded_not_raised(calibrator):
    def broken_fit(train):
        raise RuntimeError("singular matrix")

    spec = CandidateSpec("broken", broken_fit, lambda d: {"x": 1.0})
    run = calibrator.calibrate("learnable", [spec])
    outcome = run.outcomes[0]
    assert outcome.promoted is False
    # Folds fail individually, so the gate rejects for want of completed folds.
    assert outcome.verdict.promote is False


def test_parameters_are_only_extracted_for_promoted_candidates(calibrator):
    """Fitting a rejected candidate wastes time and creates a storable artifact."""
    calls: list[int] = []

    def counting_extract(data):
        calls.append(1)
        return {"avg_soc": 0.1, "intercept": 0.0}

    spec = CandidateSpec(
        "soc_only",
        linear_candidate("soc_only", ["avg_soc"]).fit_fn,
        counting_extract,
    )
    run = calibrator.calibrate("learnable", [spec])
    assert run.outcomes[0].promoted is False
    assert calls == []


def test_run_render_states_when_nothing_was_promoted(calibrator):
    run = calibrator.calibrate("learnable", [linear_candidate("soc_only", ["avg_soc"])])
    assert "No candidate was promoted" in run.render()


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------

def test_score_refuses_before_any_calibration(calibrator):
    result = calibrator.score({"avg_temp": 25.0}, battery_id="B1")
    assert isinstance(result, ScoringRefusal)
    assert bool(result) is False
    assert "no cohorts registered" in result.reason


def test_score_refuses_when_no_model_passed_the_gate(calibrator):
    """The current steady state: envelopes learned, nothing promoted."""
    data = _learnable()
    calibrator.calibrate("learnable", [linear_candidate("soc_only", ["avg_soc"])])

    result = calibrator.score(_observation(data, "COHORT_1"), battery_id="B1")
    assert isinstance(result, ScoringRefusal)
    assert "no model has passed the promotion gate" in result.reason
    assert result.prediction is None
    # And it must not have quietly used the rejected candidate.
    assert calibrator.store.active_version("GLOBAL") is None


def test_score_refuses_a_battery_outside_every_known_protocol(calibrator):
    calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    arctic = {"avg_temp": -30.0, "max_temp": -22.0, "avg_soc": 50.0,
              "aggressive_discharge_count": 5.0, "trailing_avg_temp": -30.0}

    result = calibrator.score(arctic, battery_id="ARCTIC")
    assert isinstance(result, ScoringRefusal)
    assert "outside every known protocol" in result.reason
    assert "below observed min" in result.detail


def test_a_missing_feature_refuses_rather_than_defaulting_to_zero(calibrator):
    """Absent is not zero. Same principle as the NaN-as-healthy fix."""
    data = _learnable()
    calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    observation = _observation(data, "COHORT_1")
    del observation["trailing_avg_temp"]

    result = calibrator.score(observation, battery_id="B1")
    assert isinstance(result, ScoringRefusal)
    assert "does not cover this battery's features" in result.reason


def test_a_nan_feature_refuses(calibrator):
    data = _learnable()
    calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    observation = _observation(data, "COHORT_1")
    observation["trailing_avg_temp"] = float("nan")
    assert isinstance(calibrator.score(observation), ScoringRefusal)


# ---------------------------------------------------------------------------
# Successful scoring carries its evidence
# ---------------------------------------------------------------------------

def test_a_score_carries_the_evidence_that_licenses_it(calibrator):
    data = _learnable()
    calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    result = calibrator.score(_observation(data, "COHORT_1"), battery_id="B1")

    assert isinstance(result, Scored)
    assert bool(result) is True
    assert result.cohort_id == "COHORT_1"
    assert result.model_version == 1
    assert result.loco_median_r2 is not None
    assert np.isfinite(result.prediction)


def test_screen_classifies_cells_after_calibration(calibrator):
    data = _learnable()
    calibrator.calibrate("learnable", [])
    report = calibrator.screen(data)
    assert (report["status"] == "IN_DISTRIBUTION").all()


def test_screen_before_calibration_raises(calibrator):
    with pytest.raises(RuntimeError, match="no cohorts registered"):
        calibrator.screen(_learnable())


# ---------------------------------------------------------------------------
# Against the real NASA frame
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NASA_TRAINING.exists(), reason="NASA training frame not present")
def test_nothing_is_promoted_on_the_real_nasa_data(tmp_path):
    """The steady state, verified end to end on real measurements.

    If a candidate is ever promoted here it is a finding worth investigating,
    not a test to relax: ADR 0002 established that this data does not support
    a generalising fade model.
    """
    datasets = DatasetRegistry()
    datasets.register(CallableDatasetLoader(
        "nasa", lambda: pd.read_csv(NASA_TRAINING)
    ))
    calibrator = AdaptiveCalibrator(
        store=ModelStore(tmp_path / "store"), datasets=datasets
    )

    run = calibrator.calibrate("nasa", [
        linear_candidate("temp_only", ["trailing_avg_temp"]),
        linear_candidate("temp_and_stress", ["trailing_avg_temp", "avg_stress"]),
    ])

    assert run.aborted is None
    assert run.cohorts_registered  # envelopes still learned
    assert not run.promoted, f"unexpected promotion:\n{run.render()}"
    assert calibrator.store.active_version("GLOBAL") is None

    # Rejections are still recorded, which is the point of the log.
    decisions = list(calibrator.store.decisions("GLOBAL"))
    assert decisions == [] or all(d.outcome == "REJECT" for d in decisions)


@pytest.mark.skipif(not NASA_TRAINING.exists(), reason="NASA training frame not present")
def test_a_candidate_whose_skill_is_age_is_rejected(tmp_path):
    """Beating a constant is a weak claim when everything rises with cycle count.

    `avg_soc` has a median within-cell Spearman correlation of -0.73 with
    cycle index (27 of 32 cells above 0.5 in magnitude), so it is largely a
    proxy for how far through its life a cell is. Adding it produced a
    candidate that cleared the original mean-baseline gate: LOCO R2 = +0.0125.

    Against a baseline that predicts the target from cycle count alone, the
    same candidate scores -0.0053. All of its apparent skill was age. This
    test exists because that candidate was briefly promoted before the
    confound baseline was added.
    """
    datasets = DatasetRegistry()
    datasets.register(CallableDatasetLoader("nasa", lambda: pd.read_csv(NASA_TRAINING)))
    calibrator = AdaptiveCalibrator(store=ModelStore(tmp_path / "s"), datasets=datasets)

    run = calibrator.calibrate("nasa", [
        linear_candidate("temp_stress_soc",
                         ["trailing_avg_temp", "avg_stress", "avg_soc"]),
    ])
    outcome = run.outcomes[0]

    # It still beats a constant - that is exactly why the constant is too weak.
    assert outcome.loco.median_r2 > 0
    # But not the age baseline, so it must be rejected.
    assert outcome.loco.median_r2_vs_confound < 0
    assert outcome.promoted is False
    assert any("age rather than behaviour" in r for r in outcome.verdict.reasons)


def test_the_confound_baseline_can_be_disabled_for_datasets_without_cycles(tmp_path):
    """A dataset with no cycle column simply has no confound baseline."""
    datasets = DatasetRegistry()
    datasets.register(CallableDatasetLoader("learnable", _learnable))
    calibrator = AdaptiveCalibrator(
        store=ModelStore(tmp_path / "s"), datasets=datasets, confound_col=None,
    )
    run = calibrator.calibrate(
        "learnable", [linear_candidate("temp_only", ["trailing_avg_temp"])]
    )
    assert np.isnan(run.outcomes[0].loco.median_r2_vs_confound)
    assert run.outcomes[0].promoted is True
