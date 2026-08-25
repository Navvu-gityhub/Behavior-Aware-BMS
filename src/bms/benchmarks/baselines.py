"""Reference predictors: the bar every real method has to clear.

These are not strawmen. Three of the four encode a specific, defensible claim
about battery degradation, and in this project's experience they are hard to
beat:

`train_mean`
    Predict the training fold's mean, ignoring every feature. Its R-squared
    against the harness's own `r2_vs_global_mean` is exactly zero by
    construction, which makes it a live self-test of the harness: if this
    method scores anything other than ~0, the metric is wired wrong.

`age_linear` / `age_quadratic`
    Predict from cycle count alone. This is the confound baseline that caught
    a false positive the adaptive gate had already promoted — `avg_soc` has a
    median within-cell correlation of -0.73 with cycle index, so a model can
    look skilful by learning "later cycle, more loss" and nothing about
    behaviour at all. Any behavioural method that cannot beat these has not
    demonstrated it uses behaviour.

`age_isotonic`
    The same claim without a functional form: fade is monotone in cycle count.
    Included because a behavioural method beating the linear age model but not
    the isotonic one has beaten the parameterisation, not the confound.

Why these belong in the benchmark table rather than only in the gate
--------------------------------------------------------------------
The gate uses the age baseline as a pass/fail criterion. A results table needs
it as a *row*, with the same metrics as everything else, because "the best
published method scores below predict-the-mean" is the single most legible way
to state this project's finding, and it cannot be stated unless both numbers
appear in the same column.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn
from src.bms.benchmarks.registry import BenchmarkMethod, BuildFn, Family, register

AGE_COLUMN = "cycle"


def _constant_predictor(value: float):
    def predict(test: pd.DataFrame) -> np.ndarray:
        return np.full(len(test), value, dtype=float)
    return predict


def build_train_mean(features: Sequence[str], target: str) -> FitFn:
    """Predict the training mean. Ignores `features` entirely, by design."""
    def fit(train: pd.DataFrame):
        return _constant_predictor(float(train[target].mean()))
    return fit


def _polynomial_age(degree: int) -> BuildFn:
    def build(features: Sequence[str], target: str) -> FitFn:
        def fit(train: pd.DataFrame):
            x = train[AGE_COLUMN].to_numpy(dtype=float)
            y = train[target].to_numpy(dtype=float)
            if len(np.unique(x)) <= degree:
                # Not enough distinct ages to identify the polynomial; fall
                # back to the mean rather than fitting a wild extrapolation.
                return _constant_predictor(float(y.mean()))
            coefficients = np.polyfit(x, y, degree)

            def predict(test: pd.DataFrame) -> np.ndarray:
                return np.polyval(coefficients, test[AGE_COLUMN].to_numpy(dtype=float))
            return predict
        return fit
    return build


def build_age_isotonic(features: Sequence[str], target: str) -> FitFn:
    """Monotone-in-cycle-count fade, with no assumed functional form.

    Isotonic regression is clipped to the training range by construction, so a
    held-out cohort whose cycle indices run past the training maximum receives
    the boundary value rather than an extrapolation. That is the honest
    behaviour for a nonparametric fit and matches what a deployed monotone
    model would do.
    """
    from sklearn.isotonic import IsotonicRegression

    def fit(train: pd.DataFrame):
        x = train[AGE_COLUMN].to_numpy(dtype=float)
        y = train[target].to_numpy(dtype=float)
        model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        model.fit(x, y)

        def predict(test: pd.DataFrame) -> np.ndarray:
            return np.asarray(
                model.predict(test[AGE_COLUMN].to_numpy(dtype=float)), dtype=float
            )
        return predict
    return fit


register(BenchmarkMethod(
    name="train_mean",
    family=Family.NAIVE,
    citation="Reference predictor (no features).",
    build=build_train_mean,
    default_features=(),
    feature_free=True,
    note=(
        "Scores R2 = 0 against the harness's training-mean baseline by "
        "construction; any other value indicates a metric wiring bug."
    ),
))

register(BenchmarkMethod(
    name="age_linear",
    family=Family.NAIVE,
    citation="Confound baseline; see ADR 0005 and adaptive/calibrator.py.",
    build=_polynomial_age(1),
    default_features=(AGE_COLUMN,),
    note="Cycle count alone, linear. The gate's confound baseline as a table row.",
))

register(BenchmarkMethod(
    name="age_quadratic",
    family=Family.NAIVE,
    citation="Confound baseline, second order.",
    build=_polynomial_age(2),
    default_features=(AGE_COLUMN,),
    note="Allows fade rate itself to drift with age.",
))

register(BenchmarkMethod(
    name="age_isotonic",
    family=Family.NAIVE,
    citation="Nonparametric monotone confound baseline.",
    build=build_age_isotonic,
    default_features=(AGE_COLUMN,),
    note=(
        "Monotone in cycle count with no functional form. A behavioural method "
        "that beats age_linear but not this one has beaten the "
        "parameterisation rather than the confound."
    ),
))
