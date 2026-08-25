"""Standard supervised regressors, as used throughout the battery-prognostics literature.

These are the workhorses a reviewer will expect to see: penalised linear
regression, support-vector regression, Gaussian-process regression, two tree
ensembles, and a small neural network. They are not novel and are not presented
as novel. They are here to answer one question — *does a competent, standard
model do better than this project's heuristic under leave-one-cohort-out?* —
and their value lies entirely in being evaluated under the same rules.

Three implementation decisions worth stating, because each one could otherwise
flatter the results:

**Every model is wrapped in impute -> standardise -> fit.** Not a convenience.
Support-vector and Gaussian-process regression are scale-sensitive, and this
project's features span `avg_soc` around 51 and `ambient_x_aggressive` around
2,198. An unscaled comparison would report a preprocessing failure as a
modelling result. The imputer is median-fill and exists so a fold with a
missing value degrades one column instead of crashing the fold — but note this
is imputation *inside a benchmark*, where the alternative is losing the fold
entirely, not imputation inside the scoring pipeline, which
`risk.stress_score` deliberately refuses.

**Hyperparameters are fixed, not tuned per fold.** Tuning inside each fold
would be more favourable to these methods and is the right thing to do for a
paper claiming a method is best. This benchmark makes the opposite claim — that
even reasonable defaults fail to transfer — so untuned defaults are the
conservative choice *against* the thesis: if a tuned model would score higher,
the honest reading is that the gap this study reports is an upper bound on
these methods' skill, and the study says so rather than tuning quietly.

**Gaussian-process regression subsamples its training fold.** Exact GPR is
O(n^3) in training rows; a LOBO sweep over 34 cells at ~2,600 rows per fold is
not tractable at full size. `GPR_MAX_TRAIN` caps it with a seeded random
subsample and the cap is reported in the results table, because a silently
subsampled model presented beside fully-trained ones is a rigged comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn
from src.bms.benchmarks.registry import BenchmarkMethod, BuildFn, Family, register

RANDOM_SEED = 20260821

# Exact GPR is cubic in training rows. See module docstring.
GPR_MAX_TRAIN = 800

# The behavioural feature set this project's own scores are built from, minus
# the ones the threshold-reachability audit found degenerate on NASA
# (`fast_charge_duration` is identically zero in all 2,682 observations, so it
# is excluded here rather than silently contributing a zero coefficient).
DEFAULT_FEATURES: tuple[str, ...] = (
    "avg_temp",
    "max_temp",
    "avg_stress",
    "deep_discharge_duration",
    "aggressive_discharge_count",
    "avg_soc",
    "cycle",
)


def _pipeline(estimator):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", estimator),
    ])


def _sklearn_build(make_estimator: Callable[[], object], max_train: int | None = None) -> BuildFn:
    """Turn an estimator factory into a `FitFn` factory for the validator.

    The estimator is constructed inside `fit`, once per fold, so no state
    leaks between folds — refitting per fold is the whole point of the
    harness (see `Validator.cross_validate`).
    """
    def build(features: Sequence[str], target: str) -> FitFn:
        columns = list(features)

        def fit(train: pd.DataFrame):
            frame = train
            if max_train is not None and len(frame) > max_train:
                frame = frame.sample(
                    n=max_train, random_state=RANDOM_SEED, replace=False
                )
            model = _pipeline(make_estimator())
            model.fit(frame[columns].to_numpy(dtype=float),
                      frame[target].to_numpy(dtype=float))

            def predict(test: pd.DataFrame) -> np.ndarray:
                return np.asarray(
                    model.predict(test[columns].to_numpy(dtype=float)), dtype=float
                )
            return predict
        return fit
    return build


def _elasticnet():
    from sklearn.linear_model import ElasticNetCV
    # CV over the regularisation path is internal to the training fold, so it
    # sees no held-out data. This is the one place tuning happens, and it tunes
    # only the penalty strength.
    return ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
                        random_state=RANDOM_SEED, max_iter=5000)


def _svr():
    from sklearn.svm import SVR
    return SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")


def _gpr():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )
    # Matern(nu=1.5) rather than RBF: degradation trajectories are not
    # infinitely smooth, and an RBF prior asserts they are. WhiteKernel gives
    # the model an explicit noise term, which matters because the per-cycle
    # capacity delta target is dominated by measurement noise.
    kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1.0)
    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=RANDOM_SEED, alpha=1e-8,
    )


def _random_forest():
    from sklearn.ensemble import RandomForestRegressor
    # n_jobs=1 rather than -1 deliberately. joblib's core-count probe shells
    # out to a subprocess, which fails in restricted execution contexts (CI
    # containers, sandboxed runners) and takes the whole study down with it.
    # The harness already refits per fold across dozens of folds, so
    # parallelism belongs at that level if it is wanted at all — and a
    # benchmark whose results depend on the host's core count is worse than a
    # slightly slower one.
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, n_jobs=1, random_state=RANDOM_SEED,
    )


def _hist_gradient_boosting():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=RANDOM_SEED,
    )


def _xgboost():
    """Genuine XGBoost, not a stand-in.

    `hist_gradient_boosting` is registered separately and is a *different*
    implementation of the same idea — sklearn's histogram booster, in the
    LightGBM lineage. XGBoost differs in ways that matter for a benchmark:
    a level-wise (rather than leaf-wise) growth policy by default, and an
    explicit L1/L2 penalty on leaf weights. Registering only one and labelling
    it "XGBoost" would misreport which algorithm produced the number, which is
    the kind of substitution this benchmark exists to avoid.

    Hyperparameters are conservative and fixed, matching the policy stated in
    the module docstring: no per-fold search, so the reported score is a lower
    bound on what a tuned XGBoost would achieve.
    """
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=1,          # see the RandomForest note; determinism over speed
        tree_method="hist",
        verbosity=0,
    )


def _mlp():
    from sklearn.neural_network import MLPRegressor
    # Stands in for the deep-learning family, and is labelled as a stand-in.
    # A sequence model (LSTM/CNN) over raw cycle traces is a different method
    # and is registered separately in `curves.py`, where it is correctly
    # reported as unavailable on a cycle-level summary frame.
    return MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True,
        random_state=RANDOM_SEED,
    )


_SPECS = (
    ("elasticnet", _elasticnet, None,
     "Zou & Hastie (2005); ubiquitous penalised-linear baseline.",
     "Penalty strength tuned by internal 5-fold CV on the training fold only.",
     ()),
    ("svr_rbf", _svr, None,
     "Drucker et al. (1997); widely applied to SOH/RUL regression.",
     "Fixed C=1.0, epsilon=0.1, gamma='scale'.",
     ()),
    ("gpr_matern", _gpr, GPR_MAX_TRAIN,
     "Richardson et al., J. Power Sources 357 (2017) 209-219 — GPR for battery SOH.",
     f"Training fold subsampled to {GPR_MAX_TRAIN} rows; exact GPR is O(n^3).",
     ()),
    ("random_forest", _random_forest, None,
     "Breiman (2001); common SOH-estimation baseline.",
     "300 trees, min_samples_leaf=5.",
     ()),
    ("hist_gradient_boosting", _hist_gradient_boosting, None,
     "Ke et al. (2017) LightGBM-style histogram boosting.",
     "300 iterations, lr=0.05. Handles NaN natively; imputer is a no-op here.",
     ()),
    ("xgboost", _xgboost, None,
     "Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD (2016).",
     "The actual xgboost package, not a substitute. 300 rounds, lr=0.05, "
     "depth=4, subsample=0.8, L2=1.0. Distinct from hist_gradient_boosting: "
     "level-wise growth and explicit leaf-weight regularisation.",
     ("xgboost",)),
    ("mlp", _mlp, None,
     "Stand-in for the neural-network family on tabular cycle features.",
     "Two hidden layers (64, 32), early stopping. NOT a sequence model — see "
     "the `lstm` method in sequence.py for one.",
     ()),
)

for _name, _factory, _cap, _citation, _note, _packages in _SPECS:
    register(BenchmarkMethod(
        name=_name,
        family=Family.CLASSICAL,
        citation=_citation,
        build=_sklearn_build(_factory, max_train=_cap),
        default_features=DEFAULT_FEATURES,
        requires_package=_packages,
        note=_note,
    ))
