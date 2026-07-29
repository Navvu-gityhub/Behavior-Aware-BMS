"""Cross-validation harness and the promotion gate.

This is the component the rest of the adaptive system is built around. Its job
is to answer one question about a candidate model: **is there evidence it would
work on data we have not seen?** The default answer is no.

Why the gate is strict
----------------------
`scripts/validate_health_index_versions.py` established the numbers this module
enforces against:

    v2 fitted OLS, in-sample                  rho = 0.870   inadmissible
    v2, leave-one-battery-out (LOBO)          rho = 0.841   strong
    v2, leave-one-cohort-out (LOCO)           rho = -0.295  collapses

and on per-cycle regression:

    LOBO   median R2 vs global mean = +0.008   73% of folds beat baseline
    LOCO   median R2 vs global mean = -0.167   11% of folds beat baseline

A gate that only checked LOBO would have promoted that model. LOBO always
leaves the held-out cell's cohort siblings in the training set, so it cannot
detect a model that has memorised protocols. **LOCO is therefore the binding
constraint, and a candidate that passes LOBO but fails LOCO is rejected.**

Why R-squared is measured against a baseline predictor
------------------------------------------------------
Textbook R-squared compares against the mean of the *test* fold, which no
deployed model could ever know. That flatters a model by giving it credit for
information it would not have. Here `r2_vs_global_mean` compares against the
mean of the *training* fold, which is what a naive deployed predictor would
actually emit. `r2_vs_own_mean_oracle` is retained alongside it as a diagnostic
only, and is explicitly labelled an oracle so it is never quoted as skill.

A model that cannot beat "always predict the training mean" has no skill, no
matter how good its correlation coefficient looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

# A candidate is any callable that fits on a training frame and returns a
# predictor for a test frame. Keeping it this loose means the harness can
# validate an OLS specification, a gradient booster, or a rule-based scorer
# without knowing anything about them.
FitFn = Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]]


class SupportsPredict(Protocol):
    def __call__(self, test: pd.DataFrame) -> np.ndarray: ...


def r2_against(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    """R-squared of `pred` relative to an explicit `baseline` predictor.

    Returns NaN when the baseline is already perfect (zero total sum of
    squares), because the ratio is undefined there and returning 0.0 or 1.0
    would both be assertions the data does not support.
    """
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - baseline) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class FoldResult:
    """One held-out group."""

    split: str
    held_out: str
    n_train: int
    n_test: int
    mae: float
    r2_vs_global_mean: float
    r2_vs_own_mean_oracle: float
    spearman_rho: float
    r2_vs_confound_baseline: float = float("nan")
    error: str | None = None

    @property
    def beat_baseline(self) -> bool:
        return bool(np.isfinite(self.r2_vs_global_mean) and self.r2_vs_global_mean > 0)


@dataclass(frozen=True)
class CrossValidationResult:
    """All folds for one split type, plus the summary statistics."""

    split: str
    group_col: str
    folds: list[FoldResult]

    @property
    def completed(self) -> list[FoldResult]:
        return [f for f in self.folds if f.error is None]

    @property
    def median_r2(self) -> float:
        values = [f.r2_vs_global_mean for f in self.completed]
        finite = [v for v in values if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    @property
    def median_spearman(self) -> float:
        values = [f.spearman_rho for f in self.completed]
        finite = [v for v in values if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    @property
    def median_r2_vs_confound(self) -> float:
        values = [f.r2_vs_confound_baseline for f in self.completed]
        finite = [v for v in values if np.isfinite(v)]
        return float(np.median(finite)) if finite else float("nan")

    @property
    def fraction_beating_baseline(self) -> float:
        done = self.completed
        return sum(f.beat_baseline for f in done) / len(done) if done else float("nan")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(f) for f in self.folds])


@dataclass
class Verdict:
    """The promotion decision, with every reason preserved.

    Reasons are kept for both outcomes, not just failures. A promotion whose
    justification is not recorded is indistinguishable from an unreviewed one
    six months later.
    """

    promote: bool
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.promote

    @property
    def status(self) -> str:
        return "PROMOTE" if self.promote else "REJECT"

    def render(self) -> str:
        lines = [f"{self.status}"]
        lines += [f"  - {r}" for r in self.reasons]
        if self.metrics:
            lines.append("  metrics:")
            lines += [f"    {k} = {v:.4g}" for k, v in self.metrics.items()]
        return "\n".join(lines)


class Validator:
    """Leave-one-group-out cross-validation with an explicit promotion gate.

    Usage::

        validator = Validator(data, target="capacity_loss", cohort_col="cohort")
        lobo = validator.cross_validate(fit_fn, group_col="cell_id", split="LOBO")
        loco = validator.cross_validate(fit_fn, group_col="cohort", split="LOCO")
        verdict = validator.gate(lobo, loco)
    """

    # Thresholds for the gate. Exposed as class attributes so a caller can
    # tighten them, but the defaults are set where the evidence puts them.
    #
    # `MIN_LOCO_R2 = 0.0` is not arbitrary: R2 of zero against the training
    # mean is exactly the point where a model stops being worse than emitting
    # a constant. Below it the model is actively harmful.
    MIN_LOCO_R2: float = 0.0
    MIN_LOCO_FRACTION_BEATING: float = 0.5
    MIN_LOBO_R2: float = 0.0
    MIN_TEST_ROWS: int = 10
    MIN_TRAIN_COHORTS: int = 2

    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        cohort_col: str = "cohort",
        confound_fit: FitFn | None = None,
    ) -> None:
        for column in (target, cohort_col):
            if column not in data.columns:
                raise ValueError(f"Validator: missing required column '{column}'")
        if data.empty:
            raise ValueError("Validator: no data")
        self.data = data
        self.target = target
        self.cohort_col = cohort_col
        self.confound_fit = confound_fit

    def cross_validate(
        self,
        fit_fn: FitFn,
        group_col: str,
        split: str | None = None,
        groups: Sequence[str] | None = None,
    ) -> CrossValidationResult:
        """Hold out one group at a time; refit on the remainder every fold.

        Refitting inside the fold is the whole point. Reusing shipped
        coefficients would leak the held-out group, which is precisely how the
        v2 model produced an in-sample rho of 0.870 that did not survive
        contact with a held-out protocol.
        """
        if group_col not in self.data.columns:
            raise ValueError(f"cross_validate: missing group column '{group_col}'")
        split = split or f"LO{group_col.upper()}O"
        candidates = groups if groups is not None else sorted(self.data[group_col].unique())

        folds: list[FoldResult] = []
        for group in candidates:
            test = self.data[self.data[group_col] == group]
            train = self.data[self.data[group_col] != group]

            skip = self._skip_reason(train, test)
            if skip:
                folds.append(self._errored_fold(split, group, train, test, skip))
                continue

            try:
                predict = fit_fn(train)
                pred = np.asarray(predict(test), dtype=float)
            except Exception as exc:  # a candidate that crashes has not passed
                folds.append(self._errored_fold(split, group, train, test, repr(exc)))
                continue

            y = test[self.target].to_numpy(dtype=float)
            if pred.shape != y.shape:
                folds.append(self._errored_fold(
                    split, group, train, test,
                    f"prediction shape {pred.shape} != target shape {y.shape}",
                ))
                continue

            global_mean = np.full_like(y, float(train[self.target].mean()))
            own_mean = np.full_like(y, float(y.mean()))
            rho = pd.Series(pred).corr(pd.Series(y), method="spearman")

            # Compare against the confound baseline as well, refitted on the
            # same fold. See the class docstring: beating a constant is a low
            # bar when both features and target rise with cycle count.
            confound_r2 = float("nan")
            if self.confound_fit is not None:
                try:
                    confound_pred = np.asarray(
                        self.confound_fit(train)(test), dtype=float
                    )
                    if confound_pred.shape == y.shape:
                        confound_r2 = r2_against(y, pred, confound_pred)
                except Exception:
                    # A baseline that cannot be fitted is not the candidate's
                    # fault; leave the metric undefined rather than failing
                    # the fold or, worse, silently passing it.
                    confound_r2 = float("nan")

            folds.append(FoldResult(
                split=split,
                held_out=str(group),
                n_train=int(len(train)),
                n_test=int(len(test)),
                mae=float(np.mean(np.abs(y - pred))),
                r2_vs_global_mean=r2_against(y, pred, global_mean),
                r2_vs_own_mean_oracle=r2_against(y, pred, own_mean),
                spearman_rho=float(rho) if rho is not None and np.isfinite(rho) else float("nan"),
                r2_vs_confound_baseline=confound_r2,
            ))

        return CrossValidationResult(split=split, group_col=group_col, folds=folds)

    def _skip_reason(self, train: pd.DataFrame, test: pd.DataFrame) -> str | None:
        if len(test) < self.MIN_TEST_ROWS:
            return f"test fold has {len(test)} rows, below minimum {self.MIN_TEST_ROWS}"
        if train[self.cohort_col].nunique() < self.MIN_TRAIN_COHORTS:
            return (
                f"training fold spans {train[self.cohort_col].nunique()} cohort(s), "
                f"below minimum {self.MIN_TRAIN_COHORTS}"
            )
        return None

    @staticmethod
    def _errored_fold(
        split: str, group: object, train: pd.DataFrame, test: pd.DataFrame, error: str
    ) -> FoldResult:
        nan = float("nan")
        return FoldResult(
            split=split, held_out=str(group), n_train=int(len(train)),
            n_test=int(len(test)), mae=nan, r2_vs_global_mean=nan,
            r2_vs_own_mean_oracle=nan, spearman_rho=nan, error=error,
        )

    # -- the gate -----------------------------------------------------------

    def gate(
        self,
        lobo: CrossValidationResult,
        loco: CrossValidationResult | None = None,
    ) -> Verdict:
        """Decide whether a candidate may be promoted.

        LOCO is required, not optional. Passing `loco=None` is treated as
        missing evidence and rejected, rather than waved through — a candidate
        validated only within known protocols has not been shown to
        generalise, and that is the exact gap ADR 0002 documents.
        """
        reasons: list[str] = []
        metrics: dict[str, float] = {}

        if not lobo.completed:
            return Verdict(False, ["LOBO produced no completed folds"], metrics)

        metrics["lobo_median_r2"] = lobo.median_r2
        metrics["lobo_fraction_beating_baseline"] = lobo.fraction_beating_baseline
        metrics["lobo_median_spearman"] = lobo.median_spearman

        promote = True

        if not (lobo.median_r2 > self.MIN_LOBO_R2):
            promote = False
            reasons.append(
                f"LOBO median R2 vs training mean is {lobo.median_r2:.4g}, "
                f"not above {self.MIN_LOBO_R2}. The candidate cannot beat a "
                f"constant predictor even on a held-out cell from a known protocol."
            )

        if loco is None:
            return Verdict(False, reasons + [
                "No leave-one-cohort-out result supplied. LOCO is mandatory: "
                "LOBO leaves the held-out cell's cohort siblings in training and "
                "so cannot detect protocol memorisation (ADR 0002)."
            ], metrics)

        if not loco.completed:
            return Verdict(False, reasons + [
                "LOCO produced no completed folds, so cross-protocol "
                "generalisation is untested."
            ], metrics)

        metrics["loco_median_r2"] = loco.median_r2
        metrics["loco_fraction_beating_baseline"] = loco.fraction_beating_baseline
        metrics["loco_median_spearman"] = loco.median_spearman

        if not (loco.median_r2 > self.MIN_LOCO_R2):
            promote = False
            reasons.append(
                f"LOCO median R2 vs training mean is {loco.median_r2:.4g}, "
                f"not above {self.MIN_LOCO_R2}. On an unseen protocol the "
                f"candidate is worse than emitting the training mean."
            )

        if not (loco.fraction_beating_baseline >= self.MIN_LOCO_FRACTION_BEATING):
            promote = False
            reasons.append(
                f"Only {loco.fraction_beating_baseline:.0%} of held-out protocols "
                f"beat the baseline, below the required "
                f"{self.MIN_LOCO_FRACTION_BEATING:.0%}. Median skill carried by a "
                f"minority of cohorts is not generalisation."
            )

        # Beating a constant is a weak claim when the target and every feature
        # both rise with cycle count. If a confound baseline was supplied, the
        # candidate must beat that too, or its apparent skill is age.
        for label, result in (("LOBO", lobo), ("LOCO", loco)):
            confound_r2 = result.median_r2_vs_confound
            if not np.isfinite(confound_r2):
                continue
            metrics[f"{label.lower()}_median_r2_vs_confound"] = confound_r2
            if confound_r2 <= 0:
                promote = False
                reasons.append(
                    f"{label} median R2 against the confound baseline is "
                    f"{confound_r2:.4g}, not above 0. The candidate does not "
                    f"improve on predicting the target from cycle count alone, "
                    f"so its apparent skill is age rather than behaviour."
                )

        # A large LOBO/LOCO gap is the signature of cohort memorisation even
        # when both clear their thresholds, so it is reported either way.
        if np.isfinite(lobo.median_r2) and np.isfinite(loco.median_r2):
            gap = lobo.median_r2 - loco.median_r2
            metrics["lobo_minus_loco_r2"] = gap
            if gap > 0.1:
                reasons.append(
                    f"NOTE: LOBO exceeds LOCO by {gap:.4g} R2. Skill is "
                    f"concentrated within known protocols, which is the "
                    f"signature of fitted cohort intercepts doing the work."
                )

        if promote:
            reasons.append(
                f"Beats the training-mean baseline out-of-sample on both splits "
                f"(LOBO {lobo.median_r2:.4g}, LOCO {loco.median_r2:.4g}) with "
                f"{loco.fraction_beating_baseline:.0%} of held-out protocols above baseline."
            )

        return Verdict(promote, reasons, metrics)
