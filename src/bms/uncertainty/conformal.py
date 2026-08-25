"""Conformal prediction intervals, and what happens to them under protocol shift.

WHY THIS IS THE RIGHT UNCERTAINTY TOOL FOR THIS PROJECT
--------------------------------------------------------
Every score this project emits is a point estimate, and the whole body of
validation work says those point estimates do not generalise across protocols.
A deployed battery system needs something stronger than a number that might be
wrong: it needs to know *when* it is likely to be wrong.

Split conformal prediction supplies exactly that, and it supplies it with a
finite-sample guarantee rather than an asymptotic one. Given a fitted model and
a held-out calibration set, it produces intervals with **at least 1 - alpha
marginal coverage** for any underlying model, any data distribution, and any
sample size. No Gaussian residuals, no correct model specification, no
asymptotics.

That last part matters here more than usual. This project's models are known to
be misspecified — the whole point of `physics/arrhenius.py` is that a linear
temperature coefficient has the wrong functional form. A Bayesian credible
interval from a misspecified model is confidently wrong. A conformal interval
around the same misspecified model is still valid, just wide. Wide-and-honest
is precisely what this project has spent its effort defending.

THE ASSUMPTION, AND WHY THIS PROJECT BREAKS IT ON PURPOSE
----------------------------------------------------------
The guarantee requires **exchangeability** between calibration and test data.
Cross-protocol deployment violates it directly: the held-out cohort was cycled
at a different temperature, load and cutoff, so its residual distribution is
not the calibration set's.

So the guarantee does not merely weaken across protocols, it does not apply.
That is not a limitation to be noted and moved past — it is measurable, and
measuring it restates this project's central finding in the one language a
safety case is actually written in:

    calibrate within protocol, deploy within protocol   -> coverage ~ 1 - alpha
    calibrate within protocol, deploy across protocol   -> coverage < 1 - alpha

`coverage_report` computes both. The gap is the same phenomenon as the
LOBO-to-LOCO R-squared collapse, expressed as "your 90% interval is really an
X% interval on a cell you haven't seen the protocol of."

MONDRIAN (COHORT-CONDITIONAL) CONFORMAL
---------------------------------------
Marginal coverage is a fleet-average statement. A predictor can hold 90%
overall while systematically undercovering cold cells and overcovering warm
ones, which is the failure mode that matters — the cells being failed are the
ones in the hardest conditions.

`MondrianConformal` computes a separate quantile per cohort, giving
*conditional* coverage within each. It cannot extend the guarantee to an
unseen cohort — nothing can, without an assumption about how cohorts relate —
but it makes the per-cohort breakdown visible instead of hiding it inside an
average, and it refuses rather than guessing when asked about a cohort it has
no calibration data for.

THE WIDTH REFUSAL
-----------------
An interval wide enough to span the entire plausible range of the target is
formally valid and practically useless: "this cell's fade is between 0% and
100%, with 90% confidence" is a refusal dressed as an answer. `predict` takes
a `max_useful_width` and returns an explicit refusal above it, which is the
same pattern `adaptive/calibrator.score` follows when no model has passed the
gate — a refusal carrying a reason beats a number carrying none.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn

DEFAULT_ALPHA = 0.1
DEFAULT_CALIBRATION_FRACTION = 0.3
RANDOM_SEED = 20260821

# Minimum calibration points for the quantile to mean anything. Below
# ceil((n+1)(1-alpha)) > n the required quantile exceeds the largest observed
# score and the interval becomes infinite, which is honest but useless.
MIN_CALIBRATION_ROWS = 20


@dataclass(frozen=True)
class Interval:
    """A prediction with its conformal bounds."""

    prediction: float
    lower: float
    upper: float
    cohort_id: str = ""

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def contains(self, value: float) -> bool:
        return bool(self.lower <= value <= self.upper)


@dataclass(frozen=True)
class IntervalRefusal:
    """A refusal to quote an interval, with the reason preserved."""

    reason: str
    detail: str = ""
    cohort_id: str = ""

    status: str = "REFUSED"

    def __bool__(self) -> bool:
        return False


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected quantile of the nonconformity scores.

    The correction is `ceil((n + 1)(1 - alpha)) / n`, not the plain
    `1 - alpha` empirical quantile. Without it, coverage is below nominal for
    small calibration sets — which is exactly the regime this project operates
    in, with nine cohorts and thirty-odd cells. Returns infinity when the
    required rank exceeds the sample, which is the correct answer (no finite
    interval is justified) rather than a silently truncated one.
    """
    finite = np.asarray(scores, dtype=float)
    finite = finite[np.isfinite(finite)]
    n = len(finite)
    if n == 0:
        return float("inf")

    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(finite)[rank - 1])


class SplitConformal:
    """Split conformal regression around any `FitFn`.

    The model is fitted on one part of the training data and calibrated on a
    disjoint part. The split is what buys the guarantee: reusing training rows
    for calibration gives optimistically small residuals and undercoverage.
    """

    def __init__(
        self,
        fit_fn: FitFn,
        target: str,
        alpha: float = DEFAULT_ALPHA,
        calibration_fraction: float = DEFAULT_CALIBRATION_FRACTION,
        seed: int = RANDOM_SEED,
        group_col: str | None = "cell_id",
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.fit_fn = fit_fn
        self.target = target
        self.alpha = alpha
        self.calibration_fraction = calibration_fraction
        self.seed = seed
        # Splitting by cell rather than by row. Rows from one cell are highly
        # autocorrelated, so a random row split would put near-duplicates of
        # calibration rows into training and shrink the residuals — the same
        # leakage the LOBO/LOCO harness exists to prevent, one level down.
        self.group_col = group_col

        self._predict: Callable[[pd.DataFrame], np.ndarray] | None = None
        self._quantile = float("inf")
        self._n_calibration = 0

    def _split(self, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.seed)
        if self.group_col and self.group_col in train.columns:
            groups = np.array(sorted(train[self.group_col].unique()))
            n_calibration = max(1, round(len(groups) * self.calibration_fraction))
            chosen = set(rng.choice(groups, size=n_calibration, replace=False).tolist())
            mask = train[self.group_col].isin(chosen)
            return train[~mask], train[mask]

        row_mask = rng.random(len(train)) < self.calibration_fraction
        return train[~row_mask], train[row_mask]

    def fit(self, train: pd.DataFrame) -> SplitConformal:
        proper, calibration = self._split(train)
        if proper.empty or calibration.empty:
            raise ValueError(
                "SplitConformal.fit: the split produced an empty part. With "
                "grouped splitting this happens when there are too few groups; "
                "pass group_col=None to split by row instead."
            )

        self._predict = self.fit_fn(proper)
        predicted = np.asarray(self._predict(calibration), dtype=float)
        actual = calibration[self.target].to_numpy(dtype=float)

        scores = np.abs(actual - predicted)
        self._quantile = conformal_quantile(scores, self.alpha)
        self._n_calibration = int(np.isfinite(scores).sum())
        return self

    @property
    def quantile(self) -> float:
        return self._quantile

    @property
    def n_calibration(self) -> int:
        return self._n_calibration

    def predict(
        self,
        test: pd.DataFrame,
        max_useful_width: float | None = None,
    ) -> list[Interval | IntervalRefusal]:
        """Intervals for each test row, or a refusal where none is justified."""
        if self._predict is None:
            raise RuntimeError("SplitConformal.predict called before fit()")

        if self._n_calibration < MIN_CALIBRATION_ROWS:
            return [
                IntervalRefusal(
                    reason="insufficient calibration data",
                    detail=(
                        f"{self._n_calibration} calibration rows, below the "
                        f"{MIN_CALIBRATION_ROWS} minimum for a meaningful "
                        f"{1 - self.alpha:.0%} quantile."
                    ),
                )
            ] * len(test)

        predicted = np.asarray(self._predict(test), dtype=float)
        width = 2.0 * self._quantile

        if not np.isfinite(self._quantile):
            return [
                IntervalRefusal(
                    reason="no finite interval is justified",
                    detail=(
                        f"The {1 - self.alpha:.0%} conformal quantile exceeds the "
                        f"largest observed calibration residual with "
                        f"{self._n_calibration} points."
                    ),
                )
            ] * len(test)

        if max_useful_width is not None and width > max_useful_width:
            return [
                IntervalRefusal(
                    reason="interval too wide to be actionable",
                    detail=(
                        f"width {width:.4g} exceeds the useful maximum "
                        f"{max_useful_width:.4g}. A formally valid interval "
                        f"spanning the target's whole range is a refusal "
                        f"dressed as an answer."
                    ),
                )
            ] * len(test)

        return [
            Interval(prediction=float(p), lower=float(p - self._quantile),
                     upper=float(p + self._quantile))
            for p in predicted
        ]


class MondrianConformal(SplitConformal):
    """Cohort-conditional conformal: one quantile per cohort.

    Gives conditional coverage within each *known* cohort. It cannot say
    anything about an unseen one, and `predict` refuses for cohorts absent
    from calibration rather than falling back to the pooled quantile — a
    pooled quantile applied to a new protocol is precisely the unwarranted
    extrapolation this class exists to avoid.
    """

    def __init__(self, *args, cohort_col: str = "cohort", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cohort_col = cohort_col
        self._quantiles: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def fit(self, train: pd.DataFrame) -> MondrianConformal:
        if self.cohort_col not in train.columns:
            raise ValueError(
                f"MondrianConformal.fit: missing cohort column "
                f"'{self.cohort_col}'"
            )

        proper, calibration = self._split(train)
        if proper.empty or calibration.empty:
            raise ValueError("MondrianConformal.fit: the split produced an empty part.")

        self._predict = self.fit_fn(proper)
        predicted = np.asarray(self._predict(calibration), dtype=float)
        actual = calibration[self.target].to_numpy(dtype=float)
        scores = np.abs(actual - predicted)

        frame = pd.DataFrame({
            "cohort": calibration[self.cohort_col].to_numpy(),
            "score": scores,
        })
        for cohort, group in frame.groupby("cohort"):
            values = group["score"].to_numpy(dtype=float)
            self._quantiles[str(cohort)] = conformal_quantile(values, self.alpha)
            self._counts[str(cohort)] = int(np.isfinite(values).sum())

        self._quantile = conformal_quantile(scores, self.alpha)
        self._n_calibration = int(np.isfinite(scores).sum())
        return self

    @property
    def quantiles(self) -> Mapping[str, float]:
        return dict(self._quantiles)

    def predict(
        self,
        test: pd.DataFrame,
        max_useful_width: float | None = None,
    ) -> list[Interval | IntervalRefusal]:
        if self._predict is None:
            raise RuntimeError("MondrianConformal.predict called before fit()")

        predicted = np.asarray(self._predict(test), dtype=float)
        cohorts = test[self.cohort_col].astype(str).to_numpy()

        out: list[Interval | IntervalRefusal] = []
        for value, cohort in zip(predicted, cohorts, strict=True):
            quantile = self._quantiles.get(cohort)
            if quantile is None:
                out.append(IntervalRefusal(
                    reason="no calibration data for this cohort",
                    detail=(
                        f"cohort '{cohort}' was not present in calibration. "
                        f"Known cohorts: {sorted(self._quantiles)}. The pooled "
                        f"quantile is deliberately not substituted."
                    ),
                    cohort_id=cohort,
                ))
                continue
            if not np.isfinite(quantile):
                out.append(IntervalRefusal(
                    reason="no finite interval for this cohort",
                    detail=f"only {self._counts.get(cohort, 0)} calibration points.",
                    cohort_id=cohort,
                ))
                continue
            if max_useful_width is not None and 2.0 * quantile > max_useful_width:
                out.append(IntervalRefusal(
                    reason="interval too wide to be actionable",
                    detail=f"width {2 * quantile:.4g} for cohort '{cohort}'.",
                    cohort_id=cohort,
                ))
                continue
            out.append(Interval(
                prediction=float(value), lower=float(value - quantile),
                upper=float(value + quantile), cohort_id=cohort,
            ))
        return out


def empirical_coverage(
    intervals: Sequence[Interval | IntervalRefusal],
    actual: np.ndarray,
) -> dict[str, float]:
    """Coverage and mean width over the rows where an interval was quoted.

    Refusals are excluded from the coverage rate and counted separately.
    Scoring a refusal as a miss would penalise the system for declining, and
    scoring it as a hit would reward it for saying nothing; reporting the
    refusal rate alongside is the only honest option.
    """
    quoted = [
        (i, y) for i, y in zip(intervals, actual, strict=True)
        if isinstance(i, Interval)
    ]
    n_refused = len(intervals) - len(quoted)

    if not quoted:
        return {
            "coverage": float("nan"), "mean_width": float("nan"),
            "n_quoted": 0.0, "n_refused": float(n_refused),
            "refusal_rate": 1.0 if intervals else float("nan"),
        }

    covered = sum(1 for interval, y in quoted if np.isfinite(y) and interval.contains(y))
    return {
        "coverage": covered / len(quoted),
        "mean_width": float(np.mean([i.width for i, _ in quoted])),
        "n_quoted": float(len(quoted)),
        "n_refused": float(n_refused),
        "refusal_rate": n_refused / len(intervals),
    }


def coverage_report(
    data: pd.DataFrame,
    fit_fn: FitFn,
    target: str,
    alpha: float = DEFAULT_ALPHA,
    cell_col: str = "cell_id",
    cohort_col: str = "cohort",
    mondrian: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Empirical coverage under leave-one-cell-out and leave-one-cohort-out.

    This is the module's headline experiment. Both splits calibrate on the
    training fold and quote intervals for the held-out group; the difference
    between them is the cost of protocol shift, measured in coverage rather
    than in R-squared.

    A nominal 1 - alpha interval should cover 1 - alpha of held-out points
    *if* calibration and test are exchangeable. Under LOBO they roughly are.
    Under LOCO they are not, and the shortfall is the number this project
    cares about.
    """
    rows = []
    for split, group_col in (("LOBO", cell_col), ("LOCO", cohort_col)):
        if group_col not in data.columns:
            continue
        for group in sorted(data[group_col].unique()):
            test = data[data[group_col] == group]
            train = data[data[group_col] != group]
            if len(test) < 5 or len(train) < MIN_CALIBRATION_ROWS:
                continue

            try:
                predictor = (
                    MondrianConformal(
                        fit_fn, target=target, alpha=alpha, seed=seed,
                        cohort_col=cohort_col,
                    ) if mondrian else
                    SplitConformal(fit_fn, target=target, alpha=alpha, seed=seed)
                )
                predictor.fit(train)
                intervals = predictor.predict(test)
            except Exception as exc:
                rows.append({
                    "split": split, "held_out": str(group), "n_test": len(test),
                    "coverage": float("nan"), "mean_width": float("nan"),
                    "n_quoted": 0.0, "n_refused": float(len(test)),
                    "refusal_rate": 1.0, "nominal": 1 - alpha,
                    "error": repr(exc),
                })
                continue

            stats = empirical_coverage(intervals, test[target].to_numpy(dtype=float))
            rows.append({
                "split": split, "held_out": str(group), "n_test": len(test),
                **stats, "nominal": 1 - alpha, "error": None,
            })

    return pd.DataFrame(rows)


def coverage_summary(report: pd.DataFrame) -> pd.DataFrame:
    """Per-split coverage summary, including the statistics that expose failure.

    WHY THE MEDIAN IS REPORTED BUT MUST NOT BE READ ALONE
    ------------------------------------------------------
    Measured on the NASA frame with ElasticNet against `cumulative_fade`, at a
    nominal 0.90 (`scripts/run_coverage_study.py`):

        LOBO   median 0.970   worst 0.000 (B0045)              19% below nominal
        LOCO   median 0.824   worst 0.262 (COLD4C_2A_flagged)  56% below nominal

    The medians look respectable and would pass an audit that stopped there.
    The worst-group column says something else: there is a *cell* for which
    the 90% interval contained the truth zero times, and under protocol shift
    a majority of held-out cohorts fall below nominal.

    So the aggregate is not merely uninformative here, it is
    *anti*-informative: a fleet-average coverage statistic certifies a system
    that fails precisely on the cells in the hardest operating conditions,
    because the overcovered easy cohorts pull the average up.

    The comparison with a simpler model is the other half of the point.
    `age_linear` has only 11% of cohorts below nominal — better calibrated —
    but its intervals are 27% wider (0.505 versus 0.399). The flexible model
    buys tighter intervals and pays for them in coverage, which is exactly the
    trade a safety case has to see stated.

    `min_coverage`, `worst_group` and `fraction_below_nominal` are therefore
    part of the summary rather than something a reader has to go and compute.
    A conformal system should be judged on its worst group, not its median one.
    """
    if report.empty:
        return pd.DataFrame()

    done = report[report["error"].isna()]
    if done.empty:
        return pd.DataFrame()

    summary = done.groupby("split").agg(
        n_folds=("held_out", "count"),
        median_coverage=("coverage", "median"),
        mean_coverage=("coverage", "mean"),
        min_coverage=("coverage", "min"),
        median_width=("mean_width", "median"),
        median_refusal_rate=("refusal_rate", "median"),
        nominal=("nominal", "first"),
    ).reset_index()

    summary["median_shortfall"] = summary["nominal"] - summary["median_coverage"]
    summary["worst_shortfall"] = summary["nominal"] - summary["min_coverage"]

    below = done[done["coverage"] < done["nominal"]]
    counts = below.groupby("split")["held_out"].count()
    summary["fraction_below_nominal"] = (
        summary["split"].map(counts).fillna(0) / summary["n_folds"]
    )

    worst = done.loc[done.groupby("split")["coverage"].idxmin()]
    summary["worst_group"] = summary["split"].map(
        dict(zip(worst["split"], worst["held_out"], strict=True))
    )
    return summary
