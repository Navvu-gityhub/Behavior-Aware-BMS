"""Run every registered method through the existing gate, and tabulate.

This module adds no evaluation logic of its own. It builds a
`src.bms.adaptive.validation.Validator` — the same one the project's adaptive
calibration system uses — and hands each registered method's `FitFn` to it.
That reuse is the point: a benchmark evaluated under friendlier rules than the
project applies to its own candidates would be worthless as a comparison, and
the surest way to keep the rules identical is to run the identical code.

What a row in the results table means
-------------------------------------
`lobo_r2` and `loco_r2` are median R-squared against the *training* fold mean,
across held-out cells and held-out cohorts respectively. `loco_minus_lobo` is
the headline column: it measures how much of a method's apparent skill
evaporates when the held-out group is a protocol rather than a cell. A large
negative value is the signature this study exists to detect.

`r2_ceiling` is carried alongside from `targets.signal_to_noise`, because a
method scoring 0.02 against a target whose signal fraction is 0.05 is a very
different finding from the same 0.02 against a target with a ceiling of 0.9,
and the two are indistinguishable without it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import (
    CrossValidationResult,
    FitFn,
    Validator,
    Verdict,
)
from src.bms.benchmarks.registry import (
    Availability,
    BenchmarkMethod,
    check_availability,
    load_all,
)
from src.bms.benchmarks.targets import signal_to_noise


@dataclass(frozen=True)
class MethodResult:
    """One method's outcome on one target."""

    method: str
    family: str
    availability: Availability
    lobo: CrossValidationResult | None = None
    loco: CrossValidationResult | None = None
    verdict: Verdict | None = None
    error: str | None = None
    # Set when the study target differs from the one the method declares it
    # models. Not an error — the run is legitimate — but a score carried
    # against the wrong quantity must not be read as evidence about the
    # method. The Arrhenius power law predicts cumulative *fade*, which starts
    # near zero; scored against `soh`, a level near one, it reports ~33% MAE
    # that describes a units mismatch rather than a modelling failure.
    target_mismatch: str = ""

    @property
    def ran(self) -> bool:
        """Did cross-validation actually produce a usable fold?

        Checking `lobo is not None` is not enough. `Validator.cross_validate`
        catches an exception *per fold* and records it as an errored fold
        rather than propagating, which is the right behaviour for the
        validator — one bad fold should not lose the other thirty-two. But it
        means a method that crashes on *every* fold still returns a
        `CrossValidationResult` object, and an earlier version of this property
        reported that as SCORED with NaN metrics.

        A method that never once fitted must not appear in the table beside
        methods that did.
        """
        return (
            self.error is None
            and self.lobo is not None
            and len(self.lobo.completed) > 0
        )

    @property
    def fold_error(self) -> str | None:
        """The first fold-level error, when no fold completed."""
        if self.lobo is None or self.lobo.completed:
            return None
        for fold in self.lobo.folds:
            if fold.error:
                return fold.error
        return "no folds completed and no fold recorded a reason"

    @property
    def status(self) -> str:
        """Why this row looks the way it does.

        Ordered so a failure is never masked by an earlier-passing check. An
        earlier version read the availability status first, which reported a
        method that passed availability and then crashed as "RUNNABLE" with
        blank metrics — the one outcome that must never be quiet.
        """
        if self.error or self.fold_error:
            return "ERROR"
        if not self.availability.runnable:
            return "UNAVAILABLE"
        if self.ran:
            return "SCORED"
        return "NOT_RUN"

    def row(self) -> dict[str, object]:
        base: dict[str, object] = {
            "method": self.method,
            "family": self.family,
            "status": self.status,
            "target_mismatch": self.target_mismatch,
            "reason": (
                self.error or self.fold_error or self.availability.reason
                or self.target_mismatch or ""
            ),
        }
        nan = float("nan")
        lobo = self.lobo
        loco = self.loco

        # `self.ran` already implies `lobo is not None`, but binding it to a
        # local is what lets a type checker see that — and the local is used
        # below rather than the attribute so the two cannot drift apart.
        if not self.ran or lobo is None:
            base.update({
                "lobo_mae": nan, "lobo_rmse": nan,
                "loco_mae": nan, "loco_rmse": nan,
                "lobo_r2": nan, "loco_r2": nan, "loco_minus_lobo": nan,
                "lobo_spearman": nan, "loco_spearman": nan,
                "loco_fraction_beating": nan,
                "lobo_r2_vs_age": nan, "loco_r2_vs_age": nan,
                "n_train_median": nan, "n_test_median": nan,
                "lobo_r2_ci_low": nan, "lobo_r2_ci_high": nan,
                "loco_r2_ci_low": nan, "loco_r2_ci_high": nan,
                "lobo_n_folds": 0, "loco_n_folds": 0,
                "promoted": False,
            })
            return base

        lobo_r2 = lobo.median_r2
        loco_r2 = loco.median_r2 if loco else nan
        # Every point estimate in this table is a median over a handful of
        # folds. Reporting it bare is what this project criticises other
        # benchmark tables for, so the interval travels with it.
        lobo_ci = lobo.median_ci()
        loco_ci = loco.median_ci() if loco else (nan, nan)
        base.update({
            "lobo_r2_ci_low": lobo_ci[0],
            "lobo_r2_ci_high": lobo_ci[1],
            "loco_r2_ci_low": loco_ci[0],
            "loco_r2_ci_high": loco_ci[1],
            "lobo_n_folds": lobo.n_completed,
            "loco_n_folds": loco.n_completed if loco else 0,
            "lobo_mae": lobo.median_mae,
            "lobo_rmse": lobo.median_rmse,
            "loco_mae": loco.median_mae if loco else nan,
            "loco_rmse": loco.median_rmse if loco else nan,
            "n_train_median": float(np.median([f.n_train for f in lobo.completed]))
            if lobo.completed else nan,
            "n_test_median": float(np.median([f.n_test for f in lobo.completed]))
            if lobo.completed else nan,
            "lobo_r2": lobo_r2,
            "loco_r2": loco_r2,
            "loco_minus_lobo": (loco_r2 - lobo_r2)
            if np.isfinite(lobo_r2) and np.isfinite(loco_r2) else nan,
            "lobo_spearman": lobo.median_spearman,
            "loco_spearman": loco.median_spearman if loco else nan,
            "loco_fraction_beating": loco.fraction_beating_baseline if loco else nan,
            "lobo_r2_vs_age": lobo.median_r2_vs_confound,
            "loco_r2_vs_age": loco.median_r2_vs_confound if loco else nan,
            "promoted": bool(self.verdict.promote) if self.verdict else False,
        })
        return base


@dataclass(frozen=True)
class StudyResult:
    """Every method's outcome on one target, plus that target's noise ceiling."""

    target: str
    n_rows: int
    n_cells: int
    n_cohorts: int
    results: tuple[MethodResult, ...]
    r2_ceiling: float = float("nan")

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame([r.row() for r in self.results])
        if frame.empty:
            return frame
        frame.insert(0, "target", self.target)
        frame["r2_ceiling"] = self.r2_ceiling
        return frame.sort_values(
            ["status", "loco_r2"], ascending=[True, False], na_position="last"
        ).reset_index(drop=True)

    def render(self) -> str:
        lines = [
            f"Benchmark study — target '{self.target}'",
            f"  {self.n_rows} rows, {self.n_cells} cells, {self.n_cohorts} cohorts",
            f"  R2 ceiling from target noise floor: {self.r2_ceiling:.4f}",
            "",
            f"  {'method':<30} {'LOBO R2':>9} {'LOCO R2':>9} "
            f"{'LOCO 95% CI':>18} {'delta':>9}  status",
        ]
        for result in sorted(
            self.results,
            key=lambda r: (not r.ran, -(r.loco.median_r2 if r.ran and r.loco
                                        and np.isfinite(r.loco.median_r2) else -np.inf)),
        ):
            row = result.row()
            if row["status"] != "SCORED":
                lines.append(
                    f"  {result.method:<30} {'-':>9} {'-':>9} {'-':>18} "
                    f"{'-':>9}  {row['status']}"
                )
                continue
            marker = "PROMOTED" if row["promoted"] else "rejected"
            if result.target_mismatch:
                marker += f"  [!] models '{result.target_mismatch.split(chr(39))[1]}'"
            # `row()` is typed dict[str, object], so bind through float() to
            # let the type checker see these are numbers.
            ci_low = float(row["loco_r2_ci_low"])  # type: ignore[arg-type]
            ci_high = float(row["loco_r2_ci_high"])  # type: ignore[arg-type]
            ci = (
                f"[{ci_low:.3f}, {ci_high:.3f}]"
                if np.isfinite(ci_low) and np.isfinite(ci_high)
                else f"n={row['loco_n_folds']} too few"
            )
            lines.append(
                f"  {result.method:<30} {row['lobo_r2']:>9.4f} "
                f"{row['loco_r2']:>9.4f} {ci:>18} "
                f"{row['loco_minus_lobo']:>9.4f}  {marker}"
            )
        lines.append("")
        lines.append(
            "  CI: 95% percentile bootstrap of the median, resampling "
            "LOCO folds. It covers spread across the cohorts present, "
            "NOT transfer to an unseen one."
        )
        return "\n".join(lines)


def _confound_fit(data: pd.DataFrame, target: str, column: str) -> FitFn | None:
    """Age-only baseline, refitted per fold. Mirrors AdaptiveCalibrator."""
    if column not in data.columns:
        return None

    def fit(train: pd.DataFrame):
        slope, intercept = np.polyfit(
            train[column].to_numpy(float), train[target].to_numpy(float), 1
        )
        return lambda test: intercept + slope * test[column].to_numpy(float)

    return fit


def run_study(
    data: pd.DataFrame,
    target: str = "cumulative_fade",
    features: Sequence[str] | None = None,
    methods: Sequence[BenchmarkMethod] | None = None,
    cohort_col: str = "cohort",
    cell_col: str = "cell_id",
    confound_col: str = "cycle",
    verbose: bool = False,
) -> StudyResult:
    """Evaluate every method on one target under LOBO and LOCO.

    Rows with a missing target are dropped up front rather than per method, so
    every method sees exactly the same observations — otherwise a method
    tolerant of NaN would be scored on a larger, easier set than one that is
    not, and the comparison would be meaningless. This matters for the
    `horizon_fade_*` targets, which are undefined for the last `h` cycles of
    every cell by construction.
    """
    if target not in data.columns:
        raise ValueError(f"run_study: no target column '{target}'")

    frame = data.dropna(subset=[target]).copy()
    if frame.empty:
        raise ValueError(
            f"run_study: no rows with a defined '{target}'. Horizon targets are "
            f"undefined near each cell's last cycle; check the horizon length "
            f"against the cells' cycle counts."
        )

    catalogue = tuple(methods) if methods is not None else load_all()

    validator = Validator(
        frame, target=target, cohort_col=cohort_col,
        confound_fit=_confound_fit(frame, target, confound_col),
    )

    results: list[MethodResult] = []
    for method in catalogue:
        chosen = method.features_for(features)
        availability = check_availability(method, frame, features=chosen or None)
        if not availability:
            if verbose:
                print(f"    {method.name}: UNAVAILABLE", flush=True)
            results.append(MethodResult(
                method=method.name, family=method.family, availability=availability,
            ))
            continue

        # Refitting across 42 folds takes minutes for the tree ensembles, so a
        # long run with no output is indistinguishable from a hung one.
        started = time.perf_counter()
        if verbose:
            print(f"    {method.name}: running...", end="", flush=True)

        try:
            fit_fn = method.fit_fn(features=chosen or None, target=target)
            lobo = validator.cross_validate(fit_fn, group_col=cell_col, split="LOBO")
            loco = validator.cross_validate(fit_fn, group_col=cohort_col, split="LOCO")
            verdict = validator.gate(lobo, loco)
            if verbose:
                print(f" {time.perf_counter() - started:.1f}s", flush=True)
        except Exception as exc:
            if verbose:
                print(f" ERROR {exc!r}", flush=True)
            results.append(MethodResult(
                method=method.name, family=method.family,
                availability=availability, error=repr(exc),
            ))
            continue

        mismatch = ""
        if method.assumes_target and method.assumes_target != target:
            mismatch = (
                f"method models '{method.assumes_target}' but this study "
                f"scores '{target}'; the error is against a different quantity "
                f"and is not evidence about the method"
            )

        results.append(MethodResult(
            method=method.name, family=method.family, availability=availability,
            lobo=lobo, loco=loco, verdict=verdict, target_mismatch=mismatch,
        ))

    ceiling = signal_to_noise(frame, target, cell_col=cell_col).signal_fraction

    return StudyResult(
        target=target,
        n_rows=len(frame),
        n_cells=int(frame[cell_col].nunique()),
        n_cohorts=int(frame[cohort_col].nunique()) if cohort_col in frame else 0,
        results=tuple(results),
        r2_ceiling=ceiling,
    )
