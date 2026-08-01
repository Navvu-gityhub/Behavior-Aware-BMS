"""Cross-dataset transfer validation: fit on one dataset, test on another.

This is the experiment the project has been unable to run. Leave-one-cohort-out
holds out a *protocol inside NASA*; it cannot tell you whether a model trained
on NASA says anything true about a CALCE or Stanford cell. Those differ in
chemistry, cell format, cycler hardware, temperature control and duty cycle
simultaneously, which is why transfer is a strictly harder test than LOCO and
why it is the one worth reporting.

The baseline question, which decides everything
-----------------------------------------------
When you transfer a model to a new dataset, what should it be compared against?

The tempting answer is the source dataset's mean. That is too easy: NASA and
CALCE cells have different nominal capacities, so a NASA-trained constant is
badly calibrated on CALCE and almost anything beats it. A model can post a
large positive R-squared that way while having learned nothing transferable.

The honest answer is the **target's own mean**. Anyone holding CALCE data could
compute its average fade rate in one line without any model at all. A
transferred model earns its place only by beating that. So
`r2_vs_target_mean` is the headline number here, and `r2_vs_source_mean` is
reported alongside it as a diagnostic — a large gap between the two is the
signature of a model whose apparent transfer is really just a capacity offset.

This mirrors the reasoning in `validation.py`, where R-squared is measured
against the training-fold mean rather than the test-fold mean, for the same
reason: a baseline should be something a practitioner could actually deploy.

Domain shift is reported, never silently corrected
--------------------------------------------------
`DomainShift` quantifies how far the target's feature distributions sit from the
source's. It does not rescale, standardise or otherwise adapt them. Silent
domain adaptation would make a failed transfer look like a successful one, and
the whole value of this harness is that its failures stay visible.

A transfer attempted far outside the source's observed range is extrapolation.
That is worth doing and worth reporting, but it must be labelled, which is what
`features_outside_source_range` is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.bms.adaptive.commensurability import (
    CommensurabilityReport,
    assess_commensurability,
)
from src.bms.adaptive.validation import FitFn, r2_against

# Fraction of the source's observed range a target feature's median may sit
# outside before the transfer is flagged as extrapolation. Chosen to match
# `cohort.DEFAULT_TOLERANCE` so the two guards agree.
EXTRAPOLATION_TOLERANCE = 0.15


@dataclass(frozen=True)
class CompatibilityReport:
    """Whether two datasets share enough structure to attempt transfer at all."""

    source: str
    target: str
    required: tuple[str, ...]
    shared: tuple[str, ...]
    missing_in_source: tuple[str, ...]
    missing_in_target: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.missing_in_source and not self.missing_in_target

    def __bool__(self) -> bool:
        return self.usable

    def render(self) -> str:
        if self.usable:
            return (
                f"{self.source} -> {self.target}: compatible on "
                f"{len(self.shared)} feature(s)"
            )
        parts = [f"{self.source} -> {self.target}: INCOMPATIBLE"]
        if self.missing_in_source:
            parts.append(f"  absent from source: {list(self.missing_in_source)}")
        if self.missing_in_target:
            parts.append(f"  absent from target: {list(self.missing_in_target)}")
        return "\n".join(parts)


@dataclass(frozen=True)
class DomainShift:
    """How far the target's features sit from the source's observed range."""

    per_feature: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    features_outside_source_range: tuple[str, ...] = ()

    @property
    def is_extrapolation(self) -> bool:
        return bool(self.features_outside_source_range)

    def to_frame(self) -> pd.DataFrame:
        if not self.per_feature:
            return pd.DataFrame()
        return pd.DataFrame(self.per_feature).T.reset_index(names="feature")

    def render(self) -> str:
        if not self.per_feature:
            return "  domain shift: not computed"
        lines = ["  domain shift (target median vs source observed range):"]
        for feature, stats in self.per_feature.items():
            flag = " EXTRAPOLATION" if feature in self.features_outside_source_range else ""
            lines.append(
                f"    {feature:<28} target median={stats['target_median']:>9.4g}  "
                f"source range=[{stats['source_min']:.4g}, {stats['source_max']:.4g}]"
                f"{flag}"
            )
        return "\n".join(lines)


def assess_compatibility(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    source_name: str = "source",
    target_name: str = "target",
) -> CompatibilityReport:
    """Do both frames carry the features and the label transfer needs?"""
    required = tuple(features) + (target_column,)
    missing_source = tuple(c for c in required if c not in source.columns)
    missing_target = tuple(c for c in required if c not in target.columns)
    shared = tuple(c for c in required if c in source.columns and c in target.columns)
    return CompatibilityReport(
        source=source_name, target=target_name, required=required,
        shared=shared, missing_in_source=missing_source,
        missing_in_target=missing_target,
    )


def measure_domain_shift(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: Sequence[str],
    tolerance: float = EXTRAPOLATION_TOLERANCE,
) -> DomainShift:
    """Quantify, and only quantify, how different the two feature spaces are."""
    per_feature: dict[str, dict[str, float]] = {}
    outside: list[str] = []

    for feature in features:
        if feature not in source.columns or feature not in target.columns:
            continue
        source_values = pd.to_numeric(source[feature], errors="coerce").dropna()
        target_values = pd.to_numeric(target[feature], errors="coerce").dropna()
        if source_values.empty or target_values.empty:
            continue

        low, high = float(source_values.min()), float(source_values.max())
        target_median = float(target_values.median())
        span = high - low
        pad = tolerance * span if span > 0 else max(abs(high), 1.0) * tolerance

        # Standardised mean difference, using the source's spread as the yardstick
        # because the source is what the model was fitted on.
        source_std = float(source_values.std(ddof=1))
        standardised = (
            float("nan") if source_std <= 0
            else (target_median - float(source_values.mean())) / source_std
        )

        per_feature[feature] = {
            "source_min": low,
            "source_max": high,
            "source_mean": float(source_values.mean()),
            "target_median": target_median,
            "standardised_shift": standardised,
        }
        if target_median < low - pad or target_median > high + pad:
            outside.append(feature)

    return DomainShift(
        per_feature=per_feature, features_outside_source_range=tuple(outside)
    )


@dataclass(frozen=True)
class TransferResult:
    """The outcome of one train-on-A, test-on-B experiment."""

    source: str
    target: str
    n_train: int
    n_test: int

    mae: float
    rmse: float
    spearman_rho: float

    # The headline: did the model beat what the target's own mean would give?
    r2_vs_target_mean: float
    # Diagnostic: beating this only shows the source's scale was wrong.
    r2_vs_source_mean: float

    compatibility: CompatibilityReport
    shift: DomainShift
    commensurability: CommensurabilityReport | None = None
    reasons: tuple[str, ...] = ()
    error: str | None = None

    @property
    def transferred(self) -> bool:
        """Did the model carry usable information to the target?"""
        return (
            self.error is None
            and np.isfinite(self.r2_vs_target_mean)
            and self.r2_vs_target_mean > 0
        )

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "TRANSFERRED" if self.transferred else "DID_NOT_TRANSFER"

    def render(self) -> str:
        lines = [f"{self.source} -> {self.target}: {self.status}"]
        if self.error:
            lines.append(f"  error: {self.error}")
            return "\n".join(lines)

        lines += [
            f"  trained on {self.n_train:,} rows, tested on {self.n_test:,} rows",
            f"  R2 vs TARGET mean (headline) = {self.r2_vs_target_mean:+.4f}",
            f"  R2 vs source mean (diagnostic) = {self.r2_vs_source_mean:+.4f}",
            f"  Spearman rho = {self.spearman_rho:+.4f}",
            f"  MAE = {self.mae:.5g}   RMSE = {self.rmse:.5g}",
        ]
        if self.shift.is_extrapolation:
            lines.append(
                f"  EXTRAPOLATION on: {list(self.shift.features_outside_source_range)}"
            )
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


class TransferValidator:
    """Fits a candidate on a source dataset and evaluates it on a target.

    Unlike `Validator`, which resamples inside one frame, nothing here is
    refitted on target data. That is the point: the model sees the target only
    at prediction time, which is what deployment onto an unseen fleet looks
    like.
    """

    def __init__(
        self,
        source: pd.DataFrame,
        target_column: str,
        source_name: str = "source",
    ) -> None:
        if target_column not in source.columns:
            raise ValueError(
                f"TransferValidator: source has no '{target_column}' column"
            )
        if source.empty:
            raise ValueError("TransferValidator: source frame is empty")
        self.source = source
        self.target_column = target_column
        self.source_name = source_name

    def evaluate(
        self,
        fit_fn: FitFn,
        target: pd.DataFrame,
        features: Sequence[str],
        target_name: str = "target",
    ) -> TransferResult:
        """Run one transfer experiment and report it whichever way it goes."""
        features = tuple(features)
        compatibility = assess_compatibility(
            self.source, target, features, self.target_column,
            self.source_name, target_name,
        )
        shift = measure_domain_shift(self.source, target, features)
        nan = float("nan")

        commensurability = assess_commensurability(
            self.source, target, features, self.source_name, target_name,
        )

        def failed(error: str, reasons: tuple[str, ...] = ()) -> TransferResult:
            return TransferResult(
                source=self.source_name, target=target_name,
                n_train=len(self.source), n_test=len(target),
                mae=nan, rmse=nan, spearman_rho=nan,
                r2_vs_target_mean=nan, r2_vs_source_mean=nan,
                compatibility=compatibility, shift=shift,
                commensurability=commensurability,
                reasons=reasons, error=error,
            )

        if not compatibility.usable:
            return failed(
                "feature spaces are not compatible",
                (compatibility.render(),),
            )
        if target.empty:
            return failed("target frame is empty")

        # Precondition: at least one feature must vary in BOTH datasets. If
        # none does, a transfer metric would describe an intercept rather than
        # a prediction, so no number is produced. See commensurability.py.
        if not commensurability.feasible:
            return failed(
                "no feature varies in both datasets",
                (commensurability.render(),),
            )

        try:
            predict = fit_fn(self.source)
            predicted = np.asarray(predict(target), dtype=float)
        except Exception as exc:
            return failed(f"{type(exc).__name__}: {exc}")

        observed = target[self.target_column].to_numpy(dtype=float)
        if predicted.shape != observed.shape:
            return failed(
                f"prediction shape {predicted.shape} != target shape {observed.shape}"
            )

        finite = np.isfinite(predicted) & np.isfinite(observed)
        if not finite.any():
            return failed("no finite prediction/observation pairs")
        predicted, observed = predicted[finite], observed[finite]

        target_mean = np.full_like(observed, float(observed.mean()))
        source_mean = np.full_like(
            observed, float(self.source[self.target_column].mean())
        )
        rho = pd.Series(predicted).corr(pd.Series(observed), method="spearman")

        r2_target = r2_against(observed, predicted, target_mean)
        r2_source = r2_against(observed, predicted, source_mean)

        reasons: list[str] = []
        if np.isfinite(r2_target) and r2_target <= 0:
            reasons.append(
                "Did not beat the target's own mean. Anyone holding this dataset "
                "could compute that in one line, so the model has added nothing."
            )
        if (
            np.isfinite(r2_target) and np.isfinite(r2_source)
            and r2_source > 0 and r2_source - r2_target > 0.1
        ):
            reasons.append(
                f"R2 against the source mean ({r2_source:+.4f}) far exceeds R2 "
                f"against the target mean ({r2_target:+.4f}). The apparent "
                f"transfer is mostly a capacity-scale offset between datasets, "
                f"not learned degradation behaviour."
            )
        if commensurability.status == "FEASIBLE_REDUCED":
            reasons.append(
                f"Transfer used a reduced feature set. Excluded: "
                f"{list(commensurability.constant_in_source + commensurability.constant_in_target)}. "
                f"A feature constant on either side cannot carry a transfer."
            )
        if shift.is_extrapolation:
            reasons.append(
                f"Target medians for "
                f"{list(shift.features_outside_source_range)} sit outside the "
                f"source's observed range, so this is extrapolation."
            )
        if np.isfinite(rho) and rho < 0 and np.isfinite(r2_target) and r2_target > 0:
            reasons.append(
                f"Rank correlation is negative ({rho:+.4f}) despite positive R2. "
                f"The model's error is small but its ordering of cells is wrong, "
                f"which matters if the output is used to prioritise cells."
            )

        return TransferResult(
            source=self.source_name, target=target_name,
            n_train=int(len(self.source)), n_test=int(finite.sum()),
            mae=float(np.mean(np.abs(observed - predicted))),
            rmse=float(np.sqrt(np.mean((observed - predicted) ** 2))),
            spearman_rho=float(rho) if rho is not None and np.isfinite(rho) else nan,
            r2_vs_target_mean=r2_target,
            r2_vs_source_mean=r2_source,
            compatibility=compatibility, shift=shift,
            commensurability=commensurability,
            reasons=tuple(reasons),
        )

    def evaluate_many(
        self,
        fit_fn: FitFn,
        targets: Mapping[str, pd.DataFrame],
        features: Sequence[str],
    ) -> list[TransferResult]:
        """Transfer to several target datasets, reporting each separately.

        Results are deliberately not averaged. A model that transfers to one
        dataset and fails on another has told you something specific about
        which conditions it covers, and a mean across them would erase exactly
        that information.
        """
        return [
            self.evaluate(fit_fn, frame, features, target_name=name)
            for name, frame in targets.items()
        ]


def transfer_summary(results: Sequence[TransferResult]) -> pd.DataFrame:
    """Tabulate transfer results for the report."""
    return pd.DataFrame([
        {
            "source": r.source,
            "target": r.target,
            "status": r.status,
            "n_train": r.n_train,
            "n_test": r.n_test,
            "r2_vs_target_mean": r.r2_vs_target_mean,
            "r2_vs_source_mean": r.r2_vs_source_mean,
            "spearman_rho": r.spearman_rho,
            "mae": r.mae,
            "extrapolation": r.shift.is_extrapolation,
            "notes": " | ".join(r.reasons),
        }
        for r in results
    ])
