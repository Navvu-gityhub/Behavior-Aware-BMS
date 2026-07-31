"""Feature commensurability: can two datasets support a transfer test at all?

This module answers a question that has to come before loaders, before feature
mapping, and before any transfer metric is computed: **do the two datasets vary
along the same axis?**

The finding that motivates it
----------------------------
NASA and Stanford/Severson are both excellent multi-cycle datasets. They are
also close to useless for transferring a model between, and the reason is
structural rather than technical.

NASA's 34 cells span ambient temperatures from roughly 4 C to 43 C across nine
protocols. Temperature is the axis its experiment varies, and temperature is
the one signal this project found transferable (coefficient 0.0038 Ah/degC,
correct sign in 7/7 cells across independent cohorts).

Severson's 124 A123 LFP cells were cycled in a forced-convection chamber set to
30 C, discharged identically at 4C, and varied by *charging policy* from 3.6C
to 6C. Cell temperature is recorded and does fluctuate — by up to about 10 C
from self-heating — but it is not an experimental variable. Meanwhile NASA's
`fast_charge_duration` is identically zero across all 2,682 observations.

So the axis NASA varies is held constant in Severson, and the axis Severson
varies does not exist in NASA. A temperature model transferred to Severson has
almost no target variation to predict; a charge-rate model cannot be fitted on
NASA at all.

This is worth knowing *before* writing a parser, which is why the check lives
here and runs as a precondition rather than as a post-hoc diagnostic.

What "commensurable" means here
-------------------------------
A feature is usable for transfer only if it varies in **both** datasets:

- Constant in the source, and no coefficient can be fitted for it.
- Constant in the target, and the fitted coefficient has nothing to act on:
  every target row gets the same contribution, which is an intercept, not a
  prediction.

Both failures produce a model that looks fitted and predicts nothing, which is
the same class of defect as the degenerate risk terms found by
`scripts/audit_threshold_reachability.py` — 61% of the mean risk score came
from terms whose cut points no observation ever reached.

How variation is judged, and why not by coefficient of variation
----------------------------------------------------------------
The first version of this module used the coefficient of variation (std /
|mean|) against a fixed floor. That is wrong for the quantity that matters most
here, and the error is instructive.

Celsius is an interval scale with an arbitrary zero: 70 C is not "2.8 times
more temperature" than 25 C. So a fixed CV floor makes the *same* physical
spread of +/-1.5 C count as varying at 25 C (CV 0.06) and constant at 70 C
(CV 0.02). Applied to real data it would have been worse still — NASA's 4 C
cohort with +/-1.5 C scores CV 0.375 and looks highly variable, while
Severson's 30 C chamber with a larger absolute excursion scores about 0.08 and
looks flat.

The judgement is therefore made **relative to the source's own spread**, which
is the range the coefficient was fitted over:

- The source needs enough distinct values to fit a slope at all.
- The target needs a standard deviation that is a meaningful fraction of the
  source's, because that ratio is what determines whether the fitted
  coefficient produces visible spread in target predictions.

`FeatureVariation` is consequently a pure measurement and holds no opinion
about what counts as constant. The comparison lives in
`assess_commensurability`, where both datasets are in scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Minimum ratio of target spread to source spread for a feature to carry a
# transfer.
#
# If the target varies over 10% of the range the coefficient was fitted across,
# the fitted slope still produces visible spread in target predictions. Below
# that, every target row receives nearly the same contribution and the feature
# is functioning as an intercept.
#
# This is a judgement call, exposed as an argument so it can be tightened. It is
# deliberately permissive: the intent is to exclude degenerate channels, not to
# adjudicate marginal ones, and a marginal case should surface in the report as
# FEASIBLE_REDUCED rather than be silently dropped.
MIN_TARGET_SPREAD_RATIO = 0.10

# A feature needs more than a handful of distinct values to carry a slope.
MIN_DISTINCT_VALUES = 5


@dataclass(frozen=True)
class FeatureVariation:
    """How much one feature actually varies within one dataset."""

    feature: str
    dataset: str
    n_observations: int
    n_distinct: int
    mean: float
    std: float
    minimum: float
    maximum: float

    @property
    def can_fit_a_slope(self) -> bool:
        """Enough distinct values for a coefficient to be estimable at all."""
        return self.n_distinct >= MIN_DISTINCT_VALUES and self.std > 0

    def spread_ratio_against(self, reference: "FeatureVariation") -> float:
        """This feature's spread as a fraction of the reference's.

        Scale-free in the way that matters: it compares like with like on the
        same channel, so it does not care whether the channel is Celsius,
        amperes or a count.
        """
        if reference.std <= 0:
            return float("nan")
        return self.std / reference.std

    def render(self) -> str:
        return (
            f"{self.dataset}.{self.feature}: mean={self.mean:.4g} "
            f"std={self.std:.4g} "
            f"range=[{self.minimum:.4g}, {self.maximum:.4g}] "
            f"distinct={self.n_distinct}"
        )


def measure_variation(
    data: pd.DataFrame, feature: str, dataset: str = "dataset"
) -> FeatureVariation | None:
    """Measure one feature's variation, or None if it is absent/non-numeric."""
    if feature not in data.columns:
        return None
    values = pd.to_numeric(data[feature], errors="coerce").dropna()
    if values.empty:
        return None
    return FeatureVariation(
        feature=feature,
        dataset=dataset,
        n_observations=int(len(values)),
        n_distinct=int(values.nunique()),
        mean=float(values.mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


@dataclass(frozen=True)
class CommensurabilityReport:
    """Which features can carry a transfer between two specific datasets."""

    source: str
    target: str
    usable_features: tuple[str, ...] = ()
    constant_in_source: tuple[str, ...] = ()
    constant_in_target: tuple[str, ...] = ()
    absent_in_source: tuple[str, ...] = ()
    absent_in_target: tuple[str, ...] = ()
    variation: Mapping[str, Mapping[str, FeatureVariation]] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Is there at least one feature that varies in both datasets?"""
        return bool(self.usable_features)

    def __bool__(self) -> bool:
        return self.feasible

    @property
    def status(self) -> str:
        if not self.feasible:
            return "NOT_FEASIBLE"
        blocked = (
            self.constant_in_source + self.constant_in_target
            + self.absent_in_source + self.absent_in_target
        )
        return "FEASIBLE_REDUCED" if blocked else "FEASIBLE"

    def render(self) -> str:
        lines = [f"{self.source} -> {self.target}: {self.status}"]
        if self.usable_features:
            lines.append(f"  usable: {list(self.usable_features)}")
        for label, features in (
            ("constant in source (no coefficient can be fitted)", self.constant_in_source),
            ("constant in target (coefficient has nothing to act on)", self.constant_in_target),
            ("absent from source", self.absent_in_source),
            ("absent from target", self.absent_in_target),
        ):
            if features:
                lines.append(f"  {label}: {list(features)}")
        if not self.feasible:
            lines.append(
                "  No feature varies in both datasets. A transfer metric "
                "computed here would describe an intercept, not a prediction."
            )
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """Per-feature, per-dataset variation, for the report."""
        rows = []
        for feature, per_dataset in self.variation.items():
            for dataset, measurement in per_dataset.items():
                rows.append({
                    "feature": feature,
                    "dataset": dataset,
                    "mean": measurement.mean,
                    "std": measurement.std,
                    "minimum": measurement.minimum,
                    "maximum": measurement.maximum,
                    "n_distinct": measurement.n_distinct,
                    "can_fit_a_slope": measurement.can_fit_a_slope,
                })
        return pd.DataFrame(rows)


def assess_commensurability(
    source: pd.DataFrame,
    target: pd.DataFrame,
    features: Sequence[str],
    source_name: str = "source",
    target_name: str = "target",
    min_target_spread_ratio: float = MIN_TARGET_SPREAD_RATIO,
) -> CommensurabilityReport:
    """Decide which candidate features can actually carry a transfer.

    Runs before any model is fitted. A feature that is constant on either side
    is excluded with the reason recorded, and if nothing survives the transfer
    is reported as infeasible rather than attempted.
    """
    usable: list[str] = []
    constant_source: list[str] = []
    constant_target: list[str] = []
    absent_source: list[str] = []
    absent_target: list[str] = []
    variation: dict[str, dict[str, FeatureVariation]] = {}

    for feature in features:
        source_measurement = measure_variation(source, feature, source_name)
        target_measurement = measure_variation(target, feature, target_name)

        per_dataset: dict[str, FeatureVariation] = {}
        if source_measurement is not None:
            per_dataset[source_name] = source_measurement
        if target_measurement is not None:
            per_dataset[target_name] = target_measurement
        if per_dataset:
            variation[feature] = per_dataset

        if source_measurement is None:
            absent_source.append(feature)
            continue
        if target_measurement is None:
            absent_target.append(feature)
            continue
        if not source_measurement.can_fit_a_slope:
            constant_source.append(feature)
            continue

        ratio = target_measurement.spread_ratio_against(source_measurement)
        if not target_measurement.can_fit_a_slope or (
            np.isfinite(ratio) and ratio < min_target_spread_ratio
        ):
            constant_target.append(feature)
            continue
        usable.append(feature)

    return CommensurabilityReport(
        source=source_name,
        target=target_name,
        usable_features=tuple(usable),
        constant_in_source=tuple(constant_source),
        constant_in_target=tuple(constant_target),
        absent_in_source=tuple(absent_source),
        absent_in_target=tuple(absent_target),
        variation=variation,
    )
