"""Cohort identity and distribution guards.

A "cohort" is an experimental protocol: a combination of ambient temperature,
discharge current, load profile and cutoff voltage under which a set of cells
was cycled. NASA's 34 cells span 9 of them.

Why this module exists at all
-----------------------------
`docs/adr/0002` records the finding that motivates everything here: the fitted
health model ranks an unseen cell within a **known** protocol at rho=0.841 and
an unseen **protocol** at rho=-0.295. The skill is in the fitted per-cohort
intercepts, not in the physics. Therefore the single most important question
to ask about any incoming battery is not "what is its health index?" but
"**have we ever seen its operating conditions before?**"

If the answer is no, every downstream number is extrapolation, and the system
must say so rather than emit a confident score.

What a CohortSpec is and is not
-------------------------------
A `CohortSpec` is an *observed envelope*, derived from data with
`CohortSpec.from_observations`. It is deliberately not a hand-authored
description of what a protocol "should" be. Hand-authored envelopes are how
the risk score acquired thresholds that no real data ever reaches — see
`scripts/audit_threshold_reachability.py`, which found 61% of the mean risk
score comes from terms that are constant across the entire calibration set
because their cut points sit outside the observed range.

Membership is judged on the envelope plus a tolerance, not on a distance
metric or a density model. That is a deliberate simplicity choice: with 3-4
cells per cohort there is nowhere near enough data to fit a density model
whose tails could be trusted, and a Mahalanobis distance over correlated
telemetry features would imply a covariance estimate this data cannot
support. A tolerance band is crude, but it is honest about being crude and it
fails in a predictable direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

# Features that define an operating envelope. These are the variables that
# distinguish NASA's protocols from one another (ambient temperature,
# discharge rate, depth of discharge) and therefore the ones along which a
# new dataset is either familiar or not.
#
# Deliberately excluded: anything derived from capacity or cycle count. Those
# describe how far a cell has aged, not the conditions it aged under, and
# including them would make an old cell look out-of-distribution purely for
# being old.
ENVELOPE_FEATURES: tuple[str, ...] = (
    "avg_temp",
    "max_temp",
    "avg_soc",
    "aggressive_discharge_count",
)

# How far outside the observed envelope a value may sit and still count as
# the same protocol, as a fraction of the observed range.
#
# 0.15 is a judgement call, not a fitted quantity, and is exposed as an
# argument so it can be tightened. The reasoning: NASA cohorts hold 3-4 cells,
# so the observed min/max is a small-sample estimate that certainly understates
# the true spread of the protocol. Zero tolerance would flag ordinary
# cell-to-cell variation as a new protocol. Wide tolerance would let a
# genuinely different regime pass as familiar, which is the more dangerous
# error, so the band is kept narrow.
DEFAULT_TOLERANCE = 0.15


@dataclass(frozen=True)
class CohortSpec:
    """The observed operating envelope of one experimental protocol."""

    cohort_id: str
    n_cells: int
    n_observations: int
    # feature -> (observed_min, observed_max)
    bounds: Mapping[str, tuple[float, float]]
    source: str = "unknown"

    @classmethod
    def from_observations(
        cls,
        cohort_id: str,
        data: pd.DataFrame,
        cell_col: str = "cell_id",
        features: Iterable[str] = ENVELOPE_FEATURES,
        source: str = "unknown",
    ) -> "CohortSpec":
        """Derive an envelope from data actually observed under this protocol."""
        features = tuple(features)
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise ValueError(
                f"CohortSpec.from_observations: {cohort_id} is missing envelope "
                f"features {missing}. Envelopes are derived from observations, "
                f"so every feature must be present."
            )
        if data.empty:
            raise ValueError(f"CohortSpec.from_observations: {cohort_id} has no rows")

        bounds: dict[str, tuple[float, float]] = {}
        for feature in features:
            column = pd.to_numeric(data[feature], errors="coerce").dropna()
            if column.empty:
                raise ValueError(
                    f"CohortSpec.from_observations: {cohort_id} feature "
                    f"'{feature}' has no numeric values"
                )
            bounds[feature] = (float(column.min()), float(column.max()))

        return cls(
            cohort_id=cohort_id,
            n_cells=int(data[cell_col].nunique()) if cell_col in data.columns else 0,
            n_observations=int(len(data)),
            bounds=bounds,
            source=source,
        )

    def contains(
        self, row: Mapping[str, float], tolerance: float = DEFAULT_TOLERANCE
    ) -> bool:
        """Does one observation sit inside this envelope (plus tolerance)?"""
        return not self.violations(row, tolerance)

    def violations(
        self, row: Mapping[str, float], tolerance: float = DEFAULT_TOLERANCE
    ) -> dict[str, str]:
        """Which envelope features does this observation fall outside, and how."""
        out: dict[str, str] = {}
        for feature, (low, high) in self.bounds.items():
            if feature not in row:
                out[feature] = "absent from observation"
                continue
            value = row[feature]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                out[feature] = "missing value"
                continue

            # A zero-width envelope (every observation identical) gets an
            # absolute tolerance instead, since a fraction of zero is zero and
            # would reject any value at all. This is not hypothetical:
            # `fast_charge_duration` is identically zero across all 2,682 NASA
            # observations.
            span = high - low
            pad = tolerance * span if span > 0 else max(abs(high), 1.0) * tolerance

            if value < low - pad:
                out[feature] = f"{value:.4g} below observed min {low:.4g}"
            elif value > high + pad:
                out[feature] = f"{value:.4g} above observed max {high:.4g}"
        return out


@dataclass(frozen=True)
class InDistribution:
    """A successful cohort match."""

    cohort_id: str
    n_reference_cells: int

    status: str = "IN_DISTRIBUTION"
    is_known: bool = True

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class DriftReport:
    """A failed cohort match, with the reason preserved.

    Carries *why* nothing matched rather than a bare False, because the caller
    needs to be able to tell a genuinely novel regime ("ambient temperature
    -20C, never observed") from a data problem ("avg_temp column absent").
    """

    nearest_cohort: str | None
    violations: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    status: str = "OUT_OF_DISTRIBUTION"
    is_known: bool = False

    def __bool__(self) -> bool:
        return False

    def summary(self) -> str:
        if not self.violations:
            return "No registered cohorts to compare against."
        parts = []
        for cohort_id, feature_violations in self.violations.items():
            detail = "; ".join(f"{k}: {v}" for k, v in feature_violations.items())
            parts.append(f"{cohort_id} ({detail})")
        return " | ".join(parts)


class CohortRegistry:
    """The set of protocols the system has evidence about.

    Membership here is the precondition for trusting any downstream score. A
    battery whose conditions match no registered cohort is not scored badly —
    it is refused a score, because ADR 0002 shows what the model does on an
    unseen protocol and the answer is "worse than nothing, confidently".
    """

    def __init__(self, tolerance: float = DEFAULT_TOLERANCE) -> None:
        self._specs: dict[str, CohortSpec] = {}
        self.tolerance = tolerance

    # -- construction -------------------------------------------------------

    def register(self, spec: CohortSpec) -> None:
        self._specs[spec.cohort_id] = spec

    @classmethod
    def from_training_data(
        cls,
        data: pd.DataFrame,
        cohort_col: str = "cohort",
        cell_col: str = "cell_id",
        features: Iterable[str] = ENVELOPE_FEATURES,
        source: str = "unknown",
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> "CohortRegistry":
        """Build a registry from a labelled training frame."""
        if cohort_col not in data.columns:
            raise ValueError(
                f"CohortRegistry.from_training_data: no '{cohort_col}' column. "
                f"Cohort labels are required; pooling protocols is exactly what "
                f"ADR 0002 shows produces misleading validation numbers."
            )
        registry = cls(tolerance=tolerance)
        for cohort_id, group in data.groupby(cohort_col):
            registry.register(
                CohortSpec.from_observations(
                    str(cohort_id), group, cell_col=cell_col,
                    features=features, source=source,
                )
            )
        return registry

    # -- queries ------------------------------------------------------------

    @property
    def cohort_ids(self) -> list[str]:
        return sorted(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, cohort_id: object) -> bool:
        return cohort_id in self._specs

    def get(self, cohort_id: str) -> CohortSpec:
        if cohort_id not in self._specs:
            raise KeyError(
                f"Unknown cohort '{cohort_id}'. Registered: {self.cohort_ids}"
            )
        return self._specs[cohort_id]

    def identify(self, observation: Mapping[str, float]) -> InDistribution | DriftReport:
        """Which registered protocol does this observation belong to?

        Returns a truthy `InDistribution` on a match and a falsy `DriftReport`
        otherwise, so callers can branch on the result directly while still
        having the diagnosis available.

        Ties are resolved by observation count: with overlapping envelopes the
        better-evidenced cohort wins, since it is the one whose fitted
        intercept is worth more.
        """
        matches = [
            spec for spec in self._specs.values()
            if spec.contains(observation, self.tolerance)
        ]
        if matches:
            best = max(matches, key=lambda s: s.n_observations)
            return InDistribution(cohort_id=best.cohort_id, n_reference_cells=best.n_cells)

        violations = {
            spec.cohort_id: spec.violations(observation, self.tolerance)
            for spec in self._specs.values()
        }
        nearest = min(violations, key=lambda k: len(violations[k])) if violations else None
        return DriftReport(nearest_cohort=nearest, violations=violations)

    def screen(
        self,
        data: pd.DataFrame,
        cell_col: str = "cell_id",
    ) -> pd.DataFrame:
        """Classify every cell in a frame as in- or out-of-distribution.

        Screening is per cell rather than per row on purpose: a single hot row
        is a transient, whereas a cell whose *aggregate* conditions sit outside
        every envelope is operating in a regime the model has no evidence for.
        """
        if cell_col not in data.columns:
            raise ValueError(f"CohortRegistry.screen: no '{cell_col}' column")

        rows = []
        for cell_id, group in data.groupby(cell_col):
            observation = {
                feature: float(pd.to_numeric(group[feature], errors="coerce").mean())
                for feature in ENVELOPE_FEATURES
                if feature in group.columns
            }
            result = self.identify(observation)
            rows.append({
                cell_col: cell_id,
                "status": result.status,
                "matched_cohort": getattr(result, "cohort_id", None),
                "nearest_cohort": getattr(result, "nearest_cohort", None),
                "detail": "" if result else result.summary(),
            })
        return pd.DataFrame(rows)
