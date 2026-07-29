"""The calibration orchestrator.

Wires the four components into one loop: screen a dataset, learn its cohort
envelopes, cross-validate each candidate, and offer the survivors to the store.

This module deliberately adds **no judgement of its own.** Every decision it
makes is delegated — suitability to `datasets.assess_suitability`,
generalisation to `validation.Validator.gate`, stability to
`store.ModelStore.propose`, distribution membership to
`cohort.CohortRegistry`. An orchestrator that quietly applied its own extra
criteria would be a fifth set of thresholds nobody reviewed, which is the
pattern that produced a Guardian explaining scores with a rulebook that existed
nowhere else (ADR 0003).

What happens when nothing passes
--------------------------------
On current evidence, nothing does. Both the shipped v2 specification and a
temperature-only alternative are rejected on the NASA frame. That is not a
degenerate case to be worked around — it is the expected steady state until a
second multi-cycle dataset exists, and the system's behaviour in it is the
most important thing this module defines.

`score` refuses. It does not fall back to the most recent rejected candidate,
and it does not fall back to the rule-based v1 index. A refusal carrying a
reason is more useful than a number carrying none, and ADR 0002 already
established what the alternative looks like: v1 ranks batteries against
measured fade at rho=-0.269, which is not merely unvalidated but pointed the
wrong way.

The rule-based pipeline in `main.py` still runs and still emits its severity
score. That is fine, and unchanged, because it is labelled throughout as
triage rather than measurement. What this module refuses to do is dress that
score up as a calibrated fade prediction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.bms.adaptive.cohort import CohortRegistry, DriftReport, InDistribution
from src.bms.adaptive.datasets import DatasetRegistry, SuitabilityReport
from src.bms.adaptive.store import ModelStore, ModelVersion
from src.bms.adaptive.validation import (
    CrossValidationResult,
    FitFn,
    Validator,
    Verdict,
)


@dataclass(frozen=True)
class CandidateSpec:
    """A model offered for promotion.

    `extract_params` returns the JSON-serialisable parameters that will be
    stored if the candidate is promoted. Requiring the caller to supply it,
    rather than introspecting the fitted object, keeps the store's
    inspectability constraint (see `store.py`) enforced at the point where
    someone can actually do something about it.
    """

    name: str
    fit_fn: FitFn
    extract_params: Callable[[pd.DataFrame], Mapping[str, float]]
    cohort_scope: str = "GLOBAL"


@dataclass(frozen=True)
class CandidateOutcome:
    """What happened to one candidate in one run."""

    name: str
    verdict: Verdict
    lobo: CrossValidationResult
    loco: CrossValidationResult | None
    stored: ModelVersion | None
    error: str | None = None

    @property
    def promoted(self) -> bool:
        return self.stored is not None


@dataclass(frozen=True)
class CalibrationRun:
    """Everything one calibration pass produced."""

    dataset: str
    timestamp: float
    suitability: SuitabilityReport
    outcomes: tuple[CandidateOutcome, ...] = ()
    cohorts_registered: tuple[str, ...] = ()
    aborted: str | None = None

    @property
    def promoted(self) -> list[CandidateOutcome]:
        return [o for o in self.outcomes if o.promoted]

    def render(self) -> str:
        lines = [f"Calibration run on '{self.dataset}'"]
        if self.aborted:
            lines.append(f"  ABORTED: {self.aborted}")
            lines.append("  " + self.suitability.render().replace("\n", "\n  "))
            return "\n".join(lines)

        lines.append(f"  dataset: {self.suitability.status}")
        lines.append(f"  cohorts registered: {len(self.cohorts_registered)}")
        for outcome in self.outcomes:
            if outcome.error:
                lines.append(f"  {outcome.name}: ERROR - {outcome.error}")
                continue
            marker = "PROMOTED" if outcome.promoted else "REJECTED"
            lobo = outcome.lobo.median_r2
            loco = outcome.loco.median_r2 if outcome.loco else float("nan")
            lines.append(
                f"  {outcome.name}: {marker} (LOBO R2={lobo:+.4f}, LOCO R2={loco:+.4f})"
            )
            for reason in outcome.verdict.reasons:
                lines.append(f"      - {reason}")
        if not self.promoted:
            lines.append("  No candidate was promoted. No cohort has an active model.")
        return "\n".join(lines)


@dataclass(frozen=True)
class ScoringRefusal:
    """A refusal to emit a prediction, with the reason preserved."""

    battery_id: str
    reason: str
    detail: str = ""

    status: str = "REFUSED"
    prediction: float | None = None

    def __bool__(self) -> bool:
        return False


@dataclass(frozen=True)
class Scored:
    """A prediction, with the evidence that licenses it."""

    battery_id: str
    cohort_id: str
    prediction: float
    model_version: int
    loco_median_r2: float | None = None

    status: str = "SCORED"

    def __bool__(self) -> bool:
        return True


class AdaptiveCalibrator:
    """Runs the calibrate-and-promote loop, and scores only what it can justify."""

    def __init__(
        self,
        store: ModelStore,
        datasets: DatasetRegistry,
        target: str = "capacity_loss",
        cohort_col: str = "cohort",
        cell_col: str = "cell_id",
    ) -> None:
        self.store = store
        self.datasets = datasets
        self.target = target
        self.cohort_col = cohort_col
        self.cell_col = cell_col
        self.cohorts = CohortRegistry()

    # -- calibration --------------------------------------------------------

    def calibrate(
        self,
        dataset_name: str,
        candidates: Sequence[CandidateSpec],
    ) -> CalibrationRun:
        """Screen, learn envelopes, validate every candidate, offer survivors.

        Every candidate is evaluated even after one is promoted. Stopping at
        the first success would hide the comparison, and a rejected
        alternative is evidence about the promoted one.
        """
        started = time.time()

        report = self.datasets.assess(dataset_name, cohort_col=self.cohort_col)
        if not report.usable:
            return CalibrationRun(
                dataset=dataset_name, timestamp=started, suitability=report,
                aborted="dataset cannot support fade calibration",
            )

        data, report = self.datasets.load(dataset_name, cohort_col=self.cohort_col)
        if self.target not in data.columns:
            return CalibrationRun(
                dataset=dataset_name, timestamp=started, suitability=report,
                aborted=f"target column '{self.target}' absent from the loaded frame",
            )

        registered = self._learn_cohorts(data, source=dataset_name)

        validator = Validator(data, target=self.target, cohort_col=self.cohort_col)
        outcomes = [
            self._evaluate(candidate, validator, data)
            for candidate in candidates
        ]

        return CalibrationRun(
            dataset=dataset_name,
            timestamp=started,
            suitability=report,
            outcomes=tuple(outcomes),
            cohorts_registered=tuple(registered),
        )

    def _learn_cohorts(self, data: pd.DataFrame, source: str) -> list[str]:
        if self.cohort_col not in data.columns:
            return []
        learned = CohortRegistry.from_training_data(
            data, cohort_col=self.cohort_col, cell_col=self.cell_col, source=source,
        )
        for cohort_id in learned.cohort_ids:
            self.cohorts.register(learned.get(cohort_id))
        return learned.cohort_ids

    def _evaluate(
        self,
        candidate: CandidateSpec,
        validator: Validator,
        data: pd.DataFrame,
    ) -> CandidateOutcome:
        empty = CrossValidationResult(split="LOBO", group_col=self.cell_col, folds=[])
        try:
            lobo = validator.cross_validate(
                candidate.fit_fn, group_col=self.cell_col, split="LOBO"
            )
            loco = (
                validator.cross_validate(
                    candidate.fit_fn, group_col=self.cohort_col, split="LOCO"
                )
                if self.cohort_col in data.columns
                else None
            )
        except Exception as exc:
            return CandidateOutcome(
                name=candidate.name,
                verdict=Verdict(False, [f"cross-validation raised {exc!r}"]),
                lobo=empty, loco=None, stored=None, error=repr(exc),
            )

        verdict = validator.gate(lobo, loco)

        # Only fit on the full frame if the candidate is going to be stored.
        # Fitting a rejected candidate would burn time and, worse, produce an
        # artifact that looks storable.
        stored = None
        if verdict.promote:
            try:
                params = candidate.extract_params(data)
                stored = self.store.propose(
                    candidate.cohort_scope, params, verdict, note=candidate.name,
                )
            except Exception as exc:
                return CandidateOutcome(
                    name=candidate.name, verdict=verdict, lobo=lobo, loco=loco,
                    stored=None, error=f"parameter extraction failed: {exc!r}",
                )

        return CandidateOutcome(
            name=candidate.name, verdict=verdict, lobo=lobo, loco=loco, stored=stored,
        )

    # -- scoring ------------------------------------------------------------

    def score(
        self,
        battery: Mapping[str, float],
        battery_id: str = "unknown",
        cohort_scope: str = "GLOBAL",
    ) -> Scored | ScoringRefusal:
        """Predict fade for one battery, or refuse and say why.

        Two independent conditions must hold, and both are refusals rather
        than degraded answers:

        1. The battery's operating conditions must match a known cohort. ADR
           0002 measured what the model does outside its training protocols
           (rho=-0.295) and the answer is worse than silence.
        2. A model must have survived the gate. There is no fallback to the
           most recent rejected candidate.
        """
        if len(self.cohorts) == 0:
            return ScoringRefusal(
                battery_id=battery_id,
                reason="no cohorts registered",
                detail="Run calibrate() on a usable dataset before scoring.",
            )

        membership = self.cohorts.identify(battery)
        if isinstance(membership, DriftReport):
            return ScoringRefusal(
                battery_id=battery_id,
                reason="operating conditions outside every known protocol",
                detail=membership.summary(),
            )

        active = self.store.active(cohort_scope)
        if active is None:
            return ScoringRefusal(
                battery_id=battery_id,
                reason="no model has passed the promotion gate",
                detail=(
                    "Every candidate evaluated so far failed out-of-sample "
                    "validation, so there is nothing calibrated to score with. "
                    "The rule-based severity index in main.py still runs and is "
                    "labelled as triage, not as a fade measurement."
                ),
            )

        prediction = self._apply(active, battery)
        if prediction is None:
            return ScoringRefusal(
                battery_id=battery_id,
                reason="active model does not cover this battery's features",
                detail=f"model v{active.version} expects {sorted(active.params)}",
            )

        return Scored(
            battery_id=battery_id,
            cohort_id=membership.cohort_id,
            prediction=prediction,
            model_version=active.version,
            loco_median_r2=active.metrics.get("loco_median_r2"),
        )

    @staticmethod
    def _apply(model: ModelVersion, battery: Mapping[str, float]) -> float | None:
        """Evaluate a stored linear model.

        Stored parameters are a flat feature->coefficient mapping plus an
        optional 'intercept', which is the only form the JSON constraint in
        `store.py` admits. A stored model referencing a feature the battery
        lacks returns None rather than substituting zero, since a missing
        feature is not the same as a feature measured at zero — the same
        distinction the NaN-as-healthy fix enforced elsewhere in this project.
        """
        total = float(model.params.get("intercept", 0.0))
        for feature, coefficient in model.params.items():
            if feature == "intercept":
                continue
            if feature not in battery:
                return None
            value = battery[feature]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
            total += float(coefficient) * float(value)
        return total

    def screen(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify every cell in a frame against the learned envelopes."""
        if len(self.cohorts) == 0:
            raise RuntimeError(
                "AdaptiveCalibrator.screen: no cohorts registered. Run "
                "calibrate() on a usable dataset first."
            )
        return self.cohorts.screen(data, cell_col=self.cell_col)


# ---------------------------------------------------------------------------
# Convenience constructor for the linear specifications this project uses
# ---------------------------------------------------------------------------

def linear_candidate(
    name: str,
    features: Sequence[str],
    target: str = "capacity_loss",
    cohort_scope: str = "GLOBAL",
) -> CandidateSpec:
    """Build an ordinary-least-squares candidate over `features`.

    Uses numpy's least squares rather than statsmodels so the package carries
    no hard dependency on it, and so the stored parameters are unambiguously a
    flat coefficient mapping that `_apply` can evaluate.

    Note what this deliberately cannot express: per-cohort intercepts. A
    candidate wanting those must supply its own `fit_fn`, and ADR 0002 is the
    reason to think twice — the intercepts are where the v2 model's apparent
    skill turned out to live.
    """
    features = tuple(features)

    def fit_fn(train: pd.DataFrame):
        matrix = np.column_stack(
            [train[f].to_numpy(float) for f in features] + [np.ones(len(train))]
        )
        coefficients, *_ = np.linalg.lstsq(
            matrix, train[target].to_numpy(float), rcond=None
        )

        def predict(test: pd.DataFrame) -> np.ndarray:
            test_matrix = np.column_stack(
                [test[f].to_numpy(float) for f in features] + [np.ones(len(test))]
            )
            return test_matrix @ coefficients

        return predict

    def extract_params(data: pd.DataFrame) -> dict[str, float]:
        matrix = np.column_stack(
            [data[f].to_numpy(float) for f in features] + [np.ones(len(data))]
        )
        coefficients, *_ = np.linalg.lstsq(
            matrix, data[target].to_numpy(float), rcond=None
        )
        params = {f: float(c) for f, c in zip(features, coefficients[:-1])}
        params["intercept"] = float(coefficients[-1])
        return params

    return CandidateSpec(
        name=name, fit_fn=fit_fn, extract_params=extract_params,
        cohort_scope=cohort_scope,
    )
