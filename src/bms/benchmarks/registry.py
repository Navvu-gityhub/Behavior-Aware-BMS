"""The benchmark registry: published methods, declared once, run through one gate.

WHY THIS EXISTS
---------------
Until now this project validated its own heuristic against a constant and
against an age baseline, and found it wanting. That establishes the heuristic
is not skilful. It does not establish anything about *the field*, and a
reviewer's first question is the obvious one: how do published methods fare on
the same data, under the same split?

Without that comparison there are two indistinguishable explanations for the
null result:

1. Behaviour-based fade prediction is genuinely hard under protocol shift.
2. This particular implementation is weak.

Only (1) is a contribution. Separating them requires running methods that are
already published and already believed to work, through the *same* harness, and
reporting where they land. That is this module's job.

WHAT A "METHOD" IS HERE
-----------------------
`src.bms.adaptive.validation.Validator` accepts any `FitFn` — a callable that
takes a training frame and returns a predictor for a test frame. That is the
only contract. So a benchmark method is a `FitFn` factory plus the metadata
needed to interpret its result: what it needs from the data, where it came
from, and what it is not.

This means every registered method is automatically subject to the same rules
the project's own candidates are: leave-one-cohort-out is mandatory, R-squared
is measured against a *training*-fold mean rather than a test-fold oracle, and
the age-confound baseline must be beaten. A method cannot opt into a friendlier
evaluation by being famous.

AVAILABILITY IS REPORTED, NEVER SILENTLY SKIPPED
------------------------------------------------
Several important methods in this field operate on per-cycle voltage/capacity
*curves* (Severson's Delta-Q variance, incremental capacity analysis). The
cycle-level summary frame this project retains does not carry them. The wrong
response is to quietly drop those methods from the results table, because a
table showing only what ran reads as a complete comparison when it is not.

`check_availability` therefore returns a structured verdict with a reason, and
the study script prints UNAVAILABLE rows alongside the scored ones. That
follows the same rule the rest of the project uses for missing data: absent is
reported as absent, never as a benign default (see the NaN-as-healthy fix in
`risk.stress_score.compute_risk_assessment`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from src.bms.adaptive.validation import FitFn

# A method is built from a feature list, so the same estimator can be
# benchmarked on different feature sets. `None` means "use the method's own
# declared default", which is what a published specification usually implies.
BuildFn = Callable[[Sequence[str], str], FitFn]


class Family(str):
    """Grouping for the results table. Plain strings, compared by value."""

    NAIVE = "naive"
    CLASSICAL = "classical"
    CURVE = "curve"
    PHYSICS = "physics"
    PROJECT = "project"


@dataclass(frozen=True)
class Availability:
    """Whether a method can run on a given frame, and why not if it cannot."""

    method: str
    runnable: bool
    reason: str = ""
    missing_features: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.runnable

    @property
    def status(self) -> str:
        return "RUNNABLE" if self.runnable else "UNAVAILABLE"

    def render(self) -> str:
        if self.runnable:
            return f"{self.method}: RUNNABLE"
        return f"{self.method}: UNAVAILABLE - {self.reason}"


@dataclass(frozen=True)
class BenchmarkMethod:
    """One published (or reference) method, ready to run through the gate.

    `build(features, target)` returns the `FitFn` the validator will call.
    Construction is deferred rather than eager so that a method carrying an
    expensive import or a random seed is only instantiated for the folds it
    actually runs in.

    `requires_curves` marks methods needing per-cycle voltage/capacity traces
    rather than cycle-level aggregates. It is a separate flag from
    `default_features` because the requirement is structural — no choice of
    summary columns satisfies it — and it deserves its own message in the
    results table.

    `requires_package` names third-party imports the method needs but the core
    install does not carry (`xgboost`, `torch`). Checked before any fold runs,
    so a missing optional dependency produces one UNAVAILABLE row naming the
    package rather than forty identical ImportErrors buried in fold results.
    """

    name: str
    family: str
    citation: str
    build: BuildFn
    default_features: tuple[str, ...] = ()
    requires_curves: bool = False
    requires_package: tuple[str, ...] = ()
    note: str = ""
    # Methods whose published form assumes a specific target definition.
    # Empty means the method is target-agnostic.
    assumes_target: str = ""
    # Set for methods that use no features at all. Distinct from "declares no
    # defaults": `train_mean` ignores every feature by design, and an empty
    # feature list is its correct configuration rather than a caller mistake.
    # Without this flag the two cases are indistinguishable and the guard below
    # rejects the legitimate one.
    feature_free: bool = False

    def features_for(self, features: Sequence[str] | None) -> tuple[str, ...]:
        return tuple(features) if features is not None else self.default_features

    def fit_fn(
        self,
        features: Sequence[str] | None = None,
        target: str = "capacity_loss",
    ) -> FitFn:
        chosen = self.features_for(features)
        if not chosen and not self.requires_curves and not self.feature_free:
            raise ValueError(
                f"{self.name}: no features supplied and the method declares no "
                f"defaults. Pass an explicit feature list, or set "
                f"feature_free=True if the method genuinely uses none."
            )
        return self.build(chosen, target)


def check_availability(
    method: BenchmarkMethod,
    data: pd.DataFrame,
    features: Sequence[str] | None = None,
    curve_columns: Sequence[str] = ("voltage_v", "capacity_ah_curve"),
) -> Availability:
    """Can this method run on this frame? Reports a reason when it cannot.

    Curve-level requirements are checked first because they are structural: a
    frame of cycle-level aggregates cannot be made to satisfy them by choosing
    different columns, so reporting a missing-feature list would misdescribe
    the problem.
    """
    missing_packages = tuple(
        name for name in method.requires_package
        if importlib.util.find_spec(name) is None
    )
    if missing_packages:
        return Availability(
            method=method.name,
            runnable=False,
            reason=(
                f"needs {list(missing_packages)}, which the core install does "
                f"not carry. Install with `pip install "
                f"'behavior-aware-bms[{'benchmarks'}]'` or the package directly."
            ),
        )

    if method.requires_curves and not all(c in data.columns for c in curve_columns):
        return Availability(
            method=method.name,
            runnable=False,
            reason=(
                f"needs per-cycle voltage/capacity curves (columns "
                f"{list(curve_columns)}); this frame carries cycle-level "
                f"aggregates only. Not a missing-column problem: the summary "
                f"frame cannot express the quantity this method is defined on."
            ),
        )

    chosen = method.features_for(features)
    missing = tuple(f for f in chosen if f not in data.columns)
    if missing:
        return Availability(
            method=method.name,
            runnable=False,
            reason=f"missing feature column(s) {list(missing)}",
            missing_features=missing,
        )

    degenerate = tuple(
        f for f in chosen
        if f in data.columns and float(pd.to_numeric(data[f], errors="coerce").std(ddof=0) or 0.0) == 0.0
    )
    if degenerate:
        return Availability(
            method=method.name,
            runnable=False,
            reason=(
                f"feature(s) {list(degenerate)} are constant on this frame, so no "
                f"coefficient can be fitted for them. NASA's "
                f"`fast_charge_duration` is identically zero in all 2,682 "
                f"cycle-level observations, which is the case this guards."
            ),
            missing_features=degenerate,
        )

    return Availability(method=method.name, runnable=True)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, BenchmarkMethod] = {}


def register(method: BenchmarkMethod) -> BenchmarkMethod:
    """Add a method to the global registry, refusing silent redefinition."""
    if method.name in _REGISTRY:
        raise ValueError(
            f"Benchmark method '{method.name}' is already registered. Pick a "
            f"distinct name rather than shadowing it — two methods sharing a "
            f"name in a results table is how a comparison becomes unreadable."
        )
    _REGISTRY[method.name] = method
    return method


def get(name: str) -> BenchmarkMethod:
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark method '{name}'. Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def all_methods(family: str | None = None) -> tuple[BenchmarkMethod, ...]:
    """Every registered method, optionally filtered to one family."""
    methods = sorted(_REGISTRY.values(), key=lambda m: (m.family, m.name))
    if family is not None:
        methods = [m for m in methods if m.family == family]
    return tuple(methods)


def registry_frame() -> pd.DataFrame:
    """The registry as a table, for the paper's methods section."""
    return pd.DataFrame([
        {
            "method": m.name,
            "family": m.family,
            "requires_curves": m.requires_curves,
            "requires_package": ", ".join(m.requires_package),
            "default_features": ", ".join(m.default_features),
            "assumes_target": m.assumes_target,
            "citation": m.citation,
            "note": m.note,
        }
        for m in all_methods()
    ])


def _ensure_loaded() -> None:
    """Import the modules that populate the registry.

    Kept as an explicit call rather than an import-time side effect in
    `__init__` so that importing `registry` alone (for the types) does not drag
    in scikit-learn.
    """
    from src.bms.benchmarks import (  # noqa: F401
        baselines,
        classical,
        curves,
        physics,
        sequence,
    )


def load_all() -> tuple[BenchmarkMethod, ...]:
    """Populate and return the full registry."""
    _ensure_loaded()
    return all_methods()
