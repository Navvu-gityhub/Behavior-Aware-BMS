"""Dataset registry and suitability screening.

Adding a dataset should not require touching the pipeline. More importantly,
it should not be possible to spend a week building a loader for data that
cannot answer the question being asked. This module exists mostly for the
second reason.

The lesson being encoded
------------------------
`docs/adr/0001` records that the supplied CALCE files cannot support
cross-dataset validation: all 17 workbooks are single-cycle baseline
characterisations, so there is no cycle-indexed fade target and no model error
to compute. That was discovered *after* a working loader had been written, by
inspecting the parsed output by hand.

A second attempt, at calendar aging, was closed the same way: the per-cell
capacity in those files is bit-for-bit identical to the "post-storage" column,
because both record the same measurement event.

Both are mechanical properties of the data. Both are checkable in about twenty
lines. `SuitabilityReport` runs those checks up front so the answer arrives
before the effort, not after it.

What a blocker means
--------------------
A **blocker** means the dataset cannot answer the fade question at all — not
that it would answer it poorly. There is a real difference between "this
dataset gives a weak signal" and "this dataset contains no target", and
conflating them is how a project ends up reporting a number for an experiment
it never actually ran. Blockers are not overridable by a flag.

**Warnings** are different: they describe data that is usable but whose
results will need caveats, and they do not stop a load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# Columns the adaptive pipeline needs from any dataset, whatever its origin.
REQUIRED_COLUMNS: tuple[str, ...] = ("cell_id", "cycle")

# The fade target. Without it there is nothing to calibrate against.
CAPACITY_COLUMN = "capacity_ah"

# Minimum cells in a cohort before its intercept is worth fitting.
#
# NASA's cohorts hold 3-4 cells, which is already marginal - Section 4.6 of
# the report closed the mixed-effects line precisely because 3-4 cells per
# cohort cannot identify random effects. Two is the floor below which a
# "cohort mean" is just one cell's value, so it is a warning rather than a
# blocker, with the number stated so a reader can judge it.
MIN_CELLS_PER_COHORT = 3

# Minimum distinct cycles per cell for a fade trajectory to exist at all.
# One cycle is a point; two determine a line exactly, leaving zero residual
# degrees of freedom, so the slope has no computable uncertainty and any
# R-squared or confidence interval derived from it is undefined rather than
# merely wide. Three is the floor at which a fade rate becomes an estimate
# instead of an interpolation.
MIN_CYCLES_FOR_A_SLOPE = 3

# Above the floor but below this, a fade slope is fittable but noisy. This is
# a warning threshold, not a blocker.
MIN_CYCLES_PER_CELL = 10


@dataclass(frozen=True)
class DatasetManifest:
    """What a dataset actually contains, measured rather than declared."""

    name: str
    n_cells: int
    n_cohorts: int
    n_observations: int
    min_cycles_per_cell: int
    median_cycles_per_cell: float
    max_cycles_per_cell: int
    has_capacity: bool
    capacity_varies_within_cells: bool
    cells_per_cohort: Mapping[str, int] = field(default_factory=dict)
    source_path: str = ""

    def render(self) -> str:
        lines = [
            f"{self.name}: {self.n_cells} cells, {self.n_cohorts} cohorts, "
            f"{self.n_observations:,} observations",
            f"  cycles per cell: min={self.min_cycles_per_cell} "
            f"median={self.median_cycles_per_cell:.0f} max={self.max_cycles_per_cell}",
            f"  capacity column present: {self.has_capacity}",
            f"  capacity varies within cells: {self.capacity_varies_within_cells}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class SuitabilityReport:
    """Whether a dataset can support cycle-based fade calibration."""

    dataset: str
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    manifest: DatasetManifest | None = None

    @property
    def usable(self) -> bool:
        return not self.blockers

    def __bool__(self) -> bool:
        return self.usable

    @property
    def status(self) -> str:
        if self.blockers:
            return "UNUSABLE"
        return "USABLE_WITH_CAVEATS" if self.warnings else "USABLE"

    def render(self) -> str:
        lines = [f"{self.dataset}: {self.status}"]
        for blocker in self.blockers:
            lines.append(f"  BLOCKER: {blocker}")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        if self.manifest is not None:
            lines.append("  " + self.manifest.render().replace("\n", "\n  "))
        return "\n".join(lines)


def build_manifest(
    data: pd.DataFrame,
    name: str,
    cohort_col: str = "cohort",
    source_path: str = "",
) -> DatasetManifest:
    """Measure what a loaded frame contains."""
    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(
            f"build_manifest: '{name}' is missing required columns {missing}. "
            f"Every loader must emit the unified schema."
        )

    cycles = data.groupby("cell_id")["cycle"].nunique()
    has_capacity = CAPACITY_COLUMN in data.columns

    # Does capacity actually move within a cell? This is the check that would
    # have closed the CALCE calendar-aging line in minutes: a capacity column
    # whose value is constant per cell records one measurement event, not a
    # trajectory, however many rows it occupies.
    capacity_varies = False
    if has_capacity:
        spread = data.groupby("cell_id")[CAPACITY_COLUMN].nunique(dropna=True)
        capacity_varies = bool((spread > 1).any())

    cells_per_cohort: dict[str, int] = {}
    if cohort_col in data.columns:
        cells_per_cohort = (
            data.groupby(cohort_col)["cell_id"].nunique().astype(int).to_dict()
        )

    return DatasetManifest(
        name=name,
        n_cells=int(data["cell_id"].nunique()),
        n_cohorts=len(cells_per_cohort),
        n_observations=int(len(data)),
        min_cycles_per_cell=int(cycles.min()) if len(cycles) else 0,
        median_cycles_per_cell=float(cycles.median()) if len(cycles) else 0.0,
        max_cycles_per_cell=int(cycles.max()) if len(cycles) else 0,
        has_capacity=has_capacity,
        capacity_varies_within_cells=capacity_varies,
        cells_per_cohort={str(k): int(v) for k, v in cells_per_cohort.items()},
        source_path=source_path,
    )


def assess_suitability(
    data: pd.DataFrame,
    name: str,
    cohort_col: str = "cohort",
    source_path: str = "",
) -> SuitabilityReport:
    """Decide whether a dataset can answer the fade question.

    Blockers are properties that make the question unanswerable, not merely
    hard. Each one below corresponds to a failure this project actually hit.
    """
    manifest = build_manifest(data, name, cohort_col=cohort_col, source_path=source_path)
    blockers: list[str] = []
    warnings: list[str] = []

    # BLOCKER: no fade target.
    if not manifest.has_capacity:
        blockers.append(
            f"No '{CAPACITY_COLUMN}' column. There is no fade target, so model "
            f"error cannot be computed - not poorly, but at all."
        )

    # BLOCKER: too few cycles for a fade rate to be an estimate.
    #
    # One cycle is the CALCE case recorded in ADR 0001: a baseline
    # characterisation defining no trajectory at all. Two is subtler and was
    # found by running this screen against the project's own CALCE sample -
    # two points determine a line exactly, so a slope can be computed but has
    # no residual and therefore no uncertainty. A calibration built on that
    # would report a fade rate with no way to say how wrong it might be.
    if manifest.max_cycles_per_cell < MIN_CYCLES_FOR_A_SLOPE:
        blockers.append(
            f"The best-covered cell has {manifest.max_cycles_per_cell} distinct "
            f"cycle(s), below the {MIN_CYCLES_FOR_A_SLOPE} needed for a fade rate "
            f"to be an estimate rather than an interpolation. With fewer than "
            f"three points a slope has zero residual degrees of freedom and no "
            f"computable uncertainty. One cycle is the CALCE blocker in ADR 0001."
        )
    elif manifest.median_cycles_per_cell < MIN_CYCLES_PER_CELL:
        warnings.append(
            f"Median {manifest.median_cycles_per_cell:.0f} cycles per cell, below "
            f"{MIN_CYCLES_PER_CELL}. Fade slopes fitted on this will be noisy."
        )

    # BLOCKER: capacity present but constant per cell. The calendar-aging case:
    # identical pre/post measurements recording the same event twice.
    if manifest.has_capacity and not manifest.capacity_varies_within_cells:
        blockers.append(
            f"'{CAPACITY_COLUMN}' is constant within every cell. The column "
            f"records a single measurement event, not a trajectory, so there is "
            f"no fade to predict. This is the CALCE calendar-aging blocker."
        )

    # WARNING: cohorts too thin for their intercepts to mean much.
    if cohort_col not in data.columns:
        warnings.append(
            f"No '{cohort_col}' column. Protocols cannot be held out, so "
            f"leave-one-cohort-out validation is unavailable and any model "
            f"fitted here can only be validated by the weaker LOBO split."
        )
    else:
        thin = {c: n for c, n in manifest.cells_per_cohort.items()
                if n < MIN_CELLS_PER_COHORT}
        if thin:
            warnings.append(
                f"{len(thin)} cohort(s) have fewer than {MIN_CELLS_PER_COHORT} "
                f"cells ({thin}). Their fitted intercepts rest on very few cells."
            )
        if manifest.n_cohorts < 2:
            warnings.append(
                f"Only {manifest.n_cohorts} cohort(s). Leave-one-cohort-out "
                f"needs at least 2 and is meaningful from about 3."
            )

    return SuitabilityReport(
        dataset=name,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        manifest=manifest,
    )


@runtime_checkable
class DatasetLoader(Protocol):
    """What a dataset adapter must provide.

    Deliberately minimal. A loader's only obligation is to return the unified
    schema; everything else - manifest, suitability, cohort envelopes - is
    derived from what it returns, so a new dataset cannot bypass the checks by
    declaring itself fit.
    """

    name: str

    def load(self) -> pd.DataFrame: ...


@dataclass
class CsvDatasetLoader:
    """Loader for a CSV already in the unified schema.

    This is the adapter for data that has been normalised elsewhere. Adapters
    for raw formats (Arbin workbooks, MATLAB structs) subclass or implement the
    `DatasetLoader` protocol and do their parsing in `load`.
    """

    name: str
    path: Path | str
    cohort: str | None = None
    column_map: Mapping[str, str] = field(default_factory=dict)

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"{self.name}: no such file: {path}")
        data = pd.read_csv(path)
        if self.column_map:
            data = data.rename(columns=dict(self.column_map))
        if self.cohort is not None and "cohort" not in data.columns:
            data["cohort"] = self.cohort
        missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
        if missing:
            raise ValueError(
                f"{self.name}: loaded frame is missing {missing}. Supply a "
                f"column_map to rename source columns into the unified schema."
            )
        return data


@dataclass
class CallableDatasetLoader:
    """Wraps an arbitrary function as a loader.

    Useful for adapting the existing `load_nasa_dataset` / `load_calce_capacity`
    functions without rewriting them.
    """

    name: str
    fn: Callable[[], pd.DataFrame]

    def load(self) -> pd.DataFrame:
        return self.fn()


class DatasetRegistry:
    """The set of datasets available for calibration.

    `load` refuses to return unusable data. That refusal is the point: it means
    a dataset with no fade target cannot silently reach a model-fitting step
    and produce a number that looks like a result.
    """

    def __init__(self) -> None:
        self._loaders: dict[str, DatasetLoader] = {}

    def register(self, loader: DatasetLoader) -> None:
        if not hasattr(loader, "load"):
            raise TypeError(
                f"DatasetRegistry.register: {loader!r} has no load() method"
            )
        self._loaders[loader.name] = loader

    @property
    def names(self) -> list[str]:
        return sorted(self._loaders)

    def __len__(self) -> int:
        return len(self._loaders)

    def __contains__(self, name: object) -> bool:
        return name in self._loaders

    def get(self, name: str) -> DatasetLoader:
        if name not in self._loaders:
            raise KeyError(f"Unknown dataset '{name}'. Registered: {self.names}")
        return self._loaders[name]

    def assess(self, name: str, cohort_col: str = "cohort") -> SuitabilityReport:
        """Load and screen a dataset without committing to using it."""
        loader = self.get(name)
        try:
            data = loader.load()
        except Exception as exc:
            return SuitabilityReport(
                dataset=name,
                blockers=(f"Loader raised {type(exc).__name__}: {exc}",),
            )
        return assess_suitability(
            data, name, cohort_col=cohort_col,
            source_path=str(getattr(loader, "path", "")),
        )

    def load(
        self,
        name: str,
        cohort_col: str = "cohort",
        allow_unusable: bool = False,
    ) -> tuple[pd.DataFrame, SuitabilityReport]:
        """Load a dataset, refusing unusable data by default.

        `allow_unusable` exists for inspection - looking at what is inside a
        rejected dataset is legitimate - but it is off by default and the
        report travels back with the frame either way, so a caller that opts in
        cannot claim not to have known.
        """
        data = self.get(name).load()
        report = assess_suitability(data, name, cohort_col=cohort_col)
        if not report.usable and not allow_unusable:
            raise ValueError(
                f"Dataset '{name}' cannot support fade calibration:\n"
                + report.render()
                + "\n\nPass allow_unusable=True to inspect it anyway."
            )
        return data, report

    def assess_all(self, cohort_col: str = "cohort") -> pd.DataFrame:
        """Screen every registered dataset. The triage view."""
        rows = []
        for name in self.names:
            report = self.assess(name, cohort_col=cohort_col)
            manifest = report.manifest
            rows.append({
                "dataset": name,
                "status": report.status,
                "n_cells": manifest.n_cells if manifest else 0,
                "n_cohorts": manifest.n_cohorts if manifest else 0,
                "n_observations": manifest.n_observations if manifest else 0,
                "median_cycles_per_cell": (
                    manifest.median_cycles_per_cell if manifest else 0.0
                ),
                "n_blockers": len(report.blockers),
                "n_warnings": len(report.warnings),
                "detail": " | ".join(report.blockers + report.warnings),
            })
        return pd.DataFrame(rows)
