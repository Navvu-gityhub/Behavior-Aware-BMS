"""How much of the leave-one-cohort-out penalty is the method, and how much is coverage?

THE OBSERVATION THIS EXISTS TO TURN INTO A MEASUREMENT
-------------------------------------------------------
Running the same twelve methods through the same harness on three frames
produced three mutually inconsistent rankings:

    frame              cohorts  families  rho(LOBO,LOCO)  worst delta
    CALCE CS2                5         1        --             -5.574
    NASA                     9         1        +0.084         -1.354
    CALCE CS2 + CX2          8         2        +0.818         -0.577

Five separate claims were made from those runs and all five were withdrawn
(ADR 0008, ADR 0009). Every one died the same way: a ranking read off a single
frame did not survive the next frame.

The pattern in the table suggests why. Leave-one-cohort-out asks a model to
predict an unseen protocol — but when eight cohorts remain in training, the
held-out one is barely unseen, and when four remain it genuinely is. If that
is right, the LOBO-to-LOCO gap is **not a property of a method at all**. It is
a property of how much of the experimental design space survives in training,
and a LOCO number reported without its cohort coverage is uninterpretable.

That is a testable claim rather than a story, and this module tests it: hold
everything else fixed and vary only the number of cohorts available.

THE CONFOUND THAT WOULD RUIN IT, AND WHY IT CANNOT SIMPLY BE FIXED AWAY
------------------------------------------------------------------------
More cohorts means more cells. A naive sweep would show the gap shrinking with
cohort count while actually measuring **training-set size** — a different and
far less interesting claim.

The obvious control is to fix the cell count across coverage levels, and
`cell_budget` does that. But it cannot be used at the range needed here, for a
reason worth stating because it is not obvious:

NASA has 31 cells across 9 cohorts, about 3.4 per cohort. Fixing the budget at
9 cells means that at k=9 every cohort contributes exactly **one** cell — and
then holding out a cohort *is* holding out a cell, so LOCO and LOBO become the
same split and the gap is 0.0000 by construction. The first run of this sweep
produced exactly that, and it is an artifact, not a finding. Avoiding it needs
at least two cells per cohort, so 18 cells at k=9 — but at k=3 only three
cohorts are available, holding roughly 10 cells between them. The two
requirements are incompatible on this data.

So the sweep takes every cell in the chosen cohorts, records `n_cells` on each
row, and **separates the two effects statistically rather than by design**:
`partial_gap_correlation` reports the cohort-count relationship after
regressing out cell count. Both the raw and the adjusted figure are reported,
because a reader is entitled to see how much of the raw relationship survives
the control.

`min_cells_per_cohort` additionally rejects any subframe where a cohort holds
a single cell, which is the degenerate case above.

WHY SUBSETS ARE REPEATED
------------------------
Which cohorts land in a subset matters: NASA's `COLD4C_*` protocols behave
differently from its room-temperature ones, so a single draw at k=3 measures
that draw rather than k=3. Each coverage level is therefore sampled several
times with different cohort subsets and the results are reported per repeat,
so the spread across draws is visible next to the trend across k.

WHY THE SWEEP RUNS WITHIN A DATASET, NEVER POOLED
--------------------------------------------------
NASA carries temperature, SOC and the behavioural flags; CALCE records none of
them and offers electrical channels instead. Pooling the two would mean
imputing most of both feature sets, and the sweep would measure that
imputation. Each dataset is swept separately and the two curves are compared —
which is also a stronger test, since agreement across two unrelated feature
sets is evidence the relationship is about coverage rather than about either
dataset's columns.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import Validator
from src.bms.benchmarks.registry import BenchmarkMethod

# Leave-one-cohort-out needs at least one cohort left in training after one is
# held out, and `Validator` additionally requires the training fold to span two
# (`MIN_TRAIN_COHORTS`). Three is therefore the smallest subset for which LOCO
# produces any completed fold at all.
MIN_COHORTS_IN_SUBSET = 3

CELL_COLUMN = "cell_id"
COHORT_COLUMN = "cohort"


@dataclass(frozen=True)
class SweepPoint:
    """One method, on one cohort subset, at one coverage level."""

    dataset: str
    n_cohorts: int
    repeat: int
    method: str
    n_cells: int
    n_rows: int
    lobo_r2: float
    loco_r2: float
    lobo_mae: float
    loco_mae: float

    @property
    def gap(self) -> float:
        """LOCO minus LOBO. Negative means skill was lost to protocol shift."""
        if not (np.isfinite(self.lobo_r2) and np.isfinite(self.loco_r2)):
            return float("nan")
        return self.loco_r2 - self.lobo_r2


def _sample_cells(
    data: pd.DataFrame,
    cohorts: Sequence[str],
    budget: int,
    rng: np.random.Generator,
) -> list[str]:
    """Draw `budget` cells from `cohorts`, one per cohort before any second.

    Round-robin rather than proportional. A proportional draw would give a
    four-cell cohort four times the weight of a one-cell cohort, so a subset
    could nominally span seven cohorts while being dominated by two of them —
    which is precisely the coverage the sweep is trying to vary.
    """
    by_cohort: dict[str, list[str]] = {}
    for cohort in cohorts:
        cells = sorted(
            data.loc[data[COHORT_COLUMN] == cohort, CELL_COLUMN].unique().tolist()
        )
        rng.shuffle(cells)
        by_cohort[cohort] = cells

    chosen: list[str] = []
    while len(chosen) < budget:
        added = False
        for cohort in cohorts:
            if not by_cohort[cohort]:
                continue
            chosen.append(by_cohort[cohort].pop())
            added = True
            if len(chosen) >= budget:
                break
        if not added:
            break  # every cohort exhausted
    return chosen


def sweep_cohort_coverage(
    data: pd.DataFrame,
    methods: Sequence[BenchmarkMethod],
    target: str,
    features: Sequence[str],
    dataset: str = "dataset",
    cell_budget: int | None = None,
    min_cells_per_cohort: int = 2,
    max_rows_per_cell: int | None = 200,
    repeats: int = 5,
    seed: int = 20260821,
    cell_col: str = CELL_COLUMN,
    cohort_col: str = COHORT_COLUMN,
    confound_col: str = "cycle",
    verbose: bool = False,
) -> pd.DataFrame:
    """Vary the number of cohorts available and record what varies with it.

    `cell_budget=None` takes every cell in the chosen cohorts; an integer caps
    it. See the module docstring for why capping is not usable at this range
    and why cell count is instead controlled statistically.

    Cohorts with fewer than `min_cells_per_cohort` cells are excluded, because
    a one-cell cohort makes leave-one-cohort-out identical to
    leave-one-cell-out and the gap becomes zero by construction.

    Returns one row per (coverage level, repeat, method).
    """
    for column in (target, cell_col, cohort_col):
        if column not in data.columns:
            raise ValueError(f"sweep_cohort_coverage: missing column '{column}'")

    frame = data.dropna(subset=[target]).copy()

    # Thin each cell's cycle series to bound compute.
    #
    # The sweep refits across dozens of subframes, and CALCE's partial-cycling
    # cells carry 6,000+ cycles each — a full-coverage subframe reaches 43,000
    # rows and the sweep stops being runnable.
    #
    # Thinning is uniform in cycle index and preserves the endpoints, so a
    # cell's fade trajectory keeps its shape and its range; degradation is
    # smooth in cycle count, so the intermediate points are near-redundant.
    # This is a compute bound, not a modelling choice: it applies identically
    # at every coverage level, so it cannot bias the trend the sweep measures.
    if max_rows_per_cell is not None:
        def _thin(group: pd.DataFrame) -> pd.DataFrame:
            if len(group) <= max_rows_per_cell:
                return group
            ordered = group.sort_values(confound_col)
            keep = np.linspace(0, len(ordered) - 1, max_rows_per_cell).astype(int)
            return ordered.iloc[np.unique(keep)]

        frame = (
            frame.groupby(cell_col, group_keys=False)[list(frame.columns)]
            .apply(_thin)
            .reset_index(drop=True)
        )

    # Drop cohorts too thin to distinguish LOCO from LOBO. With one cell in a
    # cohort, holding out that cohort holds out that cell, and the gap is zero
    # by construction rather than by measurement.
    sizes = frame.groupby(cohort_col)[cell_col].nunique()
    usable_cohorts = sizes[sizes >= min_cells_per_cohort].index.tolist()
    frame = frame[frame[cohort_col].isin(usable_cohorts)]

    all_cohorts = sorted(frame[cohort_col].dropna().unique().tolist())
    if len(all_cohorts) < MIN_COHORTS_IN_SUBSET:
        raise ValueError(
            f"sweep_cohort_coverage: {len(all_cohorts)} cohort(s); at least "
            f"{MIN_COHORTS_IN_SUBSET} are needed for leave-one-cohort-out to "
            f"complete a fold."
        )

    rng = np.random.default_rng(seed)
    rows: list[SweepPoint] = []

    for k in range(MIN_COHORTS_IN_SUBSET, len(all_cohorts) + 1):
        for repeat in range(repeats):
            # At full coverage every draw is the same set, so repeating it
            # would only re-measure the cell sample.
            if k == len(all_cohorts) and repeat > 0:
                break

            subset = sorted(
                rng.choice(all_cohorts, size=k, replace=False).tolist()
            )
            if cell_budget is None:
                sub = frame[frame[cohort_col].isin(subset)]
            else:
                cells = _sample_cells(frame, subset, cell_budget, rng)
                sub = frame[frame[cell_col].isin(cells)]

            if sub[cohort_col].nunique() < MIN_COHORTS_IN_SUBSET:
                continue
            if sub.groupby(cohort_col)[cell_col].nunique().min() < min_cells_per_cohort:
                continue

            validator = Validator(sub, target=target, cohort_col=cohort_col)
            for method in methods:
                try:
                    fit_fn = method.fit_fn(features=features, target=target)
                    lobo = validator.cross_validate(
                        fit_fn, group_col=cell_col, split="LOBO"
                    )
                    loco = validator.cross_validate(
                        fit_fn, group_col=cohort_col, split="LOCO"
                    )
                except Exception:
                    # A method that cannot fit this subframe contributes no
                    # point rather than a zero, which would drag the curve.
                    continue

                if not lobo.completed or not loco.completed:
                    continue

                rows.append(SweepPoint(
                    dataset=dataset,
                    n_cohorts=int(sub[cohort_col].nunique()),
                    repeat=repeat,
                    method=method.name,
                    n_cells=int(sub[cell_col].nunique()),
                    n_rows=int(len(sub)),
                    lobo_r2=lobo.median_r2,
                    loco_r2=loco.median_r2,
                    lobo_mae=lobo.median_mae,
                    loco_mae=loco.median_mae,
                ))

            if verbose:
                print(
                    f"    k={k} repeat={repeat}: {sub[cohort_col].nunique()} cohorts, "
                    f"{sub[cell_col].nunique()} cells, {len(sub)} rows",
                    flush=True,
                )

    table = pd.DataFrame([vars(r) for r in rows])
    if not table.empty:
        table["gap"] = table["loco_r2"] - table["lobo_r2"]
    return table


def summarise_sweep(table: pd.DataFrame) -> pd.DataFrame:
    """Median gap per coverage level, pooled across repeats and methods."""
    if table.empty:
        return pd.DataFrame()
    return (
        table.groupby(["dataset", "n_cohorts"])
        .agg(
            n_points=("gap", "size"),
            n_cells=("n_cells", "median"),
            n_rows=("n_rows", "median"),
            median_lobo_r2=("lobo_r2", "median"),
            median_loco_r2=("loco_r2", "median"),
            median_gap=("gap", "median"),
            worst_gap=("gap", "min"),
        )
        .reset_index()
    )


def gap_vs_coverage_correlation(table: pd.DataFrame) -> dict[str, float]:
    """Spearman correlation between cohort count and the LOBO-to-LOCO gap.

    A positive value means the gap closes as coverage grows — the module's
    hypothesis. Reported with its p-value and n, because with a handful of
    coverage levels this is easy to over-read.
    """
    if table.empty or table["n_cohorts"].nunique() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": 0.0}

    from scipy.stats import spearmanr

    usable = table.dropna(subset=["gap"])
    rho, p = spearmanr(usable["n_cohorts"], usable["gap"])
    return {"rho": float(rho), "p": float(p), "n": float(len(usable))}


def partial_gap_correlation(table: pd.DataFrame) -> dict[str, float]:
    """Cohort-count vs gap, with cell count regressed out of both.

    This is the number that decides whether the hypothesis survives. Cohort
    count and cell count rise together, so the raw correlation cannot tell
    "more protocols in training" from "more data in training". A Spearman
    partial correlation removes the second by ranking, fitting each variable
    against cell count, and correlating what is left over.

    Reported alongside the raw figure rather than instead of it: the drop
    between them is how much of the apparent relationship was sample size, and
    a reader should see it.
    """
    if table.empty or table["n_cohorts"].nunique() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": 0.0,
                "rho_gap_vs_cells": float("nan")}

    import numpy as np
    from scipy.stats import rankdata, spearmanr

    usable = table.dropna(subset=["gap", "n_cells"])
    if len(usable) < 6 or usable["n_cells"].nunique() < 2:
        return {"rho": float("nan"), "p": float("nan"), "n": float(len(usable)),
                "rho_gap_vs_cells": float("nan")}

    cohorts = rankdata(usable["n_cohorts"].to_numpy(dtype=float))
    gap = rankdata(usable["gap"].to_numpy(dtype=float))
    cells = rankdata(usable["n_cells"].to_numpy(dtype=float))

    def residual(y: np.ndarray) -> np.ndarray:
        design = np.column_stack([cells, np.ones(len(cells))])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        return y - design @ beta

    rho, p = spearmanr(residual(cohorts), residual(gap))
    gap_cells, _ = spearmanr(cells, gap)
    return {
        "rho": float(rho),
        "p": float(p),
        "n": float(len(usable)),
        "rho_gap_vs_cells": float(gap_cells),
    }
