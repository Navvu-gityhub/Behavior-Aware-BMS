"""Tests for the cohort-coverage sweep.

Two of these guard against artifacts that actually occurred while the sweep
was being built, and either would have produced a confident wrong answer:

`test_single_cell_cohorts_are_excluded` — with one cell in a cohort, holding
out the cohort *is* holding out the cell, so LOCO and LOBO are the same split
and the gap is 0.0000 by construction. The first draft reported exactly that
at full coverage and it looked like a finding.

`test_partial_correlation_removes_a_pure_size_effect` — on CALCE the raw
correlation between cohort count and the gap was +0.257 (p = 0.027) with a
convincingly monotone median. Adjusted for cell count it was +0.024 (p = 0.84).
Without this control the sweep would have confirmed a hypothesis that is false.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bms.benchmarks import get, load_all
from src.bms.benchmarks.coverage import (
    MIN_COHORTS_IN_SUBSET,
    gap_vs_coverage_correlation,
    partial_gap_correlation,
    summarise_sweep,
    sweep_cohort_coverage,
)

FEATURES = ("f_signal", "cycle")
TARGET = "soh"


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()


def sweep_frame(
    n_cohorts: int = 5,
    cells_per_cohort: int = 3,
    n_cycles: int = 40,
    seed: int = 0,
) -> pd.DataFrame:
    """A fleet with a real cohort structure and a learnable fade signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for cohort in range(n_cohorts):
        # Each cohort fades at its own rate, so a held-out cohort is genuinely
        # out of distribution rather than a relabelling.
        rate = 0.002 * (1 + cohort)
        for cell in range(cells_per_cohort):
            for cycle in range(1, n_cycles + 1):
                rows.append({
                    "cell_id": f"P{cohort}C{cell}",
                    "cohort": f"P{cohort}",
                    "cycle": cycle,
                    "f_signal": rate * cycle + rng.normal(0, 0.01),
                    TARGET: 1.0 - rate * cycle + rng.normal(0, 0.005),
                })
    return pd.DataFrame(rows)


class TestSweepMechanics:
    def test_produces_one_row_per_level_repeat_method(self):
        table = sweep_cohort_coverage(
            sweep_frame(), [get("age_linear")], target=TARGET,
            features=FEATURES, dataset="synthetic", repeats=2,
        )
        assert not table.empty
        assert set(table.columns) >= {
            "dataset", "n_cohorts", "repeat", "method",
            "n_cells", "n_rows", "lobo_r2", "loco_r2", "gap",
        }

    def test_coverage_levels_start_at_three(self):
        """Two cohorts leaves one in training; the validator skips every fold."""
        table = sweep_cohort_coverage(
            sweep_frame(n_cohorts=5), [get("age_linear")], target=TARGET,
            features=FEATURES, repeats=2,
        )
        assert table["n_cohorts"].min() >= MIN_COHORTS_IN_SUBSET

    def test_gap_is_loco_minus_lobo(self):
        table = sweep_cohort_coverage(
            sweep_frame(), [get("age_linear")], target=TARGET,
            features=FEATURES, repeats=2,
        )
        assert np.allclose(table["gap"], table["loco_r2"] - table["lobo_r2"])

    def test_full_coverage_is_measured_once(self):
        """Repeating the complete cohort set would only re-measure the cells."""
        table = sweep_cohort_coverage(
            sweep_frame(n_cohorts=4), [get("age_linear")], target=TARGET,
            features=FEATURES, repeats=5,
        )
        top = table[table["n_cohorts"] == 4]
        assert top["repeat"].nunique() == 1

    def test_too_few_cohorts_raises(self):
        with pytest.raises(ValueError, match="cohort"):
            sweep_cohort_coverage(
                sweep_frame(n_cohorts=2), [get("age_linear")],
                target=TARGET, features=FEATURES,
            )

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing column"):
            sweep_cohort_coverage(
                sweep_frame().drop(columns=["cohort"]), [get("age_linear")],
                target=TARGET, features=FEATURES,
            )


class TestDegeneracyGuard:
    def test_single_cell_cohorts_are_excluded(self):
        """LOCO collapses into LOBO when a cohort holds one cell.

        The gap is then 0.0000 by construction rather than by measurement,
        which the first draft of this sweep reported as a result.
        """
        data = sweep_frame(n_cohorts=6, cells_per_cohort=1)
        with pytest.raises(ValueError, match="cohort"):
            # Every cohort has one cell, so none survives the filter and
            # fewer than three remain.
            sweep_cohort_coverage(
                data, [get("age_linear")], target=TARGET, features=FEATURES,
            )

    def test_thin_cohorts_are_dropped_but_others_kept(self):
        thick = sweep_frame(n_cohorts=4, cells_per_cohort=3)
        thin = sweep_frame(n_cohorts=1, cells_per_cohort=1, seed=9)
        thin["cohort"] = "THIN"
        thin["cell_id"] = "THIN_C0"
        combined = pd.concat([thick, thin], ignore_index=True)

        table = sweep_cohort_coverage(
            combined, [get("age_linear")], target=TARGET,
            features=FEATURES, repeats=2,
        )
        assert table["n_cohorts"].max() == 4, "THIN should not count as coverage"

    def test_row_thinning_is_applied_and_bounded(self):
        table = sweep_cohort_coverage(
            sweep_frame(n_cohorts=4, n_cycles=500), [get("age_linear")],
            target=TARGET, features=FEATURES, repeats=1,
            max_rows_per_cell=50,
        )
        # 4 cohorts x 3 cells x 50 rows
        assert table["n_rows"].max() <= 4 * 3 * 50


class TestCorrelations:
    def test_partial_correlation_removes_a_pure_size_effect(self):
        """A gap driven only by cell count must not read as a coverage effect.

        This is the control that refuted the project's own hypothesis: on
        CALCE the raw correlation was +0.257 (p = 0.027) and vanished to
        +0.024 (p = 0.84) once cell count was regressed out.
        """
        rng = np.random.default_rng(3)
        n = 60
        cells = rng.integers(5, 40, size=n)
        table = pd.DataFrame({
            "dataset": "synthetic",
            # Cohort count rises with cell count, as it does in real data...
            "n_cohorts": (cells // 4).astype(int),
            "n_cells": cells,
            # ...but the gap depends on cell count ALONE.
            "gap": -1.0 / cells + rng.normal(0, 0.005, size=n),
        })

        raw = gap_vs_coverage_correlation(table)
        adjusted = partial_gap_correlation(table)

        assert raw["rho"] > 0.5, "raw correlation should look convincing"
        assert abs(adjusted["rho"]) < 0.3, "adjusted should collapse"
        assert adjusted["rho_gap_vs_cells"] > 0.5

    def test_genuine_coverage_effect_survives_adjustment(self):
        """The control must not erase a real relationship."""
        rng = np.random.default_rng(4)
        n = 60
        cohorts = rng.integers(3, 10, size=n)
        table = pd.DataFrame({
            "dataset": "synthetic",
            "n_cohorts": cohorts,
            # Cell count deliberately independent of cohort count here.
            "n_cells": rng.integers(10, 30, size=n),
            "gap": -1.0 / cohorts + rng.normal(0, 0.005, size=n),
        })
        assert partial_gap_correlation(table)["rho"] > 0.5

    def test_correlations_need_three_coverage_levels(self):
        table = pd.DataFrame({
            "dataset": "s", "n_cohorts": [3, 3], "n_cells": [5, 6],
            "gap": [-0.1, -0.2],
        })
        assert np.isnan(gap_vs_coverage_correlation(table)["rho"])
        assert np.isnan(partial_gap_correlation(table)["rho"])

    def test_summarise_is_empty_for_empty_input(self):
        assert summarise_sweep(pd.DataFrame()).empty
