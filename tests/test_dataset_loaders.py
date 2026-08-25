"""Tests for the concrete dataset adapters and cohort derivation.

The cohort tests carry most of the weight. Leave-one-cohort-out is this
project's binding validation constraint, and a cohort column that looks
plausible but does not correspond to real protocol groupings would make every
LOCO number meaningless while leaving it looking fine.

Two degenerate outcomes matter and are asserted on explicitly: one cohort
(LOCO unavailable, fails loudly) and one cohort per cell (LOCO silently
collapses into LOBO, which is the dangerous one).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_calce_fixture import make_cell  # noqa: E402
from make_oxford_fixture import make_oxford_mat  # noqa: E402

from src.bms.adaptive.loaders import (  # noqa: E402
    CalceCyclingLoader,
    NasaFrameLoader,
    OxfordLoader,
    default_registry,
    derive_cohorts_from_protocol,
)


def protocol_summary(specs: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Build a per-cell-cycle summary with chosen protocol descriptors.

    `specs` is (cell_id, discharge current in A, cutoff voltage in V).
    """
    rows = []
    for cell_id, current, cutoff in specs:
        for cycle in range(1, 11):
            rows.append({
                "cell_id": cell_id,
                "cycle": cycle,
                "min_current_a": current,
                "min_voltage_v": cutoff,
                "capacity_ah": 1.1 - 0.002 * cycle,
            })
    return pd.DataFrame(rows)


class TestCohortDerivation:
    def test_groups_cells_sharing_a_protocol(self):
        summary = protocol_summary([
            ("A", -1.1, 2.7), ("B", -1.1, 2.7),   # same schedule
            ("C", -2.2, 2.7),                      # twice the rate
            ("D", -1.1, 3.0),                      # different cutoff
        ])
        labels, warnings = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        summary["cohort"] = labels

        by_cell = summary.groupby("cell_id")["cohort"].first()
        assert by_cell["A"] == by_cell["B"]
        assert by_cell["C"] != by_cell["A"]
        assert by_cell["D"] != by_cell["A"]
        assert summary["cohort"].nunique() == 3
        assert warnings == ()

    def test_labels_are_marked_as_derived(self):
        """They must never be mistaken for CALCE's documented Type-1..6 groups."""
        summary = protocol_summary([("A", -1.1, 2.7), ("B", -2.2, 2.7)])
        labels, _ = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert all(str(v).startswith("DERIVED_") for v in labels.unique())

    def test_label_encodes_the_measured_descriptors(self):
        summary = protocol_summary([("A", -1.1, 2.7)])
        labels, _ = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert labels.iloc[0] == "DERIVED_1C_2.7V"

    def test_single_cohort_is_warned_about(self):
        summary = protocol_summary([("A", -1.1, 2.7), ("B", -1.1, 2.7)])
        _, warnings = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert any("one derived protocol label" in w for w in warnings)

    def test_one_cohort_per_cell_is_warned_about(self):
        """The dangerous case: LOCO silently becomes LOBO."""
        summary = protocol_summary([
            ("A", -1.1, 2.7), ("B", -2.2, 2.8), ("C", -3.3, 2.9),
        ])
        _, warnings = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert any("identical to leave-one-cell-out" in w for w in warnings)

    def test_noise_does_not_split_one_protocol(self):
        """Coarse rounding is deliberate; fine resolution invents cohorts."""
        summary = protocol_summary([
            ("A", -1.100, 2.700), ("B", -1.104, 2.703), ("C", -1.097, 2.698),
        ])
        labels, _ = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert labels.nunique() == 1

    def test_missing_descriptors_fall_back_and_warn(self):
        summary = protocol_summary([("A", -1.1, 2.7)]).drop(
            columns=["min_current_a", "min_voltage_v"]
        )
        labels, warnings = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert labels.unique().tolist() == ["DERIVED_UNKNOWN"]
        assert any("no protocol descriptor" in w for w in warnings)

    def test_uses_median_so_one_odd_cycle_cannot_reassign_a_cell(self):
        summary = protocol_summary([("A", -1.1, 2.7)])
        summary.loc[summary.index[0], "min_current_a"] = -5.5  # one outlier cycle
        labels, _ = derive_cohorts_from_protocol(summary, nominal_capacity_ah=1.1)
        assert labels.iloc[0] == "DERIVED_1C_2.7V"


@pytest.fixture(scope="module")
def calce_dir(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("calce")
    for cell_id in ("CS2_33", "CS2_34", "CS2_35"):
        make_cell(base / cell_id, cell_id=cell_id, n_files=2, cycles_per_file=6)
    return base


@pytest.fixture(scope="module")
def oxford_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("oxford") / "oxford.mat"
    make_oxford_mat(path, n_cells=3, n_characterisations=5)
    return path


class TestCalceLoader:
    def test_emits_the_columns_the_registry_requires(self, calce_dir):
        summary = CalceCyclingLoader(base_dir=calce_dir).load()
        for column in ("cell_id", "cycle", "capacity_ah", "capacity_loss",
                       "soh", "cohort"):
            assert column in summary.columns

    def test_cycle_numbers_are_reconciled_across_files(self, calce_dir):
        """Each CALCE file restarts Cycle_Index at 1; the loader must offset."""
        summary = CalceCyclingLoader(base_dir=calce_dir).load()
        per_cell = summary.groupby("cell_id")["cycle"].nunique()
        assert (per_cell == 12).all()  # 2 files x 6 cycles

    def test_capacity_declines(self, calce_dir):
        summary = CalceCyclingLoader(base_dir=calce_dir).load()
        for _, group in summary.groupby("cell_id"):
            values = group.sort_values("cycle")["capacity_ah"].to_numpy()
            assert values[-1] < values[0]

    def test_explicit_cohort_map_overrides_derivation(self, calce_dir):
        loader = CalceCyclingLoader(
            base_dir=calce_dir,
            cohort_map={"CS2_33": "TYPE_1", "CS2_34": "TYPE_1", "CS2_35": "TYPE_2"},
        )
        summary = loader.load()
        assert set(summary["cohort"].unique()) == {"TYPE_1", "TYPE_2"}
        assert not any(str(c).startswith("DERIVED_") for c in summary["cohort"])

    def test_unmapped_cell_raises_rather_than_bucketing(self, calce_dir):
        """An unmapped cell must not join a catch-all group of unrelated cells."""
        loader = CalceCyclingLoader(
            base_dir=calce_dir, cohort_map={"CS2_33": "TYPE_1"},
        )
        with pytest.raises(ValueError, match="cohort_map has no entry"):
            loader.load()

    def test_derived_cohorts_record_their_warnings(self, calce_dir):
        loader = CalceCyclingLoader(base_dir=calce_dir)
        loader.load()
        assert loader.load_warnings  # fixture cells share one schedule

    def test_absent_directory_raises_with_where_to_get_the_data(self, tmp_path):
        loader = CalceCyclingLoader(base_dir=tmp_path / "nope")
        with pytest.raises(FileNotFoundError, match="dataset_sources.yaml"):
            loader.load()


class TestOxfordLoader:
    def test_emits_the_columns_the_registry_requires(self, oxford_file):
        summary = OxfordLoader(path=oxford_file).load()
        for column in ("cell_id", "cycle", "capacity_ah", "capacity_loss",
                       "soh", "cohort"):
            assert column in summary.columns

    def test_has_exactly_one_cohort(self, oxford_file):
        summary = OxfordLoader(path=oxford_file).load()
        assert summary["cohort"].nunique() == 1

    def test_records_the_capacity_unit_decision(self, oxford_file):
        loader = OxfordLoader(path=oxford_file)
        loader.load()
        assert any("capacity read as" in w for w in loader.load_warnings)

    def test_absent_file_raises_and_says_the_transfer_is_marginal(self, tmp_path):
        loader = OxfordLoader(path=tmp_path / "nope.mat")
        with pytest.raises(FileNotFoundError, match="MARGINAL"):
            loader.load()


class TestDefaultRegistry:
    def test_screens_all_three_datasets(self, calce_dir, oxford_file):
        registry = default_registry(calce_dir=calce_dir, oxford_path=oxford_file)
        assert set(registry.names) == {"nasa", "calce", "oxford"}

        table = registry.assess_all()
        assert set(table["dataset"]) == {"nasa", "calce", "oxford"}
        assert (table["n_blockers"] == 0).all()

    def test_single_cohort_datasets_are_flagged(self, calce_dir, oxford_file):
        registry = default_registry(calce_dir=calce_dir, oxford_path=oxford_file)
        table = registry.assess_all().set_index("dataset")
        assert "Leave-one-cohort-out needs at least 2" in table.loc["oxford", "detail"]

    def test_absent_data_becomes_a_blocker_not_a_disappearance(self, tmp_path):
        """A dataset that isn't downloaded must still appear, with a reason."""
        registry = default_registry(
            calce_dir=tmp_path / "no_calce",
            oxford_path=tmp_path / "no_oxford.mat",
        )
        table = registry.assess_all().set_index("dataset")

        assert "calce" in table.index and "oxford" in table.index
        assert table.loc["calce", "status"] == "UNUSABLE"
        assert "FileNotFoundError" in table.loc["calce", "detail"]

    def test_nasa_frame_loads_from_the_tracked_csv(self):
        summary = NasaFrameLoader().load()
        assert {"cell_id", "cycle", "capacity_ah", "cohort"} <= set(summary.columns)
        assert summary["cohort"].nunique() == 9

    def test_registry_load_refuses_unusable_data(self, tmp_path):
        registry = default_registry(calce_dir=tmp_path / "no_calce")
        with pytest.raises(FileNotFoundError):
            registry.load("calce")
