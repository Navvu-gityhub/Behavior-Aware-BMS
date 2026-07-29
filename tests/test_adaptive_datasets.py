"""Tests for dataset registration and suitability screening.

The tests that carry weight here are the two reconstructing failures this
project actually hit:

`test_single_cycle_data_is_blocked` rebuilds the CALCE shape — many cells, one
cycle each — and asserts it is refused. That is the condition ADR 0001 records
as making cross-dataset validation impossible, discovered by hand after a
loader had already been written.

`test_constant_capacity_is_blocked` rebuilds the calendar-aging shape, where
the capacity column repeats one measurement event rather than tracing a
trajectory. Both are mechanical properties, checkable in seconds, and the
point of this module is that they are now checked before the effort is spent
rather than after.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.datasets import (
    CallableDatasetLoader,
    CsvDatasetLoader,
    DatasetRegistry,
    assess_suitability,
    build_manifest,
)

REPO = Path(__file__).resolve().parents[1]
NASA_TRAINING = REPO / "reports/metrics/continuous_model_training_data.csv"


def _cycling(n_cells: int = 6, n_cohorts: int = 2, cycles: int = 60) -> pd.DataFrame:
    """A well-formed multi-cycle dataset with real fade."""
    rng = np.random.default_rng(23)
    frames = []
    for i in range(n_cells):
        cohort = f"COHORT_{i % n_cohorts}"
        cap0 = 2.0 + rng.normal(0, 0.03)
        fade = 0.003 + 0.001 * (i % n_cohorts)
        frames.append(pd.DataFrame({
            "cell_id": f"CELL_{i:02d}",
            "cohort": cohort,
            "cycle": np.arange(1, cycles + 1),
            "capacity_ah": cap0 - fade * np.arange(1, cycles + 1),
            "avg_temp": 25 + rng.normal(0, 1, cycles),
        }))
    return pd.concat(frames, ignore_index=True)


def _single_cycle(n_cells: int = 138) -> pd.DataFrame:
    """The CALCE shape: many cells, exactly one cycle each."""
    rng = np.random.default_rng(5)
    return pd.DataFrame({
        "cell_id": [f"PL{i:02d}" for i in range(n_cells)],
        "cohort": "CALCE_PLN",
        "cycle": 1,
        "capacity_ah": rng.uniform(1.05, 1.15, n_cells),
        "avg_temp": rng.uniform(22, 26, n_cells),
    })


# ---------------------------------------------------------------------------
# Manifests measure rather than declare
# ---------------------------------------------------------------------------

def test_manifest_measures_what_is_present():
    manifest = build_manifest(_cycling(n_cells=6, n_cohorts=2, cycles=60), "synthetic")
    assert manifest.n_cells == 6
    assert manifest.n_cohorts == 2
    assert manifest.n_observations == 360
    assert manifest.median_cycles_per_cell == 60
    assert manifest.has_capacity is True
    assert manifest.capacity_varies_within_cells is True


def test_manifest_requires_the_unified_schema():
    with pytest.raises(ValueError, match="missing required columns"):
        build_manifest(pd.DataFrame({"foo": [1]}), "bad")


# ---------------------------------------------------------------------------
# The blockers: failures this project actually hit
# ---------------------------------------------------------------------------

def test_a_well_formed_cycling_dataset_is_usable():
    report = assess_suitability(_cycling(), "synthetic")
    assert report.usable is True
    assert bool(report) is True
    assert report.blockers == ()


def test_single_cycle_data_is_blocked():
    """The CALCE case: no fade trajectory exists, so there is nothing to fit."""
    report = assess_suitability(_single_cycle(), "calce_capacity")
    assert report.usable is False
    assert bool(report) is False
    assert report.status == "UNUSABLE"
    assert any("single-cycle" in b.lower() or "1 distinct cycle" in b
               for b in report.blockers)
    assert any("ADR 0001" in b for b in report.blockers)


def test_constant_capacity_is_blocked():
    """The calendar-aging case: one measurement event repeated, not a trajectory."""
    data = _cycling()
    data["capacity_ah"] = data.groupby("cell_id")["capacity_ah"].transform("first")
    report = assess_suitability(data, "calce_calendar")
    assert report.usable is False
    assert any("constant within every cell" in b for b in report.blockers)


def test_missing_capacity_is_blocked():
    report = assess_suitability(_cycling().drop(columns=["capacity_ah"]), "no_target")
    assert report.usable is False
    assert any("no fade target" in b.lower() for b in report.blockers)


def test_blockers_are_not_overridable_by_a_warning_path():
    """A blocker must not be downgraded into a caveat."""
    report = assess_suitability(_single_cycle(), "calce")
    assert report.blockers
    assert report.status == "UNUSABLE"


# ---------------------------------------------------------------------------
# Warnings: usable, but the results need caveats
# ---------------------------------------------------------------------------

def test_thin_cohorts_warn_without_blocking():
    data = _cycling(n_cells=4, n_cohorts=4)  # one cell per cohort
    report = assess_suitability(data, "thin")
    assert report.usable is True
    assert report.status == "USABLE_WITH_CAVEATS"
    assert any("fewer than 3" in w for w in report.warnings)


def test_absent_cohort_labels_warn_that_loco_is_unavailable():
    report = assess_suitability(_cycling().drop(columns=["cohort"]), "unlabelled")
    assert report.usable is True
    assert any("leave-one-cohort-out validation is unavailable" in w
               for w in report.warnings)


def test_short_trajectories_warn():
    report = assess_suitability(_cycling(cycles=5), "short")
    assert report.usable is True
    assert any("noisy" in w for w in report.warnings)


def test_two_cycle_data_is_blocked_as_a_degenerate_slope():
    """Two points determine a line exactly and leave no residual.

    Found by running this screen against the project's own CALCE sample,
    which has two cycles and was passing as merely 'noisy'. A slope with zero
    residual degrees of freedom has no computable uncertainty, so a fade rate
    derived from it cannot be qualified.
    """
    report = assess_suitability(_cycling(cycles=2), "two_point")
    assert report.usable is False
    assert any("zero residual degrees of freedom" in b for b in report.blockers)


def test_three_cycles_is_the_floor_not_a_blocker():
    report = assess_suitability(_cycling(cycles=3), "three_point")
    assert report.usable is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_registers_and_lists():
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("synthetic", _cycling))
    assert registry.names == ["synthetic"]
    assert "synthetic" in registry
    assert len(registry) == 1


def test_registry_refuses_unusable_data_by_default():
    """A dataset with no target must not reach a model-fitting step."""
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("calce", _single_cycle))
    with pytest.raises(ValueError, match="cannot support fade calibration"):
        registry.load("calce")


def test_unusable_data_can_be_inspected_deliberately():
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("calce", _single_cycle))
    data, report = registry.load("calce", allow_unusable=True)
    assert len(data) == 138
    assert report.usable is False  # the report travels back regardless


def test_usable_data_loads_with_its_report():
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("synthetic", _cycling))
    data, report = registry.load("synthetic")
    assert report.usable is True
    assert len(data) == 360


def test_a_loader_that_raises_becomes_a_blocker_not_a_crash():
    def broken() -> pd.DataFrame:
        raise RuntimeError("archive corrupt")

    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("broken", broken))
    report = registry.assess("broken")
    assert report.usable is False
    assert any("archive corrupt" in b for b in report.blockers)


def test_unknown_dataset_raises_with_the_list_of_known_ones():
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("synthetic", _cycling))
    with pytest.raises(KeyError, match="Registered"):
        registry.get("stanford")


def test_assess_all_gives_a_triage_table():
    registry = DatasetRegistry()
    registry.register(CallableDatasetLoader("good", _cycling))
    registry.register(CallableDatasetLoader("calce", _single_cycle))

    table = registry.assess_all().set_index("dataset")
    assert table.loc["good", "status"] == "USABLE"
    assert table.loc["calce", "status"] == "UNUSABLE"
    assert table.loc["calce", "n_blockers"] >= 1


# ---------------------------------------------------------------------------
# CSV adapter
# ---------------------------------------------------------------------------

def test_csv_loader_renames_source_columns(tmp_path):
    raw = _cycling().rename(columns={"cell_id": "Battery", "cycle": "Cycle_Index"})
    path = tmp_path / "raw.csv"
    raw.to_csv(path, index=False)

    loader = CsvDatasetLoader(
        name="renamed", path=path,
        column_map={"Battery": "cell_id", "Cycle_Index": "cycle"},
    )
    assert set(("cell_id", "cycle")) <= set(loader.load().columns)


def test_csv_loader_can_stamp_a_cohort(tmp_path):
    path = tmp_path / "raw.csv"
    _cycling().drop(columns=["cohort"]).to_csv(path, index=False)
    loader = CsvDatasetLoader(name="stamped", path=path, cohort="STANFORD_FASTCHARGE")
    assert (loader.load()["cohort"] == "STANFORD_FASTCHARGE").all()


def test_csv_loader_reports_an_unmappable_schema(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(path, index=False)
    loader = CsvDatasetLoader(name="wrong", path=path)
    with pytest.raises(ValueError, match="column_map"):
        loader.load()


def test_csv_loader_missing_file_is_explicit(tmp_path):
    loader = CsvDatasetLoader(name="absent", path=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        loader.load()


# ---------------------------------------------------------------------------
# Against the real NASA frame
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not NASA_TRAINING.exists(), reason="NASA training frame not present")
def test_the_real_nasa_frame_screens_as_usable():
    data = pd.read_csv(NASA_TRAINING)
    report = assess_suitability(data, "nasa")
    assert report.usable is True, report.render()
    assert report.manifest.n_cohorts == 9
    assert report.manifest.n_cells == 34
