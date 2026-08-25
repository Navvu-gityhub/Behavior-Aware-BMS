"""Tests for the Oxford Battery Degradation Dataset loader.

Every test runs against a fixture that reproduces the published nested-struct
layout (`tests/fixtures/make_oxford_fixture.py`), not against the real
archive, which is ~1 GB and not in this repository. That limit is stated in
the loader's own docstring and is worth restating here: these tests prove the
loader handles the format *as documented*. They cannot prove the real file
matches its documentation.

The load-bearing test is `test_recovers_the_fixtures_fade_exactly` — the
fixture is generated with a known capacity and a known fade per
characterisation, so the loader must return those numbers back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_oxford_fixture import make_oxford_mat  # noqa: E402

from src.bms.io.load_oxford import (  # noqa: E402
    CAPACITY_MEASUREMENT,
    OXFORD_COHORT,
    OXFORD_NOMINAL_CAPACITY_AH,
    _detect_capacity_scale,
    load_oxford_mat,
    oxford_capacity_loss,
    summarize_oxford_cycles,
)

INITIAL_MAH = 740.0
FADE_MAH = 12.0
N_CHARACTERISATIONS = 5
STRIDE = 100


@pytest.fixture(scope="module")
def oxford_file(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("oxford") / "oxford.mat"
    make_oxford_mat(
        path,
        n_cells=3,
        n_characterisations=N_CHARACTERISATIONS,
        cycle_stride=STRIDE,
        initial_capacity_mah=INITIAL_MAH,
        fade_per_characterisation_mah=FADE_MAH,
    )
    return path


@pytest.fixture(scope="module")
def loaded(oxford_file):
    return load_oxford_mat(oxford_file)


class TestCapacityScaleDetection:
    def test_detects_milliamp_hours(self):
        scale, units = _detect_capacity_scale(np.array([0.0, 740.0, 700.0]))
        assert scale == 0.001
        assert units == "mAh"

    def test_detects_amp_hours(self):
        scale, units = _detect_capacity_scale(np.array([0.0, 0.74, 0.70]))
        assert scale == 1.0
        assert units == "Ah"

    def test_reports_when_neither_interpretation_fits(self):
        """A silent factor-of-1000 error would be invisible in every ratio."""
        scale, units = _detect_capacity_scale(np.array([0.0, 5.0e6]))
        assert scale == 1.0
        assert "fits neither" in units

    def test_empty_input_does_not_raise(self):
        scale, units = _detect_capacity_scale(np.array([]))
        assert scale == 1.0
        assert "no finite values" in units

    def test_ignores_non_finite_values(self):
        scale, _ = _detect_capacity_scale(np.array([np.nan, np.inf, 740.0]))
        assert scale == 0.001


class TestLoad:
    def test_loads_every_cell(self, loaded):
        telemetry, report = loaded
        assert report.n_cells == 3
        assert set(telemetry["cell_id"].unique()) == {"Cell1", "Cell2", "Cell3"}

    def test_emits_unified_schema_columns(self, loaded):
        telemetry, _ = loaded
        for column in ("cell_id", "cycle", "voltage_v", "temperature_c",
                       "test_time_s", "capacity_ah_curve", "measurement",
                       "dataset", "cohort"):
            assert column in telemetry.columns

    def test_within_cycle_trace_is_named_distinctly_from_the_scalar(self, loaded):
        """`capacity_ah_curve` is a trace; `capacity_ah` is a per-cycle total.

        Overloading one name for both is the unit ambiguity that produced the
        CALCE 'initial vs post-storage' artifact (docs/calce_dataset_note.md).
        """
        telemetry, _ = loaded
        assert "capacity_ah_curve" in telemetry.columns
        assert "capacity_ah" not in telemetry.columns

    def test_cycle_numbers_follow_the_characterisation_stride(self, loaded):
        telemetry, _ = loaded
        expected = {i * STRIDE for i in range(N_CHARACTERISATIONS)}
        assert set(telemetry["cycle"].unique()) == expected

    def test_capacity_is_converted_to_amp_hours(self, loaded):
        telemetry, report = loaded
        assert report.capacity_scale == 0.001
        assert report.capacity_units_detected == "mAh"
        assert telemetry["capacity_ah_curve"].abs().max() <= 1.0

    def test_single_cohort_is_stated_not_inferred(self, loaded):
        telemetry, report = loaded
        assert report.single_cohort
        assert set(telemetry["cohort"].unique()) == {OXFORD_COHORT}
        assert "leave-one-cohort-out is unavailable" in report.render()

    def test_missing_blocks_are_reported_by_name(self, loaded):
        """The fixture omits Cell2/OCVdc; the report must say so."""
        _, report = loaded
        missing = {block for cell, block in report.blocks_missing if cell == "Cell2"}
        assert any("OCVdc" in b for b in missing)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_oxford_mat(tmp_path / "absent.mat")

    def test_file_without_cell_variables_raises_with_what_it_found(self, tmp_path):
        from scipy.io import savemat

        path = tmp_path / "wrong.mat"
        savemat(str(path), {"SomethingElse": {"a": np.array([1.0])}})
        with pytest.raises(ValueError, match="no Cell<N> variables"):
            load_oxford_mat(path)


class TestSummarize:
    def test_one_row_per_cell_characterisation(self, loaded):
        telemetry, _ = loaded
        summary = summarize_oxford_cycles(telemetry)
        assert len(summary) == 3 * N_CHARACTERISATIONS

    def test_recovers_the_fixtures_fade_exactly(self, loaded):
        """The fixture's known parameters must come back out.

        Cell1 fades at exactly FADE_MAH per characterisation; cells 2 and 3
        are scaled by 1.25x and 1.5x by the generator.
        """
        telemetry, _ = loaded
        summary = summarize_oxford_cycles(telemetry)
        cell1 = summary[summary["cell_id"] == "Cell1"].sort_values("cycle")

        assert cell1["capacity_ah"].iloc[0] == pytest.approx(INITIAL_MAH / 1000, abs=1e-6)
        steps = np.diff(cell1["capacity_ah"].to_numpy())
        assert np.allclose(steps, -FADE_MAH / 1000, atol=1e-9)

    def test_sign_convention_is_normalised(self, loaded):
        """The fixture writes discharge as negative; capacity must be positive."""
        telemetry, _ = loaded
        assert (telemetry.loc[
            telemetry["measurement"] == CAPACITY_MEASUREMENT, "capacity_ah_curve"
        ] <= 0).all()
        assert (summarize_oxford_cycles(telemetry)["capacity_ah"] > 0).all()

    def test_uses_only_the_discharge_block(self, loaded):
        """Averaging a 1C discharge with a C/18 OCV sweep mixes quantities."""
        telemetry, _ = loaded
        summary = summarize_oxford_cycles(telemetry)
        n_discharge_rows = int(
            (telemetry["measurement"] == CAPACITY_MEASUREMENT).sum()
        )
        assert int(summary["n_samples"].sum()) == n_discharge_rows

    def test_refuses_when_the_requested_block_is_absent(self, loaded):
        telemetry, _ = loaded
        with pytest.raises(ValueError, match="no 'C1ch_missing' blocks"):
            summarize_oxford_cycles(telemetry, measurement="C1ch_missing")

    def test_refuses_without_a_capacity_column(self, loaded):
        telemetry, _ = loaded
        with pytest.raises(ValueError, match="no 'capacity_ah_curve' column"):
            summarize_oxford_cycles(telemetry.drop(columns=["capacity_ah_curve"]))

    def test_missing_required_columns_raises(self):
        with pytest.raises(ValueError, match="missing"):
            summarize_oxford_cycles(pd.DataFrame({"cell_id": ["a"]}))


class TestCapacityLoss:
    def test_initial_capacity_is_the_first_characterisation(self, loaded):
        """Not a median of the first few: Oxford characterises every 100 cycles.

        A median over the first five would average across 400 cycles of real
        ageing, which is why this differs from the CALCE equivalent.
        """
        telemetry, _ = loaded
        fade = oxford_capacity_loss(summarize_oxford_cycles(telemetry))
        first = fade.sort_values(["cell_id", "cycle"]).groupby("cell_id").first()
        assert np.allclose(first["capacity_loss"], 0.0, atol=1e-9)
        assert np.allclose(first["soh"], 100.0, atol=1e-9)

    def test_fade_increases_monotonically(self, loaded):
        telemetry, _ = loaded
        fade = oxford_capacity_loss(summarize_oxford_cycles(telemetry))
        for _, group in fade.groupby("cell_id"):
            values = group.sort_values("cycle")["capacity_loss"].to_numpy()
            assert np.all(np.diff(values) > 0)

    def test_cells_have_distinguishable_fade_rates(self, loaded):
        """A dataset where every cell is identical cannot support LOBO."""
        telemetry, _ = loaded
        fade = oxford_capacity_loss(summarize_oxford_cycles(telemetry))
        finals = fade.groupby("cell_id")["capacity_loss"].max()
        assert finals.nunique() == 3

    def test_nominal_capacity_is_in_the_expected_range(self, loaded):
        """Guards the unit conversion end to end against the datasheet value."""
        telemetry, _ = loaded
        fade = oxford_capacity_loss(summarize_oxford_cycles(telemetry))
        assert fade["capacity_ah"].max() == pytest.approx(
            OXFORD_NOMINAL_CAPACITY_AH, abs=0.05
        )

    def test_requires_summarize_first(self):
        with pytest.raises(ValueError, match="no 'capacity_ah' column"):
            oxford_capacity_loss(pd.DataFrame({"cell_id": ["a"], "cycle": [0]}))
