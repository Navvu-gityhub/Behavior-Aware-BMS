"""Regression tests for CALCE CS2/CX2 cycling ingestion.

Three tests here guard failures that are invisible if you only check that the
loader returns a DataFrame:

`test_cycle_indices_are_reconciled_across_files` — every CALCE file restarts
`Cycle_Index` at 1. Concatenating naively yields fifteen rows labelled "cycle 1",
which the cycle-level layer collapses into one cycle spanning the cell's whole
life. The frame is structurally valid and the trajectory is destroyed.

`test_files_are_ordered_by_date_not_lexically` — pins a bug found while building
this loader. An unanchored date regex parsed `CS2_33_10_04_10` as month=2,
day=33, year=2010, consuming the cell number. Ordering was arbitrary and the
fade trajectory came out scrambled.

`test_temperature_is_absent_rather_than_defaulted` — CS2 was cycled at room
temperature with no thermocouple. Filling 23.0 would satisfy the schema, flow
into `high_temp_flag`, and produce a thermal stress score for a quantity nobody
measured.

The fixtures are CS2-format files, not CALCE measurements. Any number derived
from them is a property of the fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))

from make_calce_fixture import ARBIN_COLUMNS, make_cell, make_cell_file

from src.bms.io.load_calce_cycling import (
    _DATE_IN_NAME,
    CALCE_ARBIN_ALIASES,
    CALCE_UNAVAILABLE_CHANNELS,
    _sort_key,
    calce_capacity_loss,
    load_calce_cell,
    load_calce_dataset,
    summarize_calce_cycles,
)


@pytest.fixture()
def cs2_cell(tmp_path) -> Path:
    """One cell as three date-named files, five cycles each."""
    cell_dir = tmp_path / "CS2_33"
    make_cell(cell_dir, cell_id="CS2_33", n_files=3, cycles_per_file=5)
    return cell_dir


# ---------------------------------------------------------------------------
# Column mapping
# ---------------------------------------------------------------------------

def test_arbin_columns_map_into_the_unified_schema(cs2_cell):
    telemetry, _ = load_calce_cell(cs2_cell)
    for column in ("cell_id", "cycle", "voltage_v", "current_a", "capacity_ah"):
        assert column in telemetry.columns, f"missing {column}"


def test_discharge_capacity_is_mapped_because_it_is_the_fade_target():
    """The shared alias table does not cover it; without this there is no target."""
    assert CALCE_ARBIN_ALIASES["Discharge_Capacity(Ah)"] == "capacity_ah"


def test_internal_resistance_is_mapped(cs2_cell):
    telemetry, _ = load_calce_cell(cs2_cell)
    assert "resistance_ohm" in telemetry.columns


# ---------------------------------------------------------------------------
# The multi-file problem
# ---------------------------------------------------------------------------

def test_cycle_indices_are_reconciled_across_files(cs2_cell):
    """Three files of five cycles must yield fifteen cycles, not five."""
    telemetry, report = load_calce_cell(cs2_cell)

    assert report.n_files == 3
    assert report.n_cycles == 15
    cycles = sorted(telemetry["cycle"].unique())
    assert cycles == list(range(1, 16))


def test_reconciled_cycles_produce_a_monotonic_fade_trajectory(cs2_cell):
    """The point of reconciliation: the trajectory survives concatenation."""
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = calce_capacity_loss(summarize_calce_cycles(telemetry))

    assert len(summary) == 15
    assert summary["capacity_ah"].is_monotonic_decreasing
    assert summary["capacity_loss"].is_monotonic_increasing


def test_files_are_ordered_by_date_not_lexically():
    """Pins a real bug: the date regex consumed the cell number.

    'CS2_33_10_04_10' parsed as month=2, day=33, year=2010, so file ordering was
    arbitrary and a cell's fade trajectory came out scrambled. Anchoring the
    pattern to the end of the stem fixes it.
    """
    assert _DATE_IN_NAME.search("CS2_33_10_04_10").groups() == ("10", "04", "10")
    assert _DATE_IN_NAME.search("CS2_33_9_20_10").groups() == ("9", "20", "10")
    assert _DATE_IN_NAME.search("CX2_4_1_12_11").groups() == ("1", "12", "11")


def test_september_sorts_before_october_despite_lexical_order(tmp_path):
    """'9_20_10' sorts after '10_04_10' as text while preceding it in time."""
    for name in ("CS2_33_10_04_10.csv", "CS2_33_9_20_10.csv"):
        (tmp_path / name).write_text("Cycle_Index\n1\n")

    ordered = sorted(tmp_path.iterdir(), key=_sort_key)
    assert ordered[0].name == "CS2_33_9_20_10.csv"
    assert ordered[1].name == "CS2_33_10_04_10.csv"


def test_a_file_with_no_date_falls_back_to_mtime(tmp_path):
    path = tmp_path / "undated_export.csv"
    path.write_text("Cycle_Index\n1\n")
    key = _sort_key(path)
    assert key[0] == 1  # the fallback branch, ordered after dated files


# ---------------------------------------------------------------------------
# What CALCE does not record
# ---------------------------------------------------------------------------

def test_temperature_is_absent_rather_than_defaulted(cs2_cell):
    """CS2 was cycled at room temperature with no thermocouple channel.

    Filling 23.0 would satisfy the schema and then flow into high_temp_flag,
    producing a thermal stress score for a quantity nobody measured.
    """
    telemetry, report = load_calce_cell(cs2_cell)

    assert "temperature_c" not in telemetry.columns
    assert report.has_temperature is False
    assert "temperature_c" in report.unavailable_channels
    assert "not recorded by this dataset" in report.render()


def test_soc_is_absent_because_arbin_does_not_report_it(cs2_cell):
    telemetry, report = load_calce_cell(cs2_cell)
    assert "soc" not in telemetry.columns
    assert "soc" in report.unavailable_channels


def test_the_unavailable_channel_list_is_explicit():
    assert set(CALCE_UNAVAILABLE_CHANNELS) == {"temperature_c", "soc"}


def test_cx2_4_can_receive_thermocouple_data(tmp_path):
    """CX2_4 is the one CALCE cell cycled across temperatures."""
    cell_dir = tmp_path / "CX2_4"
    make_cell(cell_dir, cell_id="CX2_4", n_files=1, cycles_per_file=4)

    # Thermocouple logger output, on its own clock.
    thermal_dir = tmp_path / "Temperature"
    thermal_dir.mkdir()
    times = np.arange(0, 4 * 60 * 90.0, 137.0)
    pd.DataFrame({
        "Test_Time(s)": times,
        "Temperature (C)": 25 + 10 * (times / times.max()),
    }).to_csv(thermal_dir / "CX2_4_1_12_11.csv", index=False)

    telemetry, report = load_calce_cell(
        cell_dir, cell_id="CX2_4", temperature_dir=thermal_dir
    )
    assert report.has_temperature is True
    assert "temperature_c" in telemetry.columns
    assert telemetry["temperature_c"].notna().any()


# ---------------------------------------------------------------------------
# Per-cycle reduction
# ---------------------------------------------------------------------------

def _write_cadex_file(path: Path) -> None:
    """A minimal CADEX-format export, as CS2_8 and CS2_21 actually ship."""
    header = ("Time\tStatus code\tStatus category\tStatus color\tPgm code\t"
              "Pgm step\tPgm para\tPgm cycle\tmV\tmA\tTemperature\tDuration\t"
              "Charge count\tDischarge count\tCapacity")
    rows = [f"{i}.0\t8\t3\t3\t0\t1\t2\t{1 + i // 3}\t4210\t349\t19\t3\t1\t0\t0"
            for i in range(9)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([header, *rows]), encoding="utf-8")


def test_cadex_export_is_refused_rather_than_misread(tmp_path):
    """A CADEX file loaded through the Arbin path yields capacity off by ~1000x.

    CS2_21 read as "capacity 97.0 -> 86.0 Ah" on a 1.1 Ah cell before this
    guard: plausible-looking, monotonically declining, and nonsense. Refusing
    costs no cohort — CS2_8 and CS2_21 are both Type 1, which retains CS2_33
    and CS2_34.
    """
    cell_dir = tmp_path / "CS2_21"
    _write_cadex_file(cell_dir / "CS2_21_1_19_10.txt")

    with pytest.raises(ValueError, match="CADEX tester export"):
        load_calce_cell(cell_dir)


def test_arbin_files_are_not_mistaken_for_cadex(cs2_cell):
    """The signature check must not reject the format it is meant to accept."""
    telemetry, _ = load_calce_cell(cs2_cell)
    assert len(telemetry) > 0


def test_container_directory_is_not_reported_as_a_cell(tmp_path):
    """`CS2/` holds the Type folders and is not itself a cell."""
    from src.bms.io.load_calce_cycling import discover_calce_cells

    make_cell(tmp_path / "CS2" / "Type1" / "CS2_33",
              cell_id="CS2_33", n_files=1, cycles_per_file=3)

    found = discover_calce_cells(tmp_path)
    assert [s.cell_id for s in found] == ["CS2_33"]
    assert found[0].cohort == "CS2_Type1"


def test_cohort_comes_from_the_type_directory(tmp_path):
    """CALCE's own experiment grouping, which is what LOCO needs."""
    from src.bms.io.load_calce_cycling import discover_calce_cells

    for cell_id, type_dir in (("CS2_33", "Type1"), ("CS2_35", "Type2")):
        make_cell(tmp_path / "CS2" / type_dir / cell_id,
                  cell_id=cell_id, n_files=1, cycles_per_file=3)

    cohorts = {s.cell_id: s.cohort for s in discover_calce_cells(tmp_path)}
    assert cohorts == {"CS2_33": "CS2_Type1", "CS2_35": "CS2_Type2"}


def test_cohort_is_namespaced_by_archive_family(tmp_path):
    """CS2 and CX2 both number their types 1-6; a bare `Type1` would merge them.

    They are different cell designs (1.1 Ah prismatic vs 1.35 Ah), so merging
    would leave half of a protocol's cells in training when that protocol is
    held out — flattering leave-one-cohort-out without it noticing.
    """
    from src.bms.io.load_calce_cycling import discover_calce_cells

    make_cell(tmp_path / "CS2" / "Type1" / "CS2_33",
              cell_id="CS2_33", n_files=1, cycles_per_file=3)
    make_cell(tmp_path / "CX2" / "Type1" / "CX2_16",
              cell_id="CX2_16", n_files=1, cycles_per_file=3)

    cohorts = {s.cell_id: s.cohort for s in discover_calce_cells(tmp_path)}
    assert cohorts == {"CS2_33": "CS2_Type1", "CX2_16": "CX2_Type1"}
    assert len(set(cohorts.values())) == 2, "families must not share a cohort"


def test_sort_key_tolerates_a_path_that_does_not_exist(tmp_path):
    """Archive members are addressed as paths with no filesystem entry.

    Real CS2 archives contain `CS2_21_7_9b_10.txt` (a letter inside the date)
    and `CS_2_5_15_12_calibration.xls` (not a cycling file), neither of which
    parses as a date. An unguarded stat() on those lost the entire cell.
    """
    key = _sort_key(Path("CS2_21/CS2_21_7_9b_10.txt"))
    assert isinstance(key, tuple)
    assert key[0] == 1  # fell through to the mtime branch without raising


def test_capacity_is_the_within_cycle_span_not_the_maximum(cs2_cell):
    """Arbin's Discharge_Capacity accumulates ACROSS cycles; it does not reset.

    So the charge a cycle delivered is the span of the counter within that
    cycle, not its maximum. Taking the maximum returns cumulative throughput,
    which on real data made CS2_33 report a state of health of 383% on a
    1.1 Ah cell.
    """
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = summarize_calce_cycles(telemetry)

    first = telemetry[telemetry["cycle"] == 1]
    span = first["capacity_ah"].max() - first["capacity_ah"].min()
    assert summary.iloc[0]["capacity_ah"] == pytest.approx(span)


def test_capacity_stays_near_nominal_rather_than_accumulating(cs2_cell):
    """The regression guard for the 383%-SOH bug.

    A 1.1 Ah cell must report ~1.1 Ah on every cycle. If the counter is being
    read as a level rather than a span, later cycles climb without bound.
    """
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = summarize_calce_cycles(telemetry)

    assert summary["capacity_ah"].max() < 1.5, (
        "per-cycle capacity exceeds any plausible value for a 1.1 Ah cell — "
        "the accumulating counter is being read as a level"
    )
    assert summary["capacity_ah"].min() > 0.5


def test_cumulative_throughput_is_kept_separately(cs2_cell):
    """Retained so an accumulating export is distinguishable from a resetting one."""
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = summarize_calce_cycles(telemetry)

    assert "cumulative_capacity_ah" in summary.columns
    assert summary["cumulative_capacity_ah"].is_monotonic_increasing
    assert summary["cumulative_capacity_ah"].iloc[-1] > summary["capacity_ah"].iloc[-1]


def test_initial_capacity_uses_a_median_not_cycle_one(cs2_cell):
    """The first Arbin cycle often includes a formation step."""
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = calce_capacity_loss(summarize_calce_cycles(telemetry))
    expected = summary["capacity_ah"].head(5).median()
    assert summary["initial_capacity_ah"].iloc[0] == pytest.approx(expected)


def test_soh_is_derived_from_measured_capacity(cs2_cell):
    telemetry, _ = load_calce_cell(cs2_cell)
    summary = calce_capacity_loss(summarize_calce_cycles(telemetry))
    ratio = summary["capacity_ah"] / summary["initial_capacity_ah"] * 100
    np.testing.assert_allclose(summary["soh"], ratio)


def test_summarize_without_capacity_raises_rather_than_inventing_one():
    frame = pd.DataFrame({"cell_id": ["A"] * 4, "cycle": [1, 1, 2, 2]})
    summary = summarize_calce_cycles(frame)
    with pytest.raises(ValueError, match="no 'capacity_ah' column"):
        calce_capacity_loss(summary)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_a_corrupt_file_is_skipped_with_its_reason(tmp_path):
    """One bad export must not cost the rest of the cell."""
    cell_dir = tmp_path / "CS2_35"
    make_cell(cell_dir, cell_id="CS2_35", n_files=2, cycles_per_file=4)
    (cell_dir / "CS2_35_3_01_11.csv").write_text("")  # empty, unparseable

    telemetry, report = load_calce_cell(cell_dir)
    assert report.n_files == 2
    assert len(report.files_skipped) == 1
    assert "CS2_35_3_01_11.csv" in report.files_skipped[0][0]
    assert not telemetry.empty


def test_a_directory_with_no_readable_files_raises(tmp_path):
    empty = tmp_path / "CS2_99"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no readable cycling files"):
        load_calce_cell(empty)


def test_a_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such directory"):
        load_calce_cell(tmp_path / "absent")


def test_dataset_loads_every_cell_subdirectory(tmp_path):
    for cell in ("CS2_33", "CS2_34"):
        make_cell(tmp_path / cell, cell_id=cell, n_files=2, cycles_per_file=4)

    telemetry, reports = load_calce_dataset(tmp_path)
    assert set(telemetry["cell_id"].unique()) == {"CS2_33", "CS2_34"}
    assert len(reports) == 2
    assert all(r.n_cycles == 8 for r in reports)


def test_one_broken_cell_does_not_abort_the_dataset(tmp_path):
    make_cell(tmp_path / "CS2_33", cell_id="CS2_33", n_files=1, cycles_per_file=3)
    (tmp_path / "CS2_BROKEN").mkdir()  # no files at all

    telemetry, reports = load_calce_dataset(tmp_path)
    assert set(telemetry["cell_id"].unique()) == {"CS2_33"}
    broken = next(r for r in reports if r.cell_id == "CS2_BROKEN")
    assert broken.n_rows == 0
    assert broken.files_skipped


def test_dataset_without_cell_subdirectories_raises(tmp_path):
    (tmp_path / "loose_file.csv").write_text("Cycle_Index\n1\n")
    with pytest.raises(FileNotFoundError, match="no cells found"):
        load_calce_dataset(tmp_path)


def test_the_fixture_reproduces_the_real_arbin_column_set():
    """If this drifts, the loader is being tested against a schema CALCE
    does not actually ship."""
    assert len(ARBIN_COLUMNS) == 17
    for column in ("Cycle_Index", "Discharge_Capacity(Ah)", "Current(A)", "Voltage(V)"):
        assert column in ARBIN_COLUMNS
    assert not any("temp" in c.lower() for c in ARBIN_COLUMNS)


# ---------------------------------------------------------------------------
# Integration with the scoring pipeline
# ---------------------------------------------------------------------------

from src.bms.io.calce_analysis import (  # noqa: E402
    CHANNEL_CONSUMERS,
    analyze_calce_cell,
    measured_feasibility,
)


def test_a_cs2_cell_yields_measured_soh_without_behavioural_scores(cs2_cell):
    """What CS2 is actually good for, and what it cannot support.

    Capacity fade is measured and real. The behavioural stages are skipped
    because the channels they compute from were never recorded.
    """
    analysis = analyze_calce_cell(cs2_cell)

    assert analysis.status == "MEASURED_ONLY"
    assert analysis.has_measured_soh is True
    assert analysis.scored is False
    assert analysis.stages_completed == ("load", "capacity_fade")
    assert analysis.cycles["soh"].is_monotonic_decreasing


def test_the_blocked_stages_name_the_missing_channel(cs2_cell):
    """A bare KeyError would not tell an engineer what to instrument."""
    analysis = analyze_calce_cell(cs2_cell)
    blocked = dict(analysis.stages_unavailable)

    assert set(blocked) == {
        "behavior_flags", "risk_assessment", "health_index", "guardian",
    }
    reason = blocked["risk_assessment"]
    assert "temperature_c" in reason
    assert "soc" in reason
    assert "high_temp_flag" in reason  # names the consumer, not just the column


def test_temperature_is_not_substituted_with_room_temperature(cs2_cell):
    """23 C is documented for CS2, and filling it would be the worst option.

    A constant cannot raise high_temp_flag, so every CALCE cell would score as
    thermally unstressed regardless of what happened to it — the NaN-as-healthy
    defect with extra steps.
    """
    analysis = analyze_calce_cell(cs2_cell)
    assert "temperature_c" not in analysis.cycles.columns
    assert analysis.guardian.empty


def test_joining_thermocouple_data_unblocks_the_temperature_requirement(tmp_path):
    """CX2_4 has a real thermal channel, so the pipeline follows the
    instrumentation rather than the dataset label."""
    cell_dir = tmp_path / "CX2_4"
    make_cell(cell_dir, cell_id="CX2_4", n_files=2, cycles_per_file=6)

    thermal_dir = tmp_path / "Temperature"
    thermal_dir.mkdir()
    times = np.arange(0, 2 * 6 * 60 * 90.0, 97.0)
    pd.DataFrame({
        "Test_Time(s)": times,
        "Temperature (C)": 25 + 30 * (times / times.max()),
    }).to_csv(thermal_dir / "CX2_4_1_12_11.csv", index=False)

    with_thermal = analyze_calce_cell(
        cell_dir, cell_id="CX2_4", temperature_dir=thermal_dir
    )
    without = analyze_calce_cell(cell_dir, cell_id="CX2_4")

    # Temperature is no longer among the reasons.
    assert "temperature_c" not in dict(with_thermal.stages_unavailable)["guardian"]
    assert "temperature_c" in dict(without.stages_unavailable)["guardian"]
    assert with_thermal.load_report.has_temperature is True


def test_channel_consumers_are_documented_for_every_required_channel():
    for channel in ("temperature_c", "soc", "current_a"):
        assert channel in CHANNEL_CONSUMERS
        assert CHANNEL_CONSUMERS[channel]


@pytest.fixture()
def long_cs2_cell(tmp_path) -> Path:
    """A cell cycled long enough for its fade range to be comparable to NASA's.

    CS2 cells run for hundreds of cycles. A 15-cycle fixture fades by 0.03 Ah
    against NASA's ~0.4 Ah, a spread ratio below the commensurability floor —
    which the screen correctly rejects. That rejection is the check working, so
    the fixture is made realistic rather than the threshold relaxed.
    """
    cell_dir = tmp_path / "CS2_36"
    make_cell(cell_dir, cell_id="CS2_36", n_files=8, cycles_per_file=25)
    return cell_dir


def test_a_short_run_is_correctly_judged_incommensurable(cs2_cell):
    """Fifteen cycles cannot carry a transfer fitted over a full-life range."""
    analysis = analyze_calce_cell(cs2_cell)
    source = pd.DataFrame({
        "capacity_loss": np.linspace(0.0, 0.4, 200),
        "cell_id": ["N"] * 200,
    })
    report = measured_feasibility(source, analysis.cycles, features=["capacity_loss"])

    assert report.feasible is False
    assert "capacity_loss" in report.constant_in_target


def test_measured_feasibility_uses_loaded_data_not_metadata(long_cs2_cell):
    """ADR 0006 requires the measured screen before citing a transfer result."""
    analysis = analyze_calce_cell(long_cs2_cell)
    source = pd.DataFrame({
        "capacity_loss": np.linspace(0.0, 0.4, 200),
        "cell_id": ["N"] * 200,
    })
    report = measured_feasibility(source, analysis.cycles, features=["capacity_loss"])

    assert report.feasible is True
    assert "capacity_loss" in report.usable_features


def test_measured_feasibility_blocks_a_channel_calce_lacks(long_cs2_cell):
    """Temperature is absent from CS2, so it cannot carry a transfer."""
    analysis = analyze_calce_cell(long_cs2_cell)
    source = pd.DataFrame({
        "avg_temp": np.linspace(4.0, 43.0, 200),
        "capacity_loss": np.linspace(0.0, 0.4, 200),
    })
    report = measured_feasibility(
        source, analysis.cycles, features=["avg_temp", "capacity_loss"]
    )

    assert "avg_temp" in report.absent_in_target
    assert report.usable_features == ("capacity_loss",)


def test_the_fixture_generator_writes_one_file_per_request(tmp_path):
    """Pins a fixture bug that produced an upward fade trajectory.

    A fixed five-date list meant n_files=8 overwrote three files, so a cell's
    cycle offsets came from whichever write landed last and capacity appeared
    to increase with age. The loader was correct; the generator was not, and a
    broken fixture silently weakens every test built on it.
    """
    cell_dir = tmp_path / "CS2_40"
    make_cell(cell_dir, cell_id="CS2_40", n_files=8, cycles_per_file=5)

    assert len(list(cell_dir.iterdir())) == 8
    telemetry, report = load_calce_cell(cell_dir)
    assert report.n_cycles == 40

    summary = calce_capacity_loss(summarize_calce_cycles(telemetry))
    assert summary["capacity_ah"].is_monotonic_decreasing
