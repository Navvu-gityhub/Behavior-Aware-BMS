"""Regression tests for curve-level CALCE extraction.

The whole point of this loader is to NOT repeat the mistake that
`summarize_calce_cycles` documents at length: Arbin's `Discharge_Capacity`
counter accumulates across cycles and does not reset, and a loader that
assumes otherwise produces a frame that is structurally valid and carries the
wrong quantity. Against a fixture that resets the counter, both the right and
the wrong implementation pass.

So the central test here is `test_curve_capacity_is_rezeroed_per_cycle`. The
fixture accumulates, as real workbooks do, and a curve that starts at the
cell's lifetime throughput instead of zero would sail through every shape and
type check while making Q(V) meaningless.

The rest guard the discharge selection rule:

`test_charge_samples_are_excluded` — a Q(V) curve containing the charge leg
doubles back on itself. `interpolate_curve` sorts by voltage, so the fold is
invisible downstream and silently averages two different states.

`test_low_rate_cohort_still_yields_curves` — the threshold is a fraction of
each cycle's own current, not a fixed amperage, because CALCE's cohorts differ
by discharge rate. A fixed threshold would admit a different part of the curve
per cohort and manufacture a cohort-dependent feature.

`test_cell_missing_the_window_cycle_is_refused_not_dropped` — refusals are
values here as everywhere else in this codebase.

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

from make_calce_fixture import make_cell

from src.bms.benchmarks.curves import (
    CURVE_COLUMNS,
    delta_q_variance_feature,
    ica_peak_features,
)
from src.bms.io.load_calce_curves import (
    CURVE_FRAME_COLUMNS,
    MIN_CURVE_POINTS,
    _longest_discharge_run,
    curve_features_by_cell,
    extract_discharge_curves,
)
from src.bms.io.load_calce_cycling import load_calce_cell

# The fixture's default is 40 samples per cycle, below MIN_CURVE_POINTS. These
# tests ask for enough points to clear the floor, because a run that is refused
# for being short cannot exercise anything about the curve itself.
SAMPLES = 160
FIXTURE_INITIAL_AH = 1.10
FIXTURE_FADE_PER_CYCLE = 0.002


@pytest.fixture()
def curves(tmp_path) -> pd.DataFrame:
    """One cell, three files, 120 cycles, as curve-level rows."""
    cell_dir = tmp_path / "CS2_33"
    make_cell(
        cell_dir,
        cell_id="CS2_33",
        n_files=3,
        cycles_per_file=40,
        samples_per_cycle=SAMPLES,
    )
    telemetry, _ = load_calce_cell(cell_dir)
    frame, _ = extract_discharge_curves(telemetry)
    return frame


# ---------------------------------------------------------------------------
# The contract benchmarks/curves.py requires
# ---------------------------------------------------------------------------

def test_frame_carries_exactly_the_columns_the_feature_code_requires(curves):
    """`CURVE_COLUMNS` is the benchmark module's stated input contract."""
    for column in (*CURVE_COLUMNS, "cell_id", "cycle"):
        assert column in curves.columns, f"missing {column}"


def test_the_two_modules_agree_on_the_column_names():
    """A rename on either side must fail here, not at feature time.

    `benchmarks/curves.py` documents the format; this loader produces it. The
    two were written apart, so the only thing keeping them aligned is an
    assertion that reads both.
    """
    assert set(CURVE_COLUMNS) <= set(CURVE_FRAME_COLUMNS)


# ---------------------------------------------------------------------------
# The accumulating counter — the failure this module exists to avoid
# ---------------------------------------------------------------------------

def test_curve_capacity_is_rezeroed_per_cycle(curves):
    """Every cycle's Q(V) must start at zero, not at lifetime throughput.

    The fixture accumulates `Discharge_Capacity` monotonically across cycles,
    as real Arbin workbooks do. Without the re-zero, cycle 100 would open near
    110 Ah on a 1.1 Ah cell.
    """
    starts = curves.groupby("cycle")["capacity_ah_curve"].min()
    assert np.allclose(starts, 0.0), (
        f"curves must start at zero; got min {starts.min()}, max {starts.max()}"
    )


def test_curve_span_is_the_cycle_capacity_not_the_running_total(curves):
    """The span of a cycle's curve is that cycle's delivered charge.

    A late cycle must not span more than an early one: the fixture fades. If
    the counter were not re-zeroed the span would grow without bound, which is
    the 383%-state-of-health failure in cycle-level form.
    """
    spans = curves.groupby("cycle")["capacity_ah_curve"].max()
    assert spans.max() < FIXTURE_INITIAL_AH * 1.05, (
        f"largest span {spans.max():.3f} Ah exceeds the fixture's initial "
        f"capacity; the accumulating counter was not re-zeroed"
    )
    early, late = spans.loc[spans.index.min()], spans.loc[spans.index.max()]
    assert late < early, "fixture fades, so a later cycle must span less"


# ---------------------------------------------------------------------------
# Discharge selection
# ---------------------------------------------------------------------------

def test_charge_samples_are_excluded(curves):
    """Voltage within one cycle must fall monotonically overall.

    The fixture's charge leg runs 2.7 V upward. Including it would give a
    cycle whose voltage both falls and rises, and since `interpolate_curve`
    sorts by voltage the fold would be invisible in the interpolated result.
    """
    for cycle, rows in curves.groupby("cycle"):
        voltage = rows["voltage_v"].to_numpy()
        assert voltage[0] > voltage[-1], (
            f"cycle {cycle} does not run from high to low voltage; the charge "
            f"leg was not excluded"
        )
        # Allow sampling noise, but a real fold would be a large rise.
        assert voltage.max() - voltage[0] < 0.1, (
            f"cycle {cycle} rises well above its starting voltage"
        )


def test_one_curve_per_cycle_not_two_segments_concatenated(curves):
    """Each cycle contributes exactly one contiguous discharge."""
    per_cycle = curves.groupby("cycle").size()
    assert per_cycle.max() <= SAMPLES, (
        f"a cycle contributed {per_cycle.max()} points from {SAMPLES} "
        f"discharge samples; two segments were concatenated"
    )


def test_low_rate_cohort_still_yields_curves(tmp_path):
    """The threshold scales to the cycle's own current, not a fixed amperage.

    CALCE's cohorts vary discharge rate by design. A fixed threshold tuned to
    the 1.1 A fixture would silently return nothing for a slower protocol,
    which would drop whole cohorts out of a leave-one-cohort-out split.
    """
    cell_dir = tmp_path / "CS2_LOW"
    make_cell(
        cell_dir, cell_id="CS2_LOW", n_files=1, cycles_per_file=10,
        samples_per_cycle=SAMPLES,
    )
    telemetry, _ = load_calce_cell(cell_dir)
    # Quarter the current everywhere, leaving voltage and capacity intact.
    telemetry["current_a"] = telemetry["current_a"] * 0.25

    frame, report = extract_discharge_curves(telemetry)
    assert not frame.empty, (
        "a quarter-rate protocol produced no curves; the discharge threshold "
        "is behaving as a fixed amperage"
    )
    assert report.n_cycles_kept == 10


# ---------------------------------------------------------------------------
# Refusals are values
# ---------------------------------------------------------------------------

def test_short_cycles_are_refused_with_a_named_reason(tmp_path):
    cell_dir = tmp_path / "CS2_SHORT"
    make_cell(
        cell_dir, cell_id="CS2_SHORT", n_files=1, cycles_per_file=5,
        samples_per_cycle=20,   # below MIN_CURVE_POINTS
    )
    telemetry, _ = load_calce_cell(cell_dir, cell_id="CS2_SHORT")
    frame, report = extract_discharge_curves(telemetry)

    assert frame.empty
    assert report.n_cycles_kept == 0
    assert report.n_cycles_seen == 5
    assert sum(report.refusals.values()) == 5
    assert report.cells_without_curves == ("CS2_SHORT",)
    # The reason must distinguish "too short" from "no discharge at all";
    # these cycles do discharge, there are just 20 samples of it.
    assert str(MIN_CURVE_POINTS) in report.summary(), report.summary()
    assert "never negative" not in report.summary(), report.summary()


def test_report_yield_is_reported_not_implied(curves, tmp_path):
    cell_dir = tmp_path / "CS2_Y"
    make_cell(cell_dir, cell_id="CS2_Y", n_files=1, cycles_per_file=10,
              samples_per_cycle=SAMPLES)
    telemetry, _ = load_calce_cell(cell_dir)
    _, report = extract_discharge_curves(telemetry)
    assert report.yield_fraction == pytest.approx(1.0)
    assert "10/10" in report.summary()


def test_cell_missing_the_window_cycle_is_refused_not_dropped(tmp_path):
    """A cell that never reached cycle 100 appears, saying why.

    Dropping it would make the cohort look smaller than it is for a reason the
    reader cannot see from the artifact.
    """
    cell_dir = tmp_path / "CS2_SHORTLIFE"
    make_cell(cell_dir, cell_id="CS2_SHORTLIFE", n_files=1, cycles_per_file=20,
              samples_per_cycle=SAMPLES)
    telemetry, _ = load_calce_cell(cell_dir)
    frame, _ = extract_discharge_curves(telemetry)

    features = curve_features_by_cell(frame, cycle_a=10, cycle_b=100)
    assert len(features) == 1, "the cell must appear, not vanish"
    row = features.iloc[0]
    assert row["refusal_reason"], "a refused cell must carry its reason"
    assert "100" in row["refusal_reason"]
    assert np.isnan(row["delta_q_variance"])


def test_summarized_frame_is_refused_with_a_pointer_to_the_right_loader():
    """Passing cycle-level output here is the obvious mistake; name it."""
    summary = pd.DataFrame({"cell_id": ["A"], "cycle": [1], "capacity_ah": [1.0]})
    with pytest.raises(ValueError) as excinfo:
        extract_discharge_curves(summary)
    message = str(excinfo.value)
    assert "load_calce_dataset" in message
    assert "summarize_calce_cycles" in message


# ---------------------------------------------------------------------------
# Features computed from the frame
# ---------------------------------------------------------------------------

def test_delta_q_variance_is_finite_on_a_well_formed_cell(curves):
    value = delta_q_variance_feature(curves, 10, 100)
    assert np.isfinite(value), (
        "cycles 10 and 100 of a clean fixture must support the feature"
    )


def test_ica_area_recovers_the_discharge_capacity(curves):
    """The integral of |dQ/dV| over V is the charge delivered.

    This is an arithmetic identity, so it is a check on the differentiation
    and interpolation rather than on the battery: if smoothing or the grid
    were wrong, the area would not come back to the span.
    """
    reference = curves[curves["cycle"] == 100]
    features = ica_peak_features(
        reference["voltage_v"].to_numpy(dtype=float),
        reference["capacity_ah_curve"].to_numpy(dtype=float),
    )
    span = float(reference["capacity_ah_curve"].max())
    assert features["ica_area"] == pytest.approx(span, rel=0.05), (
        f"dQ/dV integral {features['ica_area']:.4f} does not recover the "
        f"{span:.4f} Ah span"
    )


def test_feature_frame_carries_the_cohort_for_leave_one_cohort_out(tmp_path):
    """Losing the cohort here would leave a frame that cannot be split."""
    type_dir = tmp_path / "CS2" / "Type1"
    make_cell(type_dir / "CS2_33", cell_id="CS2_33", n_files=1,
              cycles_per_file=10, samples_per_cycle=SAMPLES)
    telemetry, _ = load_calce_cell(type_dir / "CS2_33")
    telemetry["cohort"] = "CS2_Type1"

    frame, _ = extract_discharge_curves(telemetry)
    assert "cohort" in frame.columns
    features = curve_features_by_cell(frame, cycle_a=1, cycle_b=10)
    assert features.iloc[0]["cohort"] == "CS2_Type1"


# ---------------------------------------------------------------------------
# The segment picker
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask, expected", [
    ([False, True, True, False, True], slice(1, 3)),
    ([True, True, True], slice(0, 3)),
    ([False, False], None),
    ([False, True, False, True, True, True], slice(3, 6)),
    ([True, False, False], slice(0, 1)),
])
def test_longest_discharge_run(mask, expected):
    assert _longest_discharge_run(np.array(mask)) == expected
