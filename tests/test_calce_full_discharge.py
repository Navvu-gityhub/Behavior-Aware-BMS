"""Tests for CALCE full-discharge detection.

Four of these encode defects found by running the detector against real CALCE
archives rather than against a fixture, which is the third time in this project
that a structurally valid frame turned out to carry a wrong quantity.

`test_a_per_file_clock_reset_does_not_fragment_discharges` — Arbin's `Test_Time`
restarts at every file boundary (CS2_33 resets 21 times, the largest being
819352.3 -> 30.0). Sorting by it interleaves twenty-two unrelated files and
fragments every discharge; the symptom was 0.002 Ah capability on a 1.1 Ah cell.

`test_the_cutoff_is_not_defined_by_the_partial_cycles` — the failure that makes
this problem hard. A cell doing thousands of shallow cycles and tens of full
ones has a terminal-voltage distribution dominated by the shallow ones, so any
quantile of it returns the *partial* stopping voltage, every partial then
trivially "reaches cutoff", and the detector certifies exactly what it exists to
reject.

`test_a_partial_discharge_to_cutoff_is_not_full` — CS2_5's partials discharge to
the 2.7 V cutoff from a partially charged state, so 98.5% of them reach cutoff
while moving a fifth of the cell's capacity. Cutoff alone is insufficient.

`test_capacity_is_never_scaled_up_to_look_complete` — the standing rule from
`telemetry/cycles.py`, which this module reuses rather than reimplements.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.io.calce_full_discharge import (
    FULL_CHARGE_FRACTION,
    MIN_CAPABILITY_OVER_MAX,
    _estimate_cutoff,
    _monotonic_time,
    detect_full_discharges,
    full_discharge_frame,
)

NOMINAL_AH = 1.1
CUTOFF_V = 2.70
FULL_V = 4.15
PARTIAL_FLOOR_V = 3.78


def _discharge(
    start_time: float,
    charge_ah: float,
    current_a: float,
    v_start: float,
    v_end: float,
    period_s: float = 30.0,
) -> pd.DataFrame:
    """One constant-current discharge moving `charge_ah`, with a voltage ramp."""
    duration_s = charge_ah * 3600.0 / abs(current_a)
    n = max(int(duration_s / period_s), 6)
    time_s = start_time + np.arange(n) * period_s
    return pd.DataFrame({
        "test_time_s": time_s,
        "current_a": np.full(n, -abs(current_a)),
        "voltage_v": np.linspace(v_start, v_end, n),
    })


def _rest(start_time: float, n: int = 6, period_s: float = 30.0) -> pd.DataFrame:
    time_s = start_time + np.arange(n) * period_s
    return pd.DataFrame({
        "test_time_s": time_s,
        "current_a": np.zeros(n),
        "voltage_v": np.full(n, 3.9),
    })


def _charge(
    start_time: float, charge_ah: float, current_a: float = 0.55,
    period_s: float = 30.0,
) -> pd.DataFrame:
    duration_s = charge_ah * 3600.0 / current_a
    n = max(int(duration_s / period_s), 6)
    time_s = start_time + np.arange(n) * period_s
    return pd.DataFrame({
        "test_time_s": time_s,
        "current_a": np.full(n, current_a),
        "voltage_v": np.linspace(3.0, 4.15, n),
    })


def _assemble(blocks: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(blocks, ignore_index=True)
    frame["cycle"] = np.arange(len(frame)) // 50 + 1
    return frame


def _partial_protocol(n_partial: int = 60, n_full: int = 4) -> pd.DataFrame:
    """A Type-6-like protocol: many shallow cycles, a few full reference ones.

    The shallow discharges stop at 3.78 V; the reference discharges run to the
    2.70 V cutoff. This is the structure that defeats a naive cutoff test.
    """
    blocks: list[pd.DataFrame] = []
    t = 0.0
    for index in range(n_partial + n_full):
        is_reference = index % (n_partial // n_full) == 0
        if is_reference:
            block = _discharge(t, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)
        else:
            block = _discharge(t, 0.36, 0.55, FULL_V, PARTIAL_FLOOR_V)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        blocks.append(_rest(t))
        t += 6 * 30.0
        recharge = _charge(t, NOMINAL_AH if is_reference else 0.36)
        blocks.append(recharge)
        t = float(recharge["test_time_s"].iloc[-1]) + 30.0
        blocks.append(_rest(t))
        t += 6 * 30.0
    return _assemble(blocks)


# ---------------------------------------------------------------------------
# The clock reset
# ---------------------------------------------------------------------------

def test_a_per_file_clock_reset_is_stitched_not_sorted():
    """Row order is the true temporal order; the time column is not."""
    time_s = pd.Series([10.0, 20.0, 30.0, 10.0, 20.0, 30.0, 10.0, 20.0])
    stitched = _monotonic_time(time_s)

    assert stitched.is_monotonic_increasing
    # Elapsed time is preserved, not invented: three files of 20 s each.
    assert stitched.iloc[-1] == pytest.approx(50.0)


def test_a_monotonic_clock_is_left_alone():
    time_s = pd.Series([0.0, 30.0, 60.0, 90.0])
    np.testing.assert_allclose(_monotonic_time(time_s).to_numpy(), time_s.to_numpy())


def test_a_per_file_clock_reset_does_not_fragment_discharges():
    """The defect that produced 0.002 Ah capability on a 1.1 Ah cell.

    Sorting the concatenated frame by its raw time column interleaves the files
    and shatters every discharge into one-sample runs.
    """
    first = _discharge(0.0, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)
    second = _discharge(0.0, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)  # clock restarts
    frame = _assemble([first, _rest(float(first["test_time_s"].iloc[-1]) + 30), second])

    discharges, summary = detect_full_discharges(frame, "CS2_TEST")

    assert summary.n_discharges == 2, "the clock reset split or merged discharges"
    np.testing.assert_allclose(
        discharges["charge_moved_ah"].to_numpy(float), NOMINAL_AH, rtol=0.05
    )


# ---------------------------------------------------------------------------
# The cutoff
# ---------------------------------------------------------------------------

def test_the_cutoff_is_not_defined_by_the_partial_cycles():
    """A quantile of all terminal voltages is set by whichever type is common.

    Here 2,000 discharges stop at 3.78 V and 40 reach 2.70 V — roughly CS2_24's
    ratio. A 2nd-percentile cutoff lands up at the partial floor, which is the
    bug: every partial then trivially reaches it.
    """
    terminal = pd.Series([PARTIAL_FLOOR_V] * 2000 + [CUTOFF_V] * 40)

    # The naive estimator is dragged to the partial floor...
    assert terminal.quantile(0.02) > CUTOFF_V + 0.5
    # ...while this one recovers the real cutoff.
    assert _estimate_cutoff(terminal) == pytest.approx(CUTOFF_V, abs=0.01)


def test_the_cutoff_needs_a_handful_of_deep_discharges_to_be_estimable():
    """A documented limit, not an accident.

    The estimator medians the deepest `MIN_CUTOFF_SAMPLES` terminal voltages, so
    a cell with only one or two full discharges cannot be calibrated and reports
    the partial floor instead. That is the honest failure: with two reference
    points there is no way to tell a genuine cutoff from an outlier.

    It costs nothing in practice — `targets.MIN_CYCLES_FOR_SOH` already excludes
    any cell with fewer than ten usable cycles.
    """
    barely = pd.Series([PARTIAL_FLOOR_V] * 98 + [CUTOFF_V] * 2)
    assert _estimate_cutoff(barely) == pytest.approx(PARTIAL_FLOOR_V, abs=0.01)


def test_sensor_artifacts_do_not_drag_the_cutoff_below_reality():
    """CS2_5 records voltages down to -2.54 V. A cell does not go negative."""
    terminal = pd.Series([-2.54, -1.0, 0.0] + [CUTOFF_V] * 50)
    assert _estimate_cutoff(terminal) == pytest.approx(CUTOFF_V, abs=0.01)


def test_a_cell_with_no_plausible_voltage_yields_no_cutoff():
    assert np.isnan(_estimate_cutoff(pd.Series([-5.0, -3.0])))


# ---------------------------------------------------------------------------
# Full vs partial
# ---------------------------------------------------------------------------

def test_a_partial_cycling_protocol_recovers_an_absolute_reference():
    """The point of the module: Types 5 and 6 scored against a partial-cycle
    reference of ~0.26-0.42 Ah on a 1.1 Ah cell (ADR 0009)."""
    frame = _partial_protocol(n_partial=60, n_full=4)
    discharges, summary = detect_full_discharges(frame, "CS2_TEST")

    assert summary.n_full >= 3, "reference discharges were not recovered"
    # The recovered capability is the cell's real capacity, not a partial one.
    assert summary.capability_ah == pytest.approx(NOMINAL_AH, rel=0.10)
    assert summary.cutoff_v == pytest.approx(CUTOFF_V, abs=0.05)


def test_a_partial_discharge_to_cutoff_is_not_full():
    """CS2_5's partials reach the 2.7 V cutoff from a partial charge, so 98.5%
    pass a cutoff test while moving a fifth of the cell's capacity."""
    blocks: list[pd.DataFrame] = []
    t = 0.0
    # A few full discharges among many shallow ones that also end at cutoff.
    for index in range(60):
        is_reference = index % 12 == 0
        charge = NOMINAL_AH if is_reference else 0.24
        start_v = FULL_V if is_reference else 3.35
        block = _discharge(t, charge, 0.55, start_v, CUTOFF_V)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        blocks.append(_rest(t))
        t += 6 * 30.0
        recharge = _charge(t, charge)
        blocks.append(recharge)
        t = float(recharge["test_time_s"].iloc[-1]) + 30.0

    discharges, summary = detect_full_discharges(_assemble(blocks), "CS2_TEST")

    assert (discharges["reaches_cutoff"]).mean() > 0.9, "fixture assumption"
    # Only the reference discharges are full, not the 55 shallow ones that also
    # terminated at the cutoff.
    assert summary.n_full <= 8, "shallow discharges to cutoff counted as full"
    assert summary.n_full >= 3, "reference discharges were lost"
    assert summary.capability_ah == pytest.approx(NOMINAL_AH, rel=0.15)


def test_capacity_is_never_scaled_up_to_look_complete():
    """A partial's measured charge is left exactly as integrated."""
    frame = _partial_protocol(n_partial=40, n_full=4)
    discharges, _ = detect_full_discharges(frame, "CS2_TEST")

    partial = discharges[~discharges["is_full"]]
    assert not partial.empty
    assert partial["charge_moved_ah"].max() < NOMINAL_AH * FULL_CHARGE_FRACTION


def test_a_full_discharge_protocol_keeps_most_of_its_cycles():
    """Types 1 and 2 already worked; the detector must not throw them away."""
    blocks: list[pd.DataFrame] = []
    t = 0.0
    for _ in range(20):
        block = _discharge(t, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        recharge = _charge(t, NOMINAL_AH)
        blocks.append(recharge)
        t = float(recharge["test_time_s"].iloc[-1]) + 30.0

    _, summary = detect_full_discharges(_assemble(blocks), "CS2_TEST")
    assert summary.usable_fraction > 0.8


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing", ["current_a", "voltage_v", "test_time_s"])
def test_a_missing_channel_raises_rather_than_being_inferred(missing):
    """Guessing one would silently change every capacity produced."""
    frame = _assemble([_discharge(0.0, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)])
    with pytest.raises(ValueError, match=missing):
        detect_full_discharges(frame.drop(columns=[missing]), "CS2_TEST")


def test_a_cell_with_no_discharge_reports_an_empty_yield():
    frame = _assemble([_charge(0.0, NOMINAL_AH), _rest(9000.0)])
    discharges, summary = detect_full_discharges(frame, "CS2_TEST")
    assert summary.n_full == 0
    assert discharges.empty or not discharges["is_full"].any()


# ---------------------------------------------------------------------------
# The implausible-capability guard
# ---------------------------------------------------------------------------

def test_an_implausibly_small_capability_is_refused():
    """CX2_32: capability 0.0397 Ah against a 1.35 Ah cell, cutoff 1.999 V.

    The detector can confidently return a reference that is not a full
    discharge when a cell's deepest excursions are artifacts. Emitting absolute
    SOH against it would understate capacity by an order of magnitude.
    """
    blocks: list[pd.DataFrame] = []
    t = 0.0
    # Normal full discharges to 2.70 V...
    for _ in range(20):
        block = _discharge(t, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        recharge = _charge(t, NOMINAL_AH)
        blocks.append(recharge)
        t = float(recharge["test_time_s"].iloc[-1]) + 30.0
    # ...plus a handful of tiny excursions that dip far below the real cutoff,
    # which is what drags the estimated cutoff down and the capability with it.
    for _ in range(8):
        block = _discharge(t, 0.02, 0.55, 2.3, 1.9)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        blocks.append(_rest(t))
        t += 6 * 30.0

    _, summary = detect_full_discharges(_assemble(blocks), "CX2_TEST")

    assert summary.refusal, "an implausible capability was accepted"
    assert summary.n_full == 0
    assert "largest discharge" in summary.refusal


def test_the_guard_does_not_fire_on_a_healthy_cell():
    """A degraded cell still discharges most of its capacity; the guard must
    not confuse degradation with a broken reference."""
    frame = _partial_protocol(n_partial=60, n_full=6)
    _, summary = detect_full_discharges(frame, "CS2_TEST")

    assert summary.refusal == ""
    assert summary.capability_over_max > MIN_CAPABILITY_OVER_MAX


def test_a_refused_cell_contributes_no_rows():
    """Refusing must actually remove the cell, not merely annotate it."""
    blocks: list[pd.DataFrame] = []
    t = 0.0
    for _ in range(20):
        block = _discharge(t, NOMINAL_AH, 0.55, FULL_V, CUTOFF_V)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        recharge = _charge(t, NOMINAL_AH)
        blocks.append(recharge)
        t = float(recharge["test_time_s"].iloc[-1]) + 30.0
    for _ in range(8):
        block = _discharge(t, 0.02, 0.55, 2.3, 1.9)
        blocks.append(block)
        t = float(block["test_time_s"].iloc[-1]) + 30.0
        blocks.append(_rest(t))
        t += 6 * 30.0

    discharges, summary = detect_full_discharges(_assemble(blocks), "CX2_TEST")
    assert summary.refusal
    assert full_discharge_frame(discharges, cohort="CX2_Type6").empty


# ---------------------------------------------------------------------------
# Output frame
# ---------------------------------------------------------------------------

def test_the_output_frame_matches_what_the_study_consumes():
    frame = _partial_protocol(n_partial=40, n_full=4)
    discharges, _ = detect_full_discharges(frame, "CS2_TEST")
    out = full_discharge_frame(discharges, cohort="CS2_Type6")

    assert {"cell_id", "cycle", "capacity_ah", "cohort", "dataset"} <= set(out.columns)
    assert (out["dataset"] == "calce").all()
    assert (out["cohort"] == "CS2_Type6").all()


def test_cycles_are_renumbered_contiguously():
    """A gap would read as missing data rather than a filtered partial cycle."""
    frame = _partial_protocol(n_partial=40, n_full=4)
    discharges, _ = detect_full_discharges(frame, "CS2_TEST")
    out = full_discharge_frame(discharges, cohort="CS2_Type6")
    assert list(out["cycle"]) == list(range(1, len(out) + 1))


def test_an_empty_detection_yields_an_empty_frame():
    assert full_discharge_frame(pd.DataFrame()).empty


def test_the_yield_summary_reports_what_was_discarded():
    """Reporting the yield is the standing alternative to scaling partials up."""
    frame = _partial_protocol(n_partial=60, n_full=4)
    _, summary = detect_full_discharges(frame, "CS2_TEST")
    rendered = summary.render()
    assert "discharges are full" in rendered
    assert summary.usable_fraction < 0.5


def test_segmentation_is_delegated_not_reimplemented():
    """One definition of a discharge across bench rig, vehicle bus and Arbin."""
    import src.bms.io.calce_full_discharge as module
    import src.bms.telemetry.cycles as cycles

    assert module.segment_phases is cycles.segment_phases
