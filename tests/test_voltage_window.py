"""Tests for voltage-window SOH estimation.

The estimator's appeal is that it fits nothing, so the failure modes are not
the usual modelling ones. They are measurement ones, and these tests pin them:

`test_a_discharge_that_stops_inside_the_window_is_refused` — the important
one. A cycle that ends at 3.7 V has delivered no charge across the 3.7-3.6 V
portion. Reporting the charge it *did* deliver would make a shortened
discharge look like a faded cell, which is the exact error the method exists
to avoid and is invisible in the output.

`test_a_uniformly_faded_cell_reads_its_fade` — the physical assumption. If the
curve scales, the window ratio equals the capacity ratio.

`test_reference_is_a_median_not_a_single_cycle` — one anomalous early
discharge divides every later estimate of that cell, so the reference must not
be one reading.

The fixtures are synthetic discharge curves, not CALCE measurements. Any
number derived from them is a property of the fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.voltage_window import (
    MIN_PLAUSIBLE_SOH,
    MIN_WINDOW_COVERAGE,
    WindowSpec,
    per_cell_fit,
    window_charge,
    window_charge_by_cycle,
    window_soh_table,
)

SPEC = WindowSpec(3.90, 3.60)


def _curve(
    capacity_ah: float = 1.10,
    v_start: float = 4.15,
    v_end: float = 2.70,
    n: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """A monotone discharge: voltage falls linearly, charge accumulates."""
    voltage = np.linspace(v_start, v_end, n)
    fraction = (v_start - voltage) / (v_start - v_end)
    return voltage, fraction * capacity_ah


def _frame(cells: dict[str, list[float]], **kwargs) -> pd.DataFrame:
    """Long curve frame: {cell_id: [capacity per cycle]}."""
    blocks = []
    for cell_id, capacities in cells.items():
        for cycle, cap in enumerate(capacities, start=1):
            voltage, charge = _curve(cap, **kwargs)
            blocks.append(pd.DataFrame({
                "cell_id": cell_id, "cycle": cycle,
                "voltage_v": voltage, "capacity_ah_curve": charge,
            }))
    return pd.concat(blocks, ignore_index=True)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def test_window_charge_is_a_fraction_of_the_discharge():
    voltage, charge = _curve(capacity_ah=1.10)
    q = window_charge(voltage, charge, SPEC)
    # The window spans 0.30 V of a 1.45 V discharge, so on a linear curve it
    # holds 0.30/1.45 of the charge.
    assert q == pytest.approx(1.10 * 0.30 / 1.45, rel=0.02)


def test_window_charge_is_positive():
    """Q accumulates as voltage falls; a sign error would invert the trend."""
    voltage, charge = _curve()
    assert window_charge(voltage, charge, SPEC) > 0


def test_a_uniformly_faded_cell_reads_its_fade():
    """The physical assumption, stated as a test.

    If the discharge curve scales down uniformly, window charge falls in the
    same proportion as capacity, so the ratio IS the state of health.
    """
    voltage, full = _curve(capacity_ah=1.10)
    _, faded = _curve(capacity_ah=0.88)          # 80% SOH
    ratio = window_charge(voltage, faded, SPEC) / window_charge(
        voltage, full, SPEC)
    assert ratio == pytest.approx(0.80, rel=0.01)


# ---------------------------------------------------------------------------
# Refusing what it cannot measure
# ---------------------------------------------------------------------------

def test_a_discharge_that_stops_inside_the_window_is_refused():
    """A partial discharge must not read as a faded cell.

    This cycle stops at 3.75 V, so half the window was never traversed. The
    charge it delivered across the part it did reach is a real number and a
    completely wrong answer.
    """
    voltage, charge = _curve(v_start=4.15, v_end=3.75)
    assert np.isnan(window_charge(voltage, charge, SPEC))


def test_a_discharge_that_starts_below_the_window_is_refused():
    voltage, charge = _curve(v_start=3.70, v_end=2.70)
    assert np.isnan(window_charge(voltage, charge, SPEC))


def test_a_discharge_that_just_covers_the_window_is_accepted():
    voltage, charge = _curve(v_start=3.91, v_end=3.59)
    assert np.isfinite(window_charge(voltage, charge, SPEC))


def test_coverage_threshold_is_the_documented_constant():
    """A cycle covering just under the threshold is refused, just over kept."""
    span = SPEC.span_v
    short = MIN_WINDOW_COVERAGE * span - 0.02
    voltage, charge = _curve(v_start=SPEC.v_high, v_end=SPEC.v_high - short)
    assert np.isnan(window_charge(voltage, charge, SPEC))


def test_too_few_points_is_refused_not_extrapolated():
    assert np.isnan(window_charge(np.array([3.8]), np.array([0.1]), SPEC))
    assert np.isnan(window_charge(np.array([]), np.array([]), SPEC))


def test_non_finite_samples_do_not_poison_the_measurement():
    voltage, charge = _curve()
    voltage = voltage.copy()
    charge = charge.copy()
    voltage[5] = np.nan
    charge[9] = np.nan
    assert np.isfinite(window_charge(voltage, charge, SPEC))


# ---------------------------------------------------------------------------
# The per-cell table
# ---------------------------------------------------------------------------

def test_soh_window_tracks_a_known_fade_trajectory():
    capacities = [1.10, 1.10, 1.10, 1.10, 1.10, 0.99, 0.88, 0.77]
    curves = _frame({"A": capacities})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    got = table.sort_values("cycle")["soh_window"].to_numpy()
    expected = np.array(capacities) / 1.10
    assert np.allclose(got, expected, rtol=0.02)


def test_reference_is_a_median_not_a_single_cycle():
    """One bad early discharge must not rescale the whole trajectory.

    Cycle 1 here reads high. Taking it as the reference would depress every
    later estimate; the median of the first five absorbs it.
    """
    capacities = [1.60, 1.10, 1.10, 1.10, 1.10, 1.10]
    curves = _frame({"A": capacities})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    late = table[table["cycle"] == 6]["soh_window"].iloc[0]
    assert late == pytest.approx(1.0, rel=0.02), (
        "an anomalous first cycle leaked into the reference"
    )


def test_each_cell_gets_its_own_reference():
    """Cells differ in absolute capacity; a shared reference would read that
    difference as health."""
    curves = _frame({"A": [1.10] * 6, "B": [0.80] * 6})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    for cell in ("A", "B"):
        values = table[table["cell_id"] == cell]["soh_window"].dropna()
        assert np.allclose(values, 1.0, rtol=0.02), (
            f"cell {cell} should read full health against its own reference"
        )


def test_cohort_is_carried_through():
    curves = _frame({"A": [1.10] * 6})
    curves["cohort"] = "Type1"
    assert "cohort" in window_charge_by_cycle(curves, SPEC).columns


def test_missing_columns_are_named():
    with pytest.raises(ValueError) as excinfo:
        window_charge_by_cycle(pd.DataFrame({"cell_id": ["A"]}), SPEC)
    assert "voltage_v" in str(excinfo.value)


def test_a_cell_with_no_measurable_cycle_yields_nan_not_an_exception():
    curves = _frame({"A": [1.10] * 3}, v_start=3.5, v_end=2.7)
    table = window_soh_table(curves, spec=SPEC)
    assert table["soh_window"].isna().all()


# ---------------------------------------------------------------------------
# Scoring against truth
# ---------------------------------------------------------------------------

def test_per_cell_fit_recovers_unit_slope_on_a_scaling_cell():
    capacities = [1.10] * 5 + list(np.linspace(1.10, 0.80, 30))
    curves = _frame({"A": capacities})
    truth = pd.DataFrame({
        "cell_id": "A",
        "cycle": range(1, len(capacities) + 1),
        "soh": np.array(capacities) / 1.10,
    })
    table = window_soh_table(curves, truth=truth, spec=SPEC, reference_cycles=5)
    fit = per_cell_fit(table).iloc[0]
    assert fit["slope"] == pytest.approx(1.0, abs=0.05)
    assert fit["bias"] == pytest.approx(0.0, abs=0.05)
    assert fit["mae"] < 0.02


def test_per_cell_fit_without_truth_is_refused():
    curves = _frame({"A": [1.10] * 6})
    with pytest.raises(ValueError) as excinfo:
        per_cell_fit(window_soh_table(curves, spec=SPEC))
    assert "soh" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The charge leg accumulates the other way
# ---------------------------------------------------------------------------

def _charge_curve(capacity_ah: float = 1.10, v_start: float = 3.00,
                  v_end: float = 4.20, n: int = 400):
    """A constant-current charge: voltage rises, charge accumulates with it."""
    voltage = np.linspace(v_start, v_end, n)
    fraction = (voltage - v_start) / (v_end - v_start)
    return voltage, fraction * capacity_ah


def test_charge_phase_window_is_positive():
    """Q grows with voltage on charge; a signed difference would go negative."""
    voltage, charge = _charge_curve()
    assert window_charge(voltage, charge, SPEC) > 0


def test_charge_and_discharge_windows_agree_on_a_symmetric_cell():
    """Same capacity, same window, opposite legs: the charge moved matches.

    The two legs sweep the window in opposite directions, so this is the test
    that the magnitude convention holds rather than one leg being negated.
    """
    v_dis, q_dis = _curve(capacity_ah=1.10, v_start=4.20, v_end=3.00)
    v_chg, q_chg = _charge_curve(capacity_ah=1.10, v_start=3.00, v_end=4.20)
    assert window_charge(v_dis, q_dis, SPEC) == pytest.approx(
        window_charge(v_chg, q_chg, SPEC), rel=0.02)


def test_a_faded_charge_curve_reads_its_fade():
    v, full = _charge_curve(capacity_ah=1.10)
    _, faded = _charge_curve(capacity_ah=0.88)
    ratio = window_charge(v, faded, SPEC) / window_charge(v, full, SPEC)
    assert ratio == pytest.approx(0.80, rel=0.01)


# ---------------------------------------------------------------------------
# The physical plausibility gate
# ---------------------------------------------------------------------------

def _shape_shifted_cycle(capacity_ah: float = 1.10, n: int = 400):
    """Same TOTAL charge, but almost none of it inside the window.

    Charge is delivered steeply outside 3.90-3.60 V and nearly flat across it,
    so the window reading collapses while the discharge is full length. That
    separates the shape failure from a truncated cycle, which the partial
    discharge check catches instead.
    """
    voltage = np.linspace(4.15, 2.70, n)
    frac = np.where(voltage > SPEC.v_high, (4.15 - voltage) / (4.15 - SPEC.v_high) * 0.49,
           np.where(voltage > SPEC.v_low, 0.49 + (SPEC.v_high - voltage) /
                    (SPEC.v_high - SPEC.v_low) * 0.02,
                    0.51 + (SPEC.v_low - voltage) / (SPEC.v_low - 2.70) * 0.49))
    return voltage, frac * capacity_ah


def test_an_implausible_ratio_is_withheld_not_clipped():
    """A value below the floor must vanish, not be rounded into range.

    Clipping to MIN_PLAUSIBLE_SOH would put a wrong number inside the band a
    reader expects, which is harder to notice than a missing one.
    """
    blocks = []
    for cycle, cap in enumerate([1.10] * 5, start=1):
        v, q = _curve(cap)
        blocks.append(pd.DataFrame({"cell_id": "A", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    v, q = _shape_shifted_cycle(1.10)          # full discharge, dead window
    blocks.append(pd.DataFrame({"cell_id": "A", "cycle": 6,
                                "voltage_v": v, "capacity_ah_curve": q}))
    table = window_soh_table(pd.concat(blocks, ignore_index=True),
                             spec=SPEC, reference_cycles=5)
    last = table[table["cycle"] == 6].iloc[0]
    assert not last["partial_discharge"], "this cycle is full length"
    assert last["implausible"]
    assert np.isnan(last["soh_window_accepted"])
    assert last["soh_window"] < MIN_PLAUSIBLE_SOH   # raw value still inspectable
    assert "scrap" in last["refusal"]


def test_a_truncated_cycle_is_refused_as_partial_not_as_fade():
    """The failure CS2_9 exposed: a cut-short cycle reading as a dead cell."""
    capacities = [1.10] * 5 + [0.05]
    curves = _frame({"A": capacities})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    last = table[table["cycle"] == 6].iloc[0]
    assert last["partial_discharge"]
    assert np.isnan(last["soh_window"]), (
        "a truncated cycle must yield no state of health at all"
    )
    assert "truncated cycle is not a faded cell" in last["refusal"]


def test_real_fade_above_the_floor_is_still_measured():
    """The partial-discharge check must not swallow genuine degradation."""
    capacities = [1.10] * 5 + [0.88]           # 80% SOH, the retirement point
    curves = _frame({"A": capacities})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    last = table[table["cycle"] == 6].iloc[0]
    assert not last["partial_discharge"]
    assert last["soh_window_accepted"] == pytest.approx(0.80, rel=0.02)


def test_a_healthy_cell_passes_the_gate_untouched():
    capacities = [1.10] * 5 + [1.05, 1.00, 0.95, 0.90]
    curves = _frame({"A": capacities})
    table = window_soh_table(curves, spec=SPEC, reference_cycles=5)
    assert not table["implausible"].any()
    assert not table["cell_refused"].any()
    assert table["soh_window_accepted"].notna().sum() == len(capacities)


def test_a_cell_mostly_implausible_is_refused_entirely():
    """Trimming the bad rows and scoring the rest would flatter the method.

    The surviving rows would be the early ones, which is exactly where the
    estimator had not yet failed.
    """
    blocks = []
    for cycle, cap in enumerate([1.10] * 5, start=1):
        v, q = _curve(cap)
        blocks.append(pd.DataFrame({"cell_id": "A", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    for cycle in range(6, 16):
        v, q = _shape_shifted_cycle(1.10)
        blocks.append(pd.DataFrame({"cell_id": "A", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    table = window_soh_table(pd.concat(blocks, ignore_index=True),
                             spec=SPEC, reference_cycles=5)
    assert table["cell_refused"].all()
    assert table["soh_window_accepted"].isna().all()
    assert "whole cell is refused" in table.iloc[-1]["refusal"]


def test_one_bad_cell_does_not_refuse_its_neighbour():
    """The cell-level verdict must be per cell, not pooled across the frame."""
    blocks = []
    for cycle, cap in enumerate([1.10] * 5 + [1.0] * 5, start=1):
        v, q = _curve(cap)
        blocks.append(pd.DataFrame({"cell_id": "good", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    for cycle, cap in enumerate([1.10] * 5, start=1):
        v, q = _curve(cap)
        blocks.append(pd.DataFrame({"cell_id": "bad", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    for cycle in range(6, 16):
        v, q = _shape_shifted_cycle(1.10)
        blocks.append(pd.DataFrame({"cell_id": "bad", "cycle": cycle,
                                    "voltage_v": v, "capacity_ah_curve": q}))
    table = window_soh_table(pd.concat(blocks, ignore_index=True),
                             spec=SPEC, reference_cycles=5)
    assert not table[table["cell_id"] == "good"]["cell_refused"].any()
    assert table[table["cell_id"] == "bad"]["cell_refused"].all()


def test_the_floor_sits_below_automotive_retirement():
    """The gate must not fire on cells that are merely old.

    Retirement is conventionally 80% state of health; a floor at or above that
    would refuse every genuinely aged cell the estimator is meant to serve.
    """
    assert MIN_PLAUSIBLE_SOH < 0.80
