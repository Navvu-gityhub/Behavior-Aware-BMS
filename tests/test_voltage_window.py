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
