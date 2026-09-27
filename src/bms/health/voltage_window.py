"""State of health from charge delivered in a fixed voltage window.

WHAT THIS IS FOR
----------------
Every model in this project estimates SOH by fitting something. Fitting is why
the leave-one-cohort-out results collapse: a coefficient learned on one
protocol is a statement about that protocol. This module does not fit
anything.

The quantity is

    SOH_hat(n) = Q_window(n) / Q_window(reference)

where `Q_window` is the charge the cell delivers between two fixed terminal
voltages. It is a ratio of two measurements of the SAME cell, so there is no
coefficient to transfer, no training cohort, and no cross-protocol
generalisation step that could fail. A different chemistry changes which
window is sensible; it does not change the method.

WHY A WINDOW RATHER THAN THE WHOLE DISCHARGE
--------------------------------------------
Capacity is defined over a full discharge, and a vehicle almost never performs
one. Drivers charge to 80% and plug in at 40%, so the pack's controller sees
partial discharges and has no direct capacity measurement to work from. That
is the practical reason SOH is estimated rather than measured in the field.

A voltage window survives that. If the discharge passes through 3.9 V down to
3.6 V, the charge delivered across that span is measurable by coulomb counting
whatever happened before or after it, and it shrinks as the cell ages. So the
estimator runs on the data a real pack actually produces.

WHAT IT ASSUMES, AND WHERE THAT BREAKS
---------------------------------------
It assumes the discharge curve scales roughly uniformly as the cell ages, so
that charge in one voltage span falls in proportion to total capacity. That is
an approximation, and it is the method's weak point rather than a detail:

- Loss of lithium inventory SHIFTS the curve along the capacity axis, which a
  window ratio partly absorbs.
- Loss of active material changes the plateau SHAPE, which it does not, so the
  ratio drifts away from the capacity ratio as that mode dominates.
- Resistance growth depresses terminal voltage under load, which moves the
  window boundaries relative to the cell's true state. At a fixed C-rate this
  is a systematic bias, not noise.

So the estimator is expected to track SOH with a slope near one and some bias,
NOT to equal it. `window_soh_table` reports the realised slope and bias per
cell so that assumption is measured rather than asserted.

WHAT IT NEEDS THAT THE FITTED MODELS DO NOT
--------------------------------------------
One reference measurement of the same cell early in its life. A pack that has
never been characterised cannot use this. That is a real constraint, and it is
also what production BMS firmware already stores at manufacture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.benchmarks.curves import interpolate_curve

# Default window for LCO/NMC 18650 and pouch cells. Chosen inside the plateau
# rather than at the knees: above ~4.05 V and below ~3.4 V the curve is steep,
# so a small voltage error there moves the charge reading a long way, and a
# partial discharge is less likely to reach them anyway.
DEFAULT_V_HIGH = 3.90
DEFAULT_V_LOW = 3.60

# Cycles averaged to form the beginning-of-life reference. A single cycle
# carries the full measurement noise of one discharge into every subsequent
# estimate, because it divides all of them.
DEFAULT_REFERENCE_CYCLES = 5

# A cycle must span at least this fraction of the requested window to be
# measured. Below it the discharge did not really traverse the window and the
# interpolation would be reading mostly extrapolated nothing.
MIN_WINDOW_COVERAGE = 0.95


@dataclass(frozen=True)
class WindowSpec:
    """The voltage window a run used, carried with its results."""

    v_high: float = DEFAULT_V_HIGH
    v_low: float = DEFAULT_V_LOW

    @property
    def span_v(self) -> float:
        return self.v_high - self.v_low

    def __str__(self) -> str:
        return f"{self.v_high:.2f}-{self.v_low:.2f} V"


def window_charge(
    voltage: np.ndarray,
    capacity: np.ndarray,
    spec: WindowSpec = WindowSpec(),
    min_coverage: float = MIN_WINDOW_COVERAGE,
) -> float:
    """Charge delivered between `v_high` and `v_low` on one discharge.

    Returns NaN when the discharge does not cover the window. Refusing is the
    point: a cycle that stopped at 3.7 V has no charge to report for the
    3.7-3.6 V portion, and returning the charge it did deliver would silently
    report a partial window as a full one and read as capacity fade.
    """
    voltage = np.asarray(voltage, dtype=float)
    capacity = np.asarray(capacity, dtype=float)
    finite = np.isfinite(voltage) & np.isfinite(capacity)
    if finite.sum() < 2:
        return float("nan")
    voltage, capacity = voltage[finite], capacity[finite]

    covered = min(voltage.max(), spec.v_high) - max(voltage.min(), spec.v_low)
    if covered < min_coverage * spec.span_v:
        return float("nan")

    grid = np.array([spec.v_low, spec.v_high], dtype=float)
    q = interpolate_curve(voltage, capacity, grid)
    if not np.all(np.isfinite(q)):
        return float("nan")
    # Q accumulates as voltage falls, so the low-voltage end holds more charge.
    return float(q[0] - q[1])


def window_charge_by_cycle(
    curves: pd.DataFrame,
    spec: WindowSpec = WindowSpec(),
    min_coverage: float = MIN_WINDOW_COVERAGE,
) -> pd.DataFrame:
    """One row per (cell, cycle): the window charge, or NaN with a reason."""
    required = {"cell_id", "cycle", "voltage_v", "capacity_ah_curve"}
    missing = required - set(curves.columns)
    if missing:
        raise ValueError(
            f"window_charge_by_cycle: missing {sorted(missing)}. Build the "
            f"frame with load_calce_curves.extract_discharge_curves."
        )

    rows: list[dict] = []
    group_cols = ["cell_id", "cycle"]
    for (cell_id, cycle), block in curves.groupby(group_cols, sort=True):
        record = {
            "cell_id": cell_id,
            "cycle": int(cycle),
            "window_charge_ah": window_charge(
                block["voltage_v"].to_numpy(dtype=float),
                block["capacity_ah_curve"].to_numpy(dtype=float),
                spec, min_coverage,
            ),
        }
        if "cohort" in block.columns:
            record["cohort"] = block["cohort"].iloc[0]
        rows.append(record)
    return pd.DataFrame(rows)


def window_soh_table(
    curves: pd.DataFrame,
    truth: pd.DataFrame | None = None,
    spec: WindowSpec = WindowSpec(),
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
    min_coverage: float = MIN_WINDOW_COVERAGE,
) -> pd.DataFrame:
    """Per-cycle SOH estimated from the window ratio, optionally scored.

    `truth`, when given, must carry `cell_id`, `cycle` and `soh`. It is joined
    only to score the estimate; nothing about it enters the estimate, which is
    what makes the comparison meaningful rather than circular.

    The reference is the MEDIAN window charge of the first `reference_cycles`
    measurable cycles of that cell. Median rather than mean because a single
    anomalous early discharge would otherwise scale every later estimate of
    that cell.
    """
    charges = window_charge_by_cycle(curves, spec, min_coverage)

    frames: list[pd.DataFrame] = []
    for _cell_id, block in charges.groupby("cell_id", sort=True):
        block = block.sort_values("cycle").copy()
        usable = block[block["window_charge_ah"].notna()]
        if usable.empty:
            block["window_reference_ah"] = float("nan")
            block["soh_window"] = float("nan")
            frames.append(block)
            continue
        reference = float(
            usable["window_charge_ah"].head(reference_cycles).median()
        )
        block["window_reference_ah"] = reference
        block["soh_window"] = (
            block["window_charge_ah"] / reference if reference > 0
            else float("nan")
        )
        frames.append(block)

    table = pd.concat(frames, ignore_index=True)
    table["v_high"] = spec.v_high
    table["v_low"] = spec.v_low

    if truth is not None:
        keep = [c for c in ("cell_id", "cycle", "soh") if c in truth.columns]
        table = table.merge(truth[keep], on=["cell_id", "cycle"], how="left")
        table["error"] = table["soh_window"] - table["soh"]

    return table


def per_cell_fit(table: pd.DataFrame) -> pd.DataFrame:
    """Realised slope and bias of the estimate against measured SOH, per cell.

    The method assumes the window ratio falls in proportion to capacity. A
    slope of one and a bias of zero would mean that assumption held exactly.
    It will not, and the size of the departure is the honest measure of how
    far the uniform-scaling approximation can be pushed - which is why this is
    reported per cell rather than pooled into one number that hides the spread.
    """
    if "soh" not in table.columns:
        raise ValueError("per_cell_fit: table carries no `soh` column to fit against")

    rows: list[dict] = []
    for cell_id, block in table.groupby("cell_id", sort=True):
        ok = block[["soh_window", "soh"]].dropna()
        if len(ok) < 10 or np.ptp(ok["soh"].to_numpy()) <= 0:
            rows.append({"cell_id": cell_id, "n": len(ok),
                         "slope": float("nan"), "bias": float("nan"),
                         "mae": float("nan"), "spread_soh": float("nan")})
            continue
        slope, bias = np.polyfit(ok["soh"], ok["soh_window"], 1)
        rows.append({
            "cell_id": cell_id,
            "n": len(ok),
            "slope": float(slope),
            "bias": float(bias),
            "mae": float(np.mean(np.abs(ok["soh_window"] - ok["soh"]))),
            "spread_soh": float(np.ptp(ok["soh"].to_numpy())),
        })
    return pd.DataFrame(rows)
