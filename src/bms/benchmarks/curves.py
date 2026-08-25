"""Curve-based methods: Severson's Delta-Q variance and incremental capacity analysis.

These are the two most-cited feature families in data-driven battery
prognostics, and neither can run on the frame this project currently retains.
That is the point of implementing them here rather than omitting them.

WHY IMPLEMENT METHODS THAT CANNOT RUN
-------------------------------------
A benchmark table listing only the methods that happened to be computable
reads as a complete survey. It is not one, and the omission is invisible to
the reader. Registering these with `requires_curves=True` makes the gap
appear *in the results table* as an explicit UNAVAILABLE row with a reason,
which is the same discipline the rest of this project applies to missing data.

They are also not stubs. The feature computations below are the real ones, so
the moment a curve-level dataset lands — CALCE CS2/CX2 Arbin exports and the
Oxford drive-cycle set both carry per-cycle voltage/capacity traces — these
methods run without further work.

THE DELTA-Q FEATURE, AND WHY IT IS NOT A CAPACITY FEATURE
---------------------------------------------------------
Severson et al. (Nature Energy, 2019) found that the *variance of the
difference between two discharge capacity-voltage curves*, taken early in
life, predicts eventual cycle life far better than capacity fade over the
same window. The quantity is

    Delta_Q(V) = Q_cycle_b(V) - Q_cycle_a(V)

evaluated on a common voltage grid, with the feature being
log10(var(Delta_Q(V))).

The reason this matters for *this* project's thesis is specific. Severson's
result was established on 124 LFP cells in a temperature-controlled chamber
where the experimental axis was charge policy. This project's NASA frame varies
ambient temperature and holds charge rate fixed. Those are orthogonal designs,
which `adaptive/dataset_specs.py` already records — so a Delta-Q model trained
on one is not merely untested on the other, it is structurally under-determined
there. Running it under leave-one-cohort-out is how that claim gets a number
instead of an argument.

EXPECTED CURVE FORMAT
---------------------
A long frame, one row per (cell, cycle, sample point):

    cell_id | cycle | voltage_v | capacity_ah_curve

`capacity_ah_curve` is cumulative discharge capacity at that voltage, not the
cycle-level scalar `capacity_ah` used elsewhere in this codebase. The distinct
name is deliberate: silently overloading `capacity_ah` to sometimes mean a
per-cycle total and sometimes a within-cycle trace is exactly the kind of unit
ambiguity that produced the CALCE "initial vs post-storage capacity" artifact
documented in docs/calce_dataset_note.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import FitFn
from src.bms.benchmarks.registry import BenchmarkMethod, BuildFn, Family, register

CURVE_COLUMNS: tuple[str, ...] = ("voltage_v", "capacity_ah_curve")

# Discharge voltage window for the interpolation grid. Chosen wide enough to
# cover LCO (NASA/CALCE, ~2.0-4.2 V) and NMC/LCO pouch (Oxford, ~2.7-4.2 V).
# A per-chemistry window belongs on the DatasetSpec, not hard-coded here, once
# more than one chemistry is actually loaded.
DEFAULT_VOLTAGE_GRID = np.linspace(2.0, 4.2, 1000)


def interpolate_curve(
    voltage: np.ndarray,
    capacity: np.ndarray,
    grid: np.ndarray = DEFAULT_VOLTAGE_GRID,
) -> np.ndarray:
    """Put one cycle's Q(V) curve onto a common voltage grid.

    Returns NaN outside the cycle's own measured voltage range rather than
    extrapolating. A discharge curve that stopped at 3.2 V carries no
    information about what capacity would have been at 2.0 V, and filling it
    with a boundary value would manufacture a Delta-Q difference between two
    cycles that merely ended at different cutoffs.
    """
    order = np.argsort(voltage)
    v_sorted = np.asarray(voltage, dtype=float)[order]
    q_sorted = np.asarray(capacity, dtype=float)[order]

    keep = np.isfinite(v_sorted) & np.isfinite(q_sorted)
    v_sorted, q_sorted = v_sorted[keep], q_sorted[keep]
    if len(v_sorted) < 2:
        return np.full(len(grid), np.nan)

    # np.interp clamps outside the range, so mask explicitly afterwards.
    out = np.interp(grid, v_sorted, q_sorted)
    out[(grid < v_sorted[0]) | (grid > v_sorted[-1])] = np.nan
    return out


def delta_q(
    curves: pd.DataFrame,
    cycle_a: int,
    cycle_b: int,
    grid: np.ndarray = DEFAULT_VOLTAGE_GRID,
) -> np.ndarray:
    """Q_cycle_b(V) - Q_cycle_a(V) for one cell, on a common voltage grid."""
    missing = [c for c in (*CURVE_COLUMNS, "cycle") if c not in curves.columns]
    if missing:
        raise ValueError(f"delta_q: missing required columns {missing}")

    def curve_for(cycle: int) -> np.ndarray:
        rows = curves[curves["cycle"] == cycle]
        if rows.empty:
            raise KeyError(
                f"delta_q: cycle {cycle} is absent for this cell. Severson's "
                f"feature is defined on two specific cycles; substituting the "
                f"nearest available one changes what the feature measures and "
                f"is left to the caller to decide explicitly."
            )
        return interpolate_curve(
            rows["voltage_v"].to_numpy(dtype=float),
            rows["capacity_ah_curve"].to_numpy(dtype=float),
            grid,
        )

    return curve_for(cycle_b) - curve_for(cycle_a)


def delta_q_variance_feature(
    curves: pd.DataFrame,
    cycle_a: int = 10,
    cycle_b: int = 100,
    grid: np.ndarray = DEFAULT_VOLTAGE_GRID,
) -> float:
    """log10(var(Delta_Q(V))) — the Severson 'variance' model's single feature.

    Returns NaN when the two cycles share too little voltage overlap to
    estimate a variance, rather than a variance computed over a handful of
    points, which would be numerically defined and scientifically meaningless.
    """
    difference = delta_q(curves, cycle_a, cycle_b, grid)
    usable = difference[np.isfinite(difference)]
    if len(usable) < 50:
        return float("nan")
    variance = float(np.var(usable))
    return float(np.log10(variance)) if variance > 0 else float("nan")


def incremental_capacity(
    voltage: np.ndarray,
    capacity: np.ndarray,
    grid: np.ndarray = DEFAULT_VOLTAGE_GRID,
    smoothing_window: int = 51,
) -> np.ndarray:
    """dQ/dV on a common voltage grid, Savitzky-Golay smoothed.

    Incremental capacity analysis reads phase transitions off the peaks of
    dQ/dV. The derivative of raw cycler data is dominated by quantisation
    noise, so smoothing is not cosmetic — without it the peak-finding step
    returns noise maxima. The window is reported as a parameter because peak
    height depends on it, and two ICA studies with different windows are not
    directly comparable.
    """
    from scipy.signal import savgol_filter

    q = interpolate_curve(voltage, capacity, grid)
    finite = np.isfinite(q)
    if finite.sum() < smoothing_window:
        return np.full(len(grid), np.nan)

    out = np.full(len(grid), np.nan)
    v_valid, q_valid = grid[finite], q[finite]
    window = min(smoothing_window, len(q_valid) - (1 - len(q_valid) % 2))
    if window < 5:
        return out
    smoothed = savgol_filter(q_valid, window_length=window, polyorder=3)
    out[finite] = np.gradient(smoothed, v_valid)
    return out


def ica_peak_features(
    voltage: np.ndarray,
    capacity: np.ndarray,
    grid: np.ndarray = DEFAULT_VOLTAGE_GRID,
) -> dict[str, float]:
    """Peak height, peak voltage, and curve area from a dQ/dV trace."""
    dqdv = incremental_capacity(voltage, capacity, grid)
    finite = np.isfinite(dqdv)
    if finite.sum() < 10:
        return {"ica_peak_height": float("nan"),
                "ica_peak_voltage": float("nan"),
                "ica_area": float("nan")}

    values, positions = dqdv[finite], grid[finite]
    peak = int(np.argmax(np.abs(values)))
    return {
        "ica_peak_height": float(values[peak]),
        "ica_peak_voltage": float(positions[peak]),
        "ica_area": float(np.trapz(np.abs(values), positions)),
    }


def _unavailable_build(reason: str) -> BuildFn:
    """A build function that refuses, for methods with no summary-frame form.

    Registered methods must expose a `build`, and returning a `FitFn` that
    raises keeps the refusal at the point of use with a full explanation,
    rather than producing a predictor that quietly emits NaN and shows up in
    the results table as a very bad score.
    """
    def build(features: Sequence[str], target: str) -> FitFn:
        def fit(train: pd.DataFrame):
            raise NotImplementedError(reason)
        return fit
    return build


register(BenchmarkMethod(
    name="severson_delta_q_variance",
    family=Family.CURVE,
    citation=(
        "Severson et al., Data-driven prediction of battery cycle life before "
        "capacity degradation, Nature Energy 4, 383-391 (2019)."
    ),
    build=_unavailable_build(
        "severson_delta_q_variance needs per-cycle discharge Q(V) curves "
        "(columns 'voltage_v', 'capacity_ah_curve'). Compute the feature with "
        "delta_q_variance_feature() once a curve-level frame is loaded, then "
        "register a concrete variant over the resulting column."
    ),
    default_features=("delta_q_variance",),
    requires_curves=True,
    assumes_target="log_cycle_life",
    note=(
        "Established on 124 LFP cells varying charge policy at fixed 30 C. "
        "NASA varies ambient temperature at fixed charge rate, so the designs "
        "are orthogonal (adaptive/dataset_specs.py). Feature math is "
        "implemented and tested; only the data is missing."
    ),
))

register(BenchmarkMethod(
    name="ica_peak_features",
    family=Family.CURVE,
    citation=(
        "Dubarry et al., J. Power Sources 219 (2012) 204-216; incremental "
        "capacity analysis for degradation-mode identification."
    ),
    build=_unavailable_build(
        "ica_peak_features needs per-cycle Q(V) curves to differentiate. Use "
        "ica_peak_features() to build the columns from a curve-level frame."
    ),
    default_features=("ica_peak_height", "ica_peak_voltage", "ica_area"),
    requires_curves=True,
    note=(
        "Peak height and position track loss of active material and loss of "
        "lithium inventory separately, which is the physical interpretability "
        "the behavioural scores in this project lack."
    ),
))

register(BenchmarkMethod(
    name="sequence_model",
    family=Family.CURVE,
    citation="Stand-in for LSTM/CNN sequence models over raw cycle traces.",
    build=_unavailable_build(
        "sequence_model needs per-cycle time series, not cycle-level "
        "aggregates. The tabular neural network registered as 'mlp' in "
        "classical.py is NOT a substitute and is labelled accordingly."
    ),
    default_features=(),
    requires_curves=True,
    note=(
        "Registered so the results table shows the deep-learning family as "
        "untested-here rather than absent. 'mlp' covers tabular features only."
    ),
))
