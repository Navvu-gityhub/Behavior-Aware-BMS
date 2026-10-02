"""Remaining useful life by extrapolating a cell's own fade trajectory.

WHY THIS AND NOT THE EXISTING ESTIMATOR
----------------------------------------
`rul_estimation.compute_rul` multiplies a hand-picked weighting of health,
temperature, deep-discharge and fast-charge exposure by `base_cycle_life =
1000`, a constant its own docstring labels unvalidated. That docstring also
names the fix: validate against capacity-fade-to-threshold ground truth, which
CALCE provides. This module is that, done a different way.

It fits nothing across cells. At cycle n it takes the SOH history of THIS cell
up to n, fits a local fade trend, and extrapolates to the end-of-life
threshold. RUL is the distance from n to the crossing. No other cell's
coefficients enter, so there is no training cohort and no cross-protocol
transfer step - the same reason `health/voltage_window.py` has none.

WHAT IT CANNOT PREDICT, AND WHAT THE ERROR ACTUALLY DOES
----------------------------------------------------------
Lithium-ion capacity fade is not linear, so a trend fit cannot see far. The
error is therefore a function of how far ahead the estimate reaches, and
`rul_error_table` reports it against horizon rather than averaging it into one
number. Measured on CALCE at threshold 0.90:

    true RUL      median |error|    within +/-20
    0-25 cycles        9 cycles          88%
    25-50             16                 59%
    50-100            31                 21%
    200-400          146                  3%

The estimate is useful near end of life and not useful far from it. That is
the honest shape of the problem and the scope any claim must carry.

The bias is NEGATIVE at every horizon - -4 cycles near end of life, -139 at
200-400 - meaning the estimate places end of life EARLIER than it occurs. That
was worth measuring rather than assuming: the textbook expectation is the
opposite, that a pre-knee trend fit over-predicts life because the bend has
not happened yet. These CS2/CX2 cells fade quickly early and then flatten, so
the fitted slope is steeper than the life-average one and the estimate is
conservative instead.

Conservative is the right direction for a replacement decision. An estimate
that retires a cell early costs money; one that retires it late is a failure
in service. Nothing here tunes the bias away, because removing it would trade
the safe error for the unsafe one.

WHY A PLAIN LINE, WHEN FADE IS NOT A LINE
------------------------------------------
Three forms were compared on the same folds - a line, a square root (the
diffusion-limited SEI form), and a quadratic - at both full history and a
200-cycle trailing window. Within 25 cycles of end of life:

    form              window     median |error|    within +/-20
    linear            full            9.1               83%
    quadratic         200            14.2               56%
    linear            200            25.3               45%
    square root       full           43.7               14%

The line on full history wins, and the physically-motivated square root is the
worst of them. A trailing window is worse than full history in every form:
following the trajectory as it steepens buys less than averaging the
measurement noise costs. So the default is the plain line over everything,
chosen by measurement and not by which story sounds better.

THE SMOOTHING IS NOT COSMETIC
-----------------------------
Measured SOH carries cycle-to-cycle noise of a few tenths of a percent, from
temperature drift and coulomb-counting error. A slope fitted to raw SOH over a
short window is dominated by that noise, and a slope near zero extrapolates to
an end of life centuries away. The rolling median removes it without shifting
the trend the way a mean would when one cycle reads badly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# End-of-life threshold in state of health. 0.80 is the industry convention
# for automotive retirement, and it is NOT the default here, because the CALCE
# trajectories end at roughly 0.81 - one cell of 22 ever crosses it. A
# threshold with no data beyond it cannot be validated, so the default is one
# that can be, and the convention is offered as a parameter.
DEFAULT_EOL_THRESHOLD = 0.90

# Trailing cycles the trend is fitted over. None means full history, which the
# form comparison in the module docstring measured as better than any window
# tried: a 200-cycle window halves the near-end-of-life hit rate, from 83% to
# 45% within +/-20. The parameter stays so that finding can be re-run rather
# than taken on trust.
DEFAULT_WINDOW: int | None = None

# Rolling median width applied before fitting. Odd so the window is centred.
DEFAULT_SMOOTH = 11

# Below this many smoothed points no estimate is issued. A slope from three
# noisy readings is a number, not a trend.
MIN_HISTORY = 30

# An estimate may not reach further ahead than this multiple of the history it
# was fitted on.
#
# WHY A HORIZON BOUND, AND HOW IT WAS FOUND
# ------------------------------------------
# A trend fitted to a nearly-flat early trajectory solves for an end of life
# arbitrarily far away: at 1e-5 SOH per cycle a cell "reaches" 0.90 some ten
# thousand cycles out. `MIN_SLOPE` alone does not catch that, because the slope
# is finite and the arithmetic is sound - the estimate is simply reaching
# further than the evidence can see.
#
# CALCE hid this. Its cells run 500-1700 cycles, so by the time an estimate is
# issued there is enough history for the slope to be real. Running the same
# estimator on NASA, whose cells average about 79 cycles, produced a median
# absolute error of 13 cycles alongside a MEAN of 1,699 - the signature of a
# few estimates reaching into the tens of thousands.
#
# So: an extrapolation that reaches more than twice as far as the data behind
# it is refused rather than reported. Two is a judgement, not a measurement;
# what is measured is that some bound is needed, and that it is the tail rather
# than the median that needs it.
MAX_HORIZON_RATIO = 2.0

# A fade slope flatter than this is treated as no measurable fade rather than
# extrapolated. At 1e-6 SOH per cycle a cell would take 100,000 cycles to lose
# 10%, so the estimate is meaningless long before it is wrong.
MIN_SLOPE = 1e-6


@dataclass(frozen=True)
class RULEstimate:
    """One estimate, with the reason when there is no number."""

    cycle: int
    rul_cycles: float
    eol_cycle: float
    slope_per_cycle: float
    n_history: int
    refusal: str = ""


def observed_eol(
    cycles: np.ndarray,
    soh: np.ndarray,
    threshold: float = DEFAULT_EOL_THRESHOLD,
    smooth: int = DEFAULT_SMOOTH,
) -> float:
    """First cycle whose smoothed SOH is at or below `threshold`.

    Smoothed, because a single noisy discharge dipping under the threshold
    would otherwise declare end of life hundreds of cycles early and become
    the ground truth every prediction is scored against.

    Returns NaN when the cell never crosses, which must then be excluded from
    scoring rather than treated as "survives forever".
    """
    series = pd.Series(soh).rolling(smooth, center=True, min_periods=3).median()
    below = np.asarray(cycles)[series.to_numpy() <= threshold]
    return float(below[0]) if len(below) else float("nan")


def estimate_rul(
    cycles: np.ndarray,
    soh: np.ndarray,
    at_cycle: int,
    threshold: float = DEFAULT_EOL_THRESHOLD,
    window: int | None = DEFAULT_WINDOW,
    smooth: int = DEFAULT_SMOOTH,
    min_history: int = MIN_HISTORY,
    max_horizon_ratio: float = MAX_HORIZON_RATIO,
) -> RULEstimate:
    """RUL at `at_cycle`, using only this cell's history up to that cycle.

    The history cut is the whole point: an estimate that saw cycle n+1 would
    be scored against a future it had already been shown.
    """
    cycles = np.asarray(cycles, dtype=float)
    soh = np.asarray(soh, dtype=float)

    past = cycles <= at_cycle
    x, y = cycles[past], soh[past]
    if len(x) < min_history:
        return RULEstimate(at_cycle, float("nan"), float("nan"), float("nan"),
                           len(x), f"fewer than {min_history} cycles of history")

    smoothed = pd.Series(y).rolling(smooth, center=True, min_periods=3).median()
    y = smoothed.to_numpy()
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < min_history:
        return RULEstimate(at_cycle, float("nan"), float("nan"), float("nan"),
                           len(x), "too few finite points after smoothing")

    if window is not None and len(x) > window:
        x, y = x[-window:], y[-window:]

    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or -slope < MIN_SLOPE:
        return RULEstimate(
            at_cycle, float("nan"), float("nan"), float(slope), len(x),
            "no measurable fade in the recent window; a flat trend "
            "extrapolates to an arbitrary end of life",
        )

    eol = (threshold - intercept) / slope
    if not np.isfinite(eol) or eol <= at_cycle:
        # The threshold is already behind us on the fitted line. Reporting a
        # negative RUL would be arithmetic; zero is the honest statement.
        return RULEstimate(at_cycle, 0.0, float(eol), float(slope), len(x),
                           "fitted trend places end of life at or before now")

    horizon = eol - at_cycle
    observed = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    if observed > 0 and horizon > max_horizon_ratio * observed:
        return RULEstimate(
            at_cycle, float("nan"), float(eol), float(slope), len(x),
            f"extrapolation reaches {horizon:.0f} cycles ahead on "
            f"{observed:.0f} cycles of history, beyond "
            f"{max_horizon_ratio:g}x; the trend cannot see that far",
        )

    return RULEstimate(at_cycle, float(horizon), float(eol),
                       float(slope), len(x))


def rul_error_table(
    frame: pd.DataFrame,
    threshold: float = DEFAULT_EOL_THRESHOLD,
    window: int | None = DEFAULT_WINDOW,
    smooth: int = DEFAULT_SMOOTH,
    stride: int = 25,
    min_history: int = MIN_HISTORY,
    max_horizon_ratio: float = MAX_HORIZON_RATIO,
) -> pd.DataFrame:
    """Score RUL estimates along every cell's life.

    `frame` needs `cell_id`, `cycle` and `soh`. An estimate is issued every
    `stride` cycles from `min_history` onward, and scored against the cell's
    observed crossing. Cells that never cross are excluded, with a reason,
    rather than counted as correct.
    """
    required = {"cell_id", "cycle", "soh"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"rul_error_table: missing {sorted(missing)}")

    rows: list[dict] = []
    for cell_id, block in frame.groupby("cell_id", sort=True):
        block = block.sort_values("cycle")
        block = block[block["soh"].notna()]
        cycles = block["cycle"].to_numpy(dtype=float)
        soh = block["soh"].to_numpy(dtype=float)
        if len(cycles) < min_history:
            continue

        true_eol = observed_eol(cycles, soh, threshold, smooth)
        cohort = block["cohort"].iloc[0] if "cohort" in block.columns else ""
        if not np.isfinite(true_eol):
            rows.append({
                "cell_id": cell_id, "cohort": cohort, "cycle": np.nan,
                "rul_pred": np.nan, "rul_true": np.nan, "error": np.nan,
                "refusal": f"never crosses SOH {threshold}",
            })
            continue

        for at in range(int(min_history), int(true_eol), stride):
            est = estimate_rul(cycles, soh, at, threshold, window, smooth,
                               min_history, max_horizon_ratio)
            rows.append({
                "cell_id": cell_id,
                "cohort": cohort,
                "cycle": at,
                "rul_pred": est.rul_cycles,
                "rul_true": true_eol - at,
                "error": est.rul_cycles - (true_eol - at),
                "slope_per_cycle": est.slope_per_cycle,
                "refusal": est.refusal,
            })

    return pd.DataFrame(rows)


def rul_from_cycle_capacity(
    cycles: pd.DataFrame,
    threshold: float = DEFAULT_EOL_THRESHOLD,
    reference_cycles: int = 5,
    **kwargs,
) -> RULEstimate:
    """RUL at the newest cycle, from a cycle frame carrying `capacity_ah`.

    This is the adapter that lets the telemetry pipeline reach the validated
    estimator. The pipeline segments a log into cycles and each carries its
    delivered charge, so a state-of-health trajectory is available there even
    though the per-battery summary it scores on has thrown it away.

    State of health is each cycle's capacity over the median of the first
    `reference_cycles` measurable ones - the cell's own early life, not a
    nameplate. A pack whose first cycles are already degraded will therefore
    read healthy, which is the same limitation `health/voltage_window.py`
    carries and for the same reason: without a beginning-of-life reference
    there is nothing to be relative to.

    Returns a refusal rather than a number when the log is too short, the
    trend is flat, or the extrapolation reaches past what the history supports.
    A short bench capture will refuse, and that is correct.
    """
    required = {"cycle", "capacity_ah"}
    missing = required - set(cycles.columns)
    if missing:
        raise ValueError(
            f"rul_from_cycle_capacity: missing {sorted(missing)}"
        )

    frame = cycles.dropna(subset=["cycle", "capacity_ah"]).sort_values("cycle")
    if frame.empty:
        return RULEstimate(0, float("nan"), float("nan"), float("nan"), 0,
                           "no cycle carried a capacity measurement")

    capacity = pd.to_numeric(frame["capacity_ah"], errors="coerce")
    reference = float(capacity.head(reference_cycles).median())
    if not np.isfinite(reference) or reference <= 0:
        return RULEstimate(0, float("nan"), float("nan"), float("nan"),
                           len(frame),
                           "no usable beginning-of-life capacity reference")

    cycle_numbers = frame["cycle"].to_numpy(dtype=float)
    return estimate_rul(
        cycle_numbers,
        (capacity / reference).to_numpy(dtype=float),
        at_cycle=int(cycle_numbers[-1]),
        threshold=threshold,
        **kwargs,
    )

