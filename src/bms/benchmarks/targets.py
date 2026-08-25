"""Target definitions, and how much signal each one actually contains.

WHY THE TARGET IS A VARIABLE, NOT A CONSTANT
--------------------------------------------
This project's calibration work reported R-squared near zero for every
candidate against `capacity_loss`, the per-cycle change in measured discharge
capacity. That is a real result about that target. It is not automatically a
result about degradation prediction, because a per-cycle capacity delta is a
difference of two noisy measurements and inherits both their errors while
cancelling most of the underlying trend.

The numbers make the concern concrete. On the NASA cycle-level frame,
`capacity_loss` has mean 0.003 Ah and standard deviation 0.132 Ah, and ranges
from -2.61 to +1.52. A target whose spread is forty times its mean, and whose
minimum is a large *negative* value (capacity cannot un-degrade by 2.6 Ah), is
substantially measurement noise. No method can achieve high R-squared against
noise, so "every method scored ~0" may be reporting the target's noise floor
rather than the methods' skill.

That distinction matters enormously for how this project's central finding
should be read, so it is measured rather than assumed. `signal_to_noise`
below estimates, per target, what fraction of its variance is explainable
trend — which puts a *ceiling* on achievable R-squared that any honest
benchmark must be compared against.

The targets
-----------
`capacity_loss`
    Per-cycle delta. The original target. Retained so the new results are
    comparable with everything already published in docs/final_report.md.

`soh`
    Capacity as a fraction of that cell's own initial capacity. The standard
    target in the literature and the one most BMS deployments actually want.
    Far better conditioned: it is a level, not a difference, so measurement
    errors do not compound.

`cumulative_fade`
    1 - SOH. Same information, sign-flipped so that larger means worse,
    matching the direction of every other score in this codebase.

`horizon_fade_{h}`
    Fade accrued over the next `h` cycles. This is the quantity a prognostic
    is actually asked for, and averaging over a horizon suppresses per-cycle
    measurement noise without smoothing away the trend. Section 4.5 of the
    final report found longer horizons improve rank correlation; this makes
    horizon a first-class, benchmarkable target rather than a one-off
    experiment.

A note on leakage
-----------------
Every target here is derived *within* a cell from that cell's own capacity
series, never from fleet-level statistics. Normalising by a fleet mean would
leak information across the LOBO/LOCO fold boundary — the held-out cell would
have contributed to the scaling of its own target.

NOT EVERY CELL CAN CARRY AN SOH TARGET, AND THE INADMISSIBLE ONES ARE NAMED
---------------------------------------------------------------------------
State of health is capacity relative to *fresh* capacity, which presumes every
`capacity_ah` entry for a cell is the same kind of measurement. On this frame
that presumption fails for a specific, identifiable subset.

NASA's randomised-usage cells interleave reference discharge cycles with
random-walk loading, so their per-cycle `capacity_ah` mixes full reference
discharges with partial ones. The frame retained here carries no
measurement-type flag, so the two cannot be separated. The symptom is
unmistakable once looked for: normalising B0041 by its earliest cycles yields
a state of health of 22.6, and seven of thirty-four cells exceed 1.5.

A first implementation of this module used the mean of each cell's five
earliest cycles as the reference and reported a signal fraction of 0.97 for
SOH. That number was almost entirely the artifact above — spurious spread
being counted as trend. It is recorded here rather than quietly corrected,
because a noise-ceiling estimate is exactly the kind of quantity that looks
authoritative and is easy to inflate without noticing.

Two changes follow. The reference is a high quantile of the cell's own
capacity series rather than its earliest readings, which is robust to both a
single spurious high value and to early-cycle sampling problems. And every
cell is screened for whether SOH is well defined for it at all; those that
fail are excluded from the SOH-family targets by name and counted, rather than
being normalised into a plausible-looking number. `screen_cells_for_soh`
returns that report and `add_targets` applies it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CELL_COLUMN = "cell_id"
CYCLE_COLUMN = "cycle"
CAPACITY_COLUMN = "capacity_ah"

# Quantile of a cell's own capacity series taken as its fresh-capacity
# reference. Not the maximum: a single spurious high reading would then set the
# scale for every other cycle. Not the earliest cycles either — see the module
# docstring for the artifact that produced.
REFERENCE_CAPACITY_QUANTILE = 0.95

# Screening thresholds for whether SOH is well defined for a cell.
MIN_CYCLES_FOR_SOH = 10
# SOH above this is physically implausible for a cell being normalised by its
# own fresh capacity; a little headroom is allowed for measurement scatter.
SOH_UPPER_BOUND = 1.05
# How much of a cell's series may exceed that bound before the series is judged
# to be mixing measurement types rather than showing scatter.
MAX_OVERSHOOT_FRACTION = 0.10

# Degradation is gradual: consecutive cycles differ by a fraction of a percent
# of capacity, not by tens of percent. A large typical step between adjacent
# cycles means the series is alternating between different *kinds* of
# measurement — a full reference discharge one cycle, a partial drive-cycle
# discharge the next — rather than tracking one quantity as it declines.
#
# This catches what the overshoot test alone misses. Normalising by a high
# quantile bounds SOH at 1.0 by construction, so a cell alternating between
# full and partial discharges shows *no* overshoot at all; its partial cycles
# simply read as implausibly sudden capacity loss and recovery. The median
# absolute step is used rather than the mean so that a handful of genuine
# outliers does not by itself condemn a sound series.
MAX_MEDIAN_STEP_FRACTION = 0.05

# A cell must spend most of its life above this fraction of its own reference
# capacity for that reference to mean "fresh capacity".
#
# Cells are retired somewhere around 70-80% state of health, so a cell whose
# *median* life sits below 40% of its reference is not being measured against
# a fresh-capacity reference — the reference is wrong.
#
# This criterion was added after the first three failed to catch a real
# pathology in CALCE CS2. Types 3, 5 and 6 do not perform one full discharge
# per `Cycle_Index`: Type 3 switches discharge rate six times within a cycle,
# and Types 5 and 6 cycle partially by design. Normalising by a high quantile
# of the cell's own capacity then picks a reference that is either physically
# impossible (CS2_3's reference computes to 5.4 Ah on a 1.1 Ah cell) or is a
# partial-cycle capacity rather than a full one.
#
# The overshoot and step tests both pass in that situation, because a too-large
# reference *deflates* every reading rather than overshooting and does so
# smoothly. The symptom is instead an absurdly low median: CS2_3 reads as
# 99.7% degraded for most of its life.
#
# The separation is wide enough not to be a tuned number — the excluded cells
# sit at 0.003 and 0.062, the retained ones at 0.659 and above, an order of
# magnitude apart. On NASA it excludes exactly one cell, B0041, whose reference
# was already known to be unreliable.
MIN_MEDIAN_SOH = 0.40

# The reference must be a plausible fraction of the cell's own largest reading.
#
# A fifth criterion, added after loading CALCE CX2. CX2_3 records 91,279
# "cycles" whose 95th-percentile capacity is 0.017 Ah against a maximum of
# 5.67 Ah — a ratio of 0.003. Almost every row carries no discharge at all, so
# the reference lands on noise and SOH pins at 1.000 with an interquartile
# range of 0.0001. There is no trajectory in it.
#
# None of the first four criteria fire: SOH never overshoots (the reference is
# the 95th percentile), steps are tiny (everything is tiny), and the median is
# 1.0 rather than low. The cell looks pristine and is empty.
#
# It also mattered disproportionately — 91,279 of 131,580 admissible rows, 69%
# of the training set, from one degenerate cell.
#
# An interquartile floor was considered and rejected: NASA has legitimate
# short-lived cells at IQR 0.0098 against CX2_3's 0.0001, which is too narrow a
# margin to be safe. This ratio separates by 33x from the next CALCE cell and
# 175x from the lowest NASA one.
MIN_REFERENCE_OVER_MAX = 0.02


@dataclass(frozen=True)
class SignalEstimate:
    """How much of a target's variance is trend rather than noise."""

    target: str
    n_cells: int
    n_rows: int
    signal_variance: float
    noise_variance: float

    @property
    def signal_fraction(self) -> float:
        """Fraction of within-cell variance attributable to monotone trend.

        This is the ceiling on R-squared any predictor can achieve against
        this target, given that the residual is not predictable from anything.
        """
        total = self.signal_variance + self.noise_variance
        return float(self.signal_variance / total) if total > 0 else float("nan")

    def render(self) -> str:
        return (
            f"{self.target}: signal fraction {self.signal_fraction:.3f} "
            f"({self.n_cells} cells, {self.n_rows} rows) — an R2 above roughly "
            f"{self.signal_fraction:.2f} is not attainable against this target."
        )


@dataclass(frozen=True)
class CellScreen:
    """Whether SOH is well defined for one cell, and why not if it is not."""

    cell_id: str
    admissible: bool
    n_cycles: int
    reference_capacity_ah: float
    overshoot_fraction: float
    capacity_cycle_rho: float
    median_step_fraction: float = float("nan")
    median_soh: float = float("nan")
    reason: str = ""


def screen_cells_for_soh(
    data: pd.DataFrame,
    cell_col: str = CELL_COLUMN,
    cycle_col: str = CYCLE_COLUMN,
    capacity_col: str = CAPACITY_COLUMN,
) -> pd.DataFrame:
    """Decide, per cell, whether its capacity series can carry an SOH target.

    Three conditions, each of which has to hold for SOH to mean what the name
    says:

    1. Enough cycles to establish a reference at all.
    2. A positive reference capacity.
    3. Not too much of the series above that reference. A cell normalised by
       its own 95th-percentile capacity should sit at or below 1.0 nearly
       everywhere. Persistent overshoot means the series is not a single kind
       of measurement — which is the randomised-usage case described in the
       module docstring, not scatter.

    A declining trend is reported (`capacity_cycle_rho`) but is deliberately
    *not* a screening criterion. Requiring the target to trend downward before
    admitting it would select cells that agree with the hypothesis under test,
    which is the kind of circularity this project's validation work exists to
    avoid.
    """
    rows: list[CellScreen] = []

    for cell_id, group in data.groupby(cell_col):
        frame = group[[cycle_col, capacity_col]].dropna()
        n = len(frame)
        capacity = frame[capacity_col].to_numpy(dtype=float)

        if n < MIN_CYCLES_FOR_SOH:
            rows.append(CellScreen(
                str(cell_id), False, n, float("nan"), float("nan"), float("nan"),
                reason=f"only {n} usable cycles, below minimum {MIN_CYCLES_FOR_SOH}",
            ))
            continue

        reference = float(np.quantile(capacity, REFERENCE_CAPACITY_QUANTILE))
        if not np.isfinite(reference) or reference <= 0:
            rows.append(CellScreen(
                str(cell_id), False, n, reference, float("nan"), float("nan"),
                reason="reference capacity is zero or undefined",
            ))
            continue

        largest = float(np.max(capacity))
        reference_over_max = reference / largest if largest > 0 else float("nan")
        if np.isfinite(reference_over_max) and reference_over_max < MIN_REFERENCE_OVER_MAX:
            rows.append(CellScreen(
                str(cell_id), False, n, reference, float("nan"), float("nan"),
                reason=(
                    f"reference capacity is {reference_over_max:.4f} of this "
                    f"cell's maximum ({reference:.4g} vs {largest:.4g} Ah), below "
                    f"the {MIN_REFERENCE_OVER_MAX} floor. Nearly every row carries "
                    f"no discharge, so the reference lands on noise and SOH pins "
                    f"at 1.0 with no trajectory to learn"
                ),
            ))
            continue

        soh = capacity / reference
        overshoot = float(np.mean(soh > SOH_UPPER_BOUND))
        rho = pd.Series(frame[cycle_col].to_numpy(dtype=float)).corr(
            pd.Series(capacity), method="spearman"
        )
        rho = float(rho) if rho is not None and np.isfinite(rho) else float("nan")

        ordered = frame.sort_values(cycle_col)[capacity_col].to_numpy(dtype=float)
        steps = np.abs(np.diff(ordered))
        step_fraction = (
            float(np.median(steps) / reference) if len(steps) else float("nan")
        )

        median_soh = float(np.median(soh))

        if overshoot > MAX_OVERSHOOT_FRACTION:
            rows.append(CellScreen(
                str(cell_id), False, n, reference, overshoot, rho, step_fraction,
                median_soh,
                f"{overshoot:.0%} of cycles exceed SOH {SOH_UPPER_BOUND}, above "
                f"the {MAX_OVERSHOOT_FRACTION:.0%} tolerance — the capacity "
                f"series mixes measurement types (randomised-usage cells "
                f"interleave reference and partial discharges)",
            ))
            continue

        if np.isfinite(step_fraction) and step_fraction > MAX_MEDIAN_STEP_FRACTION:
            rows.append(CellScreen(
                str(cell_id), False, n, reference, overshoot, rho, step_fraction,
                median_soh,
                f"median step between consecutive cycles is {step_fraction:.1%} of "
                f"reference capacity, above the {MAX_MEDIAN_STEP_FRACTION:.0%} "
                f"tolerance. Degradation is gradual; steps this large mean the "
                f"series alternates between different kinds of measurement "
                f"rather than tracking one declining quantity",
            ))
            continue

        if median_soh < MIN_MEDIAN_SOH:
            rows.append(CellScreen(
                str(cell_id), False, n, reference, overshoot, rho, step_fraction,
                median_soh,
                f"median SOH across life is {median_soh:.1%}, below the "
                f"{MIN_MEDIAN_SOH:.0%} floor. Cells are retired near 70-80%, so a "
                f"cell reading this degraded for most of its life is being "
                f"normalised by a reference that is not its fresh capacity — "
                f"the case where a cycle index spans several discharges "
                f"(CALCE CS2 Type 3 computes a 5.4 Ah reference on a 1.1 Ah cell)",
            ))
            continue

        rows.append(CellScreen(
            str(cell_id), True, n, reference, overshoot, rho, step_fraction,
            median_soh,
        ))

    return pd.DataFrame([vars(r) for r in rows])


def add_targets(
    data: pd.DataFrame,
    horizons: tuple[int, ...] = (10, 20, 50),
    screen: bool = True,
) -> pd.DataFrame:
    """Add every derived target to a cycle-level frame.

    Requires `cell_id`, `cycle` and `capacity_ah`. Leaves any existing
    `capacity_loss` column untouched so previously published numbers stay
    reproducible.

    With `screen=True` (the default), cells for which SOH is not well defined
    get NaN in the SOH-family targets rather than a fabricated value. They
    keep `capacity_loss`, which is a within-cell difference and does not
    depend on a fresh-capacity reference. `soh_admissible` marks which is
    which, so a downstream count of usable cells is always available.
    """
    missing = [c for c in (CELL_COLUMN, CYCLE_COLUMN, CAPACITY_COLUMN)
               if c not in data.columns]
    if missing:
        raise ValueError(f"add_targets: missing required columns {missing}")

    out = data.sort_values([CELL_COLUMN, CYCLE_COLUMN]).copy()

    verdicts = screen_cells_for_soh(out)
    reference = dict(zip(
        verdicts["cell_id"], verdicts["reference_capacity_ah"], strict=True,
    ))
    admissible = dict(zip(verdicts["cell_id"], verdicts["admissible"], strict=True))

    keys = out[CELL_COLUMN].astype(str)
    out["reference_capacity_ah"] = keys.map(reference)
    out["soh_admissible"] = keys.map(admissible).fillna(False) if screen else True

    usable = (out["reference_capacity_ah"] > 0) & out["soh_admissible"]
    soh = np.where(
        usable, out[CAPACITY_COLUMN] / out["reference_capacity_ah"], np.nan
    )

    # Screening happens at two levels, because two different things go wrong.
    #
    # The cell-level screen above asks whether SOH is definable for a cell at
    # all. This observation-level mask handles the separate case of a single
    # bad reading inside an otherwise sound series: a cell with 195 good
    # cycles and one implausible spike should lose the spike, not the cell.
    #
    # Without this, one reading at SOH 1.9 survives into the variance
    # decomposition and is counted as trend, inflating the noise ceiling the
    # rest of this module exists to estimate honestly.
    out["soh"] = np.where(soh > SOH_UPPER_BOUND, np.nan, soh)
    out["soh_outlier"] = np.isfinite(soh) & (soh > SOH_UPPER_BOUND)
    out["cumulative_fade"] = 1.0 - out["soh"]

    for horizon in horizons:
        future = out.groupby(CELL_COLUMN)["cumulative_fade"].shift(-horizon)
        out[f"horizon_fade_{horizon}"] = future - out["cumulative_fade"]

    return out


def target_columns(horizons: tuple[int, ...] = (10, 20, 50)) -> tuple[str, ...]:
    return (
        "capacity_loss", "soh", "cumulative_fade",
        *(f"horizon_fade_{h}" for h in horizons),
    )


def signal_to_noise(
    data: pd.DataFrame,
    target: str,
    cell_col: str = CELL_COLUMN,
    cycle_col: str = CYCLE_COLUMN,
) -> SignalEstimate:
    """Split a target's within-cell variance into monotone trend and residual.

    Uses isotonic regression against cycle index per cell. The choice is
    deliberate: a monotone fit assumes only that degradation does not reverse,
    which is the weakest assumption that still defines a trend. A polynomial
    would attribute some genuine noise to signal by bending to fit it, which
    would overstate the achievable ceiling and understate this module's point.

    Cells with fewer than 10 usable observations are excluded — an isotonic
    fit through a handful of points can absorb nearly all variance and would
    inflate the estimate.
    """
    from sklearn.isotonic import IsotonicRegression

    if target not in data.columns:
        raise ValueError(f"signal_to_noise: no column '{target}'")

    signal_parts: list[float] = []
    noise_parts: list[float] = []
    used_cells = 0
    used_rows = 0

    for _, group in data.groupby(cell_col):
        frame = group[[cycle_col, target]].dropna()
        if len(frame) < 10:
            continue
        x = frame[cycle_col].to_numpy(dtype=float)
        y = frame[target].to_numpy(dtype=float)
        if len(np.unique(x)) < 3 or np.std(y) == 0:
            continue

        model = IsotonicRegression(increasing="auto", out_of_bounds="clip")
        trend = model.fit_transform(x, y)

        signal_parts.append(float(np.var(trend)) * len(y))
        noise_parts.append(float(np.var(y - trend)) * len(y))
        used_cells += 1
        used_rows += len(y)

    if used_rows == 0:
        return SignalEstimate(target, 0, 0, float("nan"), float("nan"))

    return SignalEstimate(
        target=target,
        n_cells=used_cells,
        n_rows=used_rows,
        signal_variance=float(np.sum(signal_parts) / used_rows),
        noise_variance=float(np.sum(noise_parts) / used_rows),
    )


def signal_report(
    data: pd.DataFrame,
    targets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Signal fraction for every target present in the frame."""
    candidates = targets if targets is not None else target_columns()
    rows = []
    for target in candidates:
        if target not in data.columns:
            continue
        estimate = signal_to_noise(data, target)
        rows.append({
            "target": estimate.target,
            "n_cells": estimate.n_cells,
            "n_rows": estimate.n_rows,
            "signal_variance": estimate.signal_variance,
            "noise_variance": estimate.noise_variance,
            "signal_fraction": estimate.signal_fraction,
            "max_attainable_r2": estimate.signal_fraction,
        })
    return pd.DataFrame(rows)
