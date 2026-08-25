"""The reversible thermal capacity effect, and why it inverts the temperature signal.

THE FINDING
-----------
Measured discharge capacity depends on the temperature at which the discharge
was performed, independently of the cell's state of health. At low
temperature, higher internal resistance and slower lithium diffusion mean the
cell reaches its voltage cutoff sooner, so less capacity is delivered. The
capacity comes back when the cell is warmed. It is reversible, and it is not
degradation.

On the NASA cycle-level frame this effect is large enough to dominate the
target. Restricting to cycle <= 20, where a cell cannot yet have degraded
meaningfully:

    ambient 4 C    mean SOH 0.918   (10 cells)
    ambient 24 C   mean SOH 0.909   (14 cells)
    ambient 43 C   mean SOH 0.979   (4 cells)
    ambient 44 C   mean SOH 0.999   (3 cells)

A cell does not lose 8% of its capacity in twenty cycles. Nearly all of that
spread is the measurement effect, and it runs *opposite* to real thermal
degradation: colder cells read as more degraded from the very first cycle.

WHY THIS MATTERS SO MUCH HERE
-----------------------------
It supplies a mechanism for this project's central unexplained result.

`docs/final_report.md` Section 4.2 reports that trailing temperature is a
real, correctly-signed predictor of fade *within* cohorts, and ADR 0002
reports that the fitted model's ranking collapses to rho = -0.295 across
cohorts. Those two facts have sat side by side as a puzzle, attributed to
cohort intercepts doing the work.

This effect explains them as one phenomenon. Within a cohort ambient
temperature is fixed, so the artifact is constant and the residual
temperature variation reflects real thermal stress — the signal is genuine and
correctly signed. Across cohorts ambient temperature changes, the artifact
switches on, and because it is anti-correlated with true thermal degradation
it does not merely add noise: it *reverses* the apparent relationship. A
coefficient fitted within cohorts and applied across them therefore gets the
sign wrong, which is exactly what LOCO measured.

It also explains why the Arrhenius fit in `arrhenius.py` returns a negative
activation energy on the uncorrected target. The model is correctly
implemented and is faithfully reporting that, as measured, this data says
degradation slows with heat.

THE CORRECTION, AND ITS LIMITS
------------------------------
`thermal_baseline` estimates each ambient level's apparent SOH over an early
window and treats it as that level's measurement offset;
`correct_thermal_measurement` divides it out.

This is a first-order empirical correction and is labelled as one. Three
limitations that a reader must have:

1. **It assumes the offset is multiplicative and constant over life.** In
   reality internal resistance grows as a cell ages, so the low-temperature
   penalty grows too. The correction will therefore under-correct late-life
   cold cycles.
2. **It is estimated from few cells at some levels** — three at 44 C. The
   offset at those levels carries the between-cell variation of a very small
   sample.
3. **It cannot separate a genuine early-life difference from the artifact.**
   If cold-cycled cells really did degrade faster in their first twenty
   cycles, this would absorb that into the baseline and remove real signal.
   The early window is kept short to limit how much real degradation can hide
   inside it, and the window is a parameter so the sensitivity can be checked.

The principled fix is a temperature-corrected capacity measurement — a
reference discharge performed at a common temperature — which NASA's raw data
supports and this derived frame does not carry. That is a concrete acceptance
criterion for the next dataset, in the same spirit as the "8-10+ batteries per
cohort" criterion Section 4.6 established for mixed-effects identifiability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_AMBIENT_COLUMN = "ambient_temperature_c"
DEFAULT_EARLY_CYCLES = 20
# Below this many cells at an ambient level, the estimated offset is reported
# but flagged: it carries the between-cell variation of a tiny sample.
MIN_CELLS_PER_LEVEL = 3


@dataclass(frozen=True)
class ThermalBaseline:
    """Apparent SOH at a given ambient temperature before real degradation."""

    ambient_temperature_c: float
    baseline_soh: float
    n_cells: int
    n_rows: int

    @property
    def well_estimated(self) -> bool:
        return self.n_cells >= MIN_CELLS_PER_LEVEL

    @property
    def implied_offset_percent(self) -> float:
        """How much capacity the measurement loses at this temperature."""
        return (1.0 - self.baseline_soh) * 100.0


def thermal_baseline(
    data: pd.DataFrame,
    soh_col: str = "soh",
    ambient_col: str = DEFAULT_AMBIENT_COLUMN,
    cycle_col: str = "cycle",
    cell_col: str = "cell_id",
    early_cycles: int = DEFAULT_EARLY_CYCLES,
) -> pd.DataFrame:
    """Estimate each ambient level's measurement offset from early cycles.

    The early window is the identifying assumption: within it, differences in
    apparent SOH across ambient levels are attributed to the measurement
    rather than to degradation. See the module docstring for what that assumes
    and how it can fail.
    """
    for column in (soh_col, ambient_col, cycle_col, cell_col):
        if column not in data.columns:
            raise ValueError(f"thermal_baseline: missing required column '{column}'")

    early = data[(data[cycle_col] <= early_cycles) & data[soh_col].notna()]
    if early.empty:
        raise ValueError(
            f"thermal_baseline: no rows with {cycle_col} <= {early_cycles} and a "
            f"defined '{soh_col}'. Widen the window or check the SOH screen."
        )

    grouped = early.groupby(ambient_col).agg(
        baseline_soh=(soh_col, "mean"),
        n_cells=(cell_col, "nunique"),
        n_rows=(soh_col, "size"),
    ).reset_index()

    grouped["well_estimated"] = grouped["n_cells"] >= MIN_CELLS_PER_LEVEL
    grouped["implied_offset_percent"] = (1.0 - grouped["baseline_soh"]) * 100.0
    return grouped.rename(columns={ambient_col: "ambient_temperature_c"})


def correct_thermal_measurement(
    data: pd.DataFrame,
    soh_col: str = "soh",
    ambient_col: str = DEFAULT_AMBIENT_COLUMN,
    cycle_col: str = "cycle",
    cell_col: str = "cell_id",
    early_cycles: int = DEFAULT_EARLY_CYCLES,
    horizons: tuple[int, ...] = (10, 20, 50),
) -> pd.DataFrame:
    """Divide out each ambient level's measurement offset.

    Adds `soh_corrected`, `cumulative_fade_corrected` and horizon variants,
    leaving the uncorrected columns untouched so that corrected and
    uncorrected results stay directly comparable in one table.

    An ambient level absent from the baseline (because it had no early cycles)
    receives NaN rather than an offset of 1.0. Treating an unmeasured offset as
    "no offset" would be the same class of error as the NaN-as-healthy defect
    fixed in `risk.stress_score` — an absent correction silently becoming a
    benign one.
    """
    baselines = thermal_baseline(
        data, soh_col=soh_col, ambient_col=ambient_col, cycle_col=cycle_col,
        cell_col=cell_col, early_cycles=early_cycles,
    )
    offsets = dict(zip(
        baselines["ambient_temperature_c"], baselines["baseline_soh"], strict=True,
    ))

    out = data.copy()
    offset = out[ambient_col].map(offsets)
    # An offset at or below zero cannot be divided by; treat as unmeasured.
    offset = offset.where(offset > 0)
    out["thermal_offset"] = offset

    out["soh_corrected"] = out[soh_col] / offset
    out["cumulative_fade_corrected"] = 1.0 - out["soh_corrected"]

    for horizon in horizons:
        future = out.groupby(cell_col)["cumulative_fade_corrected"].shift(-horizon)
        out[f"horizon_fade_corrected_{horizon}"] = (
            future - out["cumulative_fade_corrected"]
        )

    return out


def confound_report(
    data: pd.DataFrame,
    soh_col: str = "soh",
    ambient_col: str = DEFAULT_AMBIENT_COLUMN,
    cycle_col: str = "cycle",
    early_cycles: int = DEFAULT_EARLY_CYCLES,
) -> dict[str, float]:
    """Quantify the confound in one dictionary, for the report and tests.

    `early_spearman` is the headline: a positive value means warmer cells
    measure *healthier* before they have had time to degrade, which is the
    artifact. A physically clean dataset would show this near zero.
    """
    early = data[(data[cycle_col] <= early_cycles) & data[soh_col].notna()]
    all_rows = data[data[soh_col].notna()]

    def spearman(frame: pd.DataFrame, a: str, b: str) -> float:
        if len(frame) < 3:
            return float("nan")
        value = frame[a].corr(frame[b], method="spearman")
        return float(value) if value is not None and np.isfinite(value) else float("nan")

    early_soh = early.groupby(ambient_col)[soh_col].mean()
    spread = (
        float(early_soh.max() - early_soh.min()) if len(early_soh) > 1 else float("nan")
    )

    return {
        "early_cycles": float(early_cycles),
        "n_early_rows": float(len(early)),
        "n_ambient_levels": float(early[ambient_col].nunique()),
        "early_spearman_ambient_vs_soh": spearman(early, ambient_col, soh_col),
        "all_spearman_ambient_vs_soh": spearman(all_rows, ambient_col, soh_col),
        "early_soh_spread_across_levels": spread,
    }
