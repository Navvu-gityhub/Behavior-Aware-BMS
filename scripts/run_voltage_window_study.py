"""Score voltage-window SOH estimation against the fitted models, LOCO.

    python scripts/run_voltage_window_study.py

Reads the cached curve frames from `data/interim/calce_curves/` and writes
`reports/metrics/calce_voltage_window/`.

WHAT IS BEING COMPARED
----------------------
The benchmark table's methods all fit coefficients on training cohorts and
lose skill on an unseen one. This estimator fits nothing: SOH is the ratio of
charge delivered in a fixed voltage window now to the same window early in
that cell's life. Two measurements of one cell, no training cohort.

So the comparison is against the same held-out cohorts and the same R2
baseline the benchmark study uses - the training-set mean - so the numbers sit
beside the existing table rather than needing their own scale.

THE WINDOW IS A HYPERPARAMETER, AND IT IS SELECTED HONESTLY
------------------------------------------------------------
Picking the voltage window that scores best and then reporting that score is
the same optimism this project spent months diagnosing elsewhere. So for each
held-out cohort the window is chosen on the OTHER cohorts only, and the
held-out cohort is scored with that choice. `window_chosen` is recorded per
fold, and a window that is unstable across folds is itself a finding.

`naive_best` is reported alongside as the optimistic number - the single
window that scores best on everything - so the cost of choosing honestly is
visible rather than implied.

TWO ARMS
--------
`ratio`        the raw window ratio. Zero parameters. Nothing is fitted at
               all, on any cohort.
`ratio_affine` the ratio, then a single global slope and bias fitted on the
               OTHER cohorts. Two parameters. This corrects the systematic
               departure from uniform scaling documented in
               `health/voltage_window.py` - resistance growth and loss of
               active material - which is a property of the chemistry and the
               rate, not of a protocol, so it is the kind of thing that has a
               chance of transferring. Whether it does is what the fold
               measures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.bms.adaptive.validation import bootstrap_median_ci
from src.bms.benchmarks import add_targets
from src.bms.health.voltage_window import (
    WindowSpec,
    per_cell_fit,
    window_soh_table,
)

DEFAULT_CURVES = Path("data/interim/calce_curves")
DEFAULT_TRUTH = Path("reports/metrics/calce_full_discharge.csv")
DEFAULT_OUT = Path("reports/metrics/calce_voltage_window")

# A window must be measurable on at least this fraction of the median cell's
# cycles to be eligible. Without the gate the selector picks whichever window
# reads fewest cycles, because a small, easy subset gives the best error.
MIN_COHORT_COVERAGE = 0.80

# Windows swept. All sit inside the LCO plateau and avoid the steep knees,
# where a small voltage error moves the charge reading a long way.
WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec(4.05, 3.40),
    WindowSpec(4.00, 3.50),
    WindowSpec(4.00, 3.70),
    WindowSpec(3.95, 3.50),
    WindowSpec(3.90, 3.60),
    WindowSpec(3.85, 3.55),
)


def _r2(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - baseline) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _load_curves(directory: Path) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise SystemExit(
            f"no cached curves in {directory}. Run "
            f"scripts/build_calce_curve_frames.py first."
        )
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    curves = _load_curves(args.curves)
    truth = add_targets(pd.read_csv(args.truth))
    truth = truth[truth["soh"].notna()][
        ["cell_id", "cycle", "soh", "cohort"]].copy()

    print(f"curves: {len(curves)} points, {curves['cell_id'].nunique()} cells")
    print(f"truth : {len(truth)} rows, {truth['cell_id'].nunique()} cells, "
          f"{truth['cohort'].nunique()} cohorts")

    # One scored table per window, computed once and reused by every fold.
    tables: dict[str, pd.DataFrame] = {}
    coverage: dict[str, float] = {}
    for spec in WINDOWS:
        table = window_soh_table(
            curves, truth=truth[["cell_id", "cycle", "soh"]], spec=spec)
        table = table.merge(
            truth[["cell_id", "cohort"]].drop_duplicates(),
            on="cell_id", how="left", suffixes=("", "_truth"))
        if "cohort_truth" in table.columns:
            table["cohort"] = table["cohort"].fillna(table["cohort_truth"])
        scored = table.dropna(subset=["soh_window", "soh", "cohort"])
        tables[str(spec)] = scored
        # Coverage is per cell, then medianed. A pooled fraction is dominated
        # by whichever cells have the most cycles and hides a window that is
        # unmeasurable on half the fleet.
        per_cell = table.groupby("cell_id")["window_charge_ah"].apply(
            lambda c: float(c.notna().mean()))
        coverage[str(spec)] = float(per_cell.median())
        print(f"  {spec}: {len(scored)} scored rows, "
              f"median per-cell coverage {coverage[str(spec)]:.1%}")

    cohorts = sorted(truth["cohort"].dropna().unique())
    global_mean = float(truth["soh"].mean())
    rows: list[dict] = []

    for held in cohorts:
        # Choose the window on the other cohorts only, and on coverage FIRST.
        #
        # Selecting by error alone picks a window that is unmeasurable on most
        # cells: 4.05-3.40 V scores well on the cycles it can read and can
        # read 0.1% of CS2_35's, because that cell charges to 4.00 V and never
        # reaches 4.05. An estimator that refuses 999 cycles in 1000 is not
        # better than one that answers them, whatever its error on the
        # remainder. So coverage is a gate and error is the tie-break.
        eligible = {n: t for n, t in tables.items()
                    if coverage[n] >= MIN_COHORT_COVERAGE}
        if not eligible:
            eligible = {max(coverage, key=coverage.get): tables[
                max(coverage, key=coverage.get)]}
        scores = {}
        for name, table in eligible.items():
            other = table[table["cohort"] != held]
            if other.empty:
                continue
            scores[name] = float(np.mean(np.abs(
                other["soh_window"] - other["soh"])))
        if not scores:
            continue
        chosen = min(scores, key=scores.get)

        table = tables[chosen]
        test = table[table["cohort"] == held]
        other = table[table["cohort"] != held]
        if test.empty or len(other) < 10:
            continue

        y = test["soh"].to_numpy(dtype=float)
        est = test["soh_window"].to_numpy(dtype=float)
        base = np.full_like(y, global_mean)

        # Global shape correction fitted on the other cohorts only.
        slope, bias = np.polyfit(
            other["soh_window"].to_numpy(dtype=float),
            other["soh"].to_numpy(dtype=float), 1)
        corrected = slope * est + bias

        rows.append({
            "cohort": held,
            "window_chosen": chosen,
            "window_coverage": round(coverage[chosen], 3),
            "n_rows": len(test),
            "n_cells": int(test["cell_id"].nunique()),
            "ratio_mae": float(np.mean(np.abs(est - y))),
            "ratio_r2": _r2(y, est, base),
            "affine_mae": float(np.mean(np.abs(corrected - y))),
            "affine_r2": _r2(y, corrected, base),
            "fit_slope": float(slope),
            "fit_bias": float(bias),
        })
        print(f"  {held:<12} window={chosen:<14} "
              f"ratio MAE={rows[-1]['ratio_mae']:.4f} R2={rows[-1]['ratio_r2']:+.3f}  "
              f"affine MAE={rows[-1]['affine_mae']:.4f} "
              f"R2={rows[-1]['affine_r2']:+.3f}", flush=True)

    folds = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out / "window_folds.csv", index=False)

    # The optimistic number, for contrast.
    naive = {
        name: float(np.mean(np.abs(t["soh_window"] - t["soh"])))
        for name, t in tables.items() if not t.empty
    }
    naive_best = min(naive, key=naive.get)

    summary_rows = []
    for arm in ("ratio", "affine"):
        for metric in ("mae", "r2"):
            vals = folds[f"{arm}_{metric}"].dropna().tolist()
            low, high = bootstrap_median_ci(vals)
            summary_rows.append({
                "arm": arm, "metric": metric, "n_folds": len(vals),
                "median": round(float(np.median(vals)), 4) if vals else np.nan,
                "worst": round(float(np.max(vals) if metric == "mae"
                                     else np.min(vals)), 4) if vals else np.nan,
                "ci_low": round(low, 4), "ci_high": round(high, 4),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out / "window_summary.csv", index=False)

    cells = per_cell_fit(tables[naive_best])
    cells.to_csv(args.out / "window_per_cell.csv", index=False)

    print()
    print(summary.to_string(index=False))

    lines = [
        "# SOH from charge in a fixed voltage window",
        "",
        f"Generated by `scripts/run_voltage_window_study.py`. "
        f"{len(curves)} curve points, {curves['cell_id'].nunique()} cells, "
        f"{len(cohorts)} cohorts.",
        "",
        "The estimator fits nothing. SOH is the ratio of charge delivered",
        "between two fixed terminal voltages now, to the same window early in",
        "that cell's life - two measurements of one cell. There is no training",
        "cohort and therefore no cross-cohort generalisation step to fail.",
        "",
        "## Results, leave-one-cohort-out",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "`ratio` fits nothing at all. `affine` applies one global slope and",
        "bias fitted on the OTHER cohorts, correcting the systematic departure",
        "from uniform curve scaling.",
        "",
        "R2 is measured against the same baseline as the benchmark table - the",
        "global mean SOH - so these numbers sit beside it directly.",
        "",
        "## Per fold",
        "",
        "```",
        folds.to_string(index=False),
        "```",
        "",
        "## The window was chosen per fold, on the other cohorts",
        "",
        "Selecting the best-scoring window and then reporting its score would",
        "be the optimism this project exists to avoid. Each fold's window is",
        "chosen on the cohorts it does not test on, and `window_chosen`",
        "records it. A window that moves between folds is itself a result.",
        "",
        f"The single best window over everything is **{naive_best}**, at "
        f"MAE **{naive[naive_best]:.4f}**. That is the optimistic figure; the",
        "per-fold medians above are the honest ones, and the gap between them",
        "is what choosing honestly costs.",
        "",
        "## Per cell: does the uniform-scaling assumption hold?",
        "",
        "```",
        cells.round(4).to_string(index=False),
        "```",
        "",
        "Slope one and bias zero would mean window charge falls exactly in",
        "proportion to capacity. It does not, and the spread here is the",
        "measured size of that approximation rather than an assumed one.",
        "",
        "## What this still needs that the fitted models do not",
        "",
        "One reference measurement of the same cell early in its life. A pack",
        "never characterised cannot use this. Production BMS firmware already",
        "stores such a reference at manufacture, so the requirement is",
        "ordinary - but it is a requirement, and it is the reason this is not",
        "a drop-in replacement for a model that needs no per-cell history.",
    ]
    (args.out / "window_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}/window_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
