"""Calibration: does the pipeline's rule-based scoring actually track real degradation?

Uses the NASA cleaned_dataset (34 batteries, real per-cycle telemetry, real
measured discharge capacity per cycle) to answer two separate questions:

1. BATTERY-LEVEL: across the 34 batteries, does a battery that the current
   pipeline scores as higher stress/risk/lower-health actually fade faster
   or die sooner (fewer cycles to 80% capacity, the standard EOL threshold)
   than one it scores as healthy?

2. FEATURE-LEVEL (more informative if #1 is weak/absent): independent of
   the hand-picked weights, do the *individual* behavior signals
   (avg_temp, deep_discharge_duration, fast_charge_duration,
   aggressive_discharge_count) correlate with fade rate on their own? This
   tells us whether the raw inputs carry real signal even if the current
   weighting/aggregation into a single 0-100 score doesn't.

Run: python scripts/calibrate_against_nasa.py --nasa-dir /path/to/cleaned_dataset
Outputs: reports/metrics/calibration_results.csv and a printed summary with
correlation coefficients and p-values (scipy.stats.spearmanr) — not just
point estimates asserted without a significance check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.io.load_nasa import load_nasa_dataset
from src.bms.features.behavior_features import compute_behavior_flags, summarize_batteries
from src.bms.features.cycle_features import summarize_by_cycle
from src.bms.risk.stress_score import compute_stress_score, compute_risk_assessment
from src.bms.health.health_index import compute_health_index
from src.bms.rul.rul_estimation import compute_rul

EOL_FRACTION = 0.80  # standard end-of-life definition: 80% of rated/initial capacity


def compute_ground_truth_per_battery(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Per-battery: initial capacity, fade rate (Ah/cycle, linear fit), cycles-to-EOL."""
    cap = telemetry.dropna(subset=["capacity_ah"])[["cell_id", "cycle", "capacity_ah"]].drop_duplicates()
    cap["capacity_ah"] = cap["capacity_ah"].astype(float)

    rows = []
    for cell_id, g in cap.groupby("cell_id"):
        g = g.sort_values("cycle")
        if len(g) < 5:
            continue
        initial_capacity = g["capacity_ah"].iloc[:3].mean()  # average first few cycles, less noisy than a single point
        slope, intercept, r, p, se = stats.linregress(g["cycle"], g["capacity_ah"])

        eol_capacity = EOL_FRACTION * initial_capacity
        below_eol = g[g["capacity_ah"] <= eol_capacity]
        cycles_to_eol = below_eol["cycle"].iloc[0] if len(below_eol) > 0 else np.nan
        reached_eol = len(below_eol) > 0

        rows.append({
            "battery_id": cell_id,
            "initial_capacity_ah": initial_capacity,
            "fade_rate_ah_per_cycle": -slope,  # positive = faster fade
            "fade_fit_r2": r ** 2,
            "n_cycles_measured": len(g),
            "cycles_to_eol": cycles_to_eol,
            "reached_eol_in_data": reached_eol,
        })
    return pd.DataFrame(rows)


def run_pipeline_scores(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Run the existing (unmodified) battery-level pipeline exactly as main.py does."""
    flagged = compute_behavior_flags(telemetry)
    flagged["stress_score"] = compute_stress_score(flagged)
    summary = summarize_batteries(flagged)
    risk = compute_risk_assessment(summary)
    health = compute_health_index(summary)
    merged = risk.merge(
        health.drop(columns=["avg_stress", "avg_temp", "deep_discharge_duration", "fast_charge_duration", "aggressive_discharge_count", "avg_soc"]),
        on="battery_id",
    )
    rul = compute_rul(merged)
    return rul


def correlate(df: pd.DataFrame, x_cols: list[str], y_col: str) -> pd.DataFrame:
    rows = []
    for x in x_cols:
        valid = df[[x, y_col]].dropna()
        if len(valid) < 5:
            continue
        rho, p = stats.spearmanr(valid[x], valid[y_col])
        rows.append({"feature": x, "target": y_col, "n": len(valid), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nasa-dir", type=str, required=True)
    parser.add_argument("--cache-parquet", type=str, default=None, help="Optional cached telemetry parquet to skip re-parsing 7000+ files")
    parser.add_argument("--out-dir", type=str, default="reports/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_parquet and Path(args.cache_parquet).exists():
        telemetry = pd.read_parquet(args.cache_parquet)
    else:
        telemetry = load_nasa_dataset(args.nasa_dir)
        if args.cache_parquet:
            telemetry.to_parquet(args.cache_parquet)

    print(f"Loaded {len(telemetry)} telemetry rows across {telemetry.cell_id.nunique()} batteries")

    ground_truth = compute_ground_truth_per_battery(telemetry)
    ground_truth.to_csv(out_dir / "nasa_ground_truth_fade.csv", index=False)
    print(f"\nGround truth computed for {len(ground_truth)} batteries "
          f"({ground_truth['reached_eol_in_data'].sum()} reached 80% EOL within the data)")

    scores = run_pipeline_scores(telemetry)
    scores.to_csv(out_dir / "nasa_pipeline_scores.csv", index=False)

    merged = ground_truth.merge(scores, on="battery_id", how="inner")
    merged.to_csv(out_dir / "calibration_merged.csv", index=False)

    print("\n=== Question 1: does the pipeline's existing 0-100 scores track real fade? ===")
    q1 = correlate(merged, ["health_index", "risk_score", "avg_stress"], "fade_rate_ah_per_cycle")
    print(q1.to_string(index=False))

    eol_subset = merged[merged["reached_eol_in_data"]]
    if len(eol_subset) >= 5:
        print(f"\n=== RUL's predicted total cycle life vs actual cycles-to-EOL (n={len(eol_subset)} batteries that reached EOL) ===")
        q1b = correlate(eol_subset, ["estimated_total_cycles"], "cycles_to_eol")
        print(q1b.to_string(index=False))
    else:
        print(f"\nOnly {len(eol_subset)} batteries reached 80% EOL within the recorded cycles — "
              "too few for a meaningful RUL-vs-actual-EOL correlation. Reporting is skipped rather than "
              "computed on an underpowered sample.")
        q1b = pd.DataFrame()

    print("\n=== Question 2: do the raw behavior signals correlate with fade, independent of current weights? ===")
    q2 = correlate(merged, ["avg_temp", "deep_discharge_duration", "fast_charge_duration", "aggressive_discharge_count"], "fade_rate_ah_per_cycle")
    print(q2.to_string(index=False))

    all_results = pd.concat([q1, q1b, q2], ignore_index=True)
    all_results.to_csv(out_dir / "calibration_results.csv", index=False)
    print(f"\nFull results written to {out_dir / 'calibration_results.csv'}")


if __name__ == "__main__":
    main()
