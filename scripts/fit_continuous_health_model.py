"""Stage 3: fit a continuous degradation model on the NASA cycle-level data.

Deliberately conservative given what Stages 1-2 actually found
(docs/calibration_report.md Sections 3-9): only `trailing_avg_temp` is used
as a predictor, because it's the only signal that showed a significant,
correctly-signed, cross-cohort-consistent relationship with capacity loss.
`avg_stress` and current-based flags are excluded — including a predictor
that flips sign between cohorts would make the fitted model worse than the
heuristic it's replacing, not better. `deep_discharge_duration` reached
significance in only one of nine cohorts and is left out for the same
reason. This can be revisited once the current-based flags are
redesigned to be temperature-conditioned (open item in the calibration
report).

Model: OLS, capacity_loss ~ trailing_avg_temp + C(cohort), i.e. a common
temperature slope with a per-cohort intercept (cohorts differ in baseline
fade rate for reasons unrelated to temperature — different cutoff voltage,
current, etc. — see Section 2 of the calibration report; the fixed effect
absorbs that rather than pretending it doesn't exist).

Validation: leave-one-battery-out cross-validation, not a random train/test
split — a random split would let cycles from the same battery leak between
train and test, inflating the score exactly the way naive pooled p-values
did in Stage 1. Reported: per-held-out-battery correlation between
predicted and actual capacity loss, and overall out-of-sample MAE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.calibrate_cohort_cycle_level import COHORTS, build_cycle_level_table


def build_training_table(telemetry: pd.DataFrame) -> pd.DataFrame:
    cyc = build_cycle_level_table(telemetry)
    battery_to_cohort = {bid: name for name, ids in COHORTS.items() for bid in ids}
    cyc["cohort"] = cyc["cell_id"].map(battery_to_cohort)
    return cyc.dropna(subset=["cohort", "trailing_avg_temp", "capacity_loss"])


def fit_pooled_model(train: pd.DataFrame):
    model = smf.ols("capacity_loss ~ trailing_avg_temp + C(cohort)", data=train).fit()
    return model


def leave_one_battery_out_cv(cyc: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for held_out in cyc["cell_id"].unique():
        train = cyc[cyc["cell_id"] != held_out]
        test = cyc[cyc["cell_id"] == held_out]
        if test["cohort"].iloc[0] not in train["cohort"].values or len(test) < 5:
            continue  # can't predict a cohort intercept never seen in training
        model = fit_pooled_model(train)
        try:
            pred = model.predict(test)
        except Exception:
            continue
        mae = float(np.mean(np.abs(pred - test["capacity_loss"])))
        if test["capacity_loss"].nunique() > 1 and pred.nunique() > 1:
            rho, _ = stats.spearmanr(pred, test["capacity_loss"])
        else:
            rho = np.nan
        rows.append({"held_out_battery": held_out, "cohort": test["cohort"].iloc[0], "n_test_cycles": len(test), "mae": mae, "spearman_rho": rho})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-parquet", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="reports/metrics")
    args = parser.parse_args()

    telemetry = pd.read_parquet(args.cache_parquet)
    train = build_training_table(telemetry)
    print(f"Training table: {len(train)} (battery, cycle) rows across {train.cell_id.nunique()} batteries, {train.cohort.nunique()} cohorts")

    model = fit_pooled_model(train)
    print("\n=== Pooled model (in-sample, for coefficient inspection only) ===")
    print(model.summary().tables[1])
    print(f"R-squared: {model.rsquared:.4f}")

    print("\n=== Leave-one-battery-out cross-validation (the number that matters) ===")
    cv = leave_one_battery_out_cv(train)
    print(cv.to_string(index=False))
    print(f"\nOverall out-of-sample MAE: {cv['mae'].mean():.5f} Ah")
    print(f"Median per-battery Spearman rho (predicted vs actual capacity loss): {cv['spearman_rho'].median():.3f}")
    print(f"Batteries with positive rho: {(cv['spearman_rho'] > 0).sum()} / {cv['spearman_rho'].notna().sum()}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(out_dir / "health_model_v2_loocv.csv", index=False)
    with open(out_dir / "health_model_v2_coefficients.txt", "w") as f:
        f.write(str(model.summary()))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
