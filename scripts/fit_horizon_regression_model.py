"""Stage 3b: does a longer prediction horizon fix the Level-3 model?

Motivation (final_report.md Section 6, item 2): the Level-3 model
(`fit_continuous_health_model.py`) predicts capacity loss from ONE cycle to
the next. That target is dominated by measurement noise -- NASA's discharge
capacity readings jitter cycle-to-cycle by an amount that's a meaningful
fraction of the true underlying fade rate, so a model trying to predict
"how much capacity did this cell lose in the last cycle" is mostly trying
to predict noise. Predicting cumulative loss over a longer window (10, 20,
50 cycles) should average that noise down while the underlying temperature
signal accumulates, IF the signal is real and the ceiling was noise (not a
signal that's just absent).

This is exactly the counterfactual the project's own future-work list
proposed, and it's why this experiment is worth running before anything
else: it's the cheapest possible test of whether the negative headline
result (Section 4.3) reflects noise-swamping-a-real-effect, or an absent
effect, before spending more effort on new datasets or new predictors.

Data: reuses `reports/metrics/continuous_model_training_data.csv`, the
exact cached cycle-level table `fit_continuous_health_model.py` already
built and fit on (2,682 rows, 33 batteries, 9 cohorts; see that script's
docstring for how it was constructed from raw NASA telemetry). Re-deriving
horizon targets from this cache -- rather than re-running the raw-telemetry
pipeline -- is a deliberate simplification: `trailing_avg_temp` (the
predictor) is NOT horizon-dependent, only the TARGET is, so it can be
recomputed directly from the cached (cell_id, cycle, capacity_ah) triples
without touching the raw .mat files. This is exact, not approximate, for
the reasons given above -- there is nothing the raw telemetry would add
to this specific comparison.

Known, unavoidable limitation of this design (report honestly, don't
bury it): longer horizons need more remaining cycles, so short-lived
cohorts drop out entirely as the horizon grows. At H=50, four of nine
cohorts (RT_SQWAVE_4A_variedcutoff, ELEV43C_4A_CC_variedcutoff,
MIXED_24_44C_multiload, COLD4C_2A_flagged -- 15 of 33 batteries) have no
cell that ever reaches a 50-cycles-later reading, and are excluded. Any
apparent improvement at H=50 must be read against a DIFFERENT, smaller,
longer-lived-battery-skewed sample than H=1 -- it is not a controlled
comparison, and this script reports the surviving N so that's checkable
rather than assumed away.

Same validation discipline as fit_continuous_health_model.py: leave-one-
battery-out CV, R^2 against each held-out battery's own mean (does the
model beat "assume this battery's average"?), and Spearman rank
correlation. Both reported per horizon, per the reasoning in
final_report.md Section 4.3 for why R^2-vs-own-mean is the metric a
deployment decision should actually depend on.
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


def build_horizon_table(cyc: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Replace the 1-cycle `capacity_loss` target with a cumulative,
    H-cycles-ahead target: capacity_ah(cycle) - capacity_ah(cycle + H).

    Positive = more fade over the window, same sign convention as the
    original `capacity_loss`. Rows where cycle+H doesn't exist for that
    cell (end of that battery's recorded life) are dropped -- there is no
    valid target for them at this horizon, not a zero.
    """
    caps = cyc[["cell_id", "cycle", "capacity_ah"]].drop_duplicates()
    future = caps.rename(columns={"cycle": "future_cycle", "capacity_ah": "capacity_ah_future"})

    base = cyc.copy()
    base["future_cycle"] = base["cycle"] + horizon
    merged = base.merge(future, on=["cell_id", "future_cycle"], how="left")
    merged[f"loss_h{horizon}"] = merged["capacity_ah"] - merged["capacity_ah_future"]
    return merged.dropna(subset=[f"loss_h{horizon}", "trailing_avg_temp", "cohort"])


def leave_one_battery_out_cv(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    rows = []
    formula = f"{target_col} ~ trailing_avg_temp + C(cohort)"
    for held_out in df["cell_id"].unique():
        train = df[df["cell_id"] != held_out]
        test = df[df["cell_id"] == held_out]
        if test["cohort"].iloc[0] not in train["cohort"].values or len(test) < 5:
            continue  # can't predict a cohort intercept never seen in training

        model = smf.ols(formula, data=train).fit()
        try:
            pred = model.predict(test)
        except Exception:
            continue

        actual = test[target_col].to_numpy()
        mae = float(np.mean(np.abs(pred - actual)))

        # R^2 against this battery's OWN mean -- the baseline a real
        # deployment decision would be compared to (see docstring).
        battery_mean = actual.mean()
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - battery_mean) ** 2))
        r2_vs_own_mean = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        if pd.Series(actual).nunique() > 1 and pred.nunique() > 1:
            rho, _ = stats.spearmanr(pred, actual)
        else:
            rho = np.nan

        rows.append(
            {
                "held_out_battery": held_out,
                "cohort": test["cohort"].iloc[0],
                "n_test_cycles": len(test),
                "mae": mae,
                "r2_vs_own_mean": r2_vs_own_mean,
                "spearman_rho": rho,
            }
        )
    return pd.DataFrame(rows)


def run_horizon(cyc: pd.DataFrame, horizon: int, out_dir: Path) -> dict:
    target_col = f"loss_h{horizon}"
    htab = build_horizon_table(cyc, horizon)

    n_batteries = htab["cell_id"].nunique()
    n_cohorts = htab["cohort"].nunique()
    dropped_batteries = sorted(set(cyc["cell_id"]) - set(htab["cell_id"]))

    in_sample = smf.ols(f"{target_col} ~ trailing_avg_temp + C(cohort)", data=htab).fit()
    temp_coef = in_sample.params.get("trailing_avg_temp", np.nan)
    temp_p = in_sample.pvalues.get("trailing_avg_temp", np.nan)

    cv = leave_one_battery_out_cv(htab, target_col)
    cv.to_csv(out_dir / f"horizon_h{horizon}_lobo_cv.csv", index=False)

    n_testable = len(cv)
    summary = {
        "horizon_cycles": horizon,
        "n_cycle_obs": len(htab),
        "n_batteries": n_batteries,
        "n_batteries_dropped": len(dropped_batteries),
        "dropped_batteries": ",".join(dropped_batteries) if dropped_batteries else "",
        "n_cohorts": n_cohorts,
        "in_sample_r2": round(in_sample.rsquared, 4),
        "trailing_avg_temp_coef": round(float(temp_coef), 5) if pd.notna(temp_coef) else np.nan,
        "trailing_avg_temp_p": round(float(temp_p), 5) if pd.notna(temp_p) else np.nan,
        "n_batteries_testable_lobo": n_testable,
        # Mean R^2 is reported for completeness but is NOT the headline
        # number: a handful of near-flat-trajectory batteries have SS_tot
        # close to zero, so any real prediction error explodes into an
        # arbitrarily large negative R^2 for that one battery and dominates
        # the mean (see e.g. B0041 at H=50: R^2 = -402 on its own). Median
        # is the robust summary across batteries and is what should be
        # compared across horizons.
        "lobo_mean_r2_vs_own_mean": round(cv["r2_vs_own_mean"].mean(), 4) if n_testable else np.nan,
        "lobo_median_r2_vs_own_mean": round(cv["r2_vs_own_mean"].median(), 4) if n_testable else np.nan,
        "lobo_pct_beating_baseline": round((cv["r2_vs_own_mean"] > 0).mean(), 3) if n_testable else np.nan,
        "lobo_median_spearman_rho": round(cv["spearman_rho"].median(), 4) if n_testable else np.nan,
        "lobo_pct_positive_rho": round((cv["spearman_rho"] > 0).mean(), 3) if n_testable else np.nan,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-csv",
        type=str,
        default="reports/metrics/continuous_model_training_data.csv",
        help="Cached cycle-level table (cell_id, cycle, capacity_ah, cohort, trailing_avg_temp).",
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 10, 20, 50])
    parser.add_argument("--out-dir", type=str, default="reports/metrics")
    args = parser.parse_args()

    cyc = pd.read_csv(args.training_csv)
    print(f"Loaded cached table: {len(cyc)} rows, {cyc.cell_id.nunique()} batteries, {cyc.cohort.nunique()} cohorts")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_horizon(cyc, h, out_dir) for h in args.horizons]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "horizon_regression_summary.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Horizon comparison (the table that matters) ===")
    print(
        summary_df[
            [
                "horizon_cycles",
                "n_batteries",
                "n_batteries_dropped",
                "in_sample_r2",
                "trailing_avg_temp_p",
                "lobo_mean_r2_vs_own_mean",
                "lobo_median_r2_vs_own_mean",
                "lobo_pct_beating_baseline",
                "lobo_median_spearman_rho",
                "lobo_pct_positive_rho",
            ]
        ].to_string(index=False)
    )
    print(f"\nWritten to {out_dir}/horizon_regression_summary.csv and per-horizon LOBO CV files")


if __name__ == "__main__":
    main()
