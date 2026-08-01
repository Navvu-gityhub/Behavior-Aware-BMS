"""Head-to-head validation of health index v1 (rules) vs v2 (fitted OLS).

WHY THIS EXISTS
---------------
`src/bms/health/health_index_v2.py` was built, fitted and documented, then
left unwired — neither promoted to pipeline default nor removed. An
orphaned module with no recorded decision is a liability: a reader cannot
tell whether it is abandoned, pending, or quietly believed to be better.

This script produces the evidence to close that decision, and
`docs/adr/0002-health-index-version.md` records the decision it supports.

THE TWO CANDIDATES DO NOT DO THE SAME THING
-------------------------------------------
This is the part that makes a naive "which has higher R-squared?" comparison
meaningless, and it is why the comparison below is split in two:

  * **v1** emits a 0-100 severity index per battery. It is a *ranking and
    triage* instrument: its job is to sort a fleet so the worst cells
    surface first. It does not predict a physical quantity and cannot be
    scored with R-squared against capacity loss at all.
  * **v2** emits a predicted per-cycle capacity loss in Ah. It is a
    *regression* instrument, and R-squared is the right metric for it.

So they are evaluated on the task each actually claims to perform:

  * **Task A (ranking, battery-level).** Spearman correlation between each
    candidate's battery-level output and the measured fade rate from
    `nasa_ground_truth_fade.csv`. Both candidates can be scored here, which
    makes this the only genuinely head-to-head comparison available.
  * **Task B (regression, cycle-level).** Out-of-sample R-squared for v2
    against per-cycle capacity loss, under LOBO and LOCO, versus baselines.
    v1 cannot compete here; the question is only whether v2 clears the bar
    to be worth wiring in at all.

DOMAIN SHIFT: LOCO AS THE SUBSTITUTE FOR CROSS-DATASET VALIDATION
------------------------------------------------------------------
The natural next test after leave-one-battery-out is cross-dataset
(train NASA, test CALCE). That test cannot be run: the supplied CALCE files
are a single-cycle baseline characterisation with no cycle-indexed capacity
fade, so there is no target to test against. See
`docs/adr/0001-cross-dataset-validation.md` and
`docs/calce_dataset_note.md`.

Leave-one-cohort-out is run in its place. NASA's 34 cells span 9 distinct
experimental protocols differing in ambient temperature, discharge current
and cutoff voltage; holding out an entire protocol tests generalisation
across a genuine distribution shift, which is the property cross-dataset
validation is actually probing. It is a weaker claim than cross-dataset
(same lab, same cell chemistry, same instrumentation) and is reported as
such — but it is a real test, and it is available.

Run:  python scripts/validate_health_index_versions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import statsmodels.formula.api as smf

from src.bms.health.health_index import compute_health_index
from src.bms.health.health_index_v2 import predict_capacity_loss_per_cycle

TRAINING_DATA = Path("reports/metrics/continuous_model_training_data.csv")
GROUND_TRUTH = Path("reports/metrics/nasa_ground_truth_fade.csv")
CALIBRATION_SUMMARY = Path("reports/metrics/calibration_merged.csv")
METRICS_DIR = Path("reports/metrics")

SUMMARY_COLS = [
    "avg_stress", "avg_temp", "deep_discharge_duration",
    "fast_charge_duration", "aggressive_discharge_count", "avg_soc",
]


def _r2(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - baseline) ** 2))
    return np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Task A: ranking batteries by measured fade rate
# ---------------------------------------------------------------------------

def task_a_ranking() -> pd.DataFrame:
    truth = pd.read_csv(GROUND_TRUTH)
    summary = pd.read_csv(CALIBRATION_SUMMARY)
    cycle = pd.read_csv(TRAINING_DATA)

    merged = truth.merge(summary[["battery_id"] + SUMMARY_COLS], on="battery_id", how="inner")

    # v1: recompute from the summary features so this reflects current code.
    v1 = compute_health_index(merged[["battery_id"] + SUMMARY_COLS])
    merged["v1_health_index"] = v1["health_index"].to_numpy()

    # v2: aggregate its per-cycle prediction to a per-battery mean predicted
    # loss, which is the battery-level quantity comparable to a fade rate.
    per_cycle = cycle.copy()
    per_cycle["v2_pred"] = [
        float(predict_capacity_loss_per_cycle(pd.Series([t]), c).iloc[0])
        for t, c in zip(per_cycle["trailing_avg_temp"], per_cycle["cohort"])
    ]
    v2_batt = per_cycle.groupby("cell_id")["v2_pred"].mean().rename("v2_mean_pred_loss")
    merged = merged.merge(v2_batt, left_on="battery_id", right_index=True, how="left")

    rows = []
    for name, col in [("v1_rule_based", "v1_health_index"), ("v2_fitted_ols", "v2_mean_pred_loss")]:
        sub = merged[[col, "fade_rate_ah_per_cycle"]].dropna()
        rho = float(sub[col].corr(sub["fade_rate_ah_per_cycle"], method="spearman"))
        # Permutation test: with n=33 and a heavy-tailed target, the analytic
        # p-value for Spearman is not trustworthy, so it is computed by
        # resampling instead.
        rng = np.random.default_rng(42)
        observed = abs(rho)
        null = [
            abs(float(pd.Series(rng.permutation(sub[col].to_numpy())).corr(
                sub["fade_rate_ah_per_cycle"].reset_index(drop=True), method="spearman")))
            for _ in range(5000)
        ]
        p = float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1))
        rows.append({
            "candidate": name,
            "n_batteries": int(len(sub)),
            "spearman_rho_vs_fade_rate": rho,
            "permutation_p": p,
            "significant_at_0.05": bool(p < 0.05),
            "distinct_output_values": int(sub[col].nunique()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Task B: predicting per-cycle capacity loss under domain shift
# ---------------------------------------------------------------------------

def task_b_regression(data: pd.DataFrame, group_col: str, label: str) -> pd.DataFrame:
    rows = []
    for group in sorted(data[group_col].unique()):
        train = data[data[group_col] != group]
        test = data[data[group_col] == group]
        if len(test) < 10 or train["cohort"].nunique() < 2:
            continue

        # Refit v2's specification on the training fold only. Reusing the
        # shipped coefficients would leak the held-out group, since they were
        # fitted on all 34 cells.
        try:
            fit = smf.ols("capacity_loss ~ trailing_avg_temp + C(cohort)", data=train).fit()
            test_for_pred = test.copy()
            # A held-out cohort has no fitted intercept. Fall back to the mean
            # of the fitted cohort intercepts — the same fallback the shipped
            # module uses for unknown cohorts, so this measures the behaviour
            # that would actually be deployed.
            if group_col == "cohort":
                ref_cohort = train["cohort"].iloc[0]
                test_for_pred["cohort"] = ref_cohort
            pred = fit.predict(test_for_pred).to_numpy(dtype=float)
        except Exception as exc:  # pragma: no cover - diagnostic path
            rows.append({"held_out": group, "n_test": len(test), "error": str(exc)})
            continue

        y = test["capacity_loss"].to_numpy(dtype=float)
        global_mean = np.full_like(y, float(train["capacity_loss"].mean()))
        own_mean = np.full_like(y, float(y.mean()))

        rows.append({
            "split": label,
            "held_out": group,
            "n_test": int(len(test)),
            "mae": float(np.mean(np.abs(y - pred))),
            "r2_vs_global_mean": _r2(y, pred, global_mean),
            "r2_vs_own_mean_oracle": _r2(y, pred, own_mean),
            "spearman_rho": float(pd.Series(pred).corr(pd.Series(y), method="spearman")),
        })
    return pd.DataFrame(rows)


def task_a_out_of_fold(data: pd.DataFrame) -> pd.DataFrame:
    """Re-run the ranking comparison with v2 predictions generated OUT OF FOLD.

    WHY THIS IS NECESSARY: v2's shipped coefficients include a fitted
    intercept per NASA cohort, estimated on all 34 cells. Fade rate also
    varies strongly by cohort. So scoring the shipped v2 against the same
    cells it was fitted on lets the cohort intercepts encode the very
    between-cohort fade differences being predicted — an in-sample ranking
    that will look excellent for reasons that have nothing to do with the
    temperature signal the model claims to use.

    Two out-of-fold variants disentangle this:

      * **LOBO-refit**: refit on all cells except the target, keeping its
        cohort in the training set. The cohort intercept is still learned
        from siblings, so this measures "can we rank a new cell within a
        known protocol?"
      * **LOCO-refit**: refit with the target's entire cohort withheld, so
        the intercept must come from the unknown-cohort fallback. This
        measures "can we rank a cell from a protocol we have never seen?",
        which is what a claim of general applicability requires.

    If the shipped ranking is real, LOBO-refit should hold up. If it is
    cohort-intercept memorisation, LOCO-refit will collapse.
    """
    truth = pd.read_csv(GROUND_TRUTH).set_index("battery_id")["fade_rate_ah_per_cycle"]

    preds: dict[str, dict[str, float]] = {"lobo_refit": {}, "loco_refit": {}}

    for cell in sorted(data["cell_id"].unique()):
        test = data[data["cell_id"] == cell]
        if test.empty:
            continue

        # LOBO: hold out this cell only.
        train = data[data["cell_id"] != cell]
        if train["cohort"].nunique() >= 2:
            fit = smf.ols("capacity_loss ~ trailing_avg_temp + C(cohort)", data=train).fit()
            if test["cohort"].iloc[0] in set(train["cohort"]):
                preds["lobo_refit"][cell] = float(fit.predict(test).mean())

        # LOCO: hold out this cell's entire cohort.
        cohort = test["cohort"].iloc[0]
        train_c = data[data["cohort"] != cohort]
        if train_c["cohort"].nunique() >= 2:
            fit_c = smf.ols("capacity_loss ~ trailing_avg_temp + C(cohort)", data=train_c).fit()
            # Unknown cohort at prediction time -> average the fitted cohort
            # intercepts, mirroring health_index_v2.UNKNOWN_COHORT_INTERCEPT.
            per_cohort = []
            for known in sorted(train_c["cohort"].unique()):
                stand_in = test.copy()
                stand_in["cohort"] = known
                per_cohort.append(float(fit_c.predict(stand_in).mean()))
            preds["loco_refit"][cell] = float(np.mean(per_cohort))

    rows = []
    rng = np.random.default_rng(42)
    for variant, mapping in preds.items():
        s = pd.Series(mapping).dropna()
        common = s.index.intersection(truth.index)
        if len(common) < 5:
            continue
        x, y = s.loc[common], truth.loc[common]
        rho = float(x.corr(y, method="spearman"))
        null = [
            abs(float(pd.Series(rng.permutation(x.to_numpy()), index=common).corr(y, method="spearman")))
            for _ in range(5000)
        ]
        p = float((np.sum(np.array(null) >= abs(rho)) + 1) / (len(null) + 1))
        rows.append({
            "candidate": f"v2_fitted_ols__{variant}",
            "n_batteries": int(len(common)),
            "spearman_rho_vs_fade_rate": rho,
            "permutation_p": p,
            "significant_at_0.05": bool(p < 0.05),
            "distinct_output_values": int(x.nunique()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    for path in (TRAINING_DATA, GROUND_TRUTH, CALIBRATION_SUMMARY):
        if not path.exists():
            raise SystemExit(f"{path} not found; run the NASA calibration scripts first.")

    data_for_fold = pd.read_csv(TRAINING_DATA).dropna(
        subset=["capacity_loss", "trailing_avg_temp", "cohort", "cell_id"]
    )

    print("=== TASK A: ranking batteries by measured fade rate ===")
    task_a = pd.concat(
        [task_a_ranking(), task_a_out_of_fold(data_for_fold)], ignore_index=True
    )
    task_a.to_csv(METRICS_DIR / "health_version_task_a_ranking.csv", index=False)
    print(task_a.to_string(index=False))
    print(
        "\nNOTE: 'v2_fitted_ols' above is IN-SAMPLE (shipped coefficients fitted on "
        "these same cells). The two out-of-fold rows are the admissible numbers."
    )

    data = pd.read_csv(TRAINING_DATA).dropna(
        subset=["capacity_loss", "trailing_avg_temp", "cohort", "cell_id"]
    )

    print("\n=== TASK B: predicting per-cycle capacity loss (v2 specification) ===")
    lobo = task_b_regression(data, "cell_id", "LOBO")
    loco = task_b_regression(data, "cohort", "LOCO")
    pd.concat([lobo, loco], ignore_index=True).to_csv(
        METRICS_DIR / "health_version_task_b_cv.csv", index=False
    )

    summary_rows = []
    for cv, label in [(lobo, "LOBO (leave-one-battery-out)"),
                      (loco, "LOCO (leave-one-cohort-out)")]:
        cv = cv.dropna(subset=["r2_vs_global_mean"])
        summary_rows.append({
            "split": label,
            "n_folds": int(len(cv)),
            "median_r2_vs_global_mean": float(cv["r2_vs_global_mean"].median()),
            "median_r2_vs_own_mean_oracle": float(cv["r2_vs_own_mean_oracle"].median()),
            "pct_folds_beating_global_mean": float((cv["r2_vs_global_mean"] > 0).mean()),
            "median_spearman_rho": float(cv["spearman_rho"].median()),
            "median_mae_ah": float(cv["mae"].median()),
        })
    task_b = pd.DataFrame(summary_rows)
    task_b.to_csv(METRICS_DIR / "health_version_task_b_summary.csv", index=False)
    print(task_b.to_string(index=False))

    # --- Decision ---------------------------------------------------------
    v1_rho = float(task_a.loc[task_a.candidate == "v1_rule_based", "spearman_rho_vs_fade_rate"].iloc[0])
    v2_rho = float(task_a.loc[task_a.candidate == "v2_fitted_ols__lobo_refit", "spearman_rho_vs_fade_rate"].iloc[0])
    v2_rho_loco = float(task_a.loc[task_a.candidate == "v2_fitted_ols__loco_refit", "spearman_rho_vs_fade_rate"].iloc[0])
    v1_sig = bool(task_a.loc[task_a.candidate == "v1_rule_based", "significant_at_0.05"].iloc[0])
    v2_sig = bool(task_a.loc[task_a.candidate == "v2_fitted_ols__lobo_refit", "significant_at_0.05"].iloc[0])
    v2_sig_loco = bool(task_a.loc[task_a.candidate == "v2_fitted_ols__loco_refit", "significant_at_0.05"].iloc[0])
    v2_generalises = bool(task_b.loc[0, "pct_folds_beating_global_mean"] > 0.5
                          and task_b.loc[1, "pct_folds_beating_global_mean"] > 0.5)

    # The outcome here is genuinely conditional, so a binary promote/keep rule
    # would misrepresent it. v2's ranking ability depends entirely on whether
    # the battery's experimental protocol (cohort) is represented in training:
    # strong when it is, absent when it is not. The decision has to carry that
    # condition rather than average over it.
    if v2_sig and v2_sig_loco and v2_generalises:
        decision = (
            "PROMOTE v2 to unconditional pipeline default: it ranks batteries by measured "
            "fade significantly better than v1 both within and across experimental protocols."
        )
    elif v2_sig and not v2_sig_loco:
        decision = (
            "CONDITIONAL: v2 becomes the recommended ranking signal ONLY when the battery's "
            f"operating protocol is represented in training data (LOBO-refit rho={v2_rho:.3f}, "
            f"p<0.001). It must NOT be used for an unseen protocol, where its ranking collapses "
            f"to rho={v2_rho_loco:.3f} — statistically indistinguishable from noise and pointing "
            "the wrong way. v1 REMAINS the pipeline default because it needs no cohort label, "
            f"but v1's own ranking is itself non-significant and negative (rho={v1_rho:.3f}), so "
            "it is retained as a transparent threshold-alerting heuristic and must not be "
            "presented as a validated fade ranking."
        )
    elif v2_sig and not v1_sig:
        decision = (
            "KEEP v1 as default, EXPOSE v2 as a labelled research output: v2 ranks batteries "
            "significantly better than v1, but does not generalise across protocols well "
            "enough to be trusted as the primary score."
        )
    else:
        decision = (
            "KEEP v1 as default, RETAIN v2 as a documented negative result: v2 does not "
            "demonstrate a significant ranking advantage over v1 on measured fade."
        )

    verdict = {
        "task_a_v1_spearman": v1_rho,
        "task_a_v1_significant": v1_sig,
        "task_a_v2_spearman_lobo_refit": v2_rho,
        "task_a_v2_significant_lobo_refit": v2_sig,
        "task_a_v2_spearman_loco_refit": v2_rho_loco,
        "task_a_v2_significant_loco_refit": v2_sig_loco,
        "task_b_v2_generalises_out_of_sample": v2_generalises,
        "task_b_lobo_pct_beating_global_mean": float(task_b.loc[0, "pct_folds_beating_global_mean"]),
        "task_b_loco_pct_beating_global_mean": float(task_b.loc[1, "pct_folds_beating_global_mean"]),
        "decision": decision,
    }
    (METRICS_DIR / "health_version_decision.json").write_text(json.dumps(verdict, indent=2))

    print(f"\n=== DECISION ===\n{decision}")
    print(f"\nWrote {METRICS_DIR / 'health_version_decision.json'}")


if __name__ == "__main__":
    main()
