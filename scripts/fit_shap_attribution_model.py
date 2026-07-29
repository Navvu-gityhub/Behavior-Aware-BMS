"""Fit a non-additive model to MEASURED capacity fade and attribute it with SHAP.

PURPOSE
-------
`explain.attribution` answers "which term drove this battery's *rule-based
score*?" — exactly, but circularly: it can only ever tell us what the
hand-written rules already assert. It cannot tell us whether those rules
weight the right things.

This script answers the harder question: **which behavioural features
actually predict measured capacity fade in the NASA data, and does that
ranking agree with the weights the rules assign?** If the rules put their
largest penalty on aggressive discharge (30 points) but SHAP says trailing
temperature carries the signal, that is a concrete, evidence-backed
criticism of the scoring system, and the point of doing this at all.

THE GATE: SHAP EXPLAINS THE MODEL, NOT THE WORLD
------------------------------------------------
SHAP attributes a model's predictions to its inputs. It says nothing about
whether those predictions are any good. Running SHAP on a model with no
out-of-sample skill produces a confident-looking importance ranking that
describes how the model fit noise — which is worse than no analysis,
because it looks like evidence.

So this script validates BEFORE it attributes, and reports the SHAP ranking
with an explicit verdict attached:

  * If the model beats its baselines out-of-sample, the SHAP ranking is
    reported as evidence about degradation drivers.
  * If it does not, the SHAP ranking is still reported — suppressing it
    would be its own kind of dishonesty — but labelled as describing the
    model's internal fitting behaviour only, and NOT usable as a claim
    about physical degradation drivers.

The verdict is computed, not chosen in advance.

VALIDATION DESIGN
-----------------
Two splits, testing two different kinds of generalisation:

  * **LOBO** (leave-one-battery-out): can the model predict a cell it has
    never seen, from cells cycled under the same protocol? The established
    check in this project and in the SOH literature.
  * **LOCO** (leave-one-cohort-out): can it predict cells under an
    experimental protocol it has never seen? This is the harder test, and
    the one that stands in for the cross-dataset validation this project
    cannot run — see docs/adr/0001-cross-dataset-validation.md for why the
    supplied CALCE data makes a genuine NASA->CALCE test impossible.

Two baselines, because "R-squared > 0" is not a meaningful bar for a target
this noisy:

  * **global mean**: predict the training set's mean loss for everything.
    Honest — uses no test information.
  * **own mean**: predict the held-out group's OWN mean. This is an ORACLE
    baseline; it uses information the model does not have. It is included
    because it is the bar the earlier calibration work used, and because
    beating it would be a genuinely strong result. Failing to beat it is
    expected, not damning — but it must be reported as an oracle, not
    quietly presented as a fair comparison.

FEATURE SELECTION AND EXCLUDED LEAKAGE
--------------------------------------
Three columns present in the training data are deliberately NOT used as
features, each for a specific reason:

  * `capacity_ah` — the target is derived from it (loss = change in
    capacity). Including it is direct target leakage.
  * `battery_age_factor` — defined as `cycle / max(cycle for this battery)`.
    The denominator is the battery's eventual final cycle, which is not
    knowable at prediction time. Using it leaks the cell's lifetime into
    every early-cycle row. This is subtle and was not flagged in the earlier
    OLS work, where it also appears as an available column.
  * `n_rows` — a telemetry-density artifact of the source files, not a
    property of the battery.

`cohort` is not a feature either: it is the grouping variable for LOCO, and
feeding it in would let the model memorise protocol intercepts, which is
exactly what LOCO is designed to catch.

Run:  python scripts/fit_shap_attribution_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import GradientBoostingRegressor

from src.bms.risk.stress_score import RISK_TERMS

TRAINING_DATA = Path("reports/metrics/continuous_model_training_data.csv")
METRICS_DIR = Path("reports/metrics")

# Behavioural + environmental features only. See module docstring for the
# three excluded leakage columns and why `cohort` is held out as a group.
FEATURES = [
    "avg_stress",
    "avg_temp",
    "max_temp",
    "trailing_avg_temp",
    "fast_charge_duration",
    "deep_discharge_duration",
    "high_temp_duration",
    "aggressive_discharge_count",
    "trailing_deep_discharge_duration",
    "trailing_aggressive_discharge_count",
    "avg_soc",
    "ambient_temperature_c",
    "cycle",
]

TARGET = "capacity_loss"

# Deliberately shallow and heavily regularised. With 2,682 rows, 34 cells and
# a target whose per-cycle noise exceeds its signal (see calibration_report
# Section 10), an unconstrained booster will fit measurement noise and
# produce a confident, meaningless SHAP ranking. These settings are a prior
# against that, not a tuned optimum — tuning them on this data would itself
# be a form of overfitting given the sample size.
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)


def _r2(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> float:
    """R-squared of predictions against an arbitrary baseline predictor."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - baseline) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def grouped_cv(data: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Leave-one-group-out CV, scored against both baselines."""
    rows = []
    for group in sorted(data[group_col].unique()):
        train = data[data[group_col] != group]
        test = data[data[group_col] == group]
        if len(test) < 10 or len(train) < 50:
            continue

        model = GradientBoostingRegressor(**MODEL_PARAMS)
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])

        y = test[TARGET].to_numpy(dtype=float)
        global_mean = np.full_like(y, float(train[TARGET].mean()))
        own_mean = np.full_like(y, float(y.mean()))

        rows.append(
            {
                "held_out": group,
                "n_test": len(test),
                "mae": float(np.mean(np.abs(y - pred))),
                "r2_vs_global_mean": _r2(y, pred, global_mean),
                "r2_vs_own_mean_oracle": _r2(y, pred, own_mean),
                "spearman_rho": float(pd.Series(pred).corr(pd.Series(y), method="spearman")),
            }
        )
    return pd.DataFrame(rows)


def summarise(cv: pd.DataFrame, label: str) -> dict:
    beats_global = float((cv["r2_vs_global_mean"] > 0).mean())
    beats_oracle = float((cv["r2_vs_own_mean_oracle"] > 0).mean())
    return {
        "split": label,
        "n_folds": int(len(cv)),
        "median_r2_vs_global_mean": float(cv["r2_vs_global_mean"].median()),
        "median_r2_vs_own_mean_oracle": float(cv["r2_vs_own_mean_oracle"].median()),
        "pct_folds_beating_global_mean": beats_global,
        "pct_folds_beating_own_mean_oracle": beats_oracle,
        "median_spearman_rho": float(cv["spearman_rho"].median()),
        "median_mae_ah": float(cv["mae"].median()),
    }


def main() -> None:
    if not TRAINING_DATA.exists():
        raise SystemExit(
            f"{TRAINING_DATA} not found. Regenerate it with "
            "scripts/calibrate_cohort_cycle_level.py against the NASA dataset."
        )

    data = pd.read_csv(TRAINING_DATA).dropna(subset=FEATURES + [TARGET, "cohort", "cell_id"])
    print(f"Loaded {len(data)} cycle observations, "
          f"{data['cell_id'].nunique()} cells, {data['cohort'].nunique()} cohorts")

    # --- Validate first, attribute second -------------------------------
    lobo = grouped_cv(data, "cell_id")
    loco = grouped_cv(data, "cohort")
    lobo.to_csv(METRICS_DIR / "shap_model_lobo_cv.csv", index=False)
    loco.to_csv(METRICS_DIR / "shap_model_loco_cv.csv", index=False)

    summary = pd.DataFrame([summarise(lobo, "LOBO"), summarise(loco, "LOCO")])
    summary.to_csv(METRICS_DIR / "shap_model_validation.csv", index=False)
    print("\n=== Out-of-sample validation ===")
    print(summary.to_string(index=False))

    # The gate. A model earns an interpretable SHAP ranking only by beating
    # the honest (global-mean) baseline on the majority of folds in BOTH
    # splits. The oracle baseline is reported but is not part of the gate —
    # requiring a model to beat a predictor that has seen the test group's
    # mean would be an unfair bar.
    lobo_ok = summary.loc[0, "pct_folds_beating_global_mean"] > 0.5
    loco_ok = summary.loc[1, "pct_folds_beating_global_mean"] > 0.5
    has_skill = bool(lobo_ok and loco_ok)

    verdict = (
        "GENERALISES: SHAP ranking is admissible as evidence about degradation drivers."
        if has_skill
        else "NO OUT-OF-SAMPLE SKILL: SHAP ranking describes this model's internal "
             "fitting behaviour only. It is NOT evidence about physical degradation "
             "drivers and must not be cited as such."
    )
    print(f"\n=== GATE ===\n{verdict}")

    # --- SHAP attribution -----------------------------------------------
    import shap

    model = GradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(data[FEATURES], data[TARGET])

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data[FEATURES])

    importance = (
        pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                "mean_shap": shap_values.mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance["importance_rank"] = importance.index + 1
    importance["share_of_total_attribution"] = (
        importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()
    )
    importance.to_csv(METRICS_DIR / "shap_feature_importance.csv", index=False)

    print("\n=== SHAP feature importance (measured capacity fade) ===")
    print(importance[["importance_rank", "feature", "mean_abs_shap",
                      "share_of_total_attribution"]].to_string(index=False))

    # --- Rules vs data: do the hand-picked weights match the evidence? ---
    # Map each rule term to the model feature it is meant to capture, then
    # compare the rules' implied importance ranking to SHAP's.
    rule_to_feature = {
        "stress": "avg_stress",
        "temperature": "avg_temp",
        "deep_discharge": "deep_discharge_duration",
        "fast_charge": "fast_charge_duration",
        "aggressive_discharge": "aggressive_discharge_count",
    }
    # A term's implied importance is its maximum possible penalty — the most
    # the rules will ever let it move the score.
    rule_max_penalty = {
        t.name: float(np.max(t.contribution(pd.Series([0, 25, 55, 75, 150, 600])).to_numpy()))
        for t in RISK_TERMS
    }

    comparison = []
    imp_lookup = importance.set_index("feature")
    for term, feature in rule_to_feature.items():
        comparison.append(
            {
                "rule_term": term,
                "mapped_feature": feature,
                "rule_max_penalty_points": rule_max_penalty[term],
                "shap_mean_abs": float(imp_lookup.loc[feature, "mean_abs_shap"]),
                "shap_rank": int(imp_lookup.loc[feature, "importance_rank"]),
            }
        )
    comp = pd.DataFrame(comparison)
    comp["rule_rank"] = comp["rule_max_penalty_points"].rank(ascending=False, method="min").astype(int)
    comp = comp.sort_values("rule_rank")

    rho = float(comp["rule_rank"].corr(comp["shap_rank"], method="spearman"))
    comp.to_csv(METRICS_DIR / "shap_vs_rule_weights.csv", index=False)

    print("\n=== Rule weights vs SHAP importance ===")
    print(comp[["rule_term", "rule_rank", "shap_rank",
                "rule_max_penalty_points", "shap_mean_abs"]].to_string(index=False))
    print(f"\nSpearman rank correlation (rule ranking vs SHAP ranking): {rho:.3f}")

    (METRICS_DIR / "shap_attribution_verdict.json").write_text(
        json.dumps(
            {
                "model": "GradientBoostingRegressor",
                "model_params": MODEL_PARAMS,
                "n_observations": int(len(data)),
                "n_cells": int(data["cell_id"].nunique()),
                "n_cohorts": int(data["cohort"].nunique()),
                "features_used": FEATURES,
                "excluded_for_leakage": ["capacity_ah", "battery_age_factor", "n_rows"],
                "lobo_beats_global_mean_pct": summary.loc[0, "pct_folds_beating_global_mean"],
                "loco_beats_global_mean_pct": summary.loc[1, "pct_folds_beating_global_mean"],
                "has_out_of_sample_skill": has_skill,
                "verdict": verdict,
                "rule_vs_shap_rank_spearman": rho,
                "top_shap_feature": importance.loc[0, "feature"],
            },
            indent=2,
        )
    )
    print(f"\nWrote verdict to {METRICS_DIR / 'shap_attribution_verdict.json'}")


if __name__ == "__main__":
    main()
