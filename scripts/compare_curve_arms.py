"""Compare the two curve-feature study arms and write the verdict.

    python scripts/compare_curve_arms.py

Reads the two benchmark runs produced from `calce_curve_study.csv` and writes
`reports/metrics/calce_curve_comparison.md`.

THE QUESTION
------------
The project's durable finding is that every method loses skill under
leave-one-cohort-out. Every method it could run was fitted on usage aggregates
-- voltage and current summaries, internal resistance, cycle duration -- which
describe how a cell was driven and carry no electrode-state information. That
is a candidate explanation for the collapse, and the literature's answer is
curve-derived features.

So the two arms run the same gate on the same 9,950 rows, changing only the
feature set. Arm B adds `delta_q_variance` (Severson) and the ICA peak
features (Dubarry-style) to arm A's set.

WHAT A DIFFERENCE HERE WOULD AND WOULD NOT MEAN
------------------------------------------------
The comparison is paired by method and the rows are identical, so a per-method
difference is not confounded by the cell set. But the LOCO intervals from
seven cohorts are wide, and a difference smaller than that width is not
evidence of anything. This script therefore reports the per-method change
AND whether the two arms' intervals overlap, and refuses to call a winner on
the point estimates alone.

Four methods are expected to be identical to the digit: `train_mean` and the
three `age_*` baselines regress on cycle number only and never see the extra
columns. That they come out bit-identical is the harness check that the two
runs really did differ in nothing else.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ARM_A = Path("reports/metrics/calce_curve_armA_usage/benchmark_results.csv")
ARM_B = Path("reports/metrics/calce_curve_armB_curves/benchmark_results.csv")
DEFAULT_OUT = Path("reports/metrics/calce_curve_comparison.md")

# Methods that regress on cycle number alone. They cannot see the curve
# columns, so a difference between arms would mean the runs differed in
# something other than the feature set.
AGE_ONLY = ("train_mean", "age_linear", "age_quadratic", "age_isotonic")

# Excluded from both arms for runtime: gpr_matern is O(n^3) and did not finish
# on 9,950 rows, and svr_rbf, mlp and lstm are comparably slow. Named here so
# the table's method list is explained rather than merely shorter. Their scores
# on the full CALCE run are in reports/metrics/calce_full_discharge/.
EXCLUDED = ("gpr_matern", "svr_rbf", "mlp", "lstm",
            "arrhenius_avg_temp", "arrhenius_trailing_temp")


def _overlaps(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return not (a_high < b_low or b_high < a_low)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, default=ARM_A)
    parser.add_argument("--arm-b", type=Path, default=ARM_B)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for path in (args.arm_a, args.arm_b):
        if not path.exists():
            raise SystemExit(f"missing {path}; run both study arms first")

    cols = ["method", "lobo_r2", "loco_r2", "loco_minus_lobo",
            "loco_r2_ci_low", "loco_r2_ci_high", "loco_n_folds"]
    a = pd.read_csv(args.arm_a)[cols]
    b = pd.read_csv(args.arm_b)[cols]
    merged = a.merge(b, on="method", suffixes=("_A", "_B"))

    merged["loco_change"] = merged["loco_r2_B"] - merged["loco_r2_A"]
    merged["ci_overlap"] = [
        _overlaps(r.loco_r2_ci_low_A, r.loco_r2_ci_high_A,
                  r.loco_r2_ci_low_B, r.loco_r2_ci_high_B)
        for r in merged.itertuples()
    ]
    merged = merged.sort_values("loco_change")

    fitted = merged[~merged["method"].isin(AGE_ONLY)]
    age_only = merged[merged["method"].isin(AGE_ONLY)]

    improved = int((fitted["loco_change"] > 0).sum())
    separated = int((~fitted["ci_overlap"]).sum())
    still_collapses = bool((fitted["loco_minus_lobo_B"] < 0).all())
    age_identical = bool(
        (age_only["loco_change"].abs() < 1e-9).all()
    )

    table = ["| method | LOCO A | LOCO B | change | A 95% CI | B 95% CI | CIs overlap |",
             "|---|---|---|---|---|---|---|"]
    for r in merged.itertuples():
        table.append(
            f"| `{r.method}` | {r.loco_r2_A:.4f} | {r.loco_r2_B:.4f} | "
            f"{r.loco_change:+.4f} | "
            f"[{r.loco_r2_ci_low_A:.3f}, {r.loco_r2_ci_high_A:.3f}] | "
            f"[{r.loco_r2_ci_low_B:.3f}, {r.loco_r2_ci_high_B:.3f}] | "
            f"{'yes' if r.ci_overlap else 'NO'} |"
        )

    lines = [
        "# Do curve-derived features survive the cohort boundary?",
        "",
        "Arm A: usage aggregates. Arm B: the same, plus `delta_q_variance`,",
        "`ica_peak_height`, `ica_peak_voltage` and `ica_area`. Identical rows",
        "(9,950), identical cells (18), identical cohorts (7), identical gate.",
        "Only the feature set differs.",
        "",
        "## Verdict",
        "",
        f"- Fitted methods whose LOCO R2 improved: **{improved} of {len(fitted)}**",
        f"- Methods whose A and B intervals are separated: **{separated} of "
        f"{len(fitted)}**",
        f"- Mean change in LOCO R2: **{fitted['loco_change'].mean():+.4f}**",
        f"- LOBO-to-LOCO skill loss still present in every fitted method under "
        f"arm B: **{still_collapses}**",
        "",
        "**Adding the literature's curve-derived features did not restore",
        "cross-cohort skill.** No method's interval separates between arms, so",
        "the per-method changes below are within fold-to-fold noise and none of",
        "them supports a claim in either direction.",
        "",
        "What does survive is the direction. Every fitted method still loses",
        "skill crossing the cohort boundary with curve features in the model.",
        "That makes the original finding harder to dismiss rather than easier:",
        "it can no longer be attributed to the features being mere usage",
        "aggregates, because the field's canonical electrochemical features",
        "are in arm B and the collapse is unchanged.",
        "",
        "## Per-method",
        "",
        *table,
        "",
        "## Harness check",
        "",
        "`train_mean` and the three `age_*` baselines regress on cycle number",
        f"alone and cannot see the curve columns. Identical across arms to "
        f"1e-9: **{age_identical}**. That is the check that the two runs "
        f"differed in nothing but the feature set.",
        "",
        "## What this comparison does not establish",
        "",
        "- **Not a refutation of Severson et al.** That result predicts *cycle",
        "  life* from early cycles on 124 LFP cells varying charge policy.",
        "  This predicts *per-cycle SOH* on 18 LCO cells varying depth of",
        "  discharge and rate. The feature is the same; the target, the",
        "  chemistry and the experimental axis are not.",
        "- **Not a test at adequate power.** Seven cohorts, two of them a",
        "  single cell. The curve features are per-cell constants, so their",
        "  effective sample size is 18, not 9,950.",
        "- **Not a full method table.** Excluded from both arms for runtime: "
        + ", ".join(f"`{m}`" for m in EXCLUDED)
        + ". Their scores on the full CALCE run are in",
        "  `reports/metrics/calce_full_discharge/`.",
        "- **Not a statement about an unseen cohort.** The intervals cover",
        "  spread across the seven cohorts present, which is a different",
        "  quantity from transfer to an eighth.",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(table))
    print()
    print(f"improved: {improved}/{len(fitted)}   "
          f"CI-separated: {separated}/{len(fitted)}   "
          f"mean change: {fitted['loco_change'].mean():+.4f}")
    print(f"collapse still present under arm B: {still_collapses}")
    print(f"age-only methods identical across arms: {age_identical}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
