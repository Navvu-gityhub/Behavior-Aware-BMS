"""Measure conformal coverage within protocol and across it.

    python scripts/run_coverage_study.py
    python scripts/run_coverage_study.py --alpha 0.05 --mondrian

Writes to reports/metrics/:

    coverage_folds.csv     per-held-out-group coverage, width and refusal rate
    coverage_summary.csv   per-split summary, including the worst group
    coverage_summary.md    rendered, for the report

WHAT THIS MEASURES
------------------
Split conformal prediction guarantees at least `1 - alpha` coverage when
calibration and test data are exchangeable. Deploying a model on a protocol it
was not calibrated on violates that assumption directly.

So this script asks the safety-case version of the question the rest of the
project asks in R-squared: *if the dashboard quotes a 90% interval for a cell
whose protocol it has never seen, what is that interval actually worth?*

READ THE WORST GROUP, NOT THE MEDIAN
------------------------------------
Measured result on the NASA frame, ElasticNet against `cumulative_fade`,
nominal 0.90:

    LOBO   median 0.970   worst 0.000 (B0045)              19% below nominal
    LOCO   median 0.824   worst 0.262 (COLD4C_2A_flagged)  56% below nominal

The medians would pass an audit that stopped there. The worst-group column
would not: one cell's 90% interval contained the truth zero times, and under
protocol shift a majority of held-out cohorts fall below nominal.

The aggregate is therefore not just uninformative, it is anti-informative: the
overcovered easy cohorts pull the average up and certify a system that fails
exactly where failure matters. `coverage_summary` reports `min_coverage`,
`worst_group` and `fraction_below_nominal` for that reason.

For contrast, `age_linear` puts only 11% of cohorts below nominal but needs
intervals 27% wider to do it. Tighter intervals are not free.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.benchmarks import add_targets, get, load_all
from src.bms.uncertainty import coverage_report, coverage_summary

REPO_ROOT = Path(__file__).parent.parent
TRAINING_DATA = REPO_ROOT / "reports" / "metrics" / "continuous_model_training_data.csv"
METRICS_DIR = REPO_ROOT / "reports" / "metrics"

DEFAULT_METHODS = ("age_linear", "elasticnet", "random_forest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=TRAINING_DATA)
    parser.add_argument("--target", default="cumulative_fade")
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="1 - alpha is the nominal coverage (default 0.1 -> 90%%).")
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--mondrian", action="store_true",
                        help="Cohort-conditional quantiles instead of pooled.")
    parser.add_argument("--out", type=Path, default=METRICS_DIR)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"No training frame at {args.data}.")

    args.out.mkdir(parents=True, exist_ok=True)
    load_all()

    data = add_targets(pd.read_csv(args.data)).dropna(subset=[args.target])
    print(f"{len(data)} rows, {data['cell_id'].nunique()} cells, "
          f"{data['cohort'].nunique()} cohorts; target '{args.target}', "
          f"nominal coverage {1 - args.alpha:.0%}")

    fold_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for name in args.methods:
        method = get(name)
        print(f"\n--- {name} ---", flush=True)
        fit_fn = method.fit_fn(target=args.target)

        folds = coverage_report(
            data, fit_fn, target=args.target, alpha=args.alpha,
            mondrian=args.mondrian,
        )
        folds.insert(0, "method", name)
        fold_frames.append(folds)

        summary = coverage_summary(folds)
        if summary.empty:
            print("  no completed folds")
            continue
        summary.insert(0, "method", name)
        summary_frames.append(summary)

        for _, row in summary.iterrows():
            print(
                f"  {row['split']:<5} median {row['median_coverage']:.3f}  "
                f"min {row['min_coverage']:.3f} ({row['worst_group']})  "
                f"below-nominal {row['fraction_below_nominal']:.0%}  "
                f"width {row['median_width']:.4f}"
            )

    if not fold_frames:
        raise SystemExit("No method produced a coverage result.")

    folds = pd.concat(fold_frames, ignore_index=True)
    folds.to_csv(args.out / "coverage_folds.csv", index=False)

    summaries = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summaries.to_csv(args.out / "coverage_summary.csv", index=False)

    (args.out / "coverage_summary.md").write_text(
        f"# Conformal coverage — target `{args.target}`, nominal "
        f"{1 - args.alpha:.0%}\n\n"
        "Generated by `scripts/run_coverage_study.py`.\n\n"
        "Read `min_coverage` and `worst_group`, not `median_coverage` — see\n"
        "the script docstring and ADR 0007 for why the median is\n"
        "anti-informative here.\n\n```\n"
        + summaries.to_string(index=False)
        + "\n```\n",
        encoding="utf-8",
    )

    print(f"\nWrote {args.out / 'coverage_folds.csv'} ({len(folds)} folds)")
    print(f"Wrote {args.out / 'coverage_summary.csv'}")


if __name__ == "__main__":
    main()
