"""Measure how the leave-one-cohort-out penalty depends on cohort coverage.

    python scripts/run_coverage_sweep.py
    python scripts/run_coverage_sweep.py --repeats 8 --cell-budget 9

Writes to reports/metrics/:

    coverage_sweep.csv          one row per (dataset, coverage level, repeat, method)
    coverage_sweep_summary.csv  median gap per coverage level
    coverage_sweep.md           rendered, for the report

WHAT THIS TESTS
---------------
Three earlier runs produced three mutually inconsistent method rankings, and
five claims drawn from them were withdrawn (ADR 0008, ADR 0009). The surviving
hypothesis is that the LOBO-to-LOCO gap is not a property of a method but of
how many cohorts remain in training — so a LOCO number without its cohort
coverage is uninterpretable.

This sweep holds the number of cells fixed and varies only the number of
cohorts they span. If the gap closes as coverage grows, on two datasets with
disjoint feature sets, the hypothesis survives.

Both datasets are swept separately and never pooled: NASA carries temperature,
SOC and behavioural flags; CALCE records none of them. Pooling would require
imputing most of both feature sets and the sweep would measure the imputation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.benchmarks import add_targets, get, load_all  # noqa: E402
from src.bms.benchmarks.coverage import (  # noqa: E402
    gap_vs_coverage_correlation,
    partial_gap_correlation,
    summarise_sweep,
    sweep_cohort_coverage,
)

REPO_ROOT = Path(__file__).parent.parent
METRICS = REPO_ROOT / "reports" / "metrics"

# NASA's behavioural columns; CALCE has none of them and uses electrical ones.
NASA_FEATURES = ("avg_temp", "max_temp", "avg_stress", "deep_discharge_duration",
                 "aggressive_discharge_count", "avg_soc", "cycle")
CALCE_FEATURES = ("mean_voltage_v", "min_voltage_v", "mean_current_a",
                  "min_current_a", "resistance_ohm", "cycle_duration_s", "cycle")

# A deliberately small, fast method set. The sweep refits across dozens of
# subframes, and the question is about the *gap*, not about which method wins.
#
# `elasticnet` is excluded despite being the obvious linear choice: its
# internal five-fold CV makes it roughly forty times slower than the
# alternatives here, and tuning inside every fold would add a second source of
# variation to a measurement already noisy across cohort draws.
#
# `random_forest` is excluded for the same reason at greater cost: 300 trees
# refitted across every fold of every subframe did not finish a single dataset
# in eight minutes. `svr_rbf` supplies the nonlinear half of the comparison at
# a fraction of that, and the sweep measures the LOBO-to-LOCO *gap* rather
# than which model wins, so a baseline plus one nonlinear learner is enough.
SWEEP_METHODS = ("age_linear", "svr_rbf")

FRAMES = (
    ("nasa", METRICS / "continuous_model_training_data.csv", NASA_FEATURES),
    ("calce", METRICS / "calce_cycle_level.csv", CALCE_FEATURES),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="soh")
    parser.add_argument(
        "--cell-budget", type=int, default=None,
        help="Cap cells per subframe. Default None uses every cell in the "
             "chosen cohorts; see coverage.py for why capping is not usable "
             "at this range and cell count is controlled statistically.",
    )
    parser.add_argument("--repeats", type=int, default=12,
                        help="Cohort subsets per coverage level. The gap "
                             "varies a lot between draws, so this needs to be "
                             "well above a handful.")
    parser.add_argument("--methods", nargs="*", default=list(SWEEP_METHODS))
    parser.add_argument("--out", type=Path, default=METRICS)
    args = parser.parse_args()

    load_all()
    methods = [get(name) for name in args.methods]
    args.out.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []

    for name, path, features in FRAMES:
        if not path.exists():
            print(f"SKIP {name}: no frame at {path}")
            continue

        data = add_targets(pd.read_csv(path))
        usable = data.dropna(subset=[args.target])
        print(f"\n=== {name} ===")
        print(f"  {usable['cell_id'].nunique()} cells, "
              f"{usable['cohort'].nunique()} cohorts, {len(usable)} rows")

        missing = [f for f in features if f not in data.columns]
        if missing:
            print(f"  SKIP: missing features {missing}")
            continue

        table = sweep_cohort_coverage(
            data, methods, target=args.target, features=features,
            dataset=name, cell_budget=args.cell_budget, repeats=args.repeats,
            verbose=True,
        )
        if table.empty:
            print("  no usable subframes")
            continue

        tables.append(table)
        summary = summarise_sweep(table)
        print()
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        raw = gap_vs_coverage_correlation(table)
        adj = partial_gap_correlation(table)
        print(f"  raw      Spearman(n_cohorts, gap) = {raw['rho']:+.3f} "
              f"(p={raw['p']:.4f}, n={raw['n']:.0f})")
        print(f"  adjusted for cell count           = {adj['rho']:+.3f} "
              f"(p={adj['p']:.4f})")
        print(f"  (gap vs cell count alone          = {adj['rho_gap_vs_cells']:+.3f})")

    if not tables:
        raise SystemExit("No dataset produced a sweep.")

    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.out / "coverage_sweep.csv", index=False)

    summary = summarise_sweep(combined)
    summary.to_csv(args.out / "coverage_sweep_summary.csv", index=False)

    lines = ["# Cohort-coverage sweep", "",
             "Generated by `scripts/run_coverage_sweep.py`.", "",
             "Cohort count and cell count rise together, so the adjusted figure "
             "(cell count regressed out) is the one that carries the claim.", "",
             "```", summary.to_string(index=False), "```", ""]
    for name in combined["dataset"].unique():
        subset = combined[combined["dataset"] == name]
        raw = gap_vs_coverage_correlation(subset)
        adj = partial_gap_correlation(subset)
        lines.append(
            f"- **{name}** (n = {raw['n']:.0f}): raw rho = {raw['rho']:+.3f} "
            f"(p = {raw['p']:.4f}); adjusted for cell count = {adj['rho']:+.3f} "
            f"(p = {adj['p']:.4f}); gap vs cell count alone = "
            f"{adj['rho_gap_vs_cells']:+.3f}"
        )
    (args.out / "coverage_sweep.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {args.out / 'coverage_sweep.csv'} ({len(combined)} rows)")
    print(f"Wrote {args.out / 'coverage_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
