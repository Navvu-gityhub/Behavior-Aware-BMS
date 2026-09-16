"""Measure how the leave-one-cohort-out penalty depends on cohort coverage.

    python scripts/run_coverage_sweep.py
    python scripts/run_coverage_sweep.py --repeats 8 --cell-budget 9
    python scripts/run_coverage_sweep.py --frames calce_full_discharge \
        --out-prefix coverage_sweep_full_discharge

Writes three files to reports/metrics/, named from --out-prefix:

    <prefix>.csv          one row per (dataset, coverage level, repeat, method)
    <prefix>_summary.csv  median gap per coverage level
    <prefix>.md           rendered, for the report

The default prefix `coverage_sweep` is a pinned artifact -- see the comment on
DEFAULT_PREFIX before changing which frames write into it.

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

# Every frame this sweep can run on, keyed by the name written to the
# `dataset` column. `--frames` selects; the default is the pair below.
FRAMES = {
    "nasa": (METRICS / "continuous_model_training_data.csv", NASA_FEATURES),
    "calce": (METRICS / "calce_cycle_level.csv", CALCE_FEATURES),
    # The ADR 0012 full-discharge derivation of the same CALCE cells, same
    # feature set, so a run on it differs from `calce` in the target only.
    "calce_full_discharge": (METRICS / "calce_full_discharge.csv", CALCE_FEATURES),
}

DEFAULT_FRAMES = ("nasa", "calce")

# `coverage_sweep.csv` is the artifact three published figures are pinned to
# (`tests/test_reported_numbers.py`: the LOCO IQR 0.724, the paired difference
# -0.351, and the 25% nonlinear win rate). All three are recomputed by pooling
# every row of that file, so an extra frame written into it would redefine
# them silently. A run over a different frame set therefore takes a different
# `--out-prefix` and leaves the pinned artifact alone.
DEFAULT_PREFIX = "coverage_sweep"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", default="soh")
    parser.add_argument(
        "--frames", nargs="*", default=list(DEFAULT_FRAMES), choices=list(FRAMES),
        help="Which frames to sweep. Default: nasa calce.",
    )
    parser.add_argument(
        "--out-prefix", default=DEFAULT_PREFIX,
        help="Basename for the three output files. Change it whenever "
             "--frames is changed: the default artifact carries pinned "
             "figures that pool across every row it holds.",
    )
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

    # Refuse rather than overwrite: writing a different frame set into the
    # default artifact would change three figures the manuscript quotes,
    # and the test suite would report it as a prose error rather than as
    # what it is.
    if tuple(args.frames) != DEFAULT_FRAMES and args.out_prefix == DEFAULT_PREFIX:
        raise SystemExit(
            f"--frames {' '.join(args.frames)} would overwrite "
            f"{DEFAULT_PREFIX}.csv, which pins the published LOCO IQR, paired "
            f"difference and win rate (all pooled over every row of it). "
            f"Pass --out-prefix to write a separate artifact."
        )

    load_all()
    methods = [get(name) for name in args.methods]
    args.out.mkdir(parents=True, exist_ok=True)

    tables: list[pd.DataFrame] = []

    for name in args.frames:
        path, features = FRAMES[name]
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
    table_path = args.out / f"{args.out_prefix}.csv"
    summary_path = args.out / f"{args.out_prefix}_summary.csv"
    report_path = args.out / f"{args.out_prefix}.md"

    combined.to_csv(table_path, index=False)

    summary = summarise_sweep(combined)
    summary.to_csv(summary_path, index=False)

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
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote {table_path} ({len(combined)} rows)")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
