"""Run the benchmark study: every registered method, every target, LOBO and LOCO.

    python scripts/run_benchmark_study.py
    python scripts/run_benchmark_study.py --targets cumulative_fade capacity_loss
    python scripts/run_benchmark_study.py --quick     # naive + linear only

Writes to reports/metrics/:

    benchmark_registry.csv        every registered method and its citation
    benchmark_cell_screen.csv     per-cell SOH admissibility, with reasons
    benchmark_signal_report.csv   noise ceiling per target
    benchmark_results.csv         the main results table
    benchmark_summary.md          a rendered summary for the report

RUNTIME
-------
Roughly 14 minutes per target on one core, dominated by refitting the tree
ensembles and the neural network on every one of 42 folds (33 cells + 9
cohorts). `--quick` restricts to the reference methods and finishes in
seconds; use it when checking wiring rather than producing results.

Refitting per fold is not optimisable away — it is the property that makes the
result valid, since reusing a fit across folds would leak the held-out group.

WHAT THIS SCRIPT IS FOR
-----------------------
`docs/final_report.md` reports that no candidate beat a constant predictor
against `capacity_loss`. This script re-runs that question against published
methods and against several target definitions, so the finding can be stated
precisely rather than broadly.

The distinction it exists to draw: "no method can predict degradation from
behaviour" and "no method can predict *this particular target*, which is
mostly measurement noise" are different claims with different consequences,
and the original study could not separate them.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must be set before joblib is first imported, hence before pandas/sklearn.
#
# joblib determines the physical core count by shelling out to a platform
# command. In restricted execution contexts — CI containers, sandboxed
# runners, some Windows shells — that subprocess call fails and prints a
# traceback that can take the run down with it. Pinning the count skips the
# probe entirely. The value only bounds joblib's default parallelism; every
# estimator in this benchmark is configured with n_jobs=1 anyway, because a
# benchmark whose numbers depend on the host's core count is not a benchmark.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.benchmarks import (
    add_targets,
    get,
    load_all,
    registry_frame,
    run_study,
    screen_cells_for_soh,
    signal_report,
)

REPO_ROOT = Path(__file__).parent.parent
TRAINING_DATA = REPO_ROOT / "reports" / "metrics" / "continuous_model_training_data.csv"
METRICS_DIR = REPO_ROOT / "reports" / "metrics"

QUICK_METHODS = ("train_mean", "age_linear", "age_quadratic", "age_isotonic", "elasticnet")

DEFAULT_TARGETS = ("capacity_loss", "cumulative_fade", "horizon_fade_10",
                   "horizon_fade_20", "horizon_fade_50")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=TRAINING_DATA)
    parser.add_argument("--targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--quick", action="store_true",
                        help="Run only the fast reference methods.")
    parser.add_argument(
        "--methods", nargs="*", default=None,
        help=(
            "Run only these registered methods, by name. A method not named "
            "is absent from the table entirely rather than shown as rejected, "
            "so any run using this must state which methods it excluded and "
            "why. Intended for a controlled comparison where two runs must "
            "share a method set, not for trimming a table until it reads well."
        ),
    )
    parser.add_argument(
        "--features", nargs="*", default=None,
        help=(
            "Override each method's default feature set. Required for datasets "
            "that do not carry NASA's behavioural columns — CALCE records "
            "neither temperature nor SOC, so its usable features are the "
            "electrical ones (voltage, current, internal resistance, cycle "
            "duration). Note that a study run with different features is not "
            "feature-comparable with the NASA results; it tests whether the "
            "LOBO-to-LOCO phenomenon reproduces, not whether one model transfers."
        ),
    )
    parser.add_argument("--out", type=Path, default=METRICS_DIR)
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"No training frame at {args.data}. This file is tracked in git "
            f"because it cannot be regenerated without the NASA cleaned_dataset "
            f"(7.2M rows, gitignored, not redistributable)."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    load_all()
    raw = pd.read_csv(args.data)
    print(f"Loaded {len(raw)} rows, {raw['cell_id'].nunique()} cells, "
          f"{raw['cohort'].nunique()} cohorts from {args.data.name}")

    screen = screen_cells_for_soh(raw)
    screen.to_csv(args.out / "benchmark_cell_screen.csv", index=False)
    excluded = screen[~screen["admissible"]]
    print(f"\nSOH admissibility: {int(screen['admissible'].sum())}/{len(screen)} cells")
    for _, row in excluded.iterrows():
        print(f"  EXCLUDED {row['cell_id']}: {row['reason']}")

    data = add_targets(raw)
    n_outliers = int(data["soh_outlier"].sum())
    print(f"Observation-level SOH outliers masked: {n_outliers}/{len(data)}")

    registry_frame().to_csv(args.out / "benchmark_registry.csv", index=False)

    signals = signal_report(data, targets=tuple(args.targets))
    signals.to_csv(args.out / "benchmark_signal_report.csv", index=False)
    print("\nTarget noise ceilings (max attainable R2):")
    for _, row in signals.iterrows():
        print(f"  {row['target']:<20} {row['signal_fraction']:.4f}  "
              f"({row['n_cells']} cells, {row['n_rows']} rows)")

    if args.quick and args.methods:
        raise SystemExit("--quick and --methods are mutually exclusive.")
    if args.methods:
        methods = [get(name) for name in args.methods]
        names = ", ".join(args.methods)
        print(f"Method subset: {len(methods)} of the registry ({names})")
    elif args.quick:
        methods = [get(name) for name in QUICK_METHODS]
    else:
        methods = None

    frames: list[pd.DataFrame] = []
    renders: list[str] = []
    for target in args.targets:
        if target not in data.columns:
            print(f"\nSkipping '{target}': not present in the frame.")
            continue
        print(f"\n--- {target} ---", flush=True)
        result = run_study(
            data, target=target, methods=methods,
            features=args.features or None, verbose=True,
        )
        print(result.render(), flush=True)
        frames.append(result.to_frame())
        renders.append(result.render())

    if not frames:
        raise SystemExit("No target produced a result.")

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(args.out / "benchmark_results.csv", index=False)

    summary = args.out / "benchmark_summary.md"
    summary.write_text(
        "# Benchmark study results\n\n"
        "Generated by `scripts/run_benchmark_study.py`. Every method is\n"
        "evaluated by `src.bms.adaptive.validation.Validator` — the same gate\n"
        "the project applies to its own candidates.\n\n"
        "## Target noise ceilings\n\n```\n"
        + signals.to_string(index=False)
        + "\n```\n\n## Results\n\n```\n"
        + "\n\n".join(renders)
        + "\n```\n",
        encoding="utf-8",
    )

    print(f"\nWrote {args.out / 'benchmark_results.csv'} ({len(results)} rows)")
    print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
