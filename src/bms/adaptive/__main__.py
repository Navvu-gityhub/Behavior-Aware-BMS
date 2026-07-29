"""Command-line interface for the adaptive calibration system.

    python -m src.bms.adaptive screen
    python -m src.bms.adaptive calibrate --dataset nasa
    python -m src.bms.adaptive status
    python -m src.bms.adaptive rollback --cohort GLOBAL --reason "..."

The CLI is deliberately thin. It registers the datasets this project actually
has, wires them to the calibrator, and prints. Every decision it surfaces was
made by the components underneath it, and the exit codes reflect those
decisions rather than reinterpreting them: a run in which nothing is promoted
exits 0, because refusing to promote a model that fails validation is the
system working, not the system erroring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.bms.adaptive.calibrator import AdaptiveCalibrator, linear_candidate
from src.bms.adaptive.datasets import CallableDatasetLoader, DatasetRegistry
from src.bms.adaptive.store import ModelStore

DEFAULT_STORE = Path("models/adaptive")
NASA_TRAINING = Path("reports/metrics/continuous_model_training_data.csv")
CALCE_SAMPLE = Path("data/processed/calce/calce_sample_processed.csv")


def build_registry() -> DatasetRegistry:
    """Register the datasets this repository actually contains.

    CALCE is registered even though it is known to be unusable. Leaving it out
    would hide the finding; registering it means `screen` reports *why* it
    cannot be used, which is more informative than its absence.
    """
    registry = DatasetRegistry()

    if NASA_TRAINING.exists():
        registry.register(CallableDatasetLoader(
            "nasa", lambda: pd.read_csv(NASA_TRAINING)
        ))

    if CALCE_SAMPLE.exists():
        def load_calce() -> pd.DataFrame:
            data = pd.read_csv(CALCE_SAMPLE)
            if "cell_id" not in data.columns:
                data["cell_id"] = "CALCE_SAMPLE"
            if "cohort" not in data.columns:
                data["cohort"] = "CALCE_PLN"
            return data

        registry.register(CallableDatasetLoader("calce_sample", load_calce))

    return registry


def default_candidates() -> list:
    """The specifications worth testing against this project's data.

    Note the absence of a per-cohort-intercept candidate. `linear_candidate`
    cannot express one, and that is intentional: ADR 0002 found the intercepts
    were where the v2 model's apparent skill lived. A candidate wanting them
    must be written explicitly, so that choice is visible in a diff.
    """
    return [
        linear_candidate("temp_only", ["trailing_avg_temp"]),
        linear_candidate("temp_and_stress", ["trailing_avg_temp", "avg_stress"]),
        # avg_soc is deliberately absent. Its within-cell Spearman correlation
        # with cycle index is -0.73 (27 of 32 cells above 0.5 in magnitude), so
        # it is largely a proxy for how far through its life a cell is.
        # Including it produced a candidate that passed the gate on age alone,
        # which is what motivated the confound baseline now applied to every
        # candidate. Kept out of the defaults so the failure is not re-run by
        # accident; a caller wanting it must add it explicitly.
    ]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_screen(args: argparse.Namespace) -> int:
    registry = build_registry()
    if not len(registry):
        print("No datasets found. Expected at least one of:")
        print(f"  {NASA_TRAINING}")
        print(f"  {CALCE_SAMPLE}")
        return 1

    table = registry.assess_all()
    print(table[[
        "dataset", "status", "n_cells", "n_cohorts",
        "median_cycles_per_cell", "n_blockers", "n_warnings",
    ]].to_string(index=False))

    for name in registry.names:
        report = registry.assess(name)
        if report.blockers or (args.verbose and report.warnings):
            print()
            print(report.render())
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    registry = build_registry()
    if args.dataset not in registry:
        print(f"Unknown dataset '{args.dataset}'. Available: {registry.names}")
        return 1

    calibrator = AdaptiveCalibrator(
        store=ModelStore(args.store), datasets=registry
    )
    run = calibrator.calibrate(args.dataset, default_candidates())
    print(run.render())

    if run.aborted:
        # A dataset that cannot support calibration is a real failure of the
        # request, so this one does exit non-zero.
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = ModelStore(args.store)
    summary = store.summary()

    if not summary:
        print(f"No cohorts in the store at {args.store}.")
        print("Nothing has been promoted. Run 'calibrate' first.")
    else:
        print(pd.DataFrame(summary).to_string(index=False))

    decisions = list(store.decisions())
    if decisions:
        print(f"\nDecision log ({len(decisions)} entries, most recent last):")
        for decision in decisions[-args.tail:]:
            version = f"v{decision.version}" if decision.version else "-"
            loco = decision.metrics.get("loco_median_r2")
            loco_text = f"  loco_r2={loco:+.4f}" if loco is not None else ""
            print(f"  {decision.outcome:8} {decision.cohort_id:20} {version:5}{loco_text}")
            if args.verbose:
                for reason in decision.reasons:
                    print(f"      - {reason}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    store = ModelStore(args.store)
    try:
        reverted = store.rollback(args.cohort, reason=args.reason)
    except (KeyError, ValueError) as exc:
        print(f"Rollback refused: {exc}")
        return 1
    print(f"{args.cohort} reverted to v{reverted.version}")
    print(f"  params: {dict(reverted.params)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.bms.adaptive",
        description="Governed calibration: screen datasets, validate candidates, "
                    "promote only what generalises.",
    )
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE,
                        help=f"model store directory (default: {DEFAULT_STORE})")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="assess every registered dataset")
    screen.set_defaults(func=cmd_screen)

    calibrate = sub.add_parser("calibrate", help="validate candidates and offer them")
    calibrate.add_argument("--dataset", default="nasa")
    calibrate.set_defaults(func=cmd_calibrate)

    status = sub.add_parser("status", help="show the store and decision log")
    status.add_argument("--tail", type=int, default=20)
    status.set_defaults(func=cmd_status)

    rollback = sub.add_parser("rollback", help="revert a cohort to its parent version")
    rollback.add_argument("--cohort", required=True)
    rollback.add_argument("--reason", default="")
    rollback.set_defaults(func=cmd_rollback)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
