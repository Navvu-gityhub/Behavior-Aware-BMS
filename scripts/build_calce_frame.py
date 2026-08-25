"""Build the CALCE cycle-level frame once, so every later run is cheap.

    python scripts/build_calce_frame.py
    python scripts/build_calce_frame.py --base-dir data/raw/calce --limit 3

Writes `reports/metrics/calce_cycle_level.csv` — one row per cell-cycle, the
same shape as the NASA frame the rest of the project already runs on, so the
benchmark suite and the adaptive gate take it without modification.

WHY A CACHE STEP EXISTS
-----------------------
Reading the distributed archives is slow and the cost is all in openpyxl: one
45 MB CS2 archive of 23 workbooks takes about 145 seconds, and the largest
cell ships 235 MB across 39 workbooks. Loading all fifteen on every experiment
would put an hour of Excel parsing in front of a study that then runs in
minutes.

The output is roughly 13,000 rows and a few hundred kilobytes, which is the
same trade the NASA path already makes: `continuous_model_training_data.csv`
is a tracked cycle-level summary of a 7.2M-row source that is not tracked.

WHAT IS AND IS NOT LOADED
-------------------------
CS2 ships two tester formats. Thirteen cells are Arbin workbooks and load.
CS2_8 and CS2_21 are CADEX `.txt` exports with a different schema entirely —
tab-separated, millivolts and milliamps rather than volts and amps, a
`Pgm cycle` column instead of `Cycle_Index`. They are **skipped with a reason
recorded**, not silently dropped.

That costs no cohort: both are Type 1, which retains CS2_33 and CS2_34.
Supporting CADEX means writing and validating a second column map and a unit
conversion, and doing that on the basis of a schema nobody has checked against
the cycler's documentation is how a factor-of-1000 error enters a fade target.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.io.load_calce_cycling import (  # noqa: E402
    calce_capacity_loss,
    discover_calce_cells,
    load_calce_archive,
    load_calce_cell,
    summarize_calce_cycles,
)

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_BASE = REPO_ROOT / "data" / "raw" / "calce"
DEFAULT_OUT = REPO_ROOT / "reports" / "metrics" / "calce_cycle_level.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="Load at most N cells (for a quick check).")
    parser.add_argument(
        "--rebuild", action="store_true",
        help=(
            "Re-parse every cell even if it is already in the output. Without "
            "this the script resumes: cells already present are skipped, which "
            "matters because parsing one archive costs minutes and the full "
            "CS2+CX2 set is 2.5 GB."
        ),
    )
    args = parser.parse_args()

    if not args.base_dir.exists():
        raise SystemExit(
            f"No CALCE data at {args.base_dir}. Download from "
            f"https://calce.umd.edu/battery-data (CS2 / CX2 sections)."
        )

    sources = discover_calce_cells(args.base_dir)
    if not sources:
        raise SystemExit(f"No cells discovered under {args.base_dir}.")

    if args.limit:
        sources = sources[: args.limit]

    print(f"Discovered {len(sources)} cell(s) under {args.base_dir}")
    for s in sources:
        print(f"  {s.cell_id:<10} cohort={s.cohort or '(none)':<12} {s.path.name}")
    print()

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[str, str]] = []

    # Resume from a previous run. Parsing one archive costs minutes, so a
    # re-run after adding CX2 should not re-parse the thirteen CS2 cells.
    #
    # Cohort labels are NOT carried over: they are cheap to recompute from the
    # directory layout, and a cached label would silently survive a change to
    # how cohorts are derived (which is exactly what happened when CS2 and CX2
    # were namespaced). Only the expensive per-cycle rows are reused.
    already: set[str] = set()
    if args.out.exists() and not args.rebuild:
        cached = pd.read_csv(args.out)
        cohort_by_cell = {s.cell_id: s.cohort for s in sources}
        keep = cached[cached["cell_id"].isin(cohort_by_cell)].copy()
        if not keep.empty:
            keep["cohort"] = keep["cell_id"].map(cohort_by_cell)
            frames.append(keep)
            already = set(keep["cell_id"].unique())
            print(f"Resuming: {len(already)} cell(s) already in {args.out.name} "
                  f"({', '.join(sorted(already))})")
            print("Pass --rebuild to re-parse them.\n")

    for i, source in enumerate(sources, 1):
        if source.cell_id in already:
            continue
        started = time.time()
        print(f"[{i}/{len(sources)}] {source.cell_id} ...", end="", flush=True)
        try:
            if source.is_archive:
                telemetry, report = load_calce_archive(
                    source.path, cell_id=source.cell_id
                )
            else:
                telemetry, report = load_calce_cell(
                    source.path, cell_id=source.cell_id
                )
            if source.cohort:
                telemetry["cohort"] = source.cohort

            summary = calce_capacity_loss(summarize_calce_cycles(telemetry))
            frames.append(summary)

            print(
                f" {len(summary):>5} cycles  "
                f"cap {summary['capacity_ah'].iloc[0]:.3f} -> "
                f"{summary['capacity_ah'].iloc[-1]:.3f} Ah  "
                f"SOH {summary['soh'].iloc[0]:.0f}% -> {summary['soh'].iloc[-1]:.0f}%  "
                f"({time.time() - started:.0f}s, {report.n_files} files)",
                flush=True,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            skipped.append((source.cell_id, reason))
            print(f" SKIPPED — {reason[:120]}", flush=True)

    if not frames:
        raise SystemExit("No cell loaded successfully.")

    combined = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)

    print()
    print(f"Loaded {combined['cell_id'].nunique()} cells, {len(combined)} cycle rows")
    if "cohort" in combined.columns:
        print("cells per cohort:")
        for cohort, n in combined.groupby("cohort")["cell_id"].nunique().items():
            print(f"  {cohort:<10} {n}")
    for cell, reason in skipped:
        print(f"SKIPPED {cell}: {reason[:160]}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
