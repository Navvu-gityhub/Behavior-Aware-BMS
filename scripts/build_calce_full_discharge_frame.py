"""Rebuild the CALCE cycle-level frame from detected full discharges.

Addresses `docs/roadmap.md` item 0 and the open item in ADR 0009: Types 5 and 6
cycle partially by design, so grouping on Arbin's `Cycle_Index` scores them
against a partial-cycle reference and their SOH is relative fade rather than
absolute state of health.

This script segments each cell's sample-level telemetry into contiguous
discharge runs, grades each as full or partial, and writes a cycle-level frame
containing only the full ones. See `src/bms/io/calce_full_discharge.py` for why
both a voltage-cutoff and a charge-moved criterion are required, and why a
cheaper cycle-level fix does not work.

    python scripts/build_calce_full_discharge_frame.py
    python scripts/build_calce_full_discharge_frame.py --cells CS2_33 CS2_24 CS2_5
    python scripts/build_calce_full_discharge_frame.py --out reports/metrics/x.csv

Expect a long run: reading the CS2/CX2 archives is I/O bound and takes roughly
one to five minutes per cell.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.bms.io.calce_full_discharge import (  # noqa: E402
    detect_full_discharges,
    full_discharge_frame,
)
from src.bms.io.load_calce_cycling import (  # noqa: E402
    discover_calce_cells,
    load_calce_archive,
    load_calce_cell,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = REPO_ROOT / "data" / "raw" / "calce"
DEFAULT_OUT = REPO_ROOT / "reports" / "metrics" / "calce_full_discharge.csv"
DEFAULT_YIELD = REPO_ROOT / "reports" / "metrics" / "calce_full_discharge_yield.csv"


def _load(source):
    path = Path(source.path)
    if path.suffix.lower() == ".zip":
        return load_calce_archive(path, cell_id=source.cell_id)
    return load_calce_cell(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--yield-out", type=Path, default=DEFAULT_YIELD)
    parser.add_argument(
        "--cells", nargs="*", default=None,
        help="Restrict to these cell ids. Useful for validating before a full run.",
    )
    args = parser.parse_args()

    sources = discover_calce_cells(args.base)
    if args.cells:
        wanted = set(args.cells)
        sources = [s for s in sources if s.cell_id in wanted]
    if not sources:
        print("No matching CALCE cells found.", file=sys.stderr)
        return 1

    print(f"Processing {len(sources)} cell(s) from {args.base}\n")

    frames: list[pd.DataFrame] = []
    yields: list[dict] = []

    for index, source in enumerate(sources, start=1):
        started = time.time()
        label = f"[{index}/{len(sources)}] {source.cell_id} ({source.cohort})"
        try:
            telemetry, _ = _load(source)
        except Exception as exc:
            # A cell that cannot be read is reported and skipped, not fatal:
            # four CALCE cells are CADEX-format and are refused at load by
            # design (ADR 0009), and aborting on the first would lose the rest.
            print(f"{label}: SKIPPED at load - {type(exc).__name__}: {exc}")
            yields.append({
                "cell_id": source.cell_id, "cohort": source.cohort,
                "status": "load_failed", "reason": f"{type(exc).__name__}: {exc}",
                "n_discharges": 0, "n_full": 0,
            })
            continue

        try:
            discharges, summary = detect_full_discharges(telemetry, source.cell_id)
        except ValueError as exc:
            print(f"{label}: REFUSED - {exc}")
            yields.append({
                "cell_id": source.cell_id, "cohort": source.cohort,
                "status": "refused", "reason": str(exc),
                "n_discharges": 0, "n_full": 0,
            })
            continue

        frame = full_discharge_frame(discharges, cohort=source.cohort)
        if not frame.empty:
            frames.append(frame)

        elapsed = time.time() - started
        print(f"{label}: {summary.render()}  [{elapsed:.0f}s]")

        if summary.refusal:
            status = "refused_implausible_capability"
        elif summary.n_full:
            status = "ok"
        else:
            status = "no_full_discharge"

        yields.append({
            "cell_id": source.cell_id,
            "cohort": source.cohort,
            "status": status,
            "reason": summary.refusal,
            "max_discharge_ah": summary.max_discharge_ah,
            "capability_over_max": summary.capability_over_max,
            "n_rows_raw": len(telemetry),
            "n_discharges": summary.n_discharges,
            "n_full": summary.n_full,
            "usable_fraction": summary.usable_fraction,
            "capability_ah": summary.capability_ah,
            "cutoff_v": summary.cutoff_v,
            "rest_threshold_a": summary.rest_threshold_a,
            "elapsed_s": elapsed,
        })

    args.yield_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(yields).to_csv(args.yield_out, index=False)
    print(f"\nWrote yield report -> {args.yield_out}")

    if not frames:
        print("No full discharges detected in any cell.", file=sys.stderr)
        return 1

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"Wrote {len(combined)} full-discharge rows "
          f"({combined['cell_id'].nunique()} cells, "
          f"{combined['cohort'].nunique()} cohorts) -> {args.out}")

    print("\nPer-cell capability (sanity: CS2 is 1.1 Ah, CX2 is 1.35 Ah nominal):")
    check = (
        pd.DataFrame(yields)
        .query("n_full > 0")[["cell_id", "cohort", "n_full", "capability_ah"]]
        .sort_values(["cohort", "cell_id"])
    )
    print(check.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
