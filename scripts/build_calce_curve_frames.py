"""Extract full-life discharge Q(V) curves per cell and cache them.

    python scripts/build_calce_curve_frames.py

Writes one Parquet per cell under `data/interim/calce_curves/`, resumable.

WHY FULL LIFE, WHEN THE FEATURE STUDY TRUNCATED AT 130 CYCLES
--------------------------------------------------------------
`build_calce_curve_features.py` reads only far enough to reach cycle 100,
because Severson's Delta-Q is an early-life feature and the rest of the
archive is cost with no benefit for it.

Voltage-window capacity tracking is the opposite case. It asks how a cell's
charge-in-a-voltage-window falls as the cell ages, so truncating at 130 cycles
would cut the measurement off before the degradation it is meant to track has
happened. CS2 cells run to 800+ cycles and lose most of their capacity late.

This is the slow read - the whole 2.5 GB archive, roughly five hours - so the
curves are cached per cell and the run resumes. Nothing downstream re-reads
the raw workbooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.bms.io.load_calce_curves import extract_discharge_curves
from src.bms.io.load_calce_cycling import (
    discover_calce_cells,
    load_calce_archive,
    load_calce_cell,
)

DEFAULT_RAW = Path("data/raw/calce")
DEFAULT_OUT = Path("data/interim/calce_curves")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-cycles", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sources = discover_calce_cells(args.raw_dir)
    print(f"{len(sources)} cells under {args.raw_dir}", flush=True)

    for index, source in enumerate(sources, start=1):
        out = args.out_dir / f"{source.cell_id}.parquet"
        label = f"[{index}/{len(sources)}] {source.cell_id}"
        if out.exists():
            print(f"{label}: cached, skipping", flush=True)
            continue

        print(f"{label}: reading...", flush=True)
        try:
            if source.is_archive:
                telemetry, report = load_calce_archive(
                    source.path, cell_id=source.cell_id,
                    max_cycles=args.max_cycles,
                )
            else:
                telemetry, report = load_calce_cell(
                    source.path, cell_id=source.cell_id,
                    max_cycles=args.max_cycles,
                )
        except Exception as exc:
            print(f"{label}: LOAD FAILED - {type(exc).__name__}", flush=True)
            continue

        if source.cohort:
            telemetry["cohort"] = source.cohort

        curves, creport = extract_discharge_curves(telemetry)
        if curves.empty:
            print(f"{label}: no curves", flush=True)
            continue

        curves["truncated_at_cycle"] = report.truncated_at_cycle
        curves.to_parquet(out, index=False)
        print(f"{label}: {creport.n_cycles_kept}/{creport.n_cycles_seen} "
              f"cycles, {len(curves)} points -> {out.name}", flush=True)

    cached = sorted(args.out_dir.glob("*.parquet"))
    print(f"\n{len(cached)} cells cached in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
