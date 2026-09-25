"""Build curve-derived features for every CALCE cell and write the artifact.

    python scripts/build_calce_curve_features.py

Produces, under `reports/metrics/`:

    calce_curve_features.csv    one row per cell: Delta-Q variance, ICA peaks
    calce_curve_yield.csv       per-cell extraction yield and refusal counts
    calce_curve_features.md     the human-readable report

WHY THIS SCRIPT EXISTS SEPARATELY FROM THE BENCHMARK STUDY
-----------------------------------------------------------
Extraction reads every Arbin workbook in the archive -- on the full CS2 + CX2
set that is hundreds of thousands of sample rows per cell. Folding it into
`run_benchmark_study.py` would make a routine benchmark re-run take a quarter
of an hour, which is the kind of friction that stops a study being re-run at
all. The features are stable given the raw archives, so they are built once,
written as an artifact, and joined by cell id afterwards.

READING IS TRUNCATED BY DEFAULT IN PRACTICE
-------------------------------------------
Both features are early-life quantities: Delta-Q is defined on cycles 10 and
100, and the ICA peaks are read off cycle 100. Reading a cell's remaining 700
cycles costs minutes per cell for data neither feature touches, so `--max-cycles`
stops after enough files to cover the window.

That makes the frame a truncated one, and `truncated_at_cycle` is carried
through to the yield artifact for every cell. **No cycle-life, end-of-life,
total-throughput or final-state-of-health quantity may be derived from a
truncated run** -- the last cycle present is where reading stopped, not where
the cell died. Pass no `--max-cycles` to read the full archive.

WHAT THIS DOES NOT DO
---------------------
It does not score anything, and it does not register a benchmark method. It
produces the feature columns whose absence is the stated reason
`severson_delta_q_variance` and `ica_peak_features` appear as UNAVAILABLE in
the benchmark table. Turning those rows into scored rows is a separate change
to the registry, and it must go through the same validation gate as every
other method rather than arriving alongside its own data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bms.io.load_calce_curves import (
    DEFAULT_CYCLE_A,
    DEFAULT_CYCLE_B,
    curve_features_by_cell,
    extract_discharge_curves,
)
from src.bms.io.load_calce_cycling import (
    discover_calce_cells,
    load_calce_archive,
    load_calce_cell,
)

DEFAULT_RAW = Path("data/raw/calce")
DEFAULT_OUT = Path("reports/metrics")


def _append_row(path: Path, row: dict) -> None:
    """Append one record, writing the header only on the first row.

    Refusal reasons vary by cell, so the columns are not known up front. A
    plain append would misalign once a later cell carries a reason an earlier
    one did not, so the existing file is re-read and re-written on a widening.
    """
    frame = pd.DataFrame([row])
    if path.exists():
        previous = pd.read_csv(path)
        if set(frame.columns) - set(previous.columns):
            frame = pd.concat([previous, frame], ignore_index=True)
            frame.to_csv(path, index=False)
            return
    frame.to_csv(path, mode="a", index=False, header=not path.exists())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cycle-a", type=int, default=DEFAULT_CYCLE_A)
    parser.add_argument("--cycle-b", type=int, default=DEFAULT_CYCLE_B)
    parser.add_argument(
        "--restart", action="store_true",
        help="ignore any partial artifact and rebuild every cell",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=None,
        help=(
            "stop reading each cell after this many cycles. The default reads "
            "the whole archive (2.5 GB, hours). Passing cycle-b plus a margin "
            "reads only what the early-life features use -- see the note in "
            "the module docstring about what a truncated frame may not be "
            "used for."
        ),
    )
    args = parser.parse_args()

    sources = discover_calce_cells(args.raw_dir)
    if not sources:
        print(f"no CALCE cells found under {args.raw_dir}")
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.out_dir / "calce_curve_features.partial.csv"
    partial_yield_path = args.out_dir / "calce_curve_yield.partial.csv"

    # Per-cell results are appended as they are computed, and a re-run skips
    # the cells already present. Reading the full archive takes tens of
    # minutes; an interruption that discarded all of it once is enough.
    done: set[str] = set()
    if partial_path.exists() and not args.restart:
        existing = pd.read_csv(partial_path)
        done = {str(value) for value in existing["cell_id"].tolist()}
        print(f"resuming: {len(done)} cells already in {partial_path.name}")

    print(f"{len(sources)} cells discovered under {args.raw_dir}", flush=True)
    feature_rows: list[pd.DataFrame] = []
    yield_rows: list[dict] = []

    # Cell at a time rather than one concatenated frame. The full archive does
    # not need to be resident at once, and a cell that fails to load costs its
    # own row rather than the whole run.
    for index, source in enumerate(sources, start=1):
        label = f"[{index}/{len(sources)}] {source.cell_id}"
        if source.cell_id in done:
            print(f"{label}: already built, skipping", flush=True)
            continue
        print(f"{label}: reading...", flush=True)
        try:
            if source.is_archive:
                telemetry, load_report = load_calce_archive(
                    source.path, cell_id=source.cell_id,
                    max_cycles=args.max_cycles,
                )
            else:
                telemetry, load_report = load_calce_cell(
                    source.path, cell_id=source.cell_id,
                    max_cycles=args.max_cycles,
                )
        except Exception as exc:
            print(f"{label}: LOAD FAILED - {type(exc).__name__}: {exc}",
                  flush=True)
            yield_rows.append({
                "cell_id": source.cell_id,
                "cohort": source.cohort,
                "status": f"load failed: {type(exc).__name__}",
                "cycles_seen": 0,
                "cycles_kept": 0,
            })
            _append_row(partial_yield_path, yield_rows[-1])
            continue

        if source.cohort:
            telemetry["cohort"] = source.cohort

        curves, report = extract_discharge_curves(telemetry)
        print(f"{label}: {report.n_cycles_kept}/{report.n_cycles_seen} cycles "
              f"({report.yield_fraction:.1%}), {report.n_points} points",
              flush=True)

        row = {
            "cell_id": source.cell_id,
            "cohort": source.cohort,
            "status": "ok" if report.n_cycles_kept else "no curves",
            "cycles_seen": report.n_cycles_seen,
            "cycles_kept": report.n_cycles_kept,
            "yield_fraction": round(report.yield_fraction, 4),
            "curve_points": report.n_points,
            "files_read": load_report.n_files,
            "truncated_at_cycle": load_report.truncated_at_cycle,
        }
        for reason, count in report.refusals.items():
            row[f"refused: {reason}"] = count
        yield_rows.append(row)
        _append_row(partial_yield_path, row)

        if curves.empty:
            continue
        cell_features = curve_features_by_cell(curves, args.cycle_a, args.cycle_b)
        feature_rows.append(cell_features)
        # Append before moving on, so an interrupted run keeps what it earned.
        cell_features.to_csv(
            partial_path, mode="a", index=False,
            header=not partial_path.exists(),
        )

    # Read both artifacts back from the partials, so a resumed run reports the
    # whole archive rather than only the cells this invocation happened to
    # touch. A yield table covering 4 of 19 cells would understate the
    # extraction loss without saying it was partial.
    if partial_path.exists():
        feature_rows = [pd.read_csv(partial_path)]
    yields = (
        pd.read_csv(partial_yield_path)
        if partial_yield_path.exists()
        else pd.DataFrame(yield_rows)
    )
    # A cell that failed to load leaves no feature row, so a resumed run
    # retries it and appends a second yield row. Keep the latest per cell.
    if not yields.empty:
        yields = yields.drop_duplicates(subset="cell_id", keep="last")
    yields.to_csv(args.out_dir / "calce_curve_yield.csv", index=False)

    if not feature_rows:
        print("no cell produced a usable curve; no feature artifact written")
        return 1

    features = pd.concat(feature_rows, ignore_index=True)
    features.to_csv(args.out_dir / "calce_curve_features.csv", index=False)

    # `fillna("")` matters: the partial artifact is read back from CSV, where
    # an empty refusal_reason returns as NaN. Comparing to "" then matches
    # nothing and the report claims every cell was refused, which is how this
    # first ran -- "0/23 usable" against 18 rows of finite features.
    reasons = features["refusal_reason"].fillna("").astype(str).str.strip()
    usable = features[reasons == ""]
    lines = [
        "# CALCE curve-derived features",
        "",
        f"Generated by `scripts/build_calce_curve_features.py` from "
        f"`{args.raw_dir}`.",
        "",
        f"Delta-Q window: cycles {args.cycle_a} and {args.cycle_b}. ICA "
        f"features are read off cycle {args.cycle_b}.",
        "",
        f"- cells discovered: **{len(sources)}**",
        f"- cells yielding curves: **{len(features)}**",
        f"- cells with both window cycles, feature computed: **{len(usable)}**",
        f"- cycles kept: **{int(yields['cycles_kept'].sum())}** of "
        f"**{int(yields['cycles_seen'].sum())}**",
        "",
        "## Features",
        "",
        "```",
        features.to_string(index=False),
        "```",
        "",
        "## Extraction yield",
        "",
        "```",
        yields.to_string(index=False),
        "```",
        "",
        "## What these columns are for",
        "",
        "`delta_q_variance` is log10(var(Q_b(V) - Q_a(V))), the single feature",
        "of Severson et al. (Nature Energy 4, 383-391, 2019). `ica_peak_*` are",
        "read off the dQ/dV curve and track loss of active material and loss of",
        "lithium inventory separately (Dubarry et al., J. Power Sources 219,",
        "2012, 204-216).",
        "",
        "**These citations describe where the feature definitions come from.**",
        "No result from either paper is reproduced here, and no figure from",
        "either has been verified against the original by this project.",
        "",
        "Neither column has been scored yet. They exist so that",
        "`severson_delta_q_variance` and `ica_peak_features`, currently",
        "UNAVAILABLE in the benchmark table for want of curve data, can be run",
        "through the same leave-one-cohort-out gate as every other method.",
    ]
    (args.out_dir / "calce_curve_features.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"\nwrote {args.out_dir / 'calce_curve_features.csv'}")
    print(f"wrote {args.out_dir / 'calce_curve_yield.csv'}")
    print(f"wrote {args.out_dir / 'calce_curve_features.md'}")
    print(f"{len(usable)}/{len(features)} cells produced a usable feature row")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
