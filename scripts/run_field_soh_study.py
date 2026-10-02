"""Can a BMS measure SOH from its own telemetry? Validated on every CALCE cell.

    python scripts/run_field_soh_study.py            # resumes; one cell at a time
    python scripts/run_field_soh_study.py --restart

Writes `reports/metrics/calce_field_soh/`.

WHAT IS DIFFERENT FROM run_voltage_window_study.py
--------------------------------------------------
That study scored the window estimator on curves the CALCE loader extracted,
compensated with the resistance the Arbin cycler measured. Neither exists on
a vehicle. Here the estimator gets only what a BMS has - voltage, current,
time - and has to segment the discharges, build the curves and estimate its
own resistance (`health/field_soh.py`). Ground truth is the same file the
earlier study used, joined on Arbin's cycle index, and nothing about it
enters the estimate.

ARMS, FIXED BEFORE ANY CELL WAS SCORED
--------------------------------------
    raw          no ohmic correction
    cycler_r     corrected with the cycler's own resistance column (the
                 earlier study's configuration, as a reference point)
    step_r       corrected with resistance from the rest-to-load step - the
                 field method
    step_r_partial  step_r on every discharge truncated to the middle 70% of
                 its charge (85% -> 15% state of charge), re-zeroed: what a
                 driver who never fully charges or fully empties produces

GROUND TRUTH, PRIMARY: SOH as cycler capacity over the median of the cell's
first five cycles - the same reference the estimator uses, and the one a BMS
can hold. `mae_lifetime` scores against add_targets' lifetime-percentile
reference instead, for comparison with run_voltage_window_study.py; that
reference uses the cell's future and differs from the early one by a few
percent, which is an offset in the truth, not an error in the estimate.

FIELD GATES (health/field_soh.apply_field_gates) are applied to every arm,
because they are what ships: the reference must be formed within 20
equivalent full cycles of charge, and a reading above 1.10 is withheld. They
were added after a first pass found the partial arm forming its reference at
discharge 805. A second pass counted discharges rather than throughput and
wrongly refused CS2_24, whose 5,039 phases are mostly characterisation pulses;
throughput is the measure of age a BMS reports, and it is what is used.

PRIMARY WINDOW: 3.90-3.60 V, the module default, fixed before this ran.
4.05-3.55 and 3.85-3.55 are reported as sensitivity, not chosen between. The
earlier study showed choosing a window on the truth it is then scored against
is the optimism this project exists to avoid.

PER-CELL, THEN MEDIAN ACROSS CELLS. A pooled error is dominated by whichever
cells have the most cycles.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.validation import bootstrap_median_ci  # noqa: E402
from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import (  # noqa: E402
    apply_field_gates,
    curves_from_telemetry,
    monotonic_time,
    step_overpotential,
)
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402
from src.bms.io.load_calce_cycling import (  # noqa: E402
    discover_calce_cells,
    load_calce_archive,
)
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

DEFAULT_RAW = Path("data/raw/calce")
DEFAULT_TRUTH = Path("reports/metrics/calce_full_discharge.csv")
DEFAULT_OUT = Path("reports/metrics/calce_field_soh")
# Per-cell telemetry, cached after the first (slow) read of each archive.
# Gitignored: derived from data that is not redistributable.
CACHE = Path("data/interim/calce_telemetry")
CACHE_COLUMNS = ("cell_id", "test_time_s", "current_a", "voltage_v", "cycle",
                 "resistance_ohm")

# Arbin logs exactly zero current at rest. The vehicle default (0.5 A) would
# classify CALCE's 0.2C discharges, at 0.22 A, as rest.
REST_THRESHOLD_A = 0.02

PRIMARY = WindowSpec(3.90, 3.60)
SENSITIVITY = (WindowSpec(4.05, 3.55), WindowSpec(3.85, 3.55))
# EXPLORATORY, added AFTER the pre-declared partial arm was refused on every
# cell at the primary window. At ~1C a discharge cut to 85%->15% state of
# charge spans roughly 3.85-3.52 V, and the compensated primary window needs
# terminal voltage down to ~3.44 V, so it can never be read. A window has to
# sit inside the range the vehicle actually uses. 4.00-3.80 V, shifted down by
# the ~0.18 V sag, lands inside it. Reported separately and labelled; it is a
# hypothesis for the next dataset, not a validated configuration.
EXPLORATORY = (WindowSpec(4.00, 3.80),)
PARTIAL_FROM, PARTIAL_TO = 0.15, 0.85


def _arbin_cycle_per_discharge(telemetry, discharges) -> pd.Series:
    t = telemetry["test_time_s"].to_numpy(float)
    cyc = pd.to_numeric(telemetry["cycle"], errors="coerce").to_numpy(float)
    out = []
    for row in discharges.itertuples():
        a = np.searchsorted(t, row.start_time_s, "left")
        b = np.searchsorted(t, row.end_time_s, "right")
        seg = cyc[a:b]
        seg = seg[np.isfinite(seg)]
        out.append(float(pd.Series(seg).mode().iloc[0]) if len(seg) else np.nan)
    return pd.Series(out, index=discharges.index)


def _cycler_overpotential(telemetry, discharges, cell_id) -> pd.DataFrame | None:
    if "resistance_ohm" not in telemetry.columns:
        return None
    t = telemetry["test_time_s"].to_numpy(float)
    r = pd.to_numeric(telemetry["resistance_ohm"], errors="coerce").to_numpy(float)
    rows = []
    for row in discharges.itertuples():
        a = np.searchsorted(t, row.start_time_s, "left")
        b = np.searchsorted(t, row.end_time_s, "right")
        seg = r[a:b]
        seg = seg[np.isfinite(seg) & (seg > 0)]
        if len(seg):
            rows.append({"cell_id": cell_id, "cycle": int(row.cycle),
                         "ir_drop_v": abs(row.mean_current_a) * float(np.median(seg))})
    return pd.DataFrame(rows) if rows else None


def _truncate(curves: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, block in curves.groupby("cycle", sort=False):
        q = block["capacity_ah_curve"].to_numpy(float)
        total = q.max()
        keep = (q >= PARTIAL_FROM * total) & (q <= PARTIAL_TO * total)
        part = block[keep].copy()
        if len(part) >= 2:
            part["capacity_ah_curve"] = part["capacity_ah_curve"] - part["capacity_ah_curve"].iloc[0]
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else curves.iloc[0:0]


def _score(table: pd.DataFrame, truth_cell: pd.DataFrame, discharges: pd.DataFrame) -> dict:
    if table.empty:
        return {"n_scored": 0, "mae": np.nan, "refused": True, "refusal": "no curves"}
    refused = bool(table["cell_refused"].any())
    est = table[["cycle", "soh_window_accepted"]].merge(
        discharges[["cycle", "arbin_cycle"]], on="cycle", how="left")
    joined = est.merge(truth_cell, on="arbin_cycle", how="inner").dropna(
        subset=["soh_window_accepted", "soh"])
    err = (joined["soh_window_accepted"] - joined["soh"]).abs()
    err_life = (joined["soh_window_accepted"] - joined["soh_lifetime"]).abs()
    return {
        "mae_lifetime": float(err_life.mean()) if len(err_life) else np.nan,
        "reference_efc": float(table["reference_efc"].iloc[0])
        if "reference_efc" in table.columns else np.nan,
        "n_scored": int(len(joined)),
        "n_discharges": int(len(discharges)),
        # Share of ALL segmented discharges the window could read. An arm that
        # scores well on 3% of discharges has not shown it works in the field.
        "coverage": float(table["window_charge_ah"].notna().sum() / max(len(discharges), 1)),
        "mae": float(err.mean()) if len(err) else np.nan,
        "p90_abs_error": float(err.quantile(0.9)) if len(err) else np.nan,
        "refused": refused,
        "refusal": str(table.loc[table["cell_refused"], "refusal"].iloc[0]) if refused else "",
    }


def run_cell(source, truth: pd.DataFrame) -> list[dict]:
    started = time.time()
    cell = source.cell_id
    cached = CACHE / f"{cell}.parquet"
    if cached.exists():
        telemetry = pd.read_parquet(cached)
        n_files = -1
    else:
        telemetry, report = load_calce_archive(source.path, cell_id=cell)
        telemetry["test_time_s"] = monotonic_time(telemetry["test_time_s"])
        telemetry = telemetry[[c for c in CACHE_COLUMNS if c in telemetry.columns]]
        CACHE.mkdir(parents=True, exist_ok=True)
        telemetry.to_parquet(cached, index=False)
        n_files = report.n_files

    measurements = measure_cycles(telemetry, cell_id=cell, rest_threshold_a=REST_THRESHOLD_A)
    discharges = cycles_to_frame(measurements, complete_only=False)
    if discharges.empty:
        return [{"cell_id": cell, "arm": "all", "window": "", "n_scored": 0,
                 "refused": True, "refusal": "no discharges segmented"}]
    discharges["arbin_cycle"] = _arbin_cycle_per_discharge(telemetry, discharges)

    truth_cell = truth[truth["cell_id"] == cell][["arbin_cycle", "soh", "soh_lifetime"]]
    curves, steps = curves_from_telemetry(telemetry, discharges, cell, REST_THRESHOLD_A)
    step_op = step_overpotential(steps).dropna(subset=["ir_drop_v"]) if not steps.empty else None
    cycler_op = _cycler_overpotential(telemetry, discharges, cell)

    rows = []
    for spec in (PRIMARY, *SENSITIVITY, *EXPLORATORY):
        arms = {"raw": (curves, None)}
        if cycler_op is not None:
            arms["cycler_r"] = (curves[curves["cycle"].isin(set(cycler_op["cycle"]))], cycler_op)
        if step_op is not None and len(step_op):
            stepped = curves[curves["cycle"].isin(set(step_op["cycle"]))]
            arms["step_r"] = (stepped, step_op)
            arms["step_r_partial"] = (_truncate(stepped), step_op)
        for arm, (frame, op) in arms.items():
            table = window_soh_table(frame, spec=spec, overpotential=op) if len(frame) else pd.DataFrame()
            table = apply_field_gates(table, discharges)
            rows.append({"cell_id": cell, "cohort": source.cohort, "arm": arm,
                         "window": str(spec), "primary": spec == PRIMARY,
                         **_score(table, truth_cell, discharges)})
    median_r = float(steps["r_step_ohm"].median()) if not steps.empty else np.nan
    for r in rows:
        r["median_step_r_ohm"] = median_r
        r["step_found_fraction"] = float(steps["r_step_ohm"].notna().mean()) if not steps.empty else 0.0
        r["load_s"] = round(time.time() - started, 1)
        r["n_files"] = n_files
    return rows


def summarise(per_cell: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (window, arm), block in per_cell.groupby(["window", "arm"]):
        scored = block[(~block["refused"].astype(bool)) & (block["n_scored"] >= 10)]
        maes = scored["mae"].to_numpy(float)
        ci = bootstrap_median_ci(maes) if len(maes) else (np.nan, np.nan)
        out.append({
            "window": window, "arm": arm,
            "cells_scored": int(len(scored)), "cells_total": int(block["cell_id"].nunique()),
            "cells_refused": int(block["refused"].astype(bool).sum()),
            "median_coverage": float(block["coverage"].median()) if "coverage" in block else np.nan,
            "median_cell_mae_lifetime_ref": float(np.median(scored["mae_lifetime"])) if len(maes) else np.nan,
            "median_cell_mae": float(np.median(maes)) if len(maes) else np.nan,
            "ci_low": ci[0], "ci_high": ci[1],
            "worst_cell_mae": float(np.max(maes)) if len(maes) else np.nan,
            "worst_cell": scored.loc[scored["mae"].idxmax(), "cell_id"] if len(maes) else "",
        })
    return pd.DataFrame(out).sort_values(["window", "arm"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--cells", nargs="*", default=None)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    per_cell_path = args.out / "field_soh_per_cell.csv"
    done = pd.DataFrame()
    if per_cell_path.exists() and not args.restart:
        done = pd.read_csv(per_cell_path)

    raw_truth = pd.read_csv(args.truth)
    truth = add_targets(raw_truth)
    truth = truth[truth["soh"].notna()].sort_values(["cell_id", "cycle"]).copy()
    truth["soh_lifetime"] = truth["soh"]
    early = truth.groupby("cell_id")["capacity_ah"].transform(lambda c: c.head(5).median())
    truth["soh"] = truth["capacity_ah"] / early
    truth = truth[["cell_id", "arbin_cycle_index", "soh", "soh_lifetime"]]
    truth = truth.rename(columns={"arbin_cycle_index": "arbin_cycle"})

    sources = [s for s in discover_calce_cells(args.raw)
               if s.is_archive and s.cell_id in set(truth["cell_id"])]
    if args.cells:
        sources = [s for s in sources if s.cell_id in set(args.cells)]
    finished = set(done["cell_id"]) if not done.empty else set()
    print(f"{len(sources)} cells with truth; {len(finished)} already done", flush=True)

    for source in sources:
        if source.cell_id in finished:
            continue
        try:
            rows = run_cell(source, truth)
        except Exception as exc:  # recorded, not swallowed: a failed cell is a result
            rows = [{"cell_id": source.cell_id, "cohort": source.cohort, "arm": "all",
                     "window": "", "n_scored": 0, "refused": True,
                     "refusal": f"{type(exc).__name__}: {exc}"}]
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(per_cell_path, index=False)
        prim = [r for r in rows if r.get("primary")]
        print(f"{source.cell_id}: " + ", ".join(
            f"{r['arm']}={r['mae']:.4f}" if pd.notna(r.get("mae")) else f"{r['arm']}=--"
            for r in prim), flush=True)

    summary = summarise(done[done["arm"] != "all"])
    summary.to_csv(args.out / "field_soh_summary.csv", index=False)
    write_report(done, summary, args.out)
    print(summary.to_string(index=False))
    return 0


def write_report(per_cell: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    """The artifact a reader opens. Every number in it is read from the CSVs."""
    primary = summary[summary["window"] == str(PRIMARY)]
    cells = per_cell[(per_cell["window"] == str(PRIMARY))]
    wide = cells.pivot_table(index=["cell_id", "cohort"], columns="arm",
                             values="mae", aggfunc="first").round(4)
    refused = cells[cells["refused"].astype(bool)][["cell_id", "arm", "refusal"]]
    failed = per_cell[per_cell["arm"] == "all"][["cell_id", "refusal"]]
    lines = [
        "# Field SOH: voltage, current and time only",
        "",
        "Generated by `scripts/run_field_soh_study.py`. Each CALCE cell's raw "
        "cycler log is reduced to what a vehicle BMS has - voltage, current, "
        "time - and the pipeline segments the discharges, builds the curves, "
        "estimates its own resistance from the rest-to-load step and measures "
        "SOH from the voltage window. Nothing the cycler computed reaches the "
        "estimate.",
        "",
        "Error is mean absolute error in SOH (0.01 = one percentage point), per "
        "cell, against cycler capacity over the median of the cell's first five "
        "cycles - the same reference the estimator uses. The median is across "
        "cells; the 95% interval resamples cells. `median_cell_mae_lifetime_ref` "
        "is the same estimate scored against a lifetime-percentile reference, "
        "for comparison with `calce_voltage_window/`.",
        "",
        f"## Primary window {PRIMARY}",
        "",
        "```",
        primary.to_string(index=False),
        "```",
        "",
        "Arms: `raw` no ohmic correction; `cycler_r` corrected with the "
        "cycler's resistance column (not available on a vehicle); `step_r` "
        "corrected with resistance estimated from the load step (the field "
        "method); `step_r_partial` the field method on every discharge cut to "
        "the middle 70% of its charge.",
        "",
        "## Sensitivity: other windows (reported, not chosen between)",
        "",
        "```",
        summary[summary["window"].isin([str(w) for w in SENSITIVITY])].to_string(index=False),
        "```",
        "",
        "## Exploratory, added after the primary partial arm was refused",
        "",
        "A window inside the voltage range an 85%->15% partial discharge "
        "actually spans. Chosen after seeing that the primary window cannot be "
        "read on such partials, so it is a hypothesis, not a validated result.",
        "",
        "```",
        summary[summary["window"].isin([str(w) for w in EXPLORATORY])].to_string(index=False),
        "```",
        "",
        "## Per cell, primary window",
        "",
        "```",
        wide.to_string(),
        "```",
        "",
        "## Refused, with the reason the code gave",
        "",
        "```",
        refused.to_string(index=False) if not refused.empty else "(none)",
        "```",
    ]
    if not failed.empty:
        lines += ["", "## Cells that could not be processed", "", "```",
                  failed.to_string(index=False), "```"]
    (out / "field_soh_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
