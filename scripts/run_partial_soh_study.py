"""Can a BMS track SOH from partial discharges only? Real partial cycling, CALCE.

    python scripts/run_partial_soh_study.py

Reads the per-cell telemetry cache `scripts/run_field_soh_study.py` builds in
`data/interim/calce_telemetry/` and writes `reports/metrics/calce_partial_soh/`.

THE DATA
--------
CALCE CS2 Types 5 and 6 are real partial cycling, not truncated full
discharges: thousands of ~0.25 Ah discharges at 0.5C in a fixed band, with a
full capacity check every ~100 cycles.

    Type 6 (CS2_24, CS2_25)   4.07 -> 3.78 V   top of charge
    Type 5 (CS2_5,  CS2_6)    3.69 -> 2.70 V   bottom of charge

That is what a vehicle produces and what the fixed 3.90-3.60 V window cannot
read: neither band crosses it.

THE METHOD (health/field_soh.learned_field_soh)
------------------------------------------------
The window is learned from the cell's first 10 discharges only - the band
they all cover, then the flattest sub-window in it - and refused when even
that is on the knee of the curve. No ground truth, no later data.

SCORING, FIXED BEFORE RUNNING
-----------------------------
Truth: each full capacity check's capacity over the cell's FIRST check. Not
the first five: on these cells five checks span ~400 cycles, so that reference
would already include several percent of fade.

Estimate at a check: the median of the last REPORT_CYCLES accepted readings
from PARTIAL discharges at or before that check's cycle. The full discharges
are excluded from the estimate, so the capacity checks never inform it, and
only past readings are used.

CONTROL: the same learned mode on the 17 constant-current cells, where every
discharge is full. It tests that the fallback does not break what the fixed
window already does; there the estimate uses all discharges, since no partial
exists, and the result sits beside the fixed-window study's.

Nominal capacities from the CALCE cell specifications: CS2 1.1 Ah, CX2
1.35 Ah. They set the steepness scale only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import REPORT_CYCLES, learned_field_soh  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

CACHE = Path("data/interim/calce_telemetry")
TRUTH = Path("reports/metrics/calce_full_discharge.csv")
OUT = Path("reports/metrics/calce_partial_soh")
REST_THRESHOLD_A = 0.02
RATED_AH = {"CS2": 1.1, "CX2": 1.35}
PARTIAL_CELLS = {"CS2_24": "top of charge", "CS2_25": "top of charge",
                 "CS2_5": "bottom of charge", "CS2_6": "bottom of charge"}
# A discharge delivering under this share of the cell's largest is a partial.
PARTIAL_SHARE = 0.6


def score_cell(cell: str, truth: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    telemetry = pd.read_parquet(CACHE / f"{cell}.parquet")
    discharges = cycles_to_frame(
        measure_cycles(telemetry, cell, rest_threshold_a=REST_THRESHOLD_A),
        complete_only=False)
    t = telemetry["test_time_s"].to_numpy(float)
    arbin = telemetry["cycle"].to_numpy(float)
    discharges["arbin_cycle"] = [
        arbin[min(np.searchsorted(t, s), len(t) - 1)]
        for s in discharges["start_time_s"]]
    discharges["partial"] = (
        discharges["capacity_ah"] < PARTIAL_SHARE * discharges["capacity_ah"].max())

    result, table = learned_field_soh(
        telemetry, discharges, cell, REST_THRESHOLD_A, RATED_AH[cell[:3]])
    row = {"cell_id": cell, "group": PARTIAL_CELLS.get(cell, "full discharges (control)"),
           "n_discharges": len(discharges), "n_partial": int(discharges["partial"].sum()),
           "window": result.window, "refusal": result.refusal,
           "steepness": float(table["window_steepness"].iloc[0]) if not table.empty else np.nan}
    if table.empty or result.refusal and "learned" not in result.window:
        return {**row, "n_scored": 0, "mae": np.nan, "bias": np.nan}, pd.DataFrame()

    use_partials = cell in PARTIAL_CELLS
    readings = table[["cycle", "soh_window_accepted"]].merge(
        discharges[["cycle", "arbin_cycle", "partial"]], on="cycle")
    readings = readings.dropna(subset=["soh_window_accepted"])
    if use_partials:
        readings = readings[readings["partial"]]
    readings = readings.sort_values("arbin_cycle")

    checks = truth[truth["cell_id"] == cell].sort_values("arbin_cycle_index")
    first = float(checks["capacity_ah"].iloc[0])
    scored = []
    for check in checks.itertuples():
        past = readings[readings["arbin_cycle"] <= check.arbin_cycle_index].tail(REPORT_CYCLES)
        if len(past) == REPORT_CYCLES:
            scored.append({"cell_id": cell, "arbin_cycle": check.arbin_cycle_index,
                           "soh_true": check.capacity_ah / first,
                           "soh_est": float(past["soh_window_accepted"].median())})
    s = pd.DataFrame(scored)
    if s.empty or bool(table["cell_refused"].any()):
        return {**row, "n_scored": 0, "mae": np.nan, "bias": np.nan,
                "refusal": row["refusal"] or "no check had enough prior readings"}, s
    s["error"] = s["soh_est"] - s["soh_true"]
    return {**row, "n_scored": len(s), "mae": float(s["error"].abs().mean()),
            "bias": float(s["error"].mean()),
            "soh_true_min": float(s["soh_true"].min())}, s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    truth = add_targets(pd.read_csv(TRUTH))
    truth = truth[truth["soh"].notna()]

    cells = sorted(p.stem for p in CACHE.glob("*.parquet"))
    rows, traces = [], []
    for cell in cells:
        row, trace = score_cell(cell, truth)
        rows.append(row)
        if not trace.empty:
            traces.append(trace)
        print(f"{cell:7s} {row['group']:26s} {row['window']:26s} "
              f"steep={row['steepness']:.2f} mae={row['mae']:.4f} "
              f"{row['refusal'][:70]}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    per_cell = pd.DataFrame(rows)
    per_cell.to_csv(args.out / "partial_soh_per_cell.csv", index=False)
    if traces:
        pd.concat(traces).to_csv(args.out / "partial_soh_trace.csv", index=False)
    _report(per_cell, args.out)
    return 0


def _report(per_cell: pd.DataFrame, out: Path) -> None:
    def block(frame: pd.DataFrame) -> str:
        cols = ["cell_id", "group", "window", "steepness", "n_scored", "mae", "bias", "refusal"]
        return frame[cols].round(4).to_string(index=False)

    partial = per_cell[per_cell["cell_id"].isin(PARTIAL_CELLS)]
    control = per_cell[~per_cell["cell_id"].isin(PARTIAL_CELLS)]
    scored = control.dropna(subset=["mae"])
    lines = [
        "# SOH from partial discharges",
        "",
        "Generated by `scripts/run_partial_soh_study.py`. See its docstring for "
        "the data, the method and the scoring, all fixed before it ran.",
        "",
        "Error is absolute SOH error at each full capacity check, against the "
        "cell's first check (0.01 = one percentage point). The estimate uses "
        "only partial discharges before the check.",
        "",
        "## Real partial cycling",
        "",
        "```",
        block(partial),
        "```",
        "",
        "## Control: learned window on constant-current cells",
        "",
        f"{len(scored)} of {len(control)} scored; median per-cell error "
        f"{scored['mae'].median():.4f}." if len(scored) else "none scored",
        "",
        "```",
        block(control),
        "```",
    ]
    (out / "partial_soh_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
