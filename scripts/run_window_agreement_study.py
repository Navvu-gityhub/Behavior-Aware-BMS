"""Can two windows disagreeing flag the errors that consistency misses?

    python scripts/run_window_agreement_study.py

Writes reports/metrics/window_agreement/.

THE BLIND SPOT
--------------
The confidence label rests on consistency: do the readings agree with each
other once their trend is removed (field_soh.consistency_uncertainty)? That
catches scatter. It cannot catch a reading that is steadily wrong. On the
LG M50T cells every reading was rated HIGH/MEDIUM while errors were 2-3
points (reports/metrics/m50t_heldout/); on NASA B0005 a 15-point error was
rated MEDIUM.

THE CHECK, FIXED BEFORE THIS RAN
--------------------------------
If a cell has only lost capacity, its discharge curve shrinks along the
charge axis and every voltage window shrinks by the same ratio. If the curve
has changed shape or shifted - polarisation growth, electrode slippage - the
windows disagree. So, with no truth:

    d = | median of last 5 accepted readings, window 3.90-3.60 V (shipped)
        - median of last 5 accepted readings, window 4.05-3.55 V |

flag the reported estimate when d > 0.02 (two SOH points). The second
window and the threshold were chosen from that argument, not tuned.

DATA
----
Development, reported only: CALCE, NASA, Oxford, from
data/interim/cross_dataset_checks.parquet (run_cross_dataset_study.py; the
r1 and r2 columns are exactly these two windows, shipped correction).
Deciding: LG M50T, recomputed here with the shipped estimator for both
windows, truth and subset as in run_m50t_heldout_study.py. M50T is where the
blind spot was seen, but nothing in this check is fitted to it.

CRITERIA ON M50T, among estimates the consistency gate rates HIGH/MEDIUM
(u <= 0.01):
  1. flagged estimates have a median error at least twice the unflagged;
  2. the flag catches at least half of the estimates in error by > 3 points.
SUCCESS needs both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_m50t_heldout_study import FULL_AH, REST_A, files_by_cell, load_cell  # noqa: E402

from src.bms.health.field_soh import (  # noqa: E402
    apply_field_gates,
    curves_from_telemetry,
    step_overpotential,
)
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

OUT = Path("reports/metrics/window_agreement")
DEV_TABLE = Path("data/interim/cross_dataset_checks.parquet")
W1, W2 = WindowSpec(3.90, 3.60), WindowSpec(4.05, 3.55)
THRESHOLD = 0.02
TOLERANCE, RECENT, REPORT = 0.01, 10, 5


def _se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return 1.2533 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(len(values))


def points(cell: pd.DataFrame) -> list[dict]:
    """Reported estimate, consistency u and window disagreement d at each reading."""
    g = cell.sort_values("cycle")
    r1 = g.dropna(subset=["r1"])
    out = []
    soh1 = r1["r1"].to_numpy(float)
    ref_se = _se(soh1[:5])
    for k in range(9, len(r1) + 1):
        cyc = int(r1["cycle"].iloc[k - 1])
        recent = soh1[max(5, k - RECENT):k]
        idx = np.arange(len(recent))
        res = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
        u = float(np.sqrt(np.nansum([ref_se ** 2, (1.2533 * float(np.median(np.abs(res - np.median(res))))
                                                   / np.sqrt(REPORT)) ** 2])))
        est1 = float(np.median(soh1[max(5, k - REPORT):k]))
        r2 = g[(g["cycle"] <= cyc)]["r2"].dropna().to_numpy(float)
        est2 = float(np.median(r2[max(5, len(r2) - REPORT):])) if len(r2) > 5 else float("nan")
        truth = float(r1["soh_true"].iloc[k - 1])
        out.append({"cell_id": cell["cell_id"].iloc[0], "u": u, "d": abs(est1 - est2) if np.isfinite(est2) else np.nan,
                    "error": abs(est1 - truth)})
    return out


def m50t_table() -> pd.DataFrame:
    rows = []
    for (temp, cell), files in files_by_cell().items():
        cell_id = f"M50T_{temp}C_{cell}"
        tel = load_cell(files)
        d = cycles_to_frame(measure_cycles(tel, cell_id, rest_threshold_a=REST_A), complete_only=False)
        curves, steps = curves_from_telemetry(tel, d, cell_id, REST_A)
        full = d[d["capacity_ah"] >= FULL_AH].sort_values("cycle")
        if len(full) < 10:
            continue
        ref = float(full["capacity_ah"].head(5).median())
        truth = full[["cycle"]].assign(soh_true=full["capacity_ah"].to_numpy(float) / ref)
        op = step_overpotential(steps).dropna(subset=["ir_drop_v"])
        cv = curves[curves["cycle"].isin(set(op["cycle"]))]
        cols = {}
        for name, spec in (("r1", W1), ("r2", W2)):
            t = apply_field_gates(window_soh_table(cv, spec=spec, overpotential=op), d)
            cols[name] = t[["cycle", "soh_window_accepted"]].rename(columns={"soh_window_accepted": name})
        j = cols["r1"].merge(cols["r2"], on="cycle", how="outer").merge(truth, on="cycle")
        rows.append(j.assign(dataset="M50T", cell_id=cell_id))
        print(cell_id, len(j), flush=True)
    return pd.concat(rows, ignore_index=True)


def summarise(pts: pd.DataFrame, label: str) -> dict:
    g = pts[(pts["u"] <= TOLERANCE) & pts["d"].notna()]
    flag = g["d"] > THRESHOLD
    big = g["error"] > 0.03

    def cell_median(frame: pd.DataFrame) -> float:
        return float(frame.groupby("cell_id")["error"].median().median()) if len(frame) else float("nan")

    return {"data": label, "points_high_medium": len(g), "flagged": int(flag.sum()),
            "median_error_flagged": round(cell_median(g[flag]), 4),
            "median_error_unflagged": round(cell_median(g[~flag]), 4),
            "errors_over_3pts": int(big.sum()),
            "share_of_those_flagged": round(float(flag[big].mean()), 2) if big.any() else float("nan")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(DEV_TABLE)
    dev = dev[dev["soh_true"].notna()]
    rows = []
    for ds, g in dev.groupby("dataset"):
        pts = pd.DataFrame([p for _, c in g.groupby("cell_id") for p in points(c)])
        rows.append(summarise(pts, f"{ds} (development)"))
    m = m50t_table()
    test_pts = pd.DataFrame([p for _, c in m.groupby("cell_id") for p in points(c)])
    test_pts.round(5).to_csv(OUT / "m50t_points.csv", index=False)
    test = summarise(test_pts, "M50T (deciding)")
    rows.append(test)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)

    c1 = test["median_error_flagged"] >= 2 * test["median_error_unflagged"]
    c2 = test["share_of_those_flagged"] >= 0.5
    verdict = "SUCCESS" if c1 and c2 else "NOT MET"
    lines = ["# Window agreement as a bias warning", "",
             "Generated by `scripts/run_window_agreement_study.py`; window, threshold and criteria were "
             "fixed before it ran (see its docstring). Rows are estimates the consistency gate rates "
             "HIGH/MEDIUM; errors are median over cells of each cell's median.", "",
             "```", table.to_string(index=False), "```", "",
             f"- Flagged error >= 2x unflagged on M50T: {test['median_error_flagged']:.4f} vs "
             f"{test['median_error_unflagged']:.4f}: {'met' if c1 else 'not met'}",
             f"- Flag catches >= half of M50T errors over 3 points: {test['share_of_those_flagged']}: "
             f"{'met' if c2 else 'not met'}", "", f"**Verdict: {verdict}**"]
    (OUT / "window_agreement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
