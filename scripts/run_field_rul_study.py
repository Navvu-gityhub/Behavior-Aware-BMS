"""RUL when complete discharges are RARE: three methods on the same evidence.

    python scripts/run_field_rul_study.py

Reads the telemetry cache built by run_field_soh_study.py and writes
`reports/metrics/calce_field_rul/`.

THE PROBLEM
-----------
RUL extrapolates SOH to 0.90. The pipeline's SOH for RUL is capacity from
COMPLETE discharges, which a driver rarely produces - CALCE CS2_24 has 18 in
1,617 - so RUL is refused for vehicle-like use. The field SOH estimator reads
every discharge, but near 0.90 it sits a median 1.1 points below capacity,
and because fade near the threshold is slow, that small bias moves the 0.90
crossing a median 56 cycles early (measured in a first pass of this study).

THREE ARMS, ALL GIVEN THE SAME SPARSE COMPLETE DISCHARGES
--------------------------------------------------------
    capacity_sparse    the current method, on the complete discharges a
                       sparse user has
    field_raw          the field SOH trajectory from every discharge
    field_selfcal      field SOH x a correction factor from THIS cell's own
                       complete discharges so far: median, over the last 5
                       of them, of (capacity SOH / window SOH) on the same
                       discharge. Causal; fits nothing across cells.

Sparsity: on the partial-cycling cells (Types 5/6) the complete discharges
are the real periodic capacity checks. On constant-current cells, where every
discharge is complete, a sparse user is EMULATED by keeping 1 complete
discharge in 50 for capacity and calibration - a service check every 50
cycles. That emulation is labelled in every output.

SCORING, FIXED BEFORE THIS RAN
------------------------------
Truth: lab capacity over the checks in the first 10 cycles; end of life where
that crosses 0.90 (smoothed). Estimates at a cycle use only data at or before
it. Accuracy is reported by the PREDICTED remaining life - what a user sees -
and by the true one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import field_soh_table, learned_field_soh  # noqa: E402
from src.bms.rul.fade_extrapolation import (  # noqa: E402
    DEFAULT_EOL_THRESHOLD,
    estimate_rul,
    observed_eol,
)
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

CACHE = Path("data/interim/calce_telemetry")
TRUTH = Path("reports/metrics/calce_full_discharge.csv")
OUT = Path("reports/metrics/calce_field_rul")
REST = 0.02
RATED = {"CS2": 1.1, "CX2": 1.35}
PARTIAL_CELLS = {"CS2_24", "CS2_25", "CS2_5", "CS2_6"}
EMULATED_EVERY = 50
CAL_LAST = 5
STRIDE = 10
BINS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 100_000))
ARMS = ("capacity_sparse", "field_raw", "field_selfcal", "capacity_sparse_lowmin")
# ADDED AFTER THE FIRST PASS, and labelled as such: capacity_sparse refused
# every estimate because sparse checks never reach MIN_HISTORY=30 points.
# Capacity checks are accurate; only the point-count gate refused. This arm
# runs the same extrapolation on the sparse checks with 5 points minimum and a
# 3-point median. The horizon bound (2x the history span) still applies.
LOWMIN_HISTORY, LOWMIN_SMOOTH = 5, 3


def cell_data(cell: str):
    tel = pd.read_parquet(CACHE / f"{cell}.parquet")
    d = cycles_to_frame(measure_cycles(tel, cell, rest_threshold_a=REST), complete_only=False)
    t = tel["test_time_s"].to_numpy(float)
    arbin = tel["cycle"].to_numpy(float)
    d["arbin_cycle"] = [arbin[min(np.searchsorted(t, s), len(t) - 1)] for s in d["start_time_s"]]
    table = field_soh_table(tel, d, cell, REST)
    mode = "fixed"
    if table.empty or table["cell_refused"].any() or table["soh_window_accepted"].notna().sum() < 30:
        _, table = learned_field_soh(tel, d, cell, REST, RATED[cell[:3]])
        mode = "learned"
    if table.empty or table["cell_refused"].any():
        return None
    per = d[["cycle", "arbin_cycle", "capacity_ah", "is_complete"]].merge(
        table[["cycle", "soh_window_accepted"]], on="cycle", how="left")
    complete = per[per["is_complete"]].sort_values("cycle")
    if cell in PARTIAL_CELLS:
        sparse = complete
        sparsity = "real (lab capacity checks)"
    else:
        sparse = complete.iloc[::EMULATED_EVERY]
        sparsity = f"emulated: 1 complete discharge in {EMULATED_EVERY}"
    if len(sparse) < 2:
        return None
    ref = sparse["capacity_ah"].head(5).median()
    sparse = sparse.assign(cap_soh=sparse["capacity_ah"] / ref)
    return per, sparse, mode, sparsity


def trajectories(per: pd.DataFrame, sparse: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {"capacity_sparse": sparse[["arbin_cycle", "cap_soh"]].rename(columns={"cap_soh": "soh"})}
    field = per.dropna(subset=["soh_window_accepted"])[["cycle", "arbin_cycle", "soh_window_accepted"]]
    out["field_raw"] = field.rename(columns={"soh_window_accepted": "soh"})[["arbin_cycle", "soh"]]
    # Self-calibration: on discharges where BOTH were measured, the ratio of
    # capacity SOH to window SOH; each field reading is scaled by the median of
    # the last CAL_LAST ratios available AT OR BEFORE it - never later ones.
    both = sparse.dropna(subset=["soh_window_accepted"])
    ratio = (both["cap_soh"] / both["soh_window_accepted"]).to_numpy(float)
    cal_cycles = both["cycle"].to_numpy(float)
    factors = []
    for c in field["cycle"].to_numpy(float):
        k = np.searchsorted(cal_cycles, c, side="right")
        factors.append(np.median(ratio[max(0, k - CAL_LAST):k]) if k > 0 else np.nan)
    cal = field.assign(soh=field["soh_window_accepted"].to_numpy(float) * np.array(factors))
    out["field_selfcal"] = cal.dropna(subset=["soh"])[["arbin_cycle", "soh"]]
    out["capacity_sparse_lowmin"] = out["capacity_sparse"]
    return {k: v.groupby("arbin_cycle", as_index=False)["soh"].median() for k, v in out.items()}


def main() -> int:
    truth = add_targets(pd.read_csv(TRUTH))
    truth = truth[truth["soh"].notna()]
    rows, cells = [], []
    for path in sorted(CACHE.glob("*.parquet")):
        cell = path.stem
        data = cell_data(cell)
        lab = truth[truth["cell_id"] == cell].sort_values("arbin_cycle_index")
        if data is None or lab.empty:
            cells.append({"cell_id": cell, "status": "refused or no data"})
            continue
        per, sparse, mode, sparsity = data
        early = lab[lab["arbin_cycle_index"] <= 10]
        ref = (early if not early.empty else lab.head(1))["capacity_ah"].median()
        eol = observed_eol(lab["arbin_cycle_index"].to_numpy(float),
                           (lab["capacity_ah"] / ref).to_numpy(float), DEFAULT_EOL_THRESHOLD)
        if not np.isfinite(eol):
            cells.append({"cell_id": cell, "status": "lab never crosses 0.90"})
            continue
        trajs = trajectories(per, sparse)
        grid = trajs["field_raw"]["arbin_cycle"].to_numpy(float)[::STRIDE]
        for at in grid:
            if at >= eol:
                break
            for arm in ARMS:
                tr = trajs[arm]
                kw = ({"min_history": LOWMIN_HISTORY, "smooth": LOWMIN_SMOOTH}
                      if arm == "capacity_sparse_lowmin" else {})
                est = estimate_rul(tr["arbin_cycle"].to_numpy(float), tr["soh"].to_numpy(float),
                                   at_cycle=at, threshold=DEFAULT_EOL_THRESHOLD, **kw)
                rows.append({"cell_id": cell, "arm": arm, "cycle": at, "sparsity": sparsity,
                             "rul_pred": est.rul_cycles, "rul_true": eol - at,
                             "error": est.rul_cycles - (eol - at), "refused": est.refusal != ""})
        cells.append({"cell_id": cell, "status": "scored", "window": mode, "sparsity": sparsity,
                      "n_complete_used": len(sparse), "lab_eol_cycle": eol})

    est = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    est.to_csv(OUT / "field_rul_estimates.csv", index=False)
    pd.DataFrame(cells).to_csv(OUT / "field_rul_cells.csv", index=False)

    def summarise(frame: pd.DataFrame, by: str) -> pd.DataFrame:
        out = []
        for arm in ARMS:
            a = frame[frame["arm"] == arm]
            answered = a.dropna(subset=["rul_pred"])
            for lo, hi in BINS:
                b = answered[(answered[by] >= lo) & (answered[by] < hi)]
                if b.empty:
                    continue
                later = b["rul_true"] - b["rul_pred"]
                out.append({"arm": arm, "range": f"{lo}-{hi}" if hi < 100_000 else f"{lo}+",
                            "n": len(b), "cells": b["cell_id"].nunique(),
                            "median_abs_error": round(float(b["error"].abs().median()), 1),
                            "within_20": round(float((b["error"].abs() <= 20).mean()), 2),
                            "lasted_at_least_predicted": round(float((later >= 0).mean()), 2)})
        answered_share = (frame.groupby("arm")["rul_pred"].apply(lambda s: s.notna().mean())
                          .round(2).rename("share_answered").reset_index())
        return pd.DataFrame(out), answered_share

    blocks = []
    for name, frame in (("ALL CELLS", est),
                        ("PARTIAL-CYCLING CELLS (real sparsity)", est[est["cell_id"].isin(PARTIAL_CELLS)]),
                        ("CONSTANT-CURRENT CELLS (emulated sparsity)", est[~est["cell_id"].isin(PARTIAL_CELLS)])):
        if frame.empty:
            continue
        by_pred, answered = summarise(frame, "rul_pred")
        by_true, _ = summarise(frame, "rul_true")
        blocks += [f"## {name}", "", "Share of estimate points with an answer (not refused):",
                   "", "```", answered.to_string(index=False), "```", "",
                   "By PREDICTED remaining life (what a user sees):", "", "```",
                   by_pred.to_string(index=False), "```", "",
                   "By TRUE remaining life:", "", "```", by_true.to_string(index=False), "```", ""]
    lines = ["# RUL when complete discharges are rare", "",
             "Generated by `scripts/run_field_rul_study.py`; method, sparsity and "
             "scoring are in its docstring, fixed before it ran.", "", *blocks,
             "## Per cell", "", "```", pd.DataFrame(cells).to_string(index=False), "```"]
    (OUT / "field_rul_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
