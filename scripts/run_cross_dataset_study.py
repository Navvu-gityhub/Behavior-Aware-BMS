"""Cross-dataset validation, and how each kind of estimator handles unseen data.

    python scripts/run_cross_dataset_study.py            # builds the check table once, then reuses it
    python scripts/run_cross_dataset_study.py --rebuild

Writes reports/metrics/cross_dataset/. Needs the raw CALCE cache
(data/interim/calce_telemetry/, from run_field_soh_study.py), the Oxford .mat
(data/raw/oxford/) and the NASA PCoE .mat files (data/raw/nasa/mat/).

THREE DATASETS, CHOSEN TO DIFFER
--------------------------------
    CALCE   LCO prismatic, 1.1/1.35 Ah, room temperature, step resistance available
    NASA    18650, ~2 Ah, ambient 4-44 C, step resistance available
    Oxford  Kokam NMC/LCO pouch, 0.74 Ah, 40 C, no rest-to-load step

FIXED BEFORE THIS RAN
---------------------
Features, identical definitions in all three datasets: the gated window SOH
ratio (health/voltage_window + field_soh gates) for three windows -
r1 3.90-3.60 V (primary), r2 4.05-3.55 V, r3 3.85-3.55 V - corrected with
step resistance where a rest-to-load step exists (CALCE, NASA), uncorrected
where it does not (Oxford). Reference: the first 5 measurable discharges
(CALCE, NASA); the first characterisation (Oxford, checks are 100 cycles
apart). Truth: measured capacity over the same early reference.

NASA truth screen, fixed before scoring: records with capacity outside
0.3-2.5 Ah are dropped (the raw files hold zeros and constants in later
batches); cells excluded by the project's existing screen
(reports/metrics/benchmark_cell_screen.csv) stay excluded; >= 10 valid checks.
CALCE: cells with full-discharge truth; Types 5 and 6 (partial cycling) are
excluded here because the fixed-window features are not defined on them.

A. The shipped estimator, no fitting: r1 as SOH, per dataset.
B. Transfer matrix. Fitted on a SOURCE dataset, scored on each TARGET:
     identity   SOH = r1                       (no fit; the reference row)
     affine     SOH = a*r1 + b                 (fitted on source)
     hgb        HistGradientBoosting(r1,r2,r3) (fitted on source)
   In-domain baseline: leave-one-cell-out within the source.
   Metric: median per-cell MAE on target rows where the model's inputs exist,
   with the share of target rows it could score.
C. Applicability domain for hgb: a target row is IN domain when every feature
   lies within the source's 1st-99th percentile range. Error in vs out.
D. Does the shipped estimator know when it is wrong on unseen data? The
   consistency uncertainty u from run_sufficiency_study.py, tolerance 0.01,
   both fixed on CALCE, applied unchanged: at each check the estimate is the
   median of the last 5 accepted r1 readings, u = sqrt(reference SE^2 +
   recent SE^2). Error when u <= 0.01 (HIGH/MEDIUM) vs u > 0.01 (LOW):
   median over cells of each cell's median error, pooled p90, and the
   share of points within 3 SOH points.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import (  # noqa: E402
    apply_field_gates,
    curves_from_telemetry,
    step_overpotential,
)
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402
from src.bms.io.load_nasa_pcoe import (  # noqa: E402
    cohort_for,
    discharges_to_frames,
    load_nasa_pcoe_cell,
)
from src.bms.io.load_oxford import load_oxford_mat, summarize_oxford_cycles  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

OUT = Path("reports/metrics/cross_dataset")
TABLE = Path("data/interim/cross_dataset_checks.parquet")
TOLERANCE, RECENT, REPORT = 0.01, 10, 5
WINDOWS = {"r1": WindowSpec(3.90, 3.60), "r2": WindowSpec(4.05, 3.55), "r3": WindowSpec(3.85, 3.55)}
FEATURES = list(WINDOWS)
DATASETS = ("CALCE", "NASA", "Oxford")


def _windows(curves, discharges, overpotential, reference_cycles) -> pd.DataFrame:
    out = None
    for name, spec in WINDOWS.items():
        t = window_soh_table(curves, spec=spec, reference_cycles=reference_cycles,
                             overpotential=overpotential)
        t = apply_field_gates(t, discharges, reference_cycles)
        t = t[["cell_id", "cycle", "soh_window_accepted"]].rename(columns={"soh_window_accepted": name})
        out = t if out is None else out.merge(t, on=["cell_id", "cycle"], how="outer")
    return out


def build_calce() -> pd.DataFrame:
    truth = add_targets(pd.read_csv("reports/metrics/calce_full_discharge.csv"))
    truth = truth[truth["soh"].notna()]
    rows = []
    for path in sorted(Path("data/interim/calce_telemetry").glob("*.parquet")):
        cell = path.stem
        lab = truth[truth["cell_id"] == cell]
        if lab.empty or lab["cohort"].iloc[0] in ("CS2_Type5", "CS2_Type6"):
            continue
        tel = pd.read_parquet(path)
        d = cycles_to_frame(measure_cycles(tel, cell, rest_threshold_a=0.02), complete_only=False)
        t = tel["test_time_s"].to_numpy(float)
        arbin = tel["cycle"].to_numpy(float)
        d["arbin_cycle"] = [arbin[min(np.searchsorted(t, s), len(t) - 1)] for s in d["start_time_s"]]
        curves, steps = curves_from_telemetry(tel, d, cell, 0.02)
        if steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            continue
        op = step_overpotential(steps).dropna(subset=["ir_drop_v"])
        curves = curves[curves["cycle"].isin(set(op["cycle"]))]
        feats = _windows(curves, d, op, 5).merge(d[["cycle", "arbin_cycle"]], on="cycle")
        lab = lab.sort_values("arbin_cycle_index")
        ref = lab["capacity_ah"].head(5).median()
        lab = lab.assign(soh_true=lab["capacity_ah"] / ref)[["arbin_cycle_index", "soh_true"]]
        j = feats.merge(lab, left_on="arbin_cycle", right_on="arbin_cycle_index")
        j = j.assign(dataset="CALCE", cohort=truth.loc[truth["cell_id"] == cell, "cohort"].iloc[0])
        rows.append(j[["dataset", "cell_id", "cohort", "cycle", *FEATURES, "soh_true"]])
        print("CALCE", cell, len(j), flush=True)
    return pd.concat(rows, ignore_index=True)


def build_nasa() -> pd.DataFrame:
    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    rows = []
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        cell = path.stem
        if cell in excluded:
            continue
        ds = [d for d in load_nasa_pcoe_cell(path) if 0.3 <= d.capacity_ah <= 2.5]
        if len(ds) < 10:
            continue
        curves, steps = discharges_to_frames(ds)
        if steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            continue
        op = step_overpotential(steps).dropna(subset=["ir_drop_v"])
        curves = curves[curves["cycle"].isin(set(op["cycle"]))]
        disc = pd.DataFrame({"cycle": [d.index for d in ds],
                             "capacity_ah": [d.capacity_ah for d in ds]})
        feats = _windows(curves, disc, op, 5)
        ref = float(np.median([d.capacity_ah for d in ds[:5]]))
        truth = pd.DataFrame({"cycle": [d.index for d in ds],
                              "soh_true": [d.capacity_ah / ref for d in ds]})
        j = feats.merge(truth, on="cycle").assign(dataset="NASA", cohort=cohort_for(ds))
        rows.append(j[["dataset", "cell_id", "cohort", "cycle", *FEATURES, "soh_true"]])
        print("NASA", cell, len(j), flush=True)
    return pd.concat(rows, ignore_index=True)


def build_oxford() -> pd.DataFrame:
    tel, _ = load_oxford_mat("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
    truth = summarize_oxford_cycles(tel)[["cell_id", "cycle", "capacity_ah"]]
    c1 = tel[tel["measurement"] == "C1dc"][["cell_id", "cycle", "voltage_v", "capacity_ah_curve"]].copy()
    c1["capacity_ah_curve"] = c1["capacity_ah_curve"].abs()
    disc = c1.groupby(["cell_id", "cycle"], as_index=False)["capacity_ah_curve"].max().rename(
        columns={"capacity_ah_curve": "capacity_ah"})
    rows = []
    for cell, cc in c1.groupby("cell_id"):
        feats = _windows(cc, disc[disc["cell_id"] == cell], None, 1)
        lab = truth[truth["cell_id"] == cell].sort_values("cycle")
        lab = lab.assign(soh_true=lab["capacity_ah"] / lab["capacity_ah"].iloc[0])
        j = feats.merge(lab[["cycle", "soh_true"]], on="cycle").assign(dataset="Oxford", cohort="40C_artemis")
        j = j[j["cycle"] > 0]                       # the reference scores zero by construction
        rows.append(j[["dataset", "cell_id", "cohort", "cycle", *FEATURES, "soh_true"]])
    return pd.concat(rows, ignore_index=True)


def _se_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return 1.2533 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(len(values))


def consistency_points(table: pd.DataFrame) -> pd.DataFrame:
    """Reported estimate, its consistency uncertainty and its error, per check."""
    rows = []
    for (ds, cell), g in table.dropna(subset=["r1"]).groupby(["dataset", "cell_id"]):
        g = g.sort_values("cycle")
        soh, truth = g["r1"].to_numpy(float), g["soh_true"].to_numpy(float)
        ref_se = _se_median(soh[:5])
        for k in range(5 + 4, len(soh) + 1):
            recent = soh[max(5, k - RECENT):k]
            idx = np.arange(len(recent))
            res = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
            recent_se = 1.2533 * float(np.median(np.abs(res - np.median(res)))) / np.sqrt(REPORT)
            u = float(np.sqrt(np.nansum([ref_se ** 2, recent_se ** 2])))
            est = float(np.median(soh[max(5, k - REPORT):k]))
            rows.append({"dataset": ds, "cell_id": cell, "uncertainty": u,
                         "abs_error": abs(est - truth[k - 1])})
    return pd.DataFrame(rows)


def per_cell_mae(frame: pd.DataFrame, pred: np.ndarray) -> float:
    err = pd.Series(np.abs(pred - frame["soh_true"].to_numpy(float)), index=frame.index)
    return float(err.groupby(frame["cell_id"]).mean().median())


def fit_predict(model: str, src: pd.DataFrame, tgt: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predictions on tgt and a mask of rows the model could score."""
    if model == "identity":
        mask = tgt["r1"].notna().to_numpy()
        return tgt["r1"].to_numpy(float), mask
    if model == "affine":
        s = src.dropna(subset=["r1"])
        a, b = np.polyfit(s["r1"], s["soh_true"], 1)
        mask = tgt["r1"].notna().to_numpy()
        return a * tgt["r1"].to_numpy(float) + b, mask
    from sklearn.ensemble import HistGradientBoostingRegressor
    s = src.dropna(subset=FEATURES, how="all")
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4, random_state=20261006)
    m.fit(s[FEATURES], s["soh_true"])
    mask = tgt[FEATURES].notna().any(axis=1).to_numpy()
    return m.predict(tgt[FEATURES]), mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if TABLE.exists() and not args.rebuild:
        table = pd.read_parquet(TABLE)
    else:
        table = pd.concat([build_calce(), build_nasa(), build_oxford()], ignore_index=True)
        TABLE.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(TABLE, index=False)
    table = table[table["soh_true"].notna() & table[FEATURES].notna().any(axis=1)].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # A. the shipped estimator, no fitting
    a_rows = []
    for ds, g in table.groupby("dataset"):
        sc = g.dropna(subset=["r1"])
        a_rows.append({"dataset": ds, "cells_total": g["cell_id"].nunique(),
                       "cells_scored": sc["cell_id"].nunique(), "rows_scored": len(sc),
                       "share_rows_scored": round(len(sc) / len(g), 2),
                       "median_cell_mae": round(per_cell_mae(sc, sc["r1"].to_numpy(float)), 4)})
    part_a = pd.DataFrame(a_rows)
    nasa = table[table["dataset"] == "NASA"].dropna(subset=["r1"])
    by_cohort = (nasa.assign(e=(nasa["r1"] - nasa["soh_true"]).abs())
                 .groupby(["cohort", "cell_id"])["e"].mean().groupby("cohort")
                 .agg(cells="count", median_cell_mae="median").round(4).reset_index())

    # B. transfer matrix
    b_rows = []
    for src_name in DATASETS:
        src = table[table["dataset"] == src_name]
        for model in ("identity", "affine", "hgb"):
            # in-domain baseline: leave one cell out within the source
            errs = []
            for cell in src["cell_id"].unique():
                tr, te = src[src["cell_id"] != cell], src[src["cell_id"] == cell]
                pred, mask = fit_predict(model, tr, te)
                if mask.any():
                    errs.append(np.mean(np.abs(pred[mask] - te["soh_true"].to_numpy(float)[mask])))
            b_rows.append({"source": src_name, "target": f"{src_name} (leave-one-cell-out)",
                           "model": model, "median_cell_mae": round(float(np.median(errs)), 4),
                           "share_scored": 1.0})
            for tgt_name in DATASETS:
                if tgt_name == src_name:
                    continue
                tgt = table[table["dataset"] == tgt_name]
                pred, mask = fit_predict(model, src, tgt)
                b_rows.append({"source": src_name, "target": tgt_name, "model": model,
                               "median_cell_mae": round(per_cell_mae(tgt[mask], pred[mask]), 4),
                               "share_scored": round(float(mask.mean()), 2)})
    part_b = pd.DataFrame(b_rows)

    # C. applicability domain for the fitted model
    c_rows = []
    for src_name in DATASETS:
        src = table[table["dataset"] == src_name]
        lo, hi = src[FEATURES].quantile(0.01), src[FEATURES].quantile(0.99)
        for tgt_name in DATASETS:
            if tgt_name == src_name:
                continue
            tgt = table[table["dataset"] == tgt_name]
            pred, mask = fit_predict("hgb", src, tgt)
            present = tgt[FEATURES].notna()
            inside = (((tgt[FEATURES] >= lo) & (tgt[FEATURES] <= hi)) | ~present).all(axis=1).to_numpy() & mask
            err = np.abs(pred - tgt["soh_true"].to_numpy(float))
            for side, m in (("in domain", inside), ("out of domain", mask & ~inside)):
                c_rows.append({"source": src_name, "target": tgt_name, "side": side,
                               "rows": int(m.sum()),
                               "median_abs_error": round(float(np.median(err[m])), 4) if m.any() else np.nan,
                               "p90_abs_error": round(float(np.quantile(err[m], 0.9)), 4) if m.any() else np.nan})
    part_c = pd.DataFrame(c_rows)

    pts = consistency_points(table)
    d_rows = []
    for ds, g in pts.groupby("dataset"):
        for side, m in ((f"u <= {TOLERANCE} (HIGH/MEDIUM)", g["uncertainty"] <= TOLERANCE),
                        (f"u > {TOLERANCE} (LOW)", g["uncertainty"] > TOLERANCE)):
            b = g[m]
            d_rows.append({"dataset": ds, "side": side, "points": len(b), "cells": b["cell_id"].nunique(),
                           "median_cell_median_error": round(float(b.groupby("cell_id")["abs_error"].median().median()), 4) if len(b) else np.nan,
                           "p90_abs_error": round(float(b["abs_error"].quantile(0.9)), 4) if len(b) else np.nan,
                           "share_within_3pct": round(float((b["abs_error"] <= 0.03).mean()), 2) if len(b) else np.nan})
    part_d = pd.DataFrame(d_rows)

    for name, df in (("D_consistency_gate", part_d), ("A_shipped", part_a), ("A_nasa_by_cohort", by_cohort),
                     ("B_transfer", part_b), ("C_domain", part_c)):
        df.to_csv(OUT / f"{name}.csv", index=False)
    lines = ["# Cross-dataset validation and unseen-data handling", "",
             "Generated by `scripts/run_cross_dataset_study.py`; features, screens, "
             "models and metrics were fixed before it ran (see its docstring).", "",
             "## A. The shipped estimator (r1, no fitting) on each dataset", "",
             "```", part_a.to_string(index=False), "```", "",
             "NASA by cohort (ambient temperature, discharge current):", "",
             "```", by_cohort.to_string(index=False), "```", "",
             "## B. Transfer matrix (median per-cell MAE on the target)", "",
             "```", part_b.to_string(index=False), "```", "",
             "## C. Applicability domain of the fitted model (hgb)", "",
             "```", part_c.to_string(index=False), "```", "",
             "## D. The shipped consistency gate (fixed on CALCE), unchanged on each dataset", "",
             "```", part_d.to_string(index=False), "```"]
    (OUT / "cross_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
