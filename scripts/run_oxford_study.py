"""External validation on the Oxford Battery Degradation Dataset 1.

    python scripts/run_oxford_study.py

Needs `data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat`
(Howey & Birkl 2017, doi:10.5287/bodleian:KO2kdmYGg, ODC-ODbL). Writes
`reports/metrics/oxford/`.

WHY THIS DATASET
----------------
Everything else in this project was validated on CALCE: LCO prismatic cells,
one manufacturer, room temperature, trajectories that end near 81%. Oxford
differs on every one of those axes: Kokam SLPB533459H4 pouch cells (740 mAh,
NMC/LCO blend cathode), a different manufacturer and format, aged at 40 C
under an Artemis urban drive cycle, and faded to roughly 60-79% of new, so
the 80% automotive end-of-life convention is crossed and can be scored.

Capacity is measured every 100 ageing cycles: a 1C discharge (C1dc) and a
C/18 pseudo-OCV discharge (OCVdc).

FIXED BEFORE THIS RAN - nothing is re-tuned for this chemistry
-------------------------------------------------------------
SOH, estimate from each characterisation's discharge curve:
  window_1C       the 3.90-3.60 V window on the 1C discharge, uncorrected
                  (each block starts under load, so no rest-to-load step
                  exists to estimate resistance from)
  window_OCV      the same window on the C/18 discharge, where ohmic sag is
                  negligible - the window principle without the correction
  learned_1C      the learned window (health/field_soh.learn_usage_window),
                  learned on the cell's first characterisations only
Reference: the FIRST characterisation, the only beginning-of-life
measurement. Truth: 1C discharge capacity over the first characterisation's.

RUL, the shipped estimator (rul/fade_extrapolation.estimate_rul, unchanged)
on the capacity trajectory, scored against the observed crossing at 0.90 AND
0.80. By the true distance and by the prediction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import learn_usage_window  # noqa: E402
from src.bms.health.voltage_window import WindowSpec, window_charge  # noqa: E402
from src.bms.io.load_oxford import (  # noqa: E402
    OXFORD_NOMINAL_CAPACITY_AH,
    load_oxford_mat,
    summarize_oxford_cycles,
)
from src.bms.rul.fade_extrapolation import estimate_rul, observed_eol  # noqa: E402

RAW = Path("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
OUT = Path("reports/metrics/oxford")
PRIMARY = WindowSpec(3.90, 3.60)
LEARN_FIRST = 10
THRESHOLDS = (0.90, 0.80)
BINS = ((0, 500), (500, 1000), (1000, 2000), (2000, 100_000))


def curves(tel: pd.DataFrame, block: str) -> pd.DataFrame:
    c = tel[tel["measurement"] == block][["cell_id", "cycle", "voltage_v", "capacity_ah_curve"]].copy()
    c["capacity_ah_curve"] = c["capacity_ah_curve"].abs()
    return c


def soh_rows(tel: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    c1, ocv = curves(tel, "C1dc"), curves(tel, "OCVdc")
    for cell, lab in truth.groupby("cell_id"):
        lab = lab.sort_values("cycle")
        lab = lab.assign(soh=lab["capacity_ah"] / lab["capacity_ah"].iloc[0])
        learned = learn_usage_window(
            c1[c1["cell_id"] == cell].rename(columns={}), OXFORD_NOMINAL_CAPACITY_AH, LEARN_FIRST)
        for arm, frame, spec in (("window_1C", c1, PRIMARY), ("window_OCV", ocv, PRIMARY),
                                 ("learned_1C", c1, learned.spec)):
            if spec is None:
                rows.append({"cell_id": cell, "arm": arm, "refusal": learned.refusal})
                continue
            cell_curves = frame[frame["cell_id"] == cell]
            q = {}
            for cyc, b in cell_curves.groupby("cycle"):
                q[int(cyc)] = window_charge(b["voltage_v"].to_numpy(float),
                                            b["capacity_ah_curve"].to_numpy(float), spec)
            series = pd.Series(q).sort_index()
            ref = series.dropna().iloc[0] if series.notna().any() else np.nan
            for r in lab.itertuples():
                est = series.get(int(r.cycle), np.nan) / ref if np.isfinite(ref) else np.nan
                rows.append({"cell_id": cell, "arm": arm, "window": str(spec), "cycle": int(r.cycle),
                             "soh_true": r.soh, "soh_est": est, "error": est - r.soh})
    return pd.DataFrame(rows)


def rul_rows(truth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, lab in truth.groupby("cell_id"):
        lab = lab.sort_values("cycle")
        x = lab["cycle"].to_numpy(float)
        y = (lab["capacity_ah"] / lab["capacity_ah"].iloc[0]).to_numpy(float)
        for thr in THRESHOLDS:
            eol = observed_eol(x, y, thr)
            if not np.isfinite(eol):
                continue
            for at in x:
                if at >= eol:
                    break
                est = estimate_rul(x, y, at_cycle=at, threshold=thr)
                rows.append({"cell_id": cell, "threshold": thr, "cycle": at, "rul_true": eol - at,
                             "rul_pred": est.rul_cycles, "refusal": est.refusal,
                             "error": est.rul_cycles - (eol - at)})
    return pd.DataFrame(rows)


def rul_table(r: pd.DataFrame, by: str) -> pd.DataFrame:
    out = []
    for thr, g in r.dropna(subset=["rul_pred"]).groupby("threshold"):
        for lo, hi in BINS:
            b = g[(g[by] >= lo) & (g[by] < hi)]
            if b.empty:
                continue
            tol = 0.1 * b["rul_true"].clip(lower=100)   # +/-10% of remaining life, >= 10 cycles
            out.append({"threshold": thr, "range": f"{lo}-{hi}" if hi < 100_000 else f"{lo}+",
                        "n": len(b), "cells": b["cell_id"].nunique(),
                        "median_abs_error": round(float(b["error"].abs().median()), 0),
                        "within_10pct": round(float((b["error"].abs() <= tol).mean()), 2),
                        "lasted_at_least_predicted": round(float((b["rul_true"] >= b["rul_pred"]).mean()), 2)})
    return pd.DataFrame(out)


def main() -> int:
    tel, report = load_oxford_mat(RAW)
    truth = summarize_oxford_cycles(tel)[["cell_id", "cycle", "capacity_ah"]]
    OUT.mkdir(parents=True, exist_ok=True)

    soh = soh_rows(tel, truth)
    soh.to_csv(OUT / "oxford_soh_points.csv", index=False)
    scored = soh.dropna(subset=["error"])
    scored = scored[scored["cycle"] > 0]          # the reference itself scores zero by construction
    per_cell = (scored.assign(abs_error=scored["error"].abs())
                .groupby(["arm", "cell_id"])["abs_error"].mean().rename("mae").reset_index())
    summary = (per_cell.groupby("arm")["mae"]
               .agg(cells="count", median_cell_mae="median", worst="max").round(4).reset_index())
    soh_min = truth.groupby("cell_id").apply(
        lambda g: (g["capacity_ah"] / g["capacity_ah"].iloc[0]).min(), include_groups=False)

    rul = rul_rows(truth)
    rul.to_csv(OUT / "oxford_rul_points.csv", index=False)
    answered = rul.groupby("threshold")["rul_pred"].apply(lambda s: round(float(s.notna().mean()), 2))

    lines = [
        "# External validation: Oxford Battery Degradation Dataset 1", "",
        "Generated by `scripts/run_oxford_study.py`; arms, reference and scoring "
        "were fixed before it ran (see its docstring). Nothing was re-tuned for "
        "this chemistry.", "",
        f"{report.n_cells} Kokam pouch cells (NMC/LCO blend, 740 mAh, 40 C), "
        f"{report.n_characterisations} characterisations, capacity units detected "
        f"as {report.capacity_units_detected}. Lowest SOH reached per cell: "
        + ", ".join(f"{c} {v:.2f}" for c, v in soh_min.items()) + ".", "",
        "## SOH (median per-cell mean absolute error, excluding the reference itself)", "",
        "```", summary.to_string(index=False), "```", "",
        "```", per_cell.pivot(index="cell_id", columns="arm", values="mae").round(4).to_string(), "```", "",
        "## RUL, shipped estimator on the capacity trajectory", "",
        "Share of points answered (not refused): "
        + ", ".join(f"{t:.2f}: {v:.0%}" for t, v in answered.items()), "",
        "Tolerance: within 10% of the true remaining life (at least 10 cycles). "
        "Cycles here are ageing cycles, spaced 100 between capacity checks, so "
        "the CALCE +/-20-cycle tolerance is not meaningful at this scale.", "",
        "By TRUE remaining life:", "", "```", rul_table(rul, "rul_true").to_string(index=False), "```", "",
        "By PREDICTED remaining life:", "", "```", rul_table(rul, "rul_pred").to_string(index=False), "```",
    ]
    (OUT / "oxford_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
