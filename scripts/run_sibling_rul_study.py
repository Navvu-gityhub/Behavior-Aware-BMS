"""Remaining life from sibling cells of the same type: a pre-registered test.

    python scripts/fetch_imperial_summaries.py 4 5       # development
    python scripts/fetch_imperial_summaries.py 1 2 3     # deciding - only after this file is committed
    python scripts/run_sibling_rul_study.py

Writes reports/metrics/sibling_rul/.

WHY
---
Early RUL learned across datasets failed twice (validation matrix rows
28-29). The audit (reports/metrics/audit/) found 72% of the variance in log
life lies BETWEEN datasets, while within one cell type early fade predicted
life (Spearman 0.99 on M50T, 0.76 on NASA). The realistic use is a fleet of
one cell model: learn from the sibling cells that have already aged.

FIXED BEFORE THE DECIDING DATA WAS DOWNLOADED
---------------------------------------------
Cells: LG M50T, all from Kirkaldy et al. 2024 (doi:10.5281/zenodo.10637534).
  Development: Expt 4 (drive cycle) and Expt 5 (1C), 16 cells, already seen.
  Deciding:    Expt 1, 2, 3 (state-of-charge windows 0-30, 70-85, 85-100%),
               never opened by this project.
Truth: SOH = C/10 capacity at each reference performance test (RPT) over the
  cell's first RPT. Time: equivalent full cycles, EFC = charge throughput /
  (2 x 5 Ah), since throughput counts charge and discharge. Life = EFC at the
  first crossing of 0.90, linearly interpolated between RPTs. A cell that never
  crosses is right-censored at its last RPT.
Early information: the fade rate at the 2nd RPT, d = (1 - SOH_2) / EFC_2.
Arms, each giving a predicted life for a test cell from data up to its 2nd RPT:
  A population   median life of the training cells (no early information);
                 lower bound = their 10th percentile.
  B own-trend    0.10 / d: the cell's own early rate extrapolated (no learning).
  C siblings     log(life) = a + b log(d), least squares on training cells that
                 crossed; lower bound = prediction x exp(10th percentile of the
                 leave-one-cell-out residuals on the training cells).
Scoring (deciding set; training = all 16 development cells):
  point error  |pred - life| / life on cells that crossed.
  coverage     share with true life >= bound. A censored cell counts as
               covered if its last observed EFC >= bound; otherwise it is
               unknown and excluded.
  usefulness   median of bound / life on cells that crossed.
Criteria:
  0. At least 4 deciding cells cross 0.90, else UNDECIDED.
  1. C's median relative error <= 0.75 x the smaller of A's and B's.
  2. C's lower-bound coverage >= 0.80.
  3. C's median bound / life >= 0.4.
SUCCESS needs 1-3. Development (leave-one-cell-out within Expt 4+5) is
reported and decides nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw/imperial_summaries")
OUT = Path("reports/metrics/sibling_rul")
NOMINAL_AH = 5.0
THRESHOLD = 0.90
EARLY_RPT = 2
DEV, DECIDE = (4, 5), (1, 2, 3)


def cell_record(path: Path, expt: int) -> dict | None:
    d = pd.read_csv(path)
    cap = pd.to_numeric(d["C/10 Capacity [mA h]"], errors="coerce").to_numpy(float)
    thr = pd.to_numeric(d["Charge Throughput [A h]"], errors="coerce").to_numpy(float)
    ok = np.isfinite(cap) & np.isfinite(thr) & (cap > 0)
    cap, thr = cap[ok], thr[ok]
    if len(cap) <= EARLY_RPT:
        return None
    soh = cap / cap[0]
    efc = thr / (2 * NOMINAL_AH)
    below = np.flatnonzero(soh <= THRESHOLD)
    if len(below) and below[0] > 0:
        k = below[0]
        life = float(np.interp(THRESHOLD, [soh[k], soh[k - 1]], [efc[k], efc[k - 1]]))
        crossed = True
    else:
        life, crossed = float(efc[-1]), False
    rate = (1 - soh[EARLY_RPT]) / efc[EARLY_RPT] if efc[EARLY_RPT] > 0 else float("nan")
    m = re.search(r"cell (\w+) \((\-?\d+)degC\)", path.name)
    return {"expt": expt, "cell": f"E{expt}_{m.group(1) if m else path.stem}",
            "temperature_c": float(m.group(2)) if m else np.nan,
            "early_efc": float(efc[EARLY_RPT]), "early_rate": float(rate),
            "life_efc": life, "crossed": crossed, "last_efc": float(efc[-1]), "rpts": len(soh)}


def load(expts) -> pd.DataFrame:
    rows = []
    for e in expts:
        for f in sorted((RAW / f"expt{e}").glob("*.csv")):
            r = cell_record(f, e)
            if r:
                rows.append(r)
    return pd.DataFrame(rows)


def fit(train: pd.DataFrame) -> tuple[float, float, float]:
    """(a, b, q10 of leave-one-cell-out log residuals) on crossing cells."""
    t = train[train["crossed"] & (train["early_rate"] > 0)]
    x, y = np.log(t["early_rate"].to_numpy()), np.log(t["life_efc"].to_numpy())
    b, a = np.polyfit(x, y, 1)
    res = []
    for i in range(len(t)):
        m = np.arange(len(t)) != i
        bi, ai = np.polyfit(x[m], y[m], 1)
        res.append(y[i] - (ai + bi * x[i]))
    return float(a), float(b), float(np.quantile(res, 0.10))


def predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    crossed = train[train["crossed"]]["life_efc"]
    a, b, q10 = fit(train)
    out = test.copy()
    out["A_pred"], out["A_bound"] = float(crossed.median()), float(crossed.quantile(0.10))
    out["B_pred"] = np.where(out["early_rate"] > 0, 0.10 / out["early_rate"], np.nan)
    out["C_pred"] = np.where(out["early_rate"] > 0, np.exp(a + b * np.log(out["early_rate"].clip(lower=1e-12))), np.nan)
    out["C_bound"] = out["C_pred"] * np.exp(q10)
    out.attrs["fit"] = (a, b, q10)
    return out


def score(p: pd.DataFrame) -> dict:
    c = p[p["crossed"]]

    def rel(col):
        return float(((c[col] - c["life_efc"]).abs() / c["life_efc"]).median()) if len(c) else np.nan

    cov_ok = np.where(p["crossed"], p["life_efc"] >= p["C_bound"],
                      np.where(p["last_efc"] >= p["C_bound"], True, np.nan))
    cov = pd.Series(cov_ok, dtype=float).dropna()
    return {"cells": len(p), "crossed": int(p["crossed"].sum()),
            "A_rel_error": round(rel("A_pred"), 3), "B_rel_error": round(rel("B_pred"), 3),
            "C_rel_error": round(rel("C_pred"), 3),
            "C_bound_coverage": round(float(cov.mean()), 2) if len(cov) else np.nan,
            "C_bound_over_life": round(float((c["C_bound"] / c["life_efc"]).median()), 2) if len(c) else np.nan}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load(DEV)
    loo = pd.concat([predict(dev.drop(i), dev.loc[[i]]) for i in dev.index], ignore_index=True)
    dev_score = score(loo)
    rows = [{"set": "development, leave-one-cell-out (Expt 4+5)", **dev_score}]
    decide = load(DECIDE)
    verdict, notes = "UNDECIDED", []
    if decide.empty:
        notes.append("deciding data not present: run fetch_imperial_summaries.py 1 2 3")
    else:
        p = predict(dev, decide)
        p.round(4).to_csv(OUT / "deciding_cells.csv", index=False)
        s = score(p)
        rows.append({"set": "DECIDING: Expt 1-3, trained on Expt 4+5", **s})
        a, b, q10 = p.attrs["fit"]
        notes.append(f"fit on development: log(life) = {a:.2f} + {b:.2f} log(rate); 10th pct residual {q10:.2f}")
        if s["crossed"] < 4:
            notes.append(f"only {s['crossed']} deciding cells cross {THRESHOLD}: UNDECIDED")
        else:
            c1 = s["C_rel_error"] <= 0.75 * min(s["A_rel_error"], s["B_rel_error"])
            c2 = s["C_bound_coverage"] >= 0.80
            c3 = s["C_bound_over_life"] >= 0.4
            notes += [f"1. siblings error {s['C_rel_error']} vs 0.75 x min({s['A_rel_error']}, {s['B_rel_error']}): "
                      f"{'met' if c1 else 'not met'}",
                      f"2. lower-bound coverage {s['C_bound_coverage']} (>= 0.80): {'met' if c2 else 'not met'}",
                      f"3. bound / life {s['C_bound_over_life']} (>= 0.4): {'met' if c3 else 'not met'}"]
            verdict = "SUCCESS" if c1 and c2 and c3 else "NOT MET"
    dev.round(4).to_csv(OUT / "development_cells.csv", index=False)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "summary.csv", index=False)
    lines = ["# Remaining life from sibling cells (LG M50T)", "",
             "Generated by `scripts/run_sibling_rul_study.py`; arms and criteria were fixed before the "
             "deciding data was downloaded (see its docstring). Life in equivalent full cycles to 90% "
             "of C/10 capacity.", "", "```", table.to_string(index=False), "```", "",
             *[f"- {n}" for n in notes], "", f"**Verdict: {verdict}**"]
    (OUT / "sibling_rul_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
