"""Does a population prior give useful remaining life early, on unseen datasets?

    python scripts/run_population_rul_study.py

Writes reports/metrics/population_rul/. Needs the CALCE truth table (tracked),
NASA .mat (data/raw/nasa/mat), Oxford .mat (data/raw/oxford) and the
Imperial M50T per-cycle summaries (scripts/fetch_imperial_m50t.py).

THE QUESTION
------------
The shipped RUL (fade_extrapolation.estimate_rul) uses only the cell's own
history: it refuses before 30 points and is accurate only near end of life.
population_prior.posterior_rul combines a population's fade rates with the
cell's own readings. Learning from other cells is exactly what failed when
fitted SOH models moved between datasets (reports/metrics/cross_dataset/),
so the only test that counts is leave-one-DATASET-out.

FIXED BEFORE THIS RAN
---------------------
Data - capacity trajectories, SOH relative to each cell's own first 5:
  CALCE   full-discharge capacity (calce_full_discharge.csv), cohorts
          CS2_Type5/6 excluded (partial cycling: a "cycle" is not a cycle)
  NASA    NASA's discharge capacity, 0.3-2.5 Ah, the project's cell screen
  Oxford  1C characterisation capacity, at its cycle numbers
  M50T    Imperial Expt 5 discharge capacity per cycle (row order)
Threshold 0.90 for every dataset. A cell is scored if its 5-point rolling
median crosses 0.90; that crossing is its true end of life (n_eol).
Folds: each dataset held out in turn; the prior is fitted on the cells of
the other three (population_prior.cell_fade_rate, fit_prior).
Scored at f = 10%, 25%, 50%, 75% of each test cell's life (x0 -> n_eol),
using readings at or before that point. True RUL = n_eol - point.
Arms:
  population  the prior alone (no readings beyond the first)
  cell_only   same model, flat prior: the cell's own readings alone
  hybrid      prior x the cell's readings
  shipped     estimate_rul (answer rate; error when it answers)
Metrics: median over cells of |pred - true| / true; share of true RUL inside
the 80% interval (coverage).

Criteria:
  1. Early skill - at f = 10% and at f = 25%, hybrid has lower median
     relative error than BOTH cell_only and population in at least 3 of the
     4 held-out datasets.
  2. Honest uncertainty - hybrid 80%-interval coverage pooled over all
     folds and points lies between 0.65 and 0.95.
SUCCESS needs both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.io.load_nasa_pcoe import cohort_for, load_nasa_pcoe_cell  # noqa: E402
from src.bms.io.load_oxford import load_oxford_mat, summarize_oxford_cycles  # noqa: E402
from src.bms.rul.fade_extrapolation import estimate_rul  # noqa: E402
from src.bms.rul.population_prior import (  # noqa: E402
    FadePrior,
    cell_fade_rate,
    fit_prior,
    posterior_rul,
)

OUT = Path("reports/metrics/population_rul")
THRESHOLD = 0.90
FRACTIONS = (0.10, 0.25, 0.50, 0.75)
DATASETS = ("CALCE", "NASA", "Oxford", "M50T")


def _traj(dataset, cell, x, cap, temp) -> dict:
    x = np.asarray(x, float)
    cap = np.asarray(cap, float)
    ok = np.isfinite(x) & np.isfinite(cap) & (cap > 0)
    x, cap = x[ok], cap[ok]
    o = np.argsort(x, kind="stable")
    x, cap = x[o], cap[o]
    return {"dataset": dataset, "cell_id": cell, "temperature_c": temp,
            "x": x, "soh": cap / np.median(cap[:5])}


def trajectories() -> list[dict]:
    out = []
    calce = pd.read_csv("reports/metrics/calce_full_discharge.csv")
    calce = calce[~calce["cohort"].isin(["CS2_Type5", "CS2_Type6"])]
    for cell, g in calce.groupby("cell_id"):
        out.append(_traj("CALCE", cell, g["arbin_cycle_index"], g["capacity_ah"], 25.0))

    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        if path.stem in excluded:
            continue
        ds = [d for d in load_nasa_pcoe_cell(path) if 0.3 <= d.capacity_ah <= 2.5]
        if len(ds) >= 10:
            temp = float(cohort_for(ds).split("C")[0])
            out.append(_traj("NASA", path.stem, [d.index for d in ds], [d.capacity_ah for d in ds], temp))

    tel, _ = load_oxford_mat("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
    for cell, g in summarize_oxford_cycles(tel).groupby("cell_id"):
        out.append(_traj("Oxford", cell, g["cycle"], g["capacity_ah"], 40.0))

    for f in sorted(Path("data/raw/imperial_m50t").glob("*cycle_data.csv")):
        d = pd.read_csv(f)
        cell = f.name.split(" - ")[1].replace("cell ", "M50T_")
        temp = float(d["Av. Discharge Temperature [°C]"].median())
        out.append(_traj("M50T", cell, np.arange(len(d)), d["Discharge Capacity [A h]"], temp))
    return out


def end_of_life(t: dict) -> float:
    s = pd.Series(t["soh"]).rolling(5, min_periods=1).median().to_numpy()
    hit = np.flatnonzero(s <= THRESHOLD)
    return float(t["x"][hit[0]]) if len(hit) else float("nan")


def main() -> int:
    trajs = trajectories()
    for t in trajs:
        t["rate"] = cell_fade_rate(t["x"], t["soh"], THRESHOLD)
        t["n_eol"] = end_of_life(t)
    rows, priors = [], []
    for held in DATASETS:
        train = [t["rate"] for t in trajs if t["dataset"] != held]
        prior = fit_prior(np.array(train))
        priors.append({"held_out": held, "prior_mean_log_rate": prior.mean, "prior_sd": prior.sd,
                       "prior_cells": prior.n_cells,
                       "prior_median_rate_per_cycle": float(np.exp(prior.mean))})
        for t in (t for t in trajs if t["dataset"] == held and np.isfinite(t["n_eol"])):
            x0 = t["x"][0]
            if t["n_eol"] <= x0:
                continue
            for f in FRACTIONS:
                a = x0 + f * (t["n_eol"] - x0)
                true = t["n_eol"] - a
                arms = {
                    "population": posterior_rul(t["x"][:1], t["soh"][:1], a, prior, THRESHOLD),
                    "cell_only": posterior_rul(t["x"], t["soh"], a, FadePrior.flat(), THRESHOLD),
                    "hybrid": posterior_rul(t["x"], t["soh"], a, prior, THRESHOLD),
                }
                for arm, p in arms.items():
                    rows.append({"held_out": held, "cell_id": t["cell_id"], "f": f, "arm": arm,
                                 "true_rul": true, "pred": p.median, "lower": p.lower, "upper": p.upper,
                                 "n_readings": p.n_readings})
                s = estimate_rul(t["x"], t["soh"], int(a), threshold=THRESHOLD)
                rows.append({"held_out": held, "cell_id": t["cell_id"], "f": f, "arm": "shipped",
                             "true_rul": true, "pred": s.rul_cycles, "lower": np.nan, "upper": np.nan,
                             "n_readings": s.n_points})
    r = pd.DataFrame(rows)
    r["rel_error"] = (r["pred"] - r["true_rul"]).abs() / r["true_rul"]
    r["covered"] = (r["lower"] <= r["true_rul"]) & (r["true_rul"] <= r["upper"])
    OUT.mkdir(parents=True, exist_ok=True)
    r.round(5).to_csv(OUT / "points.csv", index=False)
    pd.DataFrame(priors).round(5).to_csv(OUT / "priors.csv", index=False)

    answered = r.dropna(subset=["pred"])
    table = (answered.groupby(["held_out", "f", "arm"])
             .agg(cells=("cell_id", "nunique"), median_rel_error=("rel_error", "median"))
             .round(3).reset_index())
    answer_rate = (r[r["arm"] == "shipped"].assign(ans=lambda d: d["pred"].notna())
                   .groupby(["held_out", "f"])["ans"].mean().round(2).rename("shipped_answer_rate").reset_index())
    cov = (r[r["arm"] != "shipped"].groupby(["arm", "f"])["covered"].mean().round(2)
           .rename("coverage_80").reset_index())
    table.to_csv(OUT / "relative_error.csv", index=False)
    cov.to_csv(OUT / "coverage.csv", index=False)

    wide = table.pivot_table(index=["held_out", "f"], columns="arm", values="median_rel_error")
    wins = {}
    for f in (0.10, 0.25):
        w = wide.xs(f, level="f")
        wins[f] = int(((w["hybrid"] < w["cell_only"]) & (w["hybrid"] < w["population"])).sum())
    pooled_cov = float(r[r["arm"] == "hybrid"]["covered"].mean())
    c1 = all(v >= 3 for v in wins.values())
    c2 = 0.65 <= pooled_cov <= 0.95
    verdict = "SUCCESS" if c1 and c2 else "NOT MET"

    cells = pd.DataFrame([{"dataset": t["dataset"], "cell_id": t["cell_id"], "temperature_c": t["temperature_c"],
                           "readings": len(t["x"]), "fade_rate_per_cycle": t["rate"], "n_eol_0p90": t["n_eol"]}
                          for t in trajs])
    cells.round(6).to_csv(OUT / "cells.csv", index=False)
    lines = ["# Population prior for remaining life: leave-one-dataset-out", "",
             "Generated by `scripts/run_population_rul_study.py`; data, folds, arms and criteria "
             "were fixed before it ran (see its docstring).", "",
             "## Cells per dataset (scored = crosses 0.90)", "", "```",
             cells.groupby("dataset").agg(cells=("cell_id", "count"),
                                          scored=("n_eol_0p90", lambda s: int(s.notna().sum())),
                                          median_rate=("fade_rate_per_cycle", "median")).to_string(), "```", "",
             "## Priors (learned without the held-out dataset)", "", "```",
             pd.DataFrame(priors).round(5).to_string(index=False), "```", "",
             "## Median relative error of remaining life", "", "```",
             wide.round(3).to_string(), "```", "",
             "Shipped estimator answer rate:", "", "```", answer_rate.to_string(index=False), "```", "",
             "## 80% interval coverage", "", "```", cov.to_string(index=False), "```", "",
             "## Criteria", "",
             f"- Early skill: hybrid beats both other arms in {wins[0.10]}/4 datasets at 10% of life "
             f"and {wins[0.25]}/4 at 25% (needs >= 3 each): {'met' if c1 else 'not met'}",
             f"- Coverage: hybrid pooled {pooled_cov:.2f} (needs 0.65-0.95): {'met' if c2 else 'not met'}",
             "", f"**Verdict: {verdict}**"]
    (OUT / "population_rul_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
