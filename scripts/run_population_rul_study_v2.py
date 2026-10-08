"""Population prior for remaining life, v2: development folds and a fresh test.

    python scripts/run_population_rul_study_v2.py

Writes reports/metrics/population_rul_v2/.

WHY A v2, AND WHY IT NEEDS A FRESH TEST
---------------------------------------
v1 (run_population_rul_study.py, reports/metrics/population_rul/) failed its
preset criteria. The causes were diagnosed on the same four datasets, and
v2 (population_prior.posterior_life) changes exactly those: a power-law
shape whose end-of-life cycle L and shape p are both learned as a
population, and noise from the model's own misfit inflated for residual
autocorrelation. Because v2 was designed after seeing v1's results there,
its leave-one-dataset-out scores on CALCE, NASA, Oxford and M50T are
DEVELOPMENT numbers and decide nothing.

THE DECIDING TEST, FIXED BEFORE ITS DATA WAS DOWNLOADED
-------------------------------------------------------
Imperial Expt 4 (Kirkaldy et al. 2024, doi:10.5281/zenodo.10637534): LG M50T
cycled 0-100% with a DRIVE-CYCLE discharge at 10, 25 and 40 C - a usage
pattern none of the four development datasets has. Only its per-cycle
summary files are read. The prior is fitted on all cells of the four
development datasets; nothing is fitted on Expt 4.

Same definitions as v1: SOH over each cell's first 5, threshold 0.90,
end of life where the 5-point rolling median first crosses it, scoring at
f = 10, 25, 50, 75% of life, arms population / cell_only / hybrid /
shipped, median relative error and 80%-interval coverage.

Criteria on Expt 4:
  1. At f = 10% and at f = 25%, hybrid's median relative error is lower
     than both cell_only's and population's.
  2. Hybrid 80%-interval coverage, pooled over all points, is 0.65-0.95.
SUCCESS needs both. If Expt 4 has fewer than 3 cells crossing 0.90, the
test is reported as UNDECIDED rather than scored.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_imperial_m50t import _RangeFile  # noqa: E402
from run_population_rul_study import end_of_life, trajectories  # noqa: E402

from src.bms.rul.fade_extrapolation import estimate_rul  # noqa: E402
from src.bms.rul.population_prior import (  # noqa: E402
    LifePrior,
    fit_life_curve,
    fit_life_prior,
    posterior_life,
)

OUT = Path("reports/metrics/population_rul_v2")
EXPT4_URL = ("https://zenodo.org/api/records/10637534/files/"
             "Expt%204%20-%20Drive%20Cycle%20Aging%20(Control).zip/content")
EXPT4_DIR = Path("data/raw/imperial_m50t_expt4")
THRESHOLD = 0.90
FRACTIONS = (0.10, 0.25, 0.50, 0.75)
DEV = ("CALCE", "NASA", "Oxford", "M50T")


def expt4_trajectories() -> list[dict]:
    EXPT4_DIR.mkdir(parents=True, exist_ok=True)
    if not list(EXPT4_DIR.glob("*cycle_data.csv")):
        z = zipfile.ZipFile(io.BufferedReader(_RangeFile(EXPT4_URL), buffer_size=4 << 20))
        for name in z.namelist():
            if "Summary per Cycle" in name and name.endswith(".csv"):
                (EXPT4_DIR / name.rsplit("/", 1)[1]).write_bytes(z.read(name))
    out = []
    for f in sorted(EXPT4_DIR.glob("*cycle_data.csv")):
        d = pd.read_csv(f)
        cap = d["Discharge Capacity [A h]"].to_numpy(float)
        x = np.arange(len(d), dtype=float)
        ok = np.isfinite(cap) & (cap > 0)
        x, cap = x[ok], cap[ok]
        out.append({"dataset": "Expt4", "cell_id": "E4_" + f.name.split(" - ")[1].replace("cell ", ""),
                    "temperature_c": float(d["Av. Discharge Temperature [°C]"].median()),
                    "x": x, "soh": cap / np.median(cap[:5])})
    return out


def score(test: list[dict], prior: LifePrior, fold: str) -> list[dict]:
    rows = []
    for t in test:
        n_eol = end_of_life(t)
        x0 = t["x"][0]
        if not np.isfinite(n_eol) or n_eol <= x0:
            continue
        for f in FRACTIONS:
            a = x0 + f * (n_eol - x0)
            true = n_eol - a
            arms = {"population": posterior_life(t["x"][:1], t["soh"][:1], a, prior, THRESHOLD),
                    "cell_only": posterior_life(t["x"], t["soh"], a, LifePrior.flat(), THRESHOLD),
                    "hybrid": posterior_life(t["x"], t["soh"], a, prior, THRESHOLD)}
            for arm, p in arms.items():
                rows.append({"fold": fold, "cell_id": t["cell_id"], "f": f, "arm": arm, "true_rul": true,
                             "pred": p.median, "lower": p.lower, "upper": p.upper})
            s = estimate_rul(t["x"], t["soh"], int(a), threshold=THRESHOLD)
            rows.append({"fold": fold, "cell_id": t["cell_id"], "f": f, "arm": "shipped", "true_rul": true,
                         "pred": s.rul_cycles, "lower": np.nan, "upper": np.nan})
    return rows


def prior_from(trajs: list[dict]) -> LifePrior:
    fits = [fit_life_curve(t["x"], t["soh"], THRESHOLD) for t in trajs]
    return fit_life_prior(np.array([f[0] for f in fits]), np.array([f[1] for f in fits]))


def summarise(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = r.copy()
    r["rel_error"] = (r["pred"] - r["true_rul"]).abs() / r["true_rul"]
    r["covered"] = (r["lower"] <= r["true_rul"]) & (r["true_rul"] <= r["upper"])
    err = (r.dropna(subset=["pred"]).groupby(["fold", "f", "arm"])["rel_error"].median()
           .unstack("arm").round(3))
    cov = (r[r["arm"] != "shipped"].groupby(["fold", "arm"])["covered"].mean().unstack("arm").round(2))
    return err, cov


def main() -> int:
    dev = [t for t in trajectories() if t["dataset"] in DEV]
    OUT.mkdir(parents=True, exist_ok=True)

    # A. development folds (decide nothing)
    dev_rows = []
    for held in DEV:
        prior = prior_from([t for t in dev if t["dataset"] != held])
        dev_rows += score([t for t in dev if t["dataset"] == held], prior, held)
    dev_r = pd.DataFrame(dev_rows)
    dev_r.round(5).to_csv(OUT / "development_points.csv", index=False)
    dev_err, dev_cov = summarise(dev_r)

    # B. the deciding test
    prior = prior_from(dev)
    test = expt4_trajectories()
    crossing = [t for t in test if np.isfinite(end_of_life(t))]
    test_r = pd.DataFrame(score(test, prior, "Expt4"))
    test_r.round(5).to_csv(OUT / "expt4_points.csv", index=False)
    pd.DataFrame([{"cell_id": t["cell_id"], "temperature_c": round(t["temperature_c"], 1),
                   "readings": len(t["x"]), "final_soh": round(float(t["soh"][-1]), 3),
                   "n_eol_0p90": end_of_life(t)} for t in test]).to_csv(OUT / "expt4_cells.csv", index=False)
    if len(crossing) < 3:
        verdict, lines_c = "UNDECIDED", [f"- only {len(crossing)} Expt 4 cells cross 0.90; not scored"]
        err, cov = pd.DataFrame(), pd.DataFrame()
    else:
        err, cov = summarise(test_r)
        e = err.xs("Expt4", level="fold")
        early = {f: bool(e.loc[f, "hybrid"] < e.loc[f, "cell_only"] and e.loc[f, "hybrid"] < e.loc[f, "population"])
                 for f in (0.10, 0.25)}
        rr = test_r[test_r["arm"] == "hybrid"]
        pooled = float(((rr["lower"] <= rr["true_rul"]) & (rr["true_rul"] <= rr["upper"])).mean())
        c1, c2 = all(early.values()), 0.65 <= pooled <= 0.95
        verdict = "SUCCESS" if c1 and c2 else "NOT MET"
        lines_c = [f"- Early skill: hybrid best at 10% {early[0.10]}, at 25% {early[0.25]}: "
                   f"{'met' if c1 else 'not met'}",
                   f"- Coverage: hybrid pooled {pooled:.2f} (needs 0.65-0.95): {'met' if c2 else 'not met'}"]

    lines = ["# Population prior for remaining life, v2", "",
             "Generated by `scripts/run_population_rul_study_v2.py`. v2 was designed after v1 failed; "
             "only the Expt 4 test, fixed before its data was downloaded, decides.", "",
             f"Prior from all development cells: median end of life {np.exp(prior.log_l_mean):.0f} cycles "
             f"(log sd {prior.log_l_sd:.2f}), shape p {prior.p_mean:.2f} (sd {prior.p_sd:.2f}), "
             f"{prior.n_cells} cells.", "",
             "## B. Deciding test: Imperial Expt 4 (drive-cycle ageing)", "",
             "```", pd.read_csv(OUT / "expt4_cells.csv").to_string(index=False), "```", "",
             "Median relative error:", "", "```", err.to_string() if not err.empty else "(not scored)", "```", "",
             "80% interval coverage:", "", "```", cov.to_string() if not cov.empty else "(not scored)", "```", "",
             *lines_c, "", f"**Verdict: {verdict}**", "",
             "## A. Development folds (leave-one-dataset-out; decide nothing)", "",
             "```", dev_err.to_string(), "```", "", "```", dev_cov.to_string(), "```"]
    (OUT / "population_rul_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
