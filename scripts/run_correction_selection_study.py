"""Can BEACON choose its correction per cell, with no ground truth?

    python scripts/run_temperature_fix_study.py      # writes the rows this reads
    python scripts/run_correction_selection_study.py

Writes reports/metrics/temperature_fix/selection_report.md and selection.csv.

STATUS: EXPLORATORY. This rule was proposed AFTER run_temperature_fix_study.py
showed that the anchored correction helps some cohorts and hurts others. It is
evaluated on the same cells, so a good result here is a hypothesis for fresh
data, not a validated claim.

THE RULE, FIXED BEFORE THIS RAN
-------------------------------
At each reading k of a cell, for each arm (shipped, anchored) compute the
consistency uncertainty u over that arm's accepted readings up to k (the
definition in run_sufficiency_study.py: reference SE from the first 5,
trend-removed recent SE from the last 10, needs >= 9 readings). Report the
estimate (median of the last 5) of the arm with the lower u; if only one arm
has u, report that arm. Uses no truth, and only past readings.

Scored with the same per-cell MAE and the same criteria as the fix study:
NASA cold falls by at least one third vs shipped; CALCE +0.3, Oxford +0.5,
NASA warm +0.5 points tolerance. Oxford checks are 100 cycles apart, so u
there is computed on few readings - reported, not over-read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = Path("data/interim/temperature_fix_rows.parquet")
OUT = Path("reports/metrics/temperature_fix")
RECENT, REPORT, MIN_READINGS = 10, 5, 9
REGRESSION = {"CALCE": 0.003, "Oxford": 0.005, "NASA warm": 0.005}
ARMS = ("shipped", "anchored")


def _se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return 1.2533 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(len(values))


def estimate_and_u(soh: np.ndarray) -> tuple[float, float]:
    """(reported estimate, consistency uncertainty) from readings so far."""
    if len(soh) < MIN_READINGS:
        return (float(np.median(soh[-REPORT:])) if len(soh) > 5 else float("nan"), float("nan"))
    recent = soh[max(5, len(soh) - RECENT):]
    idx = np.arange(len(recent))
    res = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
    rse = 1.2533 * float(np.median(np.abs(res - np.median(res)))) / np.sqrt(REPORT)
    u = float(np.sqrt(np.nansum([_se(soh[:5]) ** 2, rse ** 2])))
    return float(np.median(soh[max(5, len(soh) - REPORT):])), u


def cell_errors(cell: pd.DataFrame) -> dict[str, float]:
    """Per-cell MAE for each arm reported alone, and for the selection rule."""
    truth = cell.groupby("cycle")["soh_true"].first()
    series = {a: cell[(cell["arm"] == a)].dropna(subset=["soh"]).sort_values("cycle") for a in ARMS}
    errs: dict[str, list[float]] = {a: [] for a in (*ARMS, "selected")}
    picks = []
    for cyc in sorted(truth.index):
        got = {}
        for a in ARMS:
            past = series[a][series[a]["cycle"] <= cyc]["soh"].to_numpy(float)
            if len(past) > 5:
                got[a] = estimate_and_u(past)
        if not got:
            continue
        for a, (est, _) in got.items():
            if np.isfinite(est):
                errs[a].append(abs(est - truth[cyc]))
        with_u = {a: v for a, v in got.items() if np.isfinite(v[1]) and np.isfinite(v[0])}
        pool = with_u or {a: v for a, v in got.items() if np.isfinite(v[0])}
        if not pool:
            continue
        pick = min(pool, key=lambda a: pool[a][1] if np.isfinite(pool[a][1]) else np.inf)
        picks.append(pick)
        errs["selected"].append(abs(pool[pick][0] - truth[cyc]))
    out = {f"mae_{a}": float(np.mean(e)) if e else np.nan for a, e in errs.items()}
    out["share_anchored"] = float(np.mean([p == "anchored" for p in picks])) if picks else np.nan
    return out


def main() -> int:
    rows = pd.read_parquet(ROWS).dropna(subset=["soh_true"])
    recs = []
    for (ds, group, cohort, cell), g in rows.groupby(["dataset", "group", "cohort", "cell_id"]):
        recs.append({"dataset": ds, "group": group, "cohort": cohort, "cell_id": cell, **cell_errors(g)})
    pc = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    pc.round(5).to_csv(OUT / "selection.csv", index=False)
    cols = ["mae_shipped", "mae_anchored", "mae_selected", "share_anchored"]
    summ = pc.groupby("group")[cols].median().round(4)
    summ.insert(0, "cells", pc.groupby("group")["mae_selected"].count())
    cohort = pc[pc["dataset"] == "NASA"].groupby("cohort")[cols].median().round(4)

    def m(g, c):
        return float(summ.loc[g, c])

    improved = m("NASA cold", "mae_selected") <= m("NASA cold", "mae_shipped") * 2 / 3
    regress = {g: m(g, "mae_selected") - m(g, "mae_shipped") for g in REGRESSION}
    ok = all(regress[g] <= t for g, t in REGRESSION.items())
    lines = ["# Choosing the correction per cell, without ground truth (EXPLORATORY)", "",
             "Generated by `scripts/run_correction_selection_study.py`. The rule was proposed "
             "after the fix study's results and is scored on the same cells: a hypothesis for "
             "fresh data, not a validated claim.", "",
             "Each MAE here is of the reported estimate (median of the last 5 accepted readings), "
             "so it differs from the per-reading MAE in temperature_fix_report.md.", "",
             "## Median per-cell MAE", "", "```", summ.to_string(), "```", "",
             "## NASA by cohort", "", "```", cohort.to_string(), "```", "",
             "## Criteria (same as the fix study)", "",
             f"- NASA cold: {m('NASA cold', 'mae_shipped'):.4f} -> {m('NASA cold', 'mae_selected'):.4f} "
             f"({'met' if improved else 'not met'})"]
    lines += [f"- {g} change {regress[g]:+.4f} (tolerance +{t:.3f}: {'ok' if regress[g] <= t else 'REGRESSION'})"
              for g, t in REGRESSION.items()]
    lines += ["", f"**Verdict: {'MET (exploratory)' if improved and ok else 'NOT MET'}**"]
    (OUT / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
