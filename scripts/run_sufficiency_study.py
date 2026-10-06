"""How much evidence does capacity health need? Measured on CALCE, not assumed.

    python scripts/run_sufficiency_study.py

Reads the telemetry cache built by run_field_soh_study.py and writes
`reports/metrics/calce_sufficiency/`.

THREE QUESTIONS, FIXED BEFORE THIS RAN
--------------------------------------
1. Does error fall with the number of usable discharges? The card's
   "HIGH confidence from 30" was a convention; this measures it.
2. Does the size of the beginning-of-life reference matter? The estimator
   uses the median of the first 5; this compares 1, 3, 5 and 10.
3. Does a consistency-based standard error predict the real error?

   Not "has the estimate stopped changing" - a healthy estimate keeps moving
   because the cell keeps ageing. The question is whether the readings agree
   with each other once that trend is removed:

     reference SE  = 1.2533 * MAD(first R readings) / sqrt(R) / reference
     recent SE     = 1.2533 * MAD(residuals of the last 10 readings around
                     their own straight line) / sqrt(5)
     uncertainty u = sqrt(reference SE^2 + recent SE^2)

   (1.2533 * MAD / sqrt(n) is the standard error of a median under noise.)
   Tolerance fixed at u <= 0.01 (one SOH point): below the 1.7% validated
   median error, so measurement scatter is not the dominant error term.

Scored at every lab capacity check: the estimate is the median of the last 5
accepted readings at or before it (as the pipeline reports); truth is lab
capacity over the checks in the first 10 cycles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import field_soh_table, learned_field_soh  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

CACHE = Path("data/interim/calce_telemetry")
TRUTH = Path("reports/metrics/calce_full_discharge.csv")
OUT = Path("reports/metrics/calce_sufficiency")
REST = 0.02
RATED = {"CS2": 1.1, "CX2": 1.35}
TOLERANCE = 0.01
RECENT, REPORT = 10, 5
N_BUCKETS = ((6, 10), (11, 20), (21, 30), (31, 60), (61, 100_000))
REFERENCE_SIZES = (1, 3, 5, 10)


def _se_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    mad = np.median(np.abs(values - np.median(values)))
    return 1.2533 * mad / np.sqrt(len(values))


def readings(cell: str) -> pd.DataFrame | None:
    tel = pd.read_parquet(CACHE / f"{cell}.parquet")
    d = cycles_to_frame(measure_cycles(tel, cell, rest_threshold_a=REST), complete_only=False)
    t = tel["test_time_s"].to_numpy(float)
    arbin = tel["cycle"].to_numpy(float)
    d["arbin_cycle"] = [arbin[min(np.searchsorted(t, s), len(t) - 1)] for s in d["start_time_s"]]
    table = field_soh_table(tel, d, cell, REST)
    if table.empty or table["cell_refused"].any() or table["soh_window_accepted"].notna().sum() < 30:
        _, table = learned_field_soh(tel, d, cell, REST, RATED[cell[:3]])
    if table.empty or table["cell_refused"].any():
        return None
    acc = table.dropna(subset=["window_charge_ah"]).sort_values("cycle")
    acc = acc[~(acc["implausible"] | (acc["soh_window"] > 1.10))]
    acc = acc.merge(d[["cycle", "arbin_cycle"]], on="cycle")
    return acc[["cycle", "arbin_cycle", "window_charge_ah"]].reset_index(drop=True)


def score_cell(cell: str, acc: pd.DataFrame, lab: pd.DataFrame, ref_size: int) -> list[dict]:
    q = acc["window_charge_ah"].to_numpy(float)
    x = acc["arbin_cycle"].to_numpy(float)
    if len(q) <= ref_size:
        return []
    ref = float(np.median(q[:ref_size]))
    soh = q / ref
    ref_se = _se_median(q[:ref_size]) / ref if ref_size > 1 else float("nan")
    early = lab[lab["arbin_cycle_index"] <= 10]
    lref = (early if not early.empty else lab.head(1))["capacity_ah"].median()
    rows = []
    for check in lab.itertuples():
        k = int(np.searchsorted(x, check.arbin_cycle_index, side="right"))   # readings <= check
        if k < ref_size + 1 or check.arbin_cycle_index - x[k - 1] > 10:
            continue
        est = float(np.median(soh[max(ref_size, k - REPORT):k]))
        recent = soh[max(ref_size, k - RECENT):k]
        if len(recent) >= 4:
            idx = np.arange(len(recent))
            fit = np.polyval(np.polyfit(idx, recent, 1), idx)
            mad = np.median(np.abs(recent - fit - np.median(recent - fit)))
            recent_se = 1.2533 * mad / np.sqrt(REPORT)
        else:
            recent_se = float("nan")
        u = float(np.sqrt(np.nansum([ref_se ** 2, recent_se ** 2]))) if np.isfinite(recent_se) else float("nan")
        rows.append({"cell_id": cell, "ref_size": ref_size, "n_readings": k - ref_size,
                     "estimate": est, "truth": check.capacity_ah / lref,
                     "abs_error": abs(est - check.capacity_ah / lref),
                     "reference_se": ref_se, "recent_se": recent_se, "uncertainty": u})
    return rows


def main() -> int:
    truth = add_targets(pd.read_csv(TRUTH))
    truth = truth[truth["soh"].notna()]
    rows = []
    for path in sorted(CACHE.glob("*.parquet")):
        cell = path.stem
        acc = readings(cell)
        lab = truth[truth["cell_id"] == cell].sort_values("arbin_cycle_index")
        if acc is None or lab.empty:
            continue
        for r in REFERENCE_SIZES:
            rows.extend(score_cell(cell, acc, lab, r))
        print(cell, len(acc), flush=True)
    est = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    # Points for the shipped reference size only, rounded: the other sizes are
    # summarised in the report, and the full table would be ~6 MB in git.
    est[est["ref_size"] == 5].round(5).to_csv(OUT / "sufficiency_points.csv", index=False)

    main5 = est[est["ref_size"] == 5]

    def per_cell_median(frame: pd.DataFrame) -> float:
        return float(frame.groupby("cell_id")["abs_error"].median().median())

    by_n = []
    for lo, hi in N_BUCKETS:
        b = main5[(main5["n_readings"] >= lo) & (main5["n_readings"] <= hi)]
        if len(b):
            by_n.append({"usable_readings": f"{lo}-{hi}" if hi < 100_000 else f"{lo}+",
                         "points": len(b), "cells": b["cell_id"].nunique(),
                         "median_abs_error": round(per_cell_median(b), 4),
                         "p90_abs_error": round(float(b["abs_error"].quantile(0.9)), 4)})
    by_ref = [{"reference_size": r, "cells": g["cell_id"].nunique(),
               "median_abs_error": round(per_cell_median(g), 4),
               "p90_abs_error": round(float(g["abs_error"].quantile(0.9)), 4)}
              for r, g in est.groupby("ref_size")]
    u = main5.dropna(subset=["uncertainty"])
    rho = float(spearmanr(u["uncertainty"], u["abs_error"]).statistic) if len(u) > 10 else float("nan")
    cell_rho = (u.groupby("cell_id").apply(
        lambda g: spearmanr(g["uncertainty"], g["abs_error"]).statistic if len(g) > 10 else np.nan)
        .dropna())
    within = u[u["uncertainty"] <= TOLERANCE]
    beyond = u[u["uncertainty"] > TOLERANCE]
    gate = pd.DataFrame([
        {"side": f"u <= {TOLERANCE}", "points": len(within), "cells": within["cell_id"].nunique(),
         "median_abs_error": round(float(within["abs_error"].median()), 4) if len(within) else np.nan,
         "p90_abs_error": round(float(within["abs_error"].quantile(0.9)), 4) if len(within) else np.nan},
        {"side": f"u > {TOLERANCE}", "points": len(beyond), "cells": beyond["cell_id"].nunique(),
         "median_abs_error": round(float(beyond["abs_error"].median()), 4) if len(beyond) else np.nan,
         "p90_abs_error": round(float(beyond["abs_error"].quantile(0.9)), 4) if len(beyond) else np.nan},
    ])
    lines = [
        "# How much evidence capacity health needs", "",
        "Generated by `scripts/run_sufficiency_study.py`; questions, uncertainty "
        "definition and tolerance were fixed before it ran (see its docstring).", "",
        "## 1. Error against the number of usable readings (reference of 5)", "",
        "```", pd.DataFrame(by_n).to_string(index=False), "```", "",
        "## 2. Error against reference size", "",
        "```", pd.DataFrame(by_ref).to_string(index=False), "```", "",
        "## 3. Does the consistency-based uncertainty predict the real error?", "",
        f"Spearman(uncertainty, |error|), pooled: {rho:.2f}; per cell, median "
        f"{cell_rho.median():.2f} over {len(cell_rho)} cells, positive in "
        f"{int((cell_rho > 0).sum())}.", "",
        "```", gate.to_string(index=False), "```",
    ]
    (OUT / "sufficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
