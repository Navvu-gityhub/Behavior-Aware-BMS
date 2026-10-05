"""Empirical error bands for the shipped SOH and RUL estimators.

    python scripts/build_error_bands.py

Writes `reports/metrics/error_bands.csv`. `src/bms/health/error_bands.py`
carries the same values as constants, and `tests/test_error_bands.py` fails if
the two disagree - so a re-run that moves a band forces the code to follow.

SOH: the median, across measured CALCE cells, of each cell's 90th-percentile
absolute error. Fixed window from `calce_field_soh/` (step-resistance arm,
primary window); learned window from the two top-of-charge partial-cycling
cells in `calce_partial_soh/`.

RUL: indexed by the PREDICTED remaining life, because that is all a deployed
system knows. (The headline "73% within +/-20" is conditioned on the TRUE
remaining life being under 25 cycles, which a user cannot observe.) For each
predicted-RUL bin: the 5th, 50th and 95th percentiles of (true - predicted),
and how often the cell lasted at least as long as predicted.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

METRICS = Path("reports/metrics")
RUL_BINS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 100_000))


def main() -> int:
    rows = []
    field = pd.read_csv(METRICS / "calce_field_soh" / "field_soh_per_cell.csv")
    fixed = field[(field["primary"] == True) & (field["arm"] == "step_r")  # noqa: E712
                  & ~field["refused"].astype(bool) & (field["n_scored"] >= 10)]
    rows.append({"estimator": "soh_fixed_window", "bin": "all", "n": len(fixed),
                 "cells": fixed["cell_id"].nunique(),
                 "median_abs": round(float(fixed["mae"].median()), 4),
                 "p90_abs": round(float(fixed["p90_abs_error"].median()), 4)})

    trace = pd.read_csv(METRICS / "calce_partial_soh" / "partial_soh_trace.csv")
    learned = trace[trace["cell_id"].isin(["CS2_24", "CS2_25"])]
    rows.append({"estimator": "soh_learned_window", "bin": "all", "n": len(learned),
                 "cells": learned["cell_id"].nunique(),
                 "median_abs": round(float(learned["error"].abs().median()), 4),
                 "p90_abs": round(float(learned["error"].abs().quantile(0.9)), 4)})

    rul = pd.read_csv(METRICS / "calce_rul_horizon_early_ref" / "rul_folds.csv")
    rul = rul[rul["threshold"] == 0.9].dropna(subset=["rul_pred", "rul_true"])
    for lo, hi in RUL_BINS:
        b = rul[(rul["rul_pred"] >= lo) & (rul["rul_pred"] < hi)]
        later = b["rul_true"] - b["rul_pred"]
        rows.append({"estimator": "rul_fade_extrapolation", "bin": f"{lo}-{hi}",
                     "n": len(b), "cells": b["cell_id"].nunique(),
                     "true_minus_pred_q05": round(float(later.quantile(0.05)), 0),
                     "true_minus_pred_q50": round(float(later.quantile(0.50)), 0),
                     "true_minus_pred_q95": round(float(later.quantile(0.95)), 0),
                     "share_lasted_at_least_predicted": round(float((later >= 0).mean()), 2)})
    table = pd.DataFrame(rows)
    table.to_csv(METRICS / "error_bands.csv", index=False)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
