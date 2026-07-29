"""Calibration Stage 1+2: cohort-controlled, cycle-level fade correlation.

Addresses two gaps in `calibrate_against_nasa.py`:

1. That script pooled all 34 batteries despite them spanning 9 different
   NASA sub-experiments (different discharge cutoff voltage, current,
   ambient temperature — see docs/calibration_report.md Result 2). Here,
   batteries are grouped into the cohorts NASA's own documentation defines,
   and correlations are computed within-cohort.

2. That script compared ONE point per battery (whole-life average stress vs
   whole-life fade rate). Here, each battery contributes one point PER
   CYCLE: a trailing window of behavior (avg_stress, avg_temp, etc. over
   the preceding N cycles) against that cycle's actual capacity loss from
   the previous cycle. This uses the trajectory instead of collapsing it.

Statistical caveat, stated up front rather than glossed over: pooling
cycles from the same battery is NOT statistically independent (cycles
within a battery are autocorrelated). The pooled-cycle p-values below are
therefore optimistic / anti-conservative and should be read as exploratory,
not confirmatory. As a partial check, per-battery correlations are also
computed and their signs compared within each cohort — consistent signs
across independent batteries in a cohort is weaker but more trustworthy
evidence than one pooled p-value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.features.behavior_features import compute_behavior_flags
from src.bms.features.cycle_features import summarize_by_cycle
from src.bms.risk.stress_score import compute_stress_score

# Cohorts as defined by NASA's own per-batch READMEs
# (data/raw/nasa/.../extra_infos/README_*.txt), not inferred.
COHORTS: dict[str, list[str]] = {
    "RT_2A_CC_variedcutoff": ["B0005", "B0006", "B0007", "B0018"],
    "RT_SQWAVE_4A_variedcutoff": ["B0025", "B0026", "B0027", "B0028"],
    "ELEV43C_4A_CC_variedcutoff": ["B0029", "B0030", "B0031", "B0032"],
    "RT_CC_mixed_current": ["B0033", "B0034", "B0036"],
    "MIXED_24_44C_multiload": ["B0038", "B0039", "B0040"],  # heterogeneous even within group; low confidence
    "COLD4C_multiload": ["B0041", "B0042", "B0043", "B0044"],
    "COLD4C_1A": ["B0045", "B0046", "B0047", "B0048"],
    "COLD4C_2A_flagged": ["B0049", "B0050", "B0051", "B0052"],  # NASA: "experiment control software crashed"
    "COLD4C_2A": ["B0053", "B0054", "B0055", "B0056"],  # NASA: "several low capacity, not fully analyzed"
}

TRAILING_WINDOW = 5


def build_cycle_level_table(telemetry: pd.DataFrame) -> pd.DataFrame:
    flagged = compute_behavior_flags(telemetry)
    flagged["stress_score"] = compute_stress_score(flagged)
    cyc = summarize_by_cycle(flagged)

    # Ground-truth capacity per cycle (present on discharge rows only).
    cap = telemetry.dropna(subset=["capacity_ah"])[["cell_id", "cycle", "capacity_ah"]].drop_duplicates()
    cap["capacity_ah"] = cap["capacity_ah"].astype(float)

    cyc = cyc.merge(cap, on=["cell_id", "cycle"], how="inner")
    cyc = cyc.sort_values(["cell_id", "cycle"]).reset_index(drop=True)

    # Per-cycle capacity loss from the previous cycle (positive = more fade).
    cyc["capacity_loss"] = cyc.groupby("cell_id")["capacity_ah"].diff().mul(-1)

    # Trailing window average of behavior, computed from the PRIOR cycles
    # only (shift by 1 so the window doesn't include the cycle whose loss
    # we're trying to explain — otherwise this is circular).
    for col in ["avg_stress", "avg_temp", "deep_discharge_duration", "aggressive_discharge_count"]:
        cyc[f"trailing_{col}"] = (
            cyc.groupby("cell_id")[col]
            .transform(lambda s: s.shift(1).rolling(TRAILING_WINDOW, min_periods=2).mean())
        )

    return cyc.dropna(subset=["capacity_loss", "trailing_avg_stress"])


def analyze_cohort(cyc: pd.DataFrame, battery_ids: list[str]) -> dict:
    sub = cyc[cyc.cell_id.isin(battery_ids)]
    present = sub.cell_id.unique().tolist()
    result = {"n_batteries_present": len(present), "n_cycle_obs": len(sub)}

    features = ["trailing_avg_stress", "trailing_avg_temp", "trailing_deep_discharge_duration", "trailing_aggressive_discharge_count"]
    for feat in features:
        valid = sub[[feat, "capacity_loss"]].dropna()
        if len(valid) >= 10:
            rho, p = stats.spearmanr(valid[feat], valid["capacity_loss"])
            result[f"{feat}__pooled_rho"] = round(rho, 3)
            result[f"{feat}__pooled_p"] = round(p, 4)
        else:
            result[f"{feat}__pooled_rho"] = np.nan
            result[f"{feat}__pooled_p"] = np.nan

        # Per-battery signs, as a pseudoreplication check.
        signs = []
        for bid in present:
            b = sub[sub.cell_id == bid][[feat, "capacity_loss"]].dropna()
            if len(b) >= 10:
                r, _ = stats.spearmanr(b[feat], b["capacity_loss"])
                if not np.isnan(r):
                    signs.append(np.sign(r))
        result[f"{feat}__n_batteries_testable"] = len(signs)
        result[f"{feat}__n_positive_sign"] = sum(1 for s in signs if s > 0)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-parquet", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/metrics/cohort_cycle_level_results.csv")
    args = parser.parse_args()

    telemetry = pd.read_parquet(args.cache_parquet)
    print(f"Loaded {len(telemetry)} rows, {telemetry.cell_id.nunique()} batteries")

    cyc = build_cycle_level_table(telemetry)
    print(f"Cycle-level table: {len(cyc)} (battery, cycle) observations with a trailing window available")

    rows = []
    for cohort_name, battery_ids in COHORTS.items():
        res = analyze_cohort(cyc, battery_ids)
        res["cohort"] = cohort_name
        res["battery_ids"] = ",".join(battery_ids)
        rows.append(res)

    out_df = pd.DataFrame(rows)
    cols = ["cohort", "battery_ids", "n_batteries_present", "n_cycle_obs"] + [c for c in out_df.columns if c not in ("cohort", "battery_ids", "n_batteries_present", "n_cycle_obs")]
    out_df = out_df[cols]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print()
    print(out_df.to_string(index=False))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
