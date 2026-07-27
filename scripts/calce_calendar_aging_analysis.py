"""Calendar-aging analysis: does storage SOC/temperature/duration predict capacity loss?

Distinct from `calibrate_against_nasa.py` / `calibrate_cohort_cycle_level.py`,
which validate the *cycle-aging* pipeline (behavior during active use). This
answers a different, legitimate question the main pipeline doesn't model at
all: for a cell sitting in storage (not being cycled), does the SOC and
temperature it's held at, and how long, predict how much capacity it loses?

Data: CALCE PLN pouch cells.
  - Initial (pre-storage) capacity: src/bms/io/load_calce_capacity.py,
    parsed from the raw Arbin "Capacity Characterization_Initialization" export.
  - Storage condition (SOC, temperature, duration) and post-storage
    capacity: docs/calce_dataset_note.md's PLN_Number_SOC_Temp_StoragePeriod.xlsx.

This is NOT fed into the main pipeline's health_index/risk_score/RUL —
those model cycle aging, not shelf aging, and conflating the two would be
scientifically wrong, not just messy. Reported here as a standalone
finding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bms.io.load_calce_capacity import extract_initial_capacity_table

_DURATION_TO_DAYS = {"3W": 21, "3M": 90, "6M": 180}


def load_storage_conditions(pln_xlsx_path: str | Path) -> pd.DataFrame:
    import openpyxl
    wb = openpyxl.load_workbook(pln_xlsx_path, read_only=True, data_only=True)
    ws = wb["Sheet1"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    df = df.loc[:, df.columns.notna()]  # drop trailing unnamed/blank export columns
    df = df.rename(columns={
        "PLN": "pln_id",
        "SOC": "storage_soc",
        "TEMP": "storage_temp_c",
        "Time": "storage_duration_label",
        "Discharge Capacity": "post_storage_capacity_ah",
    })
    df = df[df["storage_soc"] != "NA"]
    df = df.dropna(subset=["storage_soc", "storage_temp_c", "storage_duration_label", "post_storage_capacity_ah"])

    df["storage_soc"] = pd.to_numeric(df["storage_soc"], errors="coerce")
    df["storage_temp_c"] = pd.to_numeric(df["storage_temp_c"], errors="coerce")
    df["post_storage_capacity_ah"] = pd.to_numeric(df["post_storage_capacity_ah"], errors="coerce")
    df["storage_duration_days"] = df["storage_duration_label"].map(_DURATION_TO_DAYS)
    df["pln_id"] = pd.to_numeric(df["pln_id"], errors="coerce")

    return df.dropna(subset=["storage_soc", "storage_temp_c", "post_storage_capacity_ah", "storage_duration_days", "pln_id"])[
        ["pln_id", "storage_soc", "storage_temp_c", "storage_duration_label", "storage_duration_days", "post_storage_capacity_ah"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacity-char-dir", type=str, required=True)
    parser.add_argument("--pln-storage-xlsx", type=str, required=True)
    parser.add_argument("--out", type=str, default="reports/metrics/calce_calendar_aging_results.csv")
    args = parser.parse_args()

    initial = extract_initial_capacity_table(args.capacity_char_dir)
    storage = load_storage_conditions(args.pln_storage_xlsx)

    merged = storage.merge(initial, on="pln_id", how="inner")
    print(f"Initial-capacity cells: {len(initial)} | Storage-condition cells (quality-filtered): {len(storage)} | Matched: {len(merged)}")

    merged["capacity_loss_pct"] = 100 * (1 - merged["post_storage_capacity_ah"] / merged["initial_capacity_ah"])
    merged["storage_soc_x_temp"] = merged["storage_soc"] * merged["storage_temp_c"]  # candidate interaction term

    print("\nCapacity loss (%) by storage duration:")
    print(merged.groupby("storage_duration_label")["capacity_loss_pct"].agg(["count", "mean", "std"]))

    print("\nSpearman correlations with capacity_loss_pct (pooled across all durations):")
    results = []
    for feat in ["storage_soc", "storage_temp_c", "storage_duration_days", "storage_soc_x_temp"]:
        rho, p = stats.spearmanr(merged[feat], merged["capacity_loss_pct"])
        results.append({"feature": feat, "n": len(merged), "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    print("\nWithin each storage duration separately (controls for duration as a confound):")
    within_rows = []
    for duration, g in merged.groupby("storage_duration_label"):
        if len(g) < 10:
            continue
        for feat in ["storage_soc", "storage_temp_c"]:
            rho, p = stats.spearmanr(g[feat], g["capacity_loss_pct"])
            within_rows.append({"storage_duration": duration, "feature": feat, "n": len(g), "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
    within_df = pd.DataFrame(within_rows)
    print(within_df.to_string(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(Path(args.out).with_name("calce_calendar_aging_merged.csv"), index=False)
    pd.concat([results_df.assign(scope="pooled"), within_df.rename(columns={"storage_duration": "scope"})], ignore_index=True).to_csv(args.out, index=False)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
