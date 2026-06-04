#!/usr/bin/env python3
"""
Small Day 3 smoke test.

It creates one fake NASA CSV and one fake CALCE XLSX in data/raw,
zips them, runs archive discovery, then loads one sample from each source.

Run:
    python tests/smoke_day3.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> int:
    raw = ROOT / "data/raw"
    interim = ROOT / "data/interim"
    processed = ROOT / "data/processed"

    for folder in [raw, interim, processed]:
        if folder.exists():
            shutil.rmtree(folder)

    (raw / "nasa").mkdir(parents=True, exist_ok=True)
    (raw / "calce").mkdir(parents=True, exist_ok=True)

    nasa_csv = raw / "nasa" / "nasa_sample.csv"
    pd.DataFrame(
        {
            "cycle": [1, 1, 2],
            "voltage_measured": [4.10, 4.05, 3.98],
            "current_measured": [-1.0, -1.1, -0.9],
            "temperature_measured": [28.5, 29.0, 29.5],
            "capacity": [2.00, 1.99, 1.98],
        }
    ).to_csv(nasa_csv, index=False)

    calce_xlsx = raw / "calce" / "calce_sample.xlsx"
    pd.DataFrame(
        {
            "Cycle_Index": [1, 1, 2],
            "Voltage(V)": [4.15, 4.08, 3.95],
            "Current(A)": [-0.8, -0.85, -0.75],
            "Temp_C": [27.0, 27.5, 28.1],
            "Discharge_Capacity": [1.10, 1.09, 1.08],
        }
    ).to_excel(calce_xlsx, index=False)

    with zipfile.ZipFile(raw / "nasa_sample.zip", "w") as zf:
        zf.write(nasa_csv, arcname="nasa/nasa_sample.csv")
    with zipfile.ZipFile(raw / "calce_sample.zip", "w") as zf:
        zf.write(calce_xlsx, arcname="calce/calce_sample.xlsx")

    run([sys.executable, "scripts/unpack_archives.py", "--raw", "data/raw", "--out", "data/interim", "--copy-loose-files"])
    run([sys.executable, "scripts/load_nasa.py", "--input", "data/raw/nasa/nasa_sample.csv", "--out", "data/processed/nasa/nasa_sample_processed.csv", "--max-rows", "10"])
    run([sys.executable, "scripts/load_calce.py", "--input", "data/raw/calce/calce_sample.xlsx", "--out", "data/processed/calce/calce_sample_processed.csv", "--max-rows", "10"])

    print("\nDAY 3 SMOKE TEST PASSED ✅")
    print("Manifest:", ROOT / "data/interim/discovered_files.csv")
    print("NASA processed:", ROOT / "data/processed/nasa/nasa_sample_processed.csv")
    print("CALCE processed:", ROOT / "data/processed/calce/calce_sample_processed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
