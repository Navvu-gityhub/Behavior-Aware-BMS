
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path.cwd()

INPUT_FILE = (
    ROOT /
    "data/processed/calce/calce_sample_processed.csv"
)

OUT_DIR = (
    ROOT /
    "data/processed/calce"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("Loading CALCE sample...")

df = pd.read_csv(INPUT_FILE)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# -----------------------------
# Standardize column names
# -----------------------------

df.columns = [
    str(c).strip().lower().replace(" ", "_")
    for c in df.columns
]

# -----------------------------
# Create derived features
# -----------------------------

if (
    "voltage_v" in df.columns
    and
    "current_a" in df.columns
):

    df["estimated_power_w"] = (
        df["voltage_v"] *
        df["current_a"]
    )

if "capacity_ah" in df.columns:

    initial_capacity = (
        df["capacity_ah"].max()
    )

    if initial_capacity > 0:

        df["capacity_retention_percent"] = (
            df["capacity_ah"] /
            initial_capacity
        ) * 100

# -----------------------------
# Missing value report
# -----------------------------

missing = pd.DataFrame({
    "column": df.columns,
    "missing_percent":
        df.isna().mean() * 100
})

missing.to_csv(
    OUT_DIR /
    "calce_missingness.csv",
    index=False
)

# -----------------------------
# Save analysis-ready table
# -----------------------------

output_file = (
    OUT_DIR /
    "calce_analysis_ready.csv"
)

df.to_csv(
    output_file,
    index=False
)

print("\nSaved:", output_file)
print("CALCE preprocessing complete")
