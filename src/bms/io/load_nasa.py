"""
NASA battery dataset loader for Day 3.

This loader intentionally supports common CSV/XLSX exports first.
It normalizes column names into a consistent BMS telemetry shape.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader_common import add_basic_features, normalize_columns, numeric_cleanup, read_table


REQUIRED_AT_LEAST_ONE = ["voltage_v", "current_a", "temperature_c", "capacity_ah"]


def load_nasa_file(path: str | Path, sheet_name: str | int | None = None, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    df = read_table(path, sheet_name=sheet_name)
    if max_rows:
        df = df.head(max_rows)

    df = normalize_columns(df)
    df = numeric_cleanup(df, ["cycle", "voltage_v", "current_a", "temperature_c", "capacity_ah", "soc", "soh"])
    df = add_basic_features(df, source="nasa", file_path=path)

    available = [c for c in REQUIRED_AT_LEAST_ONE if c in df.columns and df[c].notna().any()]
    if not available:
        raise ValueError(
            f"NASA file loaded but no useful battery columns were detected. "
            f"Expected one of {REQUIRED_AT_LEAST_ONE}. File: {path}"
        )

    return df


def save_processed(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
