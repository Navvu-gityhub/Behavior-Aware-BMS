"""
Common helpers for Day 3 dataset loaders.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "date_time", "datetime", "test_time", "time_s", "time_sec", "time_seconds"],
    "cycle": ["cycle", "cycle_index", "cycle_number", "cycle_no", "cycleid"],
    "voltage_v": ["voltage", "voltage_v", "voltage_measured", "voltage_measured_v", "terminal_voltage", "ewe/v"],
    "current_a": ["current", "current_a", "current_measured", "current_measured_a", "current_load", "current_charge", "current_discharge", "i/a"],
    "temperature_c": ["temperature", "temperature_c", "temp", "temp_c", "temperature_measured", "temperature_measured_c"],
    "capacity_ah": ["capacity", "capacity_ah", "discharge_capacity", "charge_capacity", "qdischarge", "qcharge", "capacity_measured"],
    "soc": ["soc", "state_of_charge"],
    "soh": ["soh", "state_of_health"],
}


def clean_column_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-\(\)\[\]{}]+", "_", text)
    text = text.replace("/", "_per_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_column_name(c) for c in df.columns]

    rename_map = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for col in df.columns:
            if col == canonical or col in aliases:
                rename_map[col] = canonical
                break

    df = df.rename(columns=rename_map)

    # Keep original extra columns also; canonical columns are added if missing.
    for col in ["timestamp", "cycle", "voltage_v", "current_a", "temperature_c", "capacity_ah", "soc", "soh"]:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def read_table(path: str | Path, sheet_name: str | int | None = None) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        # If sheet_name is None, read the first sheet. This keeps smoke tests simple.
        return pd.read_excel(path, sheet_name=0 if sheet_name is None else sheet_name)
    if suffix == ".txt":
        # Battery datasets often use comma, tab, semicolon, or whitespace.
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def numeric_cleanup(df: pd.DataFrame, numeric_cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_basic_features(df: pd.DataFrame, source: str, file_path: str | Path) -> pd.DataFrame:
    df = df.copy()
    df["source"] = source
    df["source_file"] = str(file_path)

    if "voltage_v" in df.columns and "current_a" in df.columns:
        df["power_w"] = pd.to_numeric(df["voltage_v"], errors="coerce") * pd.to_numeric(df["current_a"], errors="coerce")
    else:
        df["power_w"] = pd.NA

    # Conservative charging flag: positive current often means charge, negative often discharge.
    if "current_a" in df.columns:
        current = pd.to_numeric(df["current_a"], errors="coerce")
        df["mode_guess"] = current.apply(lambda x: "charge" if pd.notna(x) and x > 0 else ("discharge" if pd.notna(x) and x < 0 else "unknown"))
    else:
        df["mode_guess"] = "unknown"

    return df
