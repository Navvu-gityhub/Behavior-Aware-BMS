"""Unified BMS schema utilities for Behavior-Aware BMS.

Day 4 deliverable:
- Defines canonical columns for battery telemetry.
- Converts common dataset column aliases into one unified schema.
- Validates required fields and basic value ranges.
- Adds derived columns used by later feature extraction modules.

This module intentionally uses only pandas/numpy-friendly logic so it can run
inside Google Colab without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BMSField:
    """Metadata for one canonical BMS field."""

    name: str
    dtype: str
    unit: str
    required: bool
    nullable: bool
    description: str


UNIFIED_SCHEMA: Tuple[BMSField, ...] = (
    BMSField("dataset", "string", "text", True, False, "Dataset source such as nasa, calce, stanford, simulated, obd."),
    BMSField("source_file", "string", "text", False, True, "Original file name for traceability."),
    BMSField("cell_id", "string", "text", True, False, "Cell/module/pack/vehicle identifier."),
    BMSField("timestamp", "datetime64[ns]", "ISO-8601", False, True, "Measurement timestamp."),
    BMSField("cycle", "Int64", "cycle number", False, True, "Cycle index or cycle number."),
    BMSField("voltage_v", "float64", "V", True, False, "Voltage in volts."),
    BMSField("current_a", "float64", "A", True, False, "Current in amperes after sign convention correction."),
    BMSField("temperature_c", "float64", "°C", True, False, "Temperature in Celsius."),
    BMSField("capacity_ah", "float64", "Ah", False, True, "Measured or available capacity."),
    BMSField("resistance_ohm", "float64", "Ω", False, True, "Internal resistance / DCIR when available."),
    BMSField("impedance_ohm", "float64", "Ω", False, True, "AC impedance or impedance indicator when available."),
    BMSField("soc", "float64", "%", False, True, "State of Charge in percent."),
    BMSField("soh", "float64", "%", False, True, "State of Health in percent."),
    BMSField("soc_band", "string", "category", False, True, "SOC range label derived from soc."),
    BMSField("mode_guess", "string", "category", False, True, "Charge/discharge/rest mode inferred from current."),
    BMSField("power_w", "float64", "W", False, True, "Power calculated as voltage_v * current_a."),
    BMSField("label", "string", "category", False, True, "Optional ML target label."),
    BMSField("notes", "string", "text", False, True, "Dataset-specific notes."),
)

CANONICAL_COLUMNS: Tuple[str, ...] = tuple(field.name for field in UNIFIED_SCHEMA)
REQUIRED_COLUMNS: Tuple[str, ...] = tuple(field.name for field in UNIFIED_SCHEMA if field.required)
ORDER_COLUMNS: Tuple[str, ...] = ("cycle", "timestamp")

# Common aliases seen in public battery datasets and exported BMS telemetry.
FIELD_ALIASES: Mapping[str, str] = {
    # identity / traceability
    "dataset_name": "dataset",
    "source": "dataset",
    "file": "source_file",
    "filename": "source_file",
    "sourcefile": "source_file",
    "battery_id": "cell_id",
    "batteryid": "cell_id",
    "cell": "cell_id",
    "cellid": "cell_id",
    "cell_no": "cell_id",
    "cell_number": "cell_id",
    "vehicle_id": "cell_id",
    # time / cycle
    "time": "timestamp",
    "date_time": "timestamp",
    "datetime": "timestamp",
    "test_time": "timestamp",
    "cycle_index": "cycle",
    "cycle_number": "cycle",
    "cycle_no": "cycle",
    "cycleid": "cycle",
    # voltage
    "voltage": "voltage_v",
    "voltage(v)": "voltage_v",
    "voltage_v": "voltage_v",
    "voltage_measured": "voltage_v",
    "terminal_voltage": "voltage_v",
    "cell_voltage": "voltage_v",
    # current
    "current": "current_a",
    "current(a)": "current_a",
    "current_a": "current_a",
    "current_measured": "current_a",
    "load_current": "current_a",
    "battery_current": "current_a",
    # temperature
    "temperature": "temperature_c",
    "temperature(c)": "temperature_c",
    "temperature_c": "temperature_c",
    "temp": "temperature_c",
    "temp_c": "temperature_c",
    "temperature_measured": "temperature_c",
    "ambient_temperature": "temperature_c",
    # capacity / resistance / impedance
    "capacity": "capacity_ah",
    "capacity(ah)": "capacity_ah",
    "capacity_ah": "capacity_ah",
    "discharge_capacity": "capacity_ah",
    "resistance": "resistance_ohm",
    "resistance_ohm": "resistance_ohm",
    "internal_resistance": "resistance_ohm",
    "dcir": "resistance_ohm",
    "impedance": "impedance_ohm",
    "impedance_ohm": "impedance_ohm",
    "eis": "impedance_ohm",
    # states / labels
    "soc_%": "soc",
    "state_of_charge": "soc",
    "soh_%": "soh",
    "state_of_health": "soh",
    "target": "label",
    "class": "label",
    "risk_label": "label",
}

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "voltage_v",
    "current_a",
    "temperature_c",
    "capacity_ah",
    "resistance_ohm",
    "impedance_ohm",
    "soc",
    "soh",
    "power_w",
)

NON_NEGATIVE_COLUMNS: Tuple[str, ...] = (
    "voltage_v",
    "capacity_ah",
    "resistance_ohm",
    "impedance_ohm",
)

PERCENT_COLUMNS: Tuple[str, ...] = ("soc", "soh")


def normalize_column_name(name: object) -> str:
    """Normalize a raw column name for alias matching."""
    text = str(name).strip().lower()
    text = text.replace(" ", "_").replace("-", "_").replace("/", "_")
    text = text.replace("__", "_")
    return text


def build_rename_map(columns: Iterable[object], extra_aliases: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return a rename map from raw columns to canonical columns."""
    aliases = dict(FIELD_ALIASES)
    if extra_aliases:
        aliases.update({normalize_column_name(k): v for k, v in extra_aliases.items()})

    rename_map: Dict[str, str] = {}
    for col in columns:
        normalized = normalize_column_name(col)
        canonical = aliases.get(normalized, normalized)
        if canonical in CANONICAL_COLUMNS:
            rename_map[str(col)] = canonical
    return rename_map


def standardize_columns(
    df: pd.DataFrame,
    *,
    dataset: Optional[str] = None,
    cell_id: Optional[str] = None,
    source_file: Optional[str] = None,
    extra_aliases: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Rename raw columns to the unified schema and inject metadata columns.

    Parameters
    ----------
    df:
        Raw dataframe from NASA/CALCE/Stanford/simulated BMS data.
    dataset, cell_id, source_file:
        Optional metadata values added when the raw file does not contain them.
    extra_aliases:
        Optional dataset-specific aliases, e.g. {"Voltage_measured": "voltage_v"}.
    """
    out = df.copy()
    out = out.rename(columns=build_rename_map(out.columns, extra_aliases=extra_aliases))

    if dataset is not None and "dataset" not in out.columns:
        out["dataset"] = dataset
    if cell_id is not None and "cell_id" not in out.columns:
        out["cell_id"] = cell_id
    if source_file is not None and "source_file" not in out.columns:
        out["source_file"] = source_file

    return out


def infer_soc_band(soc: object) -> str:
    """Convert SOC percentage into behavior-friendly SOC band."""
    if pd.isna(soc):
        return "unknown"
    value = float(soc)
    if value < 20:
        return "low_0_20"
    if value < 80:
        return "normal_20_80"
    if value < 90:
        return "elevated_80_90"
    return "high_90_100"


def infer_mode(current_a: object, threshold_a: float = 0.02) -> str:
    """Infer battery operating mode using current.

    The default assumes positive current means charging and negative current means
    discharging. Correct the dataset sign convention before calling this function.
    """
    if pd.isna(current_a):
        return "unknown"
    current = float(current_a)
    if current > threshold_a:
        return "charge"
    if current < -threshold_a:
        return "discharge"
    return "rest"


def coerce_bms_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce canonical fields to expected dtypes where possible."""
    out = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "cycle" in out.columns:
        out["cycle"] = pd.to_numeric(out["cycle"], errors="coerce").astype("Int64")

    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    for col in ("dataset", "source_file", "cell_id", "soc_band", "mode_guess", "label", "notes"):
        if col in out.columns:
            out[col] = out[col].astype("string")

    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived fields required by Day 4 and later preprocessing."""
    out = df.copy()

    if "voltage_v" in out.columns and "current_a" in out.columns:
        out["power_w"] = pd.to_numeric(out["voltage_v"], errors="coerce") * pd.to_numeric(out["current_a"], errors="coerce")

    if "soc" in out.columns:
        out["soc_band"] = out["soc"].apply(infer_soc_band)
    elif "soc_band" not in out.columns:
        out["soc_band"] = "unknown"

    if "current_a" in out.columns:
        out["mode_guess"] = out["current_a"].apply(infer_mode)
    elif "mode_guess" not in out.columns:
        out["mode_guess"] = "unknown"

    return out


def validate_bms_schema(df: pd.DataFrame, *, strict: bool = True) -> List[str]:
    """Validate a dataframe against the unified BMS schema.

    Returns a list of human-readable issues. An empty list means validation passed.
    If strict=True, missing recommended ordering information is treated as an error.
    """
    issues: List[str] = []
    columns = set(df.columns)

    for col in REQUIRED_COLUMNS:
        if col not in columns:
            issues.append(f"Missing required column: {col}")

    if not any(col in columns for col in ORDER_COLUMNS):
        issues.append("Missing ordering column: provide at least one of cycle or timestamp")

    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().all():
            issues.append(f"Required column is fully empty: {col}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if strict and numeric.isna().all():
                issues.append(f"Numeric column could not be parsed: {col}")

    for col in NON_NEGATIVE_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if (numeric.dropna() < 0).any():
                issues.append(f"Column contains negative values but should be non-negative: {col}")

    for col in PERCENT_COLUMNS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid = numeric.dropna()
            if ((valid < 0) | (valid > 100)).any():
                issues.append(f"Column contains values outside 0–100% range: {col}")

    return issues


def standardize_validate_bms_data(
    df: pd.DataFrame,
    *,
    dataset: Optional[str] = None,
    cell_id: Optional[str] = None,
    source_file: Optional[str] = None,
    extra_aliases: Optional[Mapping[str, str]] = None,
    strict: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """One-call helper for Day 4 preprocessing.

    Steps:
    1. Rename known raw columns to canonical names.
    2. Add metadata if missing.
    3. Coerce data types.
    4. Add derived fields: power_w, soc_band, mode_guess.
    5. Validate required columns and basic ranges.
    """
    out = standardize_columns(
        df,
        dataset=dataset,
        cell_id=cell_id,
        source_file=source_file,
        extra_aliases=extra_aliases,
    )
    out = coerce_bms_dataframe(out)
    out = add_derived_features(out)
    issues = validate_bms_schema(out, strict=strict)
    return out, issues


def schema_as_dataframe() -> pd.DataFrame:
    """Return the schema metadata as a dataframe."""
    return pd.DataFrame([asdict(field) for field in UNIFIED_SCHEMA])


def save_schema_csv(path: str | Path) -> Path:
    """Save schema metadata to CSV for documentation or dashboard use."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_as_dataframe().to_csv(output_path, index=False)
    return output_path
