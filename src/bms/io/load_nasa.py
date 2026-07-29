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


# ---------------------------------------------------------------------------
# Multi-file "cleaned_dataset" format: metadata.csv + data/<uid>.csv per test
# ---------------------------------------------------------------------------
#
# This is a different, more complete NASA distribution than load_nasa_file()
# above was written for: one CSV per charge/discharge/impedance *test*
# (Voltage_measured, Current_measured, Temperature_measured, Time), linked
# by a metadata.csv that records battery_id, test type, and — critically —
# the measured end-of-discharge Capacity for each discharge test. That
# Capacity column is real ground truth for capacity-fade calibration and is
# not available from load_nasa_file() on a single CSV in isolation.
#
# Current sign convention confirmed against sample files: negative = discharge,
# positive = charge — consistent with the rest of this codebase.

def load_nasa_dataset(base_dir: str | Path, include_charge: bool = True) -> pd.DataFrame:
    """Load the full metadata-linked NASA dataset into one long-format telemetry table.

    Returns columns matching the unified schema (dataset, cell_id, cycle,
    voltage_v, current_a, temperature_c, soc, capacity_ah) plus
    `test_type` (charge/discharge) and `ambient_temperature_c`.

    `cycle` is assigned as the 1-based rank of each battery's discharge
    tests in chronological (uid) order — this dataset's tests aren't
    pre-numbered as cycles. Charge tests are stamped with the cycle number
    of the discharge test that immediately follows them, since a
    charge-then-discharge pair is what this dataset treats as one usage
    cycle. `capacity_ah` is only populated on discharge rows (it's a
    per-test measurement, not a per-row one) — it is NOT interpolated
    across the cycle, precisely so it can't be mistaken for something it
    isn't.

    SOC is not measured in this dataset. It's derived here by integrating
    the row's own current over time and normalizing by that test's known
    Capacity (discharge) or a configurable nominal rated capacity (charge,
    since no per-charge-test ground truth capacity exists) — this is a
    physically-motivated estimate, not a measurement. Treat `soc` from this
    loader as approximate.
    """
    base_dir = Path(base_dir)
    meta = pd.read_csv(base_dir / "metadata.csv")
    meta["Capacity"] = pd.to_numeric(meta["Capacity"], errors="coerce")
    meta = meta[meta["type"].isin(["charge", "discharge"])].copy()
    meta = meta.sort_values(["battery_id", "uid"]).reset_index(drop=True)

    # Assign cycle numbers from discharge-test order per battery.
    disch_mask = meta["type"] == "discharge"
    meta["cycle"] = pd.NA
    meta.loc[disch_mask, "cycle"] = meta[disch_mask].groupby("battery_id").cumcount() + 1
    # Back-fill charge tests with the cycle number of the next discharge test
    # (a charge immediately precedes the discharge that completes its cycle).
    meta["cycle"] = meta.groupby("battery_id")["cycle"].transform(lambda s: s.bfill())
    meta = meta.dropna(subset=["cycle"])
    meta["cycle"] = meta["cycle"].astype(int)

    if not include_charge:
        meta = meta[meta["type"] == "discharge"]

    frames = []
    for row in meta.itertuples():
        file_path = base_dir / "data" / row.filename
        if not file_path.exists():
            continue
        raw = pd.read_csv(file_path)
        if raw.empty or "Current_measured" not in raw.columns:
            continue

        raw = raw.rename(columns={
            "Voltage_measured": "voltage_v",
            "Current_measured": "current_a",
            "Temperature_measured": "temperature_c",
            "Time": "time_s",
        })

        dt = raw["time_s"].diff().fillna(0).clip(lower=0)
        discharged_ah = (raw["current_a"].abs() * dt / 3600.0).cumsum()

        if row.type == "discharge" and pd.notna(row.Capacity) and row.Capacity > 0:
            raw["capacity_ah"] = pd.NA
            raw.loc[raw.index[-1], "capacity_ah"] = row.Capacity  # end-of-test measurement only
            raw["soc"] = (100.0 * (1 - discharged_ah / row.Capacity)).clip(0, 100)
        else:
            nominal_capacity_ah = 2.0  # unmeasured assumption for charge-test SOC only; see docstring
            raw["capacity_ah"] = pd.NA
            raw["soc"] = (100.0 * (discharged_ah / nominal_capacity_ah)).clip(0, 100)

        raw["dataset"] = "nasa"
        raw["cell_id"] = row.battery_id
        raw["cycle"] = row.cycle
        raw["test_type"] = row.type
        raw["ambient_temperature_c"] = row.ambient_temperature
        frames.append(raw)

    if not frames:
        raise ValueError(f"No usable NASA telemetry files found under {base_dir}")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["cell_id", "cycle"]).reset_index(drop=True)
    return out
