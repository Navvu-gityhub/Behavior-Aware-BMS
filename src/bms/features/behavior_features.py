"""Behavior feature extraction: row-level flags, rolling stats, per-battery summary.

Ported from `notebooks/04_feature_refinement.ipynb`. One correctness bug from
the notebook is fixed here: `battery_age_factor` was computed as
`cycle / df["cycle"].max()` using the maximum cycle across the *entire
dataset* (all batteries pooled), which understates age for any battery that
doesn't happen to include the fleet's longest-lived cell. It is now computed
per battery (`cycle / group-max-cycle`), which is what "normalised cycle
position over battery lifetime" (per the README's feature catalog) actually
requires.
"""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOW = 50


def compute_behavior_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add the five binary behavior flags from the README's feature catalog.

    Expects `current_a`, `temperature_c`, `soc` columns (unified schema).
    """
    out = df.copy()
    out["aggressive_discharge_event"] = (out["current_a"] < -2.0).astype(int)
    out["fast_charge_flag"] = (out["current_a"] > 2.0).astype(int)
    out["high_temp_flag"] = (out["temperature_c"] > 40.0).astype(int)
    out["deep_discharge_flag"] = (out["soc"] < 20.0).astype(int)
    out["high_soc_flag"] = (out["soc"] > 90.0).astype(int)
    return out


def add_rolling_features(df: pd.DataFrame, cell_col: str = "cell_id", window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Add rolling stress/temperature stats within each battery's own row order.

    Requires a `stress_score` column (see `risk.stress_score.compute_stress_score`)
    and `temperature_c`. Assumes `df` is already sorted by (cell_id, cycle) —
    callers are responsible for sorting, since this function must not silently
    reorder rows out from under a caller holding a parallel index.
    """
    if "stress_score" not in df.columns:
        raise ValueError("add_rolling_features requires a 'stress_score' column; run risk.compute_stress_score first")

    out = df.copy()
    grouped_stress = out.groupby(cell_col)["stress_score"]
    out["stress_rolling_mean"] = grouped_stress.transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    out["stress_rolling_std"] = grouped_stress.transform(lambda x: x.rolling(window=window, min_periods=1).std()).fillna(0)

    grouped_temp = out.groupby(cell_col)["temperature_c"]
    out["temp_rolling_mean"] = grouped_temp.transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    out["temp_rolling_max"] = grouped_temp.transform(lambda x: x.rolling(window=window, min_periods=1).max())
    return out


def add_age_features(df: pd.DataFrame, cell_col: str = "cell_id", cycle_col: str = "cycle") -> pd.DataFrame:
    """Add `battery_age_factor` (per-battery normalised cycle position) and
    `cycle_stress_index` (stress x age, penalising late-life stress events).

    Requires `stress_score`. See module docstring for the per-battery-max fix.
    """
    if "stress_score" not in df.columns:
        raise ValueError("add_age_features requires a 'stress_score' column; run risk.compute_stress_score first")

    out = df.copy()
    max_cycle_per_battery = out.groupby(cell_col)[cycle_col].transform("max")
    out["battery_age_factor"] = (out[cycle_col] / max_cycle_per_battery).fillna(0)
    out["cycle_stress_index"] = out["stress_score"] * out["battery_age_factor"]
    return out


def summarize_batteries(df: pd.DataFrame, cell_col: str = "cell_id") -> pd.DataFrame:
    """Aggregate row-level telemetry/flags into one row per battery.

    Requires `stress_score`, `temperature_c`, `soc`, and the five behavior
    flags from `compute_behavior_flags`.
    """
    required = {
        "stress_score", "temperature_c", "soc",
        "fast_charge_flag", "deep_discharge_flag", "high_temp_flag",
        "aggressive_discharge_event",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summarize_batteries: missing required columns {sorted(missing)}")

    summary = (
        df.groupby(cell_col)
        .agg(
            avg_stress=("stress_score", "mean"),
            avg_temp=("temperature_c", "mean"),
            fast_charge_duration=("fast_charge_flag", "sum"),
            deep_discharge_duration=("deep_discharge_flag", "sum"),
            high_temp_duration=("high_temp_flag", "sum"),
            aggressive_discharge_count=("aggressive_discharge_event", "sum"),
            avg_soc=("soc", "mean"),
        )
        .reset_index()
        .rename(columns={cell_col: "battery_id"})
    )
    return summary
