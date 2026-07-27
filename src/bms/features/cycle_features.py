"""Per-(battery, cycle) feature aggregation.

`features.behavior_features.summarize_batteries` collapses a battery's
entire history into one row, which is the right shape for the current
health/risk scores but the wrong shape for validating them against a
capacity-fade *curve*. This module aggregates to one row per
(cell_id, cycle) instead, so behavior in a given cycle (or a trailing
window of cycles) can be correlated against that cycle's measured capacity.
"""

from __future__ import annotations

import pandas as pd


def summarize_by_cycle(df: pd.DataFrame, cell_col: str = "cell_id", cycle_col: str = "cycle") -> pd.DataFrame:
    required = {
        "stress_score", "temperature_c", "soc",
        "fast_charge_flag", "deep_discharge_flag", "high_temp_flag",
        "aggressive_discharge_event",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"summarize_by_cycle: missing required columns {sorted(missing)}")

    agg = (
        df.groupby([cell_col, cycle_col])
        .agg(
            avg_stress=("stress_score", "mean"),
            avg_temp=("temperature_c", "mean"),
            max_temp=("temperature_c", "max"),
            fast_charge_duration=("fast_charge_flag", "sum"),
            deep_discharge_duration=("deep_discharge_flag", "sum"),
            high_temp_duration=("high_temp_flag", "sum"),
            aggressive_discharge_count=("aggressive_discharge_event", "sum"),
            avg_soc=("soc", "mean"),
            n_rows=("stress_score", "size"),
        )
        .reset_index()
    )
    return agg
