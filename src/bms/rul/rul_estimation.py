"""Remaining Useful Life estimation via an Equivalent Aging Factor (EAF).

Ported from `notebooks/07_remaining_useful_life.ipynb`.

KNOWN METHODOLOGICAL LIMITATION (not fixed, documented instead — fixing it
would mean inventing a replacement model without data to validate it, which
is worse than being explicit about the flaw):

`health_index` is itself built from avg_temp, deep_discharge_duration, and
fast_charge_duration (see `health.health_index`). The EAF formula below
reuses avg_temp, deep_discharge_duration, and fast_charge_duration *again*
alongside health_index. That means temperature and charging behavior are
double-counted — once inside health_index, once directly. The practical
effect is that batteries with elevated temperature/fast-charge/deep-discharge
exposure get a compounded RUL penalty relative to what the stated 40/25/20/15
weighting implies on its own.

This should be resolved by either (a) refitting EAF on capacity-fade-to-80%
ground truth (NASA/CALCE both provide this) so the weights are no longer
hand-picked, or (b) deriving EAF from health_index alone plus one
orthogonal term. Neither is done here because both require a modeling
decision backed by held-out validation, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("health_index", "avg_temp", "deep_discharge_duration", "fast_charge_duration", "remaining_health")


@dataclass(frozen=True)
class RULConfig:
    w_health: float = 0.40
    w_temp: float = 0.25
    w_deep_discharge: float = 0.20
    w_fast_charge: float = 0.15
    eaf_min: float = 0.05
    eaf_max: float = 1.0
    base_cycle_life: float = 1000.0  # unvalidated constant; see module docstring


def replacement_policy(rul_cycles: float) -> str:
    if rul_cycles < 100:
        return "REPLACE"
    if rul_cycles < 300:
        return "PLAN_SERVICE"
    if rul_cycles < 600:
        return "MONITOR"
    return "NORMAL"


def compute_rul(battery: pd.DataFrame, config: RULConfig = RULConfig()) -> pd.DataFrame:
    """Compute equivalent_aging_factor, estimated_total_cycles, rul_cycles, replacement_policy.

    Expects the output of `health.health_index.compute_health_index`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery.columns]
    if missing:
        raise ValueError(f"compute_rul: missing required columns {missing}")

    out = battery.copy()

    dd_max = out["deep_discharge_duration"].max() or 1
    fc_max = out["fast_charge_duration"].max() or 1

    eaf = (
        config.w_health * (out["health_index"] / 100)
        + config.w_temp * (out["avg_temp"] / 50)
        + config.w_deep_discharge * (out["deep_discharge_duration"] / dd_max)
        + config.w_fast_charge * (out["fast_charge_duration"] / fc_max)
    )
    out["equivalent_aging_factor"] = eaf.clip(config.eaf_min, config.eaf_max)

    out["estimated_total_cycles"] = config.base_cycle_life / out["equivalent_aging_factor"]
    out["rul_cycles"] = (out["estimated_total_cycles"] * out["remaining_health"] / 100).round().astype(int)
    out["replacement_policy"] = out["rul_cycles"].apply(replacement_policy)
    return out
