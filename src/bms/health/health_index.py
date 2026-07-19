"""Battery Health Index via an aging-budget model.

Ported from `notebooks/06_battery_health_index.ipynb`. That notebook actually
defined `battery_state()` twice with different thresholds (cell 17 used a
WARNING cutoff of 40; cells 21-23, which ran last and produced the saved
CSV/docs, used 30). The saved output and `docs/weighted_health_index.md` both
reflect the 30 cutoff, so that is treated as the intended, canonical
definition here; the 40-cutoff version was dead code and is dropped.

The notebook also computed two different health indices in sequence — a
normalized weighted sum (cell 16: 0.35*stress + 0.25*temp + 0.15*dd +
0.15*fc + 0.10*soc_abuse) and then immediately overwrote it with
`100 - aging_budget` (cell 20 onward). Only the aging-budget version was
ever saved to `data/features/battery_health_index_v1.csv` or documented in
`docs/weighted_health_index.md`, so it is the one implemented here. The
weighted-sum version is not reproduced — keeping two silently-conflicting
health index definitions in the same module would be worse than dropping
the one that was never actually used downstream.

Like the risk score, the aging-budget penalty schedule below has hand-chosen
cut points and point deductions. It is not fit to data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "avg_stress",
    "avg_temp",
    "deep_discharge_duration",
    "fast_charge_duration",
    "aggressive_discharge_count",
    "avg_soc",
)


def _battery_state(health_index: float) -> str:
    if health_index >= 80:
        return "CRITICAL"
    if health_index >= 60:
        return "DEGRADED"
    if health_index >= 30:
        return "WARNING"
    return "HEALTHY"


def compute_health_index(battery_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute aging_budget, health_index, remaining_health, consumed_life, battery_state.

    Expects the per-battery summary table from
    `features.behavior_features.summarize_batteries`.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in battery_summary.columns]
    if missing:
        raise ValueError(f"compute_health_index: missing required columns {missing}")

    out = battery_summary.copy()

    budget = np.full(len(out), 100.0)
    budget -= np.where(out["avg_stress"] > 70, 30, np.where(out["avg_stress"] > 50, 20, 10))
    budget -= np.where(out["avg_temp"] > 40, 25, np.where(out["avg_temp"] > 30, 15, 5))
    budget -= np.where(out["deep_discharge_duration"] > 100, 20, np.where(out["deep_discharge_duration"] > 20, 10, 5))
    budget -= np.where(out["fast_charge_duration"] > 100, 15, np.where(out["fast_charge_duration"] > 20, 8, 2))
    budget -= np.where(out["aggressive_discharge_count"] > 500, 15, np.where(out["aggressive_discharge_count"] > 100, 8, 2))
    budget -= np.where((out["avg_soc"] > 80) | (out["avg_soc"] < 20), 10, 0)

    out["aging_budget"] = np.clip(budget, 0, 100)
    out["health_index"] = 100 - out["aging_budget"]
    out["remaining_health"] = out["aging_budget"]
    out["consumed_life"] = 100 - out["remaining_health"]
    out["battery_state"] = out["health_index"].apply(_battery_state)
    return out
