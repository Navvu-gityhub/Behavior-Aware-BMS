"""Synthetic EV battery telemetry generator.

This module exists so `main.py` and the test suite can run end-to-end without
requiring a NASA/CALCE download. It is a data-generation *convenience*, not a
validated battery model — do not use its capacity-fade trajectory as ground
truth for anything beyond exercising the pipeline.

Design notes / assumptions (explicit, since these are not derived from data):
- Three fleet usage archetypes (gentle / normal / aggressive) drive the
  probability of fast-charge, deep-discharge, and high-temperature events.
  This is a deliberate simplification to guarantee behavioral variance across
  the simulated fleet so downstream risk/health scoring has something to
  differentiate.
- Capacity fade per cycle is modeled as a base linear fade plus a stress
  penalty proportional to the row's rule-based stress score. This creates a
  loose, directionally-correct link between "bad behavior" and "faster fade"
  for demo purposes. It is not calibrated against measured cell data and
  should not be cited as a degradation model.
- Current sign convention: positive = charging, negative = discharging,
  consistent with `src/bms/preprocessing/schema.infer_mode`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_PROFILES = ("gentle", "normal", "aggressive")

# Per-profile probability of an "event" row (fast charge, deep discharge,
# high-temperature exposure) on any given charge/discharge step. Heuristic,
# chosen only to spread the fleet across risk bands for demonstration.
_PROFILE_EVENT_RATE = {
    "gentle": 0.05,
    "normal": 0.15,
    "aggressive": 0.35,
}


@dataclass(frozen=True)
class SimulationConfig:
    n_batteries: int = 20
    rows_per_battery: int = 400
    seed: int = 42
    nominal_voltage: float = 3.7
    ambient_temp_c: float = 25.0
    base_capacity_ah: float = 2.0


def _simulate_one_battery(cell_id: str, profile: str, cfg: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.rows_per_battery
    event_rate = _PROFILE_EVENT_RATE[profile]

    cycle = np.arange(1, n + 1)
    soc = np.zeros(n)
    current = np.zeros(n)
    temperature = np.zeros(n)
    voltage = np.zeros(n)
    capacity = np.zeros(n)
    speed = np.zeros(n)
    distance = np.zeros(n)

    soc_val = rng.uniform(60, 100)
    capacity_val = cfg.base_capacity_ah
    cumulative_distance = 0.0

    for i in range(n):
        is_fast_event = rng.random() < event_rate
        mode = "charge" if rng.random() < 0.4 else "discharge"

        if mode == "charge":
            current_val = rng.uniform(2.2, 3.5) if is_fast_event else rng.uniform(0.3, 1.8)
            soc_val = min(100.0, soc_val + current_val * rng.uniform(0.6, 1.0))
            speed_val = 0.0
        else:
            current_val = -rng.uniform(2.2, 4.0) if is_fast_event else -rng.uniform(0.3, 1.8)
            soc_val = max(0.0, soc_val + current_val * rng.uniform(0.6, 1.0))
            if is_fast_event:
                soc_val = min(soc_val, rng.uniform(5, 18))  # occasional deep discharge
            speed_val = rng.uniform(20, 110)
            distance_val = speed_val * rng.uniform(0.008, 0.02)
            cumulative_distance += distance_val

        temp_val = cfg.ambient_temp_c + abs(current_val) * rng.uniform(1.5, 2.5)
        if is_fast_event:
            temp_val += rng.uniform(8, 18)
        temp_val += rng.normal(0, 1.0)

        voltage_val = cfg.nominal_voltage + (soc_val - 50) / 250 + rng.normal(0, 0.02)

        # Loose stress proxy driving capacity fade (see module docstring).
        stress_proxy = (
            (abs(current_val) > 2.0) * 0.4
            + (temp_val > 40) * 0.35
            + (soc_val < 20) * 0.15
            + (soc_val > 90) * 0.10
        )
        capacity_val -= cfg.base_capacity_ah * (0.00015 + 0.0006 * stress_proxy)
        capacity_val = max(capacity_val, cfg.base_capacity_ah * 0.5)

        soc[i] = soc_val
        current[i] = current_val
        temperature[i] = temp_val
        voltage[i] = voltage_val
        capacity[i] = capacity_val
        speed[i] = speed_val if mode == "discharge" else 0.0
        distance[i] = cumulative_distance

    return pd.DataFrame(
        {
            "dataset": "simulated",
            "cell_id": cell_id,
            "cycle": cycle,
            "voltage_v": voltage,
            "current_a": current,
            "temperature_c": temperature,
            "soc": soc,
            "capacity_ah": capacity,
            "speed_kmh": speed,
            "distance_km": distance,
            "usage_profile": profile,  # not part of the unified schema; dropped before schema validation if needed
        }
    )


def simulate_fleet(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Generate a synthetic fleet telemetry table spanning multiple behavior profiles.

    Returns a long-format DataFrame with one row per (cell_id, cycle), using
    the unified schema's raw-friendly column names (voltage_v, current_a,
    temperature_c, soc, capacity_ah) plus two extra columns (speed_kmh,
    distance_km) referenced in the project README but not part of the
    canonical schema.
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.seed)

    frames = []
    for i in range(cfg.n_batteries):
        profile = _PROFILES[i % len(_PROFILES)]
        cell_id = f"SIM{i:03d}"
        frames.append(_simulate_one_battery(cell_id, profile, cfg, rng))

    return pd.concat(frames, ignore_index=True)
