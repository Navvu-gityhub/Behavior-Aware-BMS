"""Tests for charge_soh on synthetic constant-current charges.

Whether charging-window SOH is accurate on real cells is measured on held-out
data (scripts/run_charge_soh_study.py), not asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.charge_soh import charge_readings, charge_soh_table  # noqa: E402

DT = 10.0
I_CHG = 0.5


def _ocv(soc: np.ndarray) -> np.ndarray:
    return 3.40 + 0.80 * soc - 0.10 * (1 - soc) ** 4


def _log(n: int = 20, fade: float = 0.01, r0: float = 0.10, r_growth: float = 0.003,
         start_soc: float = 0.0) -> tuple[pd.DataFrame, np.ndarray]:
    t, i, v = [], [], []
    caps = 1.0 * (1 - fade * np.arange(n))
    now = 0.0
    for k, cap in enumerate(caps):
        r = r0 + r_growth * k
        for _ in range(6):                                   # rest at the start state
            t.append(now)
            i.append(0.0)
            v.append(float(_ocv(np.array([start_soc]))[0]))
            now += DT
        steps = int(round((1 - start_soc) * cap * 3600 / (I_CHG * DT)))
        soc = start_soc + np.arange(1, steps + 1) * I_CHG * DT / 3600 / cap
        for s in soc:                                        # constant-current charge
            t.append(now)
            i.append(I_CHG)
            v.append(float(_ocv(np.array([s]))[0]) + I_CHG * r)
            now += DT
        for _ in range(6):                                   # rest, then a discharge
            t.append(now)
            i.append(0.0)
            v.append(4.2)
            now += DT
        for _ in range(int(cap * 3600 / (I_CHG * DT))):
            t.append(now)
            i.append(-I_CHG)
            v.append(3.6)
            now += DT
    return pd.DataFrame({"test_time_s": t, "current_a": i, "voltage_v": v}), caps


def test_charge_readings_track_the_injected_fade():
    tel, caps = _log()
    table = charge_soh_table(charge_readings(tel, rest_threshold_a=0.02))
    ok = table.dropna(subset=["soh"])
    truth = caps / np.median(caps[:5])
    err = np.abs(ok["soh"].to_numpy() - truth[ok["charge"].to_numpy() - 1])
    assert len(ok) >= 15
    assert err.mean() < 0.01


def test_the_charge_step_gives_the_resistance():
    tel, _ = _log(r_growth=0.0)
    rs = [r.r_step_ohm for r in charge_readings(tel, rest_threshold_a=0.02)]
    assert np.nanmedian(rs) == pytest.approx(0.10, rel=0.1)


def test_a_top_up_that_starts_above_the_window_is_not_read():
    tel, _ = _log(start_soc=0.8)                             # starts near 4.0 V
    readings = charge_readings(tel, rest_threshold_a=0.02)
    assert all(np.isnan(r.window_charge_ah) for r in readings)
    table = charge_soh_table(readings)
    assert table["soh"].isna().all()
    assert "needed to form the reference" in table["refusal"].iloc[0]
