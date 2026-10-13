"""The bench script's Ohm's-law cross-check of the current sensor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench_session import ohm_cross_check  # noqa: E402


def _discharge(load_ohm: float, sensor_scale: float = 1.0) -> pd.DataFrame:
    t = np.arange(0, 3600 * 2, 1.0)
    v = np.concatenate([np.full(60, 4.18), np.linspace(4.05, 3.2, len(t) - 60)])
    i = np.concatenate([np.zeros(60), -v[60:] / load_ohm]) * sensor_scale
    return pd.DataFrame({"test_time_s": t, "voltage_v": v, "current_a": i})


def test_a_correct_current_sensor_agrees_with_ohms_law():
    ina, vr = ohm_cross_check(_discharge(4.7), 4.7)
    assert ina == pytest.approx(vr, rel=0.01)


def test_a_mis_scaled_current_sensor_is_exposed():
    ina, vr = ohm_cross_check(_discharge(4.7, sensor_scale=0.8), 4.7)
    assert abs(ina - vr) / vr > 0.15


def test_no_discharge_gives_no_check():
    tel = pd.DataFrame({"test_time_s": np.arange(100.0), "voltage_v": 3.9, "current_a": 0.0})
    assert ohm_cross_check(tel, 4.7) is None
