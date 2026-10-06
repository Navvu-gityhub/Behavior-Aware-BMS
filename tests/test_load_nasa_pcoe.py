"""Tests for the NASA PCoE .mat loader, on a synthetic file in NASA's layout.

The real archive is not in the repository. The fixture reproduces the struct
NASA distributes - one struct per battery whose `cycle` array mixes charge,
discharge and impedance records - so the loader is tested against the shape it
must parse, and the real run is recorded in reports/metrics/cross_dataset/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.io.load_nasa_pcoe import (  # noqa: E402
    cohort_for,
    discharges_to_frames,
    load_nasa_pcoe_cell,
)

scipy_io = pytest.importorskip("scipy.io")


def _discharge(capacity: float, r: float = 0.1) -> dict:
    n = 60
    t = np.arange(n) * 20.0
    i = np.full(n, -2.0)
    i[:2] = 0.0                                   # rest before the load
    ocv = np.linspace(4.19, 3.2, n)
    v = ocv + i * r
    return {"type": "discharge", "ambient_temperature": 24.0, "time": 0.0,
            "data": {"Voltage_measured": v, "Current_measured": i,
                     "Temperature_measured": np.full(n, 25.0), "Current_load": i,
                     "Voltage_load": v, "Time": t, "Capacity": capacity}}


def _write(tmp_path: Path) -> Path:
    cycle = np.empty((1, 4), dtype=object)
    cycle[0, 0] = {"type": "charge", "ambient_temperature": 24.0, "time": 0.0,
                   "data": {"Voltage_measured": np.ones(3)}}
    cycle[0, 1] = _discharge(1.85)
    cycle[0, 2] = {"type": "impedance", "ambient_temperature": 24.0, "time": 0.0,
                   "data": {"Re": 0.05}}
    cycle[0, 3] = _discharge(1.80)
    path = tmp_path / "B9999.mat"
    scipy_io.savemat(str(path), {"B9999": {"cycle": cycle}})
    return path


def test_only_discharge_records_are_returned(tmp_path):
    ds = load_nasa_pcoe_cell(_write(tmp_path))
    assert [d.index for d in ds] == [1, 2]
    assert [round(d.capacity_ah, 2) for d in ds] == [1.85, 1.80]


def test_cohort_names_ambient_and_load(tmp_path):
    assert cohort_for(load_nasa_pcoe_cell(_write(tmp_path))) == "24C_2.0A"


def test_frames_start_at_the_load_and_recover_the_step_resistance(tmp_path):
    curves, steps = discharges_to_frames(load_nasa_pcoe_cell(_write(tmp_path)))
    first = curves[curves["cycle"] == 1]
    assert first["capacity_ah_curve"].iloc[0] == 0.0
    assert (np.diff(first["capacity_ah_curve"]) > 0).all()
    # Injected 0.1 ohm; the step also includes one sample of OCV slope.
    assert steps["r_step_ohm"].median() == pytest.approx(0.1, rel=0.15)
