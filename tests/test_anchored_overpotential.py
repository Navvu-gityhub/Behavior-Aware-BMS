"""Tests for field_soh.anchored_overpotential on synthetic discharge curves.

Whether the correction helps on real cells is measured, not asserted here:
reports/metrics/temperature_fix/ (it improved Oxford and failed its preset
criteria overall, so it is not the default). These tests pin what it does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import anchored_overpotential  # noqa: E402
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402

CAP0 = 2.0


def _ocv(soc: np.ndarray) -> np.ndarray:
    return 3.2 + 1.0 * soc - 0.15 * (1 - soc) ** 3


def _curve(cycle: int, capacity: float, sag: float) -> pd.DataFrame:
    q = np.linspace(0, capacity * 0.98, 400)
    return pd.DataFrame({"cell_id": "X", "cycle": cycle,
                         "voltage_v": _ocv(1 - q / capacity) - sag, "capacity_ah_curve": q})


def _cell(late_sag: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    caps = [CAP0] * 5 + [1.6]
    sags = [0.10] * 5 + [late_sag]
    curves = pd.concat([_curve(i + 1, c, s) for i, (c, s) in enumerate(zip(caps, sags, strict=True))])
    base = pd.DataFrame({"cell_id": "X", "cycle": range(1, 7), "ir_drop_v": 0.10})
    return curves, base


def test_offset_recovers_overpotential_growth_the_step_missed():
    curves, base = _cell(late_sag=0.25)
    o = anchored_overpotential(curves, base=base).set_index("cycle")
    assert o.loc[1:5, "anchor_offset_v"].abs().max() < 1e-9
    # 0.15 V of growth, plus the small early state-of-charge mismatch (stated in field_soh).
    assert o.loc[6, "anchor_offset_v"] == pytest.approx(0.15, abs=0.03)
    assert o.loc[6, "ir_drop_v"] == pytest.approx(0.10 + o.loc[6, "anchor_offset_v"])


def test_anchored_window_reads_capacity_where_the_ohmic_correction_does_not():
    curves, base = _cell(late_sag=0.25)
    spec = WindowSpec(3.90, 3.60)
    shipped = window_soh_table(curves, spec=spec, overpotential=base).set_index("cycle")
    anch = window_soh_table(curves, spec=spec,
                            overpotential=anchored_overpotential(curves, base=base)).set_index("cycle")
    truth = 1.6 / CAP0
    assert abs(anch.loc[6, "soh_window"] - truth) < abs(shipped.loc[6, "soh_window"] - truth)


def test_a_discharge_not_starting_from_full_is_not_anchored():
    curves, base = _cell(late_sag=0.10)
    rest = pd.DataFrame({"cell_id": "X", "cycle": range(1, 7), "v_rest_v": [4.19] * 5 + [3.95]})
    o = anchored_overpotential(curves, base=base, rest=rest).set_index("cycle")
    assert np.isnan(o.loc[6, "anchor_offset_v"])
    assert np.isfinite(o.loc[5, "anchor_offset_v"])


def test_no_reference_level_without_a_step_means_zero_level():
    curves, _ = _cell(late_sag=0.10)
    o = anchored_overpotential(curves).set_index("cycle")
    assert o.loc[1, "ir_drop_v"] == pytest.approx(0.0, abs=1e-9)
