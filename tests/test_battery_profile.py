"""BatteryProfile must answer exactly as the whole-log estimator does.

It keeps a few numbers per discharge instead of the telemetry. That is only
worth having if nothing is lost: fed the same log in pieces, and saved and
reloaded between pieces, it must give the same health, resistance and
uncertainty as `current_field_soh` over the whole log.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.battery_profile import BatteryProfile  # noqa: E402
from src.bms.health.field_soh import current_field_soh  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402
from tests.test_field_soh import REST, _cell  # noqa: E402


def _whole(tel: pd.DataFrame):
    d = cycles_to_frame(measure_cycles(tel, "SYN", rest_threshold_a=REST), complete_only=False)
    return current_field_soh(tel, d, "SYN", REST)


def _chunks(tel: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """Split at the rest before a discharge, so every chunk holds whole discharges."""
    starts = np.flatnonzero((tel["current_a"].to_numpy() < 0) &
                            (np.r_[0.0, tel["current_a"].to_numpy()[:-1]] == 0))
    cuts = [int(starts[i]) - 6 for i in np.linspace(0, len(starts) - 1, n + 1).astype(int)[1:-1]]
    bounds = [0, *cuts, len(tel)]
    return [tel.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def _same(a, b):
    assert a.soh == pytest.approx(b.soh, abs=1e-12)
    assert a.n_accepted == b.n_accepted
    assert a.uncertainty == pytest.approx(b.uncertainty, abs=1e-12)
    assert a.resistance_reference_ohm == pytest.approx(b.resistance_reference_ohm, abs=1e-12)
    assert a.resistance_now_ohm == pytest.approx(b.resistance_now_ohm, abs=1e-12)


def test_one_update_equals_the_whole_log():
    tel, _ = _cell(n_cycles=30)
    p = BatteryProfile("SYN", REST)
    p.update(tel)
    _same(p.current(), _whole(tel))


def test_updates_in_pieces_equal_the_whole_log():
    tel, _ = _cell(n_cycles=30)
    p = BatteryProfile("SYN", REST)
    for chunk in _chunks(tel, 4):
        p.update(chunk)
    assert p.n_discharges == 30
    _same(p.current(), _whole(tel))


def test_memory_survives_save_and_reload(tmp_path):
    tel, _ = _cell(n_cycles=30)
    pieces = _chunks(tel, 3)
    p = BatteryProfile("SYN", REST)
    p.update(pieces[0])
    p.save(tmp_path / "SYN.json")
    q = BatteryProfile.load(tmp_path / "SYN.json")
    for chunk in pieces[1:]:
        q.update(chunk)
    _same(q.current(), _whole(tel))


def test_a_new_battery_refuses_with_the_whole_log_reason():
    tel, _ = _cell(n_cycles=4)
    p = BatteryProfile("SYN", REST)
    p.update(tel)
    got, want = p.current(), _whole(tel)
    assert not got.available
    assert got.refusal == want.refusal


def test_readings_carry_conditions():
    tel, _ = _cell(n_cycles=12)
    tel["temperature_c"] = 25.0
    p = BatteryProfile("SYN", REST)
    p.update(tel)
    r = p.readings()
    assert len(r) == 12
    assert r["mean_current_a"].abs().median() == pytest.approx(1.0, rel=0.01)
    assert {"soh", "r_step_ohm"} <= set(r.columns)


def test_unknown_profile_version_is_refused():
    with pytest.raises(ValueError, match="version"):
        BatteryProfile.from_dict({"version": 99})
