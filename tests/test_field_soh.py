"""Tests for SOH from raw field telemetry.

The fixture is a synthetic cell with a KNOWN open-circuit curve, a known
capacity per cycle and a known, growing resistance. That makes the right
answer exact: the curve scales uniformly by construction, so the compensated
window ratio must equal the capacity ratio, and the step estimate must
recover the resistance that was put in.

What the tests guard:

`test_step_resistance_recovers_the_injected_value` - the ohmic correction is
only as good as the resistance behind it, and a BMS must infer that from the
load step.

`test_compensation_beats_raw_when_resistance_grows` - the reason the
correction exists. Without it a fixed terminal-voltage window slides along the
curve as resistance rises and the drift reads as extra fade.

`test_resistance_uses_no_future_cycles` - a trailing median, not a centred
one. A cycle's correction must not depend on cycles not yet recorded.

`test_temperature_is_not_needed` - the point of decoupling: the behaviour
scoring refuses without a temperature channel, measured health must not.

These fixtures are synthetic, not CALCE measurements; the CALCE result is in
`reports/metrics/calce_field_soh/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import (  # noqa: E402
    current_field_soh,
    curves_from_telemetry,
    field_soh_table,
    monotonic_time,
    state_from_soh,
    step_overpotential,
)
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

REST = 0.02
DT = 10.0
I_LOAD = 1.0


def _ocv(x: np.ndarray) -> np.ndarray:
    """Open-circuit voltage against fraction of capacity discharged."""
    return 4.20 - 0.55 * x - 0.35 * x ** 3


def _cell(n_cycles: int = 40, q0: float = 1.0, fade: float = 0.004,
          r0: float = 0.10, r_growth: float = 0.002) -> tuple[pd.DataFrame, np.ndarray]:
    """Rest -> 1 A discharge -> rest -> 1 A charge, repeated. Returns (telemetry, capacity)."""
    current: list[float] = []
    voltage: list[float] = []
    caps = q0 * (1 - fade * np.arange(n_cycles))
    for n, q in enumerate(caps):
        r = r0 + r_growth * n
        steps = int(round(q * 3600 / (I_LOAD * DT)))
        x = np.arange(1, steps + 1) * I_LOAD * DT / 3600 / q
        current += [0.0] * 6                              # rest at full
        voltage += [float(_ocv(np.array([0.0]))[0])] * 6
        current += [-I_LOAD] * steps                      # discharge
        voltage += list(_ocv(x) - I_LOAD * r)
        current += [0.0] * 6                              # rest at empty
        voltage += [float(_ocv(np.array([1.0]))[0])] * 6
        current += [I_LOAD] * steps                       # charge
        voltage += [4.0] * steps
    frame = pd.DataFrame({
        "test_time_s": np.arange(len(current)) * DT,
        "current_a": current,
        "voltage_v": voltage,
    })
    frame["cell_id"] = "SYN"
    return frame, caps


def _discharges(tel: pd.DataFrame) -> pd.DataFrame:
    return cycles_to_frame(measure_cycles(tel, "SYN", rest_threshold_a=REST),
                           complete_only=False)


# ---------------------------------------------------------------------------
# Time base
# ---------------------------------------------------------------------------

def test_monotonic_time_joins_restarted_segments():
    t = np.array([0, 10, 20, 0, 10, 20], dtype=float)
    out = monotonic_time(t)
    assert (np.diff(out) > 0).all()
    assert out[3] == pytest.approx(30.0)


def test_monotonic_time_leaves_a_clean_series_alone():
    t = np.arange(0, 100, 10, dtype=float)
    assert np.array_equal(monotonic_time(t), t)


# ---------------------------------------------------------------------------
# Resistance from the step
# ---------------------------------------------------------------------------

def test_step_resistance_recovers_the_injected_value():
    tel, _ = _cell(n_cycles=10, r_growth=0.0)
    _, steps = curves_from_telemetry(tel, _discharges(tel), "SYN", REST)
    assert steps["r_step_ohm"].notna().all()
    assert steps["r_step_ohm"].median() == pytest.approx(0.10, rel=0.05)


def test_resistance_uses_no_future_cycles():
    tel, _ = _cell(n_cycles=20)
    _, steps = curves_from_telemetry(tel, _discharges(tel), "SYN", REST)
    base = step_overpotential(steps)
    tampered = steps.copy()
    tampered.loc[tampered["cycle"] > 10, "r_step_ohm"] = 5.0
    after = step_overpotential(tampered)
    early = base["cycle"] <= 10
    assert np.allclose(base.loc[early, "ir_drop_v"], after.loc[early, "ir_drop_v"])


def test_a_discharge_without_a_rest_before_it_has_no_step():
    tel, _ = _cell(n_cycles=6)
    # Remove the rest that precedes every discharge.
    tel = tel[~((tel["current_a"] == 0.0) & (tel["voltage_v"] > 4.1))].reset_index(drop=True)
    _, steps = curves_from_telemetry(tel, _discharges(tel), "SYN", REST)
    assert steps["r_step_ohm"].isna().all()


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------

def _abs_error(table: pd.DataFrame, caps: np.ndarray) -> float:
    ok = table.dropna(subset=["soh_window_accepted"])
    truth = caps / np.median(caps[:5])
    return float(np.mean(np.abs(
        ok["soh_window_accepted"].to_numpy() - truth[ok["cycle"].to_numpy() - 1])))


def test_compensation_beats_raw_when_resistance_grows():
    tel, caps = _cell(n_cycles=40, r_growth=0.002)
    d = _discharges(tel)
    comp = field_soh_table(tel, d, "SYN", REST, compensate=True)
    raw = field_soh_table(tel, d, "SYN", REST, compensate=False)
    assert _abs_error(comp, caps) < 0.01
    assert _abs_error(comp, caps) < _abs_error(raw, caps)


def test_current_soh_tracks_the_injected_fade():
    tel, caps = _cell(n_cycles=40)
    result = current_field_soh(tel, _discharges(tel), "SYN", REST)
    assert result.refusal == ""
    # Reported SOH is the median of the last five accepted discharges.
    expected = np.median(caps[-5:]) / np.median(caps[:5])
    assert result.soh == pytest.approx(expected, abs=0.01)


def test_too_few_discharges_is_refused_with_the_count():
    tel, _ = _cell(n_cycles=4)
    result = current_field_soh(tel, _discharges(tel), "SYN", REST)
    assert not result.available
    assert "needed to form the reference" in result.refusal


def test_no_voltage_channel_is_refused():
    tel, _ = _cell(n_cycles=10)
    result = current_field_soh(tel.drop(columns=["voltage_v"]), _discharges(tel), "SYN", REST)
    assert not result.available
    assert "voltage" in result.refusal


def test_no_clean_step_refuses_rather_than_running_uncompensated():
    tel, _ = _cell(n_cycles=10)
    tel = tel[~((tel["current_a"] == 0.0) & (tel["voltage_v"] > 4.1))].reset_index(drop=True)
    result = current_field_soh(tel, _discharges(tel), "SYN", REST)
    assert not result.available
    assert "resistance" in result.refusal


# ---------------------------------------------------------------------------
# State bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("soh,state", [
    (1.00, "HEALTHY"), (0.95, "HEALTHY"), (0.94, "WARNING"), (0.90, "WARNING"),
    (0.89, "DEGRADED"), (0.80, "DEGRADED"), (0.79, "CRITICAL"),
])
def test_state_bands(soh, state):
    assert state_from_soh(soh) == state


def test_state_from_a_missing_soh_raises():
    with pytest.raises(ValueError):
        state_from_soh(float("nan"))


# ---------------------------------------------------------------------------
# Pipeline: measured health survives a scoring refusal
# ---------------------------------------------------------------------------

def _score(tel: pd.DataFrame, **kwargs):
    from src.bms.telemetry.pipeline import score_telemetry_frame
    from src.bms.telemetry.sources import SignalCoverage

    coverage = SignalCoverage(
        dbc_path="synthetic", available_signals=("current_a", "voltage_v"),
        mapped_channels=("current_a", "voltage_v"),
        missing_channels=("temperature_c",), transport="serial")
    return score_telemetry_frame(
        source_name="synthetic", telemetry=tel, coverage=coverage,
        n_frames=len(tel), n_decoded=len(tel), cell_id="SYN",
        rest_threshold_a=REST, **kwargs)


def test_temperature_is_not_needed():
    tel, _ = _cell(n_cycles=40)
    result = _score(tel)
    assert not result.scored                     # behaviour scoring refused
    assert result.field_soh is not None and result.field_soh.available
    assert "measure_soh" in result.stages_completed
    assert "measured SOH" in result.render()


def test_soh_reference_defaults_to_start_of_log():
    tel, _ = _cell(n_cycles=40)
    assert _score(tel).soh_reference == "start_of_log"
    assert _score(tel, reference_is_beginning_of_life=True).soh_reference == "beginning_of_life"


# ---------------------------------------------------------------------------
# Field gates
# ---------------------------------------------------------------------------

def test_a_reference_formed_late_refuses_the_cell():
    """The CS2_38 partial-discharge failure, reproduced synthetically.

    Early discharges stop above the window, so the first measurable cycles -
    and with them the reference - come late in life.
    """
    # 45 truncated discharges of ~0.6 Ah each is ~27 equivalent full cycles
    # before the window is first crossed - past MAX_REFERENCE_EFC.
    tel, _ = _cell(n_cycles=60)
    d = _discharges(tel)
    late_start = d.sort_values("cycle")["start_time_s"].iloc[45]
    early_rows = (tel["test_time_s"] < late_start) & (tel["current_a"] < 0)
    tel = tel[~(early_rows & (tel["voltage_v"] < 3.75))].reset_index(drop=True)
    result = current_field_soh(tel, _discharges(tel), "SYN", REST)
    assert not result.available
    assert "not within the first" in result.refusal


def test_an_implausibly_high_reading_is_withheld():
    from src.bms.health.field_soh import MAX_PLAUSIBLE_SOH, apply_field_gates

    tel, _ = _cell(n_cycles=20)
    d = _discharges(tel)
    table = field_soh_table(tel, d, "SYN", REST)
    table.loc[table.index[-1], "soh_window"] = MAX_PLAUSIBLE_SOH + 0.2
    gated = apply_field_gates(table, d)
    assert np.isnan(gated["soh_window_accepted"].iloc[-1])


# ---------------------------------------------------------------------------
# Learned windows: partial discharges
# ---------------------------------------------------------------------------

def _ocv_knee(x: np.ndarray) -> np.ndarray:
    """The plateau curve plus an exponential end-of-discharge knee."""
    return _ocv(x) - 0.8 * np.exp((x - 1.0) / 0.04)


def _partial_cell(n_cycles: int, x_start: float, depth_ah: float,
                  ocv=_ocv, q0: float = 1.0, fade: float = 0.003,
                  r: float = 0.10) -> tuple[pd.DataFrame, np.ndarray]:
    """Repeated partial discharges of `depth_ah` from state `x_start`."""
    current: list[float] = []
    voltage: list[float] = []
    caps = q0 * (1 - fade * np.arange(n_cycles))
    for q in caps:
        steps = int(round(depth_ah * 3600 / (I_LOAD * DT)))
        x = x_start + np.arange(1, steps + 1) * I_LOAD * DT / 3600 / q
        rest_v = float(ocv(np.array([x_start]))[0])
        current += [0.0] * 6
        voltage += [rest_v] * 6
        current += [-I_LOAD] * steps
        voltage += list(ocv(x) - I_LOAD * r)
        current += [0.0] * 6
        voltage += [float(ocv(np.array([x[-1]]))[0])] * 6
        current += [I_LOAD] * steps
        voltage += [4.0] * steps
    frame = pd.DataFrame({"test_time_s": np.arange(len(current)) * DT,
                          "current_a": current, "voltage_v": voltage})
    frame["cell_id"] = "SYN"
    return frame, caps


def test_top_of_charge_partials_are_measured_on_a_learned_window():
    from src.bms.health.field_soh import learned_field_soh

    tel, caps = _partial_cell(60, x_start=0.02, depth_ah=0.30)
    d = _discharges(tel)
    fixed = current_field_soh(tel, d, "SYN", REST)
    assert not fixed.available                      # never reaches 3.60 V
    learned, table = learned_field_soh(tel, d, "SYN", REST, rated_capacity_ah=1.0)
    assert learned.refusal == ""
    assert learned.window_mode == "learned"
    expected = np.median(caps[-5:]) / np.median(caps[:5])
    assert learned.soh == pytest.approx(expected, abs=0.02)


def test_partials_on_the_knee_are_refused_with_the_reason():
    from src.bms.health.field_soh import learned_field_soh

    tel, _ = _partial_cell(30, x_start=0.72, depth_ah=0.27, ocv=_ocv_knee)
    learned, _ = learned_field_soh(tel, _discharges(tel), "SYN", REST,
                                   rated_capacity_ah=1.0)
    assert not learned.available
    assert "knee" in learned.refusal


def test_the_window_is_learned_from_early_discharges_only():
    """Rewriting late life must not move the learned window."""
    from src.bms.health.field_soh import curves_from_telemetry, learn_usage_window

    tel, _ = _partial_cell(40, x_start=0.02, depth_ah=0.30)
    curves, _ = curves_from_telemetry(tel, _discharges(tel), "SYN", REST)
    base = learn_usage_window(curves, 1.0)
    tampered = curves.copy()
    late = tampered["cycle"] > 20
    tampered.loc[late, "voltage_v"] = tampered.loc[late, "voltage_v"] - 0.3
    assert learn_usage_window(tampered, 1.0).spec == base.spec


def test_pipeline_falls_back_to_a_learned_window_only_with_a_rated_capacity():
    tel, _ = _partial_cell(60, x_start=0.02, depth_ah=0.30)
    without = _score(tel, rated_capacity_ah=None)
    assert not without.field_soh.available
    with_cap = _score(tel, rated_capacity_ah=1.0)
    assert with_cap.field_soh.available
    assert with_cap.field_soh.window_mode == "learned"
