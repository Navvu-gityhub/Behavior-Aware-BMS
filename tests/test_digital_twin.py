"""Tests for src.bms.digital_twin: state derivation, transitions, timeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from main import run_pipeline
from src.bms.simulation.simulate_telemetry import SimulationConfig, simulate_fleet
from src.bms.digital_twin import (
    build_health_timeline,
    detect_transition,
    evaluate_fleet,
    evaluate_twin_state,
)


def _fleet_and_guardian():
    raw = simulate_fleet(SimulationConfig(n_batteries=6, rows_per_battery=120, seed=7))
    guardian = run_pipeline(raw).guardian
    return raw, guardian


def test_evaluate_fleet_covers_every_battery_and_valid_states():
    _, guardian = _fleet_and_guardian()
    snapshots = evaluate_fleet(guardian)

    assert set(snapshots.keys()) == set(guardian["battery_id"])
    for bid, snap in snapshots.items():
        assert snap.battery_id == bid
        assert snap.twin_state in ("NORMAL", "MODERATE_RISK", "HIGH_RISK", "FAILURE_IMMINENT")
        assert 0.0 <= snap.failure_likelihood <= 1.0


def test_twin_state_relabels_battery_state_1to1_not_independently():
    """The twin state must be a deterministic relabel of health_index's
    own battery_state, not a separately-invented threshold system --
    this is the specific design decision the module docstring commits to."""
    _, guardian = _fleet_and_guardian()
    expected = {
        "HEALTHY": "NORMAL",
        "WARNING": "MODERATE_RISK",
        "DEGRADED": "HIGH_RISK",
        "CRITICAL": "FAILURE_IMMINENT",
    }
    for _, row in guardian.iterrows():
        snap = evaluate_twin_state(row)
        assert snap.twin_state == expected[row["battery_state"]]


def test_evaluate_twin_state_raises_on_missing_columns():
    row = pd.Series({"battery_id": "X"})
    with pytest.raises(ValueError, match="missing required fields"):
        evaluate_twin_state(row)


def test_evaluate_twin_state_raises_on_unrecognized_battery_state():
    row = pd.Series(
        {
            "battery_id": "X",
            "battery_state": "SOMETHING_NEW",
            "health_index": 50.0,
            "rul_cycles": 100,
            "replacement_policy": "MONITOR",
        }
    )
    with pytest.raises(ValueError, match="unrecognized battery_state"):
        evaluate_twin_state(row)


def test_detect_transition_first_evaluation_counts_as_a_transition():
    _, guardian = _fleet_and_guardian()
    snap = evaluate_twin_state(guardian.iloc[0])
    t = detect_transition(None, snap)
    assert t is not None
    assert t.from_state is None
    assert t.to_state == snap.twin_state


def test_detect_transition_same_state_is_not_a_transition():
    _, guardian = _fleet_and_guardian()
    snap = evaluate_twin_state(guardian.iloc[0])
    assert detect_transition(snap, snap) is None


def test_detect_transition_rejects_mismatched_battery_ids():
    _, guardian = _fleet_and_guardian()
    snap_a = evaluate_twin_state(guardian.iloc[0])
    snap_b = evaluate_twin_state(guardian.iloc[1])
    with pytest.raises(ValueError, match="different batteries"):
        detect_transition(snap_a, snap_b)


def test_build_health_timeline_ordered_by_cycle():
    raw, guardian = _fleet_and_guardian()
    from src.bms.preprocessing.schema import standardize_validate_bms_data
    from src.bms.features.behavior_features import compute_behavior_flags
    from src.bms.risk.stress_score import compute_stress_score

    clean, _ = standardize_validate_bms_data(raw, dataset="simulated")
    flagged = compute_behavior_flags(clean)
    flagged["stress_score"] = compute_stress_score(flagged)

    battery_id = guardian["battery_id"].iloc[0]
    timeline = build_health_timeline(flagged, battery_id)

    assert len(timeline) > 0
    cycles = [point["cycle"] for point in timeline]
    assert cycles == sorted(cycles)


def test_build_health_timeline_raises_keyerror_for_unknown_battery():
    raw, _ = _fleet_and_guardian()
    from src.bms.preprocessing.schema import standardize_validate_bms_data
    from src.bms.features.behavior_features import compute_behavior_flags
    from src.bms.risk.stress_score import compute_stress_score

    clean, _ = standardize_validate_bms_data(raw, dataset="simulated")
    flagged = compute_behavior_flags(clean)
    flagged["stress_score"] = compute_stress_score(flagged)

    with pytest.raises(KeyError):
        build_health_timeline(flagged, "NOT_A_REAL_BATTERY")


if __name__ == "__main__":
    test_evaluate_fleet_covers_every_battery_and_valid_states()
    test_twin_state_relabels_battery_state_1to1_not_independently()
    test_detect_transition_first_evaluation_counts_as_a_transition()
    test_detect_transition_same_state_is_not_a_transition()
    test_build_health_timeline_ordered_by_cycle()
    print("All digital twin tests passed.")
