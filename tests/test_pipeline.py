"""End-to-end smoke test for the full pipeline (simulate -> dashboard).

README's Quick Start has claimed `python tests/test_pipeline.py` works since
the project's first commit; the file did not exist until now. This covers
the full chain main.run_pipeline() drives, plus a couple of targeted checks
on the modules most likely to silently produce nonsense (threshold
boundaries, per-battery age normalization).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from main import run_pipeline
from src.bms.simulation.simulate_telemetry import SimulationConfig, simulate_fleet
from src.bms.features.behavior_features import compute_behavior_flags, add_age_features
from src.bms.risk.stress_score import compute_stress_score, compute_risk_assessment, RiskThresholds
from src.bms.health.health_index import compute_health_index
from src.bms.dashboard.dashboard import build_dashboard


def _small_fleet() -> pd.DataFrame:
    return simulate_fleet(SimulationConfig(n_batteries=4, rows_per_battery=60, seed=1))


def test_pipeline_end_to_end_runs_and_produces_valid_ranges():
    raw = _small_fleet()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "features"
        reports_dir = Path(tmp) / "reports"
        guardian = run_pipeline(raw, output_dir=out_dir, reports_dir=reports_dir)

        assert len(guardian) == raw["cell_id"].nunique()
        assert guardian["health_index"].between(0, 100).all()
        assert guardian["risk_score"].between(0, 100).all()
        assert guardian["rul_cycles"].ge(0).all()
        assert set(guardian["battery_state"]) <= {"HEALTHY", "WARNING", "DEGRADED", "CRITICAL"}
        assert set(guardian["risk_level"]) <= {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert (out_dir / "battery_guardian_output_v1.csv").exists()

        dash_path = build_dashboard(guardian, Path(tmp) / "dashboard.html")
        assert dash_path.exists()
        assert dash_path.stat().st_size > 1000


def test_risk_and_health_state_boundaries_are_consistent_with_docs():
    # docs/risk_rules.md: LOW 0-39, MEDIUM 40-59, HIGH 60-79, CRITICAL 80-100
    t = RiskThresholds()
    from src.bms.risk.stress_score import _risk_level
    assert _risk_level(39, t) == "LOW"
    assert _risk_level(40, t) == "MEDIUM"
    assert _risk_level(59, t) == "MEDIUM"
    assert _risk_level(60, t) == "HIGH"
    assert _risk_level(79, t) == "HIGH"
    assert _risk_level(80, t) == "CRITICAL"

    # docs/weighted_health_index.md: HEALTHY 0-29, WARNING 30-59, DEGRADED 60-79, CRITICAL 80-100
    from src.bms.health.health_index import _battery_state
    assert _battery_state(29) == "HEALTHY"
    assert _battery_state(30) == "WARNING"
    assert _battery_state(59) == "WARNING"
    assert _battery_state(60) == "DEGRADED"
    assert _battery_state(79) == "DEGRADED"
    assert _battery_state(80) == "CRITICAL"


def test_age_factor_is_normalized_per_battery_not_globally():
    # Regression test for the notebook bug where battery_age_factor used the
    # dataset-wide max cycle instead of each battery's own max cycle.
    df = pd.DataFrame({
        "cell_id": ["A", "A", "B", "B", "B"],
        "cycle": [1, 2, 1, 2, 3],
        "current_a": [1.0] * 5,
        "temperature_c": [25.0] * 5,
        "soc": [50.0] * 5,
    })
    df = compute_behavior_flags(df)
    df["stress_score"] = compute_stress_score(df)
    df = add_age_features(df)

    # Battery A's last row (cycle 2 of 2) should be fully aged (factor 1.0),
    # not 2/3 as it would be if normalized against battery B's max cycle.
    a_last = df[(df.cell_id == "A") & (df.cycle == 2)].iloc[0]
    assert a_last["battery_age_factor"] == 1.0


def test_health_index_requires_expected_columns():
    import pytest
    with pytest.raises(ValueError):
        compute_health_index(pd.DataFrame({"battery_id": ["X"]}))


if __name__ == "__main__":
    test_pipeline_end_to_end_runs_and_produces_valid_ranges()
    test_risk_and_health_state_boundaries_are_consistent_with_docs()
    test_age_factor_is_normalized_per_battery_not_globally()
    print("All tests passed.")
