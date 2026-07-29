"""Tests for the BEACON dashboard.

These concentrate on the property the dashboard most needs to hold: **it must
never display a number the pipeline did not compute.** The visual design was
mocked up with placeholder readings (state of health 92.4%, internal
resistance, a week-over-week delta), none of which this pipeline produces.
The tests below pin the unavailable-state behaviour so a future change cannot
quietly substitute a plausible value for a missing one.

The second concern is provenance: a simulated run must be labelled as
simulated. A screenshot of this dashboard being mistaken for measured results
is the most consequential failure it could have in a presentation setting.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.dashboard.beacon import build_beacon_dashboard, render_beacon_html
from src.bms.dashboard.beacon_data import UNAVAILABLE, build_beacon_data
from src.bms.guardian.guardian import generate_guardian_reports
from src.bms.health.health_index import compute_health_index
from src.bms.risk.stress_score import compute_risk_assessment
from src.bms.rul.rul_estimation import compute_rul

SUMMARY_COLS = [
    "avg_stress", "avg_temp", "deep_discharge_duration",
    "fast_charge_duration", "aggressive_discharge_count", "avg_soc",
]


def _guardian(n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    fleet = pd.DataFrame({
        "battery_id": [f"CELL{i:02d}" for i in range(n)],
        "avg_stress": rng.uniform(5, 60, n),
        "avg_temp": rng.uniform(20, 45, n),
        "deep_discharge_duration": rng.integers(0, 200, n).astype(float),
        "fast_charge_duration": rng.integers(0, 200, n).astype(float),
        "aggressive_discharge_count": rng.integers(0, 400, n).astype(float),
        "avg_soc": rng.uniform(25, 85, n),
    })
    risk = compute_risk_assessment(fleet)
    health = compute_health_index(fleet)
    merged = risk.merge(
        health[["battery_id", "aging_budget", "health_index",
                "remaining_health", "consumed_life", "battery_state"]],
        on="battery_id",
    )
    return generate_guardian_reports(compute_rul(merged))


def _telemetry(battery_ids, rows: int = 60, with_capacity: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    frames = []
    for bid in battery_ids:
        cycles = np.arange(1, rows + 1)
        frame = pd.DataFrame({
            "cell_id": bid,
            "cycle": cycles,
            "temperature_c": 25 + rng.normal(0, 2, rows),
            "temp_rolling_mean": 25 + np.linspace(0, 4, rows),
            "stress_score": rng.uniform(0, 40, rows),
            "stress_rolling_mean": np.linspace(5, 25, rows),
            "soc": rng.uniform(20, 90, rows),
            "fast_charge_flag": rng.integers(0, 2, rows),
            "deep_discharge_flag": rng.integers(0, 2, rows),
            "high_temp_flag": 0,
            "high_soc_flag": 0,
            "aggressive_discharge_event": rng.integers(0, 2, rows),
        })
        if with_capacity:
            frame["capacity_ah"] = np.linspace(2.0, 1.6, rows)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Nothing is invented
# ---------------------------------------------------------------------------

def test_state_of_health_unavailable_without_measured_capacity():
    """SOH must not be substituted from the aging budget.

    `remaining_health` is a heuristic severity score on an unrelated scale.
    Presenting it as state of health would be a false measurement claim.
    """
    g = _guardian()
    data = build_beacon_data(g, telemetry=_telemetry(g["battery_id"], with_capacity=False))
    for battery in data.batteries:
        assert battery["soh_available"] is False
        assert battery["soh_latest"] is None
        assert battery["soh_series"] == []


def test_state_of_health_computed_when_capacity_is_present():
    g = _guardian()
    data = build_beacon_data(g, telemetry=_telemetry(g["battery_id"], with_capacity=True))
    for battery in data.batteries:
        assert battery["soh_available"] is True
        # Capacity falls 2.0 -> 1.6, so final SOH should be near 80%.
        assert 78 <= battery["soh_latest"] <= 82


def test_missing_telemetry_yields_empty_series_not_zeros():
    """An absent series must render as 'no data', never as a flat line at zero.

    A flat line reads as 'measured and constant', which is a different and
    false claim.
    """
    g = _guardian()
    data = build_beacon_data(g, telemetry=None)
    for battery in data.batteries:
        assert battery["temp_series"] == []
        assert battery["stress_series"] == []
        assert battery["usage"] == []


def test_unavailable_state_renders_as_explicit_message():
    g = _guardian()
    html = render_beacon_html(build_beacon_data(g, telemetry=None))
    assert "Not available" in html
    assert UNAVAILABLE not in html  # sentinel must never leak to the page


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_simulated_runs_are_labelled_as_simulated():
    g = _guardian()
    html = render_beacon_html(build_beacon_data(g, data_source="simulated"))
    assert "Simulated data." in html
    assert "NOT measurements" in html


def test_measured_runs_are_labelled_with_their_dataset():
    g = _guardian()
    data = build_beacon_data(g, data_source="measured", dataset_label="NASA cleaned_dataset")
    html = render_beacon_html(data)
    assert "Measured data." in html
    assert "NASA cleaned_dataset" in html
    assert data.provenance["is_measured"] is True


# ---------------------------------------------------------------------------
# Attribution surfaced honestly
# ---------------------------------------------------------------------------

def test_attribution_rows_reach_the_dashboard_payload():
    g = _guardian()
    data = build_beacon_data(g)
    for battery in data.batteries:
        assert len(battery["health_attribution"]) == 6
        assert len(battery["risk_attribution"]) == 6


def test_dashboard_reports_score_resolution():
    """Fleet-level distinct-value count must be surfaced.

    The v1 index takes only 6 distinct values across 33 NASA cells; a
    dashboard that hides that makes the score look more informative than it is.
    """
    g = _guardian()
    data = build_beacon_data(g)
    assert data.fleet["distinct_health_values"] == g["health_index"].nunique()


def test_evidence_panel_states_the_negative_validation_result():
    g = _guardian()
    html = render_beacon_html(build_beacon_data(g))
    assert "-0.27" in html          # health index vs measured fade
    assert "unseen protocol" in html  # the LOCO collapse
    assert "61%" in html             # constant-term share of the risk score


# ---------------------------------------------------------------------------
# Output integrity
# ---------------------------------------------------------------------------

def test_dashboard_is_self_contained():
    """No external requests: the file must open from disk with no network."""
    g = _guardian()
    html = render_beacon_html(build_beacon_data(g, telemetry=_telemetry(g["battery_id"])))
    assert "<script src=" not in html
    assert "cdn" not in html.lower()
    for pattern in ("http://", "https://"):
        assert pattern not in html


def test_embedded_payload_is_valid_json():
    g = _guardian()
    html = render_beacon_html(build_beacon_data(g, telemetry=_telemetry(g["battery_id"])))
    match = re.search(r"window\.__BEACON__ = (\{.*?\});", html, re.S)
    assert match, "embedded payload not found"
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert len(payload["batteries"]) == len(g)


def test_build_writes_a_file(tmp_path):
    g = _guardian()
    out = build_beacon_dashboard(g, tmp_path / "nested" / "dashboard.html")
    assert out.exists() and out.stat().st_size > 10_000


def test_empty_guardian_table_raises():
    with pytest.raises(ValueError, match="empty"):
        build_beacon_data(pd.DataFrame())


def test_missing_required_columns_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        build_beacon_data(pd.DataFrame({"battery_id": ["A"], "health_index": [10.0]}))
