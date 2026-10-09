"""The validated-conditions warning: cold cells and temperature changes are LOW.

It changes no number; it only stops a reading taken where large errors were
measured from being labelled MEDIUM or HIGH (docs/weaknesses_audit_and_plan.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.api.app import app  # noqa: E402
from src.bms.api.profile_routes import forget_all, remember  # noqa: E402
from src.bms.health.evidence import temperature_flag  # noqa: E402
from tests.test_field_soh import _cell  # noqa: E402


def test_room_temperature_unchanged_is_not_flagged():
    assert temperature_flag([25, 25, 26], [27, 26, 25]) == ""


def test_cold_is_flagged():
    assert "below 15 C" in temperature_flag([5, 5], [6, 5])


def test_a_temperature_change_since_the_reference_is_flagged():
    reason = temperature_flag([30, 30, 31], [20, 21, 20])
    assert "when its reference was formed" in reason


def test_unknown_temperature_makes_no_claim():
    assert temperature_flag([float("nan")], []) == ""


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("BEACON_PROFILE_DIR", raising=False)
    forget_all()
    yield TestClient(app)
    forget_all()


def _profile(client, battery_id, temps):
    tel, _ = _cell(n_cycles=30)
    tel["temperature_c"] = temps(tel)
    remember(battery_id, tel)
    return client.get(f"/batteries/{battery_id}/profile").json()


def test_a_cold_battery_is_low_with_the_reason(client):
    body = _profile(client, "COLD", lambda t: 5.0)
    assert body["soh"] is not None                       # still measured, number unchanged
    assert body["confidence"] == "LOW"
    assert "outside the validated conditions" in body["confidence_reason"]
    listed = {r["battery_id"]: r for r in client.get("/profiles").json()}
    assert listed["COLD"]["confidence"] == "LOW"


def test_a_room_temperature_battery_is_not_downgraded(client):
    body = _profile(client, "WARM", lambda t: 25.0)
    assert body["confidence"] in ("MEDIUM", "HIGH")
    assert body["validated_error_p90"] == pytest.approx(0.0343)
