"""Integration tests for the FastAPI transport layer (src.bms.api.app).

Uses a fresh FleetStore per test (not the app's module-level singleton)
so tests don't leak state into each other -- the module-level store exists
for the real running service, not for test isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient

from src.bms.api.app import app
from src.bms.api.store import FleetStore
import src.bms.api.app as app_module


@pytest.fixture(autouse=True)
def isolated_store(monkeypatch):
    fresh = FleetStore()
    monkeypatch.setattr(app_module, "fleet_store", fresh)
    yield fresh


@pytest.fixture
def client():
    return TestClient(app)


def test_healthz_before_any_run(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_runs"] == 0
    assert body["n_batteries_tracked"] == 0


def test_batteries_empty_before_any_run(client):
    r = client.get("/batteries")
    assert r.status_code == 200
    assert r.json() == []


def test_simulate_then_list_and_detail(client):
    r = client.post("/pipeline/simulate", json={"n_batteries": 4, "rows_per_battery": 100, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["n_batteries_scored"] == 4
    assert len(body["battery_ids"]) == 4
    # Every battery's first-ever evaluation is itself a transition (from_state=None).
    assert len(body["transitions"]) == 4
    assert all(t["from_state"] is None for t in body["transitions"])

    r = client.get("/batteries")
    assert r.status_code == 200
    summaries = r.json()
    assert len(summaries) == 4
    for s in summaries:
        assert s["twin_state"] in ("NORMAL", "MODERATE_RISK", "HIGH_RISK", "FAILURE_IMMINENT")

    battery_id = summaries[0]["battery_id"]
    r = client.get(f"/batteries/{battery_id}")
    assert r.status_code == 200
    detail = r.json()
    assert detail["battery_id"] == battery_id
    assert detail["twin"]["twin_state"] == summaries[0]["twin_state"]
    assert len(detail["transitions"]) == 1


def test_unknown_battery_returns_404(client):
    r = client.get("/batteries/DOES_NOT_EXIST")
    assert r.status_code == 404

    r = client.get("/batteries/DOES_NOT_EXIST/timeline")
    assert r.status_code == 404


def test_timeline_returns_points_ordered_by_cycle(client):
    client.post("/pipeline/simulate", json={"n_batteries": 2, "rows_per_battery": 80, "seed": 3})
    battery_id = client.get("/batteries").json()[0]["battery_id"]

    r = client.get(f"/batteries/{battery_id}/timeline")
    assert r.status_code == 200
    points = r.json()
    assert len(points) == 80
    cycles = [p["cycle"] for p in points]
    assert cycles == sorted(cycles)


def test_repeated_identical_seeded_run_produces_no_new_transitions(client):
    payload = {"n_batteries": 3, "rows_per_battery": 90, "seed": 99}
    first = client.post("/pipeline/simulate", json=payload).json()
    assert len(first["transitions"]) == 3  # first-time-seen transitions

    second = client.post("/pipeline/simulate", json=payload).json()
    assert second["transitions"] == []  # same seed -> same states -> no state change

    r = client.get("/healthz")
    assert r.json()["n_runs"] == 2


def test_simulate_request_validation_rejects_absurd_values(client):
    r = client.post("/pipeline/simulate", json={"n_batteries": 0})
    assert r.status_code == 422

    r = client.post("/pipeline/simulate", json={"n_batteries": 100000})
    assert r.status_code == 422


def test_dashboard_route_serves_html(client):
    for path in ("/", "/dashboard"):
        r = client.get(path)
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "BEACON" in r.text
        # Every endpoint the dashboard's JS calls must actually exist as a route.
        for endpoint in ("/batteries", "/pipeline/simulate", "/timeline"):
            assert endpoint in r.text
