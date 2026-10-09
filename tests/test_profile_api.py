"""GET /batteries/{id}/profile: a battery's memory accumulates across runs."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.api import profile_routes  # noqa: E402
from src.bms.api.app import app  # noqa: E402
from src.bms.api.profile_routes import forget_all, remember  # noqa: E402
from tests.test_battery_profile import _chunks, _whole  # noqa: E402
from tests.test_field_soh import _cell  # noqa: E402


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("BEACON_PROFILE_DIR", raising=False)
    forget_all()
    yield TestClient(app)
    forget_all()


def test_unknown_battery_is_404(client):
    assert client.get("/batteries/NOPE/profile").status_code == 404


def test_profile_accumulates_runs_and_matches_the_whole_log(client):
    tel, _ = _cell(n_cycles=30)
    for chunk in _chunks(tel, 3):
        remember("SYN", chunk)
    body = client.get("/batteries/SYN/profile").json()
    assert body["discharges_remembered"] == 30
    assert body["telemetry_pieces"] == 3
    assert body["soh"] == pytest.approx(_whole(tel).soh, abs=1e-12)
    assert len(body["readings"]) == 30
    # The reference is the first discharges seen, not proven new.
    assert body["confidence"] in ("MEDIUM", "LOW")
    assert body["resistance_growth"] > 1.0


def test_the_same_log_twice_is_not_counted_twice(client):
    tel, _ = _cell(n_cycles=12)
    assert remember("SYN", tel) == 12
    assert remember("SYN", tel) == 0
    assert client.get("/batteries/SYN/profile").json()["discharges_remembered"] == 12


def test_a_new_battery_is_refused_with_a_reason(client):
    tel, _ = _cell(n_cycles=3)
    remember("NEW", tel)
    body = client.get("/batteries/NEW/profile").json()
    assert body["soh"] is None
    assert body["confidence"] == "NONE"
    assert body["refusal"]


def test_profiles_persist_when_a_directory_is_set(client, monkeypatch, tmp_path):
    monkeypatch.setenv("BEACON_PROFILE_DIR", str(tmp_path))
    tel, _ = _cell(n_cycles=12)
    remember("KEEP", tel)
    assert (tmp_path / "KEEP.json").exists()
    forget_all()                                   # a restart
    assert client.get("/batteries/KEEP/profile").json()["discharges_remembered"] == 12


def test_an_unsafe_battery_id_is_never_used_as_a_file_name(monkeypatch, tmp_path):
    monkeypatch.setenv("BEACON_PROFILE_DIR", str(tmp_path))
    assert profile_routes._path("../escape") is None


def test_a_serial_run_teaches_the_profile(client):
    r = client.post("/telemetry/serial/emulate",
                    json={"battery_id": "RIG_P", "n_cycles": 2, "sample_period_s": 120.0})
    assert r.status_code == 200
    assert client.get("/batteries/RIG_P/profile").status_code == 200


def test_profiles_are_listed(client):
    tel, _ = _cell(n_cycles=12)
    remember("LISTED", tel)
    rows = client.get("/profiles").json()
    assert [r["battery_id"] for r in rows] == ["LISTED"]
    assert rows[0]["discharges_remembered"] == 12


def test_a_csv_log_can_be_ingested(client, tmp_path):
    tel, _ = _cell(n_cycles=12)
    path = tmp_path / "log.csv"
    tel[["test_time_s", "current_a", "voltage_v"]].to_csv(path, index=False)
    r = client.post("/profiles/CSV1/ingest", json={"csv_path": str(path)})
    assert r.status_code == 200, r.text
    assert r.json()["discharges_added"] == 12


def test_a_csv_without_voltage_is_rejected(client, tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("test_time_s,current_a\n0,0\n", encoding="utf-8")
    r = client.post("/profiles/CSV2/ingest", json={"csv_path": str(path)})
    assert r.status_code == 422
    assert "voltage_v" in r.json()["detail"]


def test_the_reading_says_which_discharge_it_is_from(client):
    tel, _ = _cell(n_cycles=20)
    remember("AGE", tel)
    body = client.get("/batteries/AGE/profile").json()
    assert body["at_discharge"] == 20
    assert body["discharges_since_reading"] == 0


def test_the_memory_page_is_served(client):
    r = client.get("/memory")
    assert r.status_code == 200
    assert "Battery Memory" in r.text
