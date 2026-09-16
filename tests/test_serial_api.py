"""API tests for the serial rig endpoints.

Kept separate from `test_telemetry_api.py`, which skips without `cantools`. The
serial endpoints need neither `cantools` nor `pyserial`, so these run in a bare
install - which matters, because the emulated path is the one that has to work
on a machine with no hardware and no optional extras.

The convention under test throughout: a refusal is a 200 carrying its reason,
not a 4xx. HTTP errors are reserved for malformed requests and missing files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.bms.api.app import app  # noqa: E402
from src.bms.telemetry import (  # noqa: E402
    SCHEMA_ID,
    EmulatedRigSource,
    RigProfile,
)

FAST_PROFILE = RigProfile(n_cycles=2, sample_period_s=120.0)


@pytest.fixture()
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_the_wire_schema_is_served_for_firmware_authors(client):
    response = client.get("/telemetry/serial/schema")
    assert response.status_code == 200
    body = response.json()

    assert body["schema_id"] == SCHEMA_ID
    assert body["sentinel"] == "BEACON1"
    assert {field["wire_name"] for field in body["fields"]} == {
        "t", "v", "i", "tc", "soc"
    }
    assert body["example_hello"].startswith("BEACON1 HELLO")
    assert body["example_record"].startswith("BEACON1 D {")
    assert body["example_record_compact"].startswith("BEACON1 D t=")


def test_the_served_schema_states_the_two_dangerous_conventions(client):
    """Sign and scale are what actually break an integration, so the contract a
    firmware author fetches must say so rather than only giving ranges."""
    body = client.get("/telemetry/serial/schema").json()
    notes = {field["wire_name"]: field["note"] for field in body["fields"]}
    assert "Negative is discharge" in notes["i"]
    assert "PERCENT" in notes["soc"]


def test_the_served_examples_are_parseable_by_the_real_parser(client):
    """A documented example that does not parse is worse than none."""
    from src.bms.telemetry import parse_line

    body = client.get("/telemetry/serial/schema").json()
    for key in ("example_hello", "example_record", "example_record_compact"):
        kind, _ = parse_line(body[key])
        assert kind in ("hello", "data"), f"{key} did not parse"


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_a_full_rig_reports_complete_coverage(client):
    response = client.get("/telemetry/serial/coverage?fields=t,v,i,tc,soc")
    assert response.status_code == 200
    assert response.json()["complete"] is True


def test_a_rig_without_a_thermistor_reports_the_consumer_that_breaks(client):
    """The same verdict the CAN coverage endpoint gives a DBC with no
    temperature signal, through the same gate."""
    response = client.get("/telemetry/serial/coverage?fields=t,v,i,soc")
    assert response.status_code == 200
    body = response.json()

    assert body["complete"] is False
    assert "temperature_c" in body["missing_channels"]
    assert "high_temp_flag" in body["explanation"]


# ---------------------------------------------------------------------------
# Emulated runs
# ---------------------------------------------------------------------------

def test_the_emulated_rig_scores_end_to_end_over_http(client):
    response = client.post(
        "/telemetry/serial/emulate",
        json={"battery_id": "API_RIG", "n_cycles": 2, "sample_period_s": 120.0},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "SCORED"
    assert body["battery_id"] == "API_RIG"
    assert body["stages_completed"] == [
        "read", "parse", "segment_cycles", "score", "twin"
    ]
    assert body["guardian"]
    assert body["cycles"]


def test_a_serial_run_reports_its_decode_statistics(client):
    """Serial is lossy, so a capture that dropped most of its lines must not be
    indistinguishable from a clean one on its output alone."""
    body = client.post(
        "/telemetry/serial/emulate",
        json={"battery_id": "API_RIG", "n_cycles": 2, "sample_period_s": 120.0},
    ).json()

    stats = body["serial"]
    assert stats is not None
    assert stats["n_accepted"] > 0
    assert stats["accepted_fraction"] == 1.0
    assert stats["n_noise"] >= 8, "the emulator's boot banner should be counted"
    assert body["rig"] and SCHEMA_ID in body["rig"]


def test_fade_prediction_is_null_on_the_serial_path_too(client):
    """The promotion gate is not routed around by a new transport (ADR 0005)."""
    body = client.post(
        "/telemetry/serial/emulate",
        json={"battery_id": "API_RIG", "n_cycles": 2, "sample_period_s": 120.0},
    ).json()

    assert body["fade_prediction"] is None
    assert body["fade_prediction_refusal"]
    assert "promotion gate" in body["fade_prediction_refusal"] or \
           "not a validated" in body["guardian"][0]["guardian_caveat"]


def test_a_lossy_capture_is_refused_with_a_200_carrying_its_reason(client):
    """A refusal is a finding about the data, not a failure of the request."""
    response = client.post(
        "/telemetry/serial/emulate",
        json={
            "battery_id": "LOSSY_RIG", "n_cycles": 2,
            "sample_period_s": 120.0, "corrupt_every": 2,
            "min_accepted_fraction": 0.9,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "REFUSED"
    assert body["guardian"] == []
    assert any("below the" in refusal for refusal in body["refusals"])
    assert body["serial"]["accepted_fraction"] < 0.9


def test_a_serial_run_is_retrievable_from_the_latest_endpoint(client):
    """Serial runs must join the existing fleet view, not a parallel one."""
    client.post(
        "/telemetry/serial/emulate",
        json={"battery_id": "FLEET_RIG", "n_cycles": 2, "sample_period_s": 120.0},
    )
    response = client.get("/telemetry/latest/FLEET_RIG")
    assert response.status_code == 200
    assert response.json()["battery_id"] == "FLEET_RIG"


def test_an_out_of_range_request_is_rejected_by_the_schema(client):
    """Bounded like the CAN live endpoint: an unbounded run never returns."""
    response = client.post(
        "/telemetry/serial/emulate", json={"n_cycles": 5000}
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def test_a_recorded_capture_replays_over_http(client, tmp_path):
    capture = EmulatedRigSource(profile=FAST_PROFILE).write_capture(
        tmp_path / "session.txt"
    )
    response = client.post(
        "/telemetry/serial/replay",
        json={"capture_path": str(capture), "battery_id": "REPLAY_RIG"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SCORED"
    assert body["guardian"]


def test_replay_and_emulation_agree_over_http(client, tmp_path):
    """Replay must be the same computation, not a parallel implementation."""
    capture = EmulatedRigSource(profile=FAST_PROFILE).write_capture(
        tmp_path / "session.txt"
    )
    replayed = client.post(
        "/telemetry/serial/replay",
        json={"capture_path": str(capture), "battery_id": "RIG_01"},
    ).json()
    emulated = client.post(
        "/telemetry/serial/emulate",
        json={
            "battery_id": "RIG_01",
            "n_cycles": FAST_PROFILE.n_cycles,
            "sample_period_s": FAST_PROFILE.sample_period_s,
        },
    ).json()

    assert replayed["n_decoded"] == emulated["n_decoded"]
    assert len(replayed["cycles"]) == len(emulated["cycles"])
    for left, right in zip(replayed["cycles"], emulated["cycles"], strict=True):
        assert left["capacity_ah"] == pytest.approx(right["capacity_ah"], rel=1e-9)


def test_a_missing_capture_file_is_a_404(client, tmp_path):
    """A missing file is a failure of the request, unlike a refusal."""
    response = client.post(
        "/telemetry/serial/replay",
        json={"capture_path": str(tmp_path / "absent.txt")},
    )
    assert response.status_code == 404


def test_a_rig_declaring_its_own_identity_is_filed_under_it(client, tmp_path):
    """With no battery_id in the request, the run must be filed under the
    identity the rig declared - not under null."""
    capture = EmulatedRigSource(
        profile=RigProfile(
            cell_id="SELF_NAMED", n_cycles=2, sample_period_s=120.0
        )
    ).write_capture(tmp_path / "session.txt")

    body = client.post(
        "/telemetry/serial/replay", json={"capture_path": str(capture)}
    ).json()
    assert body["battery_id"] == "SELF_NAMED"
    assert client.get("/telemetry/latest/SELF_NAMED").status_code == 200


# ---------------------------------------------------------------------------
# Live port
# ---------------------------------------------------------------------------

def test_a_live_capture_without_pyserial_reports_not_implemented(client):
    """pyserial is an optional extra, and its absence must not read as a broken
    port. The message points at the path that works without it."""
    try:
        import serial  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pyserial is installed; this test covers its absence")

    response = client.post(
        "/telemetry/serial/live", json={"port": "COM_NONEXISTENT"}
    )
    assert response.status_code == 501
    assert "emulate" in response.json()["detail"]


def test_a_live_capture_requires_an_explicit_port(client):
    """Never inferred: auto-selecting would read the wrong device on a host with
    a Bluetooth serial port attached."""
    response = client.post("/telemetry/serial/live", json={})
    assert response.status_code == 422
