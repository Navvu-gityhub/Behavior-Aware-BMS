"""Tests for the telemetry, twin and transfer endpoints.

The tests that matter most here are about what the API declines to claim:

`test_fade_prediction_is_null_and_explains_why` — the field exists precisely so a
client cannot mistake the rule-based severity index for a calibrated prediction.
Omitting it would be worse than returning null: a client would reach for
`health_index` and treat it as a forecast, which is the confusion ADR 0002
documents.

`test_a_refusal_is_a_200_carrying_its_reason` — a run that decoded frames and then
declined to compute SOH has produced useful information. Collapsing that into a
4xx would discard the telemetry, the cycle measurements and the explanation.

`test_thermal_timeline_has_no_per_cell_axis` — the unified schema carries one
pack-aggregate temperature channel, so a per-cell heatmap would mean inventing
values that were never measured.

`test_coverage_states_which_signal_map_it_used` — a coverage failure can mean the
bus lacks a channel or that the map names signals this DBC does not define. Those
need different fixes, so the response says which map produced the verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
cantools = pytest.importorskip("cantools")

from fastapi.testclient import TestClient

from src.bms.api.app import app

REPO = Path(__file__).resolve().parents[1]
REFERENCE_DBC = REPO / "src/bms/io/dbc_examples/beacon_reference_pack.dbc"
TWIZY_DBC = REPO / "src/bms/io/dbc_examples/twizy_bms_1.dbc"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def drive_log(tmp_path) -> Path:
    """A CAN log with three complete drive cycles, written by python-can."""
    can = pytest.importorskip("can")
    dbc = cantools.database.load_file(str(REFERENCE_DBC))
    message = dbc.get_message_by_name("BMS_PackState")

    path = tmp_path / "drive.asc"
    timestamp, soc = 0.0, 100.0
    with can.ASCWriter(str(path)) as writer:
        for _ in range(3):
            for current, steps in ((-10.0, 40), (5.0, 40)):
                for _ in range(steps):
                    payload = message.encode({
                        "pack_voltage": 380.0 if current < 0 else 400.0,
                        "pack_current": current,
                        "pack_soc": max(min(soc, 100.0), 0.0),
                        "pack_temp_max": 31.0,
                        "pack_temp_mean": 29.0,
                    })
                    writer.on_message_received(can.Message(
                        timestamp=timestamp, arbitration_id=message.frame_id,
                        data=payload, is_extended_id=False,
                    ))
                    soc += -2.5 if current < 0 else 2.5
                    timestamp += 10.0
    return path


# ---------------------------------------------------------------------------
# Existing routes must keep working
# ---------------------------------------------------------------------------

def test_the_pre_existing_endpoints_are_untouched(client):
    """Mounting a router must not disturb the routes that were already there."""
    assert client.get("/healthz").status_code == 200
    assert client.post(
        "/pipeline/simulate", json={"n_batteries": 3, "rows_per_battery": 60, "seed": 1}
    ).status_code == 200
    assert client.get("/batteries").status_code == 200


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_the_reference_dbc_reports_complete_coverage(client):
    body = client.get("/telemetry/coverage").json()
    assert body["status"] == "COMPLETE"
    assert body["complete"] is True
    assert body["missing_channels"] == []


def test_the_twizy_dbc_reports_its_real_gap(client):
    """Temperature and voltage, not a map mismatch."""
    body = client.get(
        "/telemetry/coverage", params={"dbc_path": str(TWIZY_DBC)}
    ).json()
    assert body["status"] == "INCOMPLETE"
    assert set(body["missing_channels"]) == {"temperature_c", "voltage_v"}
    assert "high_temp_flag" in body["explanation"]


def test_coverage_states_which_signal_map_it_used(client):
    """A missing channel and a mismatched map need different fixes."""
    body = client.get(
        "/telemetry/coverage", params={"dbc_path": str(TWIZY_DBC)}
    ).json()
    assert body["signal_map_used"] == {"v_b_current": "current_a", "v_b_soc": "soc"}


def test_a_missing_dbc_is_a_404(client):
    assert client.get(
        "/telemetry/coverage", params={"dbc_path": "/nope/absent.dbc"}
    ).status_code == 404


def test_an_unparseable_dbc_is_a_422(client, tmp_path):
    bad = tmp_path / "broken.dbc"
    bad.write_text("this is not a DBC file")
    assert client.get(
        "/telemetry/coverage", params={"dbc_path": str(bad)}
    ).status_code == 422


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def test_replay_scores_a_real_can_log(client, drive_log):
    response = client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_API"},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "SCORED"
    assert body["n_decoded"] > 0
    assert "decode" in body["stages_completed"]
    assert "score" in body["stages_completed"]
    assert "twin" in body["stages_completed"]
    assert body["cycles"], "expected cycle measurements"
    assert body["guardian"], "expected a Guardian row"


def test_replay_of_a_missing_log_is_a_404(client):
    assert client.post(
        "/telemetry/replay", json={"log_path": "/nope/absent.asc"}
    ).status_code == 404


def test_capacity_yield_is_reported(client, drive_log):
    body = client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_YIELD"},
    ).json()
    summary = body["capacity_yield"]
    assert summary["n_discharges"] == 3
    assert "usable for SOH" in summary["summary"]


def test_fade_prediction_is_null_and_explains_why(client, drive_log):
    """The field exists so a client cannot mistake triage for a forecast."""
    body = client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_FADE"},
    ).json()

    assert body["fade_prediction"] is None
    refusal = body["fade_prediction_refusal"]
    assert "promotion gate" in refusal
    assert "-0.269" in refusal  # the measured rank correlation against real fade
    # And the severity index is still present, clearly labelled.
    assert "health_index" in body["guardian"][0]


def test_the_guardian_caveat_survives_the_api_boundary(client, drive_log):
    body = client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_CAVEAT"},
    ).json()
    row = body["guardian"][0]
    assert "not a validated predictor" in row["guardian_caveat"]
    assert row["attribution_method"] in ("exact_shapley", "threshold_fallback")


def test_a_refusal_is_a_200_carrying_its_reason(client, tmp_path):
    """An incomplete DBC is a finding about the data, not a bad request."""
    can = pytest.importorskip("can")
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    message = dbc.get_message_by_name("BMS_1")

    path = tmp_path / "twizy.asc"
    with can.ASCWriter(str(path)) as writer:
        for index in range(20):
            payload = message.encode({
                "v_c_climit": 0, "v_b_current": -10.0, "v_b_soc": 80.0,
            })
            writer.on_message_received(can.Message(
                timestamp=float(index) * 10.0,
                arbitration_id=message.frame_id, data=payload,
                is_extended_id=False,
            ))

    response = client.post(
        "/telemetry/replay",
        json={"log_path": str(path), "dbc_path": str(TWIZY_DBC),
              "battery_id": "VEH_REFUSED"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "REFUSED"
    assert body["refusals"]
    assert any("temperature_c" in r for r in body["refusals"])
    assert body["guardian"] == []


# ---------------------------------------------------------------------------
# Latest and live views
# ---------------------------------------------------------------------------

def test_latest_returns_the_previous_run(client, drive_log):
    client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_LATEST"},
    )
    body = client.get("/telemetry/latest/VEH_LATEST").json()
    assert body["battery_id"] == "VEH_LATEST"
    assert body["n_decoded"] > 0


def test_latest_for_an_unknown_battery_is_a_404(client):
    response = client.get("/telemetry/latest/NEVER_SEEN")
    assert response.status_code == 404
    assert "in-process" in response.json()["detail"]


def test_live_state_labels_mode_from_current_sign(client, drive_log):
    client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_LIVE"},
    )
    body = client.get("/telemetry/live/VEH_LIVE", params={"window": 50}).json()

    assert body["n_samples"] > 0
    assert body["latest"]["mode"] in ("charge", "discharge", "rest")
    assert set(body["instrumented_channels"]) >= {
        "voltage_v", "current_a", "temperature_c", "soc"
    }
    modes = {sample["mode"] for sample in body["recent"]}
    assert modes & {"charge", "discharge"}


def test_live_state_names_uninstrumented_channels(client, tmp_path):
    """A view must not imply measurements that were never taken."""
    can = pytest.importorskip("can")
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    message = dbc.get_message_by_name("BMS_1")

    path = tmp_path / "twizy2.asc"
    with can.ASCWriter(str(path)) as writer:
        for index in range(15):
            writer.on_message_received(can.Message(
                timestamp=float(index) * 10.0,
                arbitration_id=message.frame_id,
                data=message.encode({
                    "v_c_climit": 0, "v_b_current": -8.0, "v_b_soc": 70.0,
                }),
                is_extended_id=False,
            ))

    client.post("/telemetry/replay", json={
        "log_path": str(path), "dbc_path": str(TWIZY_DBC),
        "battery_id": "VEH_PARTIAL", "require_full_coverage": False,
    })
    body = client.get("/telemetry/live/VEH_PARTIAL").json()
    assert "temperature_c" in body["uninstrumented_note"]
    assert "temperature_c" not in body["instrumented_channels"]


# ---------------------------------------------------------------------------
# Thermal timeline
# ---------------------------------------------------------------------------

def test_thermal_timeline_has_no_per_cell_axis(client, drive_log):
    """One pack-aggregate channel means no per-cell resolution to display."""
    client.post(
        "/telemetry/replay",
        json={"log_path": str(drive_log), "battery_id": "VEH_THERMAL"},
    )
    body = client.get("/telemetry/thermal/VEH_THERMAL").json()

    assert body["points"], "expected thermal points"
    assert body["n_cycles"] == 3
    for point in body["points"][:5]:
        assert set(point) == {"cycle", "phase_fraction", "temperature_c"}
        assert 0.0 <= point["phase_fraction"] <= 1.0
    assert "no" in body["resolution_note"].lower()
    assert "per-cell" in body["resolution_note"]


def test_thermal_timeline_without_a_temperature_channel_is_a_404(client, tmp_path):
    can = pytest.importorskip("can")
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    message = dbc.get_message_by_name("BMS_1")

    path = tmp_path / "twizy3.asc"
    with can.ASCWriter(str(path)) as writer:
        for index in range(15):
            writer.on_message_received(can.Message(
                timestamp=float(index) * 10.0,
                arbitration_id=message.frame_id,
                data=message.encode({
                    "v_c_climit": 0, "v_b_current": -8.0, "v_b_soc": 70.0,
                }),
                is_extended_id=False,
            ))

    client.post("/telemetry/replay", json={
        "log_path": str(path), "dbc_path": str(TWIZY_DBC),
        "battery_id": "VEH_NOTEMP", "require_full_coverage": False,
    })
    response = client.get("/telemetry/thermal/VEH_NOTEMP")
    assert response.status_code == 404
    assert "did not report one" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Twin
# ---------------------------------------------------------------------------

def test_twin_history_accumulates_across_runs(client, drive_log):
    for _ in range(2):
        client.post(
            "/telemetry/replay",
            json={"log_path": str(drive_log), "battery_id": "VEH_TWIN"},
        )
    body = client.get("/telemetry/twin/VEH_TWIN").json()

    assert body["n_snapshots"] >= 2
    assert body["snapshots"][0]["twin_state"]
    # The first snapshot has no predecessor, so it is a transition from null.
    assert body["transitions"][0]["from_state"] is None


def test_twin_history_for_an_unknown_battery_is_a_404(client):
    assert client.get("/telemetry/twin/NEVER_SEEN").status_code == 404


# ---------------------------------------------------------------------------
# Transfer validation
# ---------------------------------------------------------------------------

def test_feasibility_endpoint_reports_the_stanford_obstacle(client):
    rows = client.get("/transfer/feasibility").json()
    by_target = {row["target"]: row for row in rows}

    stanford = by_target["stanford_severson"]
    assert stanford["status"] == "PREDICTED_MARGINAL"
    assert stanford["usable_axes"] == ["internal_resistance"]

    # Temperature is blocked because the target holds it fixed.
    temperature = next(
        v for v in stanford["verdicts"] if v["axis"] == "ambient_temperature"
    )
    assert temperature["usable"] is False
    assert "nothing to act on" in temperature["reason"]


def test_feasibility_marks_itself_as_a_prediction(client):
    rows = client.get("/transfer/feasibility").json()
    assert all(row["is_prediction"] for row in rows)


def test_calce_is_feasible_on_depth_of_discharge_not_temperature(client):
    rows = client.get("/transfer/feasibility").json()
    cs2 = next(row for row in rows if row["target"] == "calce_cs2")
    assert "depth_of_discharge" in cs2["usable_axes"]
    assert "ambient_temperature" not in cs2["usable_axes"]


def test_dataset_specs_carry_their_caveats(client):
    specs = {spec["name"]: spec for spec in client.get("/datasets").json()}
    assert specs["calce_cx2_4_thermal"]["n_cells"] == 1
    assert any("n=1" in c for c in specs["calce_cx2_4_thermal"]["caveats"])
    assert all(spec["citation"] for spec in specs.values())


def test_unknown_feasibility_source_is_a_404(client):
    assert client.get(
        "/transfer/feasibility", params={"source": "nope"}
    ).status_code == 404


# ---------------------------------------------------------------------------
# Per-battery isolation of the run store
# ---------------------------------------------------------------------------

@pytest.fixture()
def twizy_log(tmp_path) -> Path:
    """A CAN log recorded against the Twizy DBC, which lacks temperature."""
    can = pytest.importorskip("can")
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    message = dbc.get_message_by_name("BMS_1")

    path = tmp_path / "twizy_drive.asc"
    with can.ASCWriter(str(path)) as writer:
        for index in range(30):
            writer.on_message_received(can.Message(
                timestamp=float(index) * 10.0,
                arbitration_id=message.frame_id,
                data=message.encode({
                    "v_c_climit": 0, "v_b_current": -10.0, "v_b_soc": 80.0,
                }),
                is_extended_id=False,
            ))
    return path


def test_two_vehicles_do_not_share_a_signal_map(client, drive_log, twizy_log):
    """Each battery's run must carry the map it was actually decoded with.

    A shared store would make /telemetry/latest report one vehicle's DBC
    mapping for another, which is exactly the coverage-versus-replay
    inconsistency this store was introduced to prevent.
    """
    client.post("/telemetry/replay", json={
        "log_path": str(twizy_log), "dbc_path": str(TWIZY_DBC),
        "battery_id": "ISO_A", "require_full_coverage": False,
    })
    client.post("/telemetry/replay", json={
        "log_path": str(drive_log), "battery_id": "ISO_B",
    })

    a = client.get("/telemetry/latest/ISO_A").json()
    b = client.get("/telemetry/latest/ISO_B").json()

    assert set(a["coverage"]["signal_map_used"]) == {"v_b_current", "v_b_soc"}
    assert "pack_temp_mean" in b["coverage"]["signal_map_used"]
    assert a["coverage"]["signal_map_used"] != b["coverage"]["signal_map_used"]
    # And their outcomes differ, because one DBC cannot drive the feature layer.
    assert a["status"] == "REFUSED"
    assert b["status"] == "SCORED"


def test_the_newest_run_replaces_the_previous_one(client, drive_log, twizy_log):
    """/telemetry/latest means latest, including the map it used."""
    client.post("/telemetry/replay", json={
        "log_path": str(twizy_log), "dbc_path": str(TWIZY_DBC),
        "battery_id": "ISO_C", "require_full_coverage": False,
    })
    first = client.get("/telemetry/latest/ISO_C").json()
    assert first["status"] == "REFUSED"

    client.post("/telemetry/replay", json={
        "log_path": str(drive_log), "battery_id": "ISO_C",
    })
    second = client.get("/telemetry/latest/ISO_C").json()

    assert second["status"] == "SCORED"
    assert second["coverage"]["signal_map_used"] != first["coverage"]["signal_map_used"]


def test_twin_history_is_kept_per_battery(client, drive_log):
    client.post("/telemetry/replay",
                json={"log_path": str(drive_log), "battery_id": "ISO_D"})
    client.post("/telemetry/replay",
                json={"log_path": str(drive_log), "battery_id": "ISO_E"})

    d = client.get("/telemetry/twin/ISO_D").json()
    e = client.get("/telemetry/twin/ISO_E").json()
    assert d["battery_id"] == "ISO_D"
    assert e["battery_id"] == "ISO_E"
    assert all(s["battery_id"] == "ISO_D" for s in d["snapshots"])
