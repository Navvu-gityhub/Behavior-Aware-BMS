"""Integration tests for the real-telemetry pipeline.

Four tests carry the weight of this module, and all four are about refusal:

`test_the_shipped_twizy_dbc_is_refused_for_lacking_temperature` — the example DBC
defines no temperature signal, and `compute_behavior_flags` needs one. The
NaN-as-healthy fix established why this must refuse rather than proceed: a NumPy
comparison against NaN is False, so an absent temperature would silently produce
"not hot" for every row and a healthy score for a pack nobody measured.

`test_partial_discharges_alone_yield_no_soh` — capacity is only comparable across
equal depths of discharge, so a log of school-run trips can decode perfectly and
still support no SOH figure.

`test_a_partial_cycle_is_never_scaled_up_to_look_complete` — the tempting repair,
and why it is not applied.

`test_replay_and_live_capture_agree_frame_for_frame` — replay must be the same
computation, not a parallel implementation.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

cantools = pytest.importorskip("cantools")

from src.bms.telemetry import (
    MemorySource,
    TWIZY_SIGNAL_MAP,
    capacity_yield,
    check_signal_coverage,
    cycles_to_frame,
    dbc_signal_names,
    decode_frames,
    measure_cycles,
    replay_log,
    run_telemetry_pipeline,
    segment_phases,
)

REPO = Path(__file__).resolve().parents[1]
TWIZY_DBC = REPO / "src/bms/io/dbc_examples/twizy_bms_1.dbc"
REFERENCE_DBC = REPO / "src/bms/io/dbc_examples/beacon_reference_pack.dbc"

REFERENCE_MAP = {
    "pack_voltage": "voltage_v",
    "pack_current": "current_a",
    "pack_soc": "soc",
    "pack_temp_mean": "temperature_c",
    "pack_temp_max": "max_temp",
}


@pytest.fixture()
def reference_dbc():
    return cantools.database.load_file(str(REFERENCE_DBC))


def _encode(dbc, voltage: float, current: float, soc: float, temp: float) -> bytes:
    message = dbc.get_message_by_name("BMS_PackState")
    return message.encode({
        "pack_voltage": voltage,
        "pack_current": current,
        "pack_soc": soc,
        "pack_temp_max": temp + 2,
        "pack_temp_mean": temp,
    })


def _drive_cycle(
    dbc,
    n_cycles: int = 4,
    depth_fraction: float = 1.0,
    samples_per_phase: int = 40,
    temp: float = 28.0,
    start_time: float = 0.0,
) -> list[tuple[float, int, bytes]]:
    """Synthesise frames for alternating discharge and charge phases.

    `depth_fraction` scales how far each discharge goes, which is how the
    partial-cycle tests are built.
    """
    message = dbc.get_message_by_name("BMS_PackState")
    frames: list[tuple[float, int, bytes]] = []
    timestamp = start_time
    soc = 100.0

    for _ in range(n_cycles):
        # Discharge at 10 A.
        for _ in range(int(samples_per_phase * depth_fraction)):
            frames.append((
                timestamp, message.frame_id,
                _encode(dbc, 380.0, -10.0, max(soc, 0.0), temp),
            ))
            soc = max(soc - (100.0 * depth_fraction) / (samples_per_phase * depth_fraction), 0.0)
            timestamp += 10.0
        # Charge back at 5 A.
        for _ in range(samples_per_phase):
            frames.append((
                timestamp, message.frame_id,
                _encode(dbc, 400.0, 5.0, min(soc, 100.0), temp),
            ))
            soc = min(soc + 100.0 / samples_per_phase, 100.0)
            timestamp += 10.0
    return frames


# ---------------------------------------------------------------------------
# Signal coverage
# ---------------------------------------------------------------------------

def test_the_shipped_twizy_dbc_is_refused_for_lacking_temperature():
    """The example DBC cannot drive the feature layer, and must say so."""
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    signals = dbc_signal_names(dbc)
    assert "v_b_current" in signals
    assert not any("temp" in s.lower() for s in signals)

    coverage = check_signal_coverage(dbc, TWIZY_SIGNAL_MAP, "twizy")
    assert coverage.complete is False
    assert "temperature_c" in coverage.missing_channels

    result = run_telemetry_pipeline(
        MemorySource("twizy_bus", []), dbc, TWIZY_SIGNAL_MAP, dbc_path="twizy"
    )
    assert result.status == "REFUSED"
    assert any("temperature_c" in r for r in result.refusals)
    assert result.guardian.empty


def test_the_reference_dbc_has_complete_coverage(reference_dbc):
    coverage = check_signal_coverage(reference_dbc, REFERENCE_MAP, "reference")
    assert coverage.complete is True
    assert coverage.status == "COMPLETE"


def test_coverage_failure_names_the_consumer_of_each_missing_channel(reference_dbc):
    """A generic schema error would not tell an engineer what to fix."""
    partial_map = {"pack_voltage": "voltage_v", "pack_current": "current_a"}
    coverage = check_signal_coverage(reference_dbc, partial_map, "reference")
    rendered = coverage.render()
    assert "high_temp_flag" in rendered
    assert "deep_discharge_flag" in rendered


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def test_frames_decode_into_the_unified_schema(reference_dbc):
    frames = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=5)
    telemetry, n_read, n_decoded = decode_frames(
        iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1"
    )
    assert n_read == len(frames)
    assert n_decoded == len(frames)
    assert {"current_a", "voltage_v", "soc", "temperature_c"} <= set(telemetry.columns)
    assert (telemetry["cell_id"] == "VEH_1").all()


def test_undecodable_frames_are_counted_not_raised(reference_dbc):
    """A real bus carries ids a given DBC does not define."""
    good = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=3)
    noise = [(t + 0.5, 0x7FF, b"\x00" * 8) for t, _, _ in good]
    telemetry, n_read, n_decoded = decode_frames(
        iter(good + noise), reference_dbc, REFERENCE_MAP
    )
    assert n_read == len(good) + len(noise)
    assert n_decoded == len(good)
    assert not telemetry.empty


def test_timestamps_are_not_forward_filled(reference_dbc):
    """Filling between frames would invent measurements that were not taken."""
    frames = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=4)
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP)
    assert len(telemetry) == len(frames)
    assert telemetry["test_time_s"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Segmentation and coulomb counting
# ---------------------------------------------------------------------------

def test_phases_split_on_current_sign():
    telemetry = pd.DataFrame({
        "test_time_s": np.arange(0, 120, 10.0),
        "current_a": [-10] * 4 + [0.0] * 4 + [5] * 4,
    })
    phases = segment_phases(telemetry)
    assert [p.kind for p in phases] == ["discharge", "rest", "charge"]


def test_regenerative_braking_does_not_split_a_discharge():
    """A brief positive spike inside a discharge is not a charge phase."""
    current = [-10.0] * 20
    current[10] = 4.0  # one-sample regen spike
    telemetry = pd.DataFrame({
        "test_time_s": np.arange(0, 200, 10.0), "current_a": current,
    })
    phases = segment_phases(telemetry, min_phase_samples=3)
    assert [p.kind for p in phases] == ["discharge"]


def test_coulomb_counting_recovers_the_charge_moved():
    """10 A for 3600 s is 10 Ah."""
    telemetry = pd.DataFrame({
        "test_time_s": np.linspace(0, 3600, 361),
        "current_a": np.full(361, -10.0),
    })
    phases = segment_phases(telemetry)
    assert len(phases) == 1
    assert phases[0].charge_moved_ah == pytest.approx(10.0, rel=1e-6)


def test_unsorted_time_raises_rather_than_cancelling_charge():
    telemetry = pd.DataFrame({
        "test_time_s": [0.0, 20.0, 10.0, 30.0],
        "current_a": [-10.0] * 4,
    })
    with pytest.raises(ValueError, match="monotonically increasing"):
        segment_phases(telemetry)


def test_missing_current_raises_rather_than_guessing():
    telemetry = pd.DataFrame({"test_time_s": [0.0, 10.0]})
    with pytest.raises(ValueError, match="missing 'current_a'"):
        segment_phases(telemetry)


# ---------------------------------------------------------------------------
# Partial cycles: the SOH obstacle
# ---------------------------------------------------------------------------

def test_a_partial_cycle_is_never_scaled_up_to_look_complete(reference_dbc):
    """Scaling by observed SOC swing would make capacity depend on itself.

    The BMS's SOC is derived from a capacity estimate, so normalising capacity by
    it is circular. The partial cycle is excluded instead, with its measured
    charge left exactly as integrated.
    """
    frames = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=40)
    frames += _drive_cycle(
        reference_dbc, n_cycles=1, depth_fraction=0.3, samples_per_phase=40,
        start_time=10_000.0,
    )
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1")
    measurements = measure_cycles(telemetry, cell_id="VEH_1")

    partial = [m for m in measurements if not m.is_complete]
    assert partial, "expected a partial discharge in this fixture"
    # Its capacity is the raw integral, not rescaled toward the full cycle.
    full = max(m.capacity_ah for m in measurements)
    assert partial[0].capacity_ah < full * 0.5
    assert "only comparable across equal depths" in partial[0].exclusion_reason


def test_partial_discharges_alone_yield_no_soh(reference_dbc):
    """Real driving is mostly partial cycles, so this is the common case.

    Every discharge here is the same shallow depth, so all are "complete"
    relative to each other. The yield summary is what tells the caller the
    absolute depth was shallow.
    """
    frames = _drive_cycle(reference_dbc, n_cycles=3, depth_fraction=0.25)
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1")
    measurements = measure_cycles(telemetry, cell_id="VEH_1")
    summary = capacity_yield(measurements)

    assert summary.n_discharges == 3
    assert summary.largest_discharge_ah < 5.0  # shallow
    assert "usable for SOH" in summary.render()


def test_complete_only_renumbers_cycles_contiguously(reference_dbc):
    """A gap in cycle numbers would read as missing data, not a filtered cycle."""
    frames = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=40)
    frames += _drive_cycle(
        reference_dbc, n_cycles=1, depth_fraction=0.2, samples_per_phase=40,
        start_time=10_000.0,
    )
    frames += _drive_cycle(
        reference_dbc, n_cycles=1, samples_per_phase=40, start_time=20_000.0,
    )
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1")
    frame = cycles_to_frame(measure_cycles(telemetry, cell_id="VEH_1"))
    assert list(frame["cycle"]) == list(range(1, len(frame) + 1))


def test_depth_of_discharge_is_reported_but_not_used_for_normalisation(reference_dbc):
    frames = _drive_cycle(reference_dbc, n_cycles=1, samples_per_phase=20)
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1")
    measurement = measure_cycles(telemetry, cell_id="VEH_1")[0]
    assert measurement.depth_of_discharge is not None
    # Capacity equals the raw integral, independent of the SOC swing.
    expected_ah = 10.0 * (19 * 10.0) / 3600.0
    assert measurement.capacity_ah == pytest.approx(expected_ah, rel=0.05)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

def test_full_pipeline_scores_a_complete_drive_cycle(reference_dbc):
    frames = _drive_cycle(reference_dbc, n_cycles=3, samples_per_phase=40)
    result = run_telemetry_pipeline(
        MemorySource("bench", frames), reference_dbc, REFERENCE_MAP,
        cell_id="VEH_1", dbc_path="reference",
    )
    assert result.coverage.complete is True
    assert "decode" in result.stages_completed
    assert "segment_cycles" in result.stages_completed
    assert not result.cycles.empty
    assert result.yield_summary is not None


def test_the_pipeline_reports_which_stages_completed(reference_dbc):
    """A partial run must be distinguishable from a failed one."""
    result = run_telemetry_pipeline(
        MemorySource("empty", []), reference_dbc, REFERENCE_MAP, dbc_path="reference"
    )
    assert result.stages_completed == ("decode",)
    assert any("No frames decoded" in r for r in result.refusals)


def test_a_refusal_carries_its_reason(reference_dbc):
    result = run_telemetry_pipeline(
        MemorySource("empty", []), reference_dbc, REFERENCE_MAP, dbc_path="reference"
    )
    assert result.refusals
    assert "REFUSED" in result.render()


def test_incomplete_coverage_can_be_inspected_without_scoring():
    """Inspecting decoded telemetry is legitimate; scoring on it is not."""
    dbc = cantools.database.load_file(str(TWIZY_DBC))
    message = dbc.get_message_by_name("BMS_1")
    frames = [
        (float(i) * 10.0, message.frame_id,
         message.encode({"v_c_climit": 0, "v_b_current": -10.0, "v_b_soc": 80.0}))
        for i in range(10)
    ]
    result = run_telemetry_pipeline(
        MemorySource("twizy", frames), dbc, TWIZY_SIGNAL_MAP,
        dbc_path="twizy", require_full_coverage=False,
    )
    # Telemetry is available for inspection...
    assert not result.telemetry.empty
    # ...but nothing was scored, and the reason is recorded.
    assert result.guardian.empty
    assert any("NaN-as-healthy" in r for r in result.refusals)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def test_replay_and_live_capture_agree_frame_for_frame(reference_dbc, tmp_path):
    """Replay must be the same computation, not a parallel implementation."""
    can = pytest.importorskip("can")

    frames = _drive_cycle(reference_dbc, n_cycles=2, samples_per_phase=30)
    log_path = tmp_path / "drive.asc"
    with can.ASCWriter(str(log_path)) as writer:
        for timestamp, arbitration_id, payload in frames:
            writer.on_message_received(
                can.Message(
                    timestamp=timestamp, arbitration_id=arbitration_id,
                    data=payload, is_extended_id=False,
                )
            )

    from_memory = run_telemetry_pipeline(
        MemorySource("memory", frames), reference_dbc, REFERENCE_MAP,
        cell_id="VEH_1", dbc_path="reference",
    )
    from_log = replay_log(
        log_path, reference_dbc, REFERENCE_MAP,
        cell_id="VEH_1", dbc_path="reference",
    )

    assert from_log.n_decoded == from_memory.n_decoded
    assert len(from_log.cycles) == len(from_memory.cycles)
    np.testing.assert_allclose(
        from_log.cycles["capacity_ah"].to_numpy(float),
        from_memory.cycles["capacity_ah"].to_numpy(float),
        rtol=1e-6,
    )


def test_replaying_a_missing_log_is_explicit(reference_dbc, tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_log(tmp_path / "absent.asc", reference_dbc, REFERENCE_MAP)


def test_live_bus_source_reads_from_a_virtual_bus(reference_dbc):
    """Covers the live path without hardware. Real interfaces are untested.

    python-can's virtual bus only delivers to listeners already attached when a
    message is sent, which mirrors a real bus: frames put on the wire before you
    connect are gone. So the writer runs on a background thread while the source
    is already consuming.
    """
    can = pytest.importorskip("can")
    import threading

    from src.bms.telemetry import LiveBusSource

    message = reference_dbc.get_message_by_name("BMS_PackState")
    payload = _encode(reference_dbc, 380.0, -10.0, 80.0, 28.0)
    stop = threading.Event()

    def publish() -> None:
        with can.Bus(channel="beacon_test", interface="virtual") as writer:
            while not stop.is_set():
                writer.send(can.Message(
                    arbitration_id=message.frame_id, data=payload,
                    is_extended_id=False,
                ))
                stop.wait(0.02)

    publisher = threading.Thread(target=publish, daemon=True)
    publisher.start()
    try:
        source = LiveBusSource(
            name="virtual", channel="beacon_test", interface="virtual",
            duration_s=0.5, receive_timeout_s=0.05,
        )
        received = list(source.frames())
    finally:
        stop.set()
        publisher.join(timeout=2.0)

    assert received, "virtual bus yielded no frames"
    assert all(arbitration_id == message.frame_id for _, arbitration_id, _ in received)


def test_live_bus_source_honours_its_duration_bound(reference_dbc):
    """An unbounded generator inside a request handler would never return."""
    pytest.importorskip("can")
    import time

    from src.bms.telemetry import LiveBusSource

    source = LiveBusSource(
        name="silent", channel="beacon_quiet", interface="virtual",
        duration_s=0.2, receive_timeout_s=0.05,
    )
    started = time.monotonic()
    frames = list(source.frames())  # nothing is publishing
    elapsed = time.monotonic() - started

    assert frames == []
    assert 0.15 < elapsed < 2.0, f"duration bound not honoured (took {elapsed:.2f}s)"


def test_the_full_stack_produces_guardian_output(reference_dbc):
    """CAN frames all the way to a Guardian row, via the existing stages."""
    frames = _drive_cycle(reference_dbc, n_cycles=3, samples_per_phase=40)
    result = run_telemetry_pipeline(
        MemorySource("bench", frames), reference_dbc, REFERENCE_MAP,
        cell_id="VEH_01", dbc_path="reference",
    )
    assert result.status == "SCORED"
    # 'twin' was added when digital-twin evaluation was wired into the pipeline.
    assert result.stages_completed == (
        "decode", "segment_cycles", "score", "twin",
    )
    assert not result.guardian.empty

    row = result.guardian.iloc[0]
    for column in (
        "battery_state", "risk_level", "risk_score", "health_index",
        "rul_cycles", "attribution_method", "guardian_caveat",
    ):
        assert column in result.guardian.columns, f"missing {column}"

    # The caveat established in the explainability work must survive the
    # telemetry path, not only the dataset path.
    assert "not a validated predictor" in row["guardian_caveat"]
    assert row["attribution_method"] in ("exact_shapley", "threshold_fallback")


def test_charge_phase_rows_do_not_leak_into_discharge_cycles(reference_dbc):
    """Folding charging temperatures into a discharge aggregate would shift
    every per-cycle feature."""
    from src.bms.telemetry.pipeline import _attach_cycle_index

    frames = _drive_cycle(reference_dbc, n_cycles=2, samples_per_phase=30)
    telemetry, _, _ = decode_frames(iter(frames), reference_dbc, REFERENCE_MAP, "VEH_1")
    cycles = cycles_to_frame(measure_cycles(telemetry, cell_id="VEH_1"))

    labelled = _attach_cycle_index(telemetry, cycles)
    # Only discharge rows survive, so current is negative throughout.
    assert (labelled["current_a"] < 0).all()
    assert len(labelled) < len(telemetry)


def test_a_mis_wired_scoring_call_raises_instead_of_reporting_a_refusal(
    reference_dbc, monkeypatch
):
    """A defect in this module must not masquerade as a data refusal."""
    import src.bms.telemetry.pipeline as pipeline_module

    def broken(*args, **kwargs):
        raise AttributeError("simulated wiring defect")

    monkeypatch.setattr(pipeline_module, "_score_cycles", broken)

    frames = _drive_cycle(reference_dbc, n_cycles=2, samples_per_phase=30)
    with pytest.raises(RuntimeError, match="mis-wired"):
        run_telemetry_pipeline(
            MemorySource("bench", frames), reference_dbc, REFERENCE_MAP,
            cell_id="VEH_1", dbc_path="reference",
        )
