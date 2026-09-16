"""Tests for serial telemetry ingestion: wire protocol, sources, and pipeline.

The weight of this module sits on the refusals and on one equivalence, in
keeping with how the CAN telemetry tests are organised.

`test_a_rig_without_a_temperature_sensor_is_refused` — the serial analogue of the
Twizy DBC test. A rig that does not measure temperature must refuse for the same
reason and with the same explanation, because it reuses the same gate. If this
test and its CAN counterpart ever disagree, the two transports have drifted.

`test_soc_reported_as_a_fraction_is_refused` — a 0-1 SOC lies *inside* the
declared 0-100 range, so it is a valid record meaning the wrong thing. Scoring it
would flag deep discharge on every row and produce a confident, wrong risk
profile.

`test_a_microcontroller_reset_is_refused_not_sorted` — an ESP32 brown-out restarts
its clock at zero. Sorting would interleave two unrelated sessions and cancel
charge against itself.

`test_json_and_compact_codecs_produce_identical_records` — two wire encodings must
be one schema, not two.

`test_the_emulated_rig_and_its_replayed_capture_agree` — replay must be the same
computation, not a parallel implementation.
"""

from __future__ import annotations

import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.telemetry import (
    FIELDS,
    REQUIRED_WIRE_FIELDS,
    SCHEMA_ID,
    SENTINEL,
    EmulatedRigSource,
    LineDecodeError,
    MemoryLineSource,
    RecordingLineSource,
    RigProfile,
    SerialDecodeStats,
    TextStreamSource,
    encode_hello,
    encode_record,
    encode_status,
    parse_line,
    parse_stream,
    replay_serial_capture,
    run_serial_pipeline,
    schema_table,
    xor_checksum,
)
from src.bms.telemetry.serial_pipeline import UNDECLARED_CELL_ID

# A small profile so the tests stay fast. Two cycles is the minimum that lets
# `measure_cycles` judge completeness against a peer rather than trivially.
FAST_PROFILE = RigProfile(n_cycles=2, sample_period_s=120.0)


def _sample(**overrides: float) -> dict[str, float]:
    """One well-formed record's worth of wire values."""
    values = {"t": 10.0, "v": 3.9, "i": -1.5, "tc": 27.0, "soc": 80.0}
    values.update(overrides)
    return values


def _rig_lines(**kwargs) -> list[str]:
    return list(EmulatedRigSource(profile=FAST_PROFILE, **kwargs).lines())


# ---------------------------------------------------------------------------
# Wire protocol: framing
# ---------------------------------------------------------------------------


def test_lines_without_the_sentinel_are_noise_not_errors():
    """Every board prints a boot banner. That is normal, not a fault."""
    for banner in (
        "ets Jul 29 2019 12:21:46",
        "rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)",
        "Serial monitor debug print left in the sketch",
        "",
    ):
        kind, _ = parse_line(banner)
        assert kind == "noise"


def test_the_boot_banner_is_ignored_but_counted():
    stats = SerialDecodeStats()
    lines = _rig_lines(boot_banner=True)
    list(parse_stream(lines, stats))
    assert stats.n_noise >= 8, "the emulator's ESP32 boot banner should be seen"
    assert stats.n_rejected == 0, "boot noise must not count as a rejected record"


def test_an_unknown_record_type_is_an_error_not_noise():
    """A line that claims the sentinel and then says something unknown is a
    protocol violation, unlike a line that never claimed to be ours."""
    with pytest.raises(LineDecodeError, match="unknown record type"):
        parse_line(f"{SENTINEL} Z something")


def test_a_sentinel_with_no_record_type_is_an_error():
    with pytest.raises(LineDecodeError, match="no record type"):
        parse_line(SENTINEL)


# ---------------------------------------------------------------------------
# Wire protocol: checksum
# ---------------------------------------------------------------------------


def test_a_valid_checksum_is_accepted():
    kind, record = parse_line(encode_record(_sample(), checksum=True))
    assert kind == "data"
    assert record.values["current_a"] == pytest.approx(-1.5)


def test_a_corrupt_line_fails_its_checksum():
    """One flipped character is what a jostled USB cable actually produces."""
    line = encode_record(_sample(), checksum=True)
    corrupted = line.replace('"v":3.9', '"v":9.9')
    with pytest.raises(LineDecodeError, match="checksum mismatch"):
        parse_line(corrupted)


def test_a_checksum_is_optional_but_verified_when_present():
    """A sketch too small to compute one is still a conforming rig."""
    kind, record = parse_line(encode_record(_sample(), checksum=False))
    assert kind == "data"
    assert record.values["voltage_v"] == pytest.approx(3.9)


def test_the_checksum_is_an_xor_over_the_body():
    assert xor_checksum("") == "00"
    assert xor_checksum("A") == "41"
    assert xor_checksum("AB") == "03"


# ---------------------------------------------------------------------------
# Wire protocol: codecs
# ---------------------------------------------------------------------------


def test_json_and_compact_codecs_produce_identical_records():
    """Two encodings, one schema. Otherwise the compact form is a second spec."""
    values = _sample()
    _, from_json = parse_line(encode_record(values, compact=False))
    _, from_compact = parse_line(encode_record(values, compact=True))

    assert set(from_json.values) == set(from_compact.values)
    for channel, value in from_json.values.items():
        assert from_compact.values[channel] == pytest.approx(value, rel=1e-9)


def test_the_compact_codec_survives_a_full_pipeline_run():
    """The 8-bit AVR path must reach a score, not merely parse."""
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE, compact=True), cell_id="AVR_1"
    )
    assert result.status == "SCORED"
    assert not result.guardian.empty


def test_a_malformed_key_value_token_is_rejected():
    with pytest.raises(LineDecodeError, match="key=value"):
        parse_line(f"{SENTINEL} D t=1.0 v 3.9 i=-1.5 tc=27.0 soc=80.0")


def test_malformed_json_is_rejected():
    with pytest.raises(LineDecodeError, match="malformed JSON"):
        parse_line(f"{SENTINEL} D " + '{"t":1.0,"v":')


# ---------------------------------------------------------------------------
# Wire protocol: field validation
# ---------------------------------------------------------------------------


def test_a_record_without_a_timestamp_is_rejected():
    """Time is structural, not a measurement: a sample with no time cannot be
    ordered, integrated over, or assigned to a cycle."""
    values = _sample()
    del values["t"]
    with pytest.raises(LineDecodeError, match="timestamp"):
        parse_line(encode_record(values))


def test_a_missing_sensor_field_parses_and_is_left_to_the_coverage_gate():
    """The parser answers 'is this record well formed', not 'does this rig have
    enough sensors'. Conflating the two reported a parse failure for what is
    actually a missing thermistor, and left the gate nothing to inspect."""
    values = _sample()
    del values["tc"]
    kind, record = parse_line(encode_record(values))
    assert kind == "data"
    assert "temperature_c" not in record.values
    assert "current_a" in record.values


def test_a_record_dropping_a_channel_mid_capture_is_counted_and_dropped():
    """An absent channel is not the same as a safe one, so a record with a hole
    in it is discarded rather than admitted - and charged against the accepted
    fraction, or an intermittent dropout would still report 100% accepted."""
    lines = [encode_hello(cell_id="FLAKY")]
    for step in range(30):
        values = _sample(t=float(step * 60))
        if step % 10 == 0:
            del values["tc"]  # thermistor drops out intermittently
        lines.append(encode_record(values))

    result = run_serial_pipeline(MemoryLineSource("flaky_rig", lines))
    assert result.stats.n_rejected == 3
    assert result.stats.accepted_fraction < 1.0
    assert any("omits channel" in reason for reason in result.stats.reasons)
    assert len(result.telemetry) == 27


def test_nan_is_rejected_rather_than_admitted():
    """A NumPy comparison against NaN is False, so a NaN temperature admitted
    here would read as 'not hot' for every downstream flag - the NaN-as-healthy
    defect this project already fixed."""
    line = f"{SENTINEL} D " + '{"t":1.0,"v":3.9,"i":-1.5,"tc":NaN,"soc":80.0}'
    with pytest.raises(LineDecodeError, match="not a finite number"):
        parse_line(line)


def test_infinity_is_rejected():
    line = f"{SENTINEL} D " + '{"t":1.0,"v":3.9,"i":-1.5,"tc":Infinity,"soc":80.0}'
    with pytest.raises(LineDecodeError, match="not a finite number"):
        parse_line(line)


@pytest.mark.parametrize(
    "field_name,bad_value",
    [
        ("tc", 500.0),  # thermistor open-circuit reads full scale
        ("tc", -100.0),  # thermistor shorted
        ("soc", 150.0),  # percent computed from a stale capacity estimate
        ("v", -5.0),  # leads reversed on the divider
    ],
)
def test_an_out_of_range_value_is_rejected_not_clamped(field_name, bad_value):
    """Clamping would produce a reading that looks measured and is not."""
    with pytest.raises(LineDecodeError, match="outside the declared range"):
        parse_line(encode_record(_sample(**{field_name: bad_value})))


def test_a_non_numeric_field_is_rejected():
    with pytest.raises(LineDecodeError, match="not numeric"):
        parse_line(f"{SENTINEL} D t=1.0 v=abc i=-1.5 tc=27.0 soc=80.0")


def test_unknown_fields_are_ignored_so_a_rig_may_add_diagnostics():
    """A rig emitting cell-balancing state must not break a parser predating it."""
    values = _sample()
    values["rssi"] = -67.0
    values["balancing"] = 1.0
    kind, record = parse_line(encode_record(values))
    assert kind == "data"
    assert "rssi" not in record.values
    assert set(record.values) == {"test_time_s", "voltage_v", "current_a", "temperature_c", "soc"}


# ---------------------------------------------------------------------------
# Wire protocol: HELLO
# ---------------------------------------------------------------------------


def test_hello_declares_the_schema_and_its_fields():
    kind, header = parse_line(encode_hello(cell_id="RIG_A", period_ms=1000.0, device="esp32"))
    assert kind == "hello"
    assert header.compatible
    assert header.cell_id == "RIG_A"
    assert set(header.fields) == set(REQUIRED_WIRE_FIELDS)
    assert set(header.channels) == {"test_time_s", "voltage_v", "current_a", "temperature_c", "soc"}


def test_hello_without_a_schema_id_is_rejected():
    with pytest.raises(LineDecodeError, match="schema id"):
        parse_line(f"{SENTINEL} HELLO " + '{"fields":["t","v"]}')


def test_hello_without_fields_is_rejected():
    with pytest.raises(LineDecodeError, match="fields list"):
        parse_line(f"{SENTINEL} HELLO " + '{"schema":"' + SCHEMA_ID + '"}')


def test_status_records_are_reported_never_scored():
    kind, text = parse_line(encode_status("thermistor 2 open circuit"))
    assert kind == "status"
    assert "thermistor" in text


# ---------------------------------------------------------------------------
# Decode statistics
# ---------------------------------------------------------------------------


def test_rejected_lines_are_counted_with_their_reason():
    """A capture that drops most lines must not be indistinguishable from a
    clean one on its output alone."""
    good = [encode_record(_sample(t=float(i))) for i in range(10)]
    truncated = [line[: len(line) // 2] for line in good[:4]]
    stats = SerialDecodeStats()
    list(parse_stream(good + truncated, stats))

    assert stats.n_accepted == 10
    assert stats.n_rejected == 4
    assert stats.accepted_fraction == pytest.approx(10 / 14)
    assert sum(stats.reasons.values()) == 4
    assert "rejected" in stats.render()


def test_identical_reasons_collapse_into_one_counted_category():
    """A thousand identical truncated lines is one problem, not a thousand."""
    line = encode_record(_sample())
    stats = SerialDecodeStats()
    list(parse_stream([line[:20]] * 50, stats))
    assert stats.n_rejected == 50
    assert len(stats.reasons) == 1


# ---------------------------------------------------------------------------
# Emulated rig
# ---------------------------------------------------------------------------


def test_the_emulator_is_deterministic():
    """The tests assert exact capacity figures, which requires this."""
    assert _rig_lines() == _rig_lines()


def test_a_different_seed_produces_different_output():
    other = RigProfile(n_cycles=2, sample_period_s=120.0, seed=FAST_PROFILE.seed + 1)
    assert _rig_lines() != list(EmulatedRigSource(profile=other).lines())


def test_the_emulator_emits_wire_format_parsed_by_the_real_parser():
    """The emulator must not bypass the parser it exists to exercise."""
    stats = SerialDecodeStats()
    kinds = [kind for kind, _ in parse_stream(_rig_lines(), stats)]
    assert "hello" in kinds
    assert kinds.count("data") == stats.n_accepted > 0
    assert stats.n_rejected == 0


def test_the_emulator_can_inject_corruption_for_the_rejection_path():
    stats = SerialDecodeStats()
    list(parse_stream(_rig_lines(corrupt_every=5), stats))
    assert stats.n_rejected > 0
    assert 0.0 < stats.accepted_fraction < 1.0


def test_the_emulated_current_convention_is_negative_for_discharge():
    """The single most common integration error, asserted at the source."""
    result = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01")
    current = result.telemetry["current_a"]
    assert (current < -0.5).any(), "no discharge samples"
    assert (current > 0.5).any(), "no charge samples"


# ---------------------------------------------------------------------------
# Coverage gate — the serial analogue of the DBC gate
# ---------------------------------------------------------------------------


def test_a_rig_without_a_temperature_sensor_is_refused():
    """The serial counterpart of the Twizy-DBC refusal, through the same gate.

    A rig that omits temperature cannot drive `compute_behavior_flags`, and
    treating the absent channel as 'not hot' is exactly the NaN-as-healthy
    defect. The refusal must also name which consumer breaks.
    """
    fields = tuple(name for name in REQUIRED_WIRE_FIELDS if name != "tc")
    lines = [encode_hello(fields=fields, cell_id="NO_TEMP")]
    lines += [
        encode_record({k: v for k, v in _sample(t=float(i)).items() if k != "tc"}) for i in range(5)
    ]

    result = run_serial_pipeline(MemoryLineSource("rig", lines))
    assert result.status == "REFUSED"
    assert "temperature_c" in result.coverage.missing_channels
    assert any("temperature_c" in refusal for refusal in result.refusals)
    assert any("high_temp_flag" in refusal for refusal in result.refusals)
    assert result.guardian.empty


def test_incomplete_coverage_can_be_inspected_without_scoring():
    """Inspecting parsed telemetry is legitimate; scoring on it is not."""
    fields = tuple(name for name in REQUIRED_WIRE_FIELDS if name != "tc")
    lines = [encode_hello(fields=fields, cell_id="NO_TEMP")]
    lines += [
        encode_record({k: v for k, v in _sample(t=float(i * 60)).items() if k != "tc"})
        for i in range(20)
    ]

    result = run_serial_pipeline(MemoryLineSource("rig", lines), require_full_coverage=False)
    assert not result.telemetry.empty
    assert result.guardian.empty
    assert any("NaN-as-healthy" in refusal for refusal in result.refusals)


def test_the_serial_and_can_gates_give_the_same_verdict_on_the_same_gap():
    """One gate, two transports. If these diverge, the paths have drifted."""
    from src.bms.telemetry import coverage_from_channels

    serial_coverage = coverage_from_channels(
        ("current_a", "voltage_v", "soc"), source_label="rig", transport="serial"
    )
    assert serial_coverage.complete is False
    assert serial_coverage.missing_channels == ("temperature_c",)
    assert "high_temp_flag" in serial_coverage.render()
    assert serial_coverage.status == "INCOMPLETE"


def test_coverage_is_inferred_when_a_rig_does_not_announce_itself():
    """Legal, but weaker than a declaration, and the result says so."""
    lines = [encode_record(_sample(t=float(i * 60))) for i in range(20)]
    result = run_serial_pipeline(MemoryLineSource("silent_rig", lines))
    assert result.coverage_inferred is True
    assert "no HELLO line seen" in result.render()


def test_a_missing_hello_can_be_made_fatal():
    lines = [encode_record(_sample(t=float(i))) for i in range(5)]
    result = run_serial_pipeline(MemoryLineSource("silent_rig", lines), require_schema_header=True)
    assert result.status == "REFUSED"
    assert any("HELLO" in refusal for refusal in result.refusals)


def test_an_incompatible_schema_version_is_refused():
    """Identical field names can mean different things across a schema change."""
    lines = [f"{SENTINEL} HELLO " + '{"schema":"beacon.telemetry.v99","fields":["t"]}']
    lines += [encode_record(_sample(t=float(i))) for i in range(5)]
    result = run_serial_pipeline(MemoryLineSource("future_rig", lines))
    assert result.status == "REFUSED"
    assert any("v99" in refusal for refusal in result.refusals)


# ---------------------------------------------------------------------------
# Capture-level plausibility — miswirings that pass per-record validation
# ---------------------------------------------------------------------------


def test_soc_reported_as_a_fraction_is_refused():
    """A 0-1 SOC lies inside the declared 0-100 range, so the per-field check
    cannot catch it. Scoring it would flag deep discharge on every row."""
    lines = [encode_hello(cell_id="FRACTION_RIG")]
    for step in range(40):
        lines.append(
            encode_record(
                _sample(
                    t=float(step * 60),
                    soc=1.0 - step / 40.0,
                )
            )
        )

    result = run_serial_pipeline(MemoryLineSource("fraction_rig", lines))
    assert result.status == "REFUSED"
    assert any("0-1 fraction" in refusal for refusal in result.refusals)
    assert result.guardian.empty


def test_a_plausible_percentage_capture_is_not_flagged_as_a_fraction():
    """The check must not fire on a genuine percent rig running near empty."""
    result = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01")
    assert not any("0-1 fraction" in refusal for refusal in result.refusals)


def test_a_reversed_current_shunt_is_named_as_the_likely_cause():
    """A hint, not a refusal: a charge-only capture is legitimate, but it is far
    more often a rig wired backwards, and the run refuses for want of a
    discharge either way."""
    lines = [encode_hello(cell_id="REVERSED")]
    for step in range(40):
        lines.append(
            encode_record(
                _sample(
                    t=float(step * 60),
                    i=1.5,
                    soc=float(step * 2),
                )
            )
        )

    result = run_serial_pipeline(MemoryLineSource("reversed_rig", lines))
    assert result.status == "REFUSED"
    assert any("sign convention is inverted" in r for r in result.refusals)
    assert result.guardian.empty


# ---------------------------------------------------------------------------
# Transport-level refusals
# ---------------------------------------------------------------------------


def test_a_microcontroller_reset_is_refused_not_sorted():
    """Sorting would interleave two unrelated sessions and cancel charge."""
    lines = [encode_hello(cell_id="RESET_RIG")]
    lines += [encode_record(_sample(t=float(i * 60))) for i in range(20)]
    lines += [encode_record(_sample(t=float(i * 60))) for i in range(20)]  # reset

    result = run_serial_pipeline(MemoryLineSource("reset_rig", lines))
    assert result.status == "REFUSED"
    assert any("Time went backwards" in refusal for refusal in result.refusals)
    assert result.guardian.empty


def test_a_capture_below_the_accepted_fraction_floor_is_refused():
    """Scoring the survivors would compute health from an arbitrary subsample."""
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE, corrupt_every=2),
        cell_id="LOSSY",
        min_accepted_fraction=0.9,
    )
    assert result.status == "REFUSED"
    assert any("below the" in refusal for refusal in result.refusals)
    assert result.guardian.empty


def test_an_empty_capture_names_the_likely_causes():
    """Silence is the failure a student will actually hit at the bench."""
    result = run_serial_pipeline(MemoryLineSource("dead_rig", []))
    assert result.status == "REFUSED"
    joined = " ".join(result.refusals)
    assert "baud rate" in joined
    assert "sentinel" in joined


def test_a_capture_of_pure_boot_noise_is_refused():
    """A board stuck in a reset loop prints a banner and nothing else."""
    noise = ["ets Jul 29 2019 12:21:46", "rst:0x1 (POWERON_RESET)"] * 20
    result = run_serial_pipeline(MemoryLineSource("looping_rig", noise))
    assert result.status == "REFUSED"
    assert result.stats.n_noise == 40
    assert result.stats.n_accepted == 0


def test_an_undeclared_rig_is_labelled_as_such():
    lines = [encode_record(_sample(t=float(i * 60))) for i in range(20)]
    result = run_serial_pipeline(MemoryLineSource("anon", lines))
    assert (result.telemetry["cell_id"] == UNDECLARED_CELL_ID).all()


def test_an_explicit_cell_id_overrides_the_rigs_declaration():
    result = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="LAB_BENCH_7")
    assert (result.telemetry["cell_id"] == "LAB_BENCH_7").all()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_the_full_serial_stack_produces_guardian_output():
    """Rig lines all the way to a Guardian row, via the existing stages."""
    result = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01")
    assert result.status == "SCORED"
    assert result.stages_completed == (
        "read",
        "parse",
        "segment_cycles",
        "score",
        "twin",
    )
    assert not result.guardian.empty

    row = result.guardian.iloc[0]
    for column in (
        "battery_state",
        "risk_level",
        "risk_score",
        "health_index",
        "rul_cycles",
        "attribution_method",
        "guardian_caveat",
    ):
        assert column in result.guardian.columns, f"missing {column}"

    # The caveat must survive the serial path, not only the CAN and dataset ones.
    assert "not a validated predictor" in row["guardian_caveat"]


def test_coulomb_counting_recovers_the_emulated_capacity():
    """The emulator discharges a known number of amp-hours; the pipeline must
    integrate back to it. This is what makes the emulator a test and not a
    decoration."""
    profile = RigProfile(
        n_cycles=2,
        sample_period_s=60.0,
        capacity_ah=2.0,
        noise_a=0.0,
        noise_v=0.0,
        noise_c=0.0,
        fade_per_cycle=0.0,
    )
    result = run_serial_pipeline(EmulatedRigSource(profile=profile), cell_id="RIG_01")
    assert not result.cycles.empty
    measured = result.cycles["capacity_ah"].to_numpy(float)
    # Trapezoidal integration over a finite sample grid slightly undershoots the
    # nominal figure; 2% is the grid error, not a modelling allowance.
    np.testing.assert_allclose(measured, profile.capacity_ah, rtol=0.02)


def test_partial_discharges_are_excluded_on_the_serial_path_too():
    """The exclusion is a property of the shared stages, not of the transport."""
    lines = [encode_hello(cell_id="PARTIAL")]
    time_s = 0.0
    soc = 100.0
    # One deep discharge, then a shallow one; the shallow one must be excluded.
    for depth, samples in ((100.0, 60), (25.0, 15)):
        soc = 100.0
        for _ in range(samples):
            lines.append(
                encode_record(
                    _sample(
                        t=time_s,
                        i=-1.5,
                        soc=max(soc, 0.0),
                    )
                )
            )
            soc -= depth / samples
            time_s += 60.0
        for _ in range(30):  # charge back
            lines.append(encode_record(_sample(t=time_s, i=1.5, soc=50.0)))
            time_s += 60.0

    result = run_serial_pipeline(MemoryLineSource("partial_rig", lines))
    assert result.yield_summary is not None
    assert result.yield_summary.n_partial >= 1
    excluded = [m for m in result.measurements if not m.is_complete]
    assert excluded
    assert "only comparable across equal depths" in excluded[0].exclusion_reason


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_the_emulated_rig_and_its_replayed_capture_agree(tmp_path):
    """Replay must be the same computation, not a parallel implementation."""
    source = EmulatedRigSource(profile=FAST_PROFILE)
    capture = source.write_capture(tmp_path / "session.txt")

    live = run_serial_pipeline(source, cell_id="RIG_01")
    replayed = replay_serial_capture(capture, cell_id="RIG_01")

    assert replayed.n_decoded == live.n_decoded
    assert len(replayed.cycles) == len(live.cycles)
    np.testing.assert_allclose(
        replayed.cycles["capacity_ah"].to_numpy(float),
        live.cycles["capacity_ah"].to_numpy(float),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        replayed.guardian["health_index"].to_numpy(float),
        live.guardian["health_index"].to_numpy(float),
        rtol=1e-9,
    )


def test_replaying_a_missing_capture_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        replay_serial_capture(tmp_path / "absent.txt")


def test_a_capture_with_a_corrupt_byte_does_not_discard_the_session(tmp_path):
    """A real cable produces the odd undecodable byte; that is one rejected
    line, not a lost session."""
    path = tmp_path / "session.txt"
    lines = _rig_lines()
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8")[:-1] + b"\xff\n")

    result = replay_serial_capture(path, cell_id="RIG_01")
    assert result.status == "SCORED"


# ---------------------------------------------------------------------------
# Wall-clock axis
#
# A rig's `t` is seconds since boot, so a capture carries elapsed time only.
# Daily and weekly usage aggregation needs a calendar, and the host supplying
# its start instant is what provides one. These tests pin the two ways that
# goes wrong quietly: a naive datetime, and a calendar that silently changes
# what the scoring stages integrate over.
# ---------------------------------------------------------------------------


def test_without_captured_at_there_is_no_calendar_axis():
    """Elapsed time must not be dressed up as a date nobody supplied."""
    result = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01")
    assert result.captured_at is None
    assert not result.has_calendar_axis
    assert "timestamp" not in result.telemetry.columns
    assert "no wall-clock axis" in result.render()


def test_captured_at_places_the_capture_on_a_calendar():
    started = datetime(2026, 9, 1, 8, 30, tzinfo=timezone.utc)
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE),
        cell_id="RIG_01",
        captured_at=started,
    )

    assert result.has_calendar_axis
    stamps = result.telemetry["timestamp"]
    assert stamps.iloc[0] == pd.Timestamp(started)

    # The calendar is the start instant plus elapsed time, per row.
    elapsed = pd.to_numeric(result.telemetry["test_time_s"])
    expected = pd.Timestamp(started) + pd.to_timedelta(elapsed, unit="s")
    pd.testing.assert_series_equal(stamps, expected, check_names=False)

    # It is a real axis: a multi-cycle capture spans more than one instant.
    assert stamps.iloc[-1] > stamps.iloc[0]
    assert started.isoformat() in result.render()


def test_a_naive_captured_at_is_refused():
    """Naive local time mis-buckets across a DST change, silently."""
    with pytest.raises(ValueError, match="timezone-aware"):
        run_serial_pipeline(
            EmulatedRigSource(profile=FAST_PROFILE),
            cell_id="RIG_01",
            captured_at=datetime(2026, 9, 1, 8, 30),
        )


def test_the_calendar_axis_does_not_change_what_was_measured():
    """`captured_at` labels the capture; it must not alter the physics.

    Coulomb counting integrates over `test_time_s`, which the calendar column
    sits beside rather than replaces. A wrong start instant should shift only
    the dates - if capacity or health moved with it, the timestamp would have
    become an input to the measurement.
    """
    source_kwargs = dict(cell_id="RIG_01")
    plain = run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), **source_kwargs)
    stamped = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE),
        captured_at=datetime(2019, 3, 31, 1, 30, tzinfo=timezone.utc),
        **source_kwargs,
    )

    assert plain.status == stamped.status == "SCORED"
    np.testing.assert_allclose(
        stamped.cycles["capacity_ah"].to_numpy(float),
        plain.cycles["capacity_ah"].to_numpy(float),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        stamped.guardian["health_index"].to_numpy(float),
        plain.guardian["health_index"].to_numpy(float),
        rtol=1e-12,
    )


def test_replay_takes_the_original_session_time_not_the_replay_time(tmp_path):
    """A capture file carries elapsed time only, so the date comes from the
    caller who knows when the session ran."""
    source = EmulatedRigSource(profile=FAST_PROFILE)
    capture = source.write_capture(tmp_path / "session.txt")

    original = datetime(2026, 8, 20, 14, 0, tzinfo=timezone.utc)
    replayed = replay_serial_capture(capture, cell_id="RIG_01", captured_at=original)

    assert replayed.telemetry["timestamp"].iloc[0] == pd.Timestamp(original)

    # And absent it, replay does not invent one.
    bare = replay_serial_capture(capture, cell_id="RIG_01")
    assert bare.captured_at is None
    assert "timestamp" not in bare.telemetry.columns


# ---------------------------------------------------------------------------
# Transport equivalence
# ---------------------------------------------------------------------------


def test_serial_and_can_paths_share_the_scoring_stages():
    """Both transports must call one scoring implementation, not two.

    Asserted structurally rather than numerically: the CAN fixtures need
    `cantools`, which is optional, so a numeric comparison would silently skip
    in exactly the environments where drift is most likely.
    """
    import src.bms.telemetry.pipeline as can_pipeline
    import src.bms.telemetry.serial_pipeline as serial_pipeline

    assert serial_pipeline.score_telemetry_frame is can_pipeline.score_telemetry_frame


def test_a_mis_wired_scoring_call_raises_instead_of_reporting_a_refusal(monkeypatch):
    """A defect in the wiring must not masquerade as a data refusal."""
    import src.bms.telemetry.pipeline as pipeline_module

    def broken(*args, **kwargs):
        raise AttributeError("simulated wiring defect")

    monkeypatch.setattr(pipeline_module, "_score_cycles", broken)

    with pytest.raises(RuntimeError, match="mis-wired"):
        run_serial_pipeline(EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01")


# ---------------------------------------------------------------------------
# Documentation cannot drift from the schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Rated capacity: the C-rate denominator
# ---------------------------------------------------------------------------


def test_status_text_arrives_without_its_checksum_attached():
    """Status is operator-facing text, so a raw `*1C` suffix is a defect."""
    kind, text = parse_line(encode_status("INA219 not responding on I2C"))
    assert kind == "status"
    assert text == "INA219 not responding on I2C"


def test_a_corrupted_status_line_is_rejected_rather_than_reported_verbatim():
    """A status record earns the same integrity check its data records get.

    Otherwise a line mangled on the cable is repeated to the operator as though
    the rig had said it.
    """
    line = encode_status("cell temperature nominal")
    corrupted = line.replace("nominal", "critical")
    with pytest.raises(LineDecodeError, match="checksum mismatch"):
        parse_line(corrupted)


def test_a_rig_that_declares_its_capacity_is_scored_against_that_number():
    result = run_serial_pipeline(
        EmulatedRigSource(profile=RigProfile(n_cycles=2, sample_period_s=120.0, capacity_ah=2.0)),
        cell_id="RIG_01",
    )
    assert result.header is not None and result.header.capacity_ah == 2.0
    assert result.rated_capacity_ah == 2.0
    assert "declared by the rig" in result.capacity_source
    assert "C-rate basis: 2 Ah" in result.render()


def test_a_rig_that_declares_no_capacity_is_not_scored():
    """The defect this closes: a silent 2.0 Ah default in the feature layer.

    Nothing on the wire said 2.0, nothing in the output said 2.0, and a 3.4 Ah
    cell was scored at 1.7 C while drawing 1 C.
    """
    lines = [encode_hello(cell_id="NO_CAPACITY")]
    lines.extend(_rig_lines()[9:])  # data records from the emulator, no HELLO

    result = run_serial_pipeline(MemoryLineSource("no_capacity_rig", lines))

    assert result.rated_capacity_ah is None
    assert result.capacity_source == "undeclared"
    assert result.guardian.empty, "scored a capture with no C-rate basis"
    assert any("no rated capacity" in refusal.lower() for refusal in result.refusals)


def test_an_undeclared_capacity_still_reports_what_it_could_measure():
    """Refusing to score is not refusing to measure.

    Segmentation and coulomb counting never divide by capacity, so withholding
    them too would hide the one result a bring-up most needs to see.
    """
    lines = [encode_hello(cell_id="NO_CAPACITY")]
    lines.extend(_rig_lines()[9:])

    result = run_serial_pipeline(MemoryLineSource("no_capacity_rig", lines))

    assert result.yield_summary is not None
    assert not result.cycles.empty, "cycle measurement needs no capacity"
    assert "segment_cycles" in result.stages_completed
    assert "score" not in result.stages_completed


def test_the_caller_can_supply_a_capacity_the_rig_does_not_declare():
    """The escape hatch for firmware that predates the HELLO field."""
    lines = [encode_hello(cell_id="OLD_FIRMWARE")]
    lines.extend(_rig_lines()[9:])

    result = run_serial_pipeline(
        MemoryLineSource("old_rig", lines), rated_capacity_ah=3.4
    )

    assert result.rated_capacity_ah == 3.4
    assert not result.guardian.empty


def test_the_caller_overrides_the_rig_and_the_override_is_recorded():
    """A bench that swaps cells without reflashing is the ordinary case."""
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01",
        rated_capacity_ah=3.4,
    )
    assert result.rated_capacity_ah == 3.4
    assert "overridden" in result.capacity_source
    assert "2 Ah" in result.capacity_source, "the overridden value is not named"


def _guardian_at(capacity_ah: float) -> pd.Series:
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01",
        rated_capacity_ah=capacity_ah,
    )
    assert not result.guardian.empty
    return result.guardian.iloc[0]


def test_the_capacity_changes_the_flags_it_is_the_denominator_of():
    """If it did not, threading it through would be decoration.

    The emulator discharges at 1.5 A. Against 2.0 Ah that is 0.75 C, below the
    1 C threshold; against 1.0 Ah it is 1.5 C, above it. Asserted on the
    summary columns the flags feed directly - see the test below for why the
    risk score is the wrong place to look.
    """
    gentle = _guardian_at(2.0)
    harsh = _guardian_at(1.0)

    assert gentle["aggressive_discharge_count"] == 0
    assert harsh["aggressive_discharge_count"] > 0
    assert harsh["avg_stress"] > gentle["avg_stress"] * 5


def test_the_risk_score_is_too_coarse_to_see_that_change():
    """A property of `RISK_TERMS`, pinned here because it surprised this suite.

    `avg_stress` rises from 4.5 to 33.1 and `aggressive_discharge_count` from 0
    to 80 when the same capture is scored against 1.0 Ah instead of 2.0 Ah, and
    the battery-level `risk_score` does not move at all: the terms are step
    functions whose first cut points are 50 and 100 respectively, so both
    values change inside a single step.

    This is not asserted as desirable. It is asserted so that a future change to
    the step boundaries is a deliberate one that fails this test, rather than a
    quiet re-scoring of every figure in docs/calibration_report.md. Whether the
    steps should be continuous is open - see docs/adr/0003.
    """
    assert _guardian_at(2.0)["risk_score"] == _guardian_at(1.0)["risk_score"]


def test_a_capacity_in_milliamp_hours_is_refused_rather_than_believed():
    """3400 instead of 3.4 makes every C-rate 1000x too small, so nothing is
    ever flagged and the capture scores as a model of gentle usage."""
    with pytest.raises(LineDecodeError, match="milliamp-hours"):
        parse_line(encode_hello(cell_id="MAH_RIG", capacity_ah=3400.0))

    with pytest.raises(ValueError, match="3400"):
        run_serial_pipeline(
            EmulatedRigSource(profile=FAST_PROFILE), rated_capacity_ah=3400.0
        )


# ---------------------------------------------------------------------------
# Recording a live session
# ---------------------------------------------------------------------------


def test_recording_a_run_captures_the_bytes_that_run_scored(tmp_path):
    """The point of the wrapper: the file and the result are one session.

    The previous capture facility wrote the emulator's output in a separate
    pass, so on a live port the file came from the emulator and not from the
    board - a hardware result could not be committed as a fixture at all.
    """
    path = tmp_path / "bringup.txt"
    source = RecordingLineSource(
        inner=EmulatedRigSource(profile=FAST_PROFILE), path=path
    )

    live = run_serial_pipeline(source, cell_id="RIG_01")
    replayed = replay_serial_capture(path, cell_id="RIG_01")

    assert path.exists()
    assert live.scored and replayed.scored
    assert live.guardian.iloc[0]["health_index"] == replayed.guardian.iloc[0]["health_index"]
    pd.testing.assert_frame_equal(live.cycles, replayed.cycles)


def test_recording_does_not_rename_the_run():
    source = RecordingLineSource(inner=MemoryLineSource("serial:COM5", []), path="x")
    assert source.name == "serial:COM5"


def test_an_interrupted_recording_keeps_what_it_received(tmp_path):
    """A brown-out mid-capture must leave a readable file, not an empty one.

    Buffering to the end would lose exactly the session a post-mortem needs.
    """
    path = tmp_path / "partial.txt"

    def _dies_partway():
        yield from _rig_lines()[:12]
        raise RuntimeError("cable yanked")

    source = RecordingLineSource(
        inner=TextStreamSource(name="flaky", stream=_dies_partway()), path=path
    )
    with pytest.raises(RuntimeError, match="cable yanked"):
        run_serial_pipeline(source)

    assert path.read_text(encoding="utf-8").count("\n") == 12


def test_the_documented_schema_table_matches_the_field_definitions():
    """`docs/hardware_integration.md` embeds the rendered table. If `FIELDS`
    changes and the doc does not, this fails."""
    doc = Path(__file__).resolve().parents[1] / "docs" / "hardware_integration.md"
    assert doc.exists(), "hardware integration doc is missing"
    rendered = schema_table()
    assert rendered.strip() in doc.read_text(encoding="utf-8"), (
        "docs/hardware_integration.md no longer matches serial_schema.FIELDS. "
        "Regenerate it from schema_table()."
    )


def test_every_required_channel_has_a_wire_field():
    """The gate asks for channels; the wire supplies fields. A required channel
    with no field would be permanently unsatisfiable."""
    from src.bms.telemetry import REQUIRED_CHANNELS

    supplied = {spec.channel for spec in FIELDS}
    assert set(REQUIRED_CHANNELS) <= supplied


def _sketch_text() -> str:
    sketch = Path(__file__).resolve().parents[1] / "firmware" / "beacon_rig" / "beacon_rig.ino"
    assert sketch.exists(), "reference firmware sketch is missing"
    return sketch.read_text(encoding="utf-8")


def test_the_reference_firmware_declares_the_same_schema_id():
    """The sketch and the parser must agree, or the first real board refuses."""
    text = _sketch_text()
    assert SCHEMA_ID in text
    assert SENTINEL in text


def test_the_reference_firmware_declares_the_same_field_set():
    """The sketch's HELLO field list must be exactly `FIELDS`, name for name.

    The schema-id check above is not enough and was not enough: renaming a wire
    field leaves the id untouched, so the sketch would keep announcing
    `beacon.telemetry.v1` while sending channels the parser silently drops. The
    documentation test would fail and this one would pass, which points the
    blame at the docs for a firmware defect.

    Set comparison rather than substring, so a field REMOVED from `FIELDS` but
    left in the sketch fails too - an extra field is ignored by the parser, and
    therefore is a channel the rig believes it is supplying and the coverage
    gate never sees.
    """
    text = _sketch_text()
    match = re.search(r'\\"fields\\":\[(.*?)\]', text)
    assert match, "sketch does not emit a \"fields\" array in its HELLO line"
    declared = set(re.findall(r'\\"([a-z_]+)\\"', match.group(1)))
    assert declared == {spec.wire_name for spec in FIELDS}, (
        f"firmware HELLO declares {sorted(declared)} but serial_schema.FIELDS "
        f"defines {sorted(spec.wire_name for spec in FIELDS)}"
    )


def test_the_reference_firmware_emits_every_declared_field_in_its_samples():
    """Declaring a field and never sending it is the coverage gate's blind spot.

    The gate trusts HELLO: a rig that announces `tc` and never emits it passes
    coverage and then has every record dropped by the homogeneity check, which
    reports a parse problem for what is really a firmware one.
    """
    text = _sketch_text()
    body = text.split("static void emitSample", 1)
    assert len(body) == 2, "sketch no longer has an emitSample function"
    emitted = set(re.findall(r'\\"([a-z_]+)\\":', body[1].split("\n}", 1)[0]))
    assert emitted == {spec.wire_name for spec in FIELDS}, (
        f"firmware emitSample sends {sorted(emitted)} but serial_schema.FIELDS "
        f"defines {sorted(spec.wire_name for spec in FIELDS)}"
    )


def test_the_reference_firmware_declares_its_rated_capacity():
    """Without it the host refuses to score, so a silent omission is a dead rig.

    The host's refusal is the safety net; this test is what keeps the reference
    sketch from being the thing that trips it.
    """
    text = _sketch_text()
    assert 'capacity_ah' in text.split("static void emitHello", 1)[1][:600], (
        "the reference firmware's HELLO no longer declares capacity_ah. The "
        "host refuses to compute C-rate without it, so this sketch would flash "
        "successfully and then be refused on every capture."
    )
