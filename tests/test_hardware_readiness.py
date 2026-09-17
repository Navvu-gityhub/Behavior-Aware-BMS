"""Hardware-readiness gate: what must hold before a cell is connected.

**This suite does not validate hardware.** It cannot: no physical board has ever
been attached to this project. What it establishes is that the software side of
a bring-up is ready — that the wire contract is enforced, that the rig's rated
capacity reaches the calculations that divide by it, that a recorded capture
replays to identical numbers, and that the reference firmware still declares the
schema the host implements.

Detailed protocol behaviour is covered in `test_serial_telemetry.py`. This module
is deliberately a *cross-cutting gate* rather than a second copy of that: each
test here corresponds to something that would fail on a bench, and the failure
message says what it would look like with a cell attached.

The physical acceptance criteria that these tests cannot cover — current sign
under real load, SOC units, capacity against an independent reference — are in
`docs/hardware_integration.md`. Nothing here may be reported as evidence that
any of those passed.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.features.behavior_features import (  # noqa: E402
    DEFAULT_RATED_CAPACITY_AH,
    compute_behavior_flags,
)
from src.bms.telemetry import (  # noqa: E402
    FIELDS,
    REQUIRED_WIRE_FIELDS,
    SCHEMA_ID,
    SENTINEL,
    EmulatedRigSource,
    LineDecodeError,
    MemoryLineSource,
    RecordingLineSource,
    RigProfile,
    SerialPortSource,
    available_ports,
    encode_hello,
    encode_record,
    encode_status,
    parse_line,
    replay_serial_capture,
    run_serial_pipeline,
)

REPO = Path(__file__).resolve().parents[1]
FIXTURE_CAPTURE = REPO / "tests" / "fixtures" / "beacon_rig_capture.txt"
SKETCH = REPO / "firmware" / "beacon_rig" / "beacon_rig.ino"

#: The fixture rig declares this. Deliberately not 2.0 - see the note in
#: `make_serial_fixture.py`.
FIXTURE_CAPACITY_AH = 3.4

FAST_PROFILE = RigProfile(n_cycles=2, sample_period_s=120.0)


def _sample(**overrides: float) -> dict[str, float]:
    values = {"t": 10.0, "v": 3.9, "i": -1.5, "tc": 27.0, "soc": 80.0}
    values.update(overrides)
    return values


def _rig_lines(**kwargs) -> list[str]:
    return list(EmulatedRigSource(profile=FAST_PROFILE, **kwargs).lines())


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_a_conforming_hello_declares_the_channels_the_gate_needs():
    kind, header = parse_line(
        encode_hello(cell_id="RIG_01", period_ms=1000.0, capacity_ah=3.4)
    )
    assert kind == "hello"
    assert header.compatible, "reference HELLO no longer matches this parser"
    assert set(header.fields) == {spec.wire_name for spec in FIELDS}
    assert header.capacity_ah == 3.4


def test_a_conforming_data_record_maps_onto_unified_channels():
    kind, record = parse_line(encode_record(_sample()))
    assert kind == "data"
    assert set(record.values) == {spec.channel for spec in FIELDS}


def test_a_conforming_status_record_is_never_scored():
    kind, text = parse_line(encode_status("INA219 not responding on I2C"))
    assert kind == "status"
    assert text == "INA219 not responding on I2C", (
        "status text reached the operator with framing still attached"
    )


@pytest.mark.parametrize(
    "line, why",
    [
        ("BEACON1", "sentinel with no record type"),
        ("BEACON1 Q {}", "unknown record type"),
        ('BEACON1 D {"v":3.9}', "record with no timestamp"),
        ("BEACON1 D ", "empty body"),
        ('BEACON1 D {"t":0.0,"v":"abc"}', "non-numeric field"),
        ('BEACON1 D {"t":0.0,"tc":999.0}', "field outside declared range"),
        ('BEACON1 D {"t":0.0,"v":NaN}', "non-finite value"),
    ],
)
def test_malformed_frames_are_rejected_not_guessed_at(line, why):
    with pytest.raises(LineDecodeError):
        parse_line(line)


def test_unknown_wire_fields_are_ignored_rather_than_rejected():
    """Deliberate, and worth pinning so it is not "fixed" into a rejection.

    A rig may emit extra diagnostics - cell balancing state, RSSI - and a parser
    that predates them must not refuse the whole record. The coverage gate asks
    the different question of whether the channels the feature layer needs are
    present.
    """
    values = dict(_sample())
    values["rssi"] = -61.0
    kind, record = parse_line(encode_record(values))
    assert kind == "data"
    assert "rssi" not in record.values
    assert set(record.values) == {spec.channel for spec in FIELDS}


# ---------------------------------------------------------------------------
# Checksum integrity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("encoder", [encode_record, encode_status])
def test_a_corrupted_payload_is_caught_by_its_checksum(encoder):
    line = (
        encoder(_sample()) if encoder is encode_record
        else encoder("cell temperature nominal")
    )
    # Flip one character of the body, leaving the checksum intact.
    body_start = line.index(" ", len(SENTINEL) + 1) + 1
    corrupted = line[:body_start] + ("X" + line[body_start + 1:])
    with pytest.raises(LineDecodeError, match="checksum"):
        parse_line(corrupted)


@pytest.mark.parametrize("encoder", [encode_record, encode_status])
def test_a_corrupted_checksum_is_caught_too(encoder):
    line = (
        encoder(_sample()) if encoder is encode_record
        else encoder("rig started")
    )
    corrupted = line[:-2] + ("00" if not line.endswith("00") else "FF")
    with pytest.raises(LineDecodeError, match="checksum"):
        parse_line(corrupted)


def test_a_malformed_checksum_suffix_is_not_read_as_a_checksum():
    """`*ZZ` is not two hex digits, so it is body text, not a checksum.

    The record then fails on its own terms rather than on a checksum it never
    carried, which is the honest diagnosis.
    """
    with pytest.raises(LineDecodeError) as excinfo:
        parse_line("BEACON1 D {\"t\":0.0}*ZZ")
    assert "checksum" not in str(excinfo.value).lower()


def test_an_omitted_checksum_is_accepted_because_the_protocol_allows_it():
    kind, record = parse_line(encode_record(_sample(), checksum=False))
    assert kind == "data"
    assert record.values["current_a"] == -1.5


def test_a_checksum_failure_keeps_the_record_out_of_the_analysis():
    """The end-to-end property: a corrupted line must not reach the frame."""
    good = encode_record(_sample(t=0.0, i=-1.5))
    poisoned = encode_record(_sample(t=60.0, i=-99.0))
    poisoned = poisoned[:-2] + "00"

    lines = [encode_hello(cell_id="RIG", capacity_ah=2.0), good, poisoned]
    result = run_serial_pipeline(
        MemoryLineSource("checksum_rig", lines), require_full_coverage=False
    )

    assert result.stats is not None
    assert result.stats.n_rejected == 1
    assert not result.telemetry.empty
    assert -99.0 not in set(result.telemetry["current_a"]), (
        "a record that failed its checksum reached the telemetry frame"
    )


# ---------------------------------------------------------------------------
# Rated capacity (ADR 0014)
# ---------------------------------------------------------------------------


def test_c_rate_flags_scale_with_the_declared_capacity():
    """A 2.0 Ah and a 3.4 Ah device must not be scored identically.

    At 2.5 A: 1.25 C against 2.0 Ah (above the 1 C threshold) and 0.74 C
    against 3.4 Ah (below it). Same current, different verdict - which is the
    whole point of transmitting the capacity.
    """
    frame = pd.DataFrame(
        {"current_a": [-2.5], "temperature_c": [25.0], "soc": [50.0]}
    )
    small = compute_behavior_flags(frame, rated_capacity_ah=2.0)
    large = compute_behavior_flags(frame, rated_capacity_ah=3.4)

    assert small["aggressive_discharge_event"].iloc[0] == 1
    assert large["aggressive_discharge_event"].iloc[0] == 0


def test_the_pipeline_uses_the_capacity_the_rig_transmitted():
    result = replay_serial_capture(FIXTURE_CAPTURE, cell_id="FIXTURE_RIG")

    assert result.header is not None
    assert result.header.capacity_ah == FIXTURE_CAPACITY_AH
    assert result.rated_capacity_ah == FIXTURE_CAPACITY_AH
    assert "declared by the rig" in result.capacity_source
    assert result.scored, f"fixture capture did not score: {result.refusals}"


def test_a_capture_without_a_declared_capacity_is_not_scored():
    lines = [encode_hello(cell_id="NO_CAP")] + _rig_lines()[9:]
    result = run_serial_pipeline(MemoryLineSource("no_cap_rig", lines))

    assert result.rated_capacity_ah is None
    assert result.guardian.empty, "scored a capture with no C-rate basis"
    assert any("no rated capacity" in r.lower() for r in result.refusals)


def test_an_undeclared_capacity_still_reports_what_needs_no_capacity():
    """Segmentation and coulomb counting do not divide by capacity.

    Withholding them as well would hide the one result a bring-up most needs
    when the metadata is the only thing missing.
    """
    lines = [encode_hello(cell_id="NO_CAP")] + _rig_lines()[9:]
    result = run_serial_pipeline(MemoryLineSource("no_cap_rig", lines))

    assert not result.cycles.empty
    assert "segment_cycles" in result.stages_completed
    assert "score" not in result.stages_completed


def test_an_explicit_override_is_recorded_and_names_what_it_replaced():
    """Capacity mismatch must be visible, not silently resolved."""
    result = run_serial_pipeline(
        EmulatedRigSource(profile=FAST_PROFILE), cell_id="RIG_01",
        rated_capacity_ah=3.4,
    )
    assert result.rated_capacity_ah == 3.4
    assert "overridden" in result.capacity_source
    assert "2 Ah" in result.capacity_source


def test_no_serial_entry_point_silently_defaults_the_capacity():
    """Regression gate against reintroducing the ADR 0014 defect.

    The defect was not a wrong number - it was a number nobody passed and
    nothing printed. This asserts the *shape* of the fix: neither serial entry
    point may bind a default, so the value can only arrive from the wire or
    from an explicit argument.
    """
    import inspect

    from src.bms.telemetry import serial_pipeline

    for fn in (serial_pipeline.run_serial_pipeline,
               serial_pipeline.replay_serial_capture):
        parameter = inspect.signature(fn).parameters["rated_capacity_ah"]
        assert parameter.default is None, (
            f"{fn.__name__} binds a default rated capacity "
            f"({parameter.default!r}). The serial path must resolve capacity "
            f"from the rig's HELLO or from its caller, and refuse otherwise - "
            f"see ADR 0014."
        )


def test_the_batch_default_is_named_and_is_not_reachable_from_serial():
    """`DEFAULT_RATED_CAPACITY_AH` survives for the CAN and dataset paths.

    That asymmetry is deliberate and documented; this pins it so the constant
    is not quietly wired back into the serial path.
    """
    assert DEFAULT_RATED_CAPACITY_AH == 2.0

    lines = [encode_hello(cell_id="NO_CAP")] + _rig_lines()[9:]
    result = run_serial_pipeline(MemoryLineSource("no_cap_rig", lines))
    assert result.rated_capacity_ah != DEFAULT_RATED_CAPACITY_AH
    assert result.rated_capacity_ah is None


# ---------------------------------------------------------------------------
# Serial transport and capture
# ---------------------------------------------------------------------------


def test_the_live_port_source_can_be_constructed_without_pyserial():
    """Construction must not import pyserial - only `lines()` may.

    Otherwise the demo path, which needs no such dependency, would fail to
    import on a machine that has never installed it.
    """
    source = SerialPortSource(name="serial:COM5", port="COM5", duration_s=1.0)
    assert source.port == "COM5"
    assert source.baudrate == 115200


def test_port_enumeration_reports_absence_rather_than_raising():
    ports = available_ports()
    assert isinstance(ports, list)


def test_recording_wraps_any_source_and_preserves_its_identity():
    source = RecordingLineSource(
        inner=MemoryLineSource("serial:COM5", []), path="unused"
    )
    assert source.name == "serial:COM5"


def test_recording_persists_exactly_the_lines_that_were_scored(tmp_path):
    path = tmp_path / "bringup.txt"
    source = RecordingLineSource(
        inner=EmulatedRigSource(profile=FAST_PROFILE), path=path
    )
    live = run_serial_pipeline(source, cell_id="RIG_01")

    recorded = path.read_text(encoding="utf-8").splitlines()
    expected = list(EmulatedRigSource(profile=FAST_PROFILE).lines())
    assert recorded == expected, "the capture is not the bytes the run scored"
    assert live.scored


def test_the_fixture_capture_replays_deterministically():
    """The reproducibility requirement, against committed bytes.

    Two replays of the same file must agree exactly - not approximately - or a
    physical result could not be checked by anyone who was not in the room.
    """
    first = replay_serial_capture(FIXTURE_CAPTURE, cell_id="FIXTURE_RIG")
    second = replay_serial_capture(FIXTURE_CAPTURE, cell_id="FIXTURE_RIG")

    assert first.scored and second.scored
    pd.testing.assert_frame_equal(first.telemetry, second.telemetry)
    pd.testing.assert_frame_equal(first.cycles, second.cycles)
    pd.testing.assert_frame_equal(first.guardian, second.guardian)
    assert first.stats.n_accepted == second.stats.n_accepted


def test_recording_then_replaying_reproduces_the_live_numbers(tmp_path):
    path = tmp_path / "session.txt"
    live = run_serial_pipeline(
        RecordingLineSource(
            inner=EmulatedRigSource(profile=FAST_PROFILE), path=path
        ),
        cell_id="RIG_01",
    )
    replayed = replay_serial_capture(path, cell_id="RIG_01")

    assert live.scored and replayed.scored
    pd.testing.assert_frame_equal(live.cycles, replayed.cycles)
    assert (
        live.guardian.iloc[0]["health_index"]
        == replayed.guardian.iloc[0]["health_index"]
    )


def test_corrupted_lines_are_counted_and_the_accepted_fraction_is_exact():
    source = EmulatedRigSource(profile=FAST_PROFILE, corrupt_every=4)
    result = run_serial_pipeline(source, cell_id="RIG_01")

    stats = result.stats
    assert stats is not None
    assert stats.n_rejected > 0
    attempted = stats.n_accepted + stats.n_rejected
    assert stats.accepted_fraction == pytest.approx(stats.n_accepted / attempted)
    assert stats.reasons, "rejections were counted without recording why"


def test_a_capture_losing_most_of_its_lines_is_refused_not_scored():
    source = EmulatedRigSource(profile=FAST_PROFILE, corrupt_every=2)
    result = run_serial_pipeline(source, cell_id="RIG_01")

    if result.stats.accepted_fraction < 0.5:
        assert not result.scored
        assert any("below the" in r for r in result.refusals)


# ---------------------------------------------------------------------------
# Safety and integrity
# ---------------------------------------------------------------------------


def test_a_microcontroller_reset_is_refused_rather_than_sorted_away():
    lines = [encode_hello(cell_id="RESET_RIG", capacity_ah=2.0)]
    lines += [encode_record(_sample(t=t)) for t in (0.0, 60.0, 120.0)]
    lines += [encode_record(_sample(t=t)) for t in (0.0, 60.0)]

    result = run_serial_pipeline(MemoryLineSource("reset_rig", lines))

    assert not result.scored
    assert any("backwards" in r for r in result.refusals)


def test_a_missing_channel_cannot_become_valid_telemetry():
    """The NaN-as-healthy defect, asserted at the ingestion boundary."""
    fields = tuple(f for f in REQUIRED_WIRE_FIELDS if f != "tc")
    lines = [encode_hello(fields=fields, cell_id="NO_TEMP", capacity_ah=2.0)]
    lines += [
        encode_record({k: v for k, v in _sample(t=t).items() if k != "tc"})
        for t in (0.0, 60.0, 120.0)
    ]

    result = run_serial_pipeline(MemoryLineSource("no_temp_rig", lines))

    assert not result.scored
    assert result.guardian.empty
    assert any("temperature_c" in r for r in result.refusals)


def test_a_soc_fraction_is_refused_even_though_it_is_in_range():
    """0-1 lies inside the declared 0-100 range, so only a capture-level check
    can see it. Left unchecked it would flag deep discharge on every row."""
    lines = [encode_hello(cell_id="FRACTION_RIG", capacity_ah=2.0)]
    lines += [
        encode_record(_sample(t=t, soc=soc))
        for t, soc in ((0.0, 0.95), (60.0, 0.60), (120.0, 0.20))
    ]

    result = run_serial_pipeline(MemoryLineSource("fraction_rig", lines))

    assert not result.scored
    assert any("fraction" in r for r in result.refusals)


# ---------------------------------------------------------------------------
# Firmware contract (F3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirmwareContract:
    """The protocol declaration extracted from the reference sketch.

    Parsed once, structurally, so the assertions below are about the *contract*
    rather than about source formatting. Extraction failing is itself a failure:
    a sketch this parser cannot read is a sketch whose agreement with the host
    is unverified, which is precisely the state F3 existed to end.
    """

    sentinel: str
    schema_id: str
    hello_fields: tuple[str, ...]
    sample_fields: tuple[str, ...]
    hello_keys: frozenset[str]
    record_kinds: frozenset[str]


def _extract_firmware_contract() -> FirmwareContract:
    assert SKETCH.exists(), f"reference firmware sketch is missing: {SKETCH}"
    text = SKETCH.read_text(encoding="utf-8")

    def one_define(name: str) -> str:
        match = re.search(rf'#define\s+{name}\s+"([^"]+)"', text)
        assert match, f"sketch does not #define {name}"
        return match.group(1)

    hello_body = text.split("static void emitHello", 1)
    assert len(hello_body) == 2, "sketch has no emitHello function"
    hello_src = hello_body[1].split("\n}", 1)[0]

    fields_match = re.search(r'\\"fields\\":\[(.*?)\]', hello_src)
    assert fields_match, "emitHello does not emit a \"fields\" array"

    sample_body = text.split("static void emitSample", 1)
    assert len(sample_body) == 2, "sketch has no emitSample function"
    sample_src = sample_body[1].split("\n}", 1)[0]

    # The JSON branch is authoritative for field order; the compact branch is
    # asserted to agree separately.
    json_branch = sample_src.split("#else", 1)[-1]

    return FirmwareContract(
        sentinel=one_define("BEACON_SENTINEL"),
        schema_id=one_define("BEACON_SCHEMA"),
        hello_fields=tuple(re.findall(r'\\"([a-z_]+)\\"', fields_match.group(1))),
        sample_fields=tuple(re.findall(r'\\"([a-z_]+)\\":', json_branch)),
        hello_keys=frozenset(re.findall(r'\\"([a-z_]+)\\":', hello_src)),
        record_kinds=frozenset(re.findall(r'emit\(\s*"([A-Z]+)"', text)),
    )


@pytest.fixture(scope="module")
def firmware() -> FirmwareContract:
    return _extract_firmware_contract()


def test_firmware_and_host_agree_on_framing(firmware):
    assert firmware.sentinel == SENTINEL
    assert firmware.schema_id == SCHEMA_ID


def test_firmware_declares_exactly_the_hosts_field_set(firmware):
    """Set equality, so an added, removed or renamed field all fail.

    An extra field is not harmless: the parser ignores unknown fields, so the
    rig would believe it was supplying a channel the coverage gate never sees.
    """
    assert set(firmware.hello_fields) == {spec.wire_name for spec in FIELDS}


def test_firmware_sends_every_field_it_declares(firmware):
    """Declaring a channel and never emitting it is the coverage gate's blind
    spot: HELLO is trusted, so such a rig passes coverage and then has every
    record dropped by the homogeneity check - reporting a parse problem for
    what is really a firmware one."""
    assert set(firmware.sample_fields) == set(firmware.hello_fields)


def test_firmware_field_order_matches_the_schema_definition_order(firmware):
    """Order is not protocol-significant - both codecs are key-addressed - but
    a divergence means the sketch and `FIELDS` were edited independently, which
    is worth failing on while it is still cheap to reconcile."""
    assert list(firmware.hello_fields) == [spec.wire_name for spec in FIELDS]


def test_firmware_declares_its_rated_capacity(firmware):
    """Without it the host refuses to score, so an omission is a dead rig."""
    assert "capacity_ah" in firmware.hello_keys, (
        "the reference firmware's HELLO no longer declares capacity_ah; the "
        "host refuses to compute C-rate without it (ADR 0014), so this sketch "
        "would flash successfully and be refused on every capture"
    )


def test_firmware_emits_both_record_kinds(firmware):
    assert {"D", "S"} <= firmware.record_kinds


def test_firmware_hello_is_valid_json_this_parser_accepts(firmware):
    """End-to-end contract check: reconstruct the sketch's HELLO from the
    extracted declaration and put it through the real parser."""
    body = json.dumps(
        {
            "schema": firmware.schema_id,
            "fields": list(firmware.hello_fields),
            "capacity_ah": 3.4,
        },
        separators=(",", ":"),
    )
    kind, header = parse_line(f"{firmware.sentinel} HELLO {body}")

    assert kind == "hello"
    assert header.compatible
    assert set(header.channels) == {spec.channel for spec in FIELDS}
    assert header.capacity_ah == 3.4


# ---------------------------------------------------------------------------
# Opening a live port: the gap first contact with hardware exposed
# ---------------------------------------------------------------------------


def test_opening_a_live_port_is_retried_before_giving_up(monkeypatch):
    """A bridge that refuses one open and accepts the next must not be fatal.

    Found on first contact with a physical board: a USB-serial bridge that has
    just been flashed commonly refuses the next open for a second or two. The
    original implementation opened once and propagated the driver's exception,
    turning a recoverable timing condition into a hard failure.
    """
    import serial

    attempts = {"n": 0}

    class FakePort:
        def __init__(self, *args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise serial.SerialException("Cannot configure port")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def reset_input_buffer(self):
            pass

        def readline(self):
            return b""

    monkeypatch.setattr(serial, "Serial", FakePort)

    source = SerialPortSource(
        name="serial:TEST", port="COM_TEST", duration_s=0.0,
        open_retry_delay_s=0.0,
    )
    list(source.lines())

    assert attempts["n"] == 3, (
        f"expected the third open to succeed, saw {attempts['n']} attempts"
    )


def test_a_port_that_never_opens_names_what_to_check(monkeypatch):
    """The refusal after the last attempt must be actionable.

    A bare pyserial traceback tells a student nothing they can act on; the
    message has to name the monitor, the enumeration and the cable.
    """
    import serial

    def always_fails(*args, **kwargs):
        raise serial.SerialException("Cannot configure port")

    monkeypatch.setattr(serial, "Serial", always_fails)

    source = SerialPortSource(
        name="serial:TEST", port="COM_TEST", open_attempts=2,
        open_retry_delay_s=0.0,
    )
    with pytest.raises(OSError) as excinfo:
        list(source.lines())

    message = str(excinfo.value)
    assert "COM_TEST" in message
    assert "2 attempts" in message
    for hint in ("serial monitor", "enumerated", "cable", "replug"):
        assert hint in message, f"refusal does not mention {hint!r}"


def test_a_failed_run_does_not_destroy_a_previous_capture(tmp_path):
    """Recording must not truncate its destination before it has data.

    This deleted the first physical capture this project ever took: the board's
    USB bridge allowed one open per enumeration, the next run could not reopen
    the port, and the capture file had already been truncated to zero bytes by
    the time that failure surfaced. A capture is evidence.
    """
    import serial

    path = tmp_path / "bringup.txt"
    good = RecordingLineSource(
        inner=EmulatedRigSource(profile=FAST_PROFILE), path=path
    )
    list(good.lines())
    original = path.read_text(encoding="utf-8")
    assert original.strip(), "fixture capture was not written"

    class DeadPort:
        name = "serial:DEAD"

        def lines(self):
            raise serial.SerialException("Cannot configure port")
            yield  # pragma: no cover - generator marker

    doomed = RecordingLineSource(inner=DeadPort(), path=path)
    with pytest.raises(serial.SerialException):
        list(doomed.lines())

    assert path.read_text(encoding="utf-8") == original, (
        "a run that never produced a line overwrote the previous capture"
    )


def test_an_interrupted_recording_still_writes_what_arrived(tmp_path):
    """Lazy opening must not cost the partial-capture guarantee."""
    path = tmp_path / "partial.txt"

    def dies_partway():
        yield from _rig_lines()[:9]
        raise RuntimeError("cable yanked")

    from src.bms.telemetry import TextStreamSource

    source = RecordingLineSource(
        inner=TextStreamSource(name="flaky", stream=dies_partway()), path=path
    )
    with pytest.raises(RuntimeError, match="cable yanked"):
        list(source.lines())

    assert path.read_text(encoding="utf-8").count("\n") == 9
