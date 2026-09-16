"""BEACON serial telemetry wire protocol: line format, schema, and parsing.

This module defines the contract between a microcontroller rig and the BEACON
pipeline. It is deliberately a *text line protocol* rather than a binary one,
and deliberately transport-agnostic: it describes what a line looks like, not
what carries it. USB serial is the intended transport for the college rig, but
a line is a line, so the same parser serves a file, a socket, a pipe, or a
recorded capture with no change.

Why a separate schema module at all
-----------------------------------
The CAN path already has a schema authority: the DBC file, plus
`sources.check_signal_coverage`, which refuses before decoding when the bus
cannot supply a channel the feature layer needs. A sensor rig has no DBC. Left
implicit, its "schema" would be whatever the firmware happened to print on the
day of the demo, and a renamed field would surface as a silently absent channel
downstream — exactly the NaN-as-healthy defect this project already fixed once.

So the rig gets an explicit schema, declared on the wire, checked before any
record is scored.

Field definitions are the authority
-----------------------------------
`FIELDS` below is the single source of truth for the wire format. The firmware
reference sketch, the documentation table in `docs/hardware_integration.md`,
and the parser all derive from it, so they cannot drift apart silently. Units
are part of the definition, because the two most likely integration bugs on a
student rig are a sign convention and a unit scale:

- **Current sign.** `cycles.segment_phases` treats negative current as
  discharge. A rig that reports discharge as positive produces a stream where
  every discharge looks like a charge, no discharge phase is ever found, and the
  pipeline reports "no complete discharge cycle" while the rig appears to be
  working perfectly. `RANGE` checks cannot catch this, so it is stated loudly in
  the field table and asserted by the reference firmware's comment block.

- **SOC scale.** `features.behavior_features` compares SOC against 20.0 and
  90.0, i.e. it expects **percent, 0-100**. A rig emitting a 0-1 fraction would
  make `deep_discharge_flag` fire on every single row and `high_soc_flag` never
  fire.

Neither of those is catchable by a per-field range check, and it is worth being
precise about why, because the obvious reading of the table below is that the
ranges cover it. They do not. A 0-1 SOC fraction lies *inside* [0, 100], and an
inverted current sign lies inside the symmetric current range. Both produce
perfectly valid records that mean the wrong thing.

They are caught one level up instead, by `serial_pipeline._plausibility_refusals`,
which looks at the shape of a whole capture rather than at one record: an SOC
channel that never exceeds 1.0 while sweeping most of its observed span is a
fraction mislabelled as a percentage, and a capture with charge phases but no
discharge phases is the signature of a reversed shunt. A per-record check cannot
see either pattern. The field ranges here are a gate against a *disconnected or
unpowered* sensor, which is a different failure and is what they do catch.

Why ranges reject rather than clamp
-----------------------------------
An out-of-range value is evidence that the reading is not what the schema says
it is: a disconnected thermistor floats, an unpowered INA219 reads zero, an ADC
with a bad reference reads full-scale. Clamping such a value to the nearest
plausible number produces a reading that looks measured and is not. Rejecting
the record keeps it out of the aggregate and keeps it countable, which is what
`SerialDecodeStats` exists to report.

Why a bad line is counted rather than raised
--------------------------------------------
Serial is a lossy transport. A USB cable jostled during a demo drops bytes; an
ESP32 that browns out mid-line emits a truncated record; every board prints boot
noise on reset before the sketch's first line. Aborting the capture on the first
malformed line would make the pipeline unusable on real hardware for a reason
that has nothing to do with the battery.

The compromise is the same one `pipeline.decode_frames` already makes for CAN
frames a DBC does not define: skip and count. `SerialDecodeStats` reports the
counts, and `serial_pipeline` refuses when the accepted fraction is too low to
trust rather than scoring whatever survived.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

#: Schema identifier the rig announces and the parser validates against.
#: Bump the minor version for an additive change (a new optional field); bump
#: the major version for anything that changes the meaning of an existing one.
SCHEMA_ID = "beacon.telemetry.v1"

#: Every BEACON line starts with this token. Anything else on the wire is
#: ignored as noise, which is what makes the protocol survive an ESP32's boot
#: banner, a bootloader's ROM message, and any `Serial.println` debugging the
#: student leaves in the sketch.
SENTINEL = "BEACON1"

#: Record type markers.
RECORD_HELLO = "HELLO"   # schema announcement, sent once at rig start-up
RECORD_DATA = "D"        # one telemetry sample
RECORD_STATUS = "S"      # device-side status or fault text, never scored

#: Optional trailing XOR checksum, e.g. ``...soc=100.0*3F``. Optional because a
#: constrained sketch may not want to compute it; validated strictly whenever
#: present, because a checksum that is checked only sometimes is worse than none.
_CHECKSUM_RE = re.compile(r"\*([0-9A-Fa-f]{2})\s*$")

#: Bounds on a declared rated capacity, in amp-hours. A gate against the one
#: unit mistake this field invites: a 3400 mAh cell declared as `3400` rather
#: than `3.4`. The lower bound admits a coin cell; the upper sits above any
#: pack a bench rig measures (a large EV pack is on the order of 200 Ah) and
#: below the mAh figures cells are actually sold by, which is what makes the
#: check bite. A bound of 10,000 would have admitted every mAh rating in
#: circulation and caught nothing.
#:
#: It cannot catch every case, and the limit is worth stating: a 900 mAh cell
#: declared as `900` is rejected, but a 500 mAh cell declared as `500` is too,
#: while nothing distinguishes a genuine 500 Ah bank from that mistake. The
#: gate narrows the window; the firmware comment on `NOMINAL_CAPACITY_AH` is
#: what closes it.
MIN_CAPACITY_AH = 0.01
MAX_CAPACITY_AH = 500.0


@dataclass(frozen=True)
class FieldSpec:
    """One wire field: its name, the unified-schema channel it feeds, and its units."""

    wire_name: str
    channel: str
    unit: str
    minimum: float
    maximum: float
    required: bool
    note: str = ""

    def validate(self, value: float) -> str:
        """Return an empty string if acceptable, else the reason it is not."""
        if not math.isfinite(value):
            # NaN and infinity are rejected explicitly. A NumPy comparison
            # against NaN evaluates False, so a NaN temperature admitted here
            # would read as "not hot" for every downstream flag - the exact
            # defect this project fixed and does not intend to reintroduce
            # through a new ingestion path.
            return f"{self.wire_name}={value!r} is not a finite number"
        if not (self.minimum <= value <= self.maximum):
            return (
                f"{self.wire_name}={value:g} {self.unit} is outside the declared "
                f"range [{self.minimum:g}, {self.maximum:g}] {self.unit}"
            )
        return ""


#: The wire schema. Authoritative for the firmware, the parser and the docs.
#:
#: Ranges are deliberately generous: they are a sanity gate against a
#: disconnected sensor or a unit-scale mistake, not a specification of any
#: particular cell chemistry. A rig outside these bounds is far more likely
#: miswired than exotic.
FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        wire_name="t", channel="test_time_s", unit="s",
        minimum=0.0, maximum=1e9, required=True,
        note="Seconds since rig start. Must increase monotonically within a "
             "capture; the pipeline refuses on a time reversal rather than "
             "integrating charge against itself.",
    ),
    FieldSpec(
        wire_name="v", channel="voltage_v", unit="V",
        minimum=0.0, maximum=1000.0, required=True,
        note="Terminal voltage of the cell or pack under measurement.",
    ),
    FieldSpec(
        wire_name="i", channel="current_a", unit="A",
        minimum=-2000.0, maximum=2000.0, required=True,
        note="SIGNED. Negative is discharge, positive is charge. This is the "
             "single most common integration error: a rig with the opposite "
             "convention yields no discharge phases and therefore no SOH, "
             "while appearing to stream perfectly.",
    ),
    FieldSpec(
        wire_name="tc", channel="temperature_c", unit="degC",
        minimum=-40.0, maximum=150.0, required=True,
        note="Cell surface temperature. Required: the feature layer computes "
             "high_temp_flag from it and refuses rather than assuming ambient.",
    ),
    FieldSpec(
        wire_name="soc", channel="soc", unit="%",
        minimum=0.0, maximum=100.0, required=True,
        note="PERCENT, 0-100 - not a 0-1 fraction. The feature layer compares "
             "against 20.0 and 90.0, so a 0-1 rig would flag deep discharge on "
             "every row and never flag high SOC. A fraction passes this range "
             "check (it lies inside it); the capture-level plausibility check "
             "in serial_pipeline is what catches it.",
    ),
)

#: Fields keyed by wire name, for parsing.
FIELDS_BY_WIRE: Mapping[str, FieldSpec] = {spec.wire_name: spec for spec in FIELDS}

#: Wire name -> unified schema channel. The serial analogue of a DBC signal map.
SERIAL_CHANNEL_MAP: Mapping[str, str] = {
    spec.wire_name: spec.channel for spec in FIELDS
}

#: Wire names a fully-instrumented rig emits. This is the set the coverage gate
#: needs in order to score; it is NOT a per-record parse requirement. See
#: `_parse_data` for why those are deliberately different questions.
REQUIRED_WIRE_FIELDS: tuple[str, ...] = tuple(
    spec.wire_name for spec in FIELDS if spec.required
)

#: The one field required on every data record regardless of a rig's sensor
#: complement, because it is structural rather than a measurement.
TIME_FIELD = "t"


class LineDecodeError(ValueError):
    """One line could not be turned into a record. Carries why, for counting."""


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TelemetryRecord:
    """One validated telemetry sample, in unified-schema channel names.

    `values` is keyed by unified channel (``current_a``, ``voltage_v``, ...)
    rather than by wire name, so everything downstream of this module speaks the
    same vocabulary the CAN path already speaks.
    """

    values: Mapping[str, float]
    raw_line: str = ""

    def as_row(self, cell_id: str) -> dict[str, Any]:
        """Render as one row of the unified telemetry frame."""
        row: dict[str, Any] = dict(self.values)
        row["cell_id"] = cell_id
        return row


@dataclass(frozen=True)
class SchemaHeader:
    """A rig's declared schema, parsed from its HELLO line.

    This is the serial equivalent of loading a DBC: it states what the device
    will send *before* it sends any of it, so coverage can be checked ahead of
    decoding rather than inferred afterwards from whatever happened to arrive.
    """

    schema_id: str
    fields: tuple[str, ...]
    cell_id: str | None = None
    period_ms: float | None = None
    device: str | None = None
    firmware: str | None = None
    #: Rated capacity of the cell or pack under measurement, in amp-hours.
    #:
    #: Optional on the wire, but its absence is *not* free: the feature layer
    #: defines `aggressive_discharge_event` and `fast_charge_flag` by C-rate,
    #: which is current divided by this number. Without it there is no C-rate,
    #: and `serial_pipeline` refuses rather than assuming one - see the note
    #: there on why a default was removed.
    #:
    #: It rides in HELLO rather than on every data record because it is a
    #: property of the cell, not of the sample, and a rig that changes cells
    #: mid-capture is already re-announcing itself.
    capacity_ah: float | None = None

    @property
    def channels(self) -> tuple[str, ...]:
        """Unified channels this rig claims it can supply."""
        return tuple(
            sorted(
                {
                    SERIAL_CHANNEL_MAP[name]
                    for name in self.fields
                    if name in SERIAL_CHANNEL_MAP
                }
            )
        )

    @property
    def compatible(self) -> bool:
        """Whether the declared schema id is one this parser implements."""
        return self.schema_id == SCHEMA_ID

    def render(self) -> str:
        parts = [f"schema={self.schema_id}", f"fields={list(self.fields)}"]
        if self.device:
            parts.append(f"device={self.device}")
        if self.firmware:
            parts.append(f"firmware={self.firmware}")
        if self.cell_id:
            parts.append(f"cell_id={self.cell_id}")
        if self.period_ms is not None:
            parts.append(f"period_ms={self.period_ms:g}")
        if self.capacity_ah is not None:
            parts.append(f"capacity_ah={self.capacity_ah:g}")
        return "  ".join(parts)


@dataclass
class SerialDecodeStats:
    """What a capture actually contained, including everything discarded.

    A capture that silently drops 90% of its lines and scores the rest is
    indistinguishable, from its output alone, from a clean one. These counts are
    what make that visible, and `serial_pipeline` refuses on a low accepted
    fraction rather than scoring the remainder.
    """

    n_lines: int = 0
    n_noise: int = 0          # no sentinel: boot banners, stray prints
    n_status: int = 0         # device status records, never scored
    n_accepted: int = 0
    n_rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n_beacon(self) -> int:
        """Lines that claimed to be BEACON records, valid or not."""
        return self.n_accepted + self.n_rejected + self.n_status

    @property
    def accepted_fraction(self) -> float:
        """Accepted data records over lines that claimed to be data records."""
        attempted = self.n_accepted + self.n_rejected
        return self.n_accepted / attempted if attempted else 0.0

    def record_rejection(self, reason: str) -> None:
        self.n_rejected += 1
        # Collapse to the reason's first clause so a thousand identical
        # truncated lines produce one counted category, not a thousand strings.
        key = reason.split(":")[0].strip()[:80]
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def render(self) -> str:
        lines = [
            f"{self.n_accepted}/{self.n_accepted + self.n_rejected} data records "
            f"accepted ({self.accepted_fraction:.0%}); "
            f"{self.n_noise} non-BEACON line(s) ignored; "
            f"{self.n_status} status record(s)."
        ]
        for reason, count in sorted(
            self.reasons.items(), key=lambda item: -item[1]
        ):
            lines.append(f"  rejected x{count}: {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------

def xor_checksum(payload: str) -> str:
    """NMEA-style XOR of every byte, as two uppercase hex digits.

    Chosen because it is three lines of C on any microcontroller and catches
    the single-byte corruption and truncation that USB serial actually
    produces. It is an integrity check against a noisy cable, not a
    cryptographic one, and is documented as such.
    """
    checksum = 0
    for byte in payload.encode("utf-8"):
        checksum ^= byte
    return f"{checksum:02X}"


def _split_checksum(body: str) -> tuple[str, str]:
    """Split a record body from its optional ``*HH`` suffix."""
    match = _CHECKSUM_RE.search(body)
    if not match:
        return body, ""
    return body[: match.start()], match.group(1).upper()


# ---------------------------------------------------------------------------
# Codecs
# ---------------------------------------------------------------------------

def _decode_json_body(body: str) -> dict[str, float]:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LineDecodeError(f"malformed JSON: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise LineDecodeError("malformed JSON: expected an object")
    return parsed


def _decode_keyvalue_body(body: str) -> dict[str, Any]:
    """Parse ``t=0.0 v=4.15 i=-1.5`` - the compact form for tiny sketches.

    Offered because an 8-bit AVR with 2 KB of RAM can emit this with `print`
    calls and no JSON library, and because it is legible in a serial monitor
    during bring-up, when the student most needs to see what the board is
    actually sending.
    """
    values: dict[str, Any] = {}
    for token in body.split():
        if "=" not in token:
            raise LineDecodeError(f"malformed key=value token {token!r}")
        key, _, raw = token.partition("=")
        values[key.strip()] = raw.strip()
    if not values:
        raise LineDecodeError("empty record body")
    return values


def _coerce_float(wire_name: str, raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise LineDecodeError(
            f"field {wire_name}={raw!r} is not numeric"
        ) from exc


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

def parse_line(line: str) -> tuple[str, Any]:
    """Parse one wire line into ``(kind, payload)``.

    Returns one of:

    - ``("noise", raw_line)``   - no sentinel; ignored, not an error
    - ``("hello", SchemaHeader)``
    - ``("status", text)``
    - ``("data", TelemetryRecord)``

    Raises `LineDecodeError` for a line that claims to be a BEACON record and
    is not a usable one. The caller decides whether to count it or stop; the
    pipeline counts.
    """
    stripped = line.strip()
    if not stripped:
        return "noise", line

    if not stripped.startswith(SENTINEL):
        # Boot banners, bootloader chatter, leftover debug prints. Not an
        # error: a board that prints its own banner is behaving normally.
        return "noise", line

    remainder = stripped[len(SENTINEL):].strip()
    if not remainder:
        raise LineDecodeError("sentinel with no record type")

    kind, _, body = remainder.partition(" ")
    kind = kind.strip()
    body = body.strip()

    if kind == RECORD_STATUS:
        # Status text carries a checksum for the same reason data does - the
        # reference firmware appends one - so it is split off and verified here
        # rather than left attached. Leaving it attached printed operator-facing
        # fault text with a trailing `*1C` in the project's own documented
        # example, and meant a corrupted status line was reported verbatim as
        # though the rig had said it.
        text, checksum = _split_checksum(body)
        _verify_checksum(text, checksum)
        return "status", text.strip()

    if kind == RECORD_HELLO:
        return "hello", _parse_hello(body)

    if kind == RECORD_DATA:
        return "data", _parse_data(body, raw_line=stripped)

    raise LineDecodeError(f"unknown record type {kind!r}")


def _parse_hello(body: str) -> SchemaHeader:
    body, checksum = _split_checksum(body)
    _verify_checksum(body, checksum)

    payload = _decode_json_body(body.strip())
    schema_id = str(payload.get("schema", "")).strip()
    if not schema_id:
        raise LineDecodeError("HELLO does not declare a schema id")

    raw_fields = payload.get("fields")
    if not isinstance(raw_fields, list) or not raw_fields:
        raise LineDecodeError("HELLO does not declare a non-empty fields list")

    period = payload.get("period_ms")
    return SchemaHeader(
        schema_id=schema_id,
        fields=tuple(str(name) for name in raw_fields),
        cell_id=(str(payload["cell_id"]) if payload.get("cell_id") else None),
        period_ms=(float(period) if period is not None else None),
        device=(str(payload["device"]) if payload.get("device") else None),
        firmware=(str(payload["firmware"]) if payload.get("firmware") else None),
        capacity_ah=_parse_capacity(payload.get("capacity_ah")),
    )


def _parse_capacity(raw: Any) -> float | None:
    """Validate a declared rated capacity, or return None if none was declared.

    Rejects rather than clamps, and rejects the whole HELLO rather than dropping
    the field, because a rig that declares a capacity it cannot mean is a rig
    whose C-rate flags would all be wrong. The likely mistake is a unit one -
    a 3400 mAh cell announced as `3400` - and that value is not implausible
    enough for a downstream reader to notice on its own: it would simply make
    every C-rate 1000x too small, so no discharge would ever look aggressive
    and the capture would score as a model of gentle usage.
    """
    if raw is None:
        return None
    try:
        capacity = float(raw)
    except (TypeError, ValueError) as exc:
        raise LineDecodeError(
            f"HELLO declares capacity_ah={raw!r}, which is not numeric"
        ) from exc
    if not math.isfinite(capacity):
        raise LineDecodeError(f"HELLO declares a non-finite capacity_ah={raw!r}")
    if not (MIN_CAPACITY_AH <= capacity <= MAX_CAPACITY_AH):
        raise LineDecodeError(
            f"HELLO declares capacity_ah={capacity:g}, outside the plausible "
            f"range [{MIN_CAPACITY_AH:g}, {MAX_CAPACITY_AH:g}] Ah. The usual "
            f"cause is milliamp-hours declared as amp-hours - a 3400 mAh cell "
            f"is capacity_ah=3.4, not 3400"
        )
    return capacity


def _parse_data(body: str, raw_line: str = "") -> TelemetryRecord:
    body, checksum = _split_checksum(body)
    _verify_checksum(body, checksum)

    body = body.strip()
    if not body:
        raise LineDecodeError("empty record body")

    # Auto-detect the codec. JSON objects start with a brace; anything else is
    # treated as the compact key=value form. Detection rather than declaration
    # keeps a mixed capture readable, which matters when a student switches
    # firmware mid-session.
    raw_values = (
        _decode_json_body(body) if body.startswith("{")
        else _decode_keyvalue_body(body)
    )

    # Only the timestamp is required at *record* level. Every other field is
    # checked one level up, in `serial_pipeline`, against the channel set this
    # particular rig declared or was observed to supply.
    #
    # The distinction matters and was got wrong once. "Does this rig supply
    # every channel the feature layer needs" is the coverage gate's question,
    # and its answer for a rig without a thermistor is a refusal that names
    # `high_temp_flag` as the consumer that breaks. "Does this record carry
    # what this rig said it would" is a different question, and conflating them
    # meant a rig with no temperature sensor had every one of its records
    # rejected as malformed - which reported a parse failure for what is
    # actually a missing sensor, and left the coverage gate with nothing to
    # inspect.
    #
    # Time is the exception because it is structurally necessary rather than a
    # measurement: a sample with no timestamp cannot be ordered, cannot be
    # integrated over, and cannot be assigned to a cycle.
    if TIME_FIELD not in raw_values:
        raise LineDecodeError(
            f"record omits {TIME_FIELD!r}, the sample timestamp. A sample with "
            f"no time cannot be ordered, integrated over, or assigned to a "
            f"cycle"
        )

    values: dict[str, float] = {}
    problems: list[str] = []
    for wire_name, raw in raw_values.items():
        spec = FIELDS_BY_WIRE.get(wire_name)
        if spec is None:
            # Unknown fields are ignored rather than rejected, so a rig may
            # emit extra diagnostics (cell balancing state, RSSI) without
            # breaking a parser that predates them.
            continue
        value = _coerce_float(wire_name, raw)
        problem = spec.validate(value)
        if problem:
            problems.append(problem)
            continue
        values[spec.channel] = value

    if problems:
        raise LineDecodeError("; ".join(problems))

    return TelemetryRecord(values=values, raw_line=raw_line)


def _verify_checksum(body: str, checksum: str) -> None:
    if not checksum:
        return
    expected = xor_checksum(body)
    if expected != checksum:
        raise LineDecodeError(
            f"checksum mismatch: line carries *{checksum}, computed *{expected}"
        )


def parse_stream(
    lines: Iterable[str], stats: SerialDecodeStats | None = None
) -> Iterator[tuple[str, Any]]:
    """Parse a stream of lines, counting rather than raising on bad ones.

    Yields ``(kind, payload)`` for every usable line. Malformed lines are
    recorded in `stats` and skipped - see the module docstring for why a lossy
    transport must not abort a capture.
    """
    tally = stats if stats is not None else SerialDecodeStats()
    for line in lines:
        tally.n_lines += 1
        try:
            kind, payload = parse_line(line)
        except LineDecodeError as exc:
            tally.record_rejection(str(exc))
            continue
        if kind == "noise":
            tally.n_noise += 1
            continue
        if kind == "status":
            tally.n_status += 1
        elif kind == "data":
            tally.n_accepted += 1
        yield kind, payload


# ---------------------------------------------------------------------------
# Encoding (used by the emulator, the tests, and the firmware reference)
# ---------------------------------------------------------------------------

def encode_hello(
    fields: Sequence[str] = REQUIRED_WIRE_FIELDS,
    cell_id: str | None = None,
    period_ms: float | None = None,
    device: str | None = None,
    firmware: str | None = None,
    capacity_ah: float | None = None,
    checksum: bool = True,
) -> str:
    """Build a HELLO line. The reference firmware emits exactly this shape."""
    payload: dict[str, Any] = {"schema": SCHEMA_ID, "fields": list(fields)}
    if cell_id:
        payload["cell_id"] = cell_id
    if period_ms is not None:
        payload["period_ms"] = period_ms
    if device:
        payload["device"] = device
    if firmware:
        payload["firmware"] = firmware
    if capacity_ah is not None:
        payload["capacity_ah"] = capacity_ah
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return _frame(RECORD_HELLO, body, checksum)


def encode_record(
    values: Mapping[str, float],
    compact: bool = False,
    checksum: bool = True,
) -> str:
    """Build a data line from wire-named values.

    `compact` selects the ``key=value`` codec over JSON. Both decode to the
    same record, which is asserted by the test suite rather than assumed.
    """
    if compact:
        body = " ".join(f"{name}={_format_number(values[name])}" for name in values)
    else:
        body = json.dumps(
            {name: float(value) for name, value in values.items()},
            separators=(",", ":"),
        )
    return _frame(RECORD_DATA, body, checksum)


def encode_status(text: str, checksum: bool = True) -> str:
    """Build a status line. Status text is never scored, only reported."""
    return _frame(RECORD_STATUS, text.strip(), checksum)


def _frame(kind: str, body: str, checksum: bool) -> str:
    suffix = f"*{xor_checksum(body)}" if checksum else ""
    return f"{SENTINEL} {kind} {body}{suffix}"


def _format_number(value: float) -> str:
    """Format compactly without losing the precision the pipeline needs."""
    return f"{float(value):.6g}"


def schema_table() -> str:
    """Render the field table as Markdown, so the docs cannot drift from code.

    `docs/hardware_integration.md` contains the rendered output of this
    function. Regenerating it is a one-liner, and the test suite asserts the
    documented table still matches `FIELDS`.
    """
    header = (
        "| Wire field | Unified channel | Unit | Range | Required |\n"
        "|---|---|---|---|---|\n"
    )
    rows = "\n".join(
        f"| `{spec.wire_name}` | `{spec.channel}` | {spec.unit} | "
        f"{spec.minimum:g} to {spec.maximum:g} | "
        f"{'yes' if spec.required else 'no'} |"
        for spec in FIELDS
    )
    return header + rows
