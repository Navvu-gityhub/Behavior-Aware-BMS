"""Line sources for serial telemetry: ports, files, memory, and an emulated rig.

A source here is anything that yields text lines. That is the whole abstraction,
and it is deliberately narrower than the CAN `CanFrameSource` protocol, because
a serial rig has less structure than a bus: no arbitration ids, no DBC, just
lines. `serial_schema.parse_stream` turns those lines into records, and
`serial_pipeline` turns records into scores.

Keeping the interface at "yields lines" is what makes the hardware swappable.
A `pyserial` port, a captured `.txt` log, a list in a test, a TCP socket, and the
emulator below are interchangeable, so the demo path and the eventual hardware
path differ in exactly one constructor call and nothing else.

Nothing here names a board, a port, or a sensor
-----------------------------------------------
`SerialPortSource` takes the port as a required argument. There is no default
COM port, no default baud rate guess, no auto-detect that picks the first
enumerated device. That is a deliberate refusal: auto-selecting a port means
that on a laptop with a Bluetooth serial device or an Arduino and a USB-UART
both attached, the pipeline silently reads the wrong one and reports telemetry
from a device nobody chose. `available_ports()` lists candidates so a human or a
CLI can choose; it never chooses.

The emulator is not a shortcut past the parser
-----------------------------------------------
`EmulatedRig` generates *wire-format text lines*, which are then parsed by the
same `parse_line` a real board's output goes through. It would have been less
code to emit `TelemetryRecord` objects directly. That would also have made the
emulator prove nothing: the parser, the checksum, the schema validation and the
coverage gate would all be bypassed exactly in the path the tests exercise, and
the first real board would be the first thing that ever tested them.

So the emulator's output is byte-for-byte something an ESP32 could have printed,
including its boot banner. Replacing it with hardware removes a code path; it
does not add one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence, runtime_checkable

from src.bms.telemetry.serial_schema import (
    REQUIRED_WIRE_FIELDS,
    encode_hello,
    encode_record,
    encode_status,
)

#: Baud rate the reference firmware uses. A default here is safe in a way a
#: default port is not: a mismatched baud rate produces obvious garbage on the
#: very first line, whereas a wrong port produces plausible silence.
DEFAULT_BAUDRATE = 115200


@runtime_checkable
class LineSource(Protocol):
    """Anything that can yield telemetry lines."""

    name: str

    def lines(self) -> Iterator[str]: ...


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------

@dataclass
class MemoryLineSource:
    """Lines held in memory. The source used by tests and fixtures."""

    name: str
    _lines: Sequence[str]

    def lines(self) -> Iterator[str]:
        yield from self._lines


@dataclass
class TextStreamSource:
    """Any iterable of lines - a generator, a file handle, a socket wrapper.

    This is the seam the emulator plugs into, and the seam a transport this
    project has not thought of plugs into later.
    """

    name: str
    stream: Iterable[str]

    def lines(self) -> Iterator[str]:
        yield from self.stream


@dataclass
class LogFileLineSource:
    """A recorded serial capture on disk.

    Capturing a session to a file and replaying it is the serial analogue of
    `pipeline.replay_log`, and serves the same purpose: a result reproduced
    from a capture is the same computation the rig produced, so a disagreement
    is a data difference rather than a code-path difference. Recording a demo
    session also means the demo still runs if the hardware fails on the day.
    """

    name: str
    path: Path | str
    encoding: str = "utf-8"

    def lines(self) -> Iterator[str]:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"{self.name}: no such serial capture: {path}")
        # `errors="replace"` rather than strict: a capture from a real cable can
        # contain a corrupt byte, and that should become one rejected line in
        # the stats, not an exception that discards the entire session.
        with path.open("r", encoding=self.encoding, errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\r\n")


@dataclass
class SerialPortSource:
    """A live serial port, via pyserial.

    `port` is required and never inferred. `duration_s` bounds the capture for
    the same reason `LiveBusSource.duration_s` does: an unbounded generator
    inside a request handler never returns.

    This class is the only place in the project that touches hardware.

    **Opening is retried.** First contact with a physical board showed why: a
    USB-serial bridge that has just been flashed, or that is marginal, commonly
    refuses the next open for a second or two and then accepts it. The original
    implementation opened once and propagated the exception, which surfaced a
    recoverable timing condition as a hard failure with a driver-level message
    nobody could act on. Retrying a bounded number of times is the correct
    response to a transient, and the refusal after the last attempt still names
    what to check.

    Retries deliberately do **not** extend to reads. A port that opens and then
    stops delivering is a different fault, and the accepted-fraction gate in
    `serial_pipeline` is what judges that.
    """

    name: str
    port: str
    baudrate: int = DEFAULT_BAUDRATE
    duration_s: float | None = 60.0
    read_timeout_s: float = 1.0
    encoding: str = "utf-8"
    reset_on_open: bool = True
    #: Bounded, because an unbounded retry on a board that is simply absent
    #: would hang a request handler rather than report the absence.
    open_attempts: int = 6
    open_retry_delay_s: float = 0.75

    def lines(self) -> Iterator[str]:
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - dependency is optional
            raise ImportError(
                "SerialPortSource needs pyserial. Install with "
                "`pip install pyserial`. The emulated rig needs no such "
                "dependency, so the demo path runs without it."
            ) from exc

        import time

        last_error: Exception | None = None
        connection = None
        for attempt in range(1, self.open_attempts + 1):
            try:
                # Opened AT the target baud rather than opened and then
                # reconfigured. Some CH340 drivers accept a rate supplied at
                # open time and refuse the identical rate applied to an already
                # open handle, which is a real failure mode this project hit.
                connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.read_timeout_s,
                )
                break
            except Exception as exc:  # pyserial raises SerialException here
                last_error = exc
                if attempt < self.open_attempts:
                    time.sleep(self.open_retry_delay_s)

        if connection is None:
            raise OSError(
                f"{self.name}: could not open {self.port} after "
                f"{self.open_attempts} attempts over "
                f"{self.open_attempts * self.open_retry_delay_s:.1f}s. "
                f"Last error: {last_error}. Check that no serial monitor holds "
                f"the port, that the board is still enumerated, and that the "
                f"cable carries data. Some USB-serial bridges accept one open "
                f"per enumeration and need a physical replug."
            ) from last_error

        started = time.monotonic()
        with connection:
            if self.reset_on_open:
                # Pulse the reset line, then flush.
                #
                # This used to flush only, on the assumption that opening the
                # port asserts DTR and that this resets the board. First contact
                # with a NodeMCU showed otherwise: its auto-reset circuit is
                # driven by DTR and RTS *in opposition*, so merely opening the
                # port leaves the board free-running from whenever it last
                # booted. The practical consequence is that the HELLO line - the
                # schema handshake carrying the cell id and its rated capacity -
                # had already been sent and lost before the host was listening,
                # and every capture came back with coverage inferred and no
                # declared capacity.
                #
                # Holding EN low and releasing it restarts the sketch while the
                # host is already reading, so the handshake lands in the stream.
                # Flush FIRST, then reset. The order is the whole point.
                #
                # Flushing after the pulse discards exactly what the pulse was
                # for: the board boots in a couple of hundred milliseconds and
                # prints its banner and HELLO immediately, so a flush timed
                # after the reset throws the handshake away and the capture
                # comes back with coverage inferred - which is precisely the
                # symptom this code was added to cure. The stale bytes worth
                # discarding are the ones buffered *before* the reset.
                try:
                    connection.dtr = False
                    connection.rts = True   # EN low: hold in reset
                    time.sleep(0.1)
                    connection.rts = False  # EN high: run
                    # Flush in the gap between releasing reset and the board's
                    # first output. Timing matters in both directions and this
                    # is the only window that satisfies both:
                    #
                    #   before the pulse - too early. Anything the board sent
                    #     while the host was still retrying the open is already
                    #     in the driver's buffer and arrives anyway, so the
                    #     capture holds two sessions and is refused for a time
                    #     reversal at the join.
                    #   after the board prints - too late. The banner and the
                    #     HELLO are what the reset was for, and flushing then
                    #     discards them.
                    #
                    # The board needs ~200 ms to come out of reset; the flush
                    # below costs microseconds, so it lands cleanly in between.
                    connection.reset_input_buffer()
                except Exception:
                    # Some bridges refuse modem-control lines outright. That
                    # costs the handshake, not the capture: the coverage gate
                    # falls back to inferring channels from what arrives and
                    # says so in the result.
                    pass

            while True:
                if (
                    self.duration_s is not None
                    and time.monotonic() - started >= self.duration_s
                ):
                    return
                raw = connection.readline()
                if not raw:
                    continue  # read timeout, not end of stream
                yield raw.decode(self.encoding, errors="replace").rstrip("\r\n")


@dataclass
class RecordingLineSource:
    """Wraps any `LineSource`, writing every line to a file as it passes through.

    A decorator rather than a method on each source, because recording is a
    property of a *run*, not of a transport: a live port, a replayed capture and
    the emulator should all be recordable by the same mechanism, and each of
    them gaining its own `write_capture` is how three subtly different recorders
    come to exist.

    This is what makes a physical bring-up reproducible. Before it, the only
    capture facility wrote the *emulator's* output, so the bytes a real board
    actually sent could not be kept, and a hardware result could not be replayed
    or committed as a fixture - which meant it could not be checked by anyone
    who was not in the room.

    Lines are written as they are yielded, not buffered to the end, so a capture
    interrupted by a brown-out or a yanked cable keeps everything received up to
    that point. That capture is exactly what a post-mortem needs.
    """

    inner: LineSource
    path: Path | str
    encoding: str = "utf-8"
    #: Defaults to the wrapped source's name, so recording a run does not change
    #: how its result is labelled. A plain field rather than a property because
    #: `LineSource` declares `name` as a mutable attribute.
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.inner.name

    def lines(self) -> Iterator[str]:
        # The destination is opened lazily, on the first line that actually
        # arrives, and NOT before the inner source has produced anything.
        #
        # Opening it up front truncates it, which destroys a previous capture
        # whenever the next run fails to open the port. That is not a
        # hypothetical: it deleted this project's first physical capture, taken
        # from a board whose USB bridge permits one open per enumeration, when
        # the following run could not reopen the port. A capture is evidence,
        # and a later failure must never erase it.
        destination = Path(self.path)
        handle = None
        try:
            for line in self.inner.lines():
                if handle is None:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    handle = destination.open(
                        "w", encoding=self.encoding, newline="\n"
                    )
                handle.write(line + "\n")
                # Flushed per line for the same reason the write is streamed: an
                # aborted session must leave a readable file, and a rig sampling
                # at 1 Hz is nowhere near a rate where this costs anything.
                handle.flush()
                yield line
        finally:
            if handle is not None:
                handle.close()


def available_ports() -> list[tuple[str, str]]:
    """List serial ports as ``(device, description)``, for a human to choose from.

    Returns an empty list when pyserial is absent, rather than raising: this is
    a convenience for a CLI, and the absence of the optional dependency should
    not look like the absence of hardware.
    """
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - dependency is optional
        return []
    return [(port.device, port.description or "") for port in list_ports.comports()]


# ---------------------------------------------------------------------------
# Emulated rig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RigProfile:
    """The synthetic cell and duty cycle the emulator reproduces.

    Defaults describe a single 18650-class cell on a bench harness - roughly
    what a college rig measures - because a profile that produces currents below
    `cycles.REST_THRESHOLD_A` would be segmented as one long rest and yield no
    cycles at all. The discharge and charge currents here sit comfortably above
    that dead band.
    """

    cell_id: str = "RIG_01"
    n_cycles: int = 4
    capacity_ah: float = 2.0
    sample_period_s: float = 30.0
    discharge_current_a: float = 1.5
    charge_current_a: float = 0.75
    rest_s: float = 300.0
    voltage_full_v: float = 4.15
    voltage_empty_v: float = 3.00
    internal_resistance_ohm: float = 0.045
    ambient_c: float = 25.0
    thermal_gain_c_per_a: float = 4.0
    #: Fractional capacity lost per cycle. Kept small enough that later cycles
    #: stay above `cycles.COMPLETE_CYCLE_FRACTION` of the largest observed, so
    #: the emulated capture yields usable SOH points rather than exercising the
    #: partial-cycle exclusion by accident.
    fade_per_cycle: float = 0.004
    noise_v: float = 0.004
    noise_a: float = 0.010
    noise_c: float = 0.15
    seed: int = 20260826


def emulate_rig_lines(
    profile: RigProfile | None = None,
    compact: bool = False,
    checksum: bool = True,
    boot_banner: bool = True,
    corrupt_every: int = 0,
) -> Iterator[str]:
    """Generate the wire lines a conforming rig would print for `profile`.

    Deterministic: the same profile and seed produce identical output, which is
    what lets the test suite assert exact capacity figures rather than ranges.

    `boot_banner` emits the kind of non-BEACON chatter a board prints on reset.
    It is on by default so the happy path is always tested against a stream that
    contains noise, rather than a clean one that a real board never produces.

    `corrupt_every` truncates every Nth data line, to exercise the rejection
    counters. Zero disables it.
    """
    settings = profile or RigProfile()
    rng = random.Random(settings.seed)

    if boot_banner:
        # Representative of what an ESP32 prints from ROM before the sketch
        # runs. Reproduced here so the sentinel filter is exercised by default.
        yield "ets Jul 29 2019 12:21:46"
        yield ""
        yield "rst:0x1 (POWERON_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)"
        yield "configsip: 0, SPIWP:0xee"
        yield "clk_drv:0x00,q_drv:0x00,d_drv:0x00,cs0_drv:0x00,hd_drv:0x00,wp_drv:0x00"
        yield "mode:DIO, clock div:2"
        yield "load:0x3fff0030,len:1184"
        yield "entry 0x400805f0"

    yield encode_hello(
        fields=REQUIRED_WIRE_FIELDS,
        cell_id=settings.cell_id,
        period_ms=settings.sample_period_s * 1000.0,
        device="emulated-rig",
        firmware="beacon-emulator/1.0",
        capacity_ah=settings.capacity_ah,
        checksum=checksum,
    )
    yield encode_status(
        "emulated rig; no physical hardware attached", checksum=checksum
    )

    cursor = _RigCursor()

    for cycle_index in range(settings.n_cycles):
        cycle_capacity_ah = settings.capacity_ah * (
            1.0 - settings.fade_per_cycle * cycle_index
        )
        discharge_s = cycle_capacity_ah * 3600.0 / settings.discharge_current_a
        charge_s = cycle_capacity_ah * 3600.0 / settings.charge_current_a

        phases = (
            # Discharge: SOC 100 -> 0, current negative.
            (discharge_s, -settings.discharge_current_a, 100.0, 0.0),
            # Rest: current inside the dead band, so it segments as rest.
            (settings.rest_s, 0.0, 0.0, 0.0),
            # Charge: SOC 0 -> 100, current positive.
            (charge_s, settings.charge_current_a, 0.0, 100.0),
        )
        for duration_s, current_a, soc_start, soc_end in phases:
            yield from _phase_lines(
                settings, rng, cursor, duration_s,
                current_a=current_a, soc_start=soc_start, soc_end=soc_end,
                compact=compact, checksum=checksum, corrupt_every=corrupt_every,
            )


@dataclass
class _RigCursor:
    """Running position of the emulated rig, carried across phases.

    A small mutable holder rather than threading two counters through every
    return value: the phases form one continuous session, and an explicit
    cursor says so more plainly than a tuple that has to be unpacked and
    rebound at each call site.
    """

    elapsed_s: float = 0.0
    emitted: int = 0


def _phase_lines(
    settings: RigProfile,
    rng: random.Random,
    cursor: _RigCursor,
    duration_s: float,
    current_a: float,
    soc_start: float,
    soc_end: float,
    compact: bool,
    checksum: bool,
    corrupt_every: int,
) -> Iterator[str]:
    """Emit one charge, discharge or rest phase as wire lines."""
    n_samples = max(int(round(duration_s / settings.sample_period_s)), 2)

    for step in range(n_samples):
        progress = step / (n_samples - 1) if n_samples > 1 else 1.0
        soc = soc_start + (soc_end - soc_start) * progress

        # Open-circuit voltage tracks SOC; terminal voltage adds the IR term,
        # which is why a discharging cell reads lower and a charging one higher
        # at the same state of charge.
        open_circuit_v = (
            settings.voltage_empty_v
            + (settings.voltage_full_v - settings.voltage_empty_v) * (soc / 100.0)
        )
        terminal_v = open_circuit_v + current_a * settings.internal_resistance_ohm

        # Self-heating rises with current magnitude and accumulates through the
        # phase; a resting cell relaxes back toward ambient.
        heating_c = (
            settings.thermal_gain_c_per_a
            * abs(current_a)
            * (0.3 + 0.7 * progress)
        )
        if current_a == 0.0:
            heating_c = settings.thermal_gain_c_per_a * 0.4 * math.exp(-3.0 * progress)
        temperature_c = settings.ambient_c + heating_c

        values = {
            "t": round(cursor.elapsed_s, 3),
            "v": round(terminal_v + rng.gauss(0.0, settings.noise_v), 4),
            "i": round(current_a + rng.gauss(0.0, settings.noise_a), 4),
            "tc": round(temperature_c + rng.gauss(0.0, settings.noise_c), 3),
            "soc": round(min(max(soc, 0.0), 100.0), 2),
        }
        line = encode_record(values, compact=compact, checksum=checksum)

        cursor.emitted += 1
        if corrupt_every and cursor.emitted % corrupt_every == 0:
            # Truncate mid-line, which is what a brown-out or a yanked cable
            # actually produces.
            line = line[: max(len(line) // 2, len(line) - 12)]

        cursor.elapsed_s += settings.sample_period_s
        yield line


@dataclass
class EmulatedRigSource:
    """A `LineSource` backed by `emulate_rig_lines`.

    This is the demo path: the complete ESP32-to-dashboard route runs end to end
    with nothing plugged in. Swapping it for `SerialPortSource` is a one-line
    change at the call site and nothing else - see
    `docs/hardware_integration.md`.
    """

    name: str = "emulated_rig"
    profile: RigProfile = field(default_factory=RigProfile)
    compact: bool = False
    checksum: bool = True
    boot_banner: bool = True
    corrupt_every: int = 0

    def lines(self) -> Iterator[str]:
        return emulate_rig_lines(
            self.profile,
            compact=self.compact,
            checksum=self.checksum,
            boot_banner=self.boot_banner,
            corrupt_every=self.corrupt_every,
        )

    def write_capture(self, path: Path | str) -> Path:
        """Write this rig's output to a file, for replay or for a fixture.

        Useful for the demo: capture once, and the presentation still runs if
        the laptop, the cable or the board misbehaves on the day.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            "\n".join(self.lines()) + "\n", encoding="utf-8"
        )
        return destination
