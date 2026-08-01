"""CAN frame sources: recorded logs, live buses, and in-memory replay.

A source yields `(timestamp, arbitration_id, payload)` tuples. Anything that can
do that — a `.blf` file, a SocketCAN interface, a list of tuples in a test — is a
source, so the pipeline downstream is identical for replay and live capture.

Signal coverage is checked before decoding, not after
-----------------------------------------------------
The most likely way this pipeline produces something wrong is a DBC that decodes
cleanly but does not carry the channels the feature layer needs.

That is not hypothetical here. The example DBC shipped with this repository,
`dbc_examples/twizy_bms_1.dbc`, defines one message carrying `v_c_climit`,
`v_b_current` and `v_b_soc`. There is no temperature signal. But
`features/behavior_features.py` computes `high_temp_flag` from `temperature_c`,
and this project's NaN-as-healthy fix established that missing data must raise
rather than evaluate as safe — a NumPy comparison against NaN is False, so an
absent temperature channel would have silently produced "not hot" for every row
and a healthy-looking score for a pack nobody measured.

So `check_signal_coverage` compares a DBC's decodable signals against what the
downstream stages require, and reports what is missing before a single frame is
decoded. The pipeline refuses on a missing required channel. Refusing is the
correct output: the alternative is a health index computed from a channel that
does not exist.

Live capture is untested against hardware
-----------------------------------------
`LiveBusSource` wraps `can.Bus`, so it works with any interface python-can
supports, including `virtual` for testing and `socketcan` on Linux. The virtual
path is covered by the test suite. Real hardware is not, because none is
attached to this environment, and that limit is recorded here rather than
implied to be verified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Mapping, Protocol, Sequence, runtime_checkable

# A single raw frame: (timestamp_seconds, arbitration_id, payload_bytes).
Frame = tuple[float, int, bytes]

# Channels the downstream stages need, and which stage needs each. Used to
# explain precisely what breaks when a DBC lacks a signal, rather than emitting
# a generic schema error.
REQUIRED_CHANNELS: Mapping[str, str] = {
    "current_a": "cycle segmentation and coulomb counting",
    "voltage_v": "unified schema validation",
    "temperature_c": "behaviour features (high_temp_flag, temp_rolling_*)",
    "soc": "behaviour features (deep_discharge_flag, high_soc_flag)",
}


@runtime_checkable
class CanFrameSource(Protocol):
    """Anything that can yield raw CAN frames."""

    name: str

    def frames(self) -> Iterator[Frame]: ...


@dataclass
class MemorySource:
    """Frames held in memory. The source used by tests and by replay fixtures."""

    name: str
    _frames: Sequence[Frame]

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


@dataclass
class LogFileSource:
    """A recorded CAN log, in any format python-can can read.

    Covers .blf, .asc, .trc, .csv and .log. Format is inferred from the
    extension by python-can's own reader registry, so a new format supported
    upstream works here without a change.
    """

    name: str
    path: Path | str

    def frames(self) -> Iterator[Frame]:
        try:
            import can
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "LogFileSource needs python-can. Install with "
                "`pip install python-can`."
            ) from exc

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(f"{self.name}: no such CAN log: {path}")

        with can.LogReader(str(path)) as reader:
            for message in reader:
                if message.is_error_frame or message.is_remote_frame:
                    # Error and remote frames carry no signal payload; decoding
                    # them would produce values from undefined bytes.
                    continue
                yield (
                    float(message.timestamp),
                    int(message.arbitration_id),
                    bytes(message.data),
                )


@dataclass
class LiveBusSource:
    """A live CAN interface.

    `channel` and `interface` pass through to `can.Bus`, so this supports
    socketcan, pcan, kvaser, vector, virtual and anything else python-can
    handles.

    `duration_s` bounds the capture. Unbounded capture is available by passing
    None, but the default is bounded because an unbounded generator inside a
    request handler would never return.
    """

    name: str
    channel: str = "vcan0"
    interface: str = "socketcan"
    duration_s: float | None = 10.0
    receive_timeout_s: float = 1.0
    bus_kwargs: Mapping[str, object] = field(default_factory=dict)

    def frames(self) -> Iterator[Frame]:
        try:
            import can
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError(
                "LiveBusSource needs python-can. Install with "
                "`pip install python-can`."
            ) from exc

        started = time.monotonic()
        with can.Bus(
            channel=self.channel, interface=self.interface, **dict(self.bus_kwargs)
        ) as bus:
            while True:
                if (
                    self.duration_s is not None
                    and time.monotonic() - started >= self.duration_s
                ):
                    return
                message = bus.recv(timeout=self.receive_timeout_s)
                if message is None:
                    continue  # timeout, not end of stream
                if message.is_error_frame or message.is_remote_frame:
                    continue
                yield (
                    float(message.timestamp),
                    int(message.arbitration_id),
                    bytes(message.data),
                )


# ---------------------------------------------------------------------------
# Signal coverage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalCoverage:
    """Which required channels a DBC and signal map can actually supply."""

    dbc_path: str
    available_signals: tuple[str, ...]
    mapped_channels: tuple[str, ...]
    missing_channels: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_channels

    def __bool__(self) -> bool:
        return self.complete

    @property
    def status(self) -> str:
        return "COMPLETE" if self.complete else "INCOMPLETE"

    def render(self) -> str:
        lines = [f"{self.dbc_path}: {self.status}"]
        lines.append(f"  decodable signals: {list(self.available_signals)}")
        lines.append(f"  mapped channels: {list(self.mapped_channels)}")
        for channel in self.missing_channels:
            consumer = REQUIRED_CHANNELS.get(channel, "downstream stages")
            lines.append(f"  MISSING {channel} -> needed by {consumer}")
        if not self.complete:
            lines.append(
                "  This DBC cannot drive the full feature pipeline. Supply a "
                "DBC defining the missing signals, or extend signal_map to "
                "point at equivalents already on the bus."
            )
        return "\n".join(lines)


def dbc_signal_names(dbc) -> tuple[str, ...]:
    """Every signal name a loaded DBC can decode."""
    return tuple(
        sorted({signal.name for message in dbc.messages for signal in message.signals})
    )


def check_signal_coverage(
    dbc,
    signal_map: Mapping[str, str],
    dbc_path: str = "<dbc>",
    required: Mapping[str, str] = REQUIRED_CHANNELS,
) -> SignalCoverage:
    """Compare a DBC's signals against what the pipeline needs.

    `signal_map` maps DBC signal names to unified schema channels, e.g.
    ``{"v_b_current": "current_a"}``. Coverage is judged on the mapped result,
    because a bus may name a channel anything.
    """
    available = dbc_signal_names(dbc)
    mapped = {
        channel
        for signal, channel in signal_map.items()
        if signal in available
    }
    missing = tuple(sorted(set(required) - mapped))
    return SignalCoverage(
        dbc_path=dbc_path,
        available_signals=available,
        mapped_channels=tuple(sorted(mapped)),
        missing_channels=missing,
    )
