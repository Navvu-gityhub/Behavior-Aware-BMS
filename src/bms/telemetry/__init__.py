"""Real BMS telemetry: CAN capture, replay, and end-to-end scoring.

Three modules, each owning one concern:

- `sources.py`   raw frame acquisition (recorded logs, live bus, memory) and
                 signal-coverage validation against a DBC
- `cycles.py`    charge/discharge segmentation and coulomb counting, which is
                 what makes a capacity measurement — and therefore SOH —
                 obtainable from a bus that carries only instantaneous current
- `pipeline.py`  wiring source to the existing scoring stages, unchanged

Two transports, one pipeline
----------------------------
- **CAN/DBC** (`sources.py`, `pipeline.py`) — the production-oriented path,
  aimed at a vehicle BMS that publishes on a bus.
- **Serial** (`serial_schema.py`, `serial_source.py`, `serial_pipeline.py`) —
  the bench/demo adapter, aimed at a microcontroller rig that prints lines over
  USB. Includes a deterministic emulator so the whole path runs with no
  hardware attached.

Both converge on a unified-schema frame and then share
`pipeline.score_telemetry_frame` — segmentation, features, risk, health, RUL,
Guardian and twin. Nothing here reimplements a scoring stage. Live, replayed and
batch runs share one code path, so a discrepancy between them is a data
difference rather than an untraceable divergence between two implementations.
"""

from src.bms.telemetry.cycles import (
    CapacityYield,
    CycleMeasurement,
    Phase,
    capacity_yield,
    cycles_to_frame,
    measure_cycles,
    segment_phases,
)
from src.bms.telemetry.pipeline import (
    TWIZY_SIGNAL_MAP,
    TelemetryResult,
    decode_frames,
    replay_log,
    run_telemetry_pipeline,
    score_telemetry_frame,
)
from src.bms.telemetry.serial_pipeline import (
    DEFAULT_MIN_ACCEPTED_FRACTION,
    SerialTelemetryResult,
    records_to_frame,
    replay_serial_capture,
    run_serial_pipeline,
)
from src.bms.telemetry.serial_schema import (
    FIELDS,
    REQUIRED_WIRE_FIELDS,
    SCHEMA_ID,
    SENTINEL,
    SERIAL_CHANNEL_MAP,
    FieldSpec,
    LineDecodeError,
    SchemaHeader,
    SerialDecodeStats,
    TelemetryRecord,
    encode_hello,
    encode_record,
    encode_status,
    parse_line,
    parse_stream,
    schema_table,
    xor_checksum,
)
from src.bms.telemetry.serial_source import (
    DEFAULT_BAUDRATE,
    EmulatedRigSource,
    LineSource,
    LogFileLineSource,
    MemoryLineSource,
    RecordingLineSource,
    RigProfile,
    SerialPortSource,
    TextStreamSource,
    available_ports,
    emulate_rig_lines,
)
from src.bms.telemetry.sources import (
    REQUIRED_CHANNELS,
    CanFrameSource,
    LiveBusSource,
    LogFileSource,
    MemorySource,
    SignalCoverage,
    check_signal_coverage,
    coverage_from_channels,
    dbc_signal_names,
)
from src.bms.telemetry.twin_integration import (
    DEFAULT_HISTORY_LIMIT,
    TwinHistory,
    TwinUpdate,
    evaluate_twin_from_guardian,
    snapshots_to_frame,
    transitions_to_frame,
)

__all__ = [
    "CapacityYield",
    "CycleMeasurement",
    "Phase",
    "capacity_yield",
    "cycles_to_frame",
    "measure_cycles",
    "segment_phases",
    "TWIZY_SIGNAL_MAP",
    "TelemetryResult",
    "decode_frames",
    "replay_log",
    "run_telemetry_pipeline",
    "score_telemetry_frame",
    # Serial ingestion: wire protocol
    "FIELDS",
    "FieldSpec",
    "LineDecodeError",
    "REQUIRED_WIRE_FIELDS",
    "SCHEMA_ID",
    "SENTINEL",
    "SERIAL_CHANNEL_MAP",
    "SchemaHeader",
    "SerialDecodeStats",
    "TelemetryRecord",
    "encode_hello",
    "encode_record",
    "encode_status",
    "parse_line",
    "parse_stream",
    "schema_table",
    "xor_checksum",
    # Serial ingestion: sources
    "DEFAULT_BAUDRATE",
    "EmulatedRigSource",
    "LineSource",
    "LogFileLineSource",
    "MemoryLineSource",
    "RecordingLineSource",
    "RigProfile",
    "SerialPortSource",
    "TextStreamSource",
    "available_ports",
    "emulate_rig_lines",
    # Serial ingestion: pipeline
    "DEFAULT_MIN_ACCEPTED_FRACTION",
    "SerialTelemetryResult",
    "records_to_frame",
    "replay_serial_capture",
    "run_serial_pipeline",
    "coverage_from_channels",
    "DEFAULT_HISTORY_LIMIT",
    "TwinHistory",
    "TwinUpdate",
    "evaluate_twin_from_guardian",
    "snapshots_to_frame",
    "transitions_to_frame",
    "REQUIRED_CHANNELS",
    "CanFrameSource",
    "LiveBusSource",
    "LogFileSource",
    "MemorySource",
    "SignalCoverage",
    "check_signal_coverage",
    "dbc_signal_names",
]
