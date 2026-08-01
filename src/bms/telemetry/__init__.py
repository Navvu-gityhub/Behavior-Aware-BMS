"""Real BMS telemetry: CAN capture, replay, and end-to-end scoring.

Three modules, each owning one concern:

- `sources.py`   raw frame acquisition (recorded logs, live bus, memory) and
                 signal-coverage validation against a DBC
- `cycles.py`    charge/discharge segmentation and coulomb counting, which is
                 what makes a capacity measurement — and therefore SOH —
                 obtainable from a bus that carries only instantaneous current
- `pipeline.py`  wiring source to the existing scoring stages, unchanged

Nothing here reimplements a scoring stage. Live and batch runs share one code
path, so a discrepancy between them is a data difference rather than an
untraceable divergence between two implementations.
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
)
from src.bms.telemetry.twin_integration import (
    DEFAULT_HISTORY_LIMIT,
    TwinHistory,
    TwinUpdate,
    evaluate_twin_from_guardian,
    snapshots_to_frame,
    transitions_to_frame,
)
from src.bms.telemetry.sources import (
    REQUIRED_CHANNELS,
    CanFrameSource,
    LiveBusSource,
    LogFileSource,
    MemorySource,
    SignalCoverage,
    check_signal_coverage,
    dbc_signal_names,
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
