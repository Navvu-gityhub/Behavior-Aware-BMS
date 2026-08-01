"""Request/response contracts for the telemetry and evaluation endpoints.

Same boundary discipline as `schemas.py`: the domain modules
(`telemetry/`, `adaptive/`) return plain dataclasses that know nothing about
HTTP, and these Pydantic models are the translation. That keeps the telemetry
pipeline usable from a CLI or a batch job without dragging FastAPI along, which
matters because `python -m src.bms.adaptive` already depends on it.

One convention worth stating: every response that can represent a refusal
carries `refusals` and a `status`, rather than signalling refusal by HTTP error.
A run that decoded 40,000 frames and then declined to compute SOH because none
of the discharges was complete has produced a great deal of useful information,
and collapsing that into a 4xx would throw it away. HTTP errors are reserved for
malformed requests and missing files - failures of the request rather than
findings about the data.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Signal coverage
# ---------------------------------------------------------------------------

class SignalCoverageOut(BaseModel):
    dbc_path: str
    signal_map_used: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The DBC-signal-to-channel mapping this result was computed with. "
            "Stated explicitly because a coverage failure can mean either that "
            "the bus lacks a channel or that the map names signals this DBC "
            "does not define, and those need different fixes."
        ),
    )
    status: str = Field(description="COMPLETE or INCOMPLETE")
    complete: bool
    available_signals: list[str]
    mapped_channels: list[str]
    missing_channels: list[str]
    explanation: str = Field(
        description="Which downstream stage needs each missing channel."
    )


# ---------------------------------------------------------------------------
# Cycle measurements
# ---------------------------------------------------------------------------

class CycleMeasurementOut(BaseModel):
    cycle: int
    capacity_ah: float
    is_complete: bool
    start_time_s: float
    end_time_s: float
    n_samples: int
    mean_current_a: float
    avg_temp: Optional[float] = None
    max_temp: Optional[float] = None
    depth_of_discharge: Optional[float] = Field(
        default=None,
        description=(
            "Observed SOC swing. Reported for diagnosis only; capacity is never "
            "normalised by it, because the BMS's SOC is itself derived from a "
            "capacity estimate."
        ),
    )
    exclusion_reason: str = ""


class CapacityYieldOut(BaseModel):
    n_discharges: int
    n_complete: int
    n_partial: int
    usable_fraction: float
    largest_discharge_ah: float
    summary: str


# ---------------------------------------------------------------------------
# Twin
# ---------------------------------------------------------------------------

class TwinTransitionOut(BaseModel):
    battery_id: str
    from_state: Optional[str]
    to_state: str
    at: str


class TwinUpdateOut(BaseModel):
    evaluated: bool
    snapshots: list[dict]
    transitions: list[TwinTransitionOut]
    skipped_reason: str = ""


class TwinHistoryOut(BaseModel):
    battery_id: str
    n_snapshots: int
    snapshots: list[dict]
    transitions: list[TwinTransitionOut]


# ---------------------------------------------------------------------------
# Telemetry runs
# ---------------------------------------------------------------------------

class ReplayRequest(BaseModel):
    log_path: str = Field(description="Path to a CAN log readable by python-can.")
    dbc_path: Optional[str] = Field(
        default=None,
        description="DBC to decode with. Defaults to the bundled reference pack.",
    )
    battery_id: str = Field(default="VEHICLE_01")
    require_full_coverage: bool = Field(
        default=True,
        description=(
            "When false the run proceeds past a missing channel so decoded "
            "telemetry can be inspected. Scoring still refuses: the refusal is "
            "about the data, not about permission."
        ),
    )


class LiveCaptureRequest(BaseModel):
    channel: str = Field(default="vcan0")
    interface: str = Field(default="socketcan")
    duration_s: float = Field(default=10.0, gt=0.0, le=300.0)
    dbc_path: Optional[str] = None
    battery_id: str = Field(default="VEHICLE_01")
    require_full_coverage: bool = True


class TelemetryRunOut(BaseModel):
    source: str
    status: str = Field(description="SCORED, SCORED_WITH_REFUSALS or REFUSED")
    battery_id: str
    n_frames: int
    n_decoded: int
    stages_completed: list[str]
    coverage: SignalCoverageOut
    cycles: list[CycleMeasurementOut]
    capacity_yield: Optional[CapacityYieldOut] = None
    guardian: list[dict]
    twin: Optional[TwinUpdateOut] = None
    refusals: list[str]
    fade_prediction: Optional[float] = Field(
        default=None,
        description=(
            "Always null while no model has passed the promotion gate. The "
            "rule-based severity index in `guardian` is triage, not a "
            "calibrated fade prediction (ADR 0005)."
        ),
    )
    fade_prediction_refusal: str = ""


class LiveSampleOut(BaseModel):
    """One decoded telemetry row, for the live view."""

    test_time_s: float
    voltage_v: Optional[float] = None
    current_a: Optional[float] = None
    temperature_c: Optional[float] = None
    soc: Optional[float] = None
    mode: str = Field(description="charge, discharge or rest")


class LiveStateOut(BaseModel):
    battery_id: str
    n_samples: int
    latest: Optional[LiveSampleOut] = None
    recent: list[LiveSampleOut]
    instrumented_channels: list[str]
    uninstrumented_note: str = Field(
        default="",
        description=(
            "Channels a consumer might expect but which this bus does not "
            "report. Stated so a view cannot imply measurements that were "
            "never taken."
        ),
    )


class ThermalPointOut(BaseModel):
    """One cell of the thermal timeline.

    The axes are cycle and position within that cycle, both real. There is no
    per-cell axis because the unified schema carries a single pack-aggregate
    temperature channel; a per-cell heatmap would mean inventing values.
    """

    cycle: int
    phase_fraction: float = Field(
        ge=0.0, le=1.0, description="Position through the discharge, 0 to 1."
    )
    temperature_c: float


class ThermalTimelineOut(BaseModel):
    battery_id: str
    points: list[ThermalPointOut]
    n_cycles: int
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    resolution_note: str = Field(
        default=(
            "Pack-aggregate temperature over cycle and discharge progress. The "
            "unified schema has one temperature channel, so there is no "
            "per-cell resolution to display."
        )
    )


# ---------------------------------------------------------------------------
# Transfer validation
# ---------------------------------------------------------------------------

class AxisVerdictOut(BaseModel):
    axis: str
    source_variation: str
    target_variation: str
    usable: bool
    marginal: bool
    reason: str


class FeasibilityOut(BaseModel):
    source: str
    target: str
    status: str
    usable_axes: list[str]
    marginal_axes: list[str]
    verdicts: list[AxisVerdictOut]
    is_prediction: bool = Field(
        default=True,
        description=(
            "True when derived from published metadata rather than measured on "
            "loaded data. Confirm with the measured screen before citing."
        ),
    )


class DatasetSpecOut(BaseModel):
    name: str
    description: str
    n_cells: int
    chemistry: str
    nominal_capacity_ah: Optional[float] = None
    variation: dict[str, str]
    caveats: list[str]
    citation: str
