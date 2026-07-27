"""Request/response contracts for the API layer.

Kept separate from `digital_twin` on purpose (see that module's and this
package's docstrings): the domain module returns plain dataclasses with no
knowledge of HTTP or JSON. These Pydantic models are the translation at
the boundary, so digital_twin stays usable from a CLI, a batch job, or a
different transport without dragging FastAPI/Pydantic along.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    n_batteries: int = Field(default=20, ge=1, le=500)
    rows_per_battery: int = Field(default=400, ge=10, le=5000)
    seed: Optional[int] = Field(default=None, description="Omit for a fresh random fleet each call.")


class TwinSnapshotOut(BaseModel):
    battery_id: str
    twin_state: str
    health_index: float
    failure_likelihood: float
    rul_cycles: int
    replacement_policy: str
    evaluated_at: str


class TransitionOut(BaseModel):
    from_state: Optional[str]
    to_state: str
    at: str


class BatterySummaryOut(BaseModel):
    battery_id: str
    twin_state: str
    battery_state: str
    health_index: float
    risk_level: str
    rul_cycles: int
    replacement_policy: str


class BatteryDetailOut(BaseModel):
    battery_id: str
    twin: TwinSnapshotOut
    guardian_status: str
    primary_causes: str
    recommendation: str
    guardian_report: str
    risk_level: str
    risk_score: float
    transitions: list[TransitionOut]


class TimelinePointOut(BaseModel):
    cycle: int
    stress_score: float
    soc: float
    temperature_c: float


class PipelineRunResponseOut(BaseModel):
    n_batteries_scored: int
    transitions: list[TransitionOut]
    battery_ids: list[str]


class HealthzOut(BaseModel):
    status: str
    n_runs: int
    n_batteries_tracked: int
