"""Each battery's own memory, served: GET /batteries/{battery_id}/profile.

Every telemetry run (CAN or serial, replayed or live) teaches the battery's
`BatteryProfile` the discharges it contained. The profile answers from its
memory alone, through the shipped gates, so the answer accumulates across
runs instead of resetting with each capture.

Persistence is opt-in, as for the fleet store: set BEACON_PROFILE_DIR and each
profile is saved there as <battery_id>.json after every update and loaded on
first use. Unset, profiles live in memory for the life of the process.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.bms.health.battery_profile import BatteryProfile
from src.bms.health.evidence import _capacity_status, _resistance_status
from src.bms.telemetry.pipeline import REST_THRESHOLD_A

router = APIRouter()
_profiles: dict[str, BatteryProfile] = {}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _profile_dir() -> Path | None:
    raw = os.environ.get("BEACON_PROFILE_DIR")
    return Path(raw) if raw else None


def _path(battery_id: str) -> Path | None:
    d = _profile_dir()
    if d is None or not _SAFE_ID.match(battery_id):
        return None
    return d / f"{battery_id}.json"


def get_profile(battery_id: str, create: bool = False) -> BatteryProfile | None:
    if battery_id in _profiles:
        return _profiles[battery_id]
    path = _path(battery_id)
    if path is not None and path.exists():
        _profiles[battery_id] = BatteryProfile.load(path)
        return _profiles[battery_id]
    if create:
        _profiles[battery_id] = BatteryProfile(battery_id, REST_THRESHOLD_A)
        return _profiles[battery_id]
    return None


def remember(battery_id: str, telemetry: pd.DataFrame) -> int:
    """Teach a battery's profile the discharges in one run; return how many were new."""
    if telemetry is None or telemetry.empty or not {"test_time_s", "current_a", "voltage_v"} <= set(
            telemetry.columns):
        return 0
    profile = get_profile(battery_id, create=True)
    try:
        added = profile.update(telemetry)
    except ValueError:
        # Telemetry the segmenter cannot use teaches nothing; the run itself
        # already reports why it could not be scored.
        return 0
    path = _path(battery_id)
    if added and path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        profile.save(path)
    return added


def forget_all() -> None:
    """Drop in-memory profiles (tests; persisted files are untouched)."""
    _profiles.clear()


class ReadingOut(BaseModel):
    discharge: int
    soh: float | None
    resistance_ohm: float | None
    mean_current_a: float | None
    temperature_c: float | None


class ProfileOut(BaseModel):
    battery_id: str
    discharges_remembered: int
    telemetry_pieces: int
    soh: float | None
    refusal: str
    confidence: str
    confidence_reason: str
    resistance_confidence: str
    uncertainty: float | None
    resistance_reference_ohm: float | None
    resistance_now_ohm: float | None
    resistance_growth: float | None
    readings: list[ReadingOut]


def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


@router.get("/batteries/{battery_id}/profile", response_model=ProfileOut, tags=["batteries"])
def battery_profile(battery_id: str) -> ProfileOut:
    profile = get_profile(battery_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no telemetry has been seen for battery {battery_id!r}")
    result = profile.current()
    # The same rules as the health card (health/evidence.py). The reference is
    # the first discharges this service saw, not necessarily the cell as new,
    # so capacity confidence is at most MEDIUM until a BOL reference exists.
    cap = _capacity_status(result, bol=False)
    res = _resistance_status(result)
    readings = [ReadingOut(discharge=int(r["cycle"]), soh=_num(r["soh"]), resistance_ohm=_num(r["r_step_ohm"]),
                           mean_current_a=_num(r["mean_current_a"]), temperature_c=_num(r["temperature_c"]))
                for r in profile.readings().to_dict("records")]
    return ProfileOut(
        battery_id=battery_id, discharges_remembered=profile.n_discharges,
        telemetry_pieces=len(profile.sources), soh=_num(result.soh), refusal=result.refusal,
        confidence=cap.confidence or "NONE", confidence_reason=cap.reason,
        resistance_confidence=res.confidence or "NONE", uncertainty=_num(result.uncertainty),
        resistance_reference_ohm=_num(result.resistance_reference_ohm),
        resistance_now_ohm=_num(result.resistance_now_ohm),
        resistance_growth=_num(result.resistance_growth), readings=readings)
