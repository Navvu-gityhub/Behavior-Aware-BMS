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
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.bms.api.paths import resolve_request_path
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
    resistance_trailing_ohm: float | None
    mean_current_a: float | None
    temperature_c: float | None


class ProfileOut(BaseModel):
    battery_id: str
    discharges_remembered: int
    telemetry_pieces: int
    soh: float | None
    refusal: str
    # The discharge the reported health comes from, and how many discharges
    # since then could not be measured: a reading is only as current as this.
    at_discharge: int | None
    discharges_since_reading: int | None
    later_refusal: str
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
    rd = profile.readings()
    at = int(result.at_cycle) if result.at_cycle is not None else None
    since = profile.n_discharges - at if at is not None else None
    later = ""
    if since:
        tail = profile.table()
        tail = tail[(tail["cycle"] > at) & tail["refusal"].astype(bool)] if not tail.empty else tail
        later = str(tail["refusal"].mode().iloc[0]) if len(tail) else (
            "later discharges did not cross the measurement window")
    readings = [ReadingOut(discharge=int(r["cycle"]), soh=_num(r["soh"]), resistance_ohm=_num(r["r_step_ohm"]),
                           resistance_trailing_ohm=_num(r["r_trailing_ohm"]),
                           mean_current_a=_num(r["mean_current_a"]), temperature_c=_num(r["temperature_c"]))
                for r in rd.to_dict("records")]
    return ProfileOut(
        battery_id=battery_id, discharges_remembered=profile.n_discharges,
        telemetry_pieces=len(profile.sources), soh=_num(result.soh), refusal=result.refusal,
        at_discharge=at, discharges_since_reading=since, later_refusal=later,
        confidence=cap.confidence or "NONE", confidence_reason=cap.reason,
        resistance_confidence=res.confidence or "NONE", uncertainty=_num(result.uncertainty),
        resistance_reference_ohm=_num(result.resistance_reference_ohm),
        resistance_now_ohm=_num(result.resistance_now_ohm),
        resistance_growth=_num(result.resistance_growth), readings=readings)


class ProfileSummaryOut(BaseModel):
    battery_id: str
    discharges_remembered: int
    soh: float | None
    confidence: str
    refusal: str


def _known_ids() -> list[str]:
    ids = set(_profiles)
    d = _profile_dir()
    if d is not None and d.exists():
        ids |= {f.stem for f in d.glob("*.json") if _SAFE_ID.match(f.stem)}
    return sorted(ids)


@router.get("/profiles", response_model=list[ProfileSummaryOut], tags=["batteries"])
def list_profiles() -> list[ProfileSummaryOut]:
    """Every battery with a memory, in memory or persisted."""
    out = []
    for battery_id in _known_ids():
        profile = get_profile(battery_id)
        if profile is None:
            continue
        result = profile.current()
        cap = _capacity_status(result, bol=False)
        out.append(ProfileSummaryOut(battery_id=battery_id, discharges_remembered=profile.n_discharges,
                                     soh=_num(result.soh), confidence=cap.confidence or "NONE",
                                     refusal=result.refusal))
    return out


class IngestRequest(BaseModel):
    csv_path: str


class IngestOut(BaseModel):
    battery_id: str
    discharges_added: int
    discharges_remembered: int


@router.post("/profiles/{battery_id}/ingest", response_model=IngestOut, tags=["batteries"])
def ingest_log(battery_id: str, request: IngestRequest) -> IngestOut:
    """Teach a battery's memory from a CSV log: test_time_s, current_a, voltage_v
    (discharge current negative), optionally temperature_c. The route for real
    logs that arrive as files rather than over CAN or serial."""
    if not _SAFE_ID.match(battery_id):
        raise HTTPException(status_code=422, detail="battery_id must be 1-64 of A-Z a-z 0-9 _ . -")
    path = resolve_request_path(request.csv_path, "csv_path")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"log not found: {path}")
    try:
        tel = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not read {path.name} as CSV: {exc}") from exc
    missing = {"test_time_s", "current_a", "voltage_v"} - set(tel.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"log is missing columns {sorted(missing)}")
    added = remember(battery_id, tel)
    return IngestOut(battery_id=battery_id, discharges_added=added,
                     discharges_remembered=get_profile(battery_id).n_discharges)


_MEMORY_PAGE = Path(__file__).resolve().parents[1] / "dashboard" / "memory_dashboard.html"


@router.get("/memory", include_in_schema=False)
def memory_dashboard() -> FileResponse:
    """Each battery's memory: health, resistance, confidence and conditions."""
    return FileResponse(_MEMORY_PAGE)
