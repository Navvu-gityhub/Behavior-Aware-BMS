"""REST API for the Behavior-Aware BMS pipeline.

Transport only. This module's job is routing, request validation, JSON
serialization, error handling, and OpenAPI documentation (auto-generated
at /docs and /redoc by FastAPI) — nothing here computes a health index,
a risk score, or a twin state. That logic lives in `main.run_pipeline`
(ingestion through Guardian) and `src.bms.digital_twin` (twin state,
transitions, timeline), and would stay unchanged if this REST API were
replaced or supplemented by a CLI, MQTT, or gRPC front end — that
separation is the reason `digital_twin` is its own package rather than
living inside this one.

Every score surfaced by this API — health_index, risk_score, rul_cycles,
guardian_report — is the same rule-based, NOT-validated-against-measured-
degradation heuristic documented at length in docs/final_report.md. This
API makes those numbers reachable over HTTP; it does not make them more
trustworthy. See docs/final_report.md Sections 4 and 5 before presenting
any number from this API as a validated prediction.

Run with: uvicorn src.bms.api.app:app --reload
Docs at:  http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from main import run_pipeline
from src.bms.simulation.simulate_telemetry import SimulationConfig, simulate_fleet
from src.bms.digital_twin import build_health_timeline
from src.bms.api.store import fleet_store
from src.bms.api.telemetry_routes import router as telemetry_router
from src.bms.api.schemas import (
    BatteryDetailOut,
    BatterySummaryOut,
    HealthzOut,
    PipelineRunResponseOut,
    SimulateRequest,
    TimelinePointOut,
    TransitionOut,
    TwinSnapshotOut,
)

app = FastAPI(
    title="Behavior-Aware BMS API",
    description=(
        "REST layer over the behavior-aware battery analytics pipeline and "
        "digital twin. All scores are rule-based heuristics, not validated "
        "predictive models — see /docs and docs/final_report.md."
    ),
    version="0.1.0",
)

# Telemetry, twin and transfer endpoints live in their own router so this
# module's existing routes are untouched. See api/telemetry_routes.py.
app.include_router(telemetry_router)

_DASHBOARD_PATH = Path(__file__).parent.parent / "dashboard" / "live_dashboard.html"


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serves the live fleet dashboard (src/bms/dashboard/live_dashboard.html).

    Static HTML/CSS/JS, no build step and no external CDN dependency — it
    only needs this API running same-origin. Kept as a plain file read
    rather than a template engine dependency, matching this project's
    general preference for the fewest moving parts that do the job (see
    api/store.py's docstring on the in-memory store for the same
    reasoning). See docs/api.md for the distinction between this and the
    static, no-server dashboard main.py generates.
    """
    return FileResponse(_DASHBOARD_PATH)


def _snapshot_out(snapshot) -> TwinSnapshotOut:
    return TwinSnapshotOut(
        battery_id=snapshot.battery_id,
        twin_state=snapshot.twin_state,
        health_index=snapshot.health_index,
        failure_likelihood=snapshot.failure_likelihood,
        rul_cycles=snapshot.rul_cycles,
        replacement_policy=snapshot.replacement_policy,
        evaluated_at=snapshot.evaluated_at,
    )


def _transition_out(transition) -> TransitionOut:
    return TransitionOut(from_state=transition.from_state, to_state=transition.to_state, at=transition.at)


@app.get("/healthz", response_model=HealthzOut, tags=["service"])
def healthz() -> HealthzOut:
    """Service liveness — deliberately NOT battery health (see /batteries for that)."""
    return HealthzOut(status="ok", n_runs=fleet_store.n_runs, n_batteries_tracked=fleet_store.n_batteries)


@app.post("/pipeline/simulate", response_model=PipelineRunResponseOut, tags=["pipeline"])
def run_simulated_pipeline(request: SimulateRequest) -> PipelineRunResponseOut:
    """Run the full pipeline on a freshly simulated fleet and ingest the
    result into the fleet store. Repeated calls build up real twin-state
    transition history per battery (new random fleet each call unless a
    seed is given, so transitions across calls reflect different simulated
    batteries, not one battery evolving — this is a demonstration endpoint,
    not a live telemetry ingest; see docs/api.md for that distinction).
    """
    raw = simulate_fleet(
        SimulationConfig(
            n_batteries=request.n_batteries,
            rows_per_battery=request.rows_per_battery,
            seed=request.seed,
        )
    )
    guardian = run_pipeline(raw).guardian

    from src.bms.preprocessing.schema import standardize_validate_bms_data
    from src.bms.features.behavior_features import compute_behavior_flags
    from src.bms.risk.stress_score import compute_stress_score

    clean, _ = standardize_validate_bms_data(raw, dataset="simulated")
    flagged = compute_behavior_flags(clean)
    flagged["stress_score"] = compute_stress_score(flagged)

    transitions = fleet_store.ingest_run(guardian, flagged)

    return PipelineRunResponseOut(
        n_batteries_scored=len(guardian),
        transitions=[_transition_out(t) for t in transitions],
        battery_ids=sorted(guardian["battery_id"].tolist()),
    )


@app.get("/batteries", response_model=list[BatterySummaryOut], tags=["batteries"])
def list_batteries() -> list[BatterySummaryOut]:
    """Current twin state for every battery the store has seen. Empty
    until at least one /pipeline/simulate call has been made."""
    return [
        BatterySummaryOut(
            battery_id=record.snapshot.battery_id,
            twin_state=record.snapshot.twin_state,
            battery_state=record.guardian_row["battery_state"],
            health_index=record.snapshot.health_index,
            risk_level=record.guardian_row["risk_level"],
            rul_cycles=record.snapshot.rul_cycles,
            replacement_policy=record.snapshot.replacement_policy,
        )
        for record in fleet_store.list_batteries()
    ]


@app.get("/batteries/{battery_id}", response_model=BatteryDetailOut, tags=["batteries"])
def get_battery(battery_id: str) -> BatteryDetailOut:
    record = fleet_store.get_battery(battery_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No battery '{battery_id}' in the fleet store yet.")

    return BatteryDetailOut(
        battery_id=battery_id,
        twin=_snapshot_out(record.snapshot),
        guardian_status=record.guardian_row["guardian_status"],
        primary_causes=record.guardian_row["primary_causes"],
        recommendation=record.guardian_row["recommendation"],
        guardian_report=record.guardian_row["guardian_report"],
        evidence_confidence=record.guardian_row["evidence_confidence"],
        evidence_note=record.guardian_row["evidence_note"],
        risk_level=record.guardian_row["risk_level"],
        risk_score=record.guardian_row["risk_score"],
        transitions=[_transition_out(t) for t in record.transitions],
    )


@app.get("/batteries/{battery_id}/timeline", response_model=list[TimelinePointOut], tags=["batteries"])
def get_battery_timeline(battery_id: str) -> list[TimelinePointOut]:
    if fleet_store.get_battery(battery_id) is None:
        raise HTTPException(status_code=404, detail=f"No battery '{battery_id}' in the fleet store yet.")

    source = fleet_store.get_timeline_source()
    if source is None:
        raise HTTPException(status_code=404, detail="No telemetry retained for timelines yet.")

    try:
        points = build_health_timeline(source, battery_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"'{battery_id}' isn't in the most recent pipeline run's telemetry.",
        )
    return [TimelinePointOut(**p) for p in points]
