"""Serve the BEACON dashboard payload as JSON, for the React client.

The redesigned dashboard was prototyped as a Python-rendered static page. This
endpoint serves the *same* payload that renderer consumed, so the React client
displays the identical numbers without a second data layer existing anywhere.

That reuse is deliberate. `build_beacon_data` already decides what is available
and what is not - it marks a series unavailable rather than emitting a flat
line, and it carries the provenance block that says whether the fleet is
simulated, replayed or measured (ADR 0004). Rebuilding any of that in JavaScript
would create a second place for those judgements to live, and the two would
drift. The client renders; it does not decide.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.bms.api.store import fleet_store
from src.bms.dashboard.beacon_data import build_beacon_data

router = APIRouter()


@router.get(
    "/dashboard/beacon",
    tags=["dashboard"],
    summary="Everything the BEACON dashboard renders, as JSON",
)
def beacon_payload() -> dict[str, Any]:
    """Assemble the dashboard payload from the current fleet.

    Refuses rather than inventing a fleet. An empty store means no pipeline run
    has happened yet, which is a different thing from a fleet with no problems,
    and the client must be able to tell them apart.
    """
    records = fleet_store.list_batteries()
    if not records:
        raise HTTPException(
            status_code=404,
            detail=(
                "No pipeline run has been ingested yet, so there is no fleet to "
                "render. POST /pipeline/simulate first. This endpoint does not "
                "fabricate a fleet to fill the page."
            ),
        )

    guardian = pd.DataFrame([record.guardian_row for record in records])
    telemetry = fleet_store.get_timeline_source()

    try:
        data = build_beacon_data(
            guardian,
            telemetry,
            data_source="simulated",
            dataset_label="Synthetic fleet telemetry (simulate_fleet)",
        )
    except ValueError as exc:
        # build_beacon_data raises when the guardian table is missing a column
        # it needs. That is a property of the run, not a server fault.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return dataclasses.asdict(data)
