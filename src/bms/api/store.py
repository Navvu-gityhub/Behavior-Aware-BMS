"""In-memory fleet store backing the API.

Deliberately not a database. This is a demonstration/thesis deliverable
over an unvalidated heuristic pipeline (see docs/final_report.md) — adding
a persistence layer would suggest a durability guarantee this project
doesn't need yet and hasn't earned. State lives for the lifetime of the
API process; restarting it clears everything. If this ever needs to
survive a restart, that's a deliberate future decision (e.g. SQLite for a
single-process deployment), not an accident of this being in-memory.

Not thread-safe beyond what Python's GIL gives you for free. FastAPI's
default dev server is single-worker single-process, which is fine for a
demo; a real multi-worker deployment would need each worker's store
reconciled (e.g. moved into Redis) — out of scope here and noted so it
isn't quietly assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.bms.digital_twin import TwinSnapshot, TwinTransition, detect_transition, evaluate_fleet


@dataclass
class BatteryRecord:
    guardian_row: dict
    snapshot: TwinSnapshot
    transitions: list[TwinTransition] = field(default_factory=list)


class FleetStore:
    def __init__(self) -> None:
        self._batteries: dict[str, BatteryRecord] = {}
        self._last_behavior_features: Optional[pd.DataFrame] = None
        self._n_runs: int = 0

    def ingest_run(self, guardian_df: pd.DataFrame, behavior_features_df: pd.DataFrame) -> list[TwinTransition]:
        """Fold one pipeline run's Guardian output into the store.

        Returns the transitions produced by this run (batteries that
        changed twin state, including "first time seen" as a transition —
        see digital_twin.detect_transition). A battery not present in this
        run is left untouched, not removed — a partial re-run (e.g. one
        battery's new telemetry) shouldn't wipe out the rest of the fleet.
        """
        self._n_runs += 1
        self._last_behavior_features = behavior_features_df

        new_snapshots = evaluate_fleet(guardian_df)
        transitions: list[TwinTransition] = []

        for battery_id, snapshot in new_snapshots.items():
            guardian_row = guardian_df.loc[guardian_df["battery_id"] == battery_id].iloc[0].to_dict()
            existing = self._batteries.get(battery_id)
            previous_snapshot = existing.snapshot if existing is not None else None

            transition = detect_transition(previous_snapshot, snapshot)
            history = existing.transitions if existing is not None else []
            if transition is not None:
                history = history + [transition]
                transitions.append(transition)

            self._batteries[battery_id] = BatteryRecord(
                guardian_row=guardian_row, snapshot=snapshot, transitions=history
            )

        return transitions

    def list_batteries(self) -> list[BatteryRecord]:
        return list(self._batteries.values())

    def get_battery(self, battery_id: str) -> Optional[BatteryRecord]:
        return self._batteries.get(battery_id)

    def get_timeline_source(self) -> Optional[pd.DataFrame]:
        return self._last_behavior_features

    @property
    def n_runs(self) -> int:
        return self._n_runs

    @property
    def n_batteries(self) -> int:
        return len(self._batteries)


# Single process-wide store. FastAPI dependency injection could swap this
# for a per-request/test instance later; a module-level singleton is the
# right amount of ceremony for a single-worker demo service.
fleet_store = FleetStore()
