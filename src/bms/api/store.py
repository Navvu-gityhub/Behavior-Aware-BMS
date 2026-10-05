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

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

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
        self._state_file: Optional[Path] = None

    # -- persistence ------------------------------------------------------
    # Opt-in, so the default stays a stateless demo service. Set
    # BEACON_STATE_FILE to a path and the fleet (Guardian rows, twin snapshots,
    # transition history) is written there after every run and read back at
    # start-up, so a restart no longer loses it. The behaviour-feature
    # timeline is NOT persisted: it is large, and is rebuilt by the next run.
    def attach_state_file(self, path: str | Path) -> None:
        self._state_file = Path(path)
        if self._state_file.exists():
            self.load(self._state_file)

    def save(self, path: str | Path) -> None:
        payload = {
            "n_runs": self._n_runs,
            "batteries": {
                bid: {
                    "guardian_row": {k: _jsonable(v) for k, v in rec.guardian_row.items()},
                    "snapshot": asdict(rec.snapshot),
                    "transitions": [asdict(t) for t in rec.transitions],
                }
                for bid, rec in self._batteries.items()
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)   # atomic: a crash mid-write never leaves half a file

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._n_runs = int(data.get("n_runs", 0))
        self._batteries = {
            bid: BatteryRecord(
                guardian_row=rec["guardian_row"],
                snapshot=TwinSnapshot(**rec["snapshot"]),
                transitions=[TwinTransition(**t) for t in rec["transitions"]],
            )
            for bid, rec in data.get("batteries", {}).items()
        }

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

        if self._state_file is not None:
            self.save(self._state_file)
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
if os.environ.get("BEACON_STATE_FILE"):
    fleet_store.attach_state_file(os.environ["BEACON_STATE_FILE"])


def _jsonable(value: Any) -> Any:
    """Plain JSON for a Guardian cell: numpy scalars to Python, NaN to null."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
