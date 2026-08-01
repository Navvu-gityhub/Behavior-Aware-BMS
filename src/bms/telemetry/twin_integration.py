"""Digital twin integration for telemetry runs.

`digital_twin/twin.py` already evaluates twin state from a Guardian row and
already detects transitions between successive snapshots. What it lacked was a
path from live or replayed telemetry, because the telemetry pipeline stopped at
the Guardian stage.

This module supplies that path and the state it needs to be useful, which is
history: a twin snapshot on its own says what condition a pack is in, while a
sequence of snapshots says whether it is getting worse. `detect_transition`
needs the previous snapshot, so something has to hold it.

Why history lives here and not in the pipeline
----------------------------------------------
`run_telemetry_pipeline` is a pure function of its inputs, which is what makes
replay and live capture provably identical. Giving it memory would break that:
two runs over the same log would produce different transition output depending
on what ran before.

So the store is separate and explicit. A caller that wants transitions passes the
same `TwinHistory` across runs; a caller that wants a stateless snapshot does not
pass one at all.

Bounded by construction
-----------------------
`TwinHistory` keeps a bounded deque per battery. An unbounded history behind a
long-running API process is a memory leak with a slow fuse, and the transition
logic only needs the previous snapshot. The retained window exists for the
timeline the frontend draws, not for the transition test.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterator, Mapping, Sequence

import pandas as pd

from src.bms.digital_twin.twin import (
    TwinSnapshot,
    TwinTransition,
    detect_transition,
    evaluate_twin_state,
)

# Snapshots retained per battery. Sized for the timeline a frontend renders,
# which is a few hundred points at most before it becomes unreadable.
DEFAULT_HISTORY_LIMIT = 500


@dataclass
class TwinHistory:
    """Bounded per-battery snapshot history, and the transitions it reveals."""

    limit: int = DEFAULT_HISTORY_LIMIT
    _snapshots: dict[str, Deque[TwinSnapshot]] = field(default_factory=dict)
    _transitions: dict[str, list[TwinTransition]] = field(default_factory=dict)

    def latest(self, battery_id: str) -> TwinSnapshot | None:
        history = self._snapshots.get(battery_id)
        return history[-1] if history else None

    def snapshots(self, battery_id: str) -> tuple[TwinSnapshot, ...]:
        return tuple(self._snapshots.get(battery_id, ()))

    def transitions(self, battery_id: str) -> tuple[TwinTransition, ...]:
        return tuple(self._transitions.get(battery_id, ()))

    @property
    def battery_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._snapshots))

    def record(self, snapshot: TwinSnapshot) -> TwinTransition | None:
        """Append a snapshot and return the transition it caused, if any.

        Delegates the transition test to `detect_transition` rather than
        comparing states here, so telemetry and batch runs cannot disagree about
        what counts as a transition.
        """
        previous = self.latest(snapshot.battery_id)
        transition = detect_transition(previous, snapshot)

        history = self._snapshots.setdefault(
            snapshot.battery_id, deque(maxlen=self.limit)
        )
        history.append(snapshot)

        if transition is not None:
            self._transitions.setdefault(snapshot.battery_id, []).append(transition)
            # Transitions are rarer than snapshots but still unbounded over a
            # long-running process, so they are trimmed to the same window.
            if len(self._transitions[snapshot.battery_id]) > self.limit:
                del self._transitions[snapshot.battery_id][0]
        return transition

    def clear(self, battery_id: str | None = None) -> None:
        if battery_id is None:
            self._snapshots.clear()
            self._transitions.clear()
            return
        self._snapshots.pop(battery_id, None)
        self._transitions.pop(battery_id, None)


@dataclass(frozen=True)
class TwinUpdate:
    """The twin outcome of one telemetry run."""

    snapshots: tuple[TwinSnapshot, ...] = ()
    transitions: tuple[TwinTransition, ...] = ()
    skipped_reason: str = ""

    @property
    def evaluated(self) -> bool:
        return bool(self.snapshots)

    def render(self) -> str:
        if not self.evaluated:
            return f"twin: not evaluated ({self.skipped_reason})"
        lines = [f"twin: {len(self.snapshots)} snapshot(s)"]
        for snapshot in self.snapshots:
            lines.append(
                f"  {snapshot.battery_id}: {snapshot.twin_state} "
                f"(health {snapshot.health_index:.1f}, "
                f"failure likelihood {snapshot.failure_likelihood:.3f}, "
                f"RUL {snapshot.rul_cycles})"
            )
        for transition in self.transitions:
            lines.append(
                f"  TRANSITION {transition.battery_id}: "
                f"{transition.from_state} -> {transition.to_state}"
            )
        return "\n".join(lines)


def evaluate_twin_from_guardian(
    guardian: pd.DataFrame,
    history: TwinHistory | None = None,
) -> TwinUpdate:
    """Evaluate twin state for every battery in a Guardian frame.

    Passing `history` enables transition detection across calls. Omitting it
    yields snapshots only, which is the right choice for a stateless request.

    An empty Guardian frame is a skip with a reason, not an error: the telemetry
    pipeline legitimately produces no Guardian rows when it refuses, and that
    refusal is already reported by `TelemetryResult`.
    """
    if guardian is None or guardian.empty:
        return TwinUpdate(
            skipped_reason=(
                "no Guardian rows to evaluate. The telemetry run either refused "
                "or produced no complete cycle; see TelemetryResult.refusals."
            )
        )

    snapshots: list[TwinSnapshot] = []
    transitions: list[TwinTransition] = []

    for _, row in guardian.iterrows():
        snapshot = evaluate_twin_state(row)
        snapshots.append(snapshot)
        if history is not None:
            transition = history.record(snapshot)
            if transition is not None:
                transitions.append(transition)

    return TwinUpdate(
        snapshots=tuple(snapshots), transitions=tuple(transitions)
    )


def snapshots_to_frame(snapshots: Sequence[TwinSnapshot]) -> pd.DataFrame:
    """Tabulate snapshots for the API and the report."""
    return pd.DataFrame([
        {
            "battery_id": s.battery_id,
            "twin_state": s.twin_state,
            "health_index": s.health_index,
            "failure_likelihood": s.failure_likelihood,
            "rul_cycles": s.rul_cycles,
            "replacement_policy": s.replacement_policy,
            "evaluated_at": s.evaluated_at,
        }
        for s in snapshots
    ])


def transitions_to_frame(transitions: Sequence[TwinTransition]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "battery_id": t.battery_id,
            "from_state": t.from_state,
            "to_state": t.to_state,
            "at": t.at,
        }
        for t in transitions
    ])
