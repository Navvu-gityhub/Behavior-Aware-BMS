"""Battery Digital Twin: state, transitions, and a health timeline.

Scope, stated up front: this is a *presentation and orchestration* layer
over the existing pipeline outputs (`health.health_index`,
`rul.rul_estimation`, `guardian.guardian`) — it does not add a new
predictive model. `docs/final_report.md` documents at length that the
underlying health_index/risk_score/RUL heuristics are hand-tuned and not
validated against measured degradation (Sections 4.1, 4.3-4.6). This
module does not change that; it structures those same numbers into an
explicit state machine and exposes them for the API layer. Anywhere this
module's output is described (docs, API responses, a paper), it should
carry the same caveat the rest of the project already does: these are
rule-based diagnostics, not a validated predictive model.

Design decision: TWIN_STATE reuses `health.health_index`'s existing
HEALTHY/WARNING/DEGRADED/CRITICAL thresholds (30/60/80) via a 1:1 relabel,
rather than defining a fifth independent set of cut points for
"digital twin state" on top of the health index, risk score, and RUL
replacement-policy thresholds that already exist in this codebase. Four
parallel, independently hand-tuned threshold systems for what is largely
the same underlying signal (see `risk.stress_score`'s docstring, which
already flags this exact duplication between row-level and battery-level
scoring) would make the system harder to reason about, not more capable.
If validated calibration work (Section 4 of the final report) ever
produces a real predictive model, replacing the health_index formula
automatically updates this state machine too, since it derives from that
same field.

`failure_likelihood` is a monotonic transform of `health_index` (0-100 ->
0-1), NOT a calibrated statistical probability. It is named
`failure_likelihood` rather than `failure_probability` specifically to
avoid that implication, even though `docs/digital_twin.md`'s original
sketch used "Failure Probability" — see final_report.md Section 4 for why
this project distinguishes those terms carefully after the Guardian
"Explainable AI" naming issue.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

TWIN_STATES = ("NORMAL", "MODERATE_RISK", "HIGH_RISK", "FAILURE_IMMINENT")

# 1:1 relabel of health.health_index._battery_state's HEALTHY/WARNING/
# DEGRADED/CRITICAL categories -- same thresholds, same source column,
# different names for this layer's vocabulary (docs/digital_twin.md).
_STATE_FROM_BATTERY_STATE = {
    "HEALTHY": "NORMAL",
    "WARNING": "MODERATE_RISK",
    "DEGRADED": "HIGH_RISK",
    "CRITICAL": "FAILURE_IMMINENT",
}

REQUIRED_COLUMNS = ("battery_id", "battery_state", "health_index", "rul_cycles", "replacement_policy")


@dataclass(frozen=True)
class TwinSnapshot:
    battery_id: str
    twin_state: str
    health_index: float
    failure_likelihood: float
    rul_cycles: int
    replacement_policy: str
    evaluated_at: str  # ISO-8601 UTC


@dataclass(frozen=True)
class TwinTransition:
    battery_id: str
    from_state: Optional[str]
    to_state: str
    at: str  # ISO-8601 UTC


def evaluate_twin_state(row: pd.Series) -> TwinSnapshot:
    """Build a single battery's twin snapshot from one row of the
    Guardian output table (`guardian.generate_guardian_reports`'s
    output, itself downstream of `rul.compute_rul`).

    Raises KeyError via pandas if a required field is missing -- this
    intentionally does not silently substitute a default the way the
    bug documented in final_report.md Section 4.4 did for missing
    temperature data.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in row.index]
    if missing:
        raise ValueError(f"evaluate_twin_state: missing required fields {missing}")

    twin_state = _STATE_FROM_BATTERY_STATE.get(row["battery_state"])
    if twin_state is None:
        raise ValueError(f"evaluate_twin_state: unrecognized battery_state {row['battery_state']!r}")

    failure_likelihood = round(float(row["health_index"]) / 100.0, 4)

    return TwinSnapshot(
        battery_id=str(row["battery_id"]),
        twin_state=twin_state,
        health_index=float(row["health_index"]),
        failure_likelihood=failure_likelihood,
        rul_cycles=int(row["rul_cycles"]),
        replacement_policy=str(row["replacement_policy"]),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
    )


def evaluate_fleet(guardian_df: pd.DataFrame) -> dict[str, TwinSnapshot]:
    """Evaluate every battery in a Guardian output table. Returns a dict
    keyed by battery_id so callers (the API service layer) can look up a
    single battery in O(1) without re-scanning the DataFrame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in guardian_df.columns]
    if missing:
        raise ValueError(f"evaluate_fleet: missing required columns {missing}")
    return {str(r["battery_id"]): evaluate_twin_state(r) for _, r in guardian_df.iterrows()}


def detect_transition(previous: Optional[TwinSnapshot], current: TwinSnapshot) -> Optional[TwinTransition]:
    """Compare two snapshots of the same battery and return a
    TwinTransition if the state actually changed, else None. `previous`
    is None for a battery's first-ever evaluation -- that also counts as
    a transition (into whatever state it starts in), matching the
    'State transitions' component in docs/digital_twin.md."""
    if previous is not None and previous.battery_id != current.battery_id:
        raise ValueError("detect_transition: previous and current snapshots are for different batteries")
    if previous is not None and previous.twin_state == current.twin_state:
        return None
    return TwinTransition(
        battery_id=current.battery_id,
        from_state=previous.twin_state if previous is not None else None,
        to_state=current.twin_state,
        at=current.evaluated_at,
    )


def build_health_timeline(behavior_features_df: pd.DataFrame, battery_id: str) -> list[dict]:
    """Per-cycle timeline for one battery from the row-level
    `behavior_features_v1.csv` output (`features.behavior_features`),
    for the dashboard/API to plot health trajectory over time. This is
    the raw simulated/ingested telemetry trend, not a twin-state
    trajectory (the twin state itself is only ever computed once per
    pipeline run per battery, at the whole-battery-summary level -- see
    module docstring on why there's no per-cycle twin state).

    Ordered by `cycle`, not a wall-clock timestamp: the unified schema
    (`preprocessing.schema`) and every downstream module in this project
    key on `cell_id`/`cycle`, not a timestamp column -- NASA and CALCE's
    raw data don't reliably provide one either. `cycle` is what every
    other part of this codebase already treats as the time axis.
    """
    required = ("cell_id", "cycle", "stress_score", "soc", "temperature_c")
    missing = [c for c in required if c not in behavior_features_df.columns]
    if missing:
        raise ValueError(f"build_health_timeline: missing required columns {missing}")

    sub = behavior_features_df[behavior_features_df["cell_id"] == battery_id].sort_values("cycle")
    if sub.empty:
        raise KeyError(f"build_health_timeline: no rows for battery_id {battery_id!r}")

    return sub[["cycle", "stress_score", "soc", "temperature_c"]].to_dict(orient="records")
