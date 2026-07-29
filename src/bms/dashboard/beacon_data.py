"""Data preparation for the BEACON dashboard.

Separated from rendering (`beacon.py`) so the mapping from pipeline output to
displayed values is testable on its own, without parsing HTML.

DESIGN CONSTRAINT: NOTHING ON THE DASHBOARD IS INVENTED
-------------------------------------------------------
The visual design this dashboard implements was mocked up with placeholder
readings — state of health 92.4%, internal resistance 24.6 milliohms, "up
1.3% vs last week". None of those are quantities this pipeline computes.
Internal resistance is not in the unified schema at all; a week-over-week
delta requires timestamped history the pipeline does not retain; and state
of health requires measured capacity against a rated capacity, which exists
only for datasets that ship per-cycle capacity (NASA does, the simulator
does not).

The layout is reproduced faithfully. The *semantics* are not: every tile is
bound to a quantity the pipeline actually produces, and a tile whose
underlying data is absent renders in an explicit unavailable state rather
than showing a plausible number. A dashboard that invents a number is worse
than one that shows a gap, because the gap is self-correcting and the
invention is not.

The same principle drives `provenance`: a run on simulated telemetry is
labelled as such, prominently. A screenshot of this dashboard should never
be mistakable for measured results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

STATE_ORDER = ("HEALTHY", "WARNING", "DEGRADED", "CRITICAL")
RISK_ORDER = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Sentinel for a metric the current dataset cannot support. Rendered as an
# explicit "no data" tile; see module docstring.
UNAVAILABLE = "__unavailable__"


@dataclass
class Metric:
    """One displayed value plus everything needed to render it honestly."""

    label: str
    value: Any
    unit: str = ""
    qualifier: str = ""
    tone: str = "neutral"
    detail: str = ""
    available: bool = True

    @classmethod
    def missing(cls, label: str, reason: str) -> "Metric":
        return cls(label=label, value=UNAVAILABLE, detail=reason, available=False)


@dataclass
class BeaconData:
    provenance: dict[str, Any]
    fleet: dict[str, Any]
    batteries: list[dict[str, Any]] = field(default_factory=list)


def _tone_for_state(state: str) -> str:
    return {
        "HEALTHY": "good",
        "WARNING": "warn",
        "DEGRADED": "alert",
        "CRITICAL": "critical",
    }.get(str(state).upper(), "neutral")


def _tone_for_risk(level: str) -> str:
    return {
        "LOW": "good",
        "MEDIUM": "warn",
        "HIGH": "alert",
        "CRITICAL": "critical",
    }.get(str(level).upper(), "neutral")


def _series_for_battery(
    telemetry: pd.DataFrame | None,
    battery_id: str,
    column: str,
    max_points: int = 60,
) -> list[float]:
    """Downsample a per-cycle series for sparkline rendering.

    Returns [] when the column is absent, which the renderer turns into an
    empty-chart state rather than a flat line at zero — a flat line would
    read as "measured and constant", which is a different claim from
    "not measured".
    """
    if telemetry is None or column not in telemetry.columns:
        return []
    cell_col = "cell_id" if "cell_id" in telemetry.columns else None
    if cell_col is None:
        return []

    sub = telemetry[telemetry[cell_col] == battery_id]
    if sub.empty:
        return []

    if "cycle" in sub.columns:
        series = sub.groupby("cycle")[column].mean().sort_index()
    else:
        series = sub[column].reset_index(drop=True)

    values = series.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return []
    if len(values) > max_points:
        idx = np.linspace(0, len(values) - 1, max_points).astype(int)
        values = values[idx]
    return [round(float(v), 4) for v in values]


def _soh_series(telemetry: pd.DataFrame | None, battery_id: str) -> tuple[list[float], bool]:
    """Real state of health, only when measured capacity is present.

    SOH is capacity relative to the cell's own initial measured capacity. It
    is emphatically NOT `remaining_health` (the aging budget), which is a
    heuristic severity score on an unrelated scale. Conflating them is the
    single easiest way for this dashboard to state something false, so SOH
    is computed only where real capacity exists and is otherwise reported
    as unavailable.
    """
    if telemetry is None or "capacity_ah" not in telemetry.columns:
        return [], False
    cell_col = "cell_id" if "cell_id" in telemetry.columns else None
    if cell_col is None:
        return [], False

    sub = telemetry[telemetry[cell_col] == battery_id]
    if sub.empty or sub["capacity_ah"].dropna().empty:
        return [], False

    if "cycle" in sub.columns:
        cap = sub.groupby("cycle")["capacity_ah"].mean().sort_index().dropna()
    else:
        cap = sub["capacity_ah"].dropna()
    if len(cap) < 2:
        return [], False

    initial = float(cap.iloc[:5].mean())
    if initial <= 0:
        return [], False
    soh = (cap.to_numpy(dtype=float) / initial) * 100.0

    if len(soh) > 60:
        idx = np.linspace(0, len(soh) - 1, 60).astype(int)
        soh = soh[idx]
    return [round(float(v), 3) for v in soh], True


def _attribution_rows(row: pd.Series, prefix: str) -> list[dict[str, Any]]:
    """Extract Shapley contributions into a renderable, sorted list."""
    labels = {
        "stress": "Sustained stress",
        "temperature": "Temperature exposure",
        "deep_discharge": "Deep discharge",
        "fast_charge": "Fast charging",
        "aggressive_discharge": "Aggressive discharge",
        "soc_extremes": "SOC extremes",
    }
    rows = []
    for key, label in labels.items():
        col = f"{prefix}shap_{key}"
        if col in row.index and pd.notna(row[col]):
            rows.append({"label": label, "value": round(float(row[col]), 2)})
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return rows


def _usage_breakdown(telemetry: pd.DataFrame | None, battery_id: str) -> list[dict[str, Any]]:
    """Share of telemetry rows carrying each behaviour flag.

    Flags are not mutually exclusive (a row can be both hot and deep-
    discharging), so these are independent incidence rates, not slices of a
    partition. The renderer labels the chart accordingly rather than
    presenting it as a pie of a whole.
    """
    if telemetry is None:
        return []
    cell_col = "cell_id" if "cell_id" in telemetry.columns else None
    if cell_col is None:
        return []
    sub = telemetry[telemetry[cell_col] == battery_id]
    if sub.empty:
        return []

    flags = [
        ("fast_charge_flag", "Fast charging", "#38bdf8"),
        ("deep_discharge_flag", "Deep discharge", "#fb923c"),
        ("high_temp_flag", "High temperature", "#f87171"),
        ("high_soc_flag", "High SOC", "#a78bfa"),
        ("aggressive_discharge_event", "Aggressive discharge", "#facc15"),
    ]
    total = len(sub)
    out = []
    for col, label, color in flags:
        if col in sub.columns:
            share = float(sub[col].fillna(0).astype(float).mean())
            out.append({"label": label, "share": round(share * 100, 1), "color": color})
    nominal = max(0.0, 100.0 - sum(o["share"] for o in out))
    out.append({"label": "Nominal", "share": round(nominal, 1), "color": "#334155"})
    return out


def build_beacon_data(
    guardian: pd.DataFrame,
    telemetry: pd.DataFrame | None = None,
    data_source: str = "simulated",
    dataset_label: str = "Synthetic fleet telemetry",
) -> BeaconData:
    """Assemble everything the BEACON dashboard renders.

    `guardian` is the output of `guardian.generate_guardian_reports`.
    `telemetry` is the optional per-row feature table
    (`data/features/behavior_features_v1.csv`); without it, per-cycle trends
    and usage breakdowns render as unavailable rather than as flat or
    fabricated series.
    """
    if guardian.empty:
        raise ValueError("build_beacon_data: guardian table is empty")

    required = {"battery_id", "health_index", "battery_state", "rul_cycles"}
    missing = required - set(guardian.columns)
    if missing:
        raise ValueError(f"build_beacon_data: missing required columns {sorted(missing)}")

    is_measured = data_source.lower() not in ("simulated", "synthetic")

    provenance = {
        "data_source": data_source,
        "dataset_label": dataset_label,
        "is_measured": is_measured,
        "n_batteries": int(len(guardian)),
        "n_telemetry_rows": int(len(telemetry)) if telemetry is not None else 0,
        "warning": (
            ""
            if is_measured
            else "Simulated telemetry. Values shown are generated by the project's own "
                 "synthetic fleet model and are NOT measurements of any real battery."
        ),
    }

    state_counts = {s: int((guardian["battery_state"] == s).sum()) for s in STATE_ORDER}
    risk_counts = (
        {r: int((guardian["risk_level"] == r).sum()) for r in RISK_ORDER}
        if "risk_level" in guardian.columns
        else {}
    )

    fleet = {
        "n_batteries": int(len(guardian)),
        "state_counts": state_counts,
        "risk_counts": risk_counts,
        "mean_health_index": round(float(guardian["health_index"].mean()), 1),
        "mean_rul_cycles": int(round(float(guardian["rul_cycles"].mean()))),
        "worst_battery": str(
            guardian.loc[guardian["health_index"].idxmax(), "battery_id"]
        ),
        "n_needing_action": int(
            guardian["battery_state"].isin(["DEGRADED", "CRITICAL"]).sum()
        ),
        # Resolution of the score across the fleet. Surfaced because the
        # calibration work found the v1 index takes very few distinct values
        # on real data (6 across 33 NASA cells), and a dashboard that hides
        # that makes the score look more informative than it is.
        "distinct_health_values": int(guardian["health_index"].nunique()),
    }

    batteries = []
    for _, row in guardian.iterrows():
        bid = str(row["battery_id"])
        soh, soh_ok = _soh_series(telemetry, bid)

        params = []
        for col, label, unit, fmt in [
            ("avg_temp", "Mean temperature", "°C", "{:.1f}"),
            ("avg_soc", "Mean state of charge", "%", "{:.1f}"),
            ("avg_stress", "Mean stress score", "", "{:.1f}"),
            ("deep_discharge_duration", "Deep-discharge rows", "", "{:.0f}"),
            ("fast_charge_duration", "Fast-charge rows", "", "{:.0f}"),
            ("aggressive_discharge_count", "Aggressive-discharge events", "", "{:.0f}"),
            ("equivalent_aging_factor", "Equivalent aging factor", "", "{:.3f}"),
            ("estimated_total_cycles", "Estimated total cycles", "", "{:.0f}"),
        ]:
            if col in row.index and pd.notna(row[col]):
                params.append({"label": label, "value": fmt.format(float(row[col])), "unit": unit})

        batteries.append(
            {
                "id": bid,
                "state": str(row["battery_state"]),
                "state_tone": _tone_for_state(row["battery_state"]),
                "health_index": round(float(row["health_index"]), 1),
                "remaining_health": round(float(row.get("remaining_health", np.nan)), 1)
                if pd.notna(row.get("remaining_health", np.nan))
                else None,
                "risk_score": round(float(row["risk_score"]), 1)
                if "risk_score" in row.index and pd.notna(row["risk_score"])
                else None,
                "risk_level": str(row.get("risk_level", "")),
                "risk_tone": _tone_for_risk(row.get("risk_level", "")),
                "rul_cycles": int(row["rul_cycles"]),
                "replacement_policy": str(row.get("replacement_policy", "")),
                "soh_series": soh,
                "soh_available": soh_ok,
                "soh_latest": round(soh[-1], 1) if soh_ok and soh else None,
                "temp_series": _series_for_battery(telemetry, bid, "temp_rolling_mean")
                or _series_for_battery(telemetry, bid, "temperature_c"),
                "stress_series": _series_for_battery(telemetry, bid, "stress_rolling_mean")
                or _series_for_battery(telemetry, bid, "stress_score"),
                "soc_series": _series_for_battery(telemetry, bid, "soc"),
                "usage": _usage_breakdown(telemetry, bid),
                "health_attribution": _attribution_rows(row, "health_"),
                "risk_attribution": _attribution_rows(row, "risk_"),
                "dominant_cause": str(row.get("dominant_cause", "normal usage")),
                "guardian_report": str(row.get("guardian_report", "")),
                "guardian_status": str(row.get("guardian_status", "")),
                "targeted_action": str(row.get("targeted_action", "")),
                "guardian_caveat": str(row.get("guardian_caveat", "")),
                "recommendation": str(row.get("recommendation", "")),
                "parameters": params,
            }
        )

    batteries.sort(key=lambda b: b["health_index"], reverse=True)
    return BeaconData(provenance=provenance, fleet=fleet, batteries=batteries)
