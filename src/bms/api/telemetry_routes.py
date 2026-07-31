"""Telemetry, twin and transfer-validation endpoints.

Mounted onto the existing app as a router, so nothing in `app.py` is replaced.
Every endpoint delegates to a domain module and translates the result; none of
them recomputes anything.

Two decisions worth defending.

**A refusal is a 200, not a 4xx.** A run that decoded forty thousand frames and
then declined to compute SOH because no discharge was complete has produced a
great deal of useful information. Collapsing that into an error status would
discard the decoded telemetry, the cycle measurements and the reason. HTTP errors
are reserved for malformed requests and missing files, which are failures of the
request rather than findings about the data.

**`fade_prediction` is always null, and says why.** `AdaptiveCalibrator.score`
refuses while nothing has passed the promotion gate (ADR 0005), and this API does
not route around that. The field exists so a client cannot mistake the rule-based
severity index for a calibrated fade prediction, and carries the refusal text
alongside. Omitting the field entirely would be worse: a client would reach for
`health_index` and treat it as a prediction, which is exactly the confusion ADR
0002 documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.bms.adaptive.dataset_specs import (
    REGISTRY as SPEC_REGISTRY,
    get_spec,
    predict_transfer_feasibility,
)
from src.bms.api.telemetry_schemas import (
    AxisVerdictOut,
    CapacityYieldOut,
    CycleMeasurementOut,
    DatasetSpecOut,
    FeasibilityOut,
    LiveCaptureRequest,
    LiveSampleOut,
    LiveStateOut,
    ReplayRequest,
    SignalCoverageOut,
    TelemetryRunOut,
    ThermalPointOut,
    ThermalTimelineOut,
    TwinHistoryOut,
    TwinTransitionOut,
    TwinUpdateOut,
)
from src.bms.telemetry import (
    LiveBusSource,
    LogFileSource,
    REQUIRED_CHANNELS,
    TelemetryResult,
    TwinHistory,
    check_signal_coverage,
    run_telemetry_pipeline,
    snapshots_to_frame,
)

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DBC = REPO_ROOT / "src/bms/io/dbc_examples/beacon_reference_pack.dbc"

# Signal map for the bundled reference DBC. A deployment with its own DBC
# supplies its own map; this one exists so the endpoints are usable out of the
# box against the DBC in the repository.
REFERENCE_SIGNAL_MAP: Mapping[str, str] = {
    "pack_voltage": "voltage_v",
    "pack_current": "current_a",
    "pack_soc": "soc",
    "pack_temp_mean": "temperature_c",
    "pack_temp_max": "max_temp",
}

# Known maps, tried in order when none is supplied. Selecting the map that
# resolves the most channels means checking an arbitrary DBC reports what the
# bus actually lacks, rather than reporting that a mismatched map names nothing.
KNOWN_SIGNAL_MAPS: Mapping[str, Mapping[str, str]] = {
    "beacon_reference_pack": REFERENCE_SIGNAL_MAP,
    "twizy": {"v_b_current": "current_a", "v_b_soc": "soc"},
}


def _best_signal_map(dbc) -> tuple[str, Mapping[str, str]]:
    """Pick the known map that resolves the most channels for this DBC."""
    from src.bms.telemetry import dbc_signal_names

    available = set(dbc_signal_names(dbc))
    scored = [
        (len(available & set(mapping)), name, mapping)
        for name, mapping in KNOWN_SIGNAL_MAPS.items()
    ]
    scored.sort(reverse=True)
    _, name, mapping = scored[0]
    return name, mapping

# Twin history and the last run per battery, kept in process.
#
# Same trade-off as `api/store.py`: this is a single-process store that resets
# with the process. It is honest about being that rather than pretending to
# durability it does not have, and the bound in `TwinHistory` keeps a
# long-running process from leaking.
_twin_history = TwinHistory()

# The run and the signal map it was decoded with are stored together, so
# /telemetry/latest reports the same map /telemetry/replay used rather than
# assuming the reference one.
_last_run: dict[str, tuple[TelemetryResult, Mapping[str, str]]] = {}


def _load_dbc(dbc_path: str | None):
    """Load a DBC, raising a 4xx for a bad path rather than a 500."""
    import cantools

    path = Path(dbc_path) if dbc_path else DEFAULT_DBC
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"DBC not found: {path}")
    try:
        return cantools.database.load_file(str(path)), str(path)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Could not parse DBC {path}: {exc}"
        ) from exc


def _coverage_out(
    coverage, signal_map: Mapping[str, str] | None = None
) -> SignalCoverageOut:
    explanation = "\n".join(
        f"{channel}: needed by {REQUIRED_CHANNELS.get(channel, 'downstream stages')}"
        for channel in coverage.missing_channels
    )
    return SignalCoverageOut(
        dbc_path=coverage.dbc_path,
        signal_map_used=dict(signal_map or {}),
        status=coverage.status,
        complete=coverage.complete,
        available_signals=list(coverage.available_signals),
        mapped_channels=list(coverage.mapped_channels),
        missing_channels=list(coverage.missing_channels),
        explanation=explanation,
    )


def _twin_out(update) -> TwinUpdateOut | None:
    if update is None:
        return None
    frame = snapshots_to_frame(update.snapshots)
    return TwinUpdateOut(
        evaluated=update.evaluated,
        snapshots=frame.to_dict(orient="records") if not frame.empty else [],
        transitions=[
            TwinTransitionOut(
                battery_id=t.battery_id, from_state=t.from_state,
                to_state=t.to_state, at=t.at,
            )
            for t in update.transitions
        ],
        skipped_reason=update.skipped_reason,
    )


_FADE_REFUSAL = (
    "No model has passed the promotion gate, so no calibrated fade prediction is "
    "available. Every candidate evaluated so far failed out-of-sample validation "
    "on a held-out protocol (ADR 0005). The health_index in `guardian` is a "
    "rule-based severity index for triage and is not a validated predictor of "
    "capacity fade: measured against real NASA fade it scores Spearman "
    "rho = -0.269, p = 0.12, n = 33."
)


def _run_out(
    result: TelemetryResult,
    battery_id: str,
    signal_map: Mapping[str, str] | None = None,
) -> TelemetryRunOut:
    cycles = [
        CycleMeasurementOut(
            cycle=int(m.cycle), capacity_ah=float(m.capacity_ah),
            is_complete=bool(m.is_complete), start_time_s=float(m.start_time_s),
            end_time_s=float(m.end_time_s), n_samples=int(m.n_samples),
            mean_current_a=float(m.mean_current_a),
            avg_temp=m.mean_temperature_c, max_temp=m.max_temperature_c,
            depth_of_discharge=m.depth_of_discharge,
            exclusion_reason=m.exclusion_reason,
        )
        for m in result.measurements
    ]

    yield_out = None
    if result.yield_summary is not None:
        summary = result.yield_summary
        yield_out = CapacityYieldOut(
            n_discharges=summary.n_discharges, n_complete=summary.n_complete,
            n_partial=summary.n_partial, usable_fraction=summary.usable_fraction,
            largest_discharge_ah=summary.largest_discharge_ah,
            summary=summary.render(),
        )

    guardian_records: list[dict] = []
    if not result.guardian.empty:
        # NaN is not valid JSON; None is.
        guardian_records = (
            result.guardian.replace({np.nan: None}).to_dict(orient="records")
        )

    return TelemetryRunOut(
        source=result.source, status=result.status, battery_id=battery_id,
        n_frames=result.n_frames, n_decoded=result.n_decoded,
        stages_completed=list(result.stages_completed),
        coverage=_coverage_out(result.coverage, signal_map or REFERENCE_SIGNAL_MAP),
        cycles=cycles, capacity_yield=yield_out, guardian=guardian_records,
        twin=_twin_out(result.twin), refusals=list(result.refusals),
        fade_prediction=None, fade_prediction_refusal=_FADE_REFUSAL,
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

@router.get("/telemetry/coverage", response_model=SignalCoverageOut,
            tags=["telemetry"])
def telemetry_coverage(dbc_path: str | None = None) -> SignalCoverageOut:
    """Check whether a DBC can drive the feature pipeline, before decoding.

    The bundled `twizy_bms_1.dbc` returns INCOMPLETE: it defines no temperature
    signal, and `compute_behavior_flags` needs one.
    """
    dbc, resolved = _load_dbc(dbc_path)
    _, signal_map = _best_signal_map(dbc)
    return _coverage_out(
        check_signal_coverage(dbc, signal_map, resolved), signal_map
    )


# ---------------------------------------------------------------------------
# Replay and live capture
# ---------------------------------------------------------------------------

@router.post("/telemetry/replay", response_model=TelemetryRunOut,
             tags=["telemetry"])
def telemetry_replay(request: ReplayRequest) -> TelemetryRunOut:
    """Replay a recorded CAN log through the full scoring pipeline."""
    log_path = Path(request.log_path)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail=f"CAN log not found: {log_path}")

    dbc, resolved = _load_dbc(request.dbc_path)
    # Select the map that fits this DBC, matching what /telemetry/coverage
    # reports. Hardcoding the reference map here would make coverage and replay
    # disagree about the same file: coverage would list the channels a bus
    # supplies while replay decoded none of them.
    _, signal_map = _best_signal_map(dbc)
    try:
        result = run_telemetry_pipeline(
            LogFileSource(name=f"replay:{log_path.name}", path=log_path),
            dbc=dbc, signal_map=signal_map,
            cell_id=request.battery_id, dbc_path=resolved,
            require_full_coverage=request.require_full_coverage,
            twin_history=_twin_history,
        )
    except ValueError as exc:
        # Malformed telemetry is a property of the request's file.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    _last_run[request.battery_id] = (result, signal_map)
    return _run_out(result, request.battery_id, signal_map)


@router.post("/telemetry/live", response_model=TelemetryRunOut,
             tags=["telemetry"])
def telemetry_live(request: LiveCaptureRequest) -> TelemetryRunOut:
    """Capture from a live CAN bus for a bounded duration, then score.

    The duration is capped by the request schema. An unbounded capture inside a
    request handler would never return.
    """
    dbc, resolved = _load_dbc(request.dbc_path)
    _, signal_map = _best_signal_map(dbc)
    source = LiveBusSource(
        name=f"live:{request.interface}:{request.channel}",
        channel=request.channel, interface=request.interface,
        duration_s=request.duration_s,
    )
    try:
        result = run_telemetry_pipeline(
            source, dbc=dbc, signal_map=signal_map,
            cell_id=request.battery_id, dbc_path=resolved,
            require_full_coverage=request.require_full_coverage,
            twin_history=_twin_history,
        )
    except (OSError, ValueError) as exc:
        # A missing or misconfigured interface is a request failure.
        raise HTTPException(
            status_code=422,
            detail=(
                f"Could not capture on {request.interface}:{request.channel}: "
                f"{exc}"
            ),
        ) from exc

    _last_run[request.battery_id] = (result, signal_map)
    return _run_out(result, request.battery_id, signal_map)


@router.get("/telemetry/latest/{battery_id}", response_model=TelemetryRunOut,
            tags=["telemetry"])
def telemetry_latest(battery_id: str) -> TelemetryRunOut:
    """The most recent run for a battery, from this process."""
    if battery_id not in _last_run:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No telemetry run recorded for '{battery_id}'. Run "
                f"POST /telemetry/replay or /telemetry/live first. Note that "
                f"this store is in-process and resets with the service."
            ),
        )
    result, signal_map = _last_run[battery_id]
    return _run_out(result, battery_id, signal_map)


@router.get("/telemetry/live/{battery_id}", response_model=LiveStateOut,
            tags=["telemetry"])
def telemetry_live_state(
    battery_id: str, window: int = Query(default=120, ge=1, le=5000)
) -> LiveStateOut:
    """Recent decoded samples, for the live view.

    `instrumented_channels` and `uninstrumented_note` are returned so a view
    cannot imply measurements that were never taken.
    """
    if battery_id not in _last_run:
        raise HTTPException(
            status_code=404, detail=f"No telemetry recorded for '{battery_id}'."
        )
    telemetry = _last_run[battery_id][0].telemetry
    if telemetry.empty:
        raise HTTPException(
            status_code=404,
            detail=f"The last run for '{battery_id}' decoded no telemetry rows.",
        )

    instrumented = [
        channel for channel in
        ("voltage_v", "current_a", "temperature_c", "soc")
        if channel in telemetry.columns
    ]
    absent = [
        channel for channel in
        ("voltage_v", "current_a", "temperature_c", "soc")
        if channel not in telemetry.columns
    ]

    def sample(row: pd.Series) -> LiveSampleOut:
        current = row.get("current_a")
        if current is None or pd.isna(current):
            mode = "rest"
        elif current < -0.5:
            mode = "discharge"
        elif current > 0.5:
            mode = "charge"
        else:
            mode = "rest"

        def value(name: str) -> float | None:
            raw = row.get(name)
            return None if raw is None or pd.isna(raw) else float(raw)

        return LiveSampleOut(
            test_time_s=float(row["test_time_s"]),
            voltage_v=value("voltage_v"), current_a=value("current_a"),
            temperature_c=value("temperature_c"), soc=value("soc"),
            mode=mode,
        )

    recent_frame = telemetry.tail(window)
    recent = [sample(row) for _, row in recent_frame.iterrows()]

    return LiveStateOut(
        battery_id=battery_id, n_samples=int(len(telemetry)),
        latest=recent[-1] if recent else None, recent=recent,
        instrumented_channels=instrumented,
        uninstrumented_note=(
            f"This bus does not report {absent}. Those views are unavailable "
            f"rather than zero." if absent else ""
        ),
    )


@router.get("/telemetry/thermal/{battery_id}", response_model=ThermalTimelineOut,
            tags=["telemetry"])
def telemetry_thermal(battery_id: str) -> ThermalTimelineOut:
    """Pack temperature over cycle and discharge progress.

    Both axes are measured. There is deliberately no per-cell axis: the unified
    schema carries a single pack-aggregate temperature channel, so a per-cell
    heatmap would mean inventing values that were never measured (ADR 0004).
    """
    if battery_id not in _last_run:
        raise HTTPException(
            status_code=404, detail=f"No telemetry recorded for '{battery_id}'."
        )
    result = _last_run[battery_id][0]
    telemetry = result.telemetry
    if telemetry.empty or "temperature_c" not in telemetry.columns:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No temperature channel in the last run for '{battery_id}'. "
                f"The bus did not report one, so there is no thermal view."
            ),
        )

    points: list[ThermalPointOut] = []
    for measurement in result.measurements:
        if not measurement.is_complete:
            continue
        window = telemetry[
            (telemetry["test_time_s"] >= measurement.start_time_s)
            & (telemetry["test_time_s"] <= measurement.end_time_s)
        ]
        temperatures = pd.to_numeric(
            window["temperature_c"], errors="coerce"
        ).to_numpy(float)
        if len(temperatures) == 0:
            continue
        fractions = (
            np.linspace(0.0, 1.0, len(temperatures)) if len(temperatures) > 1
            else np.array([0.0])
        )
        for fraction, temperature in zip(fractions, temperatures):
            if np.isnan(temperature):
                continue
            points.append(ThermalPointOut(
                cycle=int(measurement.cycle),
                phase_fraction=float(fraction),
                temperature_c=float(temperature),
            ))

    values = [p.temperature_c for p in points]
    return ThermalTimelineOut(
        battery_id=battery_id, points=points,
        n_cycles=len({p.cycle for p in points}),
        temperature_min_c=min(values) if values else None,
        temperature_max_c=max(values) if values else None,
    )


# ---------------------------------------------------------------------------
# Twin
# ---------------------------------------------------------------------------

@router.get("/telemetry/twin/{battery_id}", response_model=TwinHistoryOut,
            tags=["twin"])
def twin_history(battery_id: str) -> TwinHistoryOut:
    """Twin snapshots and state transitions accumulated for a battery."""
    snapshots = _twin_history.snapshots(battery_id)
    if not snapshots:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No twin history for '{battery_id}'. Transitions accumulate "
                f"across telemetry runs within one process."
            ),
        )
    frame = snapshots_to_frame(snapshots)
    return TwinHistoryOut(
        battery_id=battery_id, n_snapshots=len(snapshots),
        snapshots=frame.to_dict(orient="records"),
        transitions=[
            TwinTransitionOut(
                battery_id=t.battery_id, from_state=t.from_state,
                to_state=t.to_state, at=t.at,
            )
            for t in _twin_history.transitions(battery_id)
        ],
    )


# ---------------------------------------------------------------------------
# Transfer validation
# ---------------------------------------------------------------------------

@router.get("/transfer/feasibility", response_model=list[FeasibilityOut],
            tags=["transfer"])
def transfer_feasibility(source: str = "nasa") -> list[FeasibilityOut]:
    """Which transfers are scientifically admissible, from published metadata.

    This is the screen that refuses invalid experiments. NASA to Stanford comes
    back MARGINAL because the datasets vary along orthogonal axes (ADR 0006).
    """
    try:
        source_spec = get_spec(source)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    out: list[FeasibilityOut] = []
    for name in sorted(SPEC_REGISTRY):
        if name == source:
            continue
        prediction = predict_transfer_feasibility(source_spec, get_spec(name))
        out.append(FeasibilityOut(
            source=prediction.source, target=prediction.target,
            status=prediction.status,
            usable_axes=[a.value for a in prediction.usable_axes],
            marginal_axes=[a.value for a in prediction.marginal_axes],
            verdicts=[
                AxisVerdictOut(
                    axis=v.axis.value,
                    source_variation=v.source_variation.value,
                    target_variation=v.target_variation.value,
                    usable=v.usable, marginal=v.marginal, reason=v.reason,
                )
                for v in prediction.verdicts
            ],
        ))
    return out


@router.get("/datasets", response_model=list[DatasetSpecOut], tags=["transfer"])
def dataset_specs() -> list[DatasetSpecOut]:
    """Every registered dataset, with its documented variation and caveats."""
    return [
        DatasetSpecOut(
            name=spec.name, description=spec.description, n_cells=spec.n_cells,
            chemistry=spec.chemistry,
            nominal_capacity_ah=spec.nominal_capacity_ah,
            variation={
                axis.value: variation.value
                for axis, variation in spec.variation.axes.items()
            },
            caveats=list(spec.caveats), citation=spec.citation,
        )
        for spec in (SPEC_REGISTRY[name] for name in sorted(SPEC_REGISTRY))
    ]
