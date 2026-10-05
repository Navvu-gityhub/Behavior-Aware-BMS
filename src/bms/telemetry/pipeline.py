"""End-to-end telemetry pipeline: CAN frames to Guardian output.

Wires the existing stages together for live and recorded telemetry:

    source -> decode (DBC) -> unified schema -> resample to cycles
           -> behaviour features -> risk -> health -> RUL -> Guardian -> twin

Nothing here reimplements a stage. Every computation is delegated to the module
that already owns it, so a telemetry run and a dataset run produce numbers by the
same code path. That matters more than convenience: if this module had its own
feature extraction, a discrepancy between live and batch results would be
untraceable.

Where this pipeline refuses
--------------------------
Three refusals, each for a reason established earlier in the project.

**Missing signal channels.** Checked before decoding. The example DBC has no
temperature signal, and `compute_behavior_flags` needs one for `high_temp_flag`.
The NaN-as-healthy fix established why this must refuse: a NumPy comparison
against NaN is False, so an absent temperature would have silently produced "not
hot" for every row and a healthy score for a pack nobody measured.

**No complete discharge cycle.** SOH is capacity relative to initial capacity,
and capacity is only comparable across equal depths of discharge. Real driving
produces mostly partial cycles, so a log can decode perfectly and still support
no SOH figure. `TelemetryResult.capacity_yield` reports how many usable points
were found.

**Fade prediction.** `AdaptiveCalibrator.score` refuses while nothing has passed
the promotion gate, which is the current state. This pipeline does not route
around that. It emits the rule-based severity index, which is labelled
throughout as triage rather than measurement, and does not present it as a
calibrated fade prediction.

The last one is the point of the whole exercise. Wiring a model the gate rejected
into a live dashboard, where it would look authoritative, is precisely the
failure this project was built to prevent.

What is measured rather than scored
-----------------------------------
SOH from the voltage window (`health/field_soh.py`) and RUL from the fade trend
(`rul/fade_extrapolation.py`) need voltage, current and time and nothing else.
They run before the behaviour scoring and survive its refusals, so a log with
no temperature channel still gets its capacity fade measured. Where measured
SOH exists and the log starts at beginning of life, it - not the heuristic
index - sets `battery_state`, and `state_basis` says which did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator, Mapping

import pandas as pd

# The one scoring-layer import at module scope. The rest are deferred inside
# `_score_cycles` to keep the API's cold start cheap, but this is a float
# constant in a module whose only dependency is pandas - which is already
# imported here - and duplicating its value would put the same assumption in
# two files that could then disagree.
from src.bms.features.behavior_features import DEFAULT_RATED_CAPACITY_AH
from src.bms.health.field_soh import (
    FieldSOH,
    current_field_soh,
    learned_field_soh,
    state_from_soh,
)
from src.bms.rul.fade_extrapolation import (
    DEFAULT_EOL_THRESHOLD,
    RULEstimate,
    rul_from_cycle_capacity,
)
from src.bms.telemetry.cycles import (
    REST_THRESHOLD_A,
    CapacityYield,
    CycleMeasurement,
    capacity_yield,
    cycles_to_frame,
    measure_cycles,
)
from src.bms.telemetry.sources import (
    CanFrameSource,
    SignalCoverage,
    check_signal_coverage,
)
from src.bms.telemetry.twin_integration import (
    TwinHistory,
    TwinUpdate,
    evaluate_twin_from_guardian,
)

# Default mapping from the example DBC's signal names to unified schema
# channels. Deliberately incomplete, because the DBC is: there is no temperature
# signal to map. See `sources.REQUIRED_CHANNELS`.
TWIZY_SIGNAL_MAP: Mapping[str, str] = {
    "v_b_current": "current_a",
    "v_b_soc": "soc",
}


@dataclass(frozen=True)
class TelemetryResult:
    """Everything one telemetry run produced, including its refusals."""

    source: str
    n_frames: int
    n_decoded: int
    coverage: SignalCoverage
    telemetry: pd.DataFrame = field(default_factory=pd.DataFrame)
    cycles: pd.DataFrame = field(default_factory=pd.DataFrame)
    measurements: tuple[CycleMeasurement, ...] = ()
    guardian: pd.DataFrame = field(default_factory=pd.DataFrame)
    twin: TwinUpdate | None = None
    yield_summary: CapacityYield | None = None
    refusals: tuple[str, ...] = ()
    stages_completed: tuple[str, ...] = ()
    # Measured health, computed from voltage, current and time alone. These do
    # not depend on the behaviour scoring and survive its refusal: a log with no
    # temperature channel cannot be scored for heat stress, but its capacity
    # fade is still measurable, and discarding it would throw away the one
    # validated result the log can support.
    field_soh: FieldSOH | None = None
    rul_estimate: RULEstimate | None = None
    soh_reference: str = "start_of_log"

    @property
    def scored(self) -> bool:
        return not self.guardian.empty

    @property
    def status(self) -> str:
        if self.refusals and not self.scored:
            return "REFUSED"
        return "SCORED_WITH_REFUSALS" if self.refusals else "SCORED"

    def render(self) -> str:
        lines = [f"{self.source}: {self.status}"]
        lines.append(f"  frames read: {self.n_frames}, decoded: {self.n_decoded}")
        lines.append(f"  stages completed: {list(self.stages_completed)}")
        if self.yield_summary is not None:
            lines.append(f"  {self.yield_summary.render()}")
        if not self.coverage.complete:
            lines.append(f"  signal coverage: {self.coverage.status}")
            for channel in self.coverage.missing_channels:
                lines.append(f"    missing: {channel}")
        if self.field_soh is not None:
            if self.field_soh.available:
                lines.append(
                    f"  measured SOH: {self.field_soh.soh:.1%} at discharge "
                    f"{self.field_soh.at_cycle} ({self.field_soh.window} window, "
                    f"{self.field_soh.n_accepted}/{self.field_soh.n_discharges} "
                    f"discharges usable, reference: {self.soh_reference})"
                )
            else:
                lines.append(f"  measured SOH: not available - {self.field_soh.refusal}")
        if self.rul_estimate is not None:
            if math.isfinite(self.rul_estimate.rul_cycles):
                lines.append(
                    f"  remaining life: {self.rul_estimate.rul_cycles:.0f} cycles "
                    f"to SOH {DEFAULT_EOL_THRESHOLD:.2f} (fade extrapolation)"
                )
            else:
                lines.append(f"  remaining life: not available - {self.rul_estimate.refusal}")
        if self.twin is not None:
            lines.append("  " + self.twin.render().replace("\n", "\n  "))
        for refusal in self.refusals:
            lines.append(f"  REFUSED: {refusal}")
        return "\n".join(lines)


def decode_frames(
    frames: Iterator[tuple[float, int, bytes]],
    dbc,
    signal_map: Mapping[str, str],
    cell_id: str = "VEHICLE_01",
) -> tuple[pd.DataFrame, int, int]:
    """Decode raw frames into a unified-schema telemetry frame.

    Returns (frame, n_read, n_decoded). Undecodable frames are counted rather
    than raised on: a real bus carries messages a given DBC does not define, and
    aborting on the first one would make the pipeline unusable on any real
    vehicle.
    """
    rows: list[dict[str, float]] = []
    n_read = 0
    n_decoded = 0

    for timestamp, arbitration_id, payload in frames:
        n_read += 1
        try:
            decoded = dbc.decode_message(arbitration_id, payload)
        except Exception:
            # Unknown id, or a payload that does not match the definition.
            continue
        n_decoded += 1

        row: dict[str, float] = {"test_time_s": float(timestamp)}
        for signal, value in decoded.items():
            channel = signal_map.get(signal)
            if channel is None:
                continue
            try:
                row[channel] = float(value)
            except (TypeError, ValueError):
                # Named-value signals decode to strings; they are not numeric
                # channels and are skipped rather than coerced to a number.
                continue
        if len(row) > 1:
            rows.append(row)

    if not rows:
        return pd.DataFrame(), n_read, n_decoded

    telemetry = pd.DataFrame(rows)
    telemetry["cell_id"] = cell_id

    # A CAN bus interleaves messages, so each frame carries a subset of
    # channels. Group by timestamp and take the first non-null per channel to
    # assemble complete rows. Forward-filling across timestamps is deliberately
    # avoided: it would invent measurements between frames.
    telemetry = (
        telemetry.groupby(["cell_id", "test_time_s"], as_index=False).first()
        .sort_values("test_time_s")
        .reset_index(drop=True)
    )
    return telemetry, n_read, n_decoded


def run_telemetry_pipeline(
    source: CanFrameSource,
    dbc,
    signal_map: Mapping[str, str] = TWIZY_SIGNAL_MAP,
    cell_id: str = "VEHICLE_01",
    dbc_path: str = "<dbc>",
    require_full_coverage: bool = True,
    twin_history: TwinHistory | None = None,
) -> TelemetryResult:
    """Run CAN telemetry through the existing scoring stages.

    Passing `twin_history` enables digital-twin transition detection across
    calls. Omitting it keeps this function a pure function of its inputs, which
    is what makes replay and live capture provably identical - see
    `twin_integration.py` for why the history is external rather than held here.

    `require_full_coverage=False` allows the run to proceed past a missing
    channel so a caller can inspect the decoded telemetry. It does not make the
    scoring stages run: those still refuse, because the refusal is about the
    data, not about permission.
    """
    coverage = check_signal_coverage(dbc, signal_map, dbc_path)
    refusals: list[str] = []
    stages: list[str] = []

    if not coverage.complete:
        refusals.append(
            f"DBC does not supply {list(coverage.missing_channels)}. "
            + coverage.render().split("\n", 1)[1].strip()
        )
        if require_full_coverage:
            return TelemetryResult(
                source=source.name, n_frames=0, n_decoded=0, coverage=coverage,
                refusals=tuple(refusals), stages_completed=(),
            )

    telemetry, n_frames, n_decoded = decode_frames(
        source.frames(), dbc, signal_map, cell_id
    )
    stages.append("decode")

    if telemetry.empty:
        refusals.append(
            f"No frames decoded into mapped channels. Read {n_frames} frame(s), "
            f"decoded {n_decoded}. Check that the DBC matches the bus and that "
            f"signal_map names signals this DBC defines."
        )
        return TelemetryResult(
            source=source.name, n_frames=n_frames, n_decoded=n_decoded,
            coverage=coverage, refusals=tuple(refusals),
            stages_completed=tuple(stages),
        )

    return score_telemetry_frame(
        source_name=source.name, telemetry=telemetry, coverage=coverage,
        n_frames=n_frames, n_decoded=n_decoded, cell_id=cell_id,
        twin_history=twin_history, refusals=refusals, stages=stages,
    )


def score_telemetry_frame(
    source_name: str,
    telemetry: pd.DataFrame,
    coverage: SignalCoverage,
    n_frames: int,
    n_decoded: int,
    cell_id: str = "VEHICLE_01",
    twin_history: TwinHistory | None = None,
    refusals: list[str] | None = None,
    stages: list[str] | None = None,
    rated_capacity_ah: float | None = DEFAULT_RATED_CAPACITY_AH,
    unit: str = "cell",
    rest_threshold_a: float = REST_THRESHOLD_A,
    reference_is_beginning_of_life: bool = False,
) -> TelemetryResult:
    """Score a unified-schema telemetry frame through the existing stages.

    This is everything downstream of acquisition: segmentation, coulomb
    counting, features, risk, health, RUL, Guardian and the twin. It is
    transport-agnostic on purpose.

    It was extracted from `run_telemetry_pipeline` when serial ingestion was
    added, and extracted rather than copied for the reason stated at the top of
    this module: if the serial path had its own segmentation and scoring, a
    disagreement between a CAN run and a serial run over the same battery would
    be untraceable. Both transports now converge on a unified-schema frame and
    share every line of what follows, so a difference in output is a difference
    in the telemetry.

    `refusals` and `stages` carry forward whatever the acquisition stage already
    recorded, so a transport-level refusal (a DBC missing a channel, a rig
    failing its schema check) survives into the final result rather than being
    replaced by scoring-stage findings.

    `rated_capacity_ah` is the denominator of every C-rate flag. `None` means no
    capacity is known, and the scoring stage is then refused rather than run
    against a guess - segmentation, capacity measurement and the yield summary
    still run, because none of them divide by it. The serial path resolves it
    from the rig's HELLO declaration or from its caller and passes `None` when
    it has neither; the CAN path still takes the module default, because a DBC
    carries no equivalent declaration and giving it one is separate work. That
    asymmetry is recorded rather than papered over - see
    `DEFAULT_RATED_CAPACITY_AH`.

    Measured health - SOH from the voltage window, RUL from the fade trend -
    is computed before and independently of the behaviour scoring, from
    voltage, current and time alone. It survives a scoring refusal.

    `reference_is_beginning_of_life` says whether this log starts when the
    cell was new. Measured SOH is always relative to the log's first
    discharges; only when those ARE beginning of life is it state of health,
    and only then does it set `battery_state`. Otherwise it is reported as
    fade since logging began and the state stays on the labelled heuristic.
    `False` is the default because a replayed bench capture is almost never
    of a new cell, and claiming otherwise would read an old cell as healthy.
    """
    refusals = list(refusals or [])
    stages = list(stages or [])

    measurements = measure_cycles(
        telemetry, cell_id=cell_id, rest_threshold_a=rest_threshold_a)
    summary = capacity_yield(measurements)
    stages.append("segment_cycles")

    soh_reference = (
        "beginning_of_life" if reference_is_beginning_of_life else "start_of_log")
    field_soh = _measure_field_soh(
        telemetry, measurements, cell_id, rest_threshold_a, rated_capacity_ah)
    if field_soh.available:
        stages.append("measure_soh")

    cycles = cycles_to_frame(measurements, complete_only=True)
    rul_estimate = rul_from_cycle_capacity(cycles) if not cycles.empty else None
    if rul_estimate is not None and rul_estimate.refusal and len(cycles) < len(measurements) // 2:
        # Say why history is short: the trend is extrapolated from COMPLETE
        # discharges, and a log of mostly partial ones has few, however long
        # it is. "Fewer than 30 cycles" alone reads as a log that is too new.
        rul_estimate = replace(rul_estimate, refusal=(
            f"{rul_estimate.refusal}. Remaining life is extrapolated from "
            f"complete discharges, and only {len(cycles)} of this log's "
            f"{len(measurements)} discharges were complete. Extrapolating the "
            f"partial-discharge SOH instead was tested and is not accurate enough "
            f"to report (reports/metrics/calce_field_rul/)"))
    if cycles.empty:
        refusals.append(
            "No complete discharge cycle found, so no capacity measurement is "
            "available and SOH cannot be computed. "
            + summary.render()
            + " Capacity is only comparable across equal depths of discharge, "
            "so partial cycles are excluded from the capacity trend rather than "
            "scaled. (Measured SOH from the voltage window does use partial "
            "discharges that cross the window; see field_soh.)"
        )
        return TelemetryResult(
            source=source_name, n_frames=n_frames, n_decoded=n_decoded,
            coverage=coverage, telemetry=telemetry,
            measurements=tuple(measurements), yield_summary=summary,
            refusals=tuple(refusals), stages_completed=tuple(stages),
            field_soh=field_soh, rul_estimate=rul_estimate,
            soh_reference=soh_reference,
        )

    # The two things the scoring stage needs and cannot substitute for: the
    # channels it reads, and the capacity it divides by. Both are refusals of
    # the same kind - a plausible stand-in would not make the answer uncertain,
    # it would make it confidently wrong - so they sit together, and both leave
    # the segmentation results above intact rather than discarding the run.
    guardian = pd.DataFrame()
    if not coverage.complete:
        refusals.append(
            "Scoring skipped: the feature layer requires channels this source "
            "does not supply, and treating them as absent-equals-safe is the "
            "NaN-as-healthy defect this project already fixed."
        )
    elif rated_capacity_ah is None:
        refusals.append(
            "Scoring skipped: no rated capacity is known for this cell, so "
            "C-rate cannot be computed. `aggressive_discharge_event` and "
            "`fast_charge_flag` are current divided by rated capacity, and the "
            "risk, health and RUL figures are all derived from them. A 3.4 Ah "
            "cell scored against an assumed 2.0 Ah would read 1.7 C while "
            "drawing 1 C and would trip both flags on every row of the "
            "capture. Declare `capacity_ah` in the rig's HELLO line, or pass "
            "`rated_capacity_ah`."
        )
    else:
        try:
            guardian = _score_cycles(
                telemetry, cycles, cell_id, rated_capacity_ah=rated_capacity_ah,
                field_soh=field_soh, rul_estimate=rul_estimate,
                soh_reference=soh_reference,
            )
            # Scope travels with the answer. Every coefficient and threshold in
            # the stages above was established on single cells; a pack is not a
            # big cell. See MEASUREMENT_UNITS in telemetry/serial_schema.py.
            if guardian is not None and not guardian.empty:
                guardian["measurement_unit"] = unit
                guardian["unit_validated"] = unit == "cell"
            stages.append("score")
        except ValueError as exc:
            # The feature and scoring layers raise ValueError deliberately when
            # data is absent or inconsistent, rather than treating missing as
            # safe. That is the explicit-failure path working.
            refusals.append(
                f"Scoring refused: {exc}. The feature layer raises on absent or "
                f"inconsistent data rather than treating it as healthy."
            )
        except (ImportError, AttributeError, KeyError) as exc:
            # These are wiring defects in this module, not properties of the
            # data. Labelling them as data refusals would hide real bugs, so
            # they are re-raised.
            raise RuntimeError(
                f"Telemetry pipeline is mis-wired against the scoring stages: "
                f"{type(exc).__name__}: {exc}. This is a defect in "
                f"telemetry/pipeline.py, not a property of the telemetry."
            ) from exc

    twin_update = evaluate_twin_from_guardian(guardian, twin_history)
    if twin_update.evaluated:
        stages.append("twin")

    return TelemetryResult(
        source=source_name, n_frames=n_frames, n_decoded=n_decoded,
        coverage=coverage, telemetry=telemetry, cycles=cycles,
        measurements=tuple(measurements), guardian=guardian,
        twin=twin_update, yield_summary=summary, refusals=tuple(refusals),
        stages_completed=tuple(stages),
        field_soh=field_soh, rul_estimate=rul_estimate,
        soh_reference=soh_reference,
    )


def _measure_field_soh(
    telemetry: pd.DataFrame,
    measurements: list[CycleMeasurement],
    cell_id: str,
    rest_threshold_a: float,
    rated_capacity_ah: float | None = None,
) -> FieldSOH:
    """Window SOH over EVERY discharge, partial ones included.

    The capacity path excludes partial discharges because their total charge
    is not comparable. The window does not need the total: a partial that
    crosses the window is a full measurement of the window. That difference is
    why this estimator exists, so it is given all of them, and its own gates
    decide which are usable.

    The fixed window is tried first, because it is the validated one. When it
    cannot be read - the driver never discharges across it - and the rated
    capacity is known, a window learned from the cell's own usage band is
    tried instead (`field_soh.learned_field_soh`). If both refuse, both
    reasons are reported, since either may be the one worth acting on.
    """
    discharges = cycles_to_frame(measurements, complete_only=False)
    try:
        fixed = current_field_soh(telemetry, discharges, cell_id, rest_threshold_a)
    except ValueError as exc:
        return FieldSOH(float("nan"), None, 0, len(discharges), True, "",
                        refusal=str(exc))
    if fixed.available or rated_capacity_ah is None or discharges.empty:
        return fixed
    try:
        learned, _ = learned_field_soh(
            telemetry, discharges, cell_id, rest_threshold_a, rated_capacity_ah)
    except ValueError:
        return fixed
    if learned.available:
        return learned
    return replace(fixed, refusal=(
        f"{fixed.refusal}. A window learned from this cell's own usage band "
        f"was also tried: {learned.refusal}"))


def _score_cycles(
    telemetry: pd.DataFrame,
    cycles: pd.DataFrame,
    cell_id: str,
    rated_capacity_ah: float = DEFAULT_RATED_CAPACITY_AH,
    field_soh: FieldSOH | None = None,
    rul_estimate: RULEstimate | None = None,
    soh_reference: str = "start_of_log",
) -> pd.DataFrame:
    """Delegate to the existing scoring stages, in the order `main.py` uses.

    The sequence, the merge and the column drops are deliberately identical to
    `run_pipeline`. Any divergence would mean a telemetry run and a dataset run
    could disagree for reasons no one could trace, which defeats the purpose of
    sharing the stages at all.

    Imported locally so `cycles.py` and `sources.py` stay importable without
    pulling in the whole scoring stack, which matters for the API's cold start.
    """
    from src.bms.features.behavior_features import (
        add_age_features,
        add_rolling_features,
        compute_behavior_flags,
        summarize_batteries,
    )
    from src.bms.guardian.guardian import generate_guardian_reports
    from src.bms.health.health_index import compute_health_index
    from src.bms.risk.stress_score import compute_risk_assessment, compute_stress_score
    from src.bms.rul.rul_estimation import compute_rul, replacement_policy

    # Attach the cycle each telemetry row belongs to, so age and rolling
    # features see a real cycle index rather than a constant.
    enriched = _attach_cycle_index(telemetry, cycles)

    flagged = compute_behavior_flags(enriched, rated_capacity_ah=rated_capacity_ah)
    flagged["stress_score"] = compute_stress_score(
        flagged, rated_capacity_ah=rated_capacity_ah
    )
    featured = add_rolling_features(flagged)
    featured = add_age_features(featured)

    summary = summarize_batteries(featured)
    risk = compute_risk_assessment(summary)
    health = compute_health_index(summary)
    merged = risk.merge(
        health.drop(columns=[
            "avg_stress", "avg_temp", "deep_discharge_duration",
            "fast_charge_duration", "aggressive_discharge_count", "avg_soc",
        ]),
        on="battery_id",
    )
    scored = compute_rul(merged)

    # Prefer the validated estimator where the log supports it. `compute_rul`
    # is a hand-picked weighting times an unvalidated constant; the trajectory
    # needed to do better is right here in `cycles`, and the per-battery
    # summary it scores on is the only reason it was ever unreachable.
    #
    # A refusal leaves the heuristic in place, still labelled unvalidated,
    # rather than blanking the field: a short bench capture legitimately
    # cannot support an extrapolation, and the caller is better served by a
    # marked estimate than by nothing.
    from src.bms.rul.rul_estimation import METHOD_FADE

    validated = (
        rul_estimate if rul_estimate is not None
        else rul_from_cycle_capacity(cycles)
    )
    if math.isfinite(validated.rul_cycles):
        scored["rul_cycles"] = int(round(validated.rul_cycles))
        scored["estimated_total_cycles"] = float(validated.eol_cycle)
        scored["replacement_policy"] = scored["rul_cycles"].apply(
            replacement_policy
        )
        scored["rul_method"] = METHOD_FADE
        scored["rul_validated"] = True
        scored["rul_refusal"] = ""
    else:
        scored["rul_refusal"] = validated.refusal

    _attach_measured_state(scored, field_soh, soh_reference)
    return generate_guardian_reports(scored)


def _attach_measured_state(
    scored: pd.DataFrame, field_soh: FieldSOH | None, soh_reference: str
) -> None:
    """Put measured SOH beside the heuristic, and let it decide state when it can.

    The heuristic health index is kept in its own column either way - it is
    still what the attribution explains - but `battery_state` is what a
    reader acts on, so it comes from a measurement whenever one exists and its
    reference is the cell's beginning of life. `state_basis` records which.
    """
    measured = field_soh is not None and field_soh.available
    scored["soh_measured"] = field_soh.soh if measured else float("nan")
    scored["soh_reference"] = soh_reference
    scored["soh_refusal"] = (
        "" if measured else (field_soh.refusal if field_soh is not None else
                             "measured SOH was not computed"))
    scored["state_basis"] = "heuristic_index"
    if measured and soh_reference == "beginning_of_life":
        scored["battery_state"] = state_from_soh(field_soh.soh)
        scored["state_basis"] = "measured_soh"


def _attach_cycle_index(
    telemetry: pd.DataFrame, cycles: pd.DataFrame
) -> pd.DataFrame:
    """Label each telemetry row with the discharge cycle it falls inside.

    Rows outside any complete discharge — charge phases, rests, and excluded
    partial discharges — are dropped rather than assigned to a neighbouring
    cycle. Assigning them would fold charging temperatures into a discharge
    cycle's aggregate and shift every per-cycle feature.
    """
    enriched = telemetry.copy()
    enriched["cycle"] = pd.NA

    for _, cycle_row in cycles.iterrows():
        inside = (
            (enriched["test_time_s"] >= cycle_row["start_time_s"])
            & (enriched["test_time_s"] <= cycle_row["end_time_s"])
        )
        enriched.loc[inside, "cycle"] = int(cycle_row["cycle"])

    enriched = enriched[enriched["cycle"].notna()].copy()
    enriched["cycle"] = enriched["cycle"].astype(int)
    return enriched.sort_values(["cell_id", "cycle"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def replay_log(
    path: Path | str,
    dbc,
    signal_map: Mapping[str, str] = TWIZY_SIGNAL_MAP,
    cell_id: str = "VEHICLE_01",
    dbc_path: str = "<dbc>",
    require_full_coverage: bool = True,
    twin_history: TwinHistory | None = None,
) -> TelemetryResult:
    """Replay a recorded CAN log through the same pipeline as live capture.

    Replay and live differ only in the source, which is the point: a result
    reproduced from a log is the same computation the vehicle produced, so a
    disagreement is a data difference rather than a code-path difference.
    """
    from src.bms.telemetry.sources import LogFileSource

    return run_telemetry_pipeline(
        LogFileSource(name=f"replay:{Path(path).name}", path=path),
        dbc=dbc, signal_map=signal_map, cell_id=cell_id, dbc_path=dbc_path,
        require_full_coverage=require_full_coverage, twin_history=twin_history,
    )
