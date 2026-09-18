"""End-to-end serial telemetry: rig lines to Guardian output.

    line source -> parse + validate -> unified schema -> coverage gate
                -> [shared scoring tail: cycles, features, risk, health, RUL,
                    Guardian, twin]

Only the first half of that is new. Everything from the coverage gate onward is
`pipeline.score_telemetry_frame`, the exact function the CAN path calls, so a
serial run and a CAN run over the same battery produce numbers by the same code.
That was the binding constraint on this module's design: a second segmentation
or a second feature extractor would make a live/batch disagreement untraceable,
which is the failure `telemetry/pipeline.py` was written to avoid in the first
place.

Serial is the demo adapter, CAN is the production path
------------------------------------------------------
This module exists so the project can be demonstrated on a bench rig a student
can actually build. It is not a replacement for the CAN path. A vehicle BMS
publishes on CAN; a breadboard with an INA219 publishes lines over USB. The two
transports converge on one unified-schema frame and one scoring pipeline, and
the CAN path remains the one aimed at a real pack.

Where this pipeline refuses
---------------------------
The CAN pipeline documents three refusals. This one inherits all three, because
it shares the stages that raise them, and adds three of its own that are
properties of a serial rig rather than of a bus.

**Incompatible or missing schema declaration.** A rig announces its fields in a
HELLO line before it streams. If it declares a schema id this parser does not
implement, the run refuses rather than guessing that the fields still mean what
they used to.

**Insufficient accepted records.** Serial is lossy, so individual bad lines are
counted and skipped. But a capture where most lines failed is not a capture with
some noise in it - it is evidence of a wrong baud rate, a half-connected cable,
or firmware that does not match this schema. Scoring the surviving minority
would produce a health index from an arbitrary subsample. The threshold is
explicit (`min_accepted_fraction`) rather than implicit.

**An SOC channel that is really a 0-1 fraction.** Caught by
`_plausibility_refusals`, not by the field range - a fraction lies inside the
declared 0-100 range and is therefore a valid record meaning the wrong thing.
The companion case, a reversed current shunt, is attached as a hint rather than
a refusal, because a charge-only capture is legitimate and the run refuses for
want of a discharge cycle regardless.

**An undeclared rated capacity.** The feature layer defines
`aggressive_discharge_event` and `fast_charge_flag` by C-rate, which is current
divided by the cell's rated capacity. That capacity used to be a default of
2.0 Ah buried in `features.behavior_features`, invisible at every call site on
this path. It was correct only by coincidence - the emulator's cell happens to
be 2.0 Ah - and a real 3.4 Ah 18650 drawing 1 C would have been scored at 1.7 C,
firing both flags on every row of the capture and propagating a fabricated
stress score into the health index and the RUL estimate. So the capacity is now
declared, by the rig in its HELLO line or by the caller, and a capture with
neither is refused. This is the one place on this path that used to default
silently instead of refusing, which made it the opposite of every other gate
here.

**Non-monotonic time.** A microcontroller that browns out and resets restarts
its millisecond counter at zero. Concatenating the two halves would place the
second session before the first in elapsed time, and coulomb counting integrates
over time - so charge from one session would cancel against the other. Sorting
the frame would hide this by silently interleaving two unrelated sessions, so
the run refuses and names the reset instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.bms.telemetry.pipeline import TelemetryResult, score_telemetry_frame
from src.bms.telemetry.serial_schema import (
    MAX_CAPACITY_AH,
    MIN_CAPACITY_AH,
    SCHEMA_ID,
    SchemaHeader,
    SerialDecodeStats,
    TelemetryRecord,
    parse_stream,
)
from src.bms.telemetry.serial_source import LineSource, LogFileLineSource
from src.bms.telemetry.sources import coverage_from_channels
from src.bms.telemetry.twin_integration import TwinHistory

#: Fraction of attempted data records that must parse for a capture to be
#: scored. Below this, the run refuses rather than scoring a subsample. Set at
#: one half because a rig losing more than half its lines has a transport or
#: firmware fault, not merely a noisy cable.
DEFAULT_MIN_ACCEPTED_FRACTION = 0.5

#: Fallback identity for a rig that does not declare `cell_id`. Named so it is
#: obvious in output that the rig did not identify itself.
UNDECLARED_CELL_ID = "SERIAL_RIG_UNDECLARED"


@dataclass(frozen=True)
class SerialTelemetryResult(TelemetryResult):
    """A `TelemetryResult` plus what only a serial capture can report.

    Subclassed rather than replaced so every existing consumer of a
    `TelemetryResult` - the API, the dashboard, the twin - accepts a serial run
    with no change. The extra fields are additive and defaulted, so nothing that
    reads the base type has to know serial ingestion exists.
    """

    stats: SerialDecodeStats | None = None
    header: SchemaHeader | None = None
    coverage_inferred: bool = False
    captured_at: datetime | None = None
    #: Rated capacity the C-rate flags were computed against, and where it came
    #: from. Carried on the result rather than left implicit so a reader of a
    #: stress score can see the denominator it was divided by.
    rated_capacity_ah: float | None = None
    capacity_source: str = ""

    @property
    def has_calendar_axis(self) -> bool:
        """Whether this capture carries wall-clock time, not just elapsed time.

        Daily and weekly usage aggregation needs a calendar, and a rig's `t`
        field is seconds since boot. Consumers should test this rather than
        probing for a `timestamp` column.
        """
        return self.captured_at is not None

    def render(self) -> str:
        lines = [super().render()]
        if self.header is not None:
            lines.append(f"  rig: {self.header.render()}")
        if self.captured_at is not None:
            lines.append(f"  capture started: {self.captured_at.isoformat()}")
        elif self.telemetry is not None and not self.telemetry.empty:
            lines.append(
                "  NOTE: elapsed time only; no wall-clock axis. Pass "
                "`captured_at` to place this capture on a calendar - daily and "
                "weekly aggregation cannot be computed without one."
            )
        if self.rated_capacity_ah is not None:
            lines.append(
                f"  C-rate basis: {self.rated_capacity_ah:g} Ah "
                f"({self.capacity_source})"
            )
        if self.coverage_inferred:
            lines.append(
                "  NOTE: no HELLO line seen; coverage was inferred from the "
                "records that arrived. A rig that declares its schema is "
                "checked before decoding rather than after."
            )
        if self.stats is not None:
            lines.append("  " + self.stats.render().replace("\n", "\n  "))
        return "\n".join(lines)


def run_serial_pipeline(
    source: LineSource,
    cell_id: str | None = None,
    require_full_coverage: bool = True,
    require_schema_header: bool = False,
    min_accepted_fraction: float = DEFAULT_MIN_ACCEPTED_FRACTION,
    twin_history: TwinHistory | None = None,
    captured_at: datetime | None = None,
    rated_capacity_ah: float | None = None,
) -> SerialTelemetryResult:
    """Run serial telemetry through the existing scoring stages.

    `cell_id` overrides whatever the rig declares. Left as None, the rig's own
    declared identity is used, falling back to `UNDECLARED_CELL_ID`.

    `rated_capacity_ah` overrides whatever the rig declares, and is the escape
    hatch for firmware that predates the `capacity_ah` HELLO field. Left as
    None, the rig's declaration is used; with neither, the run refuses. See the
    module docstring for why this stopped being a default.

    `require_full_coverage=False` allows a run to proceed past a missing channel
    so a caller can inspect the parsed telemetry. As on the CAN path, it does
    not make the scoring stages run: those still refuse, because the refusal is
    about the data, not about permission.

    `captured_at` is the wall-clock instant the capture began, supplied by the
    host. A rig's `t` field is seconds since boot, so elapsed time is all the
    wire carries; adding the host's start time is what turns it into a calendar
    axis, and a calendar axis is what daily and weekly usage aggregation needs.

    It is **not defaulted to "now"**, deliberately. Replaying a capture recorded
    last week would then stamp it with today's date and place a week-old session
    on the wrong days - inventing a calendar rather than recording one. Absent
    this argument, the frame carries elapsed time only and says so.
    """
    if captured_at is not None and captured_at.tzinfo is None:
        # Naive datetimes are the standard way a calendar axis goes quietly
        # wrong: the same local hour means different instants across a DST
        # boundary, so "usage by hour of day" over a multi-day capture would
        # mis-bucket an hour's worth of samples with nothing to indicate it.
        raise ValueError(
            "captured_at must be timezone-aware. A naive datetime cannot be "
            "placed on a calendar unambiguously - across a DST change the same "
            "local hour is two different instants - and daily/weekly "
            "aggregation would mis-bucket samples silently. Use e.g. "
            "datetime.now(timezone.utc), or attach the rig's local zone."
        )

    if rated_capacity_ah is not None and not (
        MIN_CAPACITY_AH <= rated_capacity_ah <= MAX_CAPACITY_AH
    ):
        # A caller's own argument is a programming error, so it raises rather
        # than becoming a refusal: refusals describe the data, exceptions
        # describe the call.
        raise ValueError(
            f"rated_capacity_ah={rated_capacity_ah!r} is outside the plausible "
            f"range [{MIN_CAPACITY_AH:g}, {MAX_CAPACITY_AH:g}] Ah. A 3400 mAh "
            f"cell is 3.4, not 3400."
        )

    stats = SerialDecodeStats()
    header: SchemaHeader | None = None
    records: list[TelemetryRecord] = []
    status_messages: list[str] = []

    for kind, payload in parse_stream(source.lines(), stats):
        if kind == "hello":
            # Last declaration wins. A rig that resets mid-capture re-announces
            # itself, and the later declaration describes the later records.
            header = payload
        elif kind == "status":
            status_messages.append(str(payload))
        elif kind == "data":
            records.append(payload)

    refusals: list[str] = []
    stages: list[str] = ["read"]

    resolved_cell_id = (
        cell_id or (header.cell_id if header and header.cell_id else None) or UNDECLARED_CELL_ID
    )

    # --- Schema declaration -------------------------------------------------
    if header is not None and not header.compatible:
        refusals.append(
            f"Rig declares schema {header.schema_id!r}, but this build "
            f"implements {SCHEMA_ID!r}. Field names may be identical and still "
            f"mean something different across a schema change, so the capture "
            f"is refused rather than decoded on the assumption they match."
        )
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=0,
                coverage=coverage_from_channels((), source.name),
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=False,
            captured_at=captured_at,
        )

    if header is None and require_schema_header:
        refusals.append(
            "No HELLO line seen and require_schema_header is set. The rig must "
            "declare its schema before streaming so coverage is checked before "
            "decoding rather than inferred from whatever arrived."
        )
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=0,
                coverage=coverage_from_channels((), source.name),
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=False,
            captured_at=captured_at,
        )

    coverage_inferred = header is None

    # --- Did anything arrive at all? ---------------------------------------
    # Checked before the coverage gate, and the order matters. When a capture is
    # empty, coverage is trivially incomplete, and reporting that first would
    # answer a question nobody asked: it would lecture about missing channels
    # when the actual finding is that the rig said nothing. Silence is the
    # failure a bench actually produces - wrong baud rate, sketch not running,
    # port held by a serial monitor - and it deserves its own diagnosis.
    if not records:
        refusals.append(
            f"No usable telemetry records. {stats.render()} "
            f"Check the baud rate, that the sketch is running, that no serial "
            f"monitor is holding the port, and that the rig's lines begin with "
            f"the BEACON sentinel."
        )
        if status_messages:
            refusals.append(f"Rig status: {'; '.join(status_messages[:5])}")
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=0,
                coverage=coverage_from_channels(
                    header.channels if header else (),
                    source_label=source.name,
                    transport="serial",
                ),
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=coverage_inferred,
            captured_at=captured_at,
        )
    stages.append("parse")

    # --- Coverage gate ------------------------------------------------------
    if header is not None:
        declared_channels = header.channels
    else:
        declared_channels = tuple(
            sorted({channel for record in records for channel in record.values})
        )

    coverage = coverage_from_channels(
        declared_channels, source_label=source.name, transport="serial"
    )

    if not coverage.complete:
        refusals.append(
            f"Rig does not supply {list(coverage.missing_channels)}. "
            + coverage.render().split("\n", 1)[1].strip()
        )
        if require_full_coverage:
            # The refusal is about what this rig CANNOT support, not about what
            # it measured. Those records parsed, passed their range checks and
            # carry real readings on the channels that are present; assembling
            # them costs one pass and lets a caller report the measurements
            # beside the refusal. Nothing downstream is scored from this frame
            # on this path - `stages_completed` still stops at "parse".
            return _as_serial_result(
                TelemetryResult(
                    source=source.name,
                    n_frames=stats.n_lines,
                    n_decoded=stats.n_accepted,
                    coverage=coverage,
                    telemetry=records_to_frame(
                        records, resolved_cell_id, captured_at=captured_at
                    ),
                    refusals=tuple(refusals),
                    stages_completed=tuple(stages),
                ),
                stats=stats,
                header=header,
                coverage_inferred=coverage_inferred,
                captured_at=captured_at,
            )

    # --- Record homogeneity -------------------------------------------------
    # Every record must carry the channels this rig supplies. A record that
    # drops one mid-capture leaves that channel absent for its timestamp, and an
    # absent channel is not the same as a safe one - so it is dropped and
    # counted rather than admitted with a hole in it.
    records = _enforce_record_homogeneity(records, declared_channels, stats)
    if not records:
        refusals.append(
            f"No record carried the full channel set {list(declared_channels)} "
            f"this rig supplies. {stats.render()}"
        )
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=stats.n_accepted,
                coverage=coverage,
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=coverage_inferred,
            captured_at=captured_at,
        )

    # --- Enough of them to trust? ------------------------------------------
    if stats.accepted_fraction < min_accepted_fraction:
        refusals.append(
            f"Only {stats.accepted_fraction:.0%} of data records parsed, below "
            f"the {min_accepted_fraction:.0%} floor. That is a transport or "
            f"firmware fault rather than cable noise, and scoring the surviving "
            f"records would compute a health index from an arbitrary "
            f"subsample. {stats.render()}"
        )
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=stats.n_accepted,
                coverage=coverage,
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=coverage_inferred,
            captured_at=captured_at,
        )

    telemetry = records_to_frame(records, resolved_cell_id, captured_at=captured_at)

    # --- Monotonic time -----------------------------------------------------
    reset_index = _first_time_reversal(telemetry["test_time_s"])
    if reset_index is not None:
        previous = telemetry["test_time_s"].iloc[reset_index - 1]
        current = telemetry["test_time_s"].iloc[reset_index]
        refusals.append(
            f"Time went backwards at record {reset_index} "
            f"({previous:g}s -> {current:g}s), which is what a microcontroller "
            f"reset looks like on the wire. Coulomb counting integrates over "
            f"time, so charge from the session after the reset would cancel "
            f"against charge from before it. Sorting the frame would hide the "
            f"reset by interleaving two unrelated sessions, so the capture is "
            f"refused. Split the capture at the reset and score each session."
        )
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=stats.n_accepted,
                coverage=coverage,
                telemetry=telemetry,
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=coverage_inferred,
            captured_at=captured_at,
        )

    # --- Capture-level plausibility ----------------------------------------
    fatal, hints = _plausibility_refusals(telemetry)
    refusals.extend(fatal)
    refusals.extend(hints)
    if fatal:
        return _as_serial_result(
            TelemetryResult(
                source=source.name,
                n_frames=stats.n_lines,
                n_decoded=stats.n_accepted,
                coverage=coverage,
                telemetry=telemetry,
                refusals=tuple(refusals),
                stages_completed=tuple(stages),
            ),
            stats=stats,
            header=header,
            coverage_inferred=coverage_inferred,
            captured_at=captured_at,
        )

    # --- C-rate basis ------------------------------------------------------
    # Resolved here, where the rig's declaration is visible, but *enforced* in
    # `score_telemetry_frame` beside the coverage refusal it resembles. Refusing
    # here instead would discard the cycle measurements and capacity yield,
    # which do not divide by capacity and are exactly what a bring-up needs to
    # see when the metadata is what is missing.
    resolved_capacity, capacity_source = _resolve_capacity(rated_capacity_ah, header)

    base = score_telemetry_frame(
        source_name=source.name,
        telemetry=telemetry,
        coverage=coverage,
        n_frames=stats.n_lines,
        n_decoded=stats.n_accepted,
        cell_id=resolved_cell_id,
        twin_history=twin_history,
        refusals=refusals,
        stages=stages,
        rated_capacity_ah=resolved_capacity,
    )
    return _as_serial_result(
        base,
        stats=stats,
        header=header,
        coverage_inferred=coverage_inferred,
        captured_at=captured_at,
        rated_capacity_ah=resolved_capacity,
        capacity_source=capacity_source,
    )


def _resolve_capacity(
    override: float | None, header: SchemaHeader | None
) -> tuple[float | None, str]:
    """Decide which rated capacity the C-rate flags use, and record which won.

    Caller over rig, because the caller is the one who can see the cell. A rig's
    `NOMINAL_CAPACITY_AH` is whatever was compiled into the sketch, and a bench
    that swaps cells without reflashing is the ordinary case, not an exotic one.
    """
    if override is not None:
        if header is not None and header.capacity_ah is not None and not math.isclose(
            override, header.capacity_ah, rel_tol=1e-6
        ):
            return override, (
                f"caller override; the rig declared "
                f"{header.capacity_ah:g} Ah and was overridden"
            )
        return override, "supplied by the caller"
    if header is not None and header.capacity_ah is not None:
        return header.capacity_ah, "declared by the rig in its HELLO line"
    return None, "undeclared"


def records_to_frame(
    records: list[TelemetryRecord],
    cell_id: str,
    captured_at: datetime | None = None,
) -> pd.DataFrame:
    """Assemble validated records into a unified-schema telemetry frame.

    Unlike the CAN path, no grouping by timestamp is needed: a serial record
    carries every required channel at one instant by construction, because
    `serial_schema` rejects a partially populated record. A CAN bus interleaves
    messages and must reassemble; a rig samples all its sensors and prints one
    line.

    Rows are not forward-filled and not resampled, for the same reason the CAN
    path avoids it: filling between samples invents measurements that were never
    taken.

    When `captured_at` is supplied, a `timestamp` column is added as that
    instant plus each record's elapsed `test_time_s`. This is the only place a
    calendar axis enters the serial path. `test_time_s` is kept alongside it and
    remains what the scoring stages integrate over, so a wrong `captured_at`
    shifts the calendar without touching capacity or state of health.
    """
    frame = pd.DataFrame([record.as_row(cell_id) for record in records])
    frame = frame.reset_index(drop=True)
    if captured_at is not None and "test_time_s" in frame.columns:
        elapsed = pd.to_timedelta(pd.to_numeric(frame["test_time_s"], errors="coerce"), unit="s")
        frame["timestamp"] = pd.Timestamp(captured_at) + elapsed
    return frame


def _enforce_record_homogeneity(
    records: list[TelemetryRecord],
    channels: tuple[str, ...],
    stats: SerialDecodeStats,
) -> list[TelemetryRecord]:
    """Drop records that omit a channel this rig supplies, counting each.

    Charged against the same rejection counters the parser uses, so the accepted
    fraction reported at the end reflects everything discarded for any reason -
    otherwise a rig dropping a channel intermittently would still report 100%
    accepted.
    """
    required = set(channels)
    kept: list[TelemetryRecord] = []
    for record in records:
        missing = sorted(required - set(record.values))
        if missing:
            stats.n_accepted -= 1
            stats.record_rejection(f"record omits channel(s) {missing} that this rig supplies")
            continue
        kept.append(record)
    return kept


#: Minimum SOC span, in whatever units the rig sent, before the fraction-vs-
#: percent check is willing to conclude anything. A rig that barely moved gives
#: no evidence either way.
_SOC_SPAN_FOR_SCALE_CHECK = 0.2


def _plausibility_refusals(
    telemetry: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Catch the two miswirings that produce valid records meaning the wrong thing.

    Returns ``(fatal, hints)``. Fatal entries stop the run; hints are attached to
    the result so a refusal that happens anyway carries its likely cause.

    Neither check is expressible per record - see the note in `serial_schema`.
    Both look at the shape of the whole capture.
    """
    from src.bms.telemetry.cycles import REST_THRESHOLD_A

    fatal: list[str] = []
    hints: list[str] = []

    # --- SOC reported as a 0-1 fraction rather than a percentage ------------
    if "soc" in telemetry.columns:
        soc = pd.to_numeric(telemetry["soc"], errors="coerce").dropna()
        if not soc.empty:
            span = float(soc.max() - soc.min())
            if float(soc.max()) <= 1.0 and span >= _SOC_SPAN_FOR_SCALE_CHECK:
                fatal.append(
                    f"SOC never exceeds {soc.max():.3g} while sweeping a span of "
                    f"{span:.3g}, which is a 0-1 fraction reported in a field "
                    f"declared as percent. The feature layer compares SOC "
                    f"against 20.0 and 90.0, so scoring this capture would set "
                    f"deep_discharge_flag on every row and high_soc_flag on "
                    f"none, producing a confident and wrong risk profile. "
                    f"Multiply by 100 in the firmware rather than here - the "
                    f"unit belongs to the rig that measured it."
                )

    # --- Current sign convention reversed -----------------------------------
    if "current_a" in telemetry.columns:
        current = pd.to_numeric(telemetry["current_a"], errors="coerce").dropna()
        if not current.empty:
            has_discharge = bool((current < -REST_THRESHOLD_A).any())
            has_charge = bool((current > REST_THRESHOLD_A).any())
            if has_charge and not has_discharge:
                # A hint, not a refusal: a genuinely charge-only capture is a
                # legitimate thing to record. The run will refuse anyway for
                # want of a discharge cycle; this explains the most likely why.
                hints.append(
                    f"No sample fell below -{REST_THRESHOLD_A:g} A, but "
                    f"{int((current > REST_THRESHOLD_A).sum())} sample(s) rose "
                    f"above +{REST_THRESHOLD_A:g} A. If this rig was "
                    f"discharging, its current sign convention is inverted: "
                    f"this schema requires negative for discharge. Swap the "
                    f"shunt leads or negate the reading in firmware. If the "
                    f"capture really is charge-only, no SOH is obtainable from "
                    f"it either way, because capacity is measured over a "
                    f"discharge."
                )

    return fatal, hints


def _first_time_reversal(times: pd.Series) -> int | None:
    """Index of the first sample whose time precedes its predecessor, if any."""
    numeric = pd.to_numeric(times, errors="coerce")
    backwards = numeric.diff() < 0
    if not backwards.any():
        return None
    return int(backwards.idxmax())


def _as_serial_result(base: TelemetryResult, **extra: Any) -> SerialTelemetryResult:
    """Widen a `TelemetryResult` into a `SerialTelemetryResult`.

    A shallow field copy rather than `dataclasses.asdict`, which would deep-copy
    every DataFrame the result carries.
    """
    carried = {spec.name: getattr(base, spec.name) for spec in dataclass_fields(TelemetryResult)}
    return SerialTelemetryResult(**carried, **extra)


def replay_serial_capture(
    path: Path | str,
    cell_id: str | None = None,
    require_full_coverage: bool = True,
    min_accepted_fraction: float = DEFAULT_MIN_ACCEPTED_FRACTION,
    twin_history: TwinHistory | None = None,
    captured_at: datetime | None = None,
    rated_capacity_ah: float | None = None,
) -> SerialTelemetryResult:
    """Replay a recorded serial capture through the same pipeline as live capture.

    The serial analogue of `pipeline.replay_log`, and useful for the same two
    reasons: a demo that still runs when the hardware does not, and a result
    that can be reproduced later from the bytes the rig actually sent.

    `captured_at` should be the instant the *original* session began, not the
    instant of the replay. A capture file does not record it - the wire format
    carries elapsed time only - so it has to be supplied by whoever knows when
    the session ran, and is omitted rather than guessed.
    """
    return run_serial_pipeline(
        LogFileLineSource(name=f"replay:{Path(path).name}", path=path),
        cell_id=cell_id,
        require_full_coverage=require_full_coverage,
        min_accepted_fraction=min_accepted_fraction,
        twin_history=twin_history,
        captured_at=captured_at,
        rated_capacity_ah=rated_capacity_ah,
    )
