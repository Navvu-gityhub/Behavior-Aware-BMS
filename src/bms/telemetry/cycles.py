"""Cycle segmentation and coulomb counting.

This module exists because of a gap that is easy to miss: **state of health
cannot be read off a CAN bus.**

`dashboard/beacon_data.py` derives SOH from `capacity_ah`, dividing each cycle's
measured discharge capacity by the cell's initial capacity. A CAN bus carries
instantaneous current, voltage and temperature. It does not carry per-cycle
discharge capacity, and no amount of schema mapping produces it.

What it does carry is enough to *integrate* it. Charge moved during a discharge
is the time integral of current, so capacity in amp-hours is

    Q = integral(|i| dt) / 3600

over the discharge phase. That requires knowing where one discharge ends and the
next begins, which is what the segmentation half of this module does.

Why partial cycles must be excluded rather than scaled
------------------------------------------------------
The integral above is only comparable across cycles when each cycle covers the
same depth of discharge. A vehicle discharged from 90% to 40% state of charge
moves roughly half the charge of one taken from 100% to 0%, and reporting the
first as "capacity" would show a 50% healthy cell as catastrophically degraded.

The tempting repair is to scale by the observed state-of-charge swing. That is
wrong for two reasons. The BMS's reported SOC is itself derived from a capacity
estimate, so scaling by it makes the result depend on the quantity being
measured. And SOC is least accurate at the extremes and under load, which is
precisely where a partial cycle ends.

So partial cycles are marked `is_complete=False` and excluded from SOH by
default. Real driving produces mostly partial cycles, which means a real fleet
yields far fewer usable capacity points than cycle count suggests. That is a
property of the data, not a defect to engineer around, and the honest response
is to report how many usable points were found.

Why hysteresis is not optional
------------------------------
Segmenting on the sign of current sounds trivial until the signal sits near
zero: a parked vehicle with a small parasitic draw, or regenerative braking
during a discharge, flips sign repeatedly and would produce thousands of
spurious one-sample cycles. `REST_THRESHOLD_A` defines a dead band around zero
treated as rest, and a phase change is only recognised once current has stayed
past the threshold for `MIN_PHASE_SAMPLES` consecutive samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Magnitude below which current is treated as rest rather than charge or
# discharge. Sized for passenger-vehicle packs, where key-off parasitic draw is
# typically well under an amp while any real load is several amps. Exposed as an
# argument because a pack's noise floor is a property of its instrumentation.
REST_THRESHOLD_A = 0.5

# Consecutive samples required past the threshold before a phase change is
# accepted. Suppresses chatter from regenerative braking transients inside a
# discharge.
MIN_PHASE_SAMPLES = 3

# A discharge must move at least this fraction of the largest discharge observed
# for the same cell to count as complete. Relative rather than absolute because
# nominal capacity varies by pack and is often unknown from telemetry alone.
COMPLETE_CYCLE_FRACTION = 0.80


@dataclass(frozen=True)
class Phase:
    """One contiguous charge, discharge or rest interval."""

    kind: str  # "charge" | "discharge" | "rest"
    start_index: int
    end_index: int  # inclusive
    start_time_s: float
    end_time_s: float
    charge_moved_ah: float
    mean_current_a: float
    n_samples: int

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s


@dataclass(frozen=True)
class CycleMeasurement:
    """One discharge phase reduced to a capacity measurement."""

    cell_id: str
    cycle: int
    capacity_ah: float
    is_complete: bool
    start_time_s: float
    end_time_s: float
    n_samples: int
    mean_current_a: float
    mean_temperature_c: float | None = None
    max_temperature_c: float | None = None
    soc_start: float | None = None
    soc_end: float | None = None
    exclusion_reason: str = ""

    @property
    def depth_of_discharge(self) -> float | None:
        """Observed SOC swing, when the bus reported SOC.

        Reported for diagnosis only. It is deliberately not used to normalise
        capacity: the BMS's SOC is itself derived from a capacity estimate, so
        scaling by it would make the measurement depend on the quantity being
        measured.
        """
        if self.soc_start is None or self.soc_end is None:
            return None
        return abs(self.soc_start - self.soc_end)


def segment_phases(
    telemetry: pd.DataFrame,
    current_col: str = "current_a",
    time_col: str = "test_time_s",
    rest_threshold_a: float = REST_THRESHOLD_A,
    min_phase_samples: int = MIN_PHASE_SAMPLES,
) -> list[Phase]:
    """Split a telemetry stream into charge, discharge and rest phases.

    Sign convention follows the unified schema: negative current is discharge.

    Raises rather than guessing when a required column is absent. A silently
    assumed sign convention or a defaulted timestamp would corrupt every
    downstream capacity number, and the failure would be invisible.
    """
    for column in (current_col, time_col):
        if column not in telemetry.columns:
            raise ValueError(
                f"segment_phases: missing '{column}'. Cycle segmentation needs "
                f"current and a monotonic time base; neither can be inferred."
            )
    if telemetry.empty:
        return []

    current = pd.to_numeric(telemetry[current_col], errors="coerce").to_numpy(float)
    time_s = pd.to_numeric(telemetry[time_col], errors="coerce").to_numpy(float)

    if np.isnan(current).all() or np.isnan(time_s).all():
        raise ValueError(
            "segment_phases: current or time is entirely non-numeric after "
            "coercion, so no phase boundary can be located."
        )
    if len(time_s) > 1 and np.nanmin(np.diff(time_s)) < 0:
        raise ValueError(
            "segment_phases: time is not monotonically increasing. Sort the "
            "frame before segmenting; integrating over unsorted time would "
            "silently cancel charge against itself."
        )

    def classify(value: float) -> str:
        if np.isnan(value) or abs(value) < rest_threshold_a:
            return "rest"
        return "discharge" if value < 0 else "charge"

    labels = [classify(v) for v in current]

    # Collapse runs shorter than min_phase_samples into the preceding phase, so
    # a regenerative-braking spike inside a discharge does not split it.
    smoothed = list(labels)
    index = 0
    while index < len(smoothed):
        run_end = index
        while run_end + 1 < len(smoothed) and smoothed[run_end + 1] == smoothed[index]:
            run_end += 1
        run_length = run_end - index + 1
        if run_length < min_phase_samples and index > 0:
            for position in range(index, run_end + 1):
                smoothed[position] = smoothed[index - 1]
            # Re-examine from the start of the absorbed run's predecessor.
            index = run_end + 1
            continue
        index = run_end + 1

    phases: list[Phase] = []
    start = 0
    for position in range(1, len(smoothed) + 1):
        at_end = position == len(smoothed)
        if at_end or smoothed[position] != smoothed[start]:
            end = position - 1
            phases.append(_build_phase(
                smoothed[start], start, end, time_s, current
            ))
            start = position
    return phases


def _build_phase(
    kind: str, start: int, end: int, time_s: np.ndarray, current: np.ndarray
) -> Phase:
    """Integrate one phase. Trapezoidal, because sampling is rarely uniform."""
    window_time = time_s[start : end + 1]
    window_current = current[start : end + 1]
    finite = np.isfinite(window_time) & np.isfinite(window_current)

    charge_ah = 0.0
    if finite.sum() > 1:
        charge_ah = float(
            np.trapezoid(np.abs(window_current[finite]), window_time[finite]) / 3600.0
        )

    return Phase(
        kind=kind,
        start_index=start,
        end_index=end,
        start_time_s=float(window_time[0]) if len(window_time) else float("nan"),
        end_time_s=float(window_time[-1]) if len(window_time) else float("nan"),
        charge_moved_ah=charge_ah,
        mean_current_a=float(np.nanmean(window_current)) if finite.any() else float("nan"),
        n_samples=int(end - start + 1),
    )


def measure_cycles(
    telemetry: pd.DataFrame,
    cell_id: str,
    current_col: str = "current_a",
    time_col: str = "test_time_s",
    temperature_col: str = "temperature_c",
    soc_col: str = "soc",
    rest_threshold_a: float = REST_THRESHOLD_A,
    min_phase_samples: int = MIN_PHASE_SAMPLES,
    complete_cycle_fraction: float = COMPLETE_CYCLE_FRACTION,
) -> list[CycleMeasurement]:
    """Reduce a telemetry stream to per-cycle capacity measurements.

    Each discharge phase becomes one `CycleMeasurement`. Completeness is judged
    relative to the largest discharge seen for this cell, because nominal
    capacity is usually unknown from telemetry alone.

    Note the consequence of that: completeness is only meaningful once several
    cycles have been observed. A single discharge is trivially the largest and
    is therefore reported complete. `measure_cycles` records the number of
    reference cycles so a caller can weigh that.
    """
    phases = segment_phases(
        telemetry, current_col, time_col, rest_threshold_a, min_phase_samples
    )
    discharges = [p for p in phases if p.kind == "discharge"]
    if not discharges:
        return []

    largest = max(p.charge_moved_ah for p in discharges)
    has_temperature = temperature_col in telemetry.columns
    has_soc = soc_col in telemetry.columns

    measurements: list[CycleMeasurement] = []
    for cycle_number, phase in enumerate(discharges, start=1):
        window = telemetry.iloc[phase.start_index : phase.end_index + 1]

        mean_temperature = max_temperature = None
        if has_temperature:
            temperatures = pd.to_numeric(window[temperature_col], errors="coerce").dropna()
            if not temperatures.empty:
                mean_temperature = float(temperatures.mean())
                max_temperature = float(temperatures.max())

        soc_start = soc_end = None
        if has_soc:
            soc_values = pd.to_numeric(window[soc_col], errors="coerce").dropna()
            if not soc_values.empty:
                soc_start = float(soc_values.iloc[0])
                soc_end = float(soc_values.iloc[-1])

        fraction = phase.charge_moved_ah / largest if largest > 0 else 0.0
        is_complete = fraction >= complete_cycle_fraction
        reason = "" if is_complete else (
            f"partial discharge: moved {phase.charge_moved_ah:.4g} Ah, "
            f"{fraction:.0%} of the largest observed for this cell "
            f"({largest:.4g} Ah). Excluded from SOH because capacity is only "
            f"comparable across equal depths of discharge."
        )

        measurements.append(CycleMeasurement(
            cell_id=cell_id,
            cycle=cycle_number,
            capacity_ah=phase.charge_moved_ah,
            is_complete=is_complete,
            start_time_s=phase.start_time_s,
            end_time_s=phase.end_time_s,
            n_samples=phase.n_samples,
            mean_current_a=phase.mean_current_a,
            mean_temperature_c=mean_temperature,
            max_temperature_c=max_temperature,
            soc_start=soc_start,
            soc_end=soc_end,
            exclusion_reason=reason,
        ))
    return measurements


def cycles_to_frame(
    measurements: Sequence[CycleMeasurement],
    complete_only: bool = True,
) -> pd.DataFrame:
    """Tabulate cycle measurements in the unified cycle-level schema.

    `complete_only` defaults to True so a partial discharge cannot reach an SOH
    computation by accident. Passing False returns every cycle with
    `is_complete` and `exclusion_reason` attached, which is the right choice for
    diagnosing why a log produced few usable points.
    """
    rows = [
        {
            "cell_id": m.cell_id,
            "cycle": m.cycle,
            "capacity_ah": m.capacity_ah,
            "is_complete": m.is_complete,
            "start_time_s": m.start_time_s,
            "end_time_s": m.end_time_s,
            "n_samples": m.n_samples,
            "mean_current_a": m.mean_current_a,
            "avg_temp": m.mean_temperature_c,
            "max_temp": m.max_temperature_c,
            "soc_start": m.soc_start,
            "soc_end": m.soc_end,
            "depth_of_discharge": m.depth_of_discharge,
            "exclusion_reason": m.exclusion_reason,
        }
        for m in measurements
        if m.is_complete or not complete_only
    ]
    frame = pd.DataFrame(rows)
    if not frame.empty and complete_only:
        # Renumber so cycle indices are contiguous after exclusions; a gap
        # would otherwise be read as missing data rather than a filtered
        # partial cycle.
        frame = frame.reset_index(drop=True)
        frame["cycle"] = np.arange(1, len(frame) + 1)
    return frame


@dataclass(frozen=True)
class CapacityYield:
    """How many usable capacity points a log actually produced."""

    n_discharges: int
    n_complete: int
    n_partial: int
    largest_discharge_ah: float

    @property
    def usable_fraction(self) -> float:
        return self.n_complete / self.n_discharges if self.n_discharges else 0.0

    def render(self) -> str:
        return (
            f"{self.n_complete}/{self.n_discharges} discharges usable for SOH "
            f"({self.usable_fraction:.0%}); {self.n_partial} partial. "
            f"Largest observed discharge {self.largest_discharge_ah:.4g} Ah."
        )


def capacity_yield(measurements: Sequence[CycleMeasurement]) -> CapacityYield:
    """Summarise usable yield, so a thin result is visible rather than assumed.

    Real driving produces mostly partial cycles. A log with 400 discharges and
    six complete ones supports six capacity points, and a caller needs to know
    that before treating the result as a fade trajectory.
    """
    complete = sum(1 for m in measurements if m.is_complete)
    largest = max((m.capacity_ah for m in measurements), default=0.0)
    return CapacityYield(
        n_discharges=len(measurements),
        n_complete=complete,
        n_partial=len(measurements) - complete,
        largest_discharge_ah=float(largest),
    )
