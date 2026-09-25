"""Curve-level CALCE loader: per-cycle discharge Q(V) traces.

`load_calce_cycling.summarize_calce_cycles` reduces CALCE telemetry to one row
per cell-cycle. That aggregation is what every model in this project has been
fitted on, and it throws away the thing the two most-cited feature families in
the field actually use: the shape of the discharge curve.

`benchmarks/curves.py` implements Severson's Delta-Q variance and incremental
capacity analysis, and registers both as UNAVAILABLE with the reason
*"Feature math is implemented and tested; only the data is missing."* This
module is the missing data. It produces the long frame that file documents:

    cell_id | cycle | voltage_v | capacity_ah_curve

WHY THIS MATTERS TO THE PROJECT'S CENTRAL RESULT
------------------------------------------------
The benchmark study's one durable finding is that every method loses skill
under leave-one-cohort-out. Every method it could run, that is -- and all of
them were fitted on *usage aggregates*: mean temperature, mean C-rate, cycle
duration. Those describe how a cell was treated. They carry no information
about what the treatment did to the electrodes.

That is a candidate explanation for the collapse, not merely a criticism of
it. Curve-derived features are the literature's answer, and their construction
is chemistry-agnostic even though their values are not: "difference two
discharge curves and take the spread" is defined without reference to
chemistry, while a mean-temperature coefficient fitted on LCO is not.

So the experiment this module enables is specific: **re-run leave-one-cohort-out
with curve features in place of usage aggregates.** If skill survives the
cohort boundary where it previously collapsed, that is a finding. If it
collapses identically, that is a stronger version of the existing finding,
because it would no longer be attributable to weak features.

Neither outcome is assumed here. This module loads data.

WHAT `capacity_ah_curve` IS, AND WHY IT IS NOT `capacity_ah`
------------------------------------------------------------
Arbin's `Discharge_Capacity` counter **accumulates across cycles and does not
reset** -- the discovery documented at length in `summarize_calce_cycles`,
which cost this project a cell reporting 383% state of health. Within one
discharge segment the raw counter therefore starts at whatever the cell had
already delivered in its life, not at zero.

`capacity_ah_curve` is that counter re-zeroed to the start of the segment, so
it is charge delivered *within this discharge*, which is what Q(V) means. The
name is deliberately distinct from the cycle-level scalar `capacity_ah`, for
the reason `benchmarks/curves.py` gives: overloading one name to mean a
per-cycle total in one frame and a within-cycle trace in another is the unit
ambiguity that produced the CALCE initial-capacity artifact.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.bms.benchmarks.curves import (
    delta_q_variance_feature,
    ica_peak_features,
)
from src.bms.io.load_calce_cycling import (
    CalceLoadReport,
    load_calce_dataset,
)

# Columns `benchmarks/curves.py` requires, plus the grouping keys the
# validation protocol needs. The first four are the contract; `cohort` and
# `dataset` ride along because dropping them here would leave a frame that
# looks complete and cannot be split leave-one-cohort-out.
CURVE_FRAME_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "cycle",
    "voltage_v",
    "capacity_ah_curve",
)

# The two value columns of that contract, named separately because
# `curve_features_by_cell` validates them without requiring the keys.
_CURVE_VALUE_COLUMNS: tuple[str, ...] = ("voltage_v", "capacity_ah_curve")

# Feature columns `ica_peak_features` returns, so a refusal path can fill them
# with NaN without duplicating the names.
_ICA_COLUMNS: tuple[str, ...] = (
    "ica_peak_height",
    "ica_peak_voltage",
    "ica_area",
)

# A sample counts as part of the discharge if its current is at least this
# fraction of the cycle's own median discharge current.
#
# NOT a fixed amperage. CALCE's CS2 and CX2 cohorts differ by *discharge rate*
# -- that is the experimental axis the commensurability screen picked this
# dataset for. A fixed threshold would admit a different portion of the curve
# in each cohort, which would manufacture a cohort-dependent feature and
# corrupt exactly the leave-one-cohort-out comparison this frame exists to
# support. Scaling to each cycle's own rate keeps the selection rule identical
# across protocols.
DISCHARGE_RATE_FRACTION = 0.5

# Below this many points a segment is refused rather than interpolated. The
# floor matches `delta_q_variance_feature`, which returns NaN under 50 usable
# grid points; accepting a 12-point segment here would push a curve that
# cannot support the feature downstream to fail silently as a NaN.
MIN_CURVE_POINTS = 50

# Severson's early-life window. Both cycles must be present for a cell or the
# feature is refused for that cell -- `delta_q` raises rather than substituting
# a neighbouring cycle, because that would change what the feature measures.
DEFAULT_CYCLE_A = 10
DEFAULT_CYCLE_B = 100

# A discharge spanning less than this cannot support a Q(V) curve: interpolated
# onto the 2.0-4.2 V grid it would occupy a handful of bins and return a
# variance that is numerically defined and physically meaningless.
MIN_VOLTAGE_SPAN_V = 0.1


@dataclass(frozen=True)
class CurveExtractionReport:
    """What survived extraction, and why the rest did not.

    Returned rather than logged, for the reason the rest of this codebase
    returns its refusals: "412 of 868 cycles had no resolvable discharge
    segment" is a fact that changes what a caller may claim about the result,
    not a diagnostic to print and discard.
    """

    n_cells: int = 0
    n_cycles_seen: int = 0
    n_cycles_kept: int = 0
    n_points: int = 0
    refusals: Counter = field(default_factory=Counter)
    cells_without_curves: tuple[str, ...] = ()

    @property
    def yield_fraction(self) -> float:
        if not self.n_cycles_seen:
            return 0.0
        return self.n_cycles_kept / self.n_cycles_seen

    def summary(self) -> str:
        lines = [
            f"curve extraction: {self.n_cycles_kept}/{self.n_cycles_seen} "
            f"cycles kept ({self.yield_fraction:.1%}) across {self.n_cells} "
            f"cells, {self.n_points} sample points"
        ]
        for reason, count in self.refusals.most_common():
            lines.append(f"  refused {count:>6}  {reason}")
        if self.cells_without_curves:
            lines.append(
                "  cells yielding no curve at all: "
                + ", ".join(self.cells_without_curves)
            )
        return "\n".join(lines)


def _longest_discharge_run(mask: np.ndarray) -> slice | None:
    """The longest contiguous True run in `mask`, as a slice.

    A CALCE cycle is not guaranteed to hold exactly one discharge. Rest steps,
    impedance measurements and partial steps interleave, and concatenating two
    separated discharge segments into one "curve" would produce a Q(V) trace
    that doubles back on itself -- monotonic in neither variable, and silently
    wrong once `interpolate_curve` sorts it by voltage.

    Taking the longest run is a choice, not a law: it picks the main discharge
    and discards short excursions. It is recorded here so a reader knows a
    cycle with two comparable discharges contributed only one.
    """
    if not mask.any():
        return None
    best_start = best_len = 0
    start = -1
    for i, flag in enumerate(mask):
        if flag and start < 0:
            start = i
        elif not flag and start >= 0:
            if i - start > best_len:
                best_start, best_len = start, i - start
            start = -1
    if start >= 0 and len(mask) - start > best_len:
        best_start, best_len = start, len(mask) - start
    return slice(best_start, best_start + best_len)


def extract_discharge_curves(
    telemetry: pd.DataFrame,
    rate_fraction: float = DISCHARGE_RATE_FRACTION,
    min_points: int = MIN_CURVE_POINTS,
) -> tuple[pd.DataFrame, CurveExtractionReport]:
    """Reduce CALCE sample-level telemetry to per-cycle discharge Q(V) curves.

    Returns a long frame with one row per (cell, cycle, sample point) and a
    report naming every cycle that was refused and why.

    The discharge is selected by sign and magnitude of current, not by
    `step_index`. Step numbering is a property of the Arbin schedule file and
    differs between CS2 and CX2 protocols; current sign is a property of the
    cell. Selecting on the schedule would bind this loader to one experiment
    design, which is the opposite of what a cross-cohort feature needs.
    """
    required = {"cell_id", "cycle", "voltage_v", "current_a", "capacity_ah"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"extract_discharge_curves: missing {sorted(missing)}. Load with "
            f"load_calce_dataset, which supplies all five. Note this needs "
            f"sample-level telemetry, not the output of summarize_calce_cycles "
            f"-- the aggregation this module exists to avoid."
        )

    frame = telemetry.copy()
    frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
    frame = frame.dropna(subset=["cycle"])
    frame["cycle"] = frame["cycle"].astype(int)

    sort_key = "test_time_s" if "test_time_s" in frame.columns else "cycle"
    refusals: Counter = Counter()
    pieces: list[pd.DataFrame] = []
    seen = kept = 0
    barren: list[str] = []

    has_cohort = "cohort" in frame.columns

    for cell_id, cell_rows in frame.groupby("cell_id", sort=True):
        cell_kept = 0
        for cycle, rows in cell_rows.groupby("cycle", sort=True):
            seen += 1
            rows = rows.sort_values(sort_key)
            current = pd.to_numeric(rows["current_a"], errors="coerce").to_numpy(
                dtype=float
            )

            discharging = current < 0
            if not discharging.any():
                refusals["no discharge samples (current never negative)"] += 1
                continue
            if discharging.sum() < min_points:
                # Distinct from the case above on purpose. "No discharge" and
                # "a discharge too short to interpolate" have different causes
                # -- a charge-only or rest cycle versus a coarse sampling rate
                # -- and merging them would send a reader looking for the
                # wrong one.
                refusals[
                    f"fewer than {min_points} discharge samples in the cycle"
                ] += 1
                continue

            # Threshold relative to this cycle's own rate. See the constant.
            median_rate = float(np.median(np.abs(current[discharging])))
            if not np.isfinite(median_rate) or median_rate <= 0:
                refusals["discharge current not finite"] += 1
                continue
            at_rate = current <= -(rate_fraction * median_rate)

            span = _longest_discharge_run(at_rate)
            if span is None:
                refusals["no contiguous discharge segment at rate"] += 1
                continue

            segment = rows.iloc[span]
            if len(segment) < min_points:
                refusals[
                    f"discharge segment shorter than {min_points} points"
                ] += 1
                continue

            voltage = pd.to_numeric(
                segment["voltage_v"], errors="coerce"
            ).to_numpy(dtype=float)
            capacity = pd.to_numeric(
                segment["capacity_ah"], errors="coerce"
            ).to_numpy(dtype=float)

            finite = np.isfinite(voltage) & np.isfinite(capacity)
            if finite.sum() < min_points:
                refusals["too few finite voltage/capacity pairs"] += 1
                continue
            voltage, capacity = voltage[finite], capacity[finite]

            if voltage.max() - voltage.min() < MIN_VOLTAGE_SPAN_V:
                refusals[f"voltage span under {MIN_VOLTAGE_SPAN_V} V"] += 1
                continue

            piece = pd.DataFrame({
                "cell_id": cell_id,
                "cycle": int(cycle),
                "voltage_v": voltage,
                # Re-zero the accumulating counter. See the module docstring.
                "capacity_ah_curve": capacity - capacity.min(),
            })
            if has_cohort:
                piece["cohort"] = segment["cohort"].iloc[0]
            pieces.append(piece)
            kept += 1
            cell_kept += 1

        if cell_kept == 0:
            barren.append(str(cell_id))

    if not pieces:
        empty = pd.DataFrame(columns=list(CURVE_FRAME_COLUMNS))
        return empty, CurveExtractionReport(
            n_cells=int(frame["cell_id"].nunique()),
            n_cycles_seen=seen,
            refusals=refusals,
            cells_without_curves=tuple(barren),
        )

    curves = pd.concat(pieces, ignore_index=True)
    curves["dataset"] = "calce"
    report = CurveExtractionReport(
        n_cells=int(frame["cell_id"].nunique()),
        n_cycles_seen=seen,
        n_cycles_kept=kept,
        n_points=int(len(curves)),
        refusals=refusals,
        cells_without_curves=tuple(barren),
    )
    return curves, report


def curve_features_by_cell(
    curves: pd.DataFrame,
    cycle_a: int = DEFAULT_CYCLE_A,
    cycle_b: int = DEFAULT_CYCLE_B,
) -> pd.DataFrame:
    """One row per cell: Delta-Q variance and ICA peak features.

    These are the columns `benchmarks/curves.py` registers its UNAVAILABLE
    methods against -- `delta_q_variance` for the Severson model, and
    `ica_peak_height` / `ica_peak_voltage` / `ica_area` for the Dubarry-style
    degradation-mode features.

    Both are computed on the *early-life window* only (cycles `cycle_a` and
    `cycle_b`, default 10 and 100). That is the point of Severson's result:
    the feature is predictive before capacity fade is measurable, so computing
    it from late-life cycles would be a different and much less interesting
    claim. A cell that has not reached `cycle_b` is refused rather than
    evaluated on whatever its last cycle happens to be.

    `refusal_reason` is a column, not an exception. A cell missing cycle 100
    should appear in the frame saying so, because dropping it silently would
    make the cohort look smaller than it is for reasons a reader cannot see.
    """
    missing = [
        c for c in ("cell_id", "cycle", *_CURVE_VALUE_COLUMNS)
        if c not in curves.columns
    ]
    if missing:
        raise ValueError(
            f"curve_features_by_cell: missing {sorted(missing)}. Build the "
            f"frame with extract_discharge_curves."
        )

    rows: list[dict] = []
    for cell_id, cell_curves in curves.groupby("cell_id", sort=True):
        record: dict = {"cell_id": cell_id, "refusal_reason": ""}
        if "cohort" in cell_curves.columns:
            record["cohort"] = cell_curves["cohort"].iloc[0]

        available = set(cell_curves["cycle"].unique())
        absent = [c for c in (cycle_a, cycle_b) if c not in available]
        if absent:
            record["refusal_reason"] = (
                f"cycle(s) {absent} absent; the Delta-Q feature is defined on "
                f"two specific cycles and substituting the nearest available "
                f"one changes what it measures"
            )
            record["delta_q_variance"] = float("nan")
            record.update(dict.fromkeys(_ICA_COLUMNS, float("nan")))
            rows.append(record)
            continue

        record["delta_q_variance"] = delta_q_variance_feature(
            cell_curves, cycle_a, cycle_b
        )
        reference = cell_curves[cell_curves["cycle"] == cycle_b]
        record.update(ica_peak_features(
            reference["voltage_v"].to_numpy(dtype=float),
            reference["capacity_ah_curve"].to_numpy(dtype=float),
        ))
        if not np.isfinite(record["delta_q_variance"]):
            record["refusal_reason"] = (
                f"cycles {cycle_a} and {cycle_b} share too little voltage "
                f"overlap to estimate a variance"
            )
        rows.append(record)

    return pd.DataFrame(rows)


def load_calce_curve_dataset(
    base_dir: str | Path,
    rate_fraction: float = DISCHARGE_RATE_FRACTION,
    min_points: int = MIN_CURVE_POINTS,
) -> tuple[pd.DataFrame, CurveExtractionReport, list[CalceLoadReport]]:
    """Load every CALCE cell under `base_dir` and extract its discharge curves.

    Returns the curve frame, the extraction report, and the per-cell load
    reports from `load_calce_dataset` -- three objects rather than one, because
    a thin result has two possible causes (files that would not load, cycles
    that carried no resolvable discharge) and collapsing them would hide which.
    """
    telemetry, load_reports = load_calce_dataset(base_dir)
    curves, report = extract_discharge_curves(
        telemetry, rate_fraction=rate_fraction, min_points=min_points
    )
    return curves, report, load_reports
