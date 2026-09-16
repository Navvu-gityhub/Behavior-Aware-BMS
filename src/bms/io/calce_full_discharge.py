"""Full-discharge detection for CALCE cycling data.

Why this module exists
----------------------
CALCE's cycle-level target is derived by grouping on Arbin's `Cycle_Index` and
taking the within-group span of `Discharge_Capacity`. That is correct only when
one `Cycle_Index` contains exactly one discharge, which holds for CS2/CX2 Types
1 and 2 and fails for Types 3, 5 and 6:

- Type 3 switches discharge rate six times within a cycle.
- Types 5 and 6 cycle **partially** by design, and interleave periodic full
  reference discharges with long runs of shallow ones.

ADR 0009 recorded the consequence — Types 5 and 6 are scored against a
partial-cycle reference of roughly 0.26-0.42 Ah on a 1.1 Ah cell, so their SOH
tracks relative fade of a repeated partial cycle rather than absolute state of
health — and left recovering absolute SOH as the largest open quality item.

A cycle-level fix does not work, and the reason is worth recording
---------------------------------------------------------------
The obvious cheap repair is to keep grouping by `Cycle_Index` and simply select
the groups that look full: those reaching the voltage cutoff whose capacity span
is near the cell's largest. That was tried first and is wrong. It produces
references of 1.40-1.75 Ah for the Type 5 and 6 cells — 27% to 59% **above** the
1.1 Ah nominal capacity — while the same rule on Types 1 and 2 returns a correct
1.12-1.13 Ah.

The inflation is the diagnosis: a `Cycle_Index` holding several discharges has a
`Discharge_Capacity` span covering all of them, so selecting the largest spans
preferentially selects the groups containing the *most* discharges. Any method
that keeps trusting `Cycle_Index` inherits the defect it is trying to remove.

Two criteria are needed, not one
--------------------------------
`docs/roadmap.md` proposed detecting "contiguous negative-current runs
terminating at the voltage cutoff". Measured against the data, the cutoff half
of that is necessary but nowhere near sufficient:

| Cell | Type | discharges reaching cutoff | median charge moved |
|---|---|---|---|
| CS2_24 | 6 | 53 / 4990 | 0.288 Ah |
| CS2_5 | 5 | 6951 / 6962 | 0.237 Ah |

Type 6 partials stop at 3.78 V, so the cutoff test separates them cleanly. Type
5 partials **do** reach the 2.7 V cutoff — they discharge to cutoff from a
partially charged state — so 99.8% of them pass a cutoff test while delivering a
fifth of the cell's capacity.

So a full discharge here is one that both terminates at the cutoff *and* moves
charge comparable to the most any single discharge of that cell has moved.

What this costs
---------------
Most rows. A Type 5 or 6 cell yields tens of full discharges out of thousands of
cycles. That is a property of the protocol, not a defect in the detector, and
the honest response is the one `telemetry/cycles.py` already takes for CAN logs:
report the yield rather than quietly scaling partials up to look complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.telemetry.cycles import segment_phases

# Fraction of a cell's own robust peak discharge magnitude below which current
# counts as rest. Relative rather than absolute because CALCE discharge rates
# span an order of magnitude across protocols: CS2_24 peaks at 0.68 A while
# CS2_5 peaks at 5.84 A, so `cycles.REST_THRESHOLD_A` (0.5 A, sized for a
# vehicle pack) would treat most of CS2_24's discharges as rest.
REST_FRACTION_OF_PEAK = 0.05

# Floor for the above, in amps. Guards a cell whose peak reads near zero through
# a sensor fault from having every sample classified as a discharge.
MIN_REST_THRESHOLD_A = 0.005

# A discharge terminates at the cutoff if its final voltage is within this
# margin of the cell's own observed cutoff.
CUTOFF_TOLERANCE_V = 0.10

# Voltage below which a reading is a sensor artifact rather than cell
# behaviour. CS2_5 records values down to -2.54 V; a lithium cell does not go
# negative, and admitting those would drag the estimated cutoff below anything
# the cell ever really reached.
VOLTAGE_FLOOR_V = 1.5

# The cutoff is estimated from the DEEPEST discharges a cell performed, not from
# a quantile of all of them, and the distinction is the whole difficulty.
#
# A quantile of all terminal voltages is set by whichever discharge type is most
# common. CS2_24 performs 4,990 shallow discharges that stop at 3.78 V and about
# 53 full ones that reach 2.70 V, so even a 2nd-percentile cutoff returns
# 3.78 V - the partials define the "cutoff", every partial then trivially
# "reaches" it, and the detector certifies the exact cycles it exists to reject.
#
# So the lowest `max(MIN_CUTOFF_SAMPLES, CUTOFF_DEEPEST_FRACTION)` terminal
# voltages are taken and their median used. That is robust to a few artifacts
# without depending on what fraction of a cell's cycles happen to be full.
CUTOFF_DEEPEST_FRACTION = 0.005
MIN_CUTOFF_SAMPLES = 5

# A discharge is full if it moves at least this fraction of the charge moved by
# the cell's most capable single discharge.
FULL_CHARGE_FRACTION = 0.80

# Quantile of per-discharge charge taken as that cell's full capability,
# computed over cutoff-reaching discharges only. Not the maximum: one merged or
# mis-segmented run would then set the scale for every other cycle.
#
# High (0.99 rather than 0.95) because on a partial-cycling protocol the full
# reference discharges are a small minority even among those reaching cutoff.
# CS2_5 discharges to the 2.7 V cutoff from a partially charged state, so 98.5%
# of its discharges reach cutoff while moving a fifth of the cell's capacity; a
# 95th percentile there still lands on a partial.
CAPABILITY_QUANTILE = 0.99

# A discharge shorter than this carries too few samples to integrate reliably.
MIN_DISCHARGE_SAMPLES = 5

# The detected capability must be at least this fraction of the largest single
# discharge the cell performed.
#
# A guard against the detector confidently returning a reference that is not a
# full discharge, which it can do when a cell's deepest excursions are artifacts
# rather than reference cycles. CX2_32 is the case that motivated it: the
# estimator returned a 1.999 V cutoff where every other CS2/CX2 cell returns
# 2.69-2.70 V, and a capability of 0.0397 Ah against a 1.35 Ah nominal - 3% of
# the cell's own largest discharge.
#
# Nothing downstream catches this. `targets.MIN_REFERENCE_OVER_MAX` compares the
# reference to the maximum of the *retained* rows, and here every retained row
# is equally tiny, so the ratio is ~1.0 and the cell reads as pristine. The
# median-SOH and step tests pass for the same reason. A cell can therefore be
# admitted on a reference three orders of magnitude too small unless the
# detector itself refuses.
#
# Set well below any plausible degradation floor: cells are retired near 70-80%
# state of health, and a reference at 20% of the cell's best discharge cannot be
# a full discharge on any protocol here.
MIN_CAPABILITY_OVER_MAX = 0.20


@dataclass(frozen=True)
class FullDischargeYield:
    """How many usable absolute-SOH points a cell actually produced."""

    cell_id: str
    n_discharges: int
    n_full: int
    capability_ah: float
    cutoff_v: float
    rest_threshold_a: float
    max_discharge_ah: float = float("nan")
    refusal: str = ""

    @property
    def usable_fraction(self) -> float:
        return self.n_full / self.n_discharges if self.n_discharges else 0.0

    @property
    def capability_over_max(self) -> float:
        """Detected capability as a fraction of the cell's largest discharge."""
        if not np.isfinite(self.max_discharge_ah) or self.max_discharge_ah <= 0:
            return float("nan")
        return self.capability_ah / self.max_discharge_ah

    def render(self) -> str:
        if self.refusal:
            return f"{self.cell_id}: REFUSED - {self.refusal}"
        return (
            f"{self.cell_id}: {self.n_full}/{self.n_discharges} discharges are "
            f"full ({self.usable_fraction:.1%}); capability "
            f"{self.capability_ah:.4g} Ah, cutoff {self.cutoff_v:.3f} V, rest "
            f"threshold {self.rest_threshold_a:.4g} A."
        )


def detect_full_discharges(
    telemetry: pd.DataFrame,
    cell_id: str,
    rest_fraction: float = REST_FRACTION_OF_PEAK,
    cutoff_tolerance_v: float = CUTOFF_TOLERANCE_V,
    full_charge_fraction: float = FULL_CHARGE_FRACTION,
) -> tuple[pd.DataFrame, FullDischargeYield]:
    """Segment one cell's sample-level telemetry into discharges and grade them.

    Returns ``(discharges, yield_summary)``. `discharges` has one row per
    contiguous discharge run, with `charge_moved_ah`, `end_voltage_v`,
    `reaches_cutoff`, `is_full` and the source `cycle_index` it began in.

    Segmentation is delegated to `telemetry.cycles.segment_phases`, the same
    function the CAN telemetry path uses, so a discharge means the same thing on
    a bench rig, a vehicle bus and an Arbin export. The alternative - a second
    segmentation written for CALCE - is what this project's telemetry modules
    exist to avoid.
    """
    frame = telemetry.copy()
    for column in ("current_a", "voltage_v", "test_time_s"):
        if column not in frame.columns:
            raise ValueError(
                f"detect_full_discharges: {cell_id} has no '{column}'. Full "
                f"discharge detection needs current, voltage and a time base; "
                f"none can be inferred, and guessing one would silently change "
                f"every capacity it produces."
            )

    current = pd.to_numeric(frame["current_a"], errors="coerce").reset_index(drop=True)
    voltage = pd.to_numeric(frame["voltage_v"], errors="coerce").reset_index(drop=True)
    time_s = _monotonic_time(
        pd.to_numeric(frame["test_time_s"], errors="coerce").reset_index(drop=True)
    )
    cycle_index = (
        pd.to_numeric(frame["cycle"], errors="coerce").reset_index(drop=True)
        if "cycle" in frame.columns
        else pd.Series(np.nan, index=range(len(current)))
    )
    # Internal resistance is carried through so the rebuilt target keeps the
    # same feature set the ADR 0009 CALCE study used - voltage, current,
    # internal resistance, cycle duration. Rebuilding the target on a different
    # feature set would confound the change being measured with a change in
    # what the models are allowed to see.
    resistance = (
        pd.to_numeric(frame["resistance_ohm"], errors="coerce").reset_index(drop=True)
        if "resistance_ohm" in frame.columns
        else pd.Series(np.nan, index=range(len(current)))
    )

    peak = float(np.nanquantile(np.abs(current.to_numpy(float)), 0.99))
    rest_threshold = max(peak * rest_fraction, MIN_REST_THRESHOLD_A)

    segmented = pd.DataFrame({"current_a": current, "test_time_s": time_s})
    phases = segment_phases(
        segmented, rest_threshold_a=rest_threshold, min_phase_samples=3
    )
    discharges = [p for p in phases if p.kind == "discharge"]

    rows: list[dict] = []
    for number, phase in enumerate(discharges, start=1):
        if phase.n_samples < MIN_DISCHARGE_SAMPLES:
            continue
        window_v = voltage.iloc[phase.start_index : phase.end_index + 1].dropna()
        if window_v.empty:
            continue
        rows.append({
            "cell_id": cell_id,
            "discharge": number,
            "cycle_index": _mode_or_nan(
                cycle_index.iloc[phase.start_index : phase.end_index + 1]
            ),
            "charge_moved_ah": phase.charge_moved_ah,
            "start_time_s": phase.start_time_s,
            "end_time_s": phase.end_time_s,
            "duration_s": phase.duration_s,
            "n_samples": phase.n_samples,
            "mean_current_a": phase.mean_current_a,
            "start_voltage_v": float(window_v.iloc[0]),
            "end_voltage_v": float(window_v.iloc[-1]),
            "min_voltage_v": float(window_v.min()),
            "mean_voltage_v": float(window_v.mean()),
            "min_current_a": float(
                current.iloc[phase.start_index : phase.end_index + 1].min()
            ),
            "resistance_ohm": float(
                resistance.iloc[phase.start_index : phase.end_index + 1].mean()
            ),
        })

    if not rows:
        return pd.DataFrame(), FullDischargeYield(
            cell_id=cell_id, n_discharges=0, n_full=0,
            capability_ah=float("nan"), cutoff_v=float("nan"),
            rest_threshold_a=rest_threshold,
        )

    result = pd.DataFrame(rows)

    cutoff_v = _estimate_cutoff(result["min_voltage_v"])
    result["reaches_cutoff"] = result["min_voltage_v"] <= cutoff_v + cutoff_tolerance_v

    # Capability is judged among cutoff-reaching discharges. Including partials
    # that stopped early would let a protocol made mostly of shallow cycles
    # define its own capability downward.
    reaching = result.loc[result["reaches_cutoff"], "charge_moved_ah"]
    capability = float(
        (reaching if not reaching.empty else result["charge_moved_ah"])
        .quantile(CAPABILITY_QUANTILE)
    )
    result["is_full"] = (
        result["reaches_cutoff"]
        & (result["charge_moved_ah"] >= full_charge_fraction * capability)
        & (result["charge_moved_ah"] > 0.0)
    )

    # Is the thing we called "full capability" plausibly a full discharge at
    # all? See MIN_CAPABILITY_OVER_MAX - nothing downstream can catch this.
    max_discharge = float(result["charge_moved_ah"].max())
    refusal = ""
    if max_discharge > 0 and capability / max_discharge < MIN_CAPABILITY_OVER_MAX:
        refusal = (
            f"detected capability {capability:.4g} Ah is only "
            f"{capability / max_discharge:.1%} of this cell's largest discharge "
            f"({max_discharge:.4g} Ah), so it is not a full discharge. The "
            f"estimated cutoff was {cutoff_v:.3f} V. Emitting an absolute SOH "
            f"against this reference would understate capacity by more than an "
            f"order of magnitude, and every downstream screen would pass it "
            f"because each retained row is equally wrong."
        )
        result["is_full"] = False

    summary = FullDischargeYield(
        cell_id=cell_id,
        n_discharges=len(result),
        n_full=int(result["is_full"].sum()),
        capability_ah=capability,
        cutoff_v=cutoff_v,
        rest_threshold_a=rest_threshold,
        max_discharge_ah=max_discharge,
        refusal=refusal,
    )
    return result, summary


def _estimate_cutoff(terminal_voltages: pd.Series) -> float:
    """The voltage at which this cell's deepest discharges terminate.

    See `CUTOFF_DEEPEST_FRACTION` for why this cannot be a quantile of all
    terminal voltages.
    """
    plausible = terminal_voltages[terminal_voltages >= VOLTAGE_FLOOR_V].dropna()
    if plausible.empty:
        return float("nan")
    n_deepest = max(MIN_CUTOFF_SAMPLES, int(CUTOFF_DEEPEST_FRACTION * len(plausible)))
    return float(plausible.nsmallest(n_deepest).median())


def _monotonic_time(time_s: pd.Series) -> pd.Series:
    """Stitch a per-file test clock into one monotonic axis.

    CALCE ships each cell as a series of date-named files, and **Arbin's
    `Test_Time` restarts at each file boundary**. CS2_33 resets 21 times, the
    largest being ``819352.3 -> 30.0``. The loader concatenates the files in
    date order, so row order is the true temporal order, but the time column
    alone is not.

    Sorting by the raw column - the obvious first move, and the one tried first
    here - is actively destructive: it interleaves twenty-two unrelated files
    into a single scrambled series, which fragments every real discharge into
    thousands of one-sample runs. The symptom is unmistakable once you know to
    look: detected capability of 0.002 Ah on a 1.1 Ah cell, and 12,018
    "discharges" for a cell with 4,990 cycles.

    So each reset opens a new segment, and each segment is offset by the total
    elapsed time of everything before it. Row order is preserved throughout.
    """
    values = time_s.to_numpy(float)
    if len(values) < 2:
        return time_s

    steps = np.diff(values, prepend=values[0])
    segment = np.cumsum(steps < 0)

    stitched = pd.Series(values).groupby(segment)
    # Elapsed time contributed by each segment, and the offset each one starts at.
    spans = stitched.max() - stitched.min()
    offsets = spans.cumsum().shift(1).fillna(0.0)
    starts = stitched.min()

    adjusted = (
        pd.Series(values)
        - pd.Series(segment).map(starts).to_numpy(float)
        + pd.Series(segment).map(offsets).to_numpy(float)
    )
    return adjusted


def _mode_or_nan(series: pd.Series) -> float:
    """The Arbin cycle index this discharge mostly sits in, for traceability."""
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.mode().iloc[0])


def full_discharge_frame(
    discharges: pd.DataFrame,
    cohort: str | None = None,
) -> pd.DataFrame:
    """Reduce detected discharges to a cycle-level frame of full discharges only.

    The output matches the schema `benchmarks/targets.add_targets` consumes -
    `cell_id`, `cycle`, `capacity_ah` plus electrical features - so the
    downstream study is unchanged. `cycle` is renumbered contiguously over the
    retained full discharges, because a gap would read as missing data rather
    than as a filtered partial cycle.
    """
    if discharges.empty:
        return pd.DataFrame()

    full = discharges[discharges["is_full"]].copy()
    if full.empty:
        return pd.DataFrame()

    full = full.sort_values("start_time_s").reset_index(drop=True)
    out = pd.DataFrame({
        "cell_id": full["cell_id"],
        "cycle": np.arange(1, len(full) + 1),
        "capacity_ah": full["charge_moved_ah"],
        "mean_voltage_v": full["mean_voltage_v"],
        "min_voltage_v": full["min_voltage_v"],
        "mean_current_a": full["mean_current_a"],
        "min_current_a": full["min_current_a"],
        "resistance_ohm": full["resistance_ohm"],
        "cycle_duration_s": full["duration_s"],
        "n_samples": full["n_samples"],
        "arbin_cycle_index": full["cycle_index"],
        "dataset": "calce",
    })
    if cohort is not None:
        out["cohort"] = cohort
    return out
