"""State of health from raw field telemetry: voltage, current, time - nothing else.

WHY THIS EXISTS
---------------
`voltage_window.py` validated an estimator on curve frames built by the CALCE
loader, compensated with a resistance column the Arbin cycler measured. A
vehicle BMS has neither. It has a stream of voltage, current and time, and it
has to find the discharges, build the curves and know its own resistance from
that stream alone.

This module closes that gap. It turns a unified-schema telemetry frame and the
discharge phases `telemetry.cycles` already segments into exactly the inputs
`window_soh_table` takes, so the field path and the validated path share every
line of the estimator itself.

RESISTANCE FROM THE LOAD STEP
-----------------------------
The ohmic correction needs the voltage sag under load. A cycler reports a
resistance; a BMS has to infer it. It can, from the one event every discharge
starts with: current stepping from rest to load. The voltage falls at that
step, and

    R_step = (V_rest - V_load) / |I_load - I_rest|

taken between the last rest sample and the first loaded one.

What this measures is the APPARENT resistance at the logger's first sample
after the step - ohmic plus whatever polarisation builds in that interval. On
CALCE CS2_35 it reads about 0.165 ohm against the cycler's 0.099. That is not
an error in the estimate; it is a different quantity, and it is closer to the
total sag the window actually sees mid-discharge than the pure ohmic term is.
It does depend on sample rate, so it is comparable within one log and not
across loggers.

It is noisy cycle to cycle, so each cycle uses the median of the step
estimates up to and including that cycle over a trailing window. Trailing,
not centred: a centred median would let a cycle's correction depend on
cycles not yet recorded.

WHAT IT REFUSES
---------------
A cycle with no resistance estimate yet is left unmeasured rather than scored
uncompensated. Mixing compensated and uncompensated cycles in one trajectory
would put a step change into the SOH series that is an artefact of the
correction switching on.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.bms.health.voltage_window import (
    DEFAULT_REFERENCE_CYCLES,
    WindowSpec,
    window_charge,
    window_soh_table,
)

# A rest sample and the first loaded sample further apart than this are not one
# step: the cell has relaxed or the logger dropped data in between.
MAX_STEP_GAP_S = 60.0

# The current change must be at least this large for the voltage change to be
# read as a resistance. Below it, sensor noise divided by a small current
# swamps the estimate.
MIN_STEP_CURRENT_A = 0.05

# Trailing cycles in the resistance median. Wide enough to suppress one bad
# step, narrow enough to follow resistance growth over life.
RESISTANCE_WINDOW_CYCLES = 11

# Accepted cycles averaged into the reported current SOH. One discharge
# carries the full noise of a single measurement.
REPORT_CYCLES = 5

# The reference must be formed within this many EQUIVALENT FULL CYCLES of the
# start of the record - charge discharged so far, divided by the reference
# discharge's own total - or the cell is refused.
#
# WHY THROUGHPUT AND NOT A COUNT OF DISCHARGES
# ---------------------------------------------
# The first version counted discharge phases. CALCE CS2_24, a storage-protocol
# cell, logs 5,039 of them, almost all short characterisation pulses, and its
# first full discharge through the window is phase 338: a discharge count read
# that as "late" when the cell had barely been used. Fade follows charge
# throughput, which is why a BMS reports age in equivalent full cycles; a
# pulse moves almost none and correctly costs almost nothing here.
#
# HOW THIS WAS FOUND
# ------------------
# `window_soh_table` takes its reference from the first measurable cycles,
# wherever they fall. On full laboratory discharges they fall at the start.
# On partial discharges they need not: CS2_38 truncated to 85%->15% state of
# charge could not read the window until discharge 805, as its growing
# resistance shifted the curve into range - so "as new" was set from a cell
# already well into its life, and every later reading was measured against
# the wrong baseline (58.6% mean error). A vehicle whose driver's usual charge
# range only reaches the window once the cell has aged would do the same.
#
# 20 is set from the fade rate, not tuned on the estimator's error: across 22
# CALCE cells the median capacity lost by about cycle 20 is 2.7% (up to 12% on
# the Type 6 cells), which is already the size of the estimator's own error.
# A reference formed any later carries an offset as large as what the
# estimator is trying to measure.
MAX_REFERENCE_EFC = 20.0

# A window ratio above this is withheld. Window charge can read a few percent
# above its reference from noise and temperature, but a cell holding a tenth
# more than when it was new is a window that did not span the same part of
# the curve as the reference did.
MAX_PLAUSIBLE_SOH = 1.10

METHOD = "voltage_window_field"

# State bands on MEASURED state of health. These are conventions, not fitted
# values, and each is named for where it comes from:
#
#   0.80  automotive retirement convention. Below it, CRITICAL.
#   0.90  the end-of-life threshold this project's RUL estimator is validated
#         at (fade_extrapolation.DEFAULT_EOL_THRESHOLD). Below it, DEGRADED.
#   0.95  roughly twice the field estimator's median error below new: above
#         it, a cell is not distinguishable from new by this measurement, so
#         calling it anything but HEALTHY would be reading noise.
#
# They replace health_index's 30/60/80 cut-points only where a measurement
# exists. Those were hand-picked on a score that does not track fade.
SOH_STATE_BANDS: tuple[tuple[float, str], ...] = (
    (0.95, "HEALTHY"),
    (0.90, "WARNING"),
    (0.80, "DEGRADED"),
)


def state_from_soh(soh: float) -> str:
    """HEALTHY / WARNING / DEGRADED / CRITICAL from measured state of health."""
    if not np.isfinite(soh):
        raise ValueError("state_from_soh: no measured SOH to classify")
    for floor, state in SOH_STATE_BANDS:
        if soh >= floor:
            return state
    return "CRITICAL"


@dataclass(frozen=True)
class FieldSOH:
    """The current state of health of one cell, or the reason there is none."""

    soh: float
    at_cycle: int | None
    n_accepted: int
    n_discharges: int
    compensated: bool
    window: str
    refusal: str = ""
    # "fixed": the validated default window. "learned": a window placed
    # inside the voltage band this cell's own early discharges cover, used
    # when the fixed one cannot be read (see LEARNED WINDOWS below).
    window_mode: str = "fixed"
    # Apparent resistance from the rest-to-load step: the median over the
    # reference discharges and over the latest REPORT_CYCLES, in ohms. Rising
    # resistance is power fade - the half of health capacity does not show.
    # APPARENT: it includes polarisation built up by the first loaded sample,
    # so it depends on the logger's sample rate and is comparable within one
    # log, not across loggers. NaN when no clean step was seen.
    resistance_reference_ohm: float = float("nan")
    resistance_now_ohm: float = float("nan")
    # Standard error of this estimate from the readings' own consistency -
    # see CONSISTENCY below. NaN when there are too few readings to judge.
    uncertainty: float = float("nan")

    @property
    def consistent(self) -> bool:
        return bool(np.isfinite(self.uncertainty) and self.uncertainty <= CONSISTENCY_TOLERANCE)

    @property
    def resistance_growth(self) -> float:
        """Resistance now over at the reference, e.g. 1.15 = 15% higher."""
        if np.isfinite(self.resistance_reference_ohm) and self.resistance_reference_ohm > 0:
            return self.resistance_now_ohm / self.resistance_reference_ohm
        return float("nan")

    @property
    def available(self) -> bool:
        return bool(np.isfinite(self.soh))


def monotonic_time(time_s: pd.Series | np.ndarray) -> np.ndarray:
    """Make a time base that restarts (file splits, logger resets) monotonic.

    Each backwards jump is treated as a restart: everything after it is
    offset so it continues from where the previous segment ended, plus one
    typical sample interval. Sorting by timestamp instead would interleave two
    segments whose clocks overlap, which is worse than a small offset error.
    """
    t = np.asarray(pd.to_numeric(pd.Series(time_s), errors="coerce"), dtype=float)
    if len(t) < 2:
        return t.copy()
    steps = np.diff(t)
    positive = steps[np.isfinite(steps) & (steps > 0)]
    typical = float(np.median(positive)) if len(positive) else 1.0
    out = t.copy()
    offset = 0.0
    for k in np.flatnonzero(steps < 0):
        offset_needed = (t[k] + offset) + typical - (t[k + 1] + offset)
        offset += offset_needed
        out[k + 1:] = t[k + 1:] + offset
    return out


def _phase_bounds(time_s: np.ndarray, discharges: pd.DataFrame) -> list[tuple[int, int]]:
    starts = np.searchsorted(time_s, discharges["start_time_s"].to_numpy(float), "left")
    ends = np.searchsorted(time_s, discharges["end_time_s"].to_numpy(float), "right")
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def curves_from_telemetry(
    telemetry: pd.DataFrame,
    discharges: pd.DataFrame,
    cell_id: str,
    rest_threshold_a: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discharge curves and per-cycle step resistance, from raw telemetry.

    Returns (curves, steps). `curves` has the columns `window_soh_table`
    reads. `steps` has one row per discharge with `r_step_ohm` (NaN where no
    clean step preceded it) and `mean_current_a`.

    `discharges` is the frame `cycles_to_frame(..., complete_only=False)`
    produces: every discharge phase, complete or not, because a partial
    discharge that crosses the window is exactly what this estimator exists
    to use.
    """
    for column in ("test_time_s", "current_a", "voltage_v"):
        if column not in telemetry.columns:
            raise ValueError(
                f"curves_from_telemetry: telemetry has no '{column}'. Field SOH "
                f"needs voltage, current and time; none can be inferred."
            )
    t = telemetry["test_time_s"].to_numpy(dtype=float)
    current = pd.to_numeric(telemetry["current_a"], errors="coerce").to_numpy(float)
    voltage = pd.to_numeric(telemetry["voltage_v"], errors="coerce").to_numpy(float)

    blocks: list[pd.DataFrame] = []
    steps: list[dict] = []
    for row, (a, b) in zip(discharges.itertuples(), _phase_bounds(t, discharges), strict=True):
        cycle = int(row.cycle)
        if b - a < 2:
            continue
        seg_t, seg_i = t[a:b], current[a:b]
        charge = np.concatenate([[0.0], np.cumsum(
            np.abs(0.5 * (seg_i[1:] + seg_i[:-1])) * np.diff(seg_t)) / 3600.0])
        blocks.append(pd.DataFrame({
            "cell_id": cell_id, "cycle": cycle,
            "voltage_v": voltage[a:b], "capacity_ah_curve": charge,
        }))

        r_step = float("nan")
        v_rest = float("nan")
        j = a - 1
        if j >= 0 and abs(current[j]) < rest_threshold_a and (t[a] - t[j]) <= MAX_STEP_GAP_S:
            v_rest = float(voltage[j])
            d_i = abs(current[a] - current[j])
            d_v = voltage[j] - voltage[a]
            if d_i >= MIN_STEP_CURRENT_A and np.isfinite(d_v) and d_v > 0:
                r_step = d_v / d_i
        steps.append({
            "cell_id": cell_id, "cycle": cycle, "r_step_ohm": r_step,
            "v_rest_v": v_rest, "mean_current_a": float(row.mean_current_a),
        })

    curves = pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame(
        columns=["cell_id", "cycle", "voltage_v", "capacity_ah_curve"])
    return curves, pd.DataFrame(steps)


def step_overpotential(steps: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle ohmic offset from a trailing median of step resistances."""
    steps = steps.sort_values("cycle").copy()
    steps["r_trailing_ohm"] = (
        steps["r_step_ohm"]
        .rolling(RESISTANCE_WINDOW_CYCLES, min_periods=1).median()
        .ffill()
    )
    steps["ir_drop_v"] = steps["mean_current_a"].abs() * steps["r_trailing_ohm"]
    return steps[["cell_id", "cycle", "ir_drop_v", "r_trailing_ohm"]]


# ANCHORED OVERPOTENTIAL
# ----------------------
# The step resistance sees the sag one logger sample after the load starts.
# The rest of the polarisation - charge transfer and diffusion - builds over
# the discharge, grows with age, and grows much faster in the cold. On NASA
# cells at 4 and 22 C it moved the curve through the window and the window
# ratio erred by 12-14% (reports/metrics/cross_dataset/).
#
# A discharge that starts from full charge offers a measurement of that total
# sag with no model: early in the discharge the cell is at nearly the same
# state of charge as it was in its reference discharges, so the voltage
# difference from them over the same charge drawn is overpotential growth,
# not capacity. Over 5-15% of the reference charge the state-of-charge
# mismatch from capacity loss is about 0.02 at 30% fade, worth ~10 mV on the
# upper plateau, against the 50-300 mV of overpotential growth it corrects.
#
# Applicability: the cycle must start from full (rest voltage within
# ANCHOR_REST_TOLERANCE_V of the reference cycles') and cover the band. A
# cycle that does not is not anchored - it is left out, never scored with a
# different correction.
ANCHOR_BAND = (0.05, 0.15)
ANCHOR_REST_TOLERANCE_V = 0.03
ANCHOR_GRID_POINTS = 21
#
# MEASURED (reports/metrics/temperature_fix/, criteria fixed in advance): Oxford
# 3.0% -> 0.9%, NASA cold cohorts 13.0% -> 9.7%, but CALCE 1.7% -> 2.7% and
# three NASA cohorts worse. The preset criteria were not met, so the shipped
# estimator keeps the step correction; this stays available, not default.


def _curve_on_grid(block: pd.DataFrame, grid: np.ndarray) -> np.ndarray | None:
    """Voltage at each grid charge, or None if the curve does not span it."""
    q = block["capacity_ah_curve"].to_numpy(float)
    v = block["voltage_v"].to_numpy(float)
    ok = np.isfinite(q) & np.isfinite(v)
    q, v = q[ok], v[ok]
    if len(q) < 3 or q.max() < grid[-1] or q.min() > grid[0]:
        return None
    order = np.argsort(q, kind="stable")
    return np.interp(grid, q[order], v[order])


def anchored_overpotential(
    curves: pd.DataFrame,
    base: pd.DataFrame | None = None,
    rest: pd.DataFrame | None = None,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
    band: tuple[float, float] = ANCHOR_BAND,
) -> pd.DataFrame:
    """Per-cycle window offset: the reference's ohmic sag plus the measured
    growth of total overpotential since the reference.

    `base` (cell_id, cycle, ir_drop_v) is the step-resistance correction; the
    median over the reference cycles fixes the absolute level, so a reference
    discharge reads exactly as the shipped estimator reads it. Without `base`
    the level is zero. `rest` (cell_id, cycle, v_rest_v) enables the
    starts-from-full check; where it is absent the check cannot run, and that
    is the caller's assumption to state.

    Returns cell_id, cycle, ir_drop_v, anchor_offset_v, with NaN offset for
    cycles that are not anchorable.
    """
    out: list[dict] = []
    lo, hi = band
    for cell_id, cell in curves.groupby("cell_id", sort=True):
        cycles = sorted(int(c) for c in cell["cycle"].unique())
        blocks = {c: cell[cell["cycle"] == c] for c in cycles}
        totals = {c: float(b["capacity_ah_curve"].max()) for c, b in blocks.items()}
        rest_v: dict[int, float] = {}
        if rest is not None and not rest.empty:
            r = rest[rest["cell_id"] == cell_id]
            rest_v = {int(c): float(v) for c, v in zip(r["cycle"], r["v_rest_v"], strict=True)}
        c_ref = float(np.median([totals[c] for c in cycles[:reference_cycles]])) if cycles else float("nan")
        if not np.isfinite(c_ref) or c_ref <= 0:
            continue
        grid = np.linspace(lo, hi, ANCHOR_GRID_POINTS) * c_ref
        on_grid = {c: _curve_on_grid(blocks[c], grid) for c in cycles}

        ref = [c for c in cycles if on_grid[c] is not None][:reference_cycles]
        if not ref:
            continue
        v_ref = np.median(np.vstack([on_grid[c] for c in ref]), axis=0)
        ref_rest = [rest_v[c] for c in ref if np.isfinite(rest_v.get(c, np.nan))]
        rest_floor = (float(np.median(ref_rest)) - ANCHOR_REST_TOLERANCE_V) if ref_rest else None
        level = 0.0
        if base is not None and not base.empty:
            bb = base[(base["cell_id"] == cell_id) & base["cycle"].isin(ref)]["ir_drop_v"].dropna()
            if bb.empty:
                continue
            level = float(bb.median())
        for c in cycles:
            offset = float("nan")
            v_now = on_grid[c]
            from_full = rest_floor is None or (np.isfinite(rest_v.get(c, np.nan))
                                               and rest_v[c] >= rest_floor)
            if v_now is not None and from_full:
                offset = float(np.median(v_ref - v_now))
            out.append({"cell_id": cell_id, "cycle": c, "anchor_offset_v": offset,
                        "ir_drop_v": level + offset})
    return pd.DataFrame(out, columns=["cell_id", "cycle", "ir_drop_v", "anchor_offset_v"])


def field_soh_table(
    telemetry: pd.DataFrame,
    discharges: pd.DataFrame,
    cell_id: str,
    rest_threshold_a: float,
    spec: WindowSpec = WindowSpec(),
    compensate: bool = True,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
) -> pd.DataFrame:
    """Per-discharge window SOH from raw telemetry, with every gate applied."""
    curves, steps = curves_from_telemetry(
        telemetry, discharges, cell_id, rest_threshold_a)
    if curves.empty:
        return pd.DataFrame()
    overpotential = None
    if compensate:
        if steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            return pd.DataFrame()
        overpotential = step_overpotential(steps)
        # Cycles before the first resistance estimate are dropped rather than
        # scored uncompensated - see WHAT IT REFUSES above.
        known = overpotential.dropna(subset=["ir_drop_v"])["cycle"]
        curves = curves[curves["cycle"].isin(set(known))]
        overpotential = overpotential.dropna(subset=["ir_drop_v"])
    if curves.empty:
        return pd.DataFrame()
    table = window_soh_table(
        curves, spec=spec, reference_cycles=reference_cycles,
        overpotential=overpotential,
    )
    if overpotential is not None and "r_trailing_ohm" in overpotential.columns:
        table = table.merge(overpotential[["cycle", "r_trailing_ohm"]], on="cycle", how="left")
    return apply_field_gates(table, discharges, reference_cycles)


def apply_field_gates(
    table: pd.DataFrame,
    discharges: pd.DataFrame,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
    max_reference_efc: float = MAX_REFERENCE_EFC,
) -> pd.DataFrame:
    """The two gates field data needs and laboratory full discharges did not.

    Adds `reference_efc` - equivalent full cycles of charge the cell had
    delivered when its reference was completed - and withholds, through
    `soh_window_accepted`, every row of a cell whose reference came too late
    and every reading above `MAX_PLAUSIBLE_SOH`. Needs no ground truth.
    """
    if table.empty:
        return table
    table = table.copy()
    measurable = table.dropna(subset=["window_charge_ah"]).sort_values("cycle")
    position = float("nan")
    if len(measurable) >= reference_cycles:
        last_ref = int(measurable["cycle"].iloc[reference_cycles - 1])
        delivered = float(discharges.loc[
            discharges["cycle"].astype(int) <= last_ref, "capacity_ah"].sum())
        full = float(measurable["cycle_charge_ah"].head(reference_cycles).median())
        if full > 0:
            position = delivered / full
    table["reference_efc"] = position

    too_high = table["soh_window"] > MAX_PLAUSIBLE_SOH
    table.loc[too_high, "refusal"] = (
        f"window ratio above {MAX_PLAUSIBLE_SOH}; this discharge did not span "
        f"the same part of the curve as the reference")
    table["soh_window_accepted"] = table["soh_window_accepted"].where(~too_high)

    late = not (np.isfinite(position) and position <= max_reference_efc)
    table["reference_late"] = late
    if late:
        where = (f"after {position:.0f} equivalent full cycles"
                 if np.isfinite(position) else "never")
        table["refusal"] = (
            f"the reference was formed {where}, not within the first "
            f"{max_reference_efc:.0f}, so it is not this cell as new and "
            f"every reading against it would be offset")
        table["cell_refused"] = True
        table["soh_window_accepted"] = float("nan")
    return table


def current_field_soh(
    telemetry: pd.DataFrame,
    discharges: pd.DataFrame,
    cell_id: str,
    rest_threshold_a: float,
    spec: WindowSpec = WindowSpec(),
    compensate: bool = True,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
) -> FieldSOH:
    """The cell's state of health now, relative to the first cycles of the log.

    "Relative to the first cycles of the log" is the scope, and it is stated
    in every refusal-free result too: if the log did not start when the cell
    was new, this is fade since logging began, not state of health. A
    production BMS stores the beginning-of-life reference at manufacture; a
    log replayed here has only its own start.
    """
    n_discharges = int(len(discharges))
    empty = FieldSOH(float("nan"), None, 0, n_discharges, compensate, str(spec))
    if n_discharges == 0:
        return _with(empty, "no discharge phases were found in the telemetry")
    if "voltage_v" not in telemetry.columns:
        return _with(empty, "telemetry has no voltage channel; the window needs it")

    table = field_soh_table(
        telemetry, discharges, cell_id, rest_threshold_a, spec, compensate,
        reference_cycles)
    return field_soh_from_table(table, n_discharges, spec, compensate, reference_cycles)


def field_soh_from_table(
    table: pd.DataFrame,
    n_discharges: int,
    spec: WindowSpec = WindowSpec(),
    compensate: bool = True,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
) -> FieldSOH:
    """The reported result from a gated per-discharge table.

    Shared by `current_field_soh` (table built from the whole log) and the
    stored per-battery profile (table rebuilt from its own memory).
    """
    empty = FieldSOH(float("nan"), None, 0, n_discharges, compensate, str(spec))
    if table.empty:
        reason = (
            "no discharge was preceded by a clean rest-to-load step, so the "
            "cell's resistance - and with it the ohmic correction - is unknown"
            if compensate else "no discharge produced a usable curve")
        return _with(empty, reason)
    n_measurable = int(table["window_charge_ah"].notna().sum())
    if n_measurable <= reference_cycles:
        # Said before any gate verdict: "too few" is the actionable reason,
        # and a late-reference refusal on four discharges would obscure it.
        return _with(
            FieldSOH(float("nan"), None, n_measurable, n_discharges,
                     compensate, str(spec)),
            f"only {n_measurable} discharge(s) crossed the {spec} window; "
            f"{reference_cycles} are needed to form the reference and at least "
            f"one more to measure against it")
    if bool(table["cell_refused"].any()):
        return _with(empty, str(table.loc[table["cell_refused"], "refusal"].iloc[0]))

    accepted = table.dropna(subset=["soh_window_accepted"]).sort_values("cycle")
    beyond_reference = accepted.iloc[reference_cycles:]
    if beyond_reference.empty:
        return _with(
            FieldSOH(float("nan"), None, len(accepted), n_discharges,
                     compensate, str(spec)),
            f"only {len(accepted)} discharge(s) crossed the {spec} window; "
            f"{reference_cycles} are needed to form the reference and at least "
            f"one more to measure against it")

    latest = beyond_reference.tail(REPORT_CYCLES)
    r_ref, r_now = _resistance(accepted, latest, reference_cycles)
    u = consistency_uncertainty(accepted, reference_cycles)
    return FieldSOH(
        soh=float(latest["soh_window_accepted"].median()),
        at_cycle=int(latest["cycle"].iloc[-1]),
        n_accepted=int(len(accepted)),
        n_discharges=n_discharges,
        compensated=compensate,
        window=str(spec),
        resistance_reference_ohm=r_ref,
        resistance_now_ohm=r_now,
        uncertainty=u,
    )


def _with(result: FieldSOH, refusal: str) -> FieldSOH:
    return replace(result, refusal=refusal)


# ---------------------------------------------------------------------------
# LEARNED WINDOWS: partial discharges
# ---------------------------------------------------------------------------
#
# The fixed 3.90-3.60 V window was validated on full laboratory discharges. A
# vehicle rarely produces one: a driver who keeps the battery between 90% and
# 70% never crosses 3.60 V, and one who runs it low never sees 3.90 V. CALCE
# has both, as real partial cycling with periodic full capacity checks: CS2
# Type 6 cycles 4.07 -> 3.78 V (top of charge), Type 5 cycles 3.69 -> 2.70 V
# (bottom of charge), thousands of partials each.
#
# A BMS can learn where its own driver operates. The band every one of the
# cell's first discharges covers is measurable with no ground truth, and a
# window placed inside it can be read on every later discharge in the same
# band. Learned from the first `LEARN_DISCHARGES` only, so nothing later in
# life - and nothing from a capacity test - informs it.
#
# WHERE IN THE BAND, AND WHEN TO REFUSE
# ---------------------------------------
# Window charge tracks capacity on the voltage PLATEAU, where the curve is
# set by thermodynamics. Near empty it falls into the knee, where voltage is
# set by kinetics - resistance and solid-state diffusion - so charge in a
# knee window measures how hard the cell is working, not how much it holds.
# Measured on CALCE, as window steepness in volts per unit state of charge:
#
#     learned windows, full-discharge cells    0.44-0.88    works (median 1.95%)
#     top-of-charge partials (Type 6)          0.82-0.83    works (3.2-3.4%)
#     bottom-of-charge partials (Type 5)       8.2-10.4     failed at 6.8-12.8%
#                                                           before this gate
#
# (Flattest sub-window in each band; reports/metrics/calce_partial_soh/.)
# So the flattest sub-window of the band is chosen, and refused if even that
# is steeper than `MAX_WINDOW_STEEPNESS`. The threshold is physical rather
# than tuned: any value between about 0.9 and 8 gives the same verdict on
# every cell measured, and 2.0 is a round number just above the plateau.
#
# Steepness is per unit of state of charge, so it needs the cell's rated
# capacity. Without one there is no learned window, only the fixed one.

LEARN_DISCHARGES = 10
# Fraction of the band trimmed from each end. The top of a discharge carries
# the load-step transient and the bottom the cutoff, neither of which repeats
# exactly from one discharge to the next.
BAND_MARGIN = 0.15
# Samples skipped at the start of each discharge when finding the band's top:
# the voltage is still settling from the load step.
SETTLE_SAMPLES = 2
MIN_LEARNED_SPAN_V = 0.10
MAX_LEARNED_SPAN_V = 0.30
SCAN_STEP_V = 0.01
MAX_WINDOW_STEEPNESS = 2.0


@dataclass(frozen=True)
class LearnedWindow:
    """A window chosen from a cell's own early discharges, or why there is none."""

    spec: WindowSpec | None
    steepness: float
    band: tuple[float, float]
    refusal: str = ""


def learn_usage_window(
    curves: pd.DataFrame,
    rated_capacity_ah: float,
    learn_discharges: int = LEARN_DISCHARGES,
) -> LearnedWindow:
    """The flattest window inside the band the first discharges all cover.

    `curves` must be terminal-voltage discharge curves as
    `curves_from_telemetry` builds them. The returned window is in that same
    terminal-voltage frame, as seen on the learning discharges.
    """
    cycles = sorted(curves["cycle"].unique())[:learn_discharges]
    if len(cycles) < learn_discharges:
        return LearnedWindow(None, float("nan"), (float("nan"), float("nan")),
                             f"only {len(cycles)} discharges to learn a window "
                             f"from; {learn_discharges} are needed")
    early = curves[curves["cycle"].isin(cycles)]
    tops, bottoms = [], []
    for _, block in early.groupby("cycle"):
        v = block["voltage_v"].to_numpy(float)
        settled = v[SETTLE_SAMPLES:] if len(v) > SETTLE_SAMPLES + 2 else v
        tops.append(float(np.nanmax(settled)))
        bottoms.append(float(np.nanmin(v)))
    hi, lo = min(tops), max(bottoms)
    band = (hi, lo)
    margin = BAND_MARGIN * (hi - lo)
    usable_hi, usable_lo = hi - margin, lo + margin
    span = min(MAX_LEARNED_SPAN_V, usable_hi - usable_lo)
    if span < MIN_LEARNED_SPAN_V:
        return LearnedWindow(None, float("nan"), band,
                             f"the band every early discharge covers is "
                             f"{hi:.2f}-{lo:.2f} V, too narrow to hold a window "
                             f"of {MIN_LEARNED_SPAN_V} V once its edges are trimmed")

    best: tuple[float, WindowSpec] | None = None
    low = usable_lo
    while low + span <= usable_hi + 1e-9:
        spec = WindowSpec(round(low + span, 3), round(low, 3))
        steep = _steepness(early, spec, rated_capacity_ah)
        if np.isfinite(steep) and (best is None or steep < best[0]):
            best = (steep, spec)
        low += SCAN_STEP_V
    if best is None:
        return LearnedWindow(None, float("nan"), band,
                             "no window inside the usage band could be read on "
                             "the early discharges")
    steep, spec = best
    if steep > MAX_WINDOW_STEEPNESS:
        return LearnedWindow(None, steep, band,
                             f"the flattest window in this cell's usage band "
                             f"({spec}) falls {steep:.1f} V per unit state of "
                             f"charge - the knee of the curve, where voltage "
                             f"reflects resistance rather than capacity. A "
                             f"plateau window is under {MAX_WINDOW_STEEPNESS}.")
    return LearnedWindow(spec, steep, band)


def _steepness(curves: pd.DataFrame, spec: WindowSpec, rated_capacity_ah: float) -> float:
    """Median volts per unit state of charge across the window, over discharges."""
    values = []
    for _, block in curves.groupby("cycle"):
        q = window_charge(block["voltage_v"].to_numpy(float),
                          block["capacity_ah_curve"].to_numpy(float), spec)
        if np.isfinite(q) and q > 0:
            values.append(spec.span_v / (q / rated_capacity_ah))
    return float(np.median(values)) if values else float("nan")


def learned_field_soh(
    telemetry: pd.DataFrame,
    discharges: pd.DataFrame,
    cell_id: str,
    rest_threshold_a: float,
    rated_capacity_ah: float,
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
) -> tuple[FieldSOH, pd.DataFrame]:
    """Field SOH on a window learned from the cell's own usage band.

    Returns the current estimate and the full per-discharge table, so a study
    can score every discharge, not only the latest.
    """
    n = int(len(discharges))
    curves, steps = curves_from_telemetry(telemetry, discharges, cell_id, rest_threshold_a)
    none = FieldSOH(float("nan"), None, 0, n, True, "", window_mode="learned")
    if curves.empty or steps.empty or steps["r_step_ohm"].notna().sum() == 0:
        return _with(none, "no discharge was preceded by a clean rest-to-load "
                           "step, so the ohmic correction is unknown"), pd.DataFrame()
    learned = learn_usage_window(curves, rated_capacity_ah)
    if learned.spec is None:
        return _with(none, learned.refusal), pd.DataFrame()

    overpotential = step_overpotential(steps).dropna(subset=["ir_drop_v"])
    curves = curves[curves["cycle"].isin(set(overpotential["cycle"]))]
    # The window was learned in terminal voltage on the early discharges, so
    # it is moved into the compensated frame by their own ohmic drop; each
    # later discharge then reads it shifted by ITS drop, as the fixed window
    # does.
    early = sorted(curves["cycle"].unique())[:LEARN_DISCHARGES]
    ir_ref = float(overpotential.loc[overpotential["cycle"].isin(early), "ir_drop_v"].median())
    spec = WindowSpec(learned.spec.v_high + ir_ref, learned.spec.v_low + ir_ref)
    table = window_soh_table(curves, spec=spec, reference_cycles=reference_cycles,
                             overpotential=overpotential)
    table = table.merge(overpotential[["cycle", "r_trailing_ohm"]], on="cycle", how="left")
    table = apply_field_gates(table, discharges, reference_cycles)
    table["learned_window"] = str(learned.spec)
    table["window_steepness"] = learned.steepness

    label = f"{learned.spec} (learned)"
    accepted = table.dropna(subset=["soh_window_accepted"]).sort_values("cycle")
    if bool(table["cell_refused"].any()):
        return _with(replace(none, window=label),
                     str(table.loc[table["cell_refused"], "refusal"].iloc[0])), table
    beyond = accepted.iloc[reference_cycles:]
    if beyond.empty:
        return _with(replace(none, window=label, n_accepted=len(accepted)),
                     f"only {len(accepted)} discharge(s) read the learned "
                     f"window; more are needed"), table
    latest = beyond.tail(REPORT_CYCLES)
    r_ref, r_now = _resistance(accepted, latest, reference_cycles)
    u = consistency_uncertainty(accepted, reference_cycles)
    return FieldSOH(
        soh=float(latest["soh_window_accepted"].median()),
        at_cycle=int(latest["cycle"].iloc[-1]),
        n_accepted=int(len(accepted)), n_discharges=n, compensated=True,
        window=label, window_mode="learned",
        resistance_reference_ohm=r_ref, resistance_now_ohm=r_now,
        uncertainty=u,
    ), table


# CONSISTENCY: WHEN IS THERE ENOUGH EVIDENCE?
# --------------------------------------------
# Not "has the estimate stopped changing" - a healthy estimate keeps moving,
# because the cell keeps ageing. The question is whether the readings agree
# with each other once that trend is removed. Two terms:
#
#   reference SE  standard error of the beginning-of-life median, relative
#   recent SE     scatter of the last RECENT_READINGS readings around their
#                 own straight line, as the standard error of the reported
#                 median of REPORT_CYCLES
#
# combined as u = sqrt(ref^2 + recent^2), with 1.2533 * MAD / sqrt(n) as the
# standard error of a median. Measured on CALCE (scripts/run_sufficiency_
# study.py, tolerance fixed before it ran): readings with u <= 0.01 had a
# median error of 1.3% against the lab; u > 0.01, 5.1% (90th percentile 22%).
# The same study found that the NUMBER of readings does not predict accuracy
# - error grows with age, not shrinks with count - so confidence follows
# consistency, not a discharge count.
CONSISTENCY_TOLERANCE = 0.01
RECENT_READINGS = 10


def _se_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(1.2533 * np.median(np.abs(values - np.median(values))) / np.sqrt(len(values)))


def consistency_uncertainty(accepted: pd.DataFrame, reference_cycles: int) -> float:
    """Standard error of the reported SOH from the readings' own scatter."""
    charge = accepted["window_charge_ah"].to_numpy(float)
    soh = accepted["soh_window_accepted"].to_numpy(float)
    if len(charge) <= reference_cycles:
        return float("nan")
    ref = charge[:reference_cycles]
    ref_se = _se_median(ref) / float(np.median(ref)) if reference_cycles > 1 else 0.0
    recent = soh[reference_cycles:][-RECENT_READINGS:]
    if len(recent) < 4:
        return float("nan")
    idx = np.arange(len(recent))
    resid = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
    recent_se = 1.2533 * float(np.median(np.abs(resid - np.median(resid)))) / np.sqrt(REPORT_CYCLES)
    ref_se = 0.0 if not np.isfinite(ref_se) else ref_se
    return float(np.sqrt(ref_se ** 2 + recent_se ** 2))


def _resistance(accepted: pd.DataFrame, latest: pd.DataFrame,
                reference_cycles: int) -> tuple[float, float]:
    """Median trailing step resistance over the reference and latest discharges."""
    if "r_trailing_ohm" not in accepted.columns:
        return float("nan"), float("nan")
    ref = accepted["r_trailing_ohm"].head(reference_cycles).dropna()
    now = latest["r_trailing_ohm"].dropna()
    return (float(ref.median()) if len(ref) else float("nan"),
            float(now.median()) if len(now) else float("nan"))
