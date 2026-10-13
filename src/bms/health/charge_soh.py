"""State of health read while the battery CHARGES.

WHY
---
The field estimator (field_soh.py) reads health from discharges that cross
3.90-3.60 V. A vehicle's discharges are irregular: driving current jumps, and
many trips never cross the window. Charging is the opposite. A charger holds
a steady current, the cell's voltage climbs through the same range on almost
every charge, and it happens most days. If the window method reads health
there as well as it does on discharge, BEACON can answer far more often.

THE METHOD (fixed before any charge curve was examined)
-------------------------------------------------------
Same physics as the discharge window, run in reverse:

    SOH_charge(k) = Q_window(charge k) / median Q_window(first 5 charges)

- **Window: 3.75-4.00 V of rest-equivalent voltage.** It sits in the
  constant-current part of a charge to 4.2 V, below the constant-voltage
  hold, for cells charged to 4.2 V (LCO, NMC, NMC/LCO). It is not meant for
  LFP: an LFP charge never reaches it, and the window is then refused as
  not covered.
- **Correction.** While charging, the terminal voltage sits ABOVE rest
  voltage by I*R, so the window is shifted UP by I*R_trailing. R comes from
  the voltage step at the start of each charge, from a quiet preceding
  current, screened and trailing as on discharge (field_soh).
- **Refusals.**
  - A charge that does not span the window is not read, for example a
    top-up that starts above 3.75 V or a fast charge already at its 4.2 V
    hold.
  - A ratio below 0.5 or above 1.10 is withheld.
  - Too few readings to form the reference is refused.

Charges with no rest-to-charge step before any resistance is known are left
unread rather than scored uncorrected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.bms.health.field_soh import (
    MAX_STEP_GAP_S,
    MIN_STEP_CURRENT_A,
    QUIET_FRACTION,
    RESISTANCE_WINDOW_CYCLES,
    screen_step_resistances,
)
from src.bms.health.voltage_window import (
    DEFAULT_REFERENCE_CYCLES,
    MIN_PLAUSIBLE_SOH,
    WindowSpec,
    window_charge,
)
from src.bms.telemetry.cycles import segment_phases

CHARGE_WINDOW = WindowSpec(4.00, 3.75)
MAX_PLAUSIBLE_SOH = 1.10


@dataclass(frozen=True)
class ChargeReading:
    charge: int                 # charge number within the log, from 1
    start_time_s: float
    end_time_s: float
    r_step_ohm: float
    r_trailing_ohm: float
    window_charge_ah: float     # NaN when the charge does not span the window


def charge_readings(
    telemetry: pd.DataFrame,
    rest_threshold_a: float,
    spec: WindowSpec = CHARGE_WINDOW,
) -> list[ChargeReading]:
    """One reading per charge phase, in order, with the window shifted up by I*R."""
    t = pd.to_numeric(telemetry["test_time_s"], errors="coerce").to_numpy(float)
    i = pd.to_numeric(telemetry["current_a"], errors="coerce").to_numpy(float)
    v = pd.to_numeric(telemetry["voltage_v"], errors="coerce").to_numpy(float)
    phases = [p for p in segment_phases(telemetry, rest_threshold_a=rest_threshold_a) if p.kind == "charge"]

    raw = []
    for p in phases:
        a, b = p.start_index, p.end_index + 1
        r = float("nan")
        j = a - 1
        if j >= 0:
            quiet = max(rest_threshold_a, QUIET_FRACTION * abs(i[a]))
            if abs(i[j]) <= quiet and (t[a] - t[j]) <= MAX_STEP_GAP_S:
                d_i = abs(i[a] - i[j])
                d_v = v[a] - v[j]                     # voltage RISES at a charge step
                if d_i >= MIN_STEP_CURRENT_A and np.isfinite(d_v) and d_v > 0:
                    r = d_v / d_i
        seg_t, seg_i = t[a:b], i[a:b]
        q = np.concatenate([[0.0], np.cumsum(np.abs(0.5 * (seg_i[1:] + seg_i[:-1])) * np.diff(seg_t)) / 3600.0])
        raw.append((p, r, v[a:b], q, float(np.median(np.abs(seg_i)))))

    screened = screen_step_resistances([r for _, r, _, _, _ in raw])
    out, last = [], float("nan")
    for k, (p, r, vv, q, current) in enumerate(raw):
        window = screened[max(0, k - RESISTANCE_WINDOW_CYCLES + 1):k + 1]
        finite = window[np.isfinite(window)]
        trailing = float(np.median(finite)) if len(finite) else last
        last = trailing if np.isfinite(trailing) else last
        charge_ah = float("nan")
        if np.isfinite(trailing):
            rise = current * trailing
            charge_ah = window_charge(vv, q, WindowSpec(spec.v_high + rise, spec.v_low + rise))
        out.append(ChargeReading(k + 1, p.start_time_s, p.end_time_s, float(r), float(trailing), charge_ah))
    return out


def charge_soh_table(
    readings: list[ChargeReading],
    reference_cycles: int = DEFAULT_REFERENCE_CYCLES,
) -> pd.DataFrame:
    """SOH per charge against the median of the first measurable charges."""
    df = pd.DataFrame([r.__dict__ for r in readings])
    if df.empty:
        return df.assign(soh=pd.Series(dtype=float), refusal=pd.Series(dtype=str))
    measurable = df.dropna(subset=["window_charge_ah"])
    if len(measurable) <= reference_cycles:
        df["soh"] = float("nan")
        df["refusal"] = (f"only {len(measurable)} charge(s) spanned the {CHARGE_WINDOW} window; "
                         f"{reference_cycles} are needed to form the reference and one more to measure")
        return df
    ref = float(measurable["window_charge_ah"].head(reference_cycles).median())
    soh = df["window_charge_ah"] / ref
    bad = (soh < MIN_PLAUSIBLE_SOH) | (soh > MAX_PLAUSIBLE_SOH)
    df["soh"] = soh.where(~bad)
    df["refusal"] = np.where(df["window_charge_ah"].isna(), "charge did not span the window",
                             np.where(bad, "ratio outside the plausible 0.5-1.10", ""))
    return df
