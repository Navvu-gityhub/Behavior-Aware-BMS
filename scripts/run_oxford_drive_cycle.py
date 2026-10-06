"""Can the window be read on a real drive-cycle discharge? One-discharge test.

    python scripts/run_oxford_drive_cycle.py

Needs data/raw/oxford/ExampleDC_C1.mat and Oxford_Battery_Degradation_Dataset_1.mat.
Writes reports/metrics/oxford/drive_cycle_report.md.

Every validation so far used constant-current discharges. A vehicle draws a
current that changes every second. ExampleDC_C1 is one real Artemis urban
drive-cycle discharge of Oxford Cell 1 at beginning of life (1 Hz, current in
mA). The same cell's beginning-of-life characterisation includes a C/18
pseudo-OCV discharge, where ohmic sag is negligible - an independent reference
for what the window SHOULD hold.

METHOD, FIXED BEFORE THIS RAN
-----------------------------
1. Resistance from the drive cycle's own current steps: for every 1-s step
   with |dI| >= 0.2 A, R = dV/dI; the median is used. No rest is needed - a
   drive cycle steps its current constantly.
2. Per-sample correction to an open-circuit estimate: V_oc = V - I*R
   (discharge current negative, so V_oc > V under load).
3. V_oc is not monotonic in charge (polarisation relaxes between pulses), so a
   decreasing isotonic fit of V_oc against discharged charge is taken, and the
   3.90-3.60 V window charge read from it.
4. Compared with the same window on the C/18 discharge of the same cell at
   cycle 0, and with the uncorrected drive-cycle window.

FOUND ON RUNNING, AND HANDLED IN THE OPEN
-----------------------------------------
The drive cycle is itself a partial discharge: it stops at ~483 mAh (~65% of
capacity), and once corrected the open-circuit estimate never reaches 3.60 V,
so the fixed 3.90-3.60 V window is not covered - reported as such. A BMS in
this position uses a window inside the band its discharges actually cover
(health/field_soh.learn_usage_window). So a COVERAGE window is also read: the
central 60% of the corrected curve's voltage span, chosen from coverage alone,
before any comparison with the reference was computed.

ONE DISCHARGE: this demonstrates the mechanism on real dynamic data; it is not
a population error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.voltage_window import WindowSpec, window_charge  # noqa: E402
from src.bms.io.load_oxford import load_oxford_mat  # noqa: E402

RAW = Path("data/raw/oxford")
OUT = Path("reports/metrics/oxford")
SPEC = WindowSpec(3.90, 3.60)
MIN_STEP_A = 0.2


def isotonic_window_spec(voltage: np.ndarray, charge: np.ndarray, spec: WindowSpec) -> float:
    fit = IsotonicRegression(increasing=False).fit_transform(charge, voltage)
    return window_charge(fit, charge, spec)


def isotonic_window(voltage: np.ndarray, charge: np.ndarray) -> float:
    fit = IsotonicRegression(increasing=False).fit_transform(charge, voltage)
    return window_charge(fit, charge, SPEC)


def main() -> int:
    m = loadmat(str(RAW / "ExampleDC_C1.mat"), struct_as_record=False, squeeze_me=True)
    dc = m["ExampleDC_C1"].dc
    v = np.asarray(dc.v, float)
    i = np.asarray(dc.i, float) / 1000.0          # mA -> A, discharge negative
    q = np.abs(np.asarray(dc.q, float)) / 1000.0  # mAh -> Ah, discharged charge

    dv, di = np.diff(v), np.diff(i)
    step = np.abs(di) >= MIN_STEP_A
    r_steps = dv[step] / di[step]
    r_steps = r_steps[np.isfinite(r_steps) & (r_steps > 0)]
    r = float(np.median(r_steps))

    v_oc = v - i * r
    raw_window = isotonic_window(v, q)
    comp_window = isotonic_window(v_oc, q)
    fitted = IsotonicRegression(increasing=False).fit_transform(q, v_oc)
    top, bottom = float(fitted.max()), float(fitted.min())
    pad = 0.2 * (top - bottom)
    cover = WindowSpec(round(top - pad, 3), round(bottom + pad, 3))
    cover_window = window_charge(fitted, q, cover)
    cover_window_raw = isotonic_window_spec(v, q, cover)

    tel, _ = load_oxford_mat(RAW / "Oxford_Battery_Degradation_Dataset_1.mat")
    ref = tel[(tel["cell_id"] == "Cell1") & (tel["cycle"] == 0)]
    ocv = ref[ref["measurement"] == "OCVdc"]
    c1 = ref[ref["measurement"] == "C1dc"]
    ocv_window = window_charge(ocv["voltage_v"].to_numpy(float),
                               ocv["capacity_ah_curve"].abs().to_numpy(float), SPEC)
    c1_window = window_charge(c1["voltage_v"].to_numpy(float),
                              c1["capacity_ah_curve"].abs().to_numpy(float), SPEC)

    ocv_cover = window_charge(ocv["voltage_v"].to_numpy(float),
                              ocv["capacity_ah_curve"].abs().to_numpy(float), cover)
    c1_cover = window_charge(c1["voltage_v"].to_numpy(float),
                             c1["capacity_ah_curve"].abs().to_numpy(float), cover)
    cover_rows = pd.DataFrame([
        {"reading": "drive cycle, corrected per sample", "window_charge_ah": cover_window},
        {"reading": "drive cycle, uncorrected", "window_charge_ah": cover_window_raw},
        {"reading": "1C constant-current discharge, uncorrected", "window_charge_ah": c1_cover},
        {"reading": "C/18 pseudo-OCV discharge (reference)", "window_charge_ah": ocv_cover},
    ])
    cover_rows["ratio_to_reference"] = cover_rows["window_charge_ah"] / ocv_cover
    rows = pd.DataFrame([
        {"reading": "drive cycle, corrected per sample", "window_charge_ah": comp_window},
        {"reading": "drive cycle, uncorrected", "window_charge_ah": raw_window},
        {"reading": "1C constant-current discharge, uncorrected", "window_charge_ah": c1_window},
        {"reading": "C/18 pseudo-OCV discharge (reference)", "window_charge_ah": ocv_window},
    ])
    rows["ratio_to_reference"] = rows["window_charge_ah"] / ocv_window
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# The voltage window on a real drive-cycle discharge (one discharge)", "",
        "Generated by `scripts/run_oxford_drive_cycle.py`; method fixed before it "
        "ran (see its docstring). Oxford Cell 1, beginning of life, Artemis urban "
        f"drive cycle: {len(v)} samples at 1 Hz, current {i.min():.2f} to "
        f"{i.max():.2f} A.", "",
        f"Resistance from {len(r_steps)} current steps of at least {MIN_STEP_A} A: "
        f"median {r * 1000:.0f} mOhm (interquartile "
        f"{np.percentile(r_steps, 25) * 1000:.0f}-{np.percentile(r_steps, 75) * 1000:.0f}).", "",
        f"Window {SPEC}:", "",
        "```", rows.round(4).to_string(index=False), "```", "",
        "The fixed window is NOT covered by this discharge (NaN): the drive "
        "cycle stops at ~65% depth. Coverage window, chosen from the corrected "
        f"curve's span alone before comparing: {cover}", "",
        "```", cover_rows.round(4).to_string(index=False), "```", "",
        "A ratio of 1.00 means the reading equals the near-equilibrium reference "
        "measured on the same cell at the same age.",
    ]
    (OUT / "drive_cycle_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
