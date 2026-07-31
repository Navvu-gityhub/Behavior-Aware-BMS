"""Generate CS2-format Arbin fixtures for regression tests.

These are FIXTURES, not CALCE measurements. They reproduce the seventeen-column
Arbin schedule export exactly — including the per-file Cycle_Index restart that
`_reconcile_cycle_index` exists to handle — so the loader is exercised against
the real column names and the real multi-file shape.

Any number produced from these is a property of the fixture. Nothing here may be
reported as a CALCE result.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ARBIN_COLUMNS = [
    "Data_Point", "Test_Time(s)", "Date_Time", "Step_Time(s)", "Step_Index",
    "Cycle_Index", "Current(A)", "Voltage(V)", "Charge_Capacity(Ah)",
    "Discharge_Capacity(Ah)", "Charge_Energy(Wh)", "Discharge_Energy(Wh)",
    "dV/dt(V/s)", "Internal_Resistance(Ohm)", "Is_FC_Data", "AC_Impedance(Ohm)",
    "ACI_Phase_Angle(Deg)",
]


def make_cell_file(
    path: Path,
    n_cycles: int = 5,
    samples_per_cycle: int = 40,
    initial_capacity_ah: float = 1.10,
    fade_per_cycle: float = 0.002,
    cycle_offset: int = 0,
    start_time_s: float = 0.0,
    seed: int = 0,
) -> None:
    """Write one CS2-format file. Cycle_Index restarts at 1, as CALCE's do."""
    rng = np.random.default_rng(seed)
    rows = []
    point = 1
    t = start_time_s

    for c in range(n_cycles):
        absolute_cycle = cycle_offset + c
        capacity = initial_capacity_ah - fade_per_cycle * absolute_cycle
        # Discharge: Arbin accumulates Discharge_Capacity monotonically and
        # resets it between cycles, which is why the loader takes a per-cycle max.
        for s in range(samples_per_cycle):
            frac = (s + 1) / samples_per_cycle
            rows.append({
                "Data_Point": point, "Test_Time(s)": t, "Date_Time": 0.0,
                "Step_Time(s)": frac * 3600, "Step_Index": 1,
                "Cycle_Index": c + 1,
                "Current(A)": -1.1 + rng.normal(0, 0.01),
                "Voltage(V)": 4.2 - 1.5 * frac + rng.normal(0, 0.005),
                "Charge_Capacity(Ah)": 0.0,
                "Discharge_Capacity(Ah)": capacity * frac,
                "Charge_Energy(Wh)": 0.0,
                "Discharge_Energy(Wh)": capacity * frac * 3.7,
                "dV/dt(V/s)": 0.0,
                "Internal_Resistance(Ohm)": 0.08 + 0.0005 * absolute_cycle,
                "Is_FC_Data": 0, "AC_Impedance(Ohm)": 0.0,
                "ACI_Phase_Angle(Deg)": 0.0,
            })
            point += 1
            t += 90.0
        # Charge back.
        for s in range(samples_per_cycle // 2):
            frac = (s + 1) / (samples_per_cycle // 2)
            rows.append({
                "Data_Point": point, "Test_Time(s)": t, "Date_Time": 0.0,
                "Step_Time(s)": frac * 3600, "Step_Index": 2,
                "Cycle_Index": c + 1,
                "Current(A)": 0.55 + rng.normal(0, 0.01),
                "Voltage(V)": 2.7 + 1.5 * frac + rng.normal(0, 0.005),
                "Charge_Capacity(Ah)": capacity * frac,
                "Discharge_Capacity(Ah)": capacity,
                "Charge_Energy(Wh)": capacity * frac * 3.7,
                "Discharge_Energy(Wh)": capacity * 3.7,
                "dV/dt(V/s)": 0.0,
                "Internal_Resistance(Ohm)": 0.08 + 0.0005 * absolute_cycle,
                "Is_FC_Data": 0, "AC_Impedance(Ohm)": 0.0,
                "ACI_Phase_Angle(Deg)": 0.0,
            })
            point += 1
            t += 90.0

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=ARBIN_COLUMNS).to_csv(path, index=False)


def make_cell(
    cell_dir: Path,
    cell_id: str = "CS2_33",
    n_files: int = 3,
    cycles_per_file: int = 5,
    **kwargs,
) -> None:
    """Write a cell as several date-named files, as CALCE distributes them.

    Filenames are deliberately ordered so that lexical sorting is WRONG:
    9_20_10 sorts after 10_04_10 as text while preceding it in time.
    """
    # Distinct month/year per file. A fixed five-date list silently overwrote
    # files once n_files exceeded five, leaving a cell whose cycle offsets came
    # from whichever write landed last -- a fade trajectory that went upward.
    def date_for(index: int) -> str:
        month = (index % 12) + 1
        year = 10 + index // 12
        day = 4 + (index % 3) * 8
        return f"{month}_{day:02d}_{year}"

    time_cursor = 0.0
    for i in range(n_files):
        make_cell_file(
            cell_dir / f"{cell_id}_{date_for(i)}.csv",
            n_cycles=cycles_per_file,
            cycle_offset=i * cycles_per_file,
            start_time_s=time_cursor,
            seed=i,
            **kwargs,
        )
        time_cursor += cycles_per_file * 60 * 90.0
