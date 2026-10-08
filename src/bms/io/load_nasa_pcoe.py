"""NASA PCoE Battery Data Set: the original per-battery MATLAB files.

`load_nasa.py` reads tabular exports. This reads the files NASA distributes
(Saha & Goebel 2007; `B0005.mat` ... `B0056.mat`), so the field estimators can
run on NASA exactly as they run on CALCE and Oxford: from the voltage, current
and time of each discharge.

Each file holds one struct whose `cycle` array mixes `charge`, `discharge` and
`impedance` records. A discharge record carries `Voltage_measured`,
`Current_measured` (A, discharge negative), `Temperature_measured`, `Time` (s)
and NASA's own computed `Capacity` (Ah) - the ground truth. Records begin with
one or two rest samples before the load is applied, so the rest-to-load step
the field resistance estimate needs is present.

The cohort of a cell is its ambient temperature plus its discharge current
profile, the protocol axes NASA varied; see `cohort_for`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NasaDischarge:
    cell_id: str
    index: int              # discharge number within the cell, from 1
    ambient_c: float
    capacity_ah: float      # NASA's computed capacity - ground truth only
    time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray


def load_nasa_pcoe_cell(path: str | Path) -> list[NasaDischarge]:
    """Every discharge record in one battery file, in order."""
    from scipy.io import loadmat

    path = Path(path)
    cell_id = path.stem
    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    if cell_id not in raw:
        raise ValueError(f"load_nasa_pcoe_cell: {path.name} has no struct named {cell_id}")
    out: list[NasaDischarge] = []
    for record in np.atleast_1d(raw[cell_id].cycle):
        if str(record.type).strip().lower() != "discharge":
            continue
        d = record.data
        try:
            capacity = float(np.atleast_1d(d.Capacity)[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            capacity = float("nan")
        out.append(NasaDischarge(
            cell_id=cell_id, index=len(out) + 1,
            ambient_c=float(record.ambient_temperature),
            capacity_ah=capacity,
            time_s=np.asarray(d.Time, float),
            voltage_v=np.asarray(d.Voltage_measured, float),
            current_a=np.asarray(d.Current_measured, float),
            temperature_c=np.asarray(d.Temperature_measured, float),
        ))
    return out


def cohort_for(discharges: list[NasaDischarge]) -> str:
    """Ambient temperature and discharge-current level: NASA's protocol axes."""
    if not discharges:
        return "unknown"
    ambient = round(float(np.median([d.ambient_c for d in discharges])))
    loads = [float(np.median(np.abs(d.current_a[d.current_a < -0.05])))
             for d in discharges if (d.current_a < -0.05).any()]
    level = round(float(np.median(loads)), 1) if loads else float("nan")
    return f"{ambient}C_{level}A"


def discharges_to_frames(discharges: list[NasaDischarge]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(curves, steps) in the shapes health/voltage_window and field_soh use.

    curves: cell_id, cycle, voltage_v, capacity_ah_curve - the loaded portion
            of each discharge, charge integrated from its start.
    steps:  cell_id, cycle, r_step_ohm, v_rest_v, mean_current_a - the
            rest-to-load step at the head of each record.
    """
    curve_rows, step_rows = [], []
    for d in discharges:
        on = np.flatnonzero(d.current_a < -0.05)
        if len(on) < 5:
            continue
        a = int(on[0])
        t, i, v = d.time_s[a:], d.current_a[a:], d.voltage_v[a:]
        q = np.concatenate([[0.0], np.cumsum(np.abs(0.5 * (i[1:] + i[:-1])) * np.diff(t)) / 3600.0])
        curve_rows.append(pd.DataFrame({"cell_id": d.cell_id, "cycle": d.index,
                                        "voltage_v": v, "capacity_ah_curve": q}))
        r = float("nan")
        v_rest = float(d.voltage_v[a - 1]) if a >= 1 else float("nan")
        if a >= 1:
            di = abs(d.current_a[a] - d.current_a[a - 1])
            dv = d.voltage_v[a - 1] - d.voltage_v[a]
            if di >= 0.05 and dv > 0:
                r = dv / di
        step_rows.append({"cell_id": d.cell_id, "cycle": d.index, "r_step_ohm": r,
                          "v_rest_v": v_rest, "mean_current_a": float(np.mean(i))})
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(
        columns=["cell_id", "cycle", "voltage_v", "capacity_ah_curve"])
    return curves, pd.DataFrame(step_rows)
