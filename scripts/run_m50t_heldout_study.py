"""Held-out test on a fourth dataset: LG M50T 21700 cells at 10, 25 and 40 C.

    python scripts/fetch_imperial_m50t.py        # fixed subset, ~6 GB
    python scripts/run_m50t_heldout_study.py

Writes reports/metrics/m50t_heldout/. Needs galvani (pip install galvani) to
read the Biologic .mpr files.

WHY THIS DATASET
----------------
The anchored overpotential correction (field_soh.anchored_overpotential)
improved Oxford and the NASA cold cohorts but failed its preset criteria on
the data it was designed against (reports/metrics/temperature_fix/). A
further rule tuned on those same cells would be fishing. This dataset was
chosen before any of it was read, and nothing in BEACON was changed for it:
a different maker (LG), format (21700), chemistry (NMC811 / graphite-SiOx)
and capacity (5 Ah), each ageing cycle a full 1C discharge AT the test
temperature - 10, 25 or 40 C - so truth is the cycler's own charge count,
as on NASA. Kirkaldy et al., J. Power Sources 2024,
doi:10.1016/j.jpowsour.2024.234185; data doi:10.5281/zenodo.10637534.

FIXED BEFORE ANY DATA WAS READ
------------------------------
Subset: scripts/fetch_imperial_m50t.py (sets 1, 3, ..., 15 per cell).
Segmentation: the shipped field path - telemetry.cycles.measure_cycles and
field_soh.curves_from_telemetry, rest threshold 0.02 A as on CALCE.
Truth: each discharge's charge from the cycler current, over the cell's
median of its first 5 discharges of at least 4.0 Ah (a full discharge of a
5 Ah cell; shorter ones are interrupted or partial and are not truth).
Scored discharges: those same full discharges.
Arms: shipped (step resistance) and anchored, window 3.90-3.60 V, every gate
as shipped, reference = first 5 measurable discharges.

Criteria:
  Primary - 10 C cells: anchored median per-cell MAE <= 2/3 of shipped.
  Non-regression - 25 C and 40 C: anchored no worse than shipped + 0.5 points.
Also reported, not criteria: shipped error per temperature (does the
estimator itself generalise?), cells scored per arm, and the CALCE-fixed
consistency gate (u <= 0.01) per arm.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import (  # noqa: E402
    anchored_overpotential,
    apply_field_gates,
    curves_from_telemetry,
    step_overpotential,
)
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

RAW = Path("data/raw/imperial_m50t/cycling")
OUT = Path("reports/metrics/m50t_heldout")
SPEC = WindowSpec(3.90, 3.60)
REST_A = 0.02
FULL_AH = 4.0
TOLERANCE, RECENT, REPORT = 0.01, 10, 5
NAME = re.compile(r"(\d+)degC - cell (\w)")


def load_cell(files: list[Path]) -> pd.DataFrame:
    """Unified telemetry (test_time_s, current_a, voltage_v, temperature_c)."""
    from galvani import BioLogic

    frames, offset = [], 0.0
    for f in files:
        d = pd.DataFrame(BioLogic.MPRfile(str(f)).data)
        current = d["I/mA"] if "I/mA" in d else d["control/mA"]
        t = d["time/s"].to_numpy(float)
        frame = pd.DataFrame({"test_time_s": t - t[0] + offset,
                              "current_a": current.to_numpy(float) / 1000.0,
                              "voltage_v": d["Ewe/V"].to_numpy(float)})
        temp = [c for c in d.columns if "temperature" in c.lower() or c.startswith("Temp")]
        frame["temperature_c"] = d[temp[0]].to_numpy(float) if temp else np.nan
        frames.append(frame)
        offset = float(frame["test_time_s"].iloc[-1]) + 3600.0   # sets are separated by RPTs
    tel = pd.concat(frames, ignore_index=True)
    tel["cycle"] = 0
    return tel


def files_by_cell() -> dict[tuple[int, str], list[Path]]:
    out: dict[tuple[int, str], list[tuple[int, Path]]] = {}
    for f in RAW.glob("*.mpr"):
        m = NAME.search(f.name)
        s = re.search(r"cycling(\d+)", f.name)
        set_no = int(s.group(1)) if s else 1
        out.setdefault((int(m.group(1)), m.group(2)), []).append((set_no, f))
    return {k: [f for _, f in sorted(v)] for k, v in sorted(out.items())}


def score(curves, op, disc) -> pd.DataFrame:
    if op is not None:
        op = op.dropna(subset=["ir_drop_v"])
        curves = curves[curves["cycle"].isin(set(op["cycle"]))]
    if curves.empty:
        return pd.DataFrame(columns=["cycle", "soh"])
    t = apply_field_gates(window_soh_table(curves, spec=SPEC, overpotential=op), disc)
    return t[["cycle", "soh_window_accepted"]].rename(columns={"soh_window_accepted": "soh"})


def _se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return 1.2533 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(len(values))


def gate_points(rows: pd.DataFrame) -> list[tuple[float, float]]:
    g = rows.dropna(subset=["soh", "soh_true"]).sort_values("cycle")
    soh, truth = g["soh"].to_numpy(float), g["soh_true"].to_numpy(float)
    ref_se, pts = _se(soh[:5]), []
    for k in range(9, len(soh) + 1):
        recent = soh[max(5, k - RECENT):k]
        idx = np.arange(len(recent))
        res = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
        rse = 1.2533 * float(np.median(np.abs(res - np.median(res)))) / np.sqrt(REPORT)
        pts.append((float(np.sqrt(np.nansum([ref_se ** 2, rse ** 2]))),
                    abs(float(np.median(soh[max(5, k - REPORT):k])) - truth[k - 1])))
    return pts


def main() -> int:
    per_cell, gate, info = [], [], []
    for (temp, cell), files in files_by_cell().items():
        cell_id = f"M50T_{temp}C_{cell}"
        tel = load_cell(files)
        d = cycles_to_frame(measure_cycles(tel, cell_id, rest_threshold_a=REST_A), complete_only=False)
        curves, steps = curves_from_telemetry(tel, d, cell_id, REST_A)
        full = d[d["capacity_ah"] >= FULL_AH].sort_values("cycle")
        info.append({"cell_id": cell_id, "temperature_c": temp, "files": len(files),
                     "discharges": len(d), "full_discharges": len(full),
                     "steps_with_r": int(steps["r_step_ohm"].notna().sum()) if len(steps) else 0,
                     "median_r_step_ohm": float(steps["r_step_ohm"].median()) if len(steps) else np.nan})
        print(info[-1], flush=True)
        if len(full) < 10 or steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            continue
        ref = float(full["capacity_ah"].head(5).median())
        truth = full[["cycle"]].assign(soh_true=full["capacity_ah"].to_numpy(float) / ref)
        base = step_overpotential(steps).dropna(subset=["ir_drop_v"])
        anch = anchored_overpotential(curves, base=base, rest=steps[["cell_id", "cycle", "v_rest_v"]])
        for arm, op in (("shipped", base), ("anchored", anch)):
            rows = score(curves, op, d).merge(truth, on="cycle")
            ok = rows.dropna(subset=["soh"])
            per_cell.append({"cell_id": cell_id, "temperature_c": temp, "arm": arm,
                             "rows_scored": len(ok),
                             "mae": float((ok["soh"] - ok["soh_true"]).abs().mean()) if len(ok) else np.nan,
                             "final_soh_true": float(truth["soh_true"].iloc[-1])})
            gate += [{"temperature_c": temp, "arm": arm, "cell_id": cell_id, "u": u, "e": e}
                     for u, e in gate_points(rows)]

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(info).round(5).to_csv(OUT / "cells.csv", index=False)
    pc = pd.DataFrame(per_cell)
    pc.round(5).to_csv(OUT / "per_cell.csv", index=False)
    summ = (pc.dropna(subset=["mae"]).groupby(["temperature_c", "arm"])
            .agg(cells=("mae", "count"), median_cell_mae=("mae", "median")).round(4).reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)

    def med(temp, arm):
        s = summ[(summ["temperature_c"] == temp) & (summ["arm"] == arm)]["median_cell_mae"]
        return float(s.iloc[0]) if len(s) else float("nan")

    primary = med(10, "anchored") <= med(10, "shipped") * 2 / 3
    regress = {t: med(t, "anchored") - med(t, "shipped") for t in (25, 40)}
    ok = all(np.isfinite(v) and v <= 0.005 for v in regress.values())
    verdict = "SUCCESS" if primary and ok else "NOT MET"

    gp = pd.DataFrame(gate)
    g_rows = []
    if not gp.empty:
        for (temp, arm), g in gp.groupby(["temperature_c", "arm"]):
            for side, m in (("u <= 0.01", g["u"] <= TOLERANCE), ("u > 0.01", g["u"] > TOLERANCE)):
                b = g[m]
                g_rows.append({"temperature_c": temp, "arm": arm, "side": side, "points": len(b),
                               "median_cell_median_error":
                                   round(float(b.groupby("cell_id")["e"].median().median()), 4) if len(b) else np.nan})
    gtab = pd.DataFrame(g_rows)
    gtab.to_csv(OUT / "consistency_gate.csv", index=False)

    lines = ["# Held-out test: LG M50T 21700 at 10, 25 and 40 C", "",
             "Generated by `scripts/run_m50t_heldout_study.py`; dataset, subset, arms and criteria "
             "were fixed before any of the data was read (see its docstring).", "",
             "## Cells", "", "```", pd.DataFrame(info).round(4).to_string(index=False), "```", "",
             "## Median per-cell MAE", "", "```", summ.to_string(index=False), "```", "",
             "## Criteria", "",
             f"- 10 C: shipped {med(10, 'shipped'):.4f} -> anchored {med(10, 'anchored'):.4f} "
             f"({'met' if primary else 'not met'}: needs <= {med(10, 'shipped') * 2 / 3:.4f})"]
    lines += [f"- {t} C change {v:+.4f} (tolerance +0.005: {'ok' if v <= 0.005 else 'REGRESSION'})"
              for t, v in regress.items()]
    lines += ["", f"**Verdict: {verdict}**", "",
              "## Consistency gate (u <= 0.01, fixed on CALCE)", "", "```",
              gtab.to_string(index=False) if not gtab.empty else "(no points)", "```"]
    (OUT / "m50t_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
