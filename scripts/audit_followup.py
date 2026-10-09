"""Follow-up audit: labels, temperature changes within a cell, and M50T bias.

    python scripts/audit_followup.py

Writes reports/metrics/audit/followup_*.csv and followup_report.md.
Diagnostic only, like audit_weaknesses.py.

1. NASA truth. The stored `Capacity` disagrees with the cell's own
   integrated discharge charge, by up to ~29% in the 4 C / 2 A cohort. Here
   each discharge is also scored against a label the curve itself defines:
   charge delivered until the loaded voltage first falls below 2.7 V (above
   every cohort's cutoff, so every discharge reaches it). If the cold error
   shrinks against that label, part of it was the label.
2. Ambient changes. Per NASA cell, the distinct ambient settings across its
   discharges, and the error split by whether a discharge is at the same
   ambient as the cell's reference discharges.
3. M50T bias. Cell D (25 C plate): signed error, by discharge current.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bms.health.field_soh import (  # noqa: E402
    apply_field_gates,
    curves_from_telemetry,
    step_overpotential,
)
from src.bms.health.voltage_window import WindowSpec, window_soh_table  # noqa: E402
from src.bms.io.load_nasa_pcoe import (  # noqa: E402
    cohort_for,
    discharges_to_frames,
    load_nasa_pcoe_cell,
)
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

OUT = Path("reports/metrics/audit")
SPEC = WindowSpec(3.90, 3.60)
COMMON_CUTOFF_V = 2.7


def charge_to(v: np.ndarray, q: np.ndarray, cutoff: float) -> float:
    below = np.flatnonzero(v < cutoff)
    return float(q[below[0]]) if len(below) else float(q[-1])


def nasa() -> pd.DataFrame:
    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    rows = []
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        if path.stem in excluded:
            continue
        ds = [d for d in load_nasa_pcoe_cell(path) if 0.3 <= d.capacity_ah <= 2.5]
        if len(ds) < 10:
            continue
        curves, steps = discharges_to_frames(ds)
        op = step_overpotential(steps).dropna(subset=["ir_drop_v"])
        cv = curves[curves["cycle"].isin(set(op["cycle"]))]
        disc = pd.DataFrame({"cycle": [d.index for d in ds], "capacity_ah": [d.capacity_ah for d in ds]})
        est = apply_field_gates(window_soh_table(cv, spec=SPEC, overpotential=op), disc)
        est = est.set_index("cycle")["soh_window_accepted"]
        common = {d.index: charge_to(*(lambda c: (c["voltage_v"].to_numpy(float),
                                                  c["capacity_ah_curve"].to_numpy(float)))(
            curves[curves["cycle"] == d.index]), COMMON_CUTOFF_V) for d in ds
            if not curves[curves["cycle"] == d.index].empty}
        ref_nasa = float(np.median([d.capacity_ah for d in ds[:5]]))
        ref_common = float(np.median([common[d.index] for d in ds[:5] if d.index in common]))
        ref_ambient = float(np.median([d.ambient_c for d in ds[:5]]))
        for d in ds:
            if d.index not in common:
                continue
            rows.append({"cell_id": path.stem, "cohort": cohort_for(ds), "cycle": d.index,
                         "ambient_c": d.ambient_c, "same_ambient_as_reference": abs(d.ambient_c - ref_ambient) < 2,
                         "soh_est": est.get(d.index, np.nan),
                         "truth_nasa": d.capacity_ah / ref_nasa, "truth_common_cutoff": common[d.index] / ref_common})
    return pd.DataFrame(rows)


def m50t_cell_d() -> pd.DataFrame:
    from run_m50t_heldout_study import FULL_AH, REST_A, files_by_cell, load_cell
    files = files_by_cell()[(25, "D")]
    tel = load_cell(files)
    d = cycles_to_frame(measure_cycles(tel, "D", rest_threshold_a=REST_A), complete_only=False)
    curves, steps = curves_from_telemetry(tel, d, "D", REST_A)
    op = step_overpotential(steps).dropna(subset=["ir_drop_v"])
    cv = curves[curves["cycle"].isin(set(op["cycle"]))]
    est = apply_field_gates(window_soh_table(cv, spec=SPEC, overpotential=op), d)[["cycle", "soh_window_accepted"]]
    full = d[d["capacity_ah"] >= FULL_AH].sort_values("cycle")
    ref = float(full["capacity_ah"].head(5).median())
    j = d[["cycle", "capacity_ah", "mean_current_a"]].merge(est, on="cycle")
    j["truth"] = j["capacity_ah"] / ref
    j["current_class"] = np.where(j["mean_current_a"].abs() > 3, "5 A", "slower")
    j["signed_error"] = j["soh_window_accepted"] - j["truth"]
    j["is_full"] = j["capacity_ah"] >= FULL_AH
    return j


def main() -> int:
    n = nasa()
    n["err_nasa"] = (n["soh_est"] - n["truth_nasa"]).abs()
    n["err_common"] = (n["soh_est"] - n["truth_common_cutoff"]).abs()
    n["label_gap"] = n["truth_nasa"] - n["truth_common_cutoff"]
    n.round(5).to_csv(OUT / "followup_nasa.csv", index=False)
    per_cell = n.dropna(subset=["soh_est"]).groupby(["cohort", "cell_id"]).agg(
        err_nasa=("err_nasa", "mean"), err_common=("err_common", "mean"),
        ambients=("ambient_c", lambda s: ",".join(str(int(v)) for v in sorted(set(s.round())))))
    t1 = per_cell.groupby("cohort").agg(cells=("err_nasa", "count"), median_err_vs_nasa_label=("err_nasa", "median"),
                                        median_err_vs_common_cutoff=("err_common", "median"),
                                        ambients_seen=("ambients", lambda s: " | ".join(sorted(set(s))))).round(4)
    t2 = (n.dropna(subset=["soh_est"]).groupby(["cohort", "same_ambient_as_reference"])["err_common"]
          .agg(["count", "median"]).round(4))
    m = m50t_cell_d()
    m.round(5).to_csv(OUT / "followup_m50t_D.csv", index=False)
    t3 = (m.dropna(subset=["soh_window_accepted"]).groupby(["current_class", "is_full"])["signed_error"]
          .agg(["count", "median", "mean"]).round(4))
    lines = ["# Audit follow-up", "", "Generated by `scripts/audit_followup.py`. Diagnostic only.", "",
             "## 1. NASA cold error against two truth labels", "",
             "`truth_nasa`: NASA's stored Capacity. `truth_common_cutoff`: charge until the loaded voltage first "
             f"falls below {COMMON_CUTOFF_V} V, from the cell's own current. Per-cell MAE, median per cohort.", "",
             "```", t1.to_string(), "```", "",
             "## 2. Error by whether the discharge ran at the reference ambient (common-cutoff label)", "",
             "```", t2.to_string(), "```", "",
             "## 3. LG M50T cell D: signed error (estimate - truth) by discharge current", "",
             "```", t3.to_string(), "```"]
    (OUT / "followup_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
