"""Does anchoring the overpotential fix the cold-temperature failure?

    python scripts/run_temperature_fix_study.py

Writes reports/metrics/temperature_fix/. Inputs as run_cross_dataset_study.py.

THE PROBLEM (measured 2026-10-06, reports/metrics/cross_dataset/)
-----------------------------------------------------------------
The shipped window SOH, unchanged, erred 12-14% on NASA cells at 4 and
22 C but 1.7% at 43 C. Hypothesis, stated before this ran: the step
resistance captures only the sag one sample after the load starts; the
charge-transfer and diffusion overpotential that builds during the discharge
grows with age, and much faster in the cold, moving the curve through the
fixed window.

THE FIX, FIXED BEFORE THIS RAN (field_soh.anchored_overpotential)
-----------------------------------------------------------------
For a discharge that starts from full charge, the voltage gap from the cell's
own reference discharges over 5-15% of the reference charge is overpotential
growth. Each cycle's window is shifted by the reference's ohmic sag plus that
gap. Band 5-15%, rest-voltage tolerance 30 mV and 21 grid points were chosen
from the physics above, not tuned; nothing is fitted across cells. Nothing in
the method or the criteria was changed after the first results were seen;
the only later edit saves the per-reading rows (data/interim/) for the
follow-up study run_correction_selection_study.py.

ARMS: shipped (step resistance) vs anchored, window 3.90-3.60 V, same gates
(field_soh.apply_field_gates), same truth as the cross-dataset study.
Oxford has no rest-to-load step: shipped is uncorrected, anchored uses a zero
base level and no starts-from-full check (its 1C checks follow a full charge
by protocol - an assumption, stated).

CRITERIA, FIXED BEFORE THIS RAN
-------------------------------
Success: median per-cell MAE in the NASA cold cohorts (4 C and 22 C)
  falls by at least one third, AND no regression beyond tolerance:
  CALCE +0.3 points, Oxford +0.5 points, NASA warm (24, 43, 44 C) +0.5 points.
Also reported: cells scored per arm (a fix that wins by refusing is not a
win), and the consistency gate (u <= 0.01) on the anchored arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.benchmarks import add_targets  # noqa: E402
from src.bms.health.field_soh import (  # noqa: E402
    anchored_overpotential,
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
from src.bms.io.load_oxford import load_oxford_mat, summarize_oxford_cycles  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

OUT = Path("reports/metrics/temperature_fix")
SPEC = WindowSpec(3.90, 3.60)
TOLERANCE, RECENT, REPORT = 0.01, 10, 5
REGRESSION = {"CALCE": 0.003, "Oxford": 0.005, "NASA warm": 0.005}


def score(curves, op, disc, ref_cycles) -> pd.DataFrame:
    if op is not None:
        op = op.dropna(subset=["ir_drop_v"])
        curves = curves[curves["cycle"].isin(set(op["cycle"]))]
    if curves.empty:
        return pd.DataFrame(columns=["cycle", "soh"])
    t = apply_field_gates(window_soh_table(curves, spec=SPEC, reference_cycles=ref_cycles,
                                           overpotential=op), disc, ref_cycles)
    return t[["cycle", "soh_window_accepted"]].rename(columns={"soh_window_accepted": "soh"})


def arms(curves, steps, disc, ref_cycles, has_step=True) -> dict[str, pd.DataFrame]:
    base = step_overpotential(steps).dropna(subset=["ir_drop_v"]) if has_step else None
    rest = steps[["cell_id", "cycle", "v_rest_v"]] if has_step else None
    anch = anchored_overpotential(curves, base=base, rest=rest, reference_cycles=ref_cycles)
    return {"shipped": score(curves, base, disc, ref_cycles),
            "anchored": score(curves, anch, disc, ref_cycles)}


def calce() -> list[dict]:
    truth = add_targets(pd.read_csv("reports/metrics/calce_full_discharge.csv"))
    truth = truth[truth["soh"].notna()]
    out = []
    for path in sorted(Path("data/interim/calce_telemetry").glob("*.parquet")):
        cell = path.stem
        lab = truth[truth["cell_id"] == cell]
        if lab.empty or lab["cohort"].iloc[0] in ("CS2_Type5", "CS2_Type6"):
            continue
        tel = pd.read_parquet(path)
        d = cycles_to_frame(measure_cycles(tel, cell, rest_threshold_a=0.02), complete_only=False)
        t = tel["test_time_s"].to_numpy(float)
        arbin = tel["cycle"].to_numpy(float)
        d["arbin_cycle"] = [arbin[min(np.searchsorted(t, s), len(t) - 1)] for s in d["start_time_s"]]
        curves, steps = curves_from_telemetry(tel, d, cell, 0.02)
        if steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            continue
        lab = lab.sort_values("arbin_cycle_index")
        ref = lab["capacity_ah"].head(5).median()
        lab = lab.assign(soh_true=lab["capacity_ah"] / ref)[["arbin_cycle_index", "soh_true"]]
        for arm, est in arms(curves, steps, d, 5).items():
            j = est.merge(d[["cycle", "arbin_cycle"]], on="cycle").merge(
                lab, left_on="arbin_cycle", right_on="arbin_cycle_index")
            out.append({"dataset": "CALCE", "group": "CALCE", "cell_id": cell, "arm": arm, "rows": j})
        print("CALCE", cell, flush=True)
    return out


def nasa() -> list[dict]:
    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    out = []
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        cell = path.stem
        if cell in excluded:
            continue
        ds = [d for d in load_nasa_pcoe_cell(path) if 0.3 <= d.capacity_ah <= 2.5]
        if len(ds) < 10:
            continue
        curves, steps = discharges_to_frames(ds)
        if steps.empty or steps["r_step_ohm"].notna().sum() == 0:
            continue
        disc = pd.DataFrame({"cycle": [d.index for d in ds], "capacity_ah": [d.capacity_ah for d in ds]})
        ref = float(np.median([d.capacity_ah for d in ds[:5]]))
        truth = pd.DataFrame({"cycle": [d.index for d in ds], "soh_true": [d.capacity_ah / ref for d in ds]})
        cohort = cohort_for(ds)
        group = "NASA cold" if int(cohort.split("C")[0]) <= 22 else "NASA warm"
        for arm, est in arms(curves, steps, disc, 5).items():
            out.append({"dataset": "NASA", "group": group, "cohort": cohort, "cell_id": cell,
                        "arm": arm, "rows": est.merge(truth, on="cycle")})
        print("NASA", cell, cohort, flush=True)
    return out


def oxford() -> list[dict]:
    tel, _ = load_oxford_mat("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
    truth = summarize_oxford_cycles(tel)[["cell_id", "cycle", "capacity_ah"]]
    c1 = tel[tel["measurement"] == "C1dc"][["cell_id", "cycle", "voltage_v", "capacity_ah_curve"]].copy()
    c1["capacity_ah_curve"] = c1["capacity_ah_curve"].abs()
    disc_all = c1.groupby(["cell_id", "cycle"], as_index=False)["capacity_ah_curve"].max().rename(
        columns={"capacity_ah_curve": "capacity_ah"})
    out = []
    for cell, cc in c1.groupby("cell_id"):
        lab = truth[truth["cell_id"] == cell].sort_values("cycle")
        lab = lab.assign(soh_true=lab["capacity_ah"] / lab["capacity_ah"].iloc[0])[["cycle", "soh_true"]]
        disc = disc_all[disc_all["cell_id"] == cell]
        shipped = score(cc, None, disc, 1)
        anchored = score(cc, anchored_overpotential(cc, reference_cycles=1), disc, 1)
        for arm, est in (("shipped", shipped), ("anchored", anchored)):
            j = est.merge(lab, on="cycle")
            out.append({"dataset": "Oxford", "group": "Oxford", "cell_id": cell, "arm": arm,
                        "rows": j[j["cycle"] > 0]})
    return out


def _se(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return 1.2533 * float(np.median(np.abs(values - np.median(values)))) / np.sqrt(len(values))


def gate_points(rows: pd.DataFrame) -> list[tuple[float, float]]:
    g = rows.dropna(subset=["soh"]).sort_values("cycle")
    soh, truth = g["soh"].to_numpy(float), g["soh_true"].to_numpy(float)
    ref_se, pts = _se(soh[:5]), []
    for k in range(9, len(soh) + 1):
        recent = soh[max(5, k - RECENT):k]
        idx = np.arange(len(recent))
        res = recent - np.polyval(np.polyfit(idx, recent, 1), idx)
        rse = 1.2533 * float(np.median(np.abs(res - np.median(res)))) / np.sqrt(REPORT)
        u = float(np.sqrt(np.nansum([ref_se ** 2, rse ** 2])))
        pts.append((u, abs(float(np.median(soh[max(5, k - REPORT):k])) - truth[k - 1])))
    return pts


def main() -> int:
    results = calce() + nasa() + oxford()
    per_cell = []
    gate = []
    for r in results:
        rows = r["rows"].dropna(subset=["soh", "soh_true"])
        per_cell.append({"dataset": r["dataset"], "group": r["group"], "cohort": r.get("cohort", r["group"]),
                         "cell_id": r["cell_id"], "arm": r["arm"], "rows_scored": len(rows),
                         "mae": float((rows["soh"] - rows["soh_true"]).abs().mean()) if len(rows) else np.nan})
        if r["dataset"] != "Oxford":
            for u, e in gate_points(r["rows"].dropna(subset=["soh_true"])):
                gate.append({"group": r["group"], "arm": r["arm"], "cell_id": r["cell_id"], "u": u, "e": e})
    pc = pd.DataFrame(per_cell)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    pd.concat([r["rows"].assign(dataset=r["dataset"], group=r["group"], cohort=r.get("cohort", r["group"]),
                                cell_id=r["cell_id"], arm=r["arm"])[
        ["dataset", "group", "cohort", "cell_id", "arm", "cycle", "soh", "soh_true"]] for r in results],
        ignore_index=True).to_parquet("data/interim/temperature_fix_rows.parquet", index=False)
    OUT.mkdir(parents=True, exist_ok=True)
    pc.round(5).to_csv(OUT / "per_cell.csv", index=False)

    summ = (pc.dropna(subset=["mae"]).groupby(["group", "arm"])
            .agg(cells=("mae", "count"), median_cell_mae=("mae", "median")).round(4).reset_index())
    cohort = (pc[pc["dataset"] == "NASA"].dropna(subset=["mae"]).groupby(["cohort", "arm"])
              .agg(cells=("mae", "count"), median_cell_mae=("mae", "median")).round(4).reset_index())
    summ.to_csv(OUT / "summary.csv", index=False)
    cohort.to_csv(OUT / "nasa_by_cohort.csv", index=False)

    def med(group, arm):
        s = summ[(summ["group"] == group) & (summ["arm"] == arm)]["median_cell_mae"]
        return float(s.iloc[0]) if len(s) else float("nan")

    cold_before, cold_after = med("NASA cold", "shipped"), med("NASA cold", "anchored")
    improved = cold_after <= cold_before * (2 / 3)
    regress = {g: med(g, "anchored") - med(g, "shipped") for g in REGRESSION}
    ok = all(regress[g] <= tol for g, tol in REGRESSION.items())
    verdict = "SUCCESS" if improved and ok else "NOT MET"

    gp = pd.DataFrame(gate)
    g_rows = []
    for (group, arm), g in gp.groupby(["group", "arm"]):
        for side, m in (("u <= 0.01", g["u"] <= TOLERANCE), ("u > 0.01", g["u"] > TOLERANCE)):
            b = g[m]
            g_rows.append({"group": group, "arm": arm, "side": side, "points": len(b),
                           "median_cell_median_error": round(float(b.groupby("cell_id")["e"].median().median()), 4)
                           if len(b) else np.nan})
    gtab = pd.DataFrame(g_rows)
    gtab.to_csv(OUT / "consistency_gate.csv", index=False)

    lines = ["# Anchored overpotential: the cold-temperature fix", "",
             "Generated by `scripts/run_temperature_fix_study.py`; method and criteria fixed "
             "before it ran (see its docstring).", "",
             "## Median per-cell MAE by group", "", "```", summ.to_string(index=False), "```", "",
             "## NASA by cohort", "", "```", cohort.to_string(index=False), "```", "",
             "## Criteria", "",
             f"- NASA cold: {cold_before:.4f} -> {cold_after:.4f} "
             f"({'met' if improved else 'not met'}: needs <= {cold_before * 2 / 3:.4f})"]
    lines += [f"- {g} change {regress[g]:+.4f} (tolerance +{tol:.3f}: "
              f"{'ok' if regress[g] <= tol else 'REGRESSION'})" for g, tol in REGRESSION.items()]
    lines += ["", f"**Verdict: {verdict}**", "",
              "## Consistency gate (u <= 0.01, fixed on CALCE)", "", "```", gtab.to_string(index=False), "```"]
    (OUT / "temperature_fix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
