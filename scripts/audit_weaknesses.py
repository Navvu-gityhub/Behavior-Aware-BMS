"""Audit of the three open weaknesses, before any new method is designed.

    python scripts/audit_weaknesses.py

Writes reports/metrics/audit/. Diagnostic only: nothing here changes BEACON,
and no threshold or method is chosen from it without a separate,
pre-registered test.

A. Labels      NASA's stored capacity vs the charge integrated from its own
               current; cutoff voltages per cohort.
B. Cold        For every NASA discharge: the window shift that WOULD have
               made the window ratio equal measured capacity ("required
               shift"), against the ohmic shift BEACON applied. If the cold
               error is uncorrected polarisation, the missing shift grows
               with age and is largest in the cold. Also: did cell
               temperature drift between reference and later discharges?
C. Confidence  Is the consistency uncertainty u calibrated for total error?
               Share of estimates within 2u of truth, per dataset.
D. RUL         Is early-life information predictive? Variance of log life
               between vs within datasets; within each dataset, Spearman
               between fade in the first 10% of life and total life.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import step_overpotential  # noqa: E402
from src.bms.health.voltage_window import WindowSpec, window_charge  # noqa: E402
from src.bms.io.load_nasa_pcoe import cohort_for, discharges_to_frames, load_nasa_pcoe_cell  # noqa: E402

OUT = Path("reports/metrics/audit")
SPEC = WindowSpec(3.90, 3.60)


def nasa_cells():
    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        if path.stem in excluded:
            continue
        ds = [d for d in load_nasa_pcoe_cell(path) if 0.3 <= d.capacity_ah <= 2.5]
        if len(ds) >= 10:
            yield path.stem, ds


def required_shift(v: np.ndarray, q: np.ndarray, target_ah: float) -> float:
    """Downward window shift at which the window charge equals target_ah."""
    shifts = np.linspace(-0.2, 0.8, 501)
    charges = np.array([window_charge(v, q, WindowSpec(SPEC.v_high - s, SPEC.v_low - s)) for s in shifts])
    ok = np.isfinite(charges)
    if not ok.any():
        return float("nan")
    i = int(np.argmin(np.abs(charges[ok] - target_ah)))
    return float(shifts[ok][i]) if abs(charges[ok][i] - target_ah) < 0.02 * target_ah else float("nan")


def part_a_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    label_rows, rows = [], []
    for cell, ds in nasa_cells():
        cohort = cohort_for(ds)
        curves, steps = discharges_to_frames(ds)
        op = step_overpotential(steps).set_index("cycle")
        ref_cap = float(np.median([d.capacity_ah for d in ds[:5]]))
        ref_ch = []
        for d in ds:
            c = curves[curves["cycle"] == d.index]
            if c.empty:
                continue
            q, v = c["capacity_ah_curve"].to_numpy(float), c["voltage_v"].to_numpy(float)
            label_rows.append({"cell_id": cell, "cohort": cohort, "cycle": d.index,
                               "nasa_capacity_ah": d.capacity_ah, "integrated_ah": float(q.max()),
                               "min_voltage_v": float(v.min())})
            drop = float(op["ir_drop_v"].get(d.index, np.nan)) if d.index in op.index else np.nan
            ch = window_charge(v, q, WindowSpec(SPEC.v_high - drop, SPEC.v_low - drop)) if np.isfinite(drop) else np.nan
            if len(ref_ch) < 5 and np.isfinite(ch):
                ref_ch.append(ch)
            t = d.temperature_c
            rows.append({"cell_id": cell, "cohort": cohort, "cycle": d.index, "applied_shift_v": drop,
                         "window_charge_ah": ch, "soh_true": d.capacity_ah / ref_cap,
                         "temp_start_c": float(t[0]), "temp_mean_c": float(np.mean(t)), "temp_max_c": float(np.max(t)),
                         "_v": v, "_q": q})
        if len(ref_ch) < 5:
            continue
        ref_w = float(np.median(ref_ch))
        for r in rows:
            if r["cell_id"] == cell and "ref_w" not in r:
                r["ref_w"] = ref_w
                r["soh_est"] = r["window_charge_ah"] / ref_w if np.isfinite(r["window_charge_ah"]) else np.nan
                r["required_shift_v"] = required_shift(r["_v"], r["_q"], r["soh_true"] * ref_w)
    for r in rows:
        r.pop("_v", None)
        r.pop("_q", None)
    b = pd.DataFrame(rows).dropna(subset=["ref_w"])
    b["missing_shift_v"] = b["required_shift_v"] - b["applied_shift_v"]
    b["error"] = b["soh_est"] - b["soh_true"]
    return pd.DataFrame(label_rows), b


def part_c() -> pd.DataFrame:
    rows = []
    m = Path("reports/metrics/window_agreement/m50t_points.csv")
    if m.exists():
        p = pd.read_csv(m)
        rows.append(("M50T", p["u"].to_numpy(float), p["error"].to_numpy(float)))
    table = Path("data/interim/cross_dataset_checks.parquet")
    if table.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_window_agreement_study import points
        dev = pd.read_parquet(table)
        dev = dev[dev["soh_true"].notna()]
        for ds, g in dev.groupby("dataset"):
            pts = pd.DataFrame([x for _, c in g.groupby("cell_id") for x in points(c)])
            rows.append((ds, pts["u"].to_numpy(float), pts["error"].to_numpy(float)))
    out = []
    for ds, u, e in rows:
        ok = np.isfinite(u) & np.isfinite(e) & (u > 0)
        u, e = u[ok], e[ok]
        out.append({"dataset": ds, "points": len(u), "median_u": round(float(np.median(u)), 4),
                    "median_error": round(float(np.median(e)), 4),
                    "share_within_2u": round(float(np.mean(e <= 2 * u)), 3),
                    "median_error_over_u": round(float(np.median(e / u)), 1)})
    return pd.DataFrame(out)


def part_d() -> pd.DataFrame:
    cells = pd.read_csv("reports/metrics/population_rul/cells.csv")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_population_rul_study import end_of_life, trajectories
    trajs = {t["cell_id"]: t for t in trajectories()}
    rows = []
    for _, c in cells.dropna(subset=["n_eol_0p90"]).iterrows():
        t = trajs.get(c["cell_id"])
        if t is None:
            continue
        n_eol = end_of_life(t)
        x0 = t["x"][0]
        if not np.isfinite(n_eol) or n_eol <= x0:
            continue
        early = t["x"] <= x0 + 0.10 * (n_eol - x0)
        fade_early = float(1 - np.median(t["soh"][early][-3:])) if early.sum() >= 3 else np.nan
        rows.append({"dataset": c["dataset"], "cell_id": c["cell_id"], "life": n_eol - x0,
                     "fade_first_10pct": fade_early})
    d = pd.DataFrame(rows)
    d["log_life"] = np.log(d["life"])
    total = float(d["log_life"].var())
    within = float(d.groupby("dataset")["log_life"].var().mul(d.groupby("dataset").size() - 1).sum()
                   / (len(d) - d["dataset"].nunique()))
    out = [{"dataset": "ALL", "cells": len(d), "life_median": float(d["life"].median()),
            "var_log_life_total": round(total, 3), "var_log_life_within": round(within, 3),
            "share_between_datasets": round(1 - within / total, 2), "spearman_early_fade_vs_life": np.nan}]
    for ds, g in d.groupby("dataset"):
        g = g.dropna(subset=["fade_first_10pct"])
        rho = float(spearmanr(g["fade_first_10pct"], g["life"]).statistic) if len(g) >= 4 else np.nan
        out.append({"dataset": ds, "cells": len(g), "life_median": float(g["life"].median()),
                    "var_log_life_total": round(float(g["log_life"].var()), 3), "var_log_life_within": np.nan,
                    "share_between_datasets": np.nan, "spearman_early_fade_vs_life": round(rho, 2)})
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels, b = part_a_b()
    labels["label_gap"] = (labels["nasa_capacity_ah"] - labels["integrated_ah"]) / labels["integrated_ah"]
    a = labels.groupby("cohort").agg(cells=("cell_id", "nunique"),
                                     median_label_gap=("label_gap", "median"),
                                     p95_abs_label_gap=("label_gap", lambda s: s.abs().quantile(0.95)),
                                     median_cutoff_v=("min_voltage_v", "median")).round(4).reset_index()
    b.round(5).to_csv(OUT / "nasa_cold_mechanism.csv", index=False)

    def late(g):
        g = g.sort_values("cycle")
        return g.tail(max(len(g) // 4, 3))

    bl = b.groupby("cell_id", group_keys=False).apply(late)
    first = b.sort_values("cycle").groupby("cell_id").head(5).groupby("cell_id")["temp_mean_c"].mean()
    bl = bl.assign(temp_drift_c=bl["temp_mean_c"] - bl["cell_id"].map(first))
    mech = bl.groupby("cohort").agg(
        cells=("cell_id", "nunique"), late_life_error=("error", "median"),
        applied_shift_v=("applied_shift_v", "median"), required_shift_v=("required_shift_v", "median"),
        missing_shift_v=("missing_shift_v", "median"), temp_mean_c=("temp_mean_c", "median"),
        temp_drift_c=("temp_drift_c", "median")).round(4).reset_index()
    rho = spearmanr(bl["missing_shift_v"], bl["error"], nan_policy="omit").statistic
    c = part_c()
    d = part_d()
    for name, df in (("A_labels", a), ("B_cold_mechanism", mech), ("C_calibration", c), ("D_rul_predictability", d)):
        df.to_csv(OUT / f"{name}.csv", index=False)
    lines = ["# Audit of the three open weaknesses", "",
             "Generated by `scripts/audit_weaknesses.py`. Diagnostic only.", "",
             "## A. NASA labels: stored capacity vs integrated charge, cutoff voltage", "",
             "```", a.to_string(index=False), "```", "",
             "## B. Cold mechanism (last quarter of each cell's life)", "",
             "required_shift: the window shift at which the window ratio equals measured capacity; "
             "applied_shift: BEACON's ohmic shift; missing = required - applied. "
             "temp_drift: mean discharge temperature minus that of the cell's first 5 discharges.", "",
             "```", mech.to_string(index=False), "```", "",
             f"Spearman(missing shift, SOH error), late life, all NASA cells: {rho:.2f}", "",
             "## C. Is the consistency uncertainty u calibrated for total error?", "",
             "If u were a calibrated standard error, ~95% of estimates would lie within 2u of truth.", "",
             "```", c.to_string(index=False), "```", "",
             "## D. Is early-life information predictive of remaining life?", "",
             "```", d.to_string(index=False), "```"]
    (OUT / "audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
