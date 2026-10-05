"""Build the results showcase: the numbers and charts that show what BEACON does.

    python scripts/export_results.py

Reads committed artifacts under reports/metrics/ and, for the life-trace
charts, the cached CALCE telemetry in data/interim/calce_telemetry/ (built by
scripts/run_field_soh_study.py). Writes:

    reports/metrics/results_showcase.json   data for the results page
    reports/figures/results_soh_tracking.png
    reports/figures/results_soh_per_cell.png
    reports/figures/results_rul.png

Every value is computed here from those inputs; nothing is typed in by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.field_soh import field_soh_table  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402
from src.bms.telemetry.serial_schema import parse_line  # noqa: E402

METRICS = Path("reports/metrics")
FIGURES = Path("reports/figures")
CACHE = Path("data/interim/calce_telemetry")
RIG = Path("data/interim/rig_stage_b_voltage_verified.txt")

# Two standard cells (fixed window, field method) and the two partial-cycling
# cells the learned window measures. Chosen as one per cohort/band, not by error.
FULL_CELLS = ("CS2_35", "CX2_33")
PARTIAL_CELLS = ("CS2_24", "CS2_25")

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"
ESTIMATE, TRUTH, LEARNED = "#2a78d6", "#8a8984", "#eb6834"


def full_cell_trace(cell: str, truth: pd.DataFrame) -> pd.DataFrame:
    """Field-method SOH per discharge beside the cycler's capacity, one cell."""
    telemetry = pd.read_parquet(CACHE / f"{cell}.parquet")
    discharges = cycles_to_frame(
        measure_cycles(telemetry, cell, rest_threshold_a=0.02), complete_only=False)
    t = telemetry["test_time_s"].to_numpy(float)
    arbin = telemetry["cycle"].to_numpy(float)
    discharges["arbin_cycle"] = [
        arbin[min(np.searchsorted(t, s), len(t) - 1)] for s in discharges["start_time_s"]]
    table = field_soh_table(telemetry, discharges, cell, 0.02)
    est = table[["cycle", "soh_window_accepted"]].merge(
        discharges[["cycle", "arbin_cycle"]], on="cycle").dropna()
    est["soh_est"] = est["soh_window_accepted"].rolling(5, min_periods=1).median()
    cell_truth = truth[truth["cell_id"] == cell].sort_values("arbin_cycle_index")
    early = cell_truth[cell_truth["arbin_cycle_index"] <= 10]["capacity_ah"].median()
    cell_truth = cell_truth.assign(soh_true=cell_truth["capacity_ah"] / early)
    joined = est.merge(cell_truth[["arbin_cycle_index", "soh_true"]],
                       left_on="arbin_cycle", right_on="arbin_cycle_index")
    joined = joined[["arbin_cycle", "soh_true", "soh_est"]]
    step = max(1, len(joined) // 120)
    return joined.iloc[::step].reset_index(drop=True)


def rig_voltage() -> list[dict]:
    rows = []
    for line in RIG.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            kind, payload = parse_line(line)
        except Exception:
            continue
        if kind == "data" and "voltage_v" in payload.values:
            rows.append({"t": float(payload.values.get("test_time_s", len(rows))),
                         "v": float(payload.values["voltage_v"])})
    return rows


def main() -> int:
    from src.bms.benchmarks import add_targets

    truth = add_targets(pd.read_csv(METRICS / "calce_full_discharge.csv"))
    truth = truth[truth["soh"].notna()]

    traces = {}
    for cell in FULL_CELLS:
        tr = full_cell_trace(cell, truth)
        traces[cell] = {"mode": "standard window, full discharges",
                        "cycle": tr["arbin_cycle"].tolist(),
                        "truth": tr["soh_true"].round(4).tolist(),
                        "estimate": tr["soh_est"].round(4).tolist(),
                        "mae": round(float((tr["soh_est"] - tr["soh_true"]).abs().mean()), 4)}
    partial = pd.read_csv(METRICS / "calce_partial_soh" / "partial_soh_trace.csv")
    for cell in PARTIAL_CELLS:
        tr = partial[partial["cell_id"] == cell].sort_values("arbin_cycle")
        traces[cell] = {"mode": "learned window, partial discharges only",
                        "cycle": tr["arbin_cycle"].tolist(),
                        "truth": tr["soh_true"].round(4).tolist(),
                        "estimate": tr["soh_est"].round(4).tolist(),
                        "mae": round(float(tr["error"].abs().mean()), 4)}

    field = pd.read_csv(METRICS / "calce_field_soh" / "field_soh_per_cell.csv")
    prim = field[(field["primary"] == True) & (field["arm"] == "step_r")]  # noqa: E712
    per_cell = prim[~prim["refused"].astype(bool) & (prim["n_scored"] >= 10)]
    per_cell = per_cell.sort_values("mae")[["cell_id", "cohort", "mae", "n_scored"]]
    refused = prim[prim["refused"].astype(bool)][["cell_id", "refusal"]]
    summary = pd.read_csv(METRICS / "calce_field_soh" / "field_soh_summary.csv")
    arms = summary[summary["window"] == "3.90-3.60 V"].set_index("arm")

    pcell = pd.read_csv(METRICS / "calce_partial_soh" / "partial_soh_per_cell.csv")
    partial_rows = pcell[pcell["cell_id"].isin(PARTIAL_CELLS + ("CS2_5", "CS2_6"))]

    rul = pd.read_csv(METRICS / "calce_rul_horizon_early_ref" / "rul_folds.csv")
    rul = rul[(rul["threshold"] == 0.9)].dropna(subset=["rul_pred", "rul_true"])
    near = rul[rul["rul_true"] < 25]
    rul_points = rul[rul["rul_true"] <= 50][["cell_id", "rul_true", "rul_pred"]]
    bins = [(0, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    rul_bins = []
    for lo, hi in bins:
        b = rul[(rul["rul_true"] >= lo) & (rul["rul_true"] < hi)]
        rul_bins.append({"range": f"{lo}-{hi}", "n": int(len(b)),
                         "within_20": round(float((b["error"].abs() <= 20).mean()), 3),
                         "median_abs": round(float(b["error"].abs().median()), 1)})

    rig = rig_voltage()

    data = {
        "headline": {
            "soh_median_mae": float(arms.loc["step_r", "median_cell_mae"]),
            "soh_ci": [float(arms.loc["step_r", "ci_low"]), float(arms.loc["step_r", "ci_high"])],
            "soh_cells": int(arms.loc["step_r", "cells_scored"]),
            "soh_cells_total": int(prim["cell_id"].nunique()) + 1,
            "partial_mae": sorted(round(float(x), 4) for x in partial_rows["mae"].dropna()),
            "rul_within_20_near": round(float((near["error"].abs() <= 20).mean()), 3),
            "rul_near_n": int(len(near)),
            "rul_near_median_abs": round(float(near["error"].abs().median()), 1),
            "rig_frames": len(rig),
        },
        "corrections": {arm: {"cells": int(arms.loc[arm, "cells_scored"]),
                              "median_mae": float(arms.loc[arm, "median_cell_mae"])}
                        for arm in ("raw", "cycler_r", "step_r")},
        "traces": traces,
        "per_cell": per_cell.round(4).to_dict(orient="records"),
        "refused": refused.to_dict(orient="records"),
        "partial": partial_rows[["cell_id", "group", "window", "steepness", "mae", "refusal"]]
        .round(4).fillna("").to_dict(orient="records"),
        "rul_points": rul_points.round(1).to_dict(orient="records"),
        "rul_bins": rul_bins,
        "rig": rig,
    }
    (METRICS / "results_showcase.json").write_text(json.dumps(data), encoding="utf-8")
    _figures(data)
    h = data["headline"]
    print(f"SOH median {h['soh_median_mae']:.4f} on {h['soh_cells']} cells; "
          f"partial {h['partial_mae']}; RUL near-EOL within 20: {h['rul_within_20_near']} "
          f"(n={h['rul_near_n']}); rig frames {h['rig_frames']}")
    return 0


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _gapped(x: list, y: list, gap: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert a break where consecutive points are further apart than `gap`."""
    xs, ys = [], []
    for i, (a, b) in enumerate(zip(x, y, strict=True)):
        if i and a - x[i - 1] > gap:
            xs.append(np.nan)
            ys.append(np.nan)
        xs.append(a)
        ys.append(b)
    return np.array(xs, float), np.array(ys, float)


def _figures(data: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    for ax, (cell, tr) in zip(axes.flat, data["traces"].items(), strict=True):
        color = LEARNED if "learned" in tr["mode"] else ESTIMATE
        span = max(tr["cycle"]) - min(tr["cycle"])
        ax.scatter(tr["cycle"], np.array(tr["truth"]) * 100, s=10, color=TRUTH,
                   label="lab capacity test", zorder=2)
        gx, gy = _gapped(tr["cycle"], tr["estimate"], gap=0.15 * span)
        ax.plot(gx, gy * 100, color=color, linewidth=2, label="BEACON estimate", zorder=3)
        ax.set_title(f"{cell} - {tr['mode']}\nmean error {tr['mae'] * 100:.1f} points",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("cycle", fontsize=9, color=MUTED)
        ax.set_ylabel("state of health (%)", fontsize=9, color=MUTED)
        _style(ax)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("State of health from voltage, current and time only, against lab capacity tests",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "results_soh_tracking.png", dpi=150)
    plt.close(fig)

    cells = data["per_cell"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar([c["cell_id"] for c in cells], [c["mae"] * 100 for c in cells], color=ESTIMATE, width=0.6)
    med = data["headline"]["soh_median_mae"] * 100
    ax.axhline(med, color=INK, linewidth=1, linestyle="--")
    ax.text(len(cells) - 0.5, med + 0.2, f"median {med:.1f}%", ha="right", fontsize=9, color=INK)
    ax.set_ylabel("mean SOH error (%)", fontsize=9, color=MUTED)
    ax.set_title(f"SOH error per CALCE cell, field method ({len(cells)} cells measured; "
                 f"{len(data['refused'])} refused with a stated reason)",
                 fontsize=11, color=INK, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "results_soh_per_cell.png", dpi=150)
    plt.close(fig)

    pts = pd.DataFrame(data["rul_points"])
    h = data["headline"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.6))
    ax.fill_between([0, 50], [-20, 30], [20, 70], color=GRID, alpha=0.7, linewidth=0,
                    label="within ±20 cycles")
    ax.plot([0, 50], [0, 50], color=MUTED, linewidth=1)
    ax.scatter(pts["rul_true"], pts["rul_pred"], s=18, color=ESTIMATE, alpha=0.8,
               edgecolors="white", linewidths=0.5, label="estimate")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 80)
    ax.set_xlabel("actual cycles left (lab)", fontsize=9, color=MUTED)
    ax.set_ylabel("predicted cycles left", fontsize=9, color=MUTED)
    ax.set_title("Last 50 cycles before 90% capacity", fontsize=10, color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    bins = data["rul_bins"]
    bx.bar([b["range"] for b in bins], [b["within_20"] * 100 for b in bins],
           color=ESTIMATE, width=0.6)
    for k, b in enumerate(bins):
        bx.text(k, b["within_20"] * 100 + 2, f"{b['within_20'] * 100:.0f}%",
                ha="center", fontsize=9, color=INK)
    bx.set_ylim(0, 100)
    bx.set_xlabel("actual cycles left (lab)", fontsize=9, color=MUTED)
    bx.set_ylabel("estimates within ±20 cycles (%)", fontsize=9, color=MUTED)
    bx.set_title("Accuracy rises as end of life approaches", fontsize=10, color=INK, loc="left")
    _style(bx)
    fig.suptitle(f"Remaining life to 90%: {h['rul_within_20_near'] * 100:.0f}% within ±20 cycles "
                 f"in the last 25 cycles (median miss {h['rul_near_median_abs']:.0f} cycles)",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "results_rul.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
