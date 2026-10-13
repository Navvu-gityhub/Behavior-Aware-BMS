"""Health read while CHARGING: a pre-registered test.

    python scripts/run_charge_soh_study.py

Writes reports/metrics/charge_soh/. Method: src/bms/health/charge_soh.py,
window 3.75-4.00 V rest-equivalent, shifted up by I*R, reference = first 5
measurable charges. Fixed before any charge curve in this project was looked
at; nothing below is tuned.

DATA
----
Development (reported, decides nothing): CALCE (cached telemetry), NASA
  (all charge and discharge records of each screened cell, in order).
Deciding: the charge curves of
  LG M50T Expt 5 (8 cells, 0.3C constant-current charge to 4.2 V), and
  Oxford Dataset 1 (8 cells, the 1C characterisation charge, C1ch).
  Their DISCHARGE results were seen earlier; their charge curves were not.
  Oxford's characterisation charges start under no rest-to-charge step, so
  there the window is read uncorrected, as its discharge window was.

TRUTH
-----
The capacity of the discharge that follows each charge, over the median of
the cell's first five such capacities:
  M50T    cycler charge of the next discharge, if it is full (>= 4.0 Ah, as in
          run_m50t_heldout_study.py).
  Oxford  the same characterisation's 1C discharge capacity (C1dc).
  CALCE   the next complete discharge (cycles.is_complete).
  NASA    NASA's stored capacity of the next discharge record.

METRICS AND CRITERIA (deciding sets, each separately)
-----------------------------------------------------
  per-cell MAE of each charge's SOH against truth; median over cells.
  1. at least 7 of 8 cells measured;
  2. median per-cell MAE <= 0.03 (the discharge method's validated range:
     1.7% CALCE, 2.3-3.0% on M50T and Oxford).
SUCCESS needs both, on BOTH deciding sets. Also reported: the share of
charges that give a reading, beside the share of discharges that do.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bms.health.charge_soh import CHARGE_WINDOW, charge_readings, charge_soh_table  # noqa: E402
from src.bms.health.voltage_window import window_charge  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402

OUT = Path("reports/metrics/charge_soh")
REST_A = 0.02
MAX_GAP_S = 6 * 3600.0


def score_log(tel: pd.DataFrame, discharges: pd.DataFrame, truth_ok) -> tuple[pd.DataFrame, float]:
    """Per-charge SOH matched to the next qualifying discharge's capacity."""
    table = charge_soh_table(charge_readings(tel, REST_A))
    d = discharges.sort_values("start_time_s")
    d = d[d.apply(truth_ok, axis=1)]
    if table.empty or len(d) < 6:
        return pd.DataFrame(), float("nan")
    ref = float(d["capacity_ah"].head(5).median())
    starts = d["start_time_s"].to_numpy(float)
    caps = d["capacity_ah"].to_numpy(float)
    truth = []
    for end in table["end_time_s"]:
        k = int(np.searchsorted(starts, end))
        truth.append(caps[k] / ref if k < len(starts) and starts[k] - end <= MAX_GAP_S else np.nan)
    table["truth"] = truth
    share = float(table["window_charge_ah"].notna().mean())
    return table, share


def calce() -> list[dict]:
    out = []
    for path in sorted(Path("data/interim/calce_telemetry").glob("*.parquet")):
        tel = pd.read_parquet(path)[["test_time_s", "current_a", "voltage_v"]]
        d = cycles_to_frame(measure_cycles(tel, path.stem, rest_threshold_a=REST_A), complete_only=False)
        table, share = score_log(tel, d, lambda r: bool(r["is_complete"]))
        out.append({"dataset": "CALCE", "cell": path.stem, "table": table, "share": share})
    return out


def nasa() -> list[dict]:
    from scipy.io import loadmat
    screen = pd.read_csv("reports/metrics/benchmark_cell_screen.csv")
    excluded = set(screen.loc[~screen["admissible"].astype(bool), "cell_id"])
    out = []
    for path in sorted(Path("data/raw/nasa/mat").glob("*.mat")):
        if path.stem in excluded:
            continue
        raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)[path.stem]
        frames, rows, offset = [], [], 0.0
        for rec in np.atleast_1d(raw.cycle):
            kind = str(rec.type).strip().lower()
            if kind not in ("charge", "discharge"):
                continue
            t = np.asarray(rec.data.Time, float)
            if t.size < 5:
                continue
            start = offset
            frames.append(pd.DataFrame({"test_time_s": t - t[0] + offset,
                                        "current_a": np.asarray(rec.data.Current_measured, float),
                                        "voltage_v": np.asarray(rec.data.Voltage_measured, float)}))
            offset = float(frames[-1]["test_time_s"].iloc[-1]) + 60.0
            if kind == "discharge":
                cap = np.atleast_1d(getattr(rec.data, "Capacity", np.nan))
                rows.append({"start_time_s": start, "capacity_ah": float(cap[0]) if cap.size else np.nan})
        if not frames:
            continue
        tel = pd.concat(frames, ignore_index=True)
        d = pd.DataFrame(rows)
        table, share = score_log(tel, d, lambda r: 0.3 <= r["capacity_ah"] <= 2.5)
        out.append({"dataset": "NASA", "cell": path.stem, "table": table, "share": share})
    return out


def m50t() -> list[dict]:
    from run_m50t_heldout_study import FULL_AH, files_by_cell, load_cell
    out = []
    for (temp, cell), files in files_by_cell().items():
        tel = load_cell(files)
        d = cycles_to_frame(measure_cycles(tel, cell, rest_threshold_a=REST_A), complete_only=False)
        table, share = score_log(tel, d, lambda r: r["capacity_ah"] >= FULL_AH)
        out.append({"dataset": "M50T", "cell": f"M50T_{temp}C_{cell}", "table": table, "share": share})
        print("M50T", cell, flush=True)
    return out


def oxford() -> list[dict]:
    from src.bms.io.load_oxford import load_oxford_mat, summarize_oxford_cycles
    tel, _ = load_oxford_mat("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
    truth = summarize_oxford_cycles(tel)[["cell_id", "cycle", "capacity_ah"]]
    ch = tel[tel["measurement"] == "C1ch"]
    out = []
    for cell, g in ch.groupby("cell_id"):
        rows = []
        for cyc, b in g.groupby("cycle"):
            q = b["capacity_ah_curve"].abs().to_numpy(float)
            q = q - q.min()
            rows.append({"charge": int(cyc), "window_charge_ah": window_charge(
                b["voltage_v"].to_numpy(float), q, CHARGE_WINDOW)})
        r = pd.DataFrame(rows).sort_values("charge")
        meas = r.dropna(subset=["window_charge_ah"])
        lab = truth[truth["cell_id"] == cell].sort_values("cycle").set_index("cycle")["capacity_ah"]
        if len(meas) < 2 or lab.empty:
            out.append({"dataset": "Oxford", "cell": cell, "table": pd.DataFrame(), "share": float("nan")})
            continue
        ref = float(meas["window_charge_ah"].iloc[0])            # first characterisation, as on discharge
        r["soh"] = r["window_charge_ah"] / ref
        r.loc[(r["soh"] < 0.5) | (r["soh"] > 1.10), "soh"] = np.nan
        r["truth"] = r["charge"].map(lab / lab.iloc[0])
        r = r[r["charge"] != r["charge"].min()]                  # the reference scores zero by construction
        out.append({"dataset": "Oxford", "cell": cell, "table": r,
                    "share": float(r["window_charge_ah"].notna().mean())})
    return out


def summarise(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        t = res["table"]
        ok = t.dropna(subset=["soh", "truth"]) if len(t) else t
        rows.append({"dataset": res["dataset"], "cell": res["cell"], "readings": len(ok),
                     "share_of_charges_read": round(res["share"], 3) if np.isfinite(res["share"]) else np.nan,
                     "mae": float((ok["soh"] - ok["truth"]).abs().mean()) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    per_cell = summarise(calce() + nasa() + m50t() + oxford())
    per_cell.round(5).to_csv(OUT / "per_cell.csv", index=False)
    summ = per_cell.groupby("dataset").agg(
        cells=("cell", "count"), cells_measured=("mae", lambda s: int(s.notna().sum())),
        median_cell_mae=("mae", "median"), median_share_read=("share_of_charges_read", "median")).round(4)
    summ.to_csv(OUT / "summary.csv")
    verdicts = {}
    for ds in ("M50T", "Oxford"):
        if ds not in summ.index:
            verdicts[ds] = "missing"
            continue
        row = summ.loc[ds]
        verdicts[ds] = "met" if (row["cells_measured"] >= 7 and row["median_cell_mae"] <= 0.03) else "not met"
    verdict = "SUCCESS" if all(v == "met" for v in verdicts.values()) else "NOT MET"
    lines = ["# Health read while charging", "",
             "Generated by `scripts/run_charge_soh_study.py`; method, data and criteria fixed before any "
             "charge curve was examined (see its docstring). CALCE and NASA are development; "
             "M50T and Oxford decide.", "",
             "```", summ.to_string(), "```", "",
             *[f"- {ds}: >= 7/8 cells measured and median per-cell MAE <= 3%: {v}" for ds, v in verdicts.items()],
             "", f"**Verdict: {verdict}**"]
    (OUT / "charge_soh_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
