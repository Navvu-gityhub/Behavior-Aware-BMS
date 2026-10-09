"""Does BEACON stay safe on LFP, a chemistry it was never built for?

    python scripts/run_lfp_safety_study.py

Writes reports/metrics/lfp_safety/. Data: one batch of Severson et al. 2019
(Nature Energy, doi:10.1038/s41560-019-0356-8), A123 APR18650M1A LFP/graphite
cells, 1.1 Ah, fast-charged and discharged at 4C, chamber at 30 C. The batch
2017-06-30 (48 cells, 2.0 GB) is read from data/raw/severson/.

WHY
---
LFP's voltage is almost flat near 3.3 V. BEACON's validated window,
3.90-3.60 V, is not crossed by an LFP discharge, so the shipped path
(telemetry.pipeline._measure_field_soh) falls back to a window learned from
the cell's own usage band. On a flat plateau, a few millivolts of
overpotential growth move such a window's charge a lot, so the risk is not
refusal but a confident wrong number. The project has never tested this.

FIXED BEFORE THE DATA WAS DOWNLOADED
------------------------------------
Method: the shipped decision path, unchanged - pipeline._measure_field_soh
  (fixed window, then the learned window with rated capacity 1.1 Ah), and
  the shipped confidence rules (evidence._capacity_status with the
  validated-conditions temperature flag), run on each cell's telemetry
  truncated at 25, 50, 75 and 100% of its recorded cycles.
Truth: discharge capacity per cycle (the batch's summary QDischarge) over
  the median of the cell's first 5 cycles, at the cycle the reading is for.
Outcome per checkpoint: REFUSED, LOW, or CONFIDENT (MEDIUM or HIGH), and the
  error |reported - truth| when a number is reported.
Criterion (safety, not accuracy):
  among CONFIDENT readings, at most 10% may be off by more than 5 points.
  Refusing everything passes: refusal is the safe outcome on a chemistry
  outside the validated set.
Also reported, not criteria: the share of each outcome, the error
  distribution of LOW and CONFIDENT readings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.evidence import _capacity_status, temperature_flag  # noqa: E402
from src.bms.health.field_soh import monotonic_time  # noqa: E402
from src.bms.telemetry.cycles import cycles_to_frame, measure_cycles  # noqa: E402
from src.bms.telemetry.pipeline import REST_THRESHOLD_A, _measure_field_soh  # noqa: E402

RAW = Path("data/raw/severson/2017-06-30_batchdata_updated_struct_errorcorrect.mat")
OUT = Path("reports/metrics/lfp_safety")
RATED_AH = 1.1
CHECKPOINTS = (0.25, 0.50, 0.75, 1.00)
ERROR_LIMIT = 0.05
MAX_SHARE = 0.10


def load_batch(path: Path):
    """Yield (cell_id, telemetry, truth_by_cycle) per cell; truth is Qd per cycle."""
    import h5py

    with h5py.File(path, "r") as f:
        batch = f["batch"]
        n = batch["summary"].shape[0]
        for k in range(n):
            cycles = f[batch["cycles"][k, 0]]
            summary = f[batch["summary"][k, 0]]
            qd = np.asarray(summary["QDischarge"]).ravel()
            frames, offset = [], 0.0
            for c in range(cycles["I"].shape[0]):
                # minutes -> s; some cycles restart their clock mid-record, joined as
                # the field path joins any logger's restarts (parsing only).
                t = monotonic_time(np.asarray(f[cycles["t"][c, 0]]).ravel() * 60.0)
                if t.size < 10:
                    continue
                i = np.asarray(f[cycles["I"][c, 0]]).ravel() * RATED_AH     # C-rate -> A
                v = np.asarray(f[cycles["V"][c, 0]]).ravel()
                temp = np.asarray(f[cycles["T"][c, 0]]).ravel()
                frames.append(pd.DataFrame({"test_time_s": t - t[0] + offset, "current_a": i,
                                            "voltage_v": v, "temperature_c": temp, "src_cycle": c}))
                offset = float(frames[-1]["test_time_s"].iloc[-1]) + 1.0
            if not frames:
                continue
            tel = pd.concat(frames, ignore_index=True)
            tel = tel[np.isfinite(tel["test_time_s"])].reset_index(drop=True)
            yield f"LFP_{k:02d}", tel, qd


def main() -> int:
    rows = []
    for cell_id, tel, qd in load_batch(RAW):
        ok = np.isfinite(qd) & (qd > 0.5)
        if ok.sum() < 20:
            continue
        ref = float(np.median(qd[ok][:5]))
        n_src = int(tel["src_cycle"].max()) + 1
        for frac in CHECKPOINTS:
            upto = max(int(round(frac * n_src)) - 1, 0)
            part = tel[tel["src_cycle"] <= upto]
            m = measure_cycles(part, cell_id=cell_id, rest_threshold_a=REST_THRESHOLD_A)
            soh = _measure_field_soh(part, m, cell_id, REST_THRESHOLD_A, RATED_AH)
            truth = float(qd[min(upto, len(qd) - 1)] / ref) if upto < len(qd) else np.nan
            d = cycles_to_frame(m, complete_only=False)
            cond = ""
            if soh.available and "avg_temp" in d.columns and len(d):
                cond = temperature_flag(d["avg_temp"].head(5), d["avg_temp"].tail(5))
            status = _capacity_status(soh, bol=False, condition=cond)
            outcome = "REFUSED" if not soh.available else ("LOW" if status.confidence == "LOW" else "CONFIDENT")
            rows.append({"cell_id": cell_id, "checkpoint": frac, "cycle": upto, "outcome": outcome,
                         "window": soh.window, "window_mode": soh.window_mode,
                         "reported": soh.soh if soh.available else np.nan, "truth": truth,
                         "error": abs(soh.soh - truth) if soh.available else np.nan,
                         "reason": (soh.refusal or status.reason)[:160]})
        print(cell_id, [r["outcome"] for r in rows[-len(CHECKPOINTS):]], flush=True)
    r = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    r.round(4).to_csv(OUT / "checkpoints.csv", index=False)
    share = r["outcome"].value_counts(normalize=True).round(3)
    conf = r[r["outcome"] == "CONFIDENT"]
    bad = float((conf["error"] > ERROR_LIMIT).mean()) if len(conf) else 0.0
    verdict = "SAFE" if bad <= MAX_SHARE else "UNSAFE"
    err = r.dropna(subset=["error"]).groupby("outcome")["error"].describe(percentiles=[0.5, 0.9]).round(4)
    lines = ["# BEACON on LFP (Severson 2017-06-30 batch)", "",
             "Generated by `scripts/run_lfp_safety_study.py`; method and criterion fixed before the data "
             "was downloaded (see its docstring).", "",
             f"Cells: {r['cell_id'].nunique()}; checkpoints: {len(r)}.", "",
             "Outcomes:", "", "```", share.to_string(), "```", "",
             "Errors of reported readings:", "", "```", err.to_string() if len(err) else "(none reported)", "```", "",
             "Most common refusal/confidence reasons:", "", "```",
             r["reason"].str[:110].value_counts().head(5).to_string(), "```", "",
             f"- CONFIDENT readings off by > {ERROR_LIMIT:.0%}: {bad:.1%} of {len(conf)} "
             f"(limit {MAX_SHARE:.0%})", "", f"**Verdict: {verdict}**"]
    (OUT / "lfp_safety_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
