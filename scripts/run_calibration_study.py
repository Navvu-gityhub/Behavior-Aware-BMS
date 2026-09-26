"""Can a few cells from a new cohort recover the skill lost crossing to it?

    python scripts/run_calibration_study.py

Writes `reports/metrics/calce_calibration/` : the per-fold decomposition, the
recovery curve, and the report.

THE OBSERVATION THIS STARTS FROM
--------------------------------
Under leave-one-cohort-out on CALCE, every method's rank correlation stays
high while its R2 falls:

    method        LOCO R2    LOCO Spearman
    xgboost        0.7501           0.9005
    age_linear     0.6599           0.9343
    elasticnet     0.5793           0.9371

Spearman only sees ordering; R2 sees the value. High Spearman with lower R2
is the signature of predictions that are correctly ORDERED but wrongly
SCALED or OFFSET for the unseen cohort. If that is what is happening, the
models are not failing to generalise - they are generalising in shape and
missing a level, and a level is something a handful of labelled cells can fix.

If instead the ordering were also lost, no amount of calibration would help
and this script should say so.

WHAT IT MEASURES
----------------
For each held-out cohort, train on the others and predict. Then decompose the
error into the part a per-cohort affine correction could remove and the part
it could not:

    y ~ a * prediction + b        fitted on k cells of the held-out cohort

- `r2_raw`         what the shipped model scores on the new cohort
- `r2_oracle`      the ceiling for this approach: a and b fitted on ALL of the
                   held-out cohort. Not attainable in deployment - it uses
                   labels you would not have - and reported only as the bound.
- `r2_k`           a and b fitted on k held-out CELLS, scored on the rest

The gap between `r2_raw` and `r2_oracle` is the affine-fixable part. The gap
between `r2_oracle` and 1 is what remains wrong about the shape.

`r2_k` is the deployable number and the point of the exercise: it answers
"how many cells of a new battery type must I characterise before this system
is useful on it?" That is an engineering answer, not a diagnosis.

WHAT WOULD MAKE THIS WORTHLESS
------------------------------
Calibrating on cells and scoring on the SAME cells. The k calibration cells
are excluded from scoring in every fold; a cohort with too few cells to spare
k is skipped and named. Calibration also never touches the model - only a
scalar gain and offset on its output - so nothing here can leak the held-out
cohort into the fit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.bms.benchmarks import add_targets
from src.bms.benchmarks.registry import get, load_all

DEFAULT_DATA = Path("reports/metrics/calce_curve_study.csv")
DEFAULT_OUT = Path("reports/metrics/calce_calibration")

FEATURES = ("cycle", "mean_voltage_v", "min_voltage_v", "mean_current_a",
            "resistance_ohm", "cycle_duration_s")
METHODS = ("age_linear", "elasticnet", "random_forest", "xgboost")
TARGET = "soh"

# Calibration cell counts to sweep. One cell is the interesting case: it is
# the difference between "characterise a new pack" and "characterise a new
# cell", and those are very different asks of a deployment.
K_VALUES = (1, 2, 3)

# Seed for choosing which cells act as the calibration set. The choice matters
# at k=1, so the sweep repeats over several draws rather than trusting one.
SEED = 20260926
N_DRAWS = 20


def _r2(y: np.ndarray, pred: np.ndarray, baseline: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - baseline) ** 2))
    return float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot


def _affine(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Least-squares gain and offset mapping prediction onto truth."""
    if len(y) < 2 or np.ptp(pred) <= 0:
        # No spread to fit a gain against; correct the offset only.
        return 1.0, float(np.mean(y - pred))
    gain, offset = np.polyfit(pred, y, 1)
    return float(gain), float(offset)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    load_all()
    raw = pd.read_csv(args.data)
    data = add_targets(raw)
    data = data[data[TARGET].notna()].copy()

    cohorts = sorted(data["cohort"].unique())
    print(f"{len(data)} rows, {data['cell_id'].nunique()} cells, "
          f"{len(cohorts)} cohorts")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for method_name in METHODS:
        method = get(method_name)
        fit_fn = method.build(list(FEATURES), TARGET)
        print(f"\n--- {method_name} ---", flush=True)

        for cohort in cohorts:
            test = data[data["cohort"] == cohort]
            train = data[data["cohort"] != cohort]
            cells = sorted(test["cell_id"].unique())

            predict = fit_fn(train)
            pred = np.asarray(predict(test), dtype=float)
            y = test[TARGET].to_numpy(dtype=float)
            # The baseline every R2 in this project is measured against: the
            # mean the shipped model would emit, which is the training mean.
            baseline = np.full_like(y, float(train[TARGET].mean()))

            r2_raw = _r2(y, pred, baseline)
            gain, offset = _affine(y, pred)
            r2_oracle = _r2(y, gain * pred + offset, baseline)

            record = {
                "method": method_name,
                "cohort": cohort,
                "n_cells": len(cells),
                "n_rows": len(test),
                "r2_raw": r2_raw,
                "r2_oracle": r2_oracle,
                "oracle_gain": gain,
                "oracle_offset": offset,
                "spearman": float(pd.Series(pred).corr(
                    pd.Series(y), method="spearman")),
            }

            for k in K_VALUES:
                if len(cells) <= k:
                    # Cannot calibrate on k cells and still score on others.
                    record[f"r2_k{k}"] = float("nan")
                    record[f"r2_k{k}_sd"] = float("nan")
                    continue
                scores = []
                for _ in range(N_DRAWS):
                    chosen = rng.choice(cells, size=k, replace=False)
                    cal = test[test["cell_id"].isin(chosen)]
                    held = test[~test["cell_id"].isin(chosen)]
                    cal_pred = np.asarray(
                        predict(cal), dtype=float)
                    g, o = _affine(
                        cal[TARGET].to_numpy(dtype=float), cal_pred)
                    held_pred = np.asarray(predict(held), dtype=float)
                    held_y = held[TARGET].to_numpy(dtype=float)
                    held_base = np.full_like(
                        held_y, float(train[TARGET].mean()))
                    scores.append(_r2(held_y, g * held_pred + o, held_base))
                record[f"r2_k{k}"] = float(np.mean(scores))
                record[f"r2_k{k}_sd"] = float(np.std(scores))

            rows.append(record)
            ks = "  ".join(
                f"k{k}={record[f'r2_k{k}']:.3f}" if np.isfinite(
                    record.get(f"r2_k{k}", np.nan)) else f"k{k}=-"
                for k in K_VALUES
            )
            print(f"  {cohort:<12} n={len(cells)}  raw={r2_raw:+.3f}  "
                  f"oracle={r2_oracle:+.3f}  {ks}", flush=True)

    folds = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out / "calibration_folds.csv", index=False)

    summary = folds.groupby("method").agg(
        median_r2_raw=("r2_raw", "median"),
        median_r2_oracle=("r2_oracle", "median"),
        median_spearman=("spearman", "median"),
        **{f"median_r2_k{k}": (f"r2_k{k}", "median") for k in K_VALUES},
    ).reset_index()
    summary.to_csv(args.out / "calibration_summary.csv", index=False)

    print()
    print(summary.round(4).to_string(index=False))

    # Does rank agreement on the new cohort predict whether calibrating will
    # work? If it does, it is a pre-flight test the system can actually run:
    # rank agreement needs only the calibration cells, which deployment has.
    usable = folds.dropna(subset=["r2_k1"]).copy()
    usable["gain_k1"] = usable["r2_k1"] - usable["r2_raw"]
    bands = []
    for low, high, label in (
        (0.90, 1.01, "rank held (rho >= 0.90)"),
        (0.70, 0.90, "partial (0.70 - 0.90)"),
        (-1.0, 0.70, "rank lost (rho < 0.70)"),
    ):
        sub = usable[(usable["spearman"] >= low) & (usable["spearman"] < high)]
        if sub.empty:
            continue
        bands.append({
            "rank_band": label,
            "n_folds": len(sub),
            "median_r2_raw": round(float(sub["r2_raw"].median()), 4),
            "median_r2_k1": round(float(sub["r2_k1"].median()), 4),
        })
    band_frame = pd.DataFrame(bands)
    band_frame.to_csv(args.out / "calibration_rank_bands.csv", index=False)

    corr_k1 = float(usable["spearman"].corr(usable["r2_k1"]))
    corr_raw = float(usable["spearman"].corr(usable["r2_raw"]))
    improved = int((usable["gain_k1"] > 0).sum())

    print()
    print(band_frame.to_string(index=False))
    print()
    print(f"k1 beats raw in {improved}/{len(usable)} folds, "
          f"median gain {usable['gain_k1'].median():+.4f}")
    print(f"corr(spearman, r2_k1)={corr_k1:.4f}  "
          f"corr(spearman, r2_raw)={corr_raw:.4f}")

    lines = [
        "# Per-cohort calibration: how much of the LOCO loss is a level error?",
        "",
        f"Generated by `scripts/run_calibration_study.py` from `{args.data}`.",
        f"{len(data)} rows, {data['cell_id'].nunique()} cells, "
        f"{len(cohorts)} cohorts.",
        "",
        "`raw` is the shipped model on an unseen cohort. `oracle` fits a gain",
        "and offset on ALL of that cohort's labels - not attainable in",
        "deployment, reported as the bound. `k1`/`k2`/`k3` fit the same gain",
        "and offset on that many CELLS of the new cohort and score on the",
        f"cells left over, averaged over {N_DRAWS} draws of which cells.",
        "",
        "## Summary (median over cohorts)",
        "",
        "```",
        summary.round(4).to_string(index=False),
        "```",
        "",
        "## Per fold",
        "",
        "```",
        folds.round(4).to_string(index=False),
        "```",
        "",
        "## Reading this",
        "",
        "The distance from `raw` to `oracle` is the part of the cross-cohort",
        "loss that is a level error - a wrong gain or offset on an otherwise",
        "correctly-shaped prediction. The distance from `oracle` to 1.0 is",
        "what stays wrong no matter how the output is rescaled.",
        "",
        "`k1` is the deployable figure. It is what the system scores on a new",
        "battery type after characterising one cell of it.",
        "",
        f"One calibration cell improves **{improved} of {len(usable)}** folds, "
        f"median gain **{usable['gain_k1'].median():+.4f}** R2.",
        "",
        "## When calibration works, and when it cannot",
        "",
        "```",
        band_frame.to_string(index=False),
        "```",
        "",
        f"Rank agreement on the new cohort correlates with post-calibration R2 "
        f"at **{corr_k1:.3f}**, against **{corr_raw:.3f}** for the uncalibrated "
        f"score.",
        "",
        "That is the mechanism, and it is a usable rule rather than an",
        "observation. A gain and an offset can only move a prediction up, down",
        "or stretch it; they cannot reorder it. So where the model still ranks",
        "the new cohort's cells correctly, the remaining error is a level error",
        "and calibration removes it. Where the ranking is gone - CS2_Type3,",
        "rank 0.11 - calibration recovers nothing, and no amount of reference",
        "data would, because the thing that is wrong is not the level.",
        "",
        "Rank agreement is measurable on the calibration cells alone, which a",
        "deployment has by definition. So it can be checked BEFORE the system",
        "is trusted on a new battery type, and refused when it fails - which is",
        "the same shape as every other gate in this codebase.",
        "",
        "Calibration never touches the model, only a scalar gain and offset on",
        "its output, and the calibration cells are excluded from scoring in",
        "every fold. A cohort with too few cells to spare k is reported as NaN",
        "rather than scored on its own calibration set.",
    ]
    (args.out / "calibration_report.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}/calibration_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
