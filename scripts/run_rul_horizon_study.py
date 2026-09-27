"""Does RUL estimation meet the poster's "within +/- 20 cycles"?

    python scripts/run_rul_horizon_study.py

Writes `reports/metrics/calce_rul_horizon/`.

THE CLAIM BEING TESTED
----------------------
The project poster promises "RUL Estimation Accuracy Within +/- 20 Cycles".
That is a falsifiable number and it had never been measured, so this measures
it, against capacity-fade ground truth from CALCE.

The answer is conditional, and the condition is the useful part: accuracy
depends on how far ahead the estimate reaches. One number for "RUL accuracy"
would average a good result near end of life together with a hopeless one far
from it, and the average would describe neither.

WHY THRESHOLD 0.90 AND NOT THE CONVENTIONAL 0.80
--------------------------------------------------
Automotive retirement is conventionally at 80% state of health. It cannot be
validated here: CALCE's full-discharge trajectories end at roughly 0.81, and
exactly one cell of 22 ever crosses 0.80. Scoring predictions of a crossing
the data does not contain would be measuring an extrapolation against another
extrapolation.

0.90 is crossed by 20 of 22 cells with substantial life recorded afterwards,
so predictions of it are scored against an observed crossing. 0.85 is reported
alongside as the longer-horizon case. Neither is 0.80, and no number here may
be quoted as an 80% end-of-life result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.bms.benchmarks import add_targets
from src.bms.rul.fade_extrapolation import DEFAULT_SMOOTH, rul_error_table

DEFAULT_DATA = Path("reports/metrics/calce_full_discharge.csv")
DEFAULT_OUT = Path("reports/metrics/calce_rul_horizon")

THRESHOLDS = (0.90, 0.85)
HORIZON_BINS = ((0, 25), (25, 50), (50, 100), (100, 200), (200, 400),
                (400, 10_000))
TARGET_TOLERANCE = 20


def _bin_table(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for low, high in HORIZON_BINS:
        block = scored[(scored["rul_true"] >= low) & (scored["rul_true"] < high)]
        if block.empty:
            continue
        err = block["error"]
        rows.append({
            "true_rul": f"{low}-{high}" if high < 10_000 else f"{low}+",
            "n": len(block),
            "median_abs_error": round(float(err.abs().median()), 1),
            "mae": round(float(err.abs().mean()), 1),
            f"within_{TARGET_TOLERANCE}": round(
                float((err.abs() <= TARGET_TOLERANCE).mean()), 3),
            "median_bias": round(float(err.median()), 1),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stride", type=int, default=10)
    args = parser.parse_args()

    frame = add_targets(pd.read_csv(args.data))
    frame = frame[frame["soh"].notna()][
        ["cell_id", "cycle", "soh", "cohort"]].copy()
    print(f"{len(frame)} rows, {frame['cell_id'].nunique()} cells, "
          f"{frame['cohort'].nunique()} cohorts")

    args.out.mkdir(parents=True, exist_ok=True)
    all_folds: list[pd.DataFrame] = []
    sections: list[str] = []

    for threshold in THRESHOLDS:
        table = rul_error_table(
            frame, threshold=threshold, stride=args.stride,
            smooth=DEFAULT_SMOOTH,
        )
        table["threshold"] = threshold
        all_folds.append(table)

        scored = table.dropna(subset=["error"])
        never = table[table["refusal"].astype(str).str.startswith("never")]
        bins = _bin_table(scored)

        print(f"\n--- threshold {threshold} ---")
        print(f"{len(scored)} scored estimates, "
              f"{scored['cell_id'].nunique()} cells, "
              f"{len(never)} cells never cross")
        print(bins.to_string(index=False))

        near = scored[scored["rul_true"] < 25]
        hit = float((near["error"].abs() <= TARGET_TOLERANCE).mean()) \
            if len(near) else float("nan")

        sections += [
            f"## Threshold SOH {threshold}",
            "",
            f"{len(scored)} scored estimates across "
            f"{scored['cell_id'].nunique()} cells. "
            f"{len(never)} cell(s) never cross and are excluded rather than "
            f"counted as correct.",
            "",
            "```",
            bins.to_string(index=False),
            "```",
            "",
            f"Within 25 cycles of end of life the estimate lands inside "
            f"+/-{TARGET_TOLERANCE} cycles **{hit:.1%}** of the time.",
            "",
        ]

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(args.out / "rul_folds.csv", index=False)

    primary = folds[folds["threshold"] == 0.90].dropna(subset=["error"])
    near = primary[primary["rul_true"] < 25]
    far = primary[primary["rul_true"] >= 200]
    hit_near = float((near["error"].abs() <= TARGET_TOLERANCE).mean())
    hit_far = float((far["error"].abs() <= TARGET_TOLERANCE).mean())

    lines = [
        "# RUL accuracy against horizon",
        "",
        f"Generated by `scripts/run_rul_horizon_study.py` from `{args.data}`.",
        "",
        "## The poster's claim, measured",
        "",
        'The poster promises "RUL Estimation Accuracy Within +/- 20 Cycles". '
        "That",
        "is true near end of life and false far from it:",
        "",
        f"- within 25 cycles of end of life: **{hit_near:.1%}** of estimates "
        f"land inside +/-20 cycles",
        f"- 200 or more cycles out: **{hit_far:.1%}**",
        "",
        "So the defensible form of the claim carries its condition:",
        "**within +/-20 cycles when the cell is within ~25 cycles of end of",
        "life**. Stated bare, it is not supported.",
        "",
        "## Bias direction depends on the threshold, and that is worth saying",
        "",
        "At threshold 0.90 the median bias is negative at every horizon: the",
        "estimate places end of life earlier than it occurs. That is the safe",
        "direction - an estimate that retires a cell early costs money, one",
        "that retires it late is a failure in service - and it is not tuned",
        "away, because removing it would trade the safe error for the unsafe",
        "one.",
        "",
        "At threshold 0.85 it does NOT hold. Near end of life the bias turns",
        "positive (+17.6 cycles at RUL under 25), so the estimate runs late",
        "there. The conservative behaviour is therefore a property of the 0.90",
        "threshold on these cells and not a general guarantee of the method,",
        "and it must not be quoted as one.",
        "",
        "## What is NOT shown here",
        "",
        "The conventional 80% end-of-life threshold. CALCE's full-discharge",
        "trajectories end near 0.81 and one cell of 22 crosses 0.80, so a",
        "prediction of that crossing would be scored against another",
        "extrapolation rather than an observation. Nothing here may be quoted",
        "as an 80% end-of-life result.",
        "",
        *sections,
        "## Method",
        "",
        "At cycle n, the SOH history of that cell up to n is smoothed with a",
        f"{DEFAULT_SMOOTH}-cycle rolling median, a line is fitted, and the",
        "crossing of the threshold is solved for. RUL is the distance from n",
        "to the crossing. No other cell's data enters, so there is no training",
        "cohort and no cross-protocol transfer step.",
        "",
        "The line, a square root and a quadratic were compared at full history",
        "and a 200-cycle trailing window; the line on full history won. See",
        "`src/bms/rul/fade_extrapolation.py` for that table.",
    ]
    (args.out / "rul_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}/rul_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
