"""Join curve-derived features onto the CALCE cycle-level frame.

    python scripts/build_calce_curve_study_frame.py

Writes `reports/metrics/calce_curve_study.csv`: the full-discharge frame
restricted to cells that have curve features, with those features attached as
per-cell constant columns.

THE EXPERIMENT THIS FRAME EXISTS FOR
------------------------------------
The benchmark study's durable finding is that every method loses skill under
leave-one-cohort-out. Every method it could run was fitted on *usage
aggregates* -- voltage and current summaries, internal resistance, cycle
duration. Those describe how a cell was driven and carry no information about
electrode state, which is a candidate explanation for the collapse.

So: run the same gate twice on the same rows, changing only the feature set.

    arm A   CALCE_FEATURES                      (usage aggregates)
    arm B   CALCE_FEATURES + curve features     (electrochemical)

If LOCO skill survives in arm B where it collapsed in arm A, the collapse was
attributable to the features. If both collapse identically, the finding is
stronger than before, because it can no longer be blamed on weak features.

BOTH ARMS MUST RUN ON THE SAME CELLS
------------------------------------
Four of the 22 full-discharge cells have no curve features -- CS2_5, CS2_6,
CX2_3 and CX2_8 lack cycle 10 or 100, or yielded too few resolvable
discharges. This script drops them from BOTH arms rather than letting arm A
keep them.

That costs rows, and it is not optional. Leaving them in arm A would confound
the feature set with the cell set, and a difference in LOCO R2 could then be
read as "curve features help" when it meant "arm A was also asked to
generalise to four extra cells". Dropping CX2_Type3, CX2_Type4 and CS2_Type5
entirely also takes the cohort count from 10 to 7, which is reported rather
than absorbed.

THE CURVE COLUMNS ARE PER-CELL CONSTANTS
----------------------------------------
`delta_q_variance` is computed once per cell from cycles 10 and 100; it does
not vary along a cell's life. Broadcasting it across that cell's ~550 cycle
rows does not create 550 independent observations of it, and the effective
sample size for these four columns is 18, not 9,951.

Leave-one-cohort-out is what keeps that honest: a per-cell constant cannot be
memorised across the fold boundary, because the held-out cohort's cells never
appear in training. A random k-fold split over rows would let the model look
up the held-out cell's own constant from its other rows and report a score
that means nothing -- which is the same failure mode, at row level, that this
project documented at cell level.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CYCLES = Path("reports/metrics/calce_full_discharge.csv")
DEFAULT_FEATURES = Path("reports/metrics/calce_curve_features.csv")
DEFAULT_OUT = Path("reports/metrics/calce_curve_study.csv")

CURVE_FEATURE_COLUMNS: tuple[str, ...] = (
    "delta_q_variance",
    "ica_peak_height",
    "ica_peak_voltage",
    "ica_area",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=Path, default=DEFAULT_CYCLES)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    for path in (args.cycles, args.features):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Build the curve features first with "
                f"scripts/build_calce_curve_features.py."
            )

    cycles = pd.read_csv(args.cycles)
    features = pd.read_csv(args.features)

    # A refused cell carries its reason and NaN features. Reading it back from
    # CSV turns the empty reason into NaN, so test on the feature itself --
    # which is what the join actually needs -- rather than on the reason text.
    usable = features[features["delta_q_variance"].notna()].copy()
    refused = sorted(set(features["cell_id"]) - set(usable["cell_id"]))

    keep = set(usable["cell_id"])
    missing = sorted(set(cycles["cell_id"]) - keep)

    joined = cycles[cycles["cell_id"].isin(keep)].merge(
        usable[["cell_id", *CURVE_FEATURE_COLUMNS]], on="cell_id", how="inner"
    )

    if joined.empty:
        raise SystemExit(
            "no rows survived the join; no cell in the cycle frame has curve "
            "features"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(args.out, index=False)

    print(f"cycle frame   : {len(cycles):>6} rows, "
          f"{cycles['cell_id'].nunique()} cells, "
          f"{cycles['cohort'].nunique()} cohorts")
    print(f"curve features: {len(features)} cells, {len(usable)} usable")
    if refused:
        print(f"  refused (no delta_q_variance): {', '.join(refused)}")
    print(f"joined        : {len(joined):>6} rows, "
          f"{joined['cell_id'].nunique()} cells, "
          f"{joined['cohort'].nunique()} cohorts")
    if missing:
        print(f"  dropped from BOTH arms: {', '.join(missing)}")
    print()
    print("cells per cohort:")
    print(joined.groupby("cohort")["cell_id"].nunique().to_string())
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
