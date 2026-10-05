"""Battery health report card from real telemetry, checked against the lab.

    python scripts/health_report.py --calce data/raw/calce/CS2/Type2/CS2_35.zip --upto-cycle 300
    python scripts/health_report.py --serial data/interim/rig_stage_b_voltage_verified.txt

`--calce` runs a CALCE cell's raw cycler log through the same pipeline a vehicle
log would take - voltage, current and time only, nothing the cycler computed -
and prints the card as it would have read at `--upto-cycle`. Because CALCE
kept cycling past that point, the script then prints what actually happened:
the capacity the cycler measured at that cycle, and the cycle where the cell
really crossed the end-of-life threshold. The card is scored against a future
it was not shown.

`--serial` replays a bench-rig capture. Every capture to date is a cell at
rest, so expect the card to refuse - that is the correct output for a log with
no discharge in it.

CALCE is not redistributable and is not in a fresh clone; see
docs/reproducing.md for where to download it.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.guardian.health_card import render_health_card  # noqa: E402
from src.bms.health.field_soh import monotonic_time  # noqa: E402
from src.bms.rul.fade_extrapolation import (  # noqa: E402
    DEFAULT_EOL_THRESHOLD,
    observed_eol,
)
from src.bms.telemetry.pipeline import score_telemetry_frame  # noqa: E402
from src.bms.telemetry.sources import REQUIRED_CHANNELS, SignalCoverage  # noqa: E402

TRUTH = Path("reports/metrics/calce_full_discharge.csv")

# Arbin logs exactly zero current at rest; the vehicle default of 0.5 A would
# read CALCE's 0.2C discharges as rest. See run_field_soh_study.py.
CALCE_REST_THRESHOLD_A = 0.02

# Nominal capacities from the CALCE cell specifications. Passed as the rated
# capacity, which is what lets the pipeline fall back to a learned window on
# a cell that never discharges across the standard one.
CALCE_RATED_AH = {"CS2": 1.1, "CX2": 1.35}

# Written by scripts/run_field_soh_study.py; reading an archive takes minutes.
CALCE_CACHE = Path("data/interim/calce_telemetry")


def _coverage(telemetry: pd.DataFrame, label: str) -> SignalCoverage:
    present = tuple(c for c in REQUIRED_CHANNELS if c in telemetry.columns)
    missing = tuple(c for c in REQUIRED_CHANNELS if c not in telemetry.columns)
    return SignalCoverage(dbc_path=label, available_signals=present,
                          mapped_channels=present, missing_channels=missing,
                          transport="dataset")


def _calce(path: Path, upto: int | None) -> int:
    from src.bms.io.load_calce_cycling import load_calce_archive

    cached = CALCE_CACHE / f"{path.stem}.parquet"
    if cached.exists():
        telemetry = pd.read_parquet(cached)
        cell = path.stem
    else:
        print(f"Loading {path.name} (a full CALCE archive takes a minute or two)...",
              flush=True)
        telemetry, report = load_calce_archive(path)
        telemetry["test_time_s"] = monotonic_time(telemetry["test_time_s"])
        cell = report.cell_id
    if upto is not None:
        telemetry = telemetry[pd.to_numeric(telemetry["cycle"], errors="coerce") <= upto]
    # The cycler's own capacity and resistance columns are dropped, so nothing
    # a vehicle BMS would not have can reach the estimate.
    telemetry = telemetry[[c for c in ("cell_id", "test_time_s", "current_a",
                                       "voltage_v", "cycle") if c in telemetry.columns]]

    result = score_telemetry_frame(
        source_name=f"calce:{cell}", telemetry=telemetry,
        coverage=_coverage(telemetry, path.name), n_frames=len(telemetry),
        n_decoded=len(telemetry), cell_id=cell,
        rated_capacity_ah=CALCE_RATED_AH.get(cell[:3]),
        rest_threshold_a=CALCE_REST_THRESHOLD_A,
        # CALCE cycled these cells from new, so the log's first discharges
        # are beginning of life. A replayed vehicle log would not say this.
        reference_is_beginning_of_life=True,
    )
    print()
    print(render_health_card(result, battery_label=f"{cell} (CALCE)"))
    _check_against_lab(cell, upto, result)
    return 0


def _check_against_lab(cell: str, upto: int | None, result) -> None:
    if not TRUTH.exists():
        return
    from src.bms.benchmarks import add_targets

    truth = add_targets(pd.read_csv(TRUTH))
    truth = truth[(truth["cell_id"] == cell) & truth["soh"].notna()].sort_values("arbin_cycle_index")
    if truth.empty:
        return
    # "As new" is the capacity checks within the cell's first 10 cycles - the
    # first five on a normally cycled cell, only the first on a
    # partial-cycling cell whose checks are ~100 cycles apart (its first five
    # would already include several percent of fade). add_targets' own `soh`
    # divides by a lifetime percentile, which uses the future.
    early = truth[truth["arbin_cycle_index"] <= 10]
    reference = early if not early.empty else truth.head(1)
    truth["soh"] = truth["capacity_ah"] / reference["capacity_ah"].median()
    print()
    print("CHECK AGAINST THE LAB  (the card above was not shown any of this)")
    print("-" * 60)
    upto_c = upto if upto is not None else int(truth["arbin_cycle_index"].max())
    seen = truth[truth["arbin_cycle_index"] <= upto_c]
    soh = result.field_soh
    if not seen.empty:
        recent = seen[seen["arbin_cycle_index"] > upto_c - 10]
        lab = float((recent if not recent.empty else seen.tail(1))["soh"].median())
        line = f"   Cycler-measured capacity at cycle {upto_c}: {lab:.1%} of initial"
        if soh is not None and soh.available:
            line += f"; the card said {soh.soh:.1%} ({(soh.soh - lab) * 100:+.1f} points)"
        print(line)
    eol = observed_eol(truth["arbin_cycle_index"].to_numpy(float),
                       truth["soh"].to_numpy(float), DEFAULT_EOL_THRESHOLD)
    rul = result.rul_estimate
    if math.isfinite(eol):
        true_rul = max(eol - upto_c, 0.0)
        line = (f"   The cell actually crossed {DEFAULT_EOL_THRESHOLD:.0%} at cycle "
                f"{eol:.0f}: {true_rul:.0f} cycles after cycle {upto_c}")
        if rul is not None and np.isfinite(rul.rul_cycles):
            # rul_cycles counts discharges the pipeline segmented, not Arbin
            # cycle indices. On CALCE they agree to within the handful of
            # non-discharge cycles, which is noted rather than corrected.
            line += f"; the card said {rul.rul_cycles:.0f}"
        print(line)
    else:
        print(f"   The cell never crossed {DEFAULT_EOL_THRESHOLD:.0%} in the lab record.")


def _serial(path: Path) -> int:
    from src.bms.telemetry.serial_pipeline import replay_serial_capture

    result = replay_serial_capture(path, require_full_coverage=False)
    print(render_health_card(result, battery_label=f"bench rig ({path.name})"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--calce", type=Path, help="a CALCE cell archive (.zip)")
    source.add_argument("--serial", type=Path, help="a bench-rig serial capture")
    parser.add_argument("--upto-cycle", type=int, default=None,
                        help="report as the card would have read at this cycle")
    args = parser.parse_args()
    if args.calce:
        return _calce(args.calce, args.upto_cycle)
    return _serial(args.serial)


if __name__ == "__main__":
    raise SystemExit(main())
