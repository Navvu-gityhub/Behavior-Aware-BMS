"""Give real cells a memory, piece by piece, for the /memory dashboard.

    python scripts/demo_battery_memory.py
    BEACON_PROFILE_DIR=data/interim/profiles uvicorn src.bms.api.app:app
    # then open http://127.0.0.1:8000/memory

Each cell's log is fed to its BatteryProfile in pieces, the way a BMS
receives telemetry, and the profile is saved after each piece. Cells are
chosen to show the range of what BEACON does, not only where it does well:

  NASA_B0029_43C   warm NASA cell - where the estimator is accurate
  NASA_B0005_24C   room-temperature NASA cell
  NASA_B0047_4C    cold NASA cell - where it is weak (12-14% error class)
  CALCE_CS2_35     the development chemistry
  M50T_25C_D       LG 21700 held-out cell, one ageing set per piece

Uses only data already downloaded (data/raw, data/interim); cells whose data
is missing are skipped with a message. Writes to BEACON_PROFILE_DIR, default
data/interim/profiles/ (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.bms.health.battery_profile import BatteryProfile  # noqa: E402
from src.bms.io.load_nasa_pcoe import load_nasa_pcoe_cell  # noqa: E402

OUT = Path(os.environ.get("BEACON_PROFILE_DIR", "data/interim/profiles"))
REST_A = 0.02
PIECES = 4


def nasa_telemetry(path: Path) -> pd.DataFrame:
    """NASA discharge records laid end to end, each with its own rest head."""
    frames, offset = [], 0.0
    for d in load_nasa_pcoe_cell(path):
        if not (0.3 <= d.capacity_ah <= 2.5):
            continue
        frames.append(pd.DataFrame({"test_time_s": d.time_s - d.time_s[0] + offset, "current_a": d.current_a,
                                    "voltage_v": d.voltage_v, "temperature_c": d.temperature_c}))
        offset = float(frames[-1]["test_time_s"].iloc[-1]) + 600.0
    return pd.concat(frames, ignore_index=True)


def split_at_rest(tel: pd.DataFrame, n: int) -> list[pd.DataFrame]:
    """n pieces, cut just before a discharge's rest head so no discharge is split."""
    i = tel["current_a"].to_numpy(float)
    starts = np.flatnonzero((i[1:] < -REST_A) & (np.abs(i[:-1]) <= REST_A)) + 1
    if len(starts) < n + 1:
        return [tel]
    cuts = [max(int(starts[k]) - 2, 0) for k in np.linspace(0, len(starts) - 1, n + 1).astype(int)[1:-1]]
    bounds = [0, *cuts, len(tel)]
    return [tel.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:], strict=True) if b > a]


def feed(battery_id: str, pieces: list[pd.DataFrame]) -> None:
    profile = BatteryProfile(battery_id, REST_A)
    for k, piece in enumerate(pieces, 1):
        added = profile.update(piece)
        profile.save(OUT / f"{battery_id}.json")
        r = profile.current()
        state = f"{r.soh:.1%}" if r.available else f"refused: {r.refusal[:70]}"
        print(f"  {battery_id} piece {k}/{len(pieces)}: +{added} discharges -> {state}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for cell, label in (("B0029", "43C"), ("B0005", "24C"), ("B0047", "4C")):
        path = Path("data/raw/nasa/mat") / f"{cell}.mat"
        if path.exists():
            feed(f"NASA_{cell}_{label}", split_at_rest(nasa_telemetry(path), PIECES))
        else:
            print(f"skip NASA {cell}: {path} not found (see scripts/run_cross_dataset_study.py)")

    calce = Path("data/interim/calce_telemetry/CS2_35.parquet")
    if calce.exists():
        tel = pd.read_parquet(calce)
        feed("CALCE_CS2_35", split_at_rest(tel[["test_time_s", "current_a", "voltage_v"]], PIECES))
    else:
        print(f"skip CALCE: {calce} not found (built by scripts/run_field_soh_study.py)")

    try:
        from run_m50t_heldout_study import files_by_cell, load_cell
        files = files_by_cell().get((25, "D"), [])
    except ImportError:
        files = []
    if files:
        feed("M50T_25C_D", [load_cell([f]) for f in files])
    else:
        print("skip M50T: no .mpr files (scripts/fetch_imperial_m50t.py)")
    print(f"\nprofiles in {OUT.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
