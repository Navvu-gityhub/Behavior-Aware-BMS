"""CALCE CS2/CX2 cycling data: multi-file Arbin ingestion.

`load_calce.py` reads one Excel file. `load_calce_capacity.py` reads the
single-cycle capacity-characterisation workbooks, which ADR 0001 records as
unusable for fade modelling. Neither reads the cycling data, which is the part
that carries a fade trajectory and therefore the part BEACON needs.

Why this cannot reuse `load_nasa_dataset`
-----------------------------------------
NASA ships one `metadata.csv` indexing every test, and cycle numbers are derived
by ranking discharge tests per battery. CALCE ships **many files per cell**,
named by recording date, and each file's `Cycle_Index` restarts at 1.

Concatenating them naively produces forty rows labelled "cycle 1", which the
cycle-level feature layer would collapse into a single cycle spanning the cell's
whole life. That is not a hypothetical: it is the default outcome of the obvious
implementation, and it fails silently because the resulting frame is
structurally valid.

`_reconcile_cycle_index` offsets each file's indices by the running maximum, so
cycle numbers increase monotonically across a cell's recording history. Files
are ordered by the date embedded in their filename, falling back to filesystem
mtime, because lexical ordering puts `10_20_2011` before `9_20_2011`.

What this loader does not invent
--------------------------------
CS2 and CX2 Arbin exports carry seventeen columns and **none of them is
temperature**. The cells were cycled at room temperature, around 23 C, and the
ambient was a property of the room rather than a recorded channel.

So `temperature_c` is absent from the output rather than filled with 23.0. A
constant stand-in would satisfy the schema, flow into `high_temp_flag` and
`temp_rolling_mean`, and produce a thermal stress score for a quantity nobody
measured. The NaN-as-healthy fix established the principle: absent data must
stay absent and fail loudly downstream, not acquire a plausible default.

The one exception is CX2_4, which was cycled across 25, 35, 45 and 55 C with
separate thermocouple files. `load_calce_cell` accepts a `temperature_dir` for
that case, and only that case.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.bms.preprocessing.schema import build_rename_map

# Arbin column names the shared alias table does not already cover.
#
# `build_rename_map` already resolves Current(A), Voltage(V), Cycle_Index and
# Date_Time. It does NOT resolve Discharge_Capacity(Ah), which is the fade
# target, so without these the loader would silently produce frames with no
# capacity column.
CALCE_ARBIN_ALIASES: Mapping[str, str] = {
    "Discharge_Capacity(Ah)": "capacity_ah",
    "Charge_Capacity(Ah)": "charge_capacity_ah",
    "Discharge_Energy(Wh)": "discharge_energy_wh",
    "Charge_Energy(Wh)": "charge_energy_wh",
    "Test_Time(s)": "test_time_s",
    "Step_Time(s)": "step_time_s",
    "Step_Index": "step_index",
    "Internal_Resistance(Ohm)": "resistance_ohm",
    "AC_Impedance(Ohm)": "impedance_ohm",
    "Data_Point": "data_point",
}

# Channels a CS2/CX2 Arbin export never contains. Recorded so the loader can
# state what is missing rather than leaving a downstream consumer to discover it
# as a KeyError.
CALCE_UNAVAILABLE_CHANNELS: tuple[str, ...] = (
    "temperature_c",
    "soc",
)

# Filenames look like CS2_33_10_04_10.xlsx: cell id, then month_day_year.
#
# The date must be anchored to the END of the stem. An unanchored pattern
# consumes the cell number instead: 'CS2_33_10_04_10' matches as month=2,
# day=33, year=2010, which is not a date and orders files arbitrarily. On real
# archives that silently scrambles a cell's recording sequence and produces a
# fade trajectory in the wrong order -- structurally valid and completely wrong.
_DATE_IN_NAME = re.compile(r"(\d{1,2})[_-](\d{1,2})[_-](\d{2,4})$")
_CELL_IN_NAME = re.compile(r"((?:CS2|CX2|CS_2|CX_2)[_-]?\d+)", re.IGNORECASE)

_READERS = {
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".csv": pd.read_csv,
    ".txt": lambda p: pd.read_csv(p, sep="\t"),
}


@dataclass(frozen=True)
class CalceLoadReport:
    """What a load produced, and what it could not."""

    cell_id: str
    n_files: int
    n_rows: int
    n_cycles: int
    files_used: tuple[str, ...] = ()
    files_skipped: tuple[tuple[str, str], ...] = ()
    unavailable_channels: tuple[str, ...] = ()
    has_temperature: bool = False

    def render(self) -> str:
        lines = [
            f"{self.cell_id}: {self.n_rows:,} rows across {self.n_cycles} cycles "
            f"from {self.n_files} file(s)"
        ]
        if self.unavailable_channels:
            lines.append(
                f"  not recorded by this dataset: {list(self.unavailable_channels)}"
            )
        for name, reason in self.files_skipped:
            lines.append(f"  skipped {name}: {reason}")
        return "\n".join(lines)


def _sort_key(path: Path) -> tuple:
    """Order a cell's files chronologically.

    Lexical ordering is wrong here: '10_04_10' sorts before '9_20_10' as text
    while being three months later. The date in the filename is parsed when
    present, and mtime is the fallback for files that do not carry one.
    """
    match = _DATE_IN_NAME.search(path.stem)
    if match:
        month, day, year = (int(g) for g in match.groups())
        if year < 100:
            year += 2000
        return (0, year, month, day, path.name)
    return (1, path.stat().st_mtime, 0, 0, path.name)


def _infer_cell_id(path: Path) -> str:
    match = _CELL_IN_NAME.search(path.stem)
    return match.group(1).upper().replace("-", "_") if match else path.stem


def _read_one(path: Path) -> pd.DataFrame:
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f"unsupported extension '{path.suffix}'")
    frame = reader(path)
    if frame.empty:
        raise ValueError("file contains no rows")

    # Two renames, deliberately separate.
    #
    # `build_rename_map` resolves aliases and then keeps only names in
    # CANONICAL_COLUMNS. That filter is correct: the unified schema is a shared
    # contract, and widening it so CALCE can carry `step_index` would push an
    # Arbin implementation detail into every other dataset's frames.
    #
    # But CALCE genuinely needs some of those columns. `test_time_s` is how
    # cycle duration and the CX2_4 thermocouple join are computed, and neither
    # is expressible without it. So canonical names go through the shared map,
    # and the rest are renamed here, in the loader that actually needs them.
    canonical = build_rename_map(frame.columns, extra_aliases=CALCE_ARBIN_ALIASES)
    extra = {
        column: CALCE_ARBIN_ALIASES[column]
        for column in frame.columns
        if column in CALCE_ARBIN_ALIASES and column not in canonical
    }
    return frame.rename(columns={**canonical, **extra})


def _reconcile_cycle_index(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a cell's files with monotonically increasing cycle numbers.

    Each CALCE file restarts `Cycle_Index` at 1, so the naive concatenation
    labels forty separate cycles as cycle 1. The cycle-level feature layer would
    then collapse them into one cycle spanning the cell's entire life — a
    structurally valid frame carrying a completely wrong trajectory, which is
    why this is done here rather than left to the caller.
    """
    reconciled: list[pd.DataFrame] = []
    offset = 0

    for frame in frames:
        block = frame.copy()
        if "cycle" not in block.columns:
            # A file with no cycle index contributes one cycle, positioned after
            # everything already read.
            block["cycle"] = offset + 1
            offset += 1
            reconciled.append(block)
            continue

        cycles = pd.to_numeric(block["cycle"], errors="coerce")
        if cycles.notna().sum() == 0:
            block["cycle"] = offset + 1
            offset += 1
            reconciled.append(block)
            continue

        # Rebase to 1 within the file before offsetting, so a file whose indices
        # start at 0 (Arbin sometimes emits a pre-cycle row) does not collide
        # with the previous file's last cycle.
        block["cycle"] = cycles - cycles.min() + 1 + offset
        offset = int(block["cycle"].max())
        reconciled.append(block)

    return pd.concat(reconciled, ignore_index=True)


def _attach_temperature(
    telemetry: pd.DataFrame, temperature_dir: Path, cell_id: str
) -> pd.DataFrame:
    """Join thermocouple data for CX2_4, the one thermally-varied CALCE cell.

    Joined on test time by nearest match, because the thermocouple logger and
    the cycler are separate instruments sampling on their own clocks. An exact
    join would drop nearly every row.
    """
    files = sorted(
        (p for p in temperature_dir.iterdir()
         if p.suffix.lower() in _READERS and cell_id.lower() in p.stem.lower()),
        key=_sort_key,
    )
    if not files:
        return telemetry

    blocks = []
    for path in files:
        try:
            blocks.append(_read_one(path))
        except Exception:
            continue
    if not blocks:
        return telemetry

    thermal = pd.concat(blocks, ignore_index=True)
    # The thermocouple export names its column "Temperature (C)", which the
    # shared alias table does resolve to `temperature_c`. Accept either the
    # canonical name or any remaining temperature-ish column.
    temp_col = (
        "temperature_c" if "temperature_c" in thermal.columns
        else next((c for c in thermal.columns if "temp" in str(c).lower()), None)
    )
    if temp_col is None or "test_time_s" not in thermal.columns:
        return telemetry
    if "test_time_s" not in telemetry.columns:
        return telemetry

    thermal = thermal[["test_time_s", temp_col]].rename(
        columns={temp_col: "temperature_c"}
    )
    thermal["test_time_s"] = pd.to_numeric(thermal["test_time_s"], errors="coerce")
    thermal = thermal.dropna(subset=["test_time_s"]).sort_values("test_time_s")

    merged = telemetry.sort_values("test_time_s")
    return pd.merge_asof(
        merged, thermal, on="test_time_s", direction="nearest",
    ).reset_index(drop=True)


def load_calce_cell(
    cell_dir: str | Path,
    cell_id: str | None = None,
    temperature_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, CalceLoadReport]:
    """Load every cycling file for one CALCE cell into one telemetry frame.

    Returns the frame and a report naming the files used, the files skipped with
    reasons, and the channels this dataset does not record. The report is
    returned rather than logged because "CS2 has no temperature channel" is a
    fact a caller must act on, not a diagnostic to discard.
    """
    cell_dir = Path(cell_dir)
    if not cell_dir.exists():
        raise FileNotFoundError(f"load_calce_cell: no such directory: {cell_dir}")

    candidates = sorted(
        (p for p in cell_dir.iterdir() if p.suffix.lower() in _READERS),
        key=_sort_key,
    )
    if not candidates:
        raise FileNotFoundError(
            f"load_calce_cell: no readable cycling files in {cell_dir}. "
            f"Expected .xlsx, .xls, .csv or .txt."
        )

    resolved_id = cell_id or _infer_cell_id(candidates[0])
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    skipped: list[tuple[str, str]] = []

    for path in candidates:
        try:
            frames.append(_read_one(path))
            used.append(path.name)
        except Exception as exc:
            # A corrupt or non-cycling file in the directory must not abort the
            # cell. It is recorded so a thin result is explainable.
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))

    if not frames:
        raise ValueError(
            f"load_calce_cell: every file in {cell_dir} failed to parse: "
            + "; ".join(f"{n} ({r})" for n, r in skipped)
        )

    telemetry = _reconcile_cycle_index(frames)
    telemetry["cell_id"] = resolved_id
    telemetry["dataset"] = "calce"

    if temperature_dir is not None:
        telemetry = _attach_temperature(
            telemetry, Path(temperature_dir), resolved_id
        )

    has_temperature = "temperature_c" in telemetry.columns
    unavailable = tuple(
        c for c in CALCE_UNAVAILABLE_CHANNELS if c not in telemetry.columns
    )

    report = CalceLoadReport(
        cell_id=resolved_id,
        n_files=len(used),
        n_rows=int(len(telemetry)),
        n_cycles=int(pd.to_numeric(telemetry["cycle"], errors="coerce").nunique()),
        files_used=tuple(used),
        files_skipped=tuple(skipped),
        unavailable_channels=unavailable,
        has_temperature=has_temperature,
    )
    return telemetry, report


def load_calce_dataset(
    base_dir: str | Path,
    temperature_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[CalceLoadReport]]:
    """Load every cell under `base_dir`, one subdirectory per cell.

    Expected layout, matching how CALCE distributes the archives::

        data/raw/calce/
          CS2_33/  CS2_33_10_04_10.xlsx  CS2_33_10_20_10.xlsx  ...
          CS2_34/  ...

    Cells that fail to load are skipped with their reason preserved in the
    returned reports rather than aborting the whole dataset, because one corrupt
    archive should not cost the other thirteen cells.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"load_calce_dataset: no such directory: {base_dir}")

    cell_dirs = sorted(p for p in base_dir.iterdir() if p.is_dir())
    if not cell_dirs:
        raise FileNotFoundError(
            f"load_calce_dataset: no cell subdirectories in {base_dir}. Expected "
            f"one directory per cell, e.g. {base_dir}/CS2_33/."
        )

    frames: list[pd.DataFrame] = []
    reports: list[CalceLoadReport] = []

    for cell_dir in cell_dirs:
        try:
            frame, report = load_calce_cell(
                cell_dir, temperature_dir=temperature_dir
            )
        except Exception as exc:
            reports.append(CalceLoadReport(
                cell_id=cell_dir.name, n_files=0, n_rows=0, n_cycles=0,
                files_skipped=((cell_dir.name, f"{type(exc).__name__}: {exc}"),),
            ))
            continue
        frames.append(frame)
        reports.append(report)

    if not frames:
        raise ValueError(
            f"load_calce_dataset: no cell in {base_dir} could be loaded. "
            + "; ".join(r.files_skipped[0][1] for r in reports if r.files_skipped)
        )

    return pd.concat(frames, ignore_index=True), reports


def summarize_calce_cycles(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Reduce CALCE telemetry to one row per cell-cycle.

    `capacity_ah` is taken as the **maximum** within a cycle, not the mean or
    the last value. Arbin's `Discharge_Capacity` accumulates monotonically
    through a discharge and resets between cycles, so its per-cycle maximum is
    the charge that cycle actually delivered. The mean would report roughly half
    of it, and the last value is unreliable when a file ends mid-cycle.
    """
    required = {"cell_id", "cycle"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"summarize_calce_cycles: missing {sorted(missing)}. Load with "
            f"load_calce_cell, which supplies both."
        )

    frame = telemetry.copy()
    frame["cycle"] = pd.to_numeric(frame["cycle"], errors="coerce")
    frame = frame.dropna(subset=["cycle"])
    frame["cycle"] = frame["cycle"].astype(int)

    aggregations: dict[str, tuple[str, str]] = {}
    if "capacity_ah" in frame.columns:
        aggregations["capacity_ah"] = ("capacity_ah", "max")
    if "voltage_v" in frame.columns:
        aggregations["mean_voltage_v"] = ("voltage_v", "mean")
        aggregations["min_voltage_v"] = ("voltage_v", "min")
    if "current_a" in frame.columns:
        aggregations["mean_current_a"] = ("current_a", "mean")
        aggregations["min_current_a"] = ("current_a", "min")
    if "resistance_ohm" in frame.columns:
        aggregations["resistance_ohm"] = ("resistance_ohm", "mean")
    if "temperature_c" in frame.columns:
        aggregations["avg_temp"] = ("temperature_c", "mean")
        aggregations["max_temp"] = ("temperature_c", "max")
    if "test_time_s" in frame.columns:
        aggregations["cycle_duration_s"] = ("test_time_s", lambda s: s.max() - s.min())

    aggregations["n_samples"] = ("cycle", "size")

    summary = (
        frame.groupby(["cell_id", "cycle"], as_index=False)
        .agg(**aggregations)
        .sort_values(["cell_id", "cycle"])
        .reset_index(drop=True)
    )
    summary["dataset"] = "calce"
    return summary


def calce_capacity_loss(summary: pd.DataFrame) -> pd.DataFrame:
    """Add per-cycle capacity loss relative to each cell's initial capacity.

    Initial capacity is the median of the first five cycles rather than cycle
    one alone: the first cycle of an Arbin schedule often includes a formation
    or conditioning step whose capacity is not representative.

    This mirrors the target the NASA calibration uses, so a model fitted on one
    is dimensionally comparable with the other.
    """
    if "capacity_ah" not in summary.columns:
        raise ValueError(
            "calce_capacity_loss: no 'capacity_ah' column. CS2/CX2 Arbin exports "
            "provide Discharge_Capacity(Ah); check the loader mapped it."
        )

    frame = summary.sort_values(["cell_id", "cycle"]).copy()

    def initial(group: pd.Series) -> float:
        head = group.head(5).dropna()
        return float(head.median()) if not head.empty else float("nan")

    initials = frame.groupby("cell_id")["capacity_ah"].apply(initial)
    frame["initial_capacity_ah"] = frame["cell_id"].map(initials)
    frame["capacity_loss"] = frame["initial_capacity_ah"] - frame["capacity_ah"]
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["soh"] = (frame["capacity_ah"] / frame["initial_capacity_ah"]) * 100.0
    return frame.reset_index(drop=True)
