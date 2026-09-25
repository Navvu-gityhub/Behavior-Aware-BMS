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

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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

# CALCE files its cells under `Type1` .. `Type6`, one directory per documented
# experiment type. That directory name IS the cohort, which is what
# leave-one-cohort-out needs; see `discover_calce_cells`.
_COHORT_IN_DIR = re.compile(r"^type[_\s-]?\d+$", re.IGNORECASE)

# Looser than `_CELL_IN_NAME`, which requires a cell number. Used to recognise
# a directory that is *meant* to hold a cell even when it holds nothing, so an
# empty or unreadable one is reported rather than quietly omitted.
#
# The trailing separator matters: without it this also matches the container
# directory `CS2/`, which holds the Type folders and is not a cell. That
# produced a phantom sixteenth cell reported as unloadable.
_CELL_PREFIX = re.compile(r"^(?:cs2|cx2|cs_2|cx_2)[_\s-]", re.IGNORECASE)

# CADEX tester exports (CS2_8, CS2_21, CX2_4, CX2_31) share none of the Arbin
# schema: tab-separated, millivolts and milliamps rather than volts and amps,
# `Pgm cycle` instead of `Cycle_Index`, and a `Capacity` column whose units are
# not stated in the export.
#
# Loading one through the Arbin path does not fail — it produces a frame whose
# `capacity_ah` is off by three orders of magnitude. CS2_21 read as capacity
# "97.0 -> 86.0 Ah" on a 1.1 Ah cell before this guard existed, which is the
# most dangerous kind of wrong: plausible-looking, monotonically declining, and
# nonsense.
#
# Refusing is the correct behaviour until someone maps the schema against the
# cycler's documentation and confirms the units. It costs no cohort: CS2_8 and
# CS2_21 are both Type 1, which retains CS2_33 and CS2_34.
_CADEX_SIGNATURE: frozenset[str] = frozenset({"mV", "mA", "Pgm cycle"})

_READERS = {
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".csv": pd.read_csv,
    ".txt": lambda p: pd.read_csv(p, sep="\t"),
}

# Extensions that are data. Anything else inside an archive is ignored rather
# than attempted: the CS2_7 archive ships a `Thumbs.db`, and trying to parse it
# would fill the skip list with noise that hides a real failure.
_DATA_SUFFIXES: frozenset[str] = frozenset(_READERS)

# An Arbin workbook has TWO sheets: a human-readable `Info` header block and
# the actual channel log, named `Channel_<n>-<nnn>`. `pd.read_excel` defaults
# to the first sheet, which is `Info` — a 23-column metadata block that parses
# without error and contains no telemetry whatsoever.
#
# This is the failure mode this project keeps meeting: structurally valid,
# silently wrong. The fixtures are single-sheet CSVs, so no test caught it;
# only running against a real workbook did.
_ARBIN_SHEET_PREFIX = "channel"


def _pick_data_sheet(sheets: Sequence[str]) -> str | int:
    """Choose the telemetry sheet from an Arbin workbook.

    Prefers a sheet named `Channel_*`. Falls back to the last sheet when no
    name matches, because Arbin appends the channel log after the header
    block — never the first, which is where `Info` lives.
    """
    for name in sheets:
        if str(name).strip().lower().startswith(_ARBIN_SHEET_PREFIX):
            return name
    return sheets[-1] if sheets else 0


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
    truncated_at_cycle: int | None = None

    def render(self) -> str:
        lines = [
            f"{self.cell_id}: {self.n_rows:,} rows across {self.n_cycles} cycles "
            f"from {self.n_files} file(s)"
        ]
        if self.truncated_at_cycle is not None:
            lines.append(
                f"  TRUNCATED: reading stopped after cycle "
                f"{self.truncated_at_cycle}; this cell's later life was not "
                f"read and no end-of-life or full-trajectory quantity may be "
                f"derived from this frame"
            )
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

    # Modification time is the fallback for a file whose name carries no date.
    # It has to tolerate a path that does not exist on disk: archive members
    # are addressed as `Path("CS2_21/CS2_21_7_9b_10.txt")` and stat() raises
    # for them. Real CS2 archives contain both cases — `..._7_9b_10.txt` has a
    # letter inside the date, and `CS_2_5_15_12_calibration.xls` is not a
    # cycling file at all — and an unguarded stat() lost the entire cell over
    # one such member.
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (1, mtime, 0, 0, path.name)


def _infer_cell_id(path: Path) -> str:
    match = _CELL_IN_NAME.search(path.stem)
    return match.group(1).upper().replace("-", "_") if match else path.stem


def _read_frame(source, suffix: str) -> pd.DataFrame:
    """Parse one telemetry file from a path or an in-memory buffer.

    Split out from `_read_one` so archive members can be read without being
    extracted to disk: a CS2 cell is up to 235 MB compressed and unpacking
    fifteen of them to read them once is a lot of I/O for no benefit.
    """
    suffix = suffix.lower()
    if suffix in (".xlsx", ".xls"):
        workbook = pd.ExcelFile(source)
        return workbook.parse(_pick_data_sheet(workbook.sheet_names))

    reader = _READERS.get(suffix)
    if reader is None:
        raise ValueError(f"unsupported extension '{suffix}'")
    return reader(source)


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map source column names onto the unified schema."""
    if frame.empty:
        raise ValueError("file contains no rows")

    present = {str(c).strip() for c in frame.columns}
    if len(_CADEX_SIGNATURE & present) >= 2:
        raise ValueError(
            "CADEX tester export, not Arbin — columns include "
            f"{sorted(_CADEX_SIGNATURE & present)}. Units are mV/mA and the "
            "capacity column's units are unstated, so mapping it through the "
            "Arbin aliases yields a capacity wrong by ~1000x. Refused until "
            "the schema is confirmed against the cycler documentation."
        )

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


def _read_one(path: Path) -> pd.DataFrame:
    """Read and normalise one telemetry file from disk."""
    return _normalise_columns(_read_frame(path, path.suffix))


def _read_member(archive: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read and normalise one telemetry file from inside a zip archive."""
    with archive.open(name) as handle:
        buffer = io.BytesIO(handle.read())
    return _normalise_columns(_read_frame(buffer, Path(name).suffix))


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
    max_cycles: int | None = None,
) -> tuple[pd.DataFrame, CalceLoadReport]:
    """Load every cycling file for one CALCE cell into one telemetry frame.

    Returns the frame and a report naming the files used, the files skipped with
    reasons, and the channels this dataset does not record. The report is
    returned rather than logged because "CS2 has no temperature channel" is a
    fact a caller must act on, not a diagnostic to discard.

    `max_cycles` stops reading early and sets `truncated_at_cycle` on the
    report; see `load_calce_archive` for what may and may not be derived from a
    truncated frame.
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
    truncated: int | None = None
    running_cycles = 0

    for path in candidates:
        try:
            frame = _read_one(path)
            frames.append(frame)
            used.append(path.name)
            running_cycles += _cycles_in(frame)
        except Exception as exc:
            # A corrupt or non-cycling file in the directory must not abort the
            # cell. It is recorded so a thin result is explainable.
            skipped.append((path.name, f"{type(exc).__name__}: {exc}"))
            continue
        # See load_calce_archive for what truncation does and does not permit.
        if max_cycles is not None and running_cycles >= max_cycles:
            truncated = running_cycles
            break

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
        truncated_at_cycle=truncated,
    )
    return telemetry, report


def _cycles_in(frame: pd.DataFrame) -> int:
    """How many cycles one file contributes once reconciled.

    Mirrors the offset arithmetic in `_reconcile_cycle_index`. It is separate
    so a caller can count cycles as files are read, without concatenating
    first -- which is the whole point of being able to stop early.
    """
    if "cycle" not in frame.columns:
        return 1
    cycles = pd.to_numeric(frame["cycle"], errors="coerce")
    if cycles.notna().sum() == 0:
        return 1
    return int(cycles.max() - cycles.min() + 1)


def load_calce_archive(
    archive_path: str | Path,
    cell_id: str | None = None,
    max_cycles: int | None = None,
) -> tuple[pd.DataFrame, CalceLoadReport]:
    """Load one cell directly from its distributed `.zip`, without extracting.

    CALCE ships one archive per cell, each containing a `CS2_nn/` folder of
    date-named files. A CS2 cell runs to 235 MB compressed, so unpacking
    fifteen of them to read each once is a lot of I/O for nothing.

    Members are ordered by the date in their filename, exactly as the
    directory loader does, because a zip's member order is arbitrary and
    lexical order puts `10_04_10` before `9_20_10`.

    `max_cycles` stops reading once that many cycles have been accumulated.
    Default `None` reads everything, so existing callers are unaffected.

    THIS IS A TRUNCATION, AND THE REPORT SAYS SO
    --------------------------------------------
    An early-life feature -- Severson's Delta-Q between cycles 10 and 100 --
    needs the first few files of a 23-file archive, and reading the remaining
    twenty costs minutes per cell for data the feature discards. But a
    truncated frame is NOT a frame of that cell: its last cycle is an artifact
    of where reading stopped, not of where the cell died.

    So `truncated_at_cycle` is set on the report, and anything derived from a
    truncated frame that depends on the cell's full life -- cycle life,
    end-of-life, total throughput, final state of health -- is wrong. The
    field exists so that is visible rather than inferred from a suspiciously
    round cycle count.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"load_calce_archive: no such file: {archive_path}")

    resolved_id = cell_id or _infer_cell_id(archive_path)
    frames: list[pd.DataFrame] = []
    used: list[str] = []
    skipped: list[tuple[str, str]] = []
    truncated: int | None = None

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            name for name in archive.namelist()
            if not name.endswith("/") and Path(name).suffix.lower() in _DATA_SUFFIXES
        ]
        if not members:
            raise ValueError(
                f"load_calce_archive: {archive_path.name} contains no readable "
                f"telemetry files (looked for {sorted(_DATA_SUFFIXES)})."
            )

        running_cycles = 0
        for name in sorted(members, key=lambda n: _sort_key(Path(n))):
            try:
                frame = _read_member(archive, name)
                frames.append(frame)
                used.append(Path(name).name)
                running_cycles += _cycles_in(frame)
            except Exception as exc:
                skipped.append((Path(name).name, f"{type(exc).__name__}: {exc}"))
                continue
            if max_cycles is not None and running_cycles >= max_cycles:
                truncated = running_cycles
                break

    if not frames:
        raise ValueError(
            f"load_calce_archive: every member of {archive_path.name} failed to "
            f"parse: " + "; ".join(f"{n} ({r})" for n, r in skipped[:5])
        )

    telemetry = _reconcile_cycle_index(frames)
    telemetry["cell_id"] = resolved_id
    telemetry["dataset"] = "calce"

    return telemetry, CalceLoadReport(
        cell_id=resolved_id,
        n_files=len(used),
        n_rows=int(len(telemetry)),
        n_cycles=int(pd.to_numeric(telemetry["cycle"], errors="coerce").nunique()),
        files_used=tuple(used),
        files_skipped=tuple(skipped),
        unavailable_channels=tuple(
            c for c in CALCE_UNAVAILABLE_CHANNELS if c not in telemetry.columns
        ),
        has_temperature="temperature_c" in telemetry.columns,
        truncated_at_cycle=truncated,
    )


@dataclass(frozen=True)
class CalceSource:
    """One cell's data, and the cohort implied by where it was filed."""

    cell_id: str
    path: Path
    is_archive: bool
    cohort: str = ""


def _cohort_for(type_dir: Path) -> str:
    """Build a cohort label from the `<archive>/Type<n>/` directory pair.

    The archive name is part of the label, not decoration. CS2 and CX2 both
    number their experiment types 1 to 6, so a bare `Type1` would silently
    merge CS2's four 0.5C cells with CX2's four — different cell designs
    (1.1 Ah prismatic versus 1.35 Ah), different chemistry batches, and
    different absolute capacities.

    Merging them would corrupt leave-one-cohort-out in the direction that
    flatters it: holding out "Type1" would still leave half of that protocol's
    cells in training, so a model could memorise the protocol and the split
    would not notice. Namespacing gives twelve genuinely disjoint cohorts
    instead of six overlapping ones.
    """
    if not _COHORT_IN_DIR.match(type_dir.name):
        return ""
    family = type_dir.parent.name
    return f"{family}_{type_dir.name}" if family else type_dir.name


def discover_calce_cells(base_dir: str | Path) -> list[CalceSource]:
    """Find every cell under `base_dir`, however the archive was laid out.

    Two layouts are supported because both occur in practice:

        calce/CS2_33/CS2_33_10_04_10.xlsx      extracted, flat
        calce/CS2/Type1/CS2_33.zip             as distributed, nested by type

    **The second carries the cohort for free, and that matters.** CALCE groups
    CS2 and CX2 into six experiment types each, and leave-one-cohort-out needs
    that grouping. Reading it from the directory the file was filed in is
    strictly better than inferring it from telemetry
    (`adaptive.loaders.derive_cohorts_from_protocol`) and better than a
    transcribed lookup table, because it is the organisation the data actually
    shipped in.

    A cell not filed under a recognisable type folder gets an empty cohort,
    and the caller decides what to do about it rather than being handed a
    fabricated group.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"discover_calce_cells: no such directory: {base_dir}")

    found: dict[str, CalceSource] = {}

    for archive in sorted(base_dir.rglob("*.zip")):
        cell_id = _infer_cell_id(archive)
        found[cell_id] = CalceSource(
            cell_id, archive, True, _cohort_for(archive.parent),
        )

    for directory in sorted(p for p in base_dir.rglob("*") if p.is_dir()):
        if _COHORT_IN_DIR.match(directory.name):
            continue
        data_files = [
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in _DATA_SUFFIXES
        ]
        if not data_files:
            # A directory NAMED like a cell but holding nothing is still
            # registered, so the loader reports it as unloadable rather than
            # omitting it. A cell that disappears from a report is
            # indistinguishable from one that was never there.
            if _CELL_PREFIX.match(directory.name) and directory.name not in found:
                parent_name = directory.parent.name
                found[directory.name] = CalceSource(
                    directory.name, directory, False,
                    parent_name if _COHORT_IN_DIR.match(parent_name) else "",
                )
            continue
        cell_id = _infer_cell_id(data_files[0])
        if cell_id in found:
            continue
        found[cell_id] = CalceSource(
            cell_id, directory, False, _cohort_for(directory.parent),
        )

    return [found[k] for k in sorted(found)]


def load_calce_dataset(
    base_dir: str | Path,
    temperature_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, list[CalceLoadReport]]:
    """Load every cell under `base_dir`, in either supported layout.

    Both of these work, and the second is how CALCE actually distributes the
    data::

        data/raw/calce/CS2_33/CS2_33_10_04_10.xlsx     extracted, flat
        data/raw/calce/CS2/Type1/CS2_33.zip            as downloaded

    When cells are filed under `Type<n>` directories, that name is carried
    through as a `cohort` column — CALCE's own experiment grouping, which is
    exactly what leave-one-cohort-out needs and is better evidence than any
    grouping inferred from telemetry.

    Cells that fail to load are skipped with their reason preserved in the
    returned reports rather than aborting the whole dataset, because one corrupt
    archive should not cost the other twelve cells.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"load_calce_dataset: no such directory: {base_dir}")

    sources = discover_calce_cells(base_dir)
    if not sources:
        raise FileNotFoundError(
            f"load_calce_dataset: no cells found under {base_dir}. Expected "
            f"either one directory per cell (e.g. {base_dir}/CS2_33/) or the "
            f"distributed archives (e.g. {base_dir}/CS2/Type1/CS2_33.zip)."
        )

    frames: list[pd.DataFrame] = []
    reports: list[CalceLoadReport] = []

    for source in sources:
        try:
            if source.is_archive:
                frame, report = load_calce_archive(source.path, cell_id=source.cell_id)
            else:
                frame, report = load_calce_cell(
                    source.path, cell_id=source.cell_id,
                    temperature_dir=temperature_dir,
                )
            if source.cohort:
                frame["cohort"] = source.cohort
        except Exception as exc:
            reports.append(CalceLoadReport(
                cell_id=source.cell_id, n_files=0, n_rows=0, n_cycles=0,
                files_skipped=((source.path.name, f"{type(exc).__name__}: {exc}"),),
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

    `capacity_ah` is the **span** of `Discharge_Capacity` within the cycle —
    its maximum minus its minimum — which is the charge that cycle actually
    delivered.

    THIS WAS WRONG UNTIL REAL DATA WAS RUN THROUGH IT
    --------------------------------------------------
    A previous version took the per-cycle *maximum*, on the stated assumption
    that Arbin's `Discharge_Capacity` "accumulates through a discharge and
    resets between cycles". **It does not reset.** In the CS2 workbooks the
    counter accumulates monotonically across every cycle in a file::

        Cycle 1:  0.0000 -> 1.0849      delta 1.0849
        Cycle 2:  1.0849 -> 2.1718      delta 1.0869
        Cycle 3:  2.1718 -> 3.1423      delta 0.9705   (fading)

    Taking the maximum therefore returned the cell's cumulative throughput, so
    CS2_33 reported capacity climbing 1.16 -> 4.45 Ah and a state of health of
    383% on a 1.1 Ah cell. The span gives 1.08 Ah on cycle 1, which is the
    right answer for that cell.

    The regression fixtures reproduced the same false assumption — they write
    a counter that resets — so no test caught it. Only loading a real workbook
    did, which is the third time in this project that a structurally valid
    frame turned out to carry a wrong quantity.
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

    def _span(series: pd.Series) -> float:
        """Charge delivered within this cycle. See the docstring."""
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.max() - values.min()) if len(values) else float("nan")

    aggregations: dict[str, tuple] = {}
    if "capacity_ah" in frame.columns:
        aggregations["capacity_ah"] = ("capacity_ah", _span)
        # Retained so a reader can tell an accumulating counter from a
        # resetting one without re-deriving it: on an accumulating export this
        # rises without bound, on a resetting one it tracks capacity_ah.
        aggregations["cumulative_capacity_ah"] = ("capacity_ah", "max")
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

    # Carry the cohort through the aggregation. It is a per-cell constant, so
    # a map from cell id is exact — and losing it here would silently strip the
    # column leave-one-cohort-out depends on, leaving a frame that looks fine
    # and cannot be validated across protocols.
    if "cohort" in telemetry.columns:
        by_cell = telemetry.groupby("cell_id")["cohort"].first()
        summary["cohort"] = summary["cell_id"].map(by_cell)

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
