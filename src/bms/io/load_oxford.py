"""Oxford Battery Degradation Dataset 1: nested MATLAB struct ingestion.

Eight Kokam SLPB533459H4 pouch cells (740 mAh nominal, NMC/LCO blend cathode),
aged at 40 C with a constant-current-constant-voltage charge and an ARTEMIS
urban **drive-cycle** discharge. Every 100 ageing cycles the schedule pauses
for a characterisation: a 1C charge and discharge, then a pseudo-OCV sweep at
C/18.

WHY THIS DATASET IS WORTH LOADING, AND WHAT IT IS NOT FOR
---------------------------------------------------------
Two things it genuinely adds:

**A third chemistry.** NASA and CALCE are both LCO. Oxford is an NMC/LCO blend
pouch cell, so a model that transfers to it has crossed a chemistry boundary,
not just a protocol boundary.

**A real behavioural load.** The discharge is an actual drive cycle rather than
a constant current. This project is called "behavior-aware"; Oxford is the only
one of the candidate datasets where the discharge profile resembles driving.

One thing it is emphatically not for, and the commensurability screen says so
before the download (`python -m src.bms.adaptive feasibility`):

    nasa -> oxford_degradation: PREDICTED_MARGINAL
        ambient_temperature   blocked   varied -> fixed
        depth_of_discharge    blocked   varied -> fixed
        discharge_rate        blocked   varied -> fixed
        internal_resistance   MARGINAL  incidental -> incidental

**All eight cells share one protocol at one temperature.** There are no
cohorts, so Oxford cannot support a leave-one-cohort-out test on its own, and
it cannot receive a fitted ambient-temperature coefficient because ambient
does not vary. It is an external held-out target for cell-level
generalisation, and `OxfordLoadReport` states that rather than leaving a
caller to infer it from a suitability warning.

CAPACITY IS SPARSE HERE, AND THAT IS STRUCTURAL
-----------------------------------------------
Discharge capacity is only measurable in the periodic characterisation cycles,
because the drive-cycle ageing cycles are partial discharges of varying depth.
A cell run to 8,000 cycles therefore yields roughly 80 usable fade points, not
8,000. `summarize_oxford_cycles` returns one row per characterisation, and the
count is reported so a thin result is explainable rather than surprising.

This is the same distinction `telemetry/cycles.py` draws for CAN logs: a
partial discharge is not a capacity measurement, and scaling it by the observed
SOC swing would make the measurement depend on the quantity being measured.

THE FILE FORMAT
---------------
One `.mat` file containing eight top-level structs::

    Cell1
      cyc0000
        C1ch    t, v, q, T      1C charge
        C1dc    t, v, q, T      1C discharge   <- the capacity measurement
        OCVch   t, v, q, T      C/18 charge
        OCVdc   t, v, q, T      C/18 discharge
      cyc0100
        ...

`t` is time, `v` voltage, `q` charge, `T` temperature. Field names are read
from the file rather than assumed, so a cell missing a measurement block is
reported instead of raising.

**This loader is written against the published format and tested against a
fixture that reproduces it** (`tests/fixtures/make_oxford_fixture.py`). It has
not been run against the real archive, because that archive is not in this
repository. Anything it produces on real data should be checked against
`OxfordLoadReport` before being trusted — in particular the capacity-unit
detection described below.

UNITS ARE DETECTED AND REPORTED, NOT ASSUMED
--------------------------------------------
The published `q` field is in mAh for these cells, but different mirrors and
re-exports of this dataset have shipped it in Ah. Rather than hard-code either,
`_detect_capacity_scale` compares the observed maximum against the 740 mAh
nominal and reports which interpretation it chose. A silent factor-of-1000
error in a fade target would be invisible in every downstream ratio — SOH,
capacity_loss and cumulative_fade are all scale-free — and would only surface
as an absurd absolute capacity nobody was looking at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Nominal capacity of the Kokam SLPB533459H4, in ampere-hours.
OXFORD_NOMINAL_CAPACITY_AH = 0.740

# All eight cells were aged in a thermal chamber at this temperature.
OXFORD_AMBIENT_TEMPERATURE_C = 40.0

# The single protocol every cell shares. Named rather than left implicit so a
# frame carrying it cannot be mistaken for one with real cohort structure.
OXFORD_COHORT = "OXFORD_40C_ARTEMIS"

# The four measurement blocks inside each characterisation cycle.
#
# C1dc is the one that yields capacity: a full 1C discharge from full to the
# cutoff. C1ch is its charge counterpart. The OCV sweeps are at C/18 and are
# for differential-voltage analysis rather than capacity.
CAPACITY_MEASUREMENT = "C1dc"
MEASUREMENT_BLOCKS: tuple[str, ...] = ("C1ch", "C1dc", "OCVch", "OCVdc")

# Field names inside a measurement block, mapped to the unified schema.
FIELD_MAP: dict[str, str] = {
    "t": "test_time_s",
    "v": "voltage_v",
    "q": "capacity_ah_curve",
    "T": "temperature_c",
}

_CELL_KEY = re.compile(r"^Cell\s*_?(\d+)$", re.IGNORECASE)
_CYCLE_KEY = re.compile(r"^cyc\s*_?(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class OxfordLoadReport:
    """What a load produced, and the assumptions it had to make."""

    n_cells: int
    n_rows: int
    n_characterisations: int
    capacity_scale: float
    capacity_units_detected: str
    cells_loaded: tuple[str, ...] = ()
    blocks_missing: tuple[tuple[str, str], ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def single_cohort(self) -> bool:
        """Always true for this dataset. Stated, not inferred."""
        return True

    def render(self) -> str:
        lines = [
            f"Oxford: {self.n_cells} cells, {self.n_rows:,} rows, "
            f"{self.n_characterisations} characterisation cycles",
            f"  capacity read as {self.capacity_units_detected} "
            f"(scale x{self.capacity_scale:g} to Ah)",
            f"  all cells share one protocol at "
            f"{OXFORD_AMBIENT_TEMPERATURE_C:g} C ({OXFORD_COHORT}), so "
            f"leave-one-cohort-out is unavailable on this dataset alone",
        ]
        for cell, block in self.blocks_missing[:10]:
            lines.append(f"  missing block: {cell}/{block}")
        if len(self.blocks_missing) > 10:
            lines.append(f"  ... and {len(self.blocks_missing) - 10} more")
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        return "\n".join(lines)


def _fieldnames(obj) -> list[str]:
    """Field names of a scipy `mat_struct`, or [] for anything else."""
    names = getattr(obj, "_fieldnames", None)
    return list(names) if names else []


def _detect_capacity_scale(
    values: np.ndarray,
    nominal_ah: float = OXFORD_NOMINAL_CAPACITY_AH,
) -> tuple[float, str]:
    """Decide whether `q` is in Ah or mAh, by comparison with the nominal.

    Returns the multiplier that converts to ampere-hours, and a label for the
    report. A degraded cell delivers less than nominal and a fresh one a little
    more, so the observed maximum should land within a factor of about two of
    nominal under the correct interpretation — the two candidate scales differ
    by a factor of a thousand, so the test has enormous margin and does not
    need to be delicate.

    Falls back to Ah with a recorded caveat when the data is empty or the
    magnitude fits neither, rather than picking the closer of two bad options.
    """
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 1.0, "Ah (assumed; no finite values to test)"

    observed = float(np.max(np.abs(finite)))
    if observed <= 0:
        return 1.0, "Ah (assumed; all values zero)"

    # Ratio to nominal under each interpretation.
    as_ah = observed / nominal_ah
    as_mah = (observed / 1000.0) / nominal_ah

    if 0.2 <= as_ah <= 5.0:
        return 1.0, "Ah"
    if 0.2 <= as_mah <= 5.0:
        return 0.001, "mAh"

    return 1.0, (
        f"Ah (assumed; observed maximum {observed:g} fits neither Ah nor mAh "
        f"against a {nominal_ah:g} Ah nominal — check the source)"
    )


def _block_to_frame(block, cell_id: str, cycle: int, name: str) -> pd.DataFrame | None:
    """Turn one measurement block into a long frame, or None if unusable."""
    present = _fieldnames(block)
    if not present:
        return None

    columns: dict[str, np.ndarray] = {}
    for source, target in FIELD_MAP.items():
        if source not in present:
            continue
        values = np.atleast_1d(np.asarray(getattr(block, source), dtype=float))
        columns[target] = values

    if not columns:
        return None

    # Blocks are parallel arrays sampled together; a length mismatch means the
    # file is not what this loader expects, so it is reported rather than
    # padded into agreement.
    lengths = {len(v) for v in columns.values()}
    if len(lengths) != 1:
        return None

    frame = pd.DataFrame(columns)
    frame["cell_id"] = cell_id
    frame["cycle"] = cycle
    frame["measurement"] = name
    return frame


def load_oxford_mat(
    path: str | Path,
    measurements: tuple[str, ...] = MEASUREMENT_BLOCKS,
) -> tuple[pd.DataFrame, OxfordLoadReport]:
    """Load the Oxford `.mat` archive into one long telemetry frame.

    Returns the frame and a report. The report carries the capacity-unit
    decision and the single-cohort fact, both of which a caller has to act on.
    """
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - scipy is a core dependency
        raise ImportError(
            "load_oxford_mat needs scipy. Install with `pip install scipy`."
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"load_oxford_mat: no such file: {path}")

    # `struct_as_record=False, squeeze_me=True` gives attribute access to
    # nested structs instead of nested object arrays, which is the difference
    # between `cell.cyc0100.C1dc.q` and four levels of [0][0] indexing.
    raw = loadmat(str(path), struct_as_record=False, squeeze_me=True)

    cell_keys = sorted(
        (k for k in raw if _CELL_KEY.match(str(k))),
        key=lambda k: int(_CELL_KEY.match(str(k)).group(1)),
    )
    if not cell_keys:
        raise ValueError(
            f"load_oxford_mat: no Cell<N> variables in {path.name}. Found "
            f"{sorted(k for k in raw if not str(k).startswith('__'))}. This "
            f"loader expects Oxford Battery Degradation Dataset 1's layout."
        )

    frames: list[pd.DataFrame] = []
    missing: list[tuple[str, str]] = []
    cells_loaded: list[str] = []

    for key in cell_keys:
        cell = raw[key]
        cell_id = str(key)
        cycle_keys = sorted(
            (f for f in _fieldnames(cell) if _CYCLE_KEY.match(f)),
            key=lambda f: int(_CYCLE_KEY.match(f).group(1)),
        )
        if not cycle_keys:
            missing.append((cell_id, "no cyc<NNNN> fields"))
            continue

        for cycle_key in cycle_keys:
            cycle = int(_CYCLE_KEY.match(cycle_key).group(1))
            characterisation = getattr(cell, cycle_key)
            available = _fieldnames(characterisation)

            for name in measurements:
                if name not in available:
                    missing.append((cell_id, f"{cycle_key}/{name}"))
                    continue
                frame = _block_to_frame(
                    getattr(characterisation, name), cell_id, cycle, name
                )
                if frame is None:
                    missing.append((cell_id, f"{cycle_key}/{name} (unreadable)"))
                    continue
                frames.append(frame)

        cells_loaded.append(cell_id)

    if not frames:
        raise ValueError(
            f"load_oxford_mat: no readable measurement blocks in {path.name}. "
            f"Expected Cell<N>.cyc<NNNN>.{{{','.join(measurements)}}} with "
            f"fields {sorted(FIELD_MAP)}."
        )

    telemetry = pd.concat(frames, ignore_index=True)

    scale, units = (1.0, "Ah (no capacity column)")
    if "capacity_ah_curve" in telemetry.columns:
        discharge = telemetry.loc[
            telemetry["measurement"] == CAPACITY_MEASUREMENT, "capacity_ah_curve"
        ]
        reference = discharge if len(discharge) else telemetry["capacity_ah_curve"]
        scale, units = _detect_capacity_scale(reference.to_numpy(dtype=float))
        telemetry["capacity_ah_curve"] = telemetry["capacity_ah_curve"] * scale

    telemetry["dataset"] = "oxford"
    telemetry["cohort"] = OXFORD_COHORT
    telemetry["ambient_temperature_c"] = OXFORD_AMBIENT_TEMPERATURE_C

    warnings: list[str] = []
    if "temperature_c" not in telemetry.columns:
        warnings.append(
            "no temperature channel found in any block; downstream thermal "
            "features will be unavailable rather than defaulted"
        )

    report = OxfordLoadReport(
        n_cells=len(cells_loaded),
        n_rows=int(len(telemetry)),
        n_characterisations=int(
            telemetry.groupby("cell_id")["cycle"].nunique().sum()
        ),
        capacity_scale=scale,
        capacity_units_detected=units,
        cells_loaded=tuple(cells_loaded),
        blocks_missing=tuple(missing),
        warnings=tuple(warnings),
    )
    return telemetry, report


def summarize_oxford_cycles(
    telemetry: pd.DataFrame,
    measurement: str = CAPACITY_MEASUREMENT,
) -> pd.DataFrame:
    """Reduce Oxford telemetry to one row per cell-characterisation.

    Capacity is the **maximum absolute** `q` within the 1C discharge block.
    Oxford records `q` as accumulated charge over the discharge, so its
    magnitude at the end of the block is the charge that discharge delivered.
    The absolute value is taken because sign conventions for discharge differ
    between exports of this dataset, and a sign flip would otherwise turn every
    capacity into a negative number and every fade into a gain.

    Only the discharge block is used. Averaging over all four blocks would mix
    a 1C discharge with a C/18 OCV sweep, which measure different quantities.
    """
    required = {"cell_id", "cycle", "measurement"}
    missing = required - set(telemetry.columns)
    if missing:
        raise ValueError(
            f"summarize_oxford_cycles: missing {sorted(missing)}. Load with "
            f"load_oxford_mat, which supplies all three."
        )
    if "capacity_ah_curve" not in telemetry.columns:
        raise ValueError(
            "summarize_oxford_cycles: no 'capacity_ah_curve' column, so no "
            "capacity can be recovered. The 'q' field was absent from every "
            "measurement block — check OxfordLoadReport.blocks_missing."
        )

    discharge = telemetry[telemetry["measurement"] == measurement]
    if discharge.empty:
        raise ValueError(
            f"summarize_oxford_cycles: no '{measurement}' blocks in this frame. "
            f"Present: {sorted(telemetry['measurement'].unique())}. Capacity is "
            f"only measurable in the 1C discharge; the OCV sweeps are for "
            f"differential-voltage analysis."
        )

    aggregations: dict[str, tuple] = {
        "capacity_ah": ("capacity_ah_curve", lambda s: float(np.nanmax(np.abs(s)))),
        "n_samples": ("cycle", "size"),
    }
    if "voltage_v" in discharge.columns:
        aggregations["mean_voltage_v"] = ("voltage_v", "mean")
        aggregations["min_voltage_v"] = ("voltage_v", "min")
    if "temperature_c" in discharge.columns:
        aggregations["avg_temp"] = ("temperature_c", "mean")
        aggregations["max_temp"] = ("temperature_c", "max")
    if "test_time_s" in discharge.columns:
        aggregations["cycle_duration_s"] = (
            "test_time_s", lambda s: float(s.max() - s.min()),
        )

    summary = (
        discharge.groupby(["cell_id", "cycle"], as_index=False)
        .agg(**aggregations)
        .sort_values(["cell_id", "cycle"])
        .reset_index(drop=True)
    )
    summary["dataset"] = "oxford"
    summary["cohort"] = OXFORD_COHORT
    summary["ambient_temperature_c"] = OXFORD_AMBIENT_TEMPERATURE_C
    return summary


def oxford_capacity_loss(summary: pd.DataFrame) -> pd.DataFrame:
    """Add per-cycle capacity loss relative to each cell's initial capacity.

    Initial capacity is taken from the **first characterisation** rather than a
    median of the first few, because Oxford characterises only every 100
    cycles: a median of the first five would average over 400 cycles of real
    ageing. This differs from `load_calce_cycling.calce_capacity_loss`
    deliberately, and the reason is the measurement cadence rather than a
    preference.
    """
    if "capacity_ah" not in summary.columns:
        raise ValueError(
            "oxford_capacity_loss: no 'capacity_ah' column. Run "
            "summarize_oxford_cycles first."
        )

    frame = summary.sort_values(["cell_id", "cycle"]).copy()
    initials = frame.groupby("cell_id")["capacity_ah"].first()
    frame["initial_capacity_ah"] = frame["cell_id"].map(initials)
    frame["capacity_loss"] = frame["initial_capacity_ah"] - frame["capacity_ah"]
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["soh"] = (frame["capacity_ah"] / frame["initial_capacity_ah"]) * 100.0
    return frame.reset_index(drop=True)
