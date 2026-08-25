"""Concrete dataset adapters for the calibration registry.

`datasets.py` owns the screening logic and the `DatasetLoader` protocol. This
module owns the adapters that satisfy it, so a new dataset can be added
without touching the screen — and, more importantly, so a new dataset cannot
skip the screen by declaring itself fit.

Each adapter's job is narrow: read the raw archive with the loader that
already understands its format, reduce it to one row per cell-cycle, attach a
fade target, and attach a cohort label. Nothing here decides whether the
result is usable; `assess_suitability` does that from what is returned.

THE COHORT PROBLEM, WHICH IS THE INTERESTING PART
--------------------------------------------------
Leave-one-cohort-out is this project's binding validation constraint (ADR
0005), and it needs a cohort column. NASA supplies one because its cells are
documented into nine named test protocols. Neither CALCE nor Oxford ships a
protocol label in the data files.

Three ways to respond, and only one of them is honest:

**Refuse.** Defensible but wasteful: CS2 genuinely does vary depth of
discharge and discharge rate across its cells, so protocol structure exists
even though it is not written down in the export.

**Invent a mapping from memory.** CALCE documents CS2 groupings as Type-1
through Type-6 on its website. Hard-coding a cell-to-type table from
recollection would put an unverified claim about which cell ran which protocol
at the root of every downstream LOCO result, where it would be invisible. This
module does not do that.

**Derive the cohort from what was measured, and label it as derived.** A
cell's protocol shows up in its telemetry: discharge rate and cutoff voltage
are measurable per cell. `derive_cohorts_from_protocol` groups cells by those
observables and names the result `DERIVED_<rate>C_<cutoff>V` so it can never be
mistaken for CALCE's own documented grouping.

That is the same rule the CALCE cycling loader already follows about
temperature — the pipeline follows the instrumentation, not the dataset label
(`load_calce_cycling`, module docstring).

An authoritative mapping is still strictly better, so every adapter accepts a
`cohort_map`. When one is supplied it wins, and the derived labels are not
used at all.

**CALCE's own grouping has since been located** and is provided as
`CALCE_CS2_COHORTS` / `CALCE_CX2_COHORTS`, transcribed from the CALCE dataset
page rather than recalled. Prefer it::

    CalceCyclingLoader(base_dir=..., cohort_map=CALCE_CS2_COHORTS)

The derived path remains for archives with no published grouping, and as the
fallback when a cell is present that the map does not cover. Note that the
axes CALCE actually varied — discharge rate and cutoff voltage — are exactly
the two the derivation measures, so the two approaches should broadly agree;
comparing them on real data is a cheap check on both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

# Rounding applied to the measured protocol descriptors before they become a
# cohort label. Coarse on purpose: the point is to group cells that ran the
# same schedule, and finer resolution would split one protocol into several
# cohorts on measurement noise, which inflates the apparent number of
# protocols and makes LOCO look better-powered than it is.
DEFAULT_C_RATE_DECIMALS = 1
DEFAULT_VOLTAGE_DECIMALS = 1


# ---------------------------------------------------------------------------
# Authoritative CALCE cohort maps
#
# Transcribed from the CALCE Battery Research Group's own dataset page,
# https://calce.umd.edu/battery-data, which documents CS2 and CX2 as six
# experiment types each and lists the cells in every type.
#
# These are NOT derived from telemetry and are not a guess. They are the
# dataset authors' own grouping, and they are strictly preferable to
# `derive_cohorts_from_protocol` — pass one as `cohort_map` and the derived
# labels are not used at all.
#
# VERIFY BEFORE PUBLISHING ON THEM. This mapping was read off a web page, not
# out of the archive. Four independent facts already agree with it, which is
# why it is here rather than in a comment: the cell counts total 15 for CS2
# and 12 for CX2, matching `dataset_specs.CALCE_CS2_SPEC.n_cells` and
# `CALCE_CX2_SPEC.n_cells`; CX2-4 appears as the temperature-cycled type,
# matching `CALCE_CX2_4_THERMAL_SPEC`; and CS2-8, CS2-21, CX2-4 and CX2-31 are
# the CADEX-tester cells already recorded as `.txt` rather than Arbin Excel in
# those specs. Agreement on four checks is good evidence and is not proof, so
# confirm against the downloaded archive's own readme before any published
# LOCO result rests on it.
#
# Cell ids use the underscore form the loader emits (`_infer_cell_id`
# uppercases and converts hyphens), not the hyphenated form the website uses.
CALCE_CS2_COHORTS: Mapping[str, str] = {
    # Type 1 - constant current discharge at 0.5C
    "CS2_8": "CS2_TYPE1_0.5C", "CS2_21": "CS2_TYPE1_0.5C",
    "CS2_33": "CS2_TYPE1_0.5C", "CS2_34": "CS2_TYPE1_0.5C",
    # Type 2 - constant current discharge at 1C
    "CS2_35": "CS2_TYPE2_1C", "CS2_36": "CS2_TYPE2_1C",
    "CS2_37": "CS2_TYPE2_1C", "CS2_38": "CS2_TYPE2_1C",
    # Type 3 - discharge rate switched among six predetermined rates per cycle
    "CS2_3": "CS2_TYPE3_VARRATE", "CS2_9": "CS2_TYPE3_VARRATE",
    # Type 4 - randomised cutoff voltage
    "CS2_7": "CS2_TYPE4_RANDCUTOFF",
    # Type 5 - low-voltage partial charge/discharge cycles
    "CS2_5": "CS2_TYPE5_LOWPARTIAL", "CS2_6": "CS2_TYPE5_LOWPARTIAL",
    # Type 6 - high-voltage partial charge/discharge cycles
    "CS2_24": "CS2_TYPE6_HIGHPARTIAL", "CS2_25": "CS2_TYPE6_HIGHPARTIAL",
}

CALCE_CX2_COHORTS: Mapping[str, str] = {
    "CX2_16": "CX2_TYPE1_0.5C", "CX2_31": "CX2_TYPE1_0.5C",
    "CX2_33": "CX2_TYPE1_0.5C", "CX2_35": "CX2_TYPE1_0.5C",
    "CX2_34": "CX2_TYPE2_0.5C_ALT", "CX2_36": "CX2_TYPE2_0.5C_ALT",
    "CX2_37": "CX2_TYPE2_0.5C_ALT", "CX2_38": "CX2_TYPE2_0.5C_ALT",
    "CX2_8": "CX2_TYPE3_3C",
    "CX2_3": "CX2_TYPE4_PULSED",
    # The one CALCE cell with a thermal axis (25/35/45/55 C). Load it with
    # `temperature_dir=` so the thermocouple files are joined; see
    # `io.load_calce_cycling.load_calce_cell`.
    "CX2_4": "CX2_TYPE5_TEMPCYCLED",
    "CX2_32": "CX2_TYPE6_PULSEDPROFILE",
}

# Both archives together, for a run spanning CS2 and CX2.
CALCE_ALL_COHORTS: Mapping[str, str] = {**CALCE_CS2_COHORTS, **CALCE_CX2_COHORTS}


def derive_cohorts_from_protocol(
    summary: pd.DataFrame,
    nominal_capacity_ah: float,
    cell_col: str = "cell_id",
    current_col: str = "min_current_a",
    voltage_col: str = "min_voltage_v",
    c_rate_decimals: int = DEFAULT_C_RATE_DECIMALS,
    voltage_decimals: int = DEFAULT_VOLTAGE_DECIMALS,
) -> tuple[pd.Series, tuple[str, ...]]:
    """Group cells by their measured discharge rate and cutoff voltage.

    Returns a per-row cohort label and any warnings the grouping produced.

    Uses the **median across a cell's cycles** of each descriptor, not a single
    cycle, so one anomalous cycle cannot reassign a cell's protocol. Labels are
    prefixed `DERIVED_` because they are an inference from telemetry, not the
    dataset's own grouping — see the module docstring.

    Degenerate outcomes are reported rather than silently accepted. One cohort
    means LOCO is unavailable; one cohort per cell means LOCO has collapsed
    into LOBO and would produce a number that looks like cross-protocol
    validation while being nothing of the kind. That second case is the
    dangerous one, because it fails quietly.
    """
    warnings: list[str] = []

    available = [c for c in (current_col, voltage_col) if c in summary.columns]
    if not available:
        warnings.append(
            f"neither '{current_col}' nor '{voltage_col}' is present, so no "
            f"protocol descriptor could be measured; every cell is assigned to "
            f"one cohort and leave-one-cohort-out is unavailable"
        )
        return pd.Series("DERIVED_UNKNOWN", index=summary.index), tuple(warnings)

    per_cell = summary.groupby(cell_col)[available].median()
    parts: list[pd.Series] = []

    if current_col in available:
        c_rate = (per_cell[current_col].abs() / nominal_capacity_ah).round(
            c_rate_decimals
        )
        parts.append(c_rate.map(lambda v: f"{v:g}C"))
    if voltage_col in available:
        cutoff = per_cell[voltage_col].round(voltage_decimals)
        parts.append(cutoff.map(lambda v: f"{v:g}V"))

    labels = "DERIVED_" + parts[0].astype(str)
    for extra in parts[1:]:
        labels = labels + "_" + extra.astype(str)

    n_cohorts = int(labels.nunique())
    n_cells = int(len(labels))

    if n_cohorts <= 1:
        warnings.append(
            f"all {n_cells} cells share one derived protocol label, so "
            f"leave-one-cohort-out is unavailable on this dataset alone"
        )
    elif n_cohorts == n_cells:
        warnings.append(
            f"every one of {n_cells} cells received a distinct protocol label, "
            f"so leave-one-cohort-out would be identical to leave-one-cell-out "
            f"and would not test cross-protocol transfer at all. Supply an "
            f"authoritative cohort_map, or coarsen the rounding."
        )

    return summary[cell_col].map(labels), tuple(warnings)


@dataclass
class CalceCyclingLoader:
    """CALCE CS2/CX2 cycling archives as a fade-calibration dataset.

    Wraps `io.load_calce_cycling`, which already handles the parts that make
    CALCE awkward: many date-named files per cell whose `Cycle_Index` each
    restart at 1, and the absence of a temperature channel.

    `temperature_dir` applies only to CX2_4, the one CALCE cell cycled across
    several temperatures with separate thermocouple files. For every other cell
    the correct value is None, and passing a directory would not invent a
    temperature — the join simply finds no matching files.
    """

    name: str = "calce"
    base_dir: Path | str = Path("data/raw/calce")
    nominal_capacity_ah: float = 1.1
    temperature_dir: Path | str | None = None
    cohort_map: Mapping[str, str] = field(default_factory=dict)
    load_warnings: tuple[str, ...] = field(default=(), init=False)

    def load(self) -> pd.DataFrame:
        from src.bms.io.load_calce_cycling import (
            calce_capacity_loss,
            load_calce_dataset,
            summarize_calce_cycles,
        )

        base = Path(self.base_dir)
        if not base.exists():
            raise FileNotFoundError(
                f"{self.name}: no such directory: {base}. CALCE archives are "
                f"gitignored and not redistributable; see "
                f"configs/dataset_sources.yaml for where to obtain them, and "
                f"docs/calce_integration.md for the expected layout."
            )

        telemetry, reports = load_calce_dataset(
            base, temperature_dir=self.temperature_dir
        )
        summary = summarize_calce_cycles(telemetry)
        summary = calce_capacity_loss(summary)

        summary["cohort"], warnings = self._assign_cohorts(summary)
        skipped = tuple(
            f"{r.cell_id}: {r.files_skipped[0][1]}"
            for r in reports if r.files_skipped and r.n_rows == 0
        )
        self.load_warnings = warnings + skipped
        return summary

    def _assign_cohorts(self, summary: pd.DataFrame) -> tuple[pd.Series, tuple[str, ...]]:
        if self.cohort_map:
            labels = summary["cell_id"].map(dict(self.cohort_map))
            unmapped = sorted(summary.loc[labels.isna(), "cell_id"].unique())
            if unmapped:
                # An unmapped cell must not quietly fall into a catch-all
                # bucket: it would form a fake cohort of unrelated cells and
                # LOCO would hold out a group that never shared a protocol.
                raise ValueError(
                    f"{self.name}: cohort_map has no entry for {unmapped}. "
                    f"Add them, or omit cohort_map to derive labels from "
                    f"measured protocol descriptors."
                )
            return labels, ()
        return derive_cohorts_from_protocol(
            summary, nominal_capacity_ah=self.nominal_capacity_ah
        )


@dataclass
class OxfordLoader:
    """Oxford Battery Degradation Dataset 1 as a calibration dataset.

    All eight cells ran one protocol at 40 C, so this dataset has exactly one
    cohort and cannot support leave-one-cohort-out on its own. That is not a
    defect to be worked around: it makes Oxford an *external* target — a third
    chemistry on which to test a model fitted elsewhere — rather than a
    training set. `assess_suitability` will emit the one-cohort warning, and
    that warning is correct.
    """

    name: str = "oxford"
    path: Path | str = Path("data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat")
    cohort_map: Mapping[str, str] = field(default_factory=dict)
    load_warnings: tuple[str, ...] = field(default=(), init=False)

    def load(self) -> pd.DataFrame:
        from src.bms.io.load_oxford import (
            load_oxford_mat,
            oxford_capacity_loss,
            summarize_oxford_cycles,
        )

        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: no such file: {path}. The Oxford archive is "
                f"gitignored; see configs/dataset_sources.yaml. Note that the "
                f"commensurability screen rates nasa -> oxford as MARGINAL "
                f"(internal resistance only), so this is a chemistry check "
                f"rather than a transfer target."
            )

        telemetry, report = load_oxford_mat(path)
        summary = summarize_oxford_cycles(telemetry)
        summary = oxford_capacity_loss(summary)

        if self.cohort_map:
            labels = summary["cell_id"].map(dict(self.cohort_map))
            unmapped = sorted(summary.loc[labels.isna(), "cell_id"].unique())
            if unmapped:
                raise ValueError(
                    f"{self.name}: cohort_map has no entry for {unmapped}."
                )
            summary["cohort"] = labels

        self.load_warnings = report.warnings + (
            f"capacity read as {report.capacity_units_detected}",
        )
        return summary


@dataclass
class NasaFrameLoader:
    """The NASA cycle-level frame this project already has on disk.

    Not a raw-archive parser: `reports/metrics/continuous_model_training_data.csv`
    is the derived cycle-level table, tracked in git because the 7.2M-row
    source is gitignored and not redistributable. It is registered here so the
    three datasets are screened and calibrated through one interface.
    """

    name: str = "nasa"
    path: Path | str = Path("reports/metrics/continuous_model_training_data.csv")

    def load(self) -> pd.DataFrame:
        path = Path(self.path)
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: no such file: {path}. This file is tracked in "
                f"git and cannot be regenerated from a clean checkout — see "
                f"the README's note on what is and isn't tracked."
            )
        return pd.read_csv(path)


def default_registry(
    calce_dir: Path | str = Path("data/raw/calce"),
    oxford_path: Path | str = Path(
        "data/raw/oxford/Oxford_Battery_Degradation_Dataset_1.mat"
    ),
    nasa_path: Path | str = Path(
        "reports/metrics/continuous_model_training_data.csv"
    ),
):
    """A registry with all three datasets registered.

    Every dataset is registered whether or not its files are present. A loader
    for absent data raises `FileNotFoundError` on `load`, which
    `DatasetRegistry.assess` converts into a blocker with the reason attached —
    so `assess_all()` reports "not downloaded" as a distinct state rather than
    the dataset simply not appearing in the table.
    """
    from src.bms.adaptive.datasets import DatasetRegistry

    registry = DatasetRegistry()
    registry.register(NasaFrameLoader(path=nasa_path))
    registry.register(CalceCyclingLoader(base_dir=calce_dir))
    registry.register(OxfordLoader(path=oxford_path))
    return registry
