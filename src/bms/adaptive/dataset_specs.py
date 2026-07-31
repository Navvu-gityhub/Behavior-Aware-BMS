"""Declarative dataset specifications.

Adding a dataset should require configuration, not code. A `DatasetSpec` states
where a dataset's columns live, what its experiment varied, and what it held
constant — so the commensurability screen in `commensurability.py` can run
against published metadata *before* anyone downloads gigabytes and writes a
parser.

Why documented variation profiles are worth encoding
----------------------------------------------------
Batch 7 established that NASA and Stanford/Severson vary along orthogonal axes:
NASA varies ambient temperature across nine protocols while holding
`fast_charge_duration` at identically zero, and Severson holds temperature at
30 C in a chamber while varying charge policy from 3.6C to 6C. That made a
temperature-model transfer between them structurally under-determined.

That conclusion was reachable from each dataset's published description. It did
not need the files. `VariationProfile` therefore records, per dataset and per
axis, whether the experiment *varied* that axis, held it *fixed*, or did not
record it at all — and `predict_transfer_feasibility` reports what the
commensurability screen will conclude, before the download.

This is a screening aid and is labelled as one. It reflects what the dataset
authors documented, which is not always what the files contain, so it never
substitutes for `assess_commensurability` on loaded data. Its job is to stop
effort being spent on a transfer that published metadata already rules out.

The CALCE finding this makes visible
------------------------------------
CALCE CS2 (15 LCO prismatic cells) were all cycled at room temperature, around
23 C. So a NASA temperature model transferred to CS2 hits the same wall as
Stanford: the predictor does not vary on the target.

CALCE CX2 is different in one specific respect. Most CX2 cells were also room
temperature, but CX2_4 was cycled across 25, 35, 45 and 55 C with separate
thermocouple data. That single cell is the only CALCE unit with a thermal axis
comparable to NASA's — which makes it the one scientifically admissible CALCE
target for a temperature model, and also means any such test is n=1 and cannot
support a cell-level generalisation claim. Both facts are recorded on the spec
so neither gets lost.

What CS2 does vary is depth of discharge and discharge rate, and NASA's
`deep_discharge_duration` is far from constant (std 430 over a range of
77-3768). Depth of discharge, not temperature, is the axis on which a
NASA-to-CALCE transfer is most likely to be well posed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import pandas as pd


class Variation(str, Enum):
    """How a dataset's experiment treated one physical axis."""

    VARIED = "varied"          # an experimental variable
    FIXED = "fixed"            # deliberately held constant
    INCIDENTAL = "incidental"  # recorded, moves only as a side effect
    ABSENT = "absent"          # not recorded at all

    @property
    def can_fit_a_coefficient(self) -> bool:
        """Enough variation on the source side to estimate a slope."""
        return self in (Variation.VARIED, Variation.INCIDENTAL)

    @property
    def can_receive_a_coefficient(self) -> bool:
        """Enough variation on the target side for a slope to produce spread.

        `INCIDENTAL` is admitted but is the marginal case: Severson's cell
        temperature moves by up to about 10 C from self-heating, which is real
        variation but an order of magnitude below NASA's designed 4-43 C spread.
        A transfer resting on it should be reported as marginal, which is what
        `predict_transfer_feasibility` does.
        """
        return self in (Variation.VARIED, Variation.INCIDENTAL)


# The physical axes a behaviour-aware model might use. Named independently of
# any one dataset's column names so specs can be compared across datasets.
class Axis(str, Enum):
    AMBIENT_TEMPERATURE = "ambient_temperature"
    CHARGE_RATE = "charge_rate"
    DISCHARGE_RATE = "discharge_rate"
    DEPTH_OF_DISCHARGE = "depth_of_discharge"
    CUTOFF_VOLTAGE = "cutoff_voltage"
    INTERNAL_RESISTANCE = "internal_resistance"


@dataclass(frozen=True)
class VariationProfile:
    """What one dataset's experiment varied, per axis, as documented."""

    axes: Mapping[Axis, Variation]
    note: str = ""

    def get(self, axis: Axis) -> Variation:
        return self.axes.get(axis, Variation.ABSENT)


@dataclass(frozen=True)
class DatasetSpec:
    """Everything needed to load and screen a dataset, declared not coded."""

    name: str
    description: str
    # Source column name -> unified schema column name.
    column_map: Mapping[str, str]
    variation: VariationProfile
    # Cells, if known from the publication. 0 means unknown.
    n_cells: int = 0
    nominal_capacity_ah: float | None = None
    chemistry: str = ""
    citation: str = ""
    # Facts a user must see before drawing conclusions from this dataset.
    caveats: tuple[str, ...] = ()

    def rename(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the declared column mapping, leaving unmapped columns alone."""
        present = {src: dst for src, dst in self.column_map.items() if src in data.columns}
        return data.rename(columns=present)

    def missing_columns(self, data: pd.DataFrame) -> tuple[str, ...]:
        """Declared source columns absent from a loaded frame."""
        return tuple(src for src in self.column_map if src not in data.columns)

    def render(self) -> str:
        lines = [f"{self.name}: {self.description}"]
        if self.n_cells:
            lines.append(f"  cells: {self.n_cells}")
        if self.chemistry:
            lines.append(f"  chemistry: {self.chemistry}")
        lines.append("  documented variation:")
        for axis, variation in self.variation.axes.items():
            lines.append(f"    {axis.value:<22} {variation.value}")
        if self.variation.note:
            lines.append(f"  note: {self.variation.note}")
        for caveat in self.caveats:
            lines.append(f"  CAVEAT: {caveat}")
        if self.citation:
            lines.append(f"  cite: {self.citation}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feasibility prediction from published metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisVerdict:
    axis: Axis
    source_variation: Variation
    target_variation: Variation
    usable: bool
    marginal: bool
    reason: str


@dataclass(frozen=True)
class FeasibilityPrediction:
    """What the commensurability screen is expected to conclude, pre-download."""

    source: str
    target: str
    verdicts: tuple[AxisVerdict, ...]

    @property
    def usable_axes(self) -> tuple[Axis, ...]:
        return tuple(v.axis for v in self.verdicts if v.usable)

    @property
    def marginal_axes(self) -> tuple[Axis, ...]:
        return tuple(v.axis for v in self.verdicts if v.usable and v.marginal)

    @property
    def feasible(self) -> bool:
        return bool(self.usable_axes)

    def __bool__(self) -> bool:
        return self.feasible

    @property
    def status(self) -> str:
        if not self.feasible:
            return "PREDICTED_NOT_FEASIBLE"
        if len(self.marginal_axes) == len(self.usable_axes):
            return "PREDICTED_MARGINAL"
        return "PREDICTED_FEASIBLE"

    def render(self) -> str:
        lines = [
            f"{self.source} -> {self.target}: {self.status}",
            "  (predicted from published metadata; confirm with "
            "assess_commensurability on loaded data)",
        ]
        for verdict in self.verdicts:
            mark = "USABLE" if verdict.usable else "blocked"
            if verdict.usable and verdict.marginal:
                mark = "MARGINAL"
            lines.append(
                f"    {verdict.axis.value:<22} {mark:<8} "
                f"{verdict.source_variation.value} -> {verdict.target_variation.value}"
            )
            lines.append(f"      {verdict.reason}")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "source": self.source,
                "target": self.target,
                "axis": v.axis.value,
                "source_variation": v.source_variation.value,
                "target_variation": v.target_variation.value,
                "usable": v.usable,
                "marginal": v.marginal,
                "reason": v.reason,
            }
            for v in self.verdicts
        ])


def predict_transfer_feasibility(
    source: DatasetSpec,
    target: DatasetSpec,
    axes: Sequence[Axis] | None = None,
) -> FeasibilityPrediction:
    """Predict, from published metadata, whether a transfer is well posed.

    Cheap and approximate by design: it is meant to be run before committing to
    a download, and it never replaces measuring the loaded data.
    """
    candidate_axes = list(axes) if axes is not None else sorted(
        set(source.variation.axes) | set(target.variation.axes),
        key=lambda a: a.value,
    )

    verdicts: list[AxisVerdict] = []
    for axis in candidate_axes:
        source_variation = source.variation.get(axis)
        target_variation = target.variation.get(axis)

        if not source_variation.can_fit_a_coefficient:
            verdicts.append(AxisVerdict(
                axis, source_variation, target_variation, False, False,
                f"{source.name} records this axis as '{source_variation.value}', "
                f"so no coefficient can be fitted for it.",
            ))
            continue
        if not target_variation.can_receive_a_coefficient:
            verdicts.append(AxisVerdict(
                axis, source_variation, target_variation, False, False,
                f"{target.name} records this axis as '{target_variation.value}', "
                f"so a fitted coefficient would have nothing to act on and "
                f"would function as an intercept.",
            ))
            continue

        marginal = (
            source_variation is Variation.INCIDENTAL
            or target_variation is Variation.INCIDENTAL
        )
        reason = (
            f"Varied in {source.name} and {target.name}."
            if not marginal else
            f"One side records this axis only incidentally, so the usable "
            f"range is likely far narrower than the fitted range. Treat any "
            f"result as marginal and check the measured spread ratio."
        )
        verdicts.append(AxisVerdict(
            axis, source_variation, target_variation, True, marginal, reason,
        ))

    return FeasibilityPrediction(source.name, target.name, tuple(verdicts))


# ---------------------------------------------------------------------------
# The specifications themselves
# ---------------------------------------------------------------------------

NASA_SPEC = DatasetSpec(
    name="nasa",
    description=(
        "NASA Ames PCoE randomised battery usage and cycling data, as "
        "preprocessed into this project's cycle-level frame."
    ),
    column_map={},  # already in unified schema by this stage
    n_cells=34,
    chemistry="LCO 18650",
    variation=VariationProfile(
        axes={
            Axis.AMBIENT_TEMPERATURE: Variation.VARIED,
            Axis.DISCHARGE_RATE: Variation.VARIED,
            Axis.CUTOFF_VOLTAGE: Variation.VARIED,
            Axis.DEPTH_OF_DISCHARGE: Variation.VARIED,
            Axis.CHARGE_RATE: Variation.FIXED,
            Axis.INTERNAL_RESISTANCE: Variation.INCIDENTAL,
        },
        note=(
            "Nine protocols spanning roughly 4-43 C ambient. Charge rate is "
            "effectively fixed: fast_charge_duration is identically zero across "
            "all 2,682 cycle-level observations."
        ),
    ),
    caveats=(
        "Nine sub-protocols; pooled analysis obscures cohort effects.",
        "fast_charge_duration is degenerate and cannot support a charge-rate term.",
    ),
    citation="NASA Ames Prognostics Center of Excellence battery data repository",
)


CALCE_CS2_SPEC = DatasetSpec(
    name="calce_cs2",
    description=(
        "CALCE CS2: 15 LCO prismatic cells, grouped Type-1 to Type-6, varying "
        "depth and range of partial charge/discharge and C-rate."
    ),
    column_map={
        # Arbin cycler export column names.
        "Cycle_Index": "cycle",
        "Test_Time(s)": "test_time_s",
        "Current(A)": "current_a",
        "Voltage(V)": "voltage_v",
        "Discharge_Capacity(Ah)": "capacity_ah",
        "Charge_Capacity(Ah)": "charge_capacity_ah",
        "Internal_Resistance(Ohm)": "internal_resistance_ohm",
        "Temperature (C)_1": "temperature_c",
    },
    n_cells=15,
    nominal_capacity_ah=1.1,
    chemistry="LCO prismatic",
    variation=VariationProfile(
        axes={
            Axis.AMBIENT_TEMPERATURE: Variation.FIXED,
            Axis.DEPTH_OF_DISCHARGE: Variation.VARIED,
            Axis.DISCHARGE_RATE: Variation.VARIED,
            Axis.CUTOFF_VOLTAGE: Variation.VARIED,
            Axis.CHARGE_RATE: Variation.FIXED,
            Axis.INTERNAL_RESISTANCE: Variation.INCIDENTAL,
        },
        note=(
            "Cycled at room temperature, about 23 C. Charging was a standard "
            "0.5C CC-CV to 4.2 V for all cells, so charge rate is fixed. The "
            "experimental axes are depth of discharge, discharge rate and "
            "cutoff voltage."
        ),
    ),
    caveats=(
        "Room temperature only: cannot receive a NASA temperature coefficient.",
        "Multiple date-named files per cell require concatenation and "
        "cycle-index reconciliation before use.",
        "CS2_8 and CS2_21 used a CADEX tester and are .txt, not Arbin Excel.",
        "Type-1 cells are 0.9 Ah and Type-2 onward are 1.1 Ah; capacity must be "
        "normalised per cell before cross-cell comparison.",
    ),
    citation="CALCE Battery Group, University of Maryland (doi:10.21227/w9rg-7173)",
)


CALCE_CX2_SPEC = DatasetSpec(
    name="calce_cx2",
    description="CALCE CX2: 12 LCO prismatic cells, 1.35 Ah rated.",
    column_map=dict(CALCE_CS2_SPEC.column_map),
    n_cells=12,
    nominal_capacity_ah=1.35,
    chemistry="LCO prismatic",
    variation=VariationProfile(
        axes={
            Axis.AMBIENT_TEMPERATURE: Variation.FIXED,
            Axis.DEPTH_OF_DISCHARGE: Variation.VARIED,
            Axis.DISCHARGE_RATE: Variation.VARIED,
            Axis.CUTOFF_VOLTAGE: Variation.VARIED,
            Axis.CHARGE_RATE: Variation.FIXED,
            Axis.INTERNAL_RESISTANCE: Variation.INCIDENTAL,
        },
        note=(
            "Same format and largely the same protocols as CS2. The exception "
            "is CX2_4, which was cycled across several temperatures and is "
            "specified separately as calce_cx2_4_thermal."
        ),
    ),
    caveats=(
        "Room temperature for all cells except CX2_4.",
        "CX2_4 and CX2_31 used a CADEX tester and are .txt, not Arbin Excel.",
    ),
    citation="CALCE Battery Group, University of Maryland (doi:10.21227/w9rg-7173)",
)


CALCE_CX2_4_THERMAL_SPEC = DatasetSpec(
    name="calce_cx2_4_thermal",
    description=(
        "CALCE CX2_4 alone: the one CALCE cell cycled across a range of "
        "temperatures (25, 35, 45 and 55 C), with separate thermocouple data."
    ),
    column_map=dict(CALCE_CS2_SPEC.column_map),
    n_cells=1,
    nominal_capacity_ah=1.35,
    chemistry="LCO prismatic",
    variation=VariationProfile(
        axes={
            Axis.AMBIENT_TEMPERATURE: Variation.VARIED,
            Axis.DEPTH_OF_DISCHARGE: Variation.VARIED,
            Axis.DISCHARGE_RATE: Variation.VARIED,
            Axis.CHARGE_RATE: Variation.FIXED,
            Axis.INTERNAL_RESISTANCE: Variation.INCIDENTAL,
        },
        note=(
            "The only CALCE unit with a thermal axis comparable to NASA's, and "
            "therefore the only scientifically admissible CALCE target for a "
            "temperature model."
        ),
    ),
    caveats=(
        "n=1 cell. A transfer test here can characterise the temperature "
        "relationship on one unit but cannot support any cell-level "
        "generalisation claim, because there is no between-cell variance to "
        "estimate.",
        "Thermocouple data ships in a separate Temperature folder and must be "
        "joined to the cycler log on test time.",
        "CX2_4 was recorded on a CADEX tester in .txt format, so the Arbin "
        "column map does not apply directly.",
    ),
    citation="CALCE Battery Group, University of Maryland (doi:10.21227/w9rg-7173)",
)


STANFORD_SEVERSON_SPEC = DatasetSpec(
    name="stanford_severson",
    description=(
        "Severson et al. 2019: 124 A123 LFP/graphite 18650 cells cycled to "
        "failure under 72 one- or two-step fast-charging policies."
    ),
    column_map={
        "cycle_number": "cycle",
        "QDischarge": "capacity_ah",
        "QCharge": "charge_capacity_ah",
        "IR": "internal_resistance_ohm",
        "Tavg": "avg_temp",
        "Tmin": "min_temp",
        "Tmax": "max_temp",
        "chargetime": "charge_time_min",
    },
    n_cells=124,
    nominal_capacity_ah=1.1,
    chemistry="LFP/graphite 18650 (A123 APR18650M1A)",
    variation=VariationProfile(
        axes={
            Axis.CHARGE_RATE: Variation.VARIED,
            Axis.AMBIENT_TEMPERATURE: Variation.FIXED,
            Axis.DISCHARGE_RATE: Variation.FIXED,
            Axis.DEPTH_OF_DISCHARGE: Variation.FIXED,
            Axis.INTERNAL_RESISTANCE: Variation.INCIDENTAL,
        },
        note=(
            "Forced-convection chamber set to 30 C, so ambient temperature is "
            "held. Cell temperature is recorded and moves by up to about 10 C "
            "from self-heating, which is incidental rather than experimental. "
            "All cells discharge identically at 4C to 2.0 V; the experimental "
            "axis is charge policy, 3.6C to 6C."
        ),
    ),
    caveats=(
        "Chamber-controlled temperature: cannot receive a NASA ambient "
        "temperature coefficient. Cell temperature is available but its spread "
        "is roughly an order of magnitude below NASA's designed range.",
        "NASA cannot fit a charge-rate coefficient, so Severson's own "
        "experimental axis is unavailable as a transfer feature.",
        "Cycle 1 is excluded upstream for sampling-rate reasons.",
        "Batch 1 channels 4 and 8 have no data; thermocouples for channels 15 "
        "and 16 were swapped.",
    ),
    citation=(
        "Severson et al., Data-driven prediction of battery cycle life before "
        "capacity degradation, Nature Energy 4, 383-391 (2019)"
    ),
)


REGISTRY: Mapping[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        NASA_SPEC,
        CALCE_CS2_SPEC,
        CALCE_CX2_SPEC,
        CALCE_CX2_4_THERMAL_SPEC,
        STANFORD_SEVERSON_SPEC,
    )
}


def get_spec(name: str) -> DatasetSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset spec '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def feasibility_matrix(
    source_name: str = "nasa",
    target_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Predicted transfer feasibility from one source to every known target."""
    source = get_spec(source_name)
    targets = target_names if target_names is not None else [
        name for name in REGISTRY if name != source_name
    ]
    frames = [
        predict_transfer_feasibility(source, get_spec(name)).to_frame()
        for name in targets
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
