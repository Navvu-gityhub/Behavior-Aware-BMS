"""Registration of the physics-grounded models into the benchmark registry.

Kept separate from `src/bms/physics/`, which owns the model itself. The split
matters: the physics module must be usable without importing the benchmark
machinery, and the benchmark registry must be able to list a method without
committing to how it is implemented.

Two temperature variants are registered because the frame carries two columns
and they answer different questions. `avg_temp` is the cell's mean temperature
during that cycle. `trailing_avg_temp` is a rolling mean over preceding
cycles, which is the better match to the model's physics: SEI growth
integrates thermal exposure over history, so the fade observed at cycle N was
driven by the temperatures the cell has been through, not the one it happened
to be at during the measurement.

Registering both means the comparison is measured rather than argued.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.bms.adaptive.validation import FitFn
from src.bms.benchmarks.registry import BenchmarkMethod, Family, register
from src.bms.physics.arrhenius import build_arrhenius

_CITATION = (
    "Arrhenius power-law capacity fade; cf. Bloom et al., J. Power Sources 101 "
    "(2001) 238-247, and Wang et al., J. Power Sources 196 (2011) 3942-3948."
)


def _variant(temperature_col: str):
    def build(features: Sequence[str], target: str) -> FitFn:
        return build_arrhenius(features, target, temperature_col=temperature_col)
    return build


register(BenchmarkMethod(
    name="arrhenius_avg_temp",
    family=Family.PHYSICS,
    citation=_CITATION,
    build=_variant("avg_temp"),
    default_features=("avg_temp", "cycle"),
    assumes_target="cumulative_fade",
    note=(
        "fade = A*exp(-Ea/RT)*N^z, fitted by OLS in log space. Ea has physical "
        "units and a published reference range, so the fit has an external "
        "check no purely empirical model here has. Assumes a cumulative fade "
        "target; a per-cycle delta target is not what the power law describes."
    ),
))

register(BenchmarkMethod(
    name="arrhenius_trailing_temp",
    family=Family.PHYSICS,
    citation=_CITATION,
    build=_variant("trailing_avg_temp"),
    default_features=("trailing_avg_temp", "cycle"),
    assumes_target="cumulative_fade",
    note=(
        "Same model against trailing mean temperature. Better matched to the "
        "physics — thermally activated side reactions integrate exposure over "
        "history rather than responding to the instantaneous measurement."
    ),
))
