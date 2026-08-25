"""Physics-grounded degradation models.

Distinct from `src/bms/health/` and `src/bms/risk/`, which are rule-based
severity heuristics with hand-chosen weights. The models here have functional
forms taken from degradation physics, and their fitted parameters have
physical units and published reference values to be checked against.

That difference is the point. A fitted linear coefficient can only be
validated against held-out data. An activation energy can additionally be
compared with the literature, which is a second, independent check that no
purely empirical model in this project has.
"""

from src.bms.physics.arrhenius import (
    ADMISSIBLE_EA_RANGE,
    GAS_CONSTANT,
    PLAUSIBLE_EA_RANGE,
    ArrheniusFit,
    IdentifiabilityReport,
    arrhenius_stability,
    assess_identifiability,
    build_arrhenius,
    fit_arrhenius,
    predict_arrhenius,
)
from src.bms.physics.thermal_confound import (
    ThermalBaseline,
    confound_report,
    correct_thermal_measurement,
    thermal_baseline,
)

__all__ = [
    "ADMISSIBLE_EA_RANGE",
    "GAS_CONSTANT",
    "PLAUSIBLE_EA_RANGE",
    "ArrheniusFit",
    "IdentifiabilityReport",
    "ThermalBaseline",
    "arrhenius_stability",
    "assess_identifiability",
    "build_arrhenius",
    "confound_report",
    "correct_thermal_measurement",
    "fit_arrhenius",
    "predict_arrhenius",
    "thermal_baseline",
]
