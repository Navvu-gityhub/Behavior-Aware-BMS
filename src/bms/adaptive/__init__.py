"""Adaptive calibration: governed model promotion across datasets and protocols.

The organising principle of this package is that **the gate is the product.**

A naive adaptive system refits on new data and keeps whatever fits better.
Applied to this project that would be actively harmful: `docs/adr/0002` records
that the fitted health model scores rho=0.841 ranking an unseen cell inside a
known protocol and rho=-0.295 on an unseen protocol. Its apparent skill lives
in fitted per-cohort intercepts. An unguarded retraining loop would get
steadily better at memorising protocols while its reported metrics improved,
which is the failure mode this project spent its calibration effort
diagnosing.

So the default answer here is REJECT. A candidate is promoted only when it
beats a naive baseline out-of-sample on a held-out *protocol*, not merely a
held-out cell, and only when its coefficients are stable across the refit.
"""

from src.bms.adaptive.calibrator import (
    AdaptiveCalibrator,
    CalibrationRun,
    CandidateOutcome,
    CandidateSpec,
    Scored,
    ScoringRefusal,
    linear_candidate,
)
from src.bms.adaptive.commensurability import (
    CommensurabilityReport,
    FeatureVariation,
    assess_commensurability,
    measure_variation,
)
from src.bms.adaptive.cohort import (
    CohortRegistry,
    CohortSpec,
    DriftReport,
    InDistribution,
)
from src.bms.adaptive.dataset_specs import (
    Axis,
    DatasetSpec,
    FeasibilityPrediction,
    Variation,
    VariationProfile,
    feasibility_matrix,
    get_spec,
    predict_transfer_feasibility,
)
from src.bms.adaptive.datasets import (
    CallableDatasetLoader,
    CsvDatasetLoader,
    DatasetManifest,
    DatasetRegistry,
    SuitabilityReport,
    assess_suitability,
)
from src.bms.adaptive.store import (
    Decision,
    ModelStore,
    ModelVersion,
    StabilityError,
)
from src.bms.adaptive.transfer import (
    CompatibilityReport,
    DomainShift,
    TransferResult,
    TransferValidator,
    transfer_summary,
)
from src.bms.adaptive.validation import (
    CrossValidationResult,
    FoldResult,
    Validator,
    Verdict,
)

__all__ = [
    "AdaptiveCalibrator",
    "CalibrationRun",
    "CandidateOutcome",
    "CandidateSpec",
    "Scored",
    "ScoringRefusal",
    "linear_candidate",
    "CommensurabilityReport",
    "FeatureVariation",
    "assess_commensurability",
    "measure_variation",
    "CohortRegistry",
    "CohortSpec",
    "DriftReport",
    "InDistribution",
    "Axis",
    "DatasetSpec",
    "FeasibilityPrediction",
    "Variation",
    "VariationProfile",
    "feasibility_matrix",
    "get_spec",
    "predict_transfer_feasibility",
    "CallableDatasetLoader",
    "CsvDatasetLoader",
    "DatasetManifest",
    "DatasetRegistry",
    "SuitabilityReport",
    "assess_suitability",
    "Decision",
    "ModelStore",
    "ModelVersion",
    "StabilityError",
    "CompatibilityReport",
    "DomainShift",
    "TransferResult",
    "TransferValidator",
    "transfer_summary",
    "CrossValidationResult",
    "FoldResult",
    "Validator",
    "Verdict",
]
