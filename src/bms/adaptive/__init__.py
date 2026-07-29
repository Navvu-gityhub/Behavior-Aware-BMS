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

from src.bms.adaptive.cohort import (
    CohortRegistry,
    CohortSpec,
    DriftReport,
    InDistribution,
)
from src.bms.adaptive.store import (
    Decision,
    ModelStore,
    ModelVersion,
    StabilityError,
)
from src.bms.adaptive.validation import (
    CrossValidationResult,
    FoldResult,
    Validator,
    Verdict,
)

__all__ = [
    "CohortRegistry",
    "CohortSpec",
    "DriftReport",
    "InDistribution",
    "Decision",
    "ModelStore",
    "ModelVersion",
    "StabilityError",
    "CrossValidationResult",
    "FoldResult",
    "Validator",
    "Verdict",
]
