"""health_index_v2: a fitted, cross-validated alternative to health.health_index.

Built from `scripts/fit_continuous_health_model.py` against the NASA
calibration data (see docs/calibration_report.md Section 10 for full
methodology and honest results). Read the limitations below before using
this for anything — it is a genuine improvement in *methodology* over the
original bucketed heuristic (real coefficients, real out-of-sample
validation) but NOT a strong predictive model, and that distinction matters.

WHAT THIS MODEL IS: an OLS fit of per-cycle capacity loss on a trailing
5-cycle average temperature, with a per-cohort intercept fit on 9 NASA
experimental cohorts (34 batteries, 2,682 cycle-level observations).
`trailing_avg_temp` was the only input feature from Stages 1-2 that showed
a statistically significant, correctly-signed, cross-cohort-consistent
relationship with real capacity loss — see calibration_report.md Sections
3-8 for why `avg_stress` and current-based flags were excluded rather than
also included with their (unreliable) fitted weights.

WHAT THIS MODEL IS NOT: accurate at the single-cycle level. Leave-one-
battery-out cross-validation: R-squared = 0.015 in-sample; out-of-sample
median Spearman correlation between predicted and actual per-cycle
capacity loss = 0.052 (24 of 33 held-out batteries positive, i.e. better
than chance but weak). Do not present this model's per-cycle predictions
as reliable; the honest summary is "temperature has a detectable,
directionally-correct effect on fade, but a single validated feature at
single-cycle resolution explains very little of the noise in individual
capacity readings." A longer-horizon target (e.g. cumulative fade over 20+
cycles rather than one cycle) would likely show a stronger R-squared,
since it would average out per-cycle measurement noise — untested here,
noted as a next step.

COHORT INTERCEPTS ARE FIT DATA, NOT PORTABLE CONSTANTS: they absorb
NASA-specific experimental differences (cutoff voltage, discharge current,
etc. — calibration_report.md Section 2) rather than anything about the
input battery. `UNKNOWN_COHORT_INTERCEPT` below is the mean of the 9 fitted
cohort intercepts, used only as a fallback for data with no matching
cohort — it is explicitly a weaker, unvalidated default, not a calibrated
value.
"""

from __future__ import annotations

import pandas as pd

# Coefficients from scripts/fit_continuous_health_model.py, run against
# reports/metrics/health_model_v2_coefficients.txt (NASA cleaned_dataset,
# 34 batteries). Re-run that script and update these if the training data
# changes — they are not derived at runtime, to keep this module fast and
# dependency-light for the main pipeline.
TRAILING_TEMP_COEF = 0.0038  # Ah of capacity loss per degree C of trailing 5-cycle avg temp
INTERCEPT = -0.0209

COHORT_INTERCEPT_ADJUSTMENT = {
    "RT_2A_CC_variedcutoff": -0.0749,
    "RT_SQWAVE_4A_variedcutoff": -0.0841,
    "ELEV43C_4A_CC_variedcutoff": -0.1473,
    "RT_CC_mixed_current": -0.0815,
    "MIXED_24_44C_multiload": -0.1386,
    "COLD4C_multiload": -0.0246,
    "COLD4C_1A": 0.0,  # reference cohort (absorbed into INTERCEPT)
    "COLD4C_2A_flagged": 0.0362,
    "COLD4C_2A": -0.0025,
}
UNKNOWN_COHORT_INTERCEPT = sum(COHORT_INTERCEPT_ADJUSTMENT.values()) / len(COHORT_INTERCEPT_ADJUSTMENT)

# Model quality, reported honestly rather than omitted — see module docstring.
MODEL_QUALITY = {
    "r_squared_in_sample": 0.0150,
    "loocv_median_spearman_rho": 0.052,
    "loocv_batteries_positive_rho": "24/33",
    "loocv_overall_mae_ah": 0.0523,
}


def predict_capacity_loss_per_cycle(trailing_avg_temp: pd.Series, cohort: str | None = None) -> pd.Series:
    """Predict per-cycle capacity loss (Ah) from trailing average temperature.

    `cohort` should be one of COHORT_INTERCEPT_ADJUSTMENT's keys if the
    input data matches a known NASA experimental protocol; otherwise the
    unvalidated fallback average is used and the caller should treat the
    output with additional caution (see module docstring).
    """
    adjustment = COHORT_INTERCEPT_ADJUSTMENT.get(cohort, UNKNOWN_COHORT_INTERCEPT)
    return INTERCEPT + adjustment + TRAILING_TEMP_COEF * trailing_avg_temp
