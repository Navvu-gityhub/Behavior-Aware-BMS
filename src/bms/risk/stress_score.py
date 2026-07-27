"""Degradation risk scoring: row-level stress score and battery-level risk assessment.

Ported from `notebooks/n05_T1v1.ipynb` (battery-level risk) and the stress-score
convention assumed (but never defined in code) upstream of
`notebooks/04_feature_refinement.ipynb`. Both are rule-based heuristics with
hand-chosen weights and thresholds — they are NOT fit to data and have not
been validated against measured degradation outcomes. Treat them as a
transparent, auditable starting point, not a scientific result.

Two independent scoring passes exist in this codebase and are deliberately
kept as two functions rather than merged:

- `compute_stress_score`: per-telemetry-row score (0-100), used as an input
  to the rolling/aggregate features in `features.behavior_features`.
- `compute_risk_assessment`: per-battery score (0-100) computed from the
  battery summary table, independent of the row-level stress score.

These use overlapping signals (temperature, deep discharge, fast charging)
but different weights and thresholds. That overlap is a known limitation
(see docs/risk_rules.md) — collapsing them into one score is future work
that requires deciding, with data, whether row-level and fleet-level risk
should in fact be the same quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

REQUIRED_ROW_COLUMNS = ("current_a", "temperature_c", "soc")
REQUIRED_SUMMARY_COLUMNS = (
    "avg_stress",
    "avg_temp",
    "deep_discharge_duration",
    "fast_charge_duration",
    "aggressive_discharge_count",
    "avg_soc",
)


# ---------------------------------------------------------------------------
# Row-level rule-based stress score
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StressScoreWeights:
    """Point values added to the 0-100 row-level stress score per triggered flag.

    Chosen to be roughly ordered by assumed severity (aggressive discharge and
    sustained high temperature are the most commonly cited drivers of
    accelerated lithium-ion aging in the literature; SOC extremes are
    included but weighted lowest). Not calibrated — see module docstring.
    """

    aggressive_discharge: float = 30.0
    high_temp: float = 25.0
    fast_charge: float = 20.0
    deep_discharge: float = 15.0
    high_soc: float = 10.0


def compute_stress_score(df: pd.DataFrame, weights: StressScoreWeights = StressScoreWeights(), rated_capacity_ah: float = 2.0, c_rate_threshold: float = 1.0) -> pd.Series:
    """Rule-based row-level stress score in [0, 100].

    Expects `current_a`, `temperature_c`, `soc` columns (unified schema).
    If behavior flag columns (`aggressive_discharge_event`, `high_temp_flag`,
    `fast_charge_flag`, `deep_discharge_flag`, `high_soc_flag`) already exist
    (e.g. from `features.behavior_features.compute_behavior_flags`), those are
    reused instead of being recomputed, so flag thresholds stay in one place.
    The `rated_capacity_ah`/`c_rate_threshold` fallback thresholds below only
    apply if flags haven't already been computed — see behavior_features.py
    for why C-rate rather than a flat Amp threshold.
    """
    missing = [c for c in REQUIRED_ROW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"compute_stress_score: missing required columns {missing}")

    c_rate = df["current_a"] / rated_capacity_ah
    aggressive = df["aggressive_discharge_event"] if "aggressive_discharge_event" in df.columns else (c_rate < -c_rate_threshold)
    high_temp = df["high_temp_flag"] if "high_temp_flag" in df.columns else (df["temperature_c"] > 40.0)
    fast_charge = df["fast_charge_flag"] if "fast_charge_flag" in df.columns else (c_rate > c_rate_threshold)
    deep_discharge = df["deep_discharge_flag"] if "deep_discharge_flag" in df.columns else (df["soc"] < 20.0)
    high_soc = df["high_soc_flag"] if "high_soc_flag" in df.columns else (df["soc"] > 90.0)

    score = (
        aggressive.astype(float) * weights.aggressive_discharge
        + high_temp.astype(float) * weights.high_temp
        + fast_charge.astype(float) * weights.fast_charge
        + deep_discharge.astype(float) * weights.deep_discharge
        + high_soc.astype(float) * weights.high_soc
    )
    return score.clip(0, 100)


# ---------------------------------------------------------------------------
# Battery-level risk assessment (ported from notebooks/n05_T1v1.ipynb)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskThresholds:
    """Battery-level risk score bands.

    Aligned to `docs/risk_rules.md`, which is what the original notebook
    (n05_T1v1) actually implemented. The README previously stated a
    different 3-band scale (Low 0-30 / Medium 31-60 / High 61-100) that did
    not match any implemented code — that was a documentation bug, now
    corrected to match this 4-band scale.
    """

    low_max: int = 39
    medium_max: int = 59
    high_max: int = 79
    # 80-100 -> CRITICAL


def _risk_level(score: float, t: RiskThresholds) -> str:
    if score > t.high_max:
        return "CRITICAL"
    if score > t.medium_max:
        return "HIGH"
    if score > t.low_max:
        return "MEDIUM"
    return "LOW"


def _risk_reason(row: pd.Series) -> str:
    reasons = []
    if row["avg_stress"] > 50:
        reasons.append("high stress")
    if row["avg_temp"] > 30:
        reasons.append("high temperature")
    if row["deep_discharge_duration"] > 20:
        reasons.append("deep discharge")
    if row["fast_charge_duration"] > 20:
        reasons.append("frequent fast charging")
    if row["aggressive_discharge_count"] > 100:
        reasons.append("aggressive discharge")
    if row["avg_soc"] > 80:
        reasons.append("high SOC exposure")
    if row["avg_soc"] < 20:
        reasons.append("low SOC exposure")
    if not reasons:
        reasons.append("healthy usage")
    return ", ".join(reasons)


def _risk_recommendation(level: str) -> str:
    return {
        "CRITICAL": "Immediate inspection recommended. Avoid fast charging and maintain 20-80 SOC.",
        "HIGH": "Reduce fast charging frequency and follow 20-80 charging rule.",
        "MEDIUM": "Avoid deep discharge and prolonged high SOC exposure.",
        "LOW": "Battery behavior is healthy.",
    }[level]


def compute_risk_assessment(battery_summary: pd.DataFrame, thresholds: RiskThresholds = RiskThresholds()) -> pd.DataFrame:
    """Compute battery-level risk_score, risk_level, risk_reason, recommended_action.

    Expects a per-battery summary table with the columns produced by
    `features.behavior_features.summarize_batteries`.
    """
    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in battery_summary.columns]
    if missing:
        raise ValueError(f"compute_risk_assessment: missing required columns {missing}")

    nulls = {c: int(battery_summary[c].isna().sum()) for c in REQUIRED_SUMMARY_COLUMNS if battery_summary[c].isna().any()}
    if nulls:
        # NumPy comparisons against NaN (e.g. `NaN >= 40`) evaluate to False, which
        # would otherwise silently fall through to the LOWEST-risk bucket for a
        # battery whose data is actually missing, not actually healthy. Caught
        # by src/bms/io/load_calce_capacity.py's data (no temperature channel
        # recorded) — see docs/calce_dataset_note.md.
        raise ValueError(
            f"compute_risk_assessment: NaN values in required columns {nulls} — "
            "would silently score as low-risk rather than reflect missing data. "
            "Impute or explicitly exclude these rows/columns before scoring."
        )

    out = battery_summary.copy()

    score = np.zeros(len(out))
    score += np.where(out["avg_stress"] >= 70, 30, np.where(out["avg_stress"] >= 50, 20, 10))
    score += np.where(out["avg_temp"] >= 40, 25, np.where(out["avg_temp"] >= 30, 15, 5))
    score += np.where(out["deep_discharge_duration"] >= 100, 20, np.where(out["deep_discharge_duration"] >= 20, 10, 5))
    score += np.where(out["fast_charge_duration"] >= 100, 15, np.where(out["fast_charge_duration"] >= 20, 8, 2))
    score += np.where(out["aggressive_discharge_count"] >= 500, 15, np.where(out["aggressive_discharge_count"] >= 100, 8, 2))
    score += np.where((out["avg_soc"] > 80) | (out["avg_soc"] < 20), 10, 0)

    out["risk_score"] = np.clip(score, 0, 100)
    out["risk_level"] = out["risk_score"].apply(lambda s: _risk_level(s, thresholds))
    out["risk_reason"] = out.apply(_risk_reason, axis=1)
    out["recommended_action"] = out["risk_level"].apply(_risk_recommendation)
    return out


# ---------------------------------------------------------------------------
# Optional ML scorer
# ---------------------------------------------------------------------------

def try_ml_stress_score(df: pd.DataFrame, model_path: str, feature_columns: Optional[list[str]] = None) -> Optional[pd.Series]:
    """Attempt to score rows with the pre-trained model in `models/`.

    Returns None (with no exception) if the model can't be loaded or its
    expected feature set doesn't match `feature_columns`/`df`. This is
    intentionally conservative: `models/stress_score_rf_sample_v1.joblib`
    has no recorded training data, feature list, or evaluation metrics in
    this repository, so silently trusting its output would be scientifically
    indefensible. Wire this up for real once the model's provenance
    (training set, features, validation metrics) is documented.
    """
    try:
        import joblib
    except ImportError:
        return None

    try:
        model = joblib.load(model_path)
    except (FileNotFoundError, OSError, ValueError):
        return None

    expected = feature_columns or list(getattr(model, "feature_names_in_", []))
    if not expected or any(c not in df.columns for c in expected):
        return None

    try:
        preds = model.predict(df[expected])
    except Exception:
        return None

    return pd.Series(np.clip(preds, 0, 100), index=df.index)
