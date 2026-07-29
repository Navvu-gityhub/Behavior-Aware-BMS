"""Audit whether the rule-based scores' thresholds are reachable in real data.

WHY THIS EXISTS
---------------
`docs/calibration_report.md` Section 5 observed that the pipeline's scores
"have almost no resolution" — nearly every NASA battery received the same
risk score and health index. That was reported as an empirical fact without
a mechanism.

This script supplies the mechanism, and it turns out to be simple and
damning: **several of the hand-chosen threshold cut points sit outside the
range the real data ever occupies.** A term whose feature never crosses its
cut points is a constant. A constant contributes nothing to between-battery
variation, cannot be validated against anything, and — critically — cannot
be falsified by the calibration exercise that was supposed to test it.

This reframes the earlier "no significant correlation" result. The scores
did not fail to correlate with fade because the *hypothesis* was wrong;
they failed because most of the score was a constant, and a constant cannot
correlate with anything. Those are very different diagnoses with very
different fixes, and the distinction was not visible before this audit.

WHAT IT MEASURES
----------------
For every term in `RISK_TERMS` and `HEALTH_TERMS`, over the real NASA
calibration data:

  * the observed range of the term's input feature;
  * the fraction of observations landing in each penalty band;
  * whether the term is DEGENERATE (one band captures everything, so the
    term is constant), PARTIAL (some bands unreachable), or ACTIVE.

A term is only meaningfully calibrated if it is ACTIVE. Anything else is a
weight that was chosen, never tested, and cannot be defended by pointing at
the calibration work.

Run:  python scripts/audit_threshold_reachability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.health.health_index import HEALTH_TERMS
from src.bms.risk.stress_score import RISK_TERMS

TRAINING_DATA = Path("reports/metrics/continuous_model_training_data.csv")
CALIBRATION_SUMMARY = Path("reports/metrics/calibration_merged.csv")
METRICS_DIR = Path("reports/metrics")


def audit_terms(terms, data: pd.DataFrame, score_name: str) -> pd.DataFrame:
    rows = []
    for term in terms:
        if term.feature not in data.columns:
            continue
        feature = data[term.feature].dropna()
        if feature.empty:
            continue

        contributions = term.contribution(feature)
        band_counts = contributions.value_counts(normalize=True).sort_index(ascending=False)

        # The full set of penalty levels this term can emit, probed across a
        # wide synthetic sweep of its input. Comparing this to the levels
        # actually observed is what identifies unreachable bands.
        probe = pd.Series(np.linspace(-50, 5000, 20001))
        possible = sorted(set(np.round(term.contribution(probe).to_numpy(), 6)))
        observed = sorted(set(np.round(contributions.to_numpy(), 6)))
        unreachable = [p for p in possible if p not in observed]

        if len(observed) == 1:
            status = "DEGENERATE"
        elif unreachable:
            status = "PARTIAL"
        else:
            status = "ACTIVE"

        rows.append(
            {
                "score": score_name,
                "term": term.name,
                "feature": term.feature,
                "feature_min": float(feature.min()),
                "feature_max": float(feature.max()),
                "penalty_levels_possible": len(possible),
                "penalty_levels_observed": len(observed),
                "unreachable_penalty_levels": ", ".join(str(u) for u in unreachable) or "none",
                "observed_penalty_min": float(contributions.min()),
                "observed_penalty_max": float(contributions.max()),
                "penalty_std": float(contributions.std()),
                "modal_band_share": float(band_counts.iloc[0]),
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    if not TRAINING_DATA.exists():
        raise SystemExit(f"{TRAINING_DATA} not found; run the NASA calibration scripts first.")

    cycle_level = pd.read_csv(TRAINING_DATA)
    print(f"Cycle-level NASA data: {len(cycle_level)} observations, "
          f"{cycle_level['cell_id'].nunique()} cells\n")

    risk_audit = audit_terms(RISK_TERMS, cycle_level, "risk_score")
    health_audit = audit_terms(HEALTH_TERMS, cycle_level, "health_index")
    audit = pd.concat([risk_audit, health_audit], ignore_index=True)
    audit.to_csv(METRICS_DIR / "threshold_reachability_audit.csv", index=False)

    print("=== Threshold reachability, cycle-level NASA data ===")
    print(
        audit[
            ["score", "term", "feature_min", "feature_max",
             "penalty_levels_possible", "penalty_levels_observed",
             "modal_band_share", "status"]
        ].to_string(index=False)
    )

    # How much of each score is actually variable, per battery?
    if CALIBRATION_SUMMARY.exists():
        summary = pd.read_csv(CALIBRATION_SUMMARY)
        print("\n=== Per-battery score decomposition (33 NASA batteries) ===")
        contrib_rows = []
        for term in RISK_TERMS:
            if term.feature not in summary.columns:
                continue
            c = term.contribution(summary[term.feature])
            contrib_rows.append(
                {
                    "term": term.name,
                    "mean_points": float(c.mean()),
                    "std_points": float(c.std()),
                    "distinct_values": int(c.nunique()),
                    "is_constant_across_fleet": bool(c.nunique() == 1),
                }
            )
        contrib = pd.DataFrame(contrib_rows)
        contrib.to_csv(METRICS_DIR / "risk_term_variability.csv", index=False)
        print(contrib.to_string(index=False))

        constant_points = contrib.loc[contrib["is_constant_across_fleet"], "mean_points"].sum()
        total_points = contrib["mean_points"].sum()
        print(
            f"\nOf a mean risk score of {total_points:.1f} points, "
            f"{constant_points:.1f} points ({100 * constant_points / total_points:.0f}%) "
            f"come from terms that are IDENTICAL for every battery in the dataset "
            f"and therefore carry zero discriminative information."
        )

    degenerate = audit[audit["status"] == "DEGENERATE"]
    if not degenerate.empty:
        print("\n=== DEGENERATE terms (constant in real data, cannot be calibrated) ===")
        for _, r in degenerate.iterrows():
            print(
                f"  {r['score']}.{r['term']:22s} feature '{r['feature']}' spans "
                f"[{r['feature_min']:.2f}, {r['feature_max']:.2f}] — never leaves one band."
            )

    print(f"\nWrote {METRICS_DIR / 'threshold_reachability_audit.csv'}")


if __name__ == "__main__":
    main()
