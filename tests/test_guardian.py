"""Tests for src.bms.guardian.guardian's evidence labeling.

The point of these tests isn't just "does it run" -- it's confirming the
confidence label actually tracks what final_report.md found: temperature
causes get VALIDATED, current-based/health_index causes get HEURISTIC,
mixing them gets MIXED, and existing fields (guardian_report,
primary_causes, etc.) are unchanged by this addition.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from src.bms.guardian.guardian import (
    TEMPERATURE_COEFFICIENT_AH_PER_C,
    generate_guardian_reports,
)


def _row(**overrides):
    base = dict(
        battery_id="B1",
        battery_state="WARNING",
        health_index=40,
        rul_cycles=500,
        avg_temp=25,
        fast_charge_duration=0,
        deep_discharge_duration=0,
    )
    base.update(overrides)
    return base


def test_temperature_only_cause_gets_validated_confidence():
    df = pd.DataFrame([_row(avg_temp=40)])  # only trips the temperature threshold
    out = generate_guardian_reports(df)
    assert out.iloc[0]["primary_causes"] == "high temperature exposure"
    assert out.iloc[0]["evidence_confidence"] == "VALIDATED"
    assert str(TEMPERATURE_COEFFICIENT_AH_PER_C) in out.iloc[0]["evidence_note"]
    assert "final_report.md" in out.iloc[0]["evidence_note"]


def test_current_based_causes_get_heuristic_confidence():
    df = pd.DataFrame([_row(fast_charge_duration=100, deep_discharge_duration=100)])
    out = generate_guardian_reports(df)
    assert out.iloc[0]["evidence_confidence"] == "HEURISTIC"
    assert "did NOT transfer" in out.iloc[0]["evidence_note"] or "did not transfer" in out.iloc[0]["evidence_note"]


def test_mixed_causes_get_mixed_confidence():
    df = pd.DataFrame([_row(avg_temp=40, fast_charge_duration=100)])
    out = generate_guardian_reports(df)
    causes = out.iloc[0]["primary_causes"]
    assert "high temperature exposure" in causes
    assert "frequent fast charging" in causes
    assert out.iloc[0]["evidence_confidence"] == "MIXED"


def test_no_causes_gets_na_confidence():
    df = pd.DataFrame([_row()])  # nothing trips any threshold
    out = generate_guardian_reports(df)
    assert out.iloc[0]["primary_causes"] == "normal battery usage"
    assert out.iloc[0]["evidence_confidence"] == "N/A"


def test_health_index_driven_cause_is_heuristic_not_validated():
    """health_index itself was found to have no significant relationship
    with real fade rate (Section 4.1) -- a battery flagged ONLY because
    health_index > 60 should not read as validated."""
    df = pd.DataFrame([_row(health_index=70)])
    out = generate_guardian_reports(df)
    assert out.iloc[0]["primary_causes"] == "accelerated battery aging"
    assert out.iloc[0]["evidence_confidence"] == "HEURISTIC"


def test_an_unvalidated_rul_is_qualified_in_the_report():
    """A remaining-life figure must not read as measured when it is not.

    `rul_estimation.compute_rul` is a hand-picked weighting times
    `base_cycle_life = 1000`, a constant no measurement supports. A frame
    carrying no provenance is treated as unvalidated, because the safe default
    for an unknown estimator is not to vouch for it.
    """
    df = pd.DataFrame([_row(avg_temp=40, battery_state="DEGRADED", rul_cycles=250)])
    out = generate_guardian_reports(df)
    row = out.iloc[0]
    assert row["guardian_report"] == (
        "Battery B1 is in DEGRADED state with an UNVALIDATED remaining-life "
        "estimate of 250 cycles. Primary degradation factors include high "
        "temperature exposure. Recommended action: Reduce fast charging and "
        "monitor temperature"
    )
    assert row["guardian_status"] == "Battery performance degradation detected"
    assert row["recommendation"] == "Reduce fast charging and monitor temperature"


def test_a_validated_rul_reads_plainly():
    """The qualifier must lift when the estimate has been scored."""
    df = pd.DataFrame([_row(avg_temp=40, battery_state="DEGRADED", rul_cycles=250)])
    df["rul_validated"] = True
    out = generate_guardian_reports(df)
    assert out.iloc[0]["guardian_report"] == (
        "Battery B1 is in DEGRADED state with estimated remaining life of 250 "
        "cycles. Primary degradation factors include high temperature exposure"
        ". Recommended action: Reduce fast charging and monitor temperature"
    )


def test_the_other_evidence_fields_are_untouched_by_the_qualifier():
    """Only the report sentence changes; consumers of the rest see no shift."""
    df = pd.DataFrame([_row(avg_temp=40, battery_state="DEGRADED", rul_cycles=250)])
    plain = generate_guardian_reports(df).iloc[0]
    df["rul_validated"] = True
    marked = generate_guardian_reports(df).iloc[0]
    for field in ("guardian_status", "recommendation", "primary_causes",
                  "evidence_confidence"):
        assert plain[field] == marked[field], field


def test_missing_required_columns_still_raises():
    df = pd.DataFrame([{"battery_id": "B1"}])
    with pytest.raises(ValueError, match="missing required columns"):
        generate_guardian_reports(df)


if __name__ == "__main__":
    test_temperature_only_cause_gets_validated_confidence()
    test_current_based_causes_get_heuristic_confidence()
    test_mixed_causes_get_mixed_confidence()
    test_no_causes_gets_na_confidence()
    test_health_index_driven_cause_is_heuristic_not_validated()
    test_an_unvalidated_rul_is_qualified_in_the_report()
    test_a_validated_rul_reads_plainly()
    test_the_other_evidence_fields_are_untouched_by_the_qualifier()
    print("All guardian evidence-labeling tests passed.")


# ---------------------------------------------------------------------------
# Measured state, heat advice, and labelled general guidance
# ---------------------------------------------------------------------------

def test_a_measured_state_gets_a_measured_recommendation():
    df = pd.DataFrame([_row(avg_temp=25, battery_state="DEGRADED", rul_cycles=40)])
    df["state_basis"] = "measured_soh"
    df["soh_measured"] = 0.87
    df["rul_validated"] = True
    row = generate_guardian_reports(df).iloc[0]
    assert row["recommendation"].startswith("Below 90% of original capacity")
    assert "MEASURED state of health of 87%" in row["guardian_report"]
    assert "UNVALIDATED" not in row["guardian_report"]


def test_a_heuristic_state_keeps_its_labelled_recommendation():
    df = pd.DataFrame([_row(avg_temp=25, battery_state="DEGRADED", rul_cycles=40)])
    df["state_basis"] = "heuristic_index"
    row = generate_guardian_reports(df).iloc[0]
    assert row["recommendation"] == "Reduce fast charging and monitor temperature"


def test_heat_advice_names_the_temperature_and_gives_no_invented_figure():
    hot = generate_guardian_reports(pd.DataFrame([_row(avg_temp=41.0)])).iloc[0]
    assert "41.0 °C" in hot["heat_advice"]
    assert "no figure is given" in hot["heat_advice"]
    cool = generate_guardian_reports(pd.DataFrame([_row(avg_temp=24.0)])).iloc[0]
    assert "not flagged" in cool["heat_advice"]


def test_general_guidance_is_labelled_unconfirmed():
    row = generate_guardian_reports(pd.DataFrame([_row()])).iloc[0]
    assert "NOT confirmed by this project's data" in row["general_guidance"]
