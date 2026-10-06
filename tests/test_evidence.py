"""Tests for the evidence and data-sufficiency block.

It must say, per output, whether that output's inputs exist - and never mark
something available that the run could not compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.guardian.health_card import render_health_card  # noqa: E402
from src.bms.health.evidence import (  # noqa: E402
    HIGH,
    MEDIUM,
    evidence_from_result,
    resistance_health,
)
from tests.test_field_soh import _cell, _score  # noqa: E402


@pytest.fixture(scope="module")
def result():
    tel, _ = _cell(n_cycles=40)
    return _score(tel, reference_is_beginning_of_life=True)


def test_missing_temperature_is_reported_per_output(result):
    ev = evidence_from_result(result, rated_capacity_known=False)
    assert ev.channels["temperature"] is False
    heat = ev.output("heat exposure")
    assert not heat.available
    assert "temperature" in heat.reason


def test_capacity_health_confidence_follows_the_stated_rules(result):
    ev = evidence_from_result(result, rated_capacity_known=False)
    cap = ev.output("capacity health")
    assert cap.available
    # 40 synthetic discharges, fixed window, beginning-of-life reference.
    assert cap.confidence == HIGH


def test_a_start_of_log_reference_lowers_confidence():
    tel, _ = _cell(n_cycles=40)
    ev = evidence_from_result(_score(tel), rated_capacity_known=False)
    cap = ev.output("capacity health")
    assert cap.confidence == MEDIUM
    assert "may not be the cell as new" in cap.reason


def test_behaviour_risk_is_never_marked_measured(result):
    ev = evidence_from_result(result, rated_capacity_known=False)
    assert not ev.output("behaviour risk score").available


def test_resistance_health_is_the_inverse_of_growth(result):
    soh = result.field_soh
    assert resistance_health(soh) == pytest.approx(1 / soh.resistance_growth)
    # The fixture's resistance grows, so resistance health falls below 100%.
    assert resistance_health(soh) < 1.0


def test_the_card_opens_with_the_evidence(result):
    card = render_health_card(result, battery_label="SYN", rated_capacity_known=False)
    assert card.index("0. EVIDENCE") < card.index("1. HOW HEALTHY IT IS")
    assert "resistance health" in card
    assert "NOT AVAILABLE" in card
