"""Tests for the battery health report card.

The card computes nothing; it lays out a `TelemetryResult`. So the tests guard
what it is allowed to SAY:

- a measurement is labelled MEASURED and an estimate ESTIMATE;
- fade measured against the start of a log is never called state of health,
  and gets no HEALTHY/WARNING state, because the log may not start at new;
- general advice is labelled as not confirmed by this project's data;
- every refusal the run produced reaches the card.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.guardian.health_card import render_health_card  # noqa: E402
from tests.test_field_soh import _cell, _score  # noqa: E402


def _card(**kwargs) -> str:
    tel, _ = _cell(n_cycles=40)
    return render_health_card(_score(tel, **kwargs), battery_label="SYN")


def test_measured_soh_is_labelled_measured():
    card = _card(reference_is_beginning_of_life=True)
    assert "capacity health  [MEASURED" in card
    assert "State: " in card


def test_start_of_log_fade_is_not_called_state_of_health():
    card = _card()
    assert "capacity relative to the start of this log" in card
    assert "No state is assigned" in card
    assert "State: " not in card


def test_general_guidance_is_labelled_unconfirmed():
    assert "NOT confirmed by this project's data" in _card()


def test_missing_temperature_is_said_not_assumed():
    assert "No temperature was measured" in _card()


def test_refusals_reach_the_card():
    card = _card()
    assert "WHAT THIS REPORT CANNOT TELL YOU" in card
    assert "Scoring skipped" in card          # the behaviour-scoring refusal


def test_scope_of_the_evidence_is_always_stated():
    assert "one LCO family" in _card()

