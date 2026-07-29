"""Tests for the versioned model store.

The two tests worth reading first:

`test_rejections_are_recorded_not_discarded` — a store that logged only
successes would present a history of unbroken progress. The rejections are the
record of what the gate caught, and they have to survive.

`test_rollback_preserves_the_reverted_version` — reverting must never delete.
A bad promotion that leaves no trace teaches nobody anything, and makes the
same mistake repeatable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.store import ModelStore, ModelVersion
from src.bms.adaptive.validation import Verdict


def _pass(**metrics) -> Verdict:
    return Verdict(True, ["beat baseline on both splits"], metrics or {"loco_median_r2": 0.12})


def _fail(**metrics) -> Verdict:
    return Verdict(False, ["LOCO median R2 below zero"], metrics or {"loco_median_r2": -0.17})


@pytest.fixture()
def store(tmp_path) -> ModelStore:
    return ModelStore(tmp_path / "adaptive")


# ---------------------------------------------------------------------------
# Promotion and rejection
# ---------------------------------------------------------------------------

def test_a_passing_candidate_is_promoted_to_v1(store):
    record = store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    assert isinstance(record, ModelVersion)
    assert record.version == 1
    assert record.parent_version is None
    assert store.active_version("RT_2A") == 1


def test_a_failing_candidate_is_not_promoted(store):
    assert store.propose("RT_2A", {"temp_coef": 0.0038}, _fail()) is None
    assert store.active_version("RT_2A") is None


def test_rejections_are_recorded_not_discarded(store):
    """The rejection log is the record of what the gate caught."""
    store.propose("RT_2A", {"temp_coef": 0.0038}, _fail())
    store.propose("RT_2A", {"temp_coef": 0.0041}, _fail())

    decisions = list(store.decisions("RT_2A"))
    assert len(decisions) == 2
    assert all(d.outcome == "REJECT" for d in decisions)
    assert all("LOCO" in " ".join(d.reasons) for d in decisions)


def test_the_verdict_that_justified_a_promotion_is_stored_with_it(store):
    """A promotion whose justification wasn't recorded is an unreviewed one."""
    record = store.propose("RT_2A", {"temp_coef": 0.0038}, _pass(loco_median_r2=0.15))
    assert record.metrics["loco_median_r2"] == pytest.approx(0.15)
    assert record.reasons

    reloaded = store.load("RT_2A", 1)
    assert reloaded.metrics == record.metrics
    assert reloaded.reasons == record.reasons


def test_versions_increment_and_track_their_parent(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    second = store.propose("RT_2A", {"temp_coef": 0.0040}, _pass())
    assert second.version == 2
    assert second.parent_version == 1
    assert store.active_version("RT_2A") == 2


def test_cohorts_are_versioned_independently(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("COLD4C", {"temp_coef": 0.0021}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0039}, _pass())

    assert store.active_version("RT_2A") == 2
    assert store.active_version("COLD4C") == 1


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def test_a_sign_flip_blocks_promotion_even_with_a_passing_gate(store):
    """Sign flips are the signature of a fit driven by which cells arrived.

    The project already observed this: current-based flags flip sign between
    room-temperature and 4C cohorts.
    """
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    result = store.propose("RT_2A", {"temp_coef": -0.0035}, _pass())

    assert result is None
    assert store.active_version("RT_2A") == 1
    last = list(store.decisions("RT_2A"))[-1]
    assert last.outcome == "REJECT"
    assert "flipped sign" in " ".join(last.reasons)


def test_a_large_coefficient_move_blocks_promotion(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    result = store.propose("RT_2A", {"temp_coef": 0.0200}, _pass())
    assert result is None
    assert "above the 50% tolerance" in " ".join(list(store.decisions("RT_2A"))[-1].reasons)


def test_a_small_coefficient_move_is_allowed(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    assert store.propose("RT_2A", {"temp_coef": 0.0042}, _pass()) is not None


def test_first_version_has_nothing_to_compare_against(store):
    assert store.check_stability("NEW_COHORT", {"temp_coef": 0.0038}) == []


def test_a_near_zero_parameter_does_not_produce_a_false_instability(store):
    """Relative change is not computable against ~0 and must not be asserted."""
    store.propose("RT_2A", {"fast_charge_coef": 0.0}, _pass())
    assert store.check_stability("RT_2A", {"fast_charge_coef": 0.0}) == []


def test_appearing_and_disappearing_parameters_are_flagged(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    problems = store.check_stability("RT_2A", {"soc_coef": 0.5})
    assert any("disappeared" in p for p in problems)
    assert any("is new" in p for p in problems)


def test_stability_enforcement_can_be_disabled_explicitly(store):
    """An intentional respecification should be possible, but never by default."""
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    forced = store.propose(
        "RT_2A", {"temp_coef": -0.02}, _pass(),
        note="deliberate respecification", enforce_stability=False,
    )
    assert forced is not None and forced.version == 2


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def test_rollback_reverts_the_active_pointer(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0042}, _pass())
    assert store.active_version("RT_2A") == 2

    reverted = store.rollback("RT_2A", reason="production R2 dropped")
    assert reverted.version == 1
    assert store.active_version("RT_2A") == 1


def test_rollback_preserves_the_reverted_version(store):
    """Reverting must never delete. The bad version stays reviewable."""
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0042}, _pass())
    store.rollback("RT_2A")

    assert store.load("RT_2A", 2).params["temp_coef"] == pytest.approx(0.0042)
    assert [v.version for v in store.history("RT_2A")] == [1, 2]


def test_rollback_is_recorded_in_the_log(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0042}, _pass())
    store.rollback("RT_2A", reason="drift detected")

    last = list(store.decisions("RT_2A"))[-1]
    assert last.outcome == "ROLLBACK"
    assert "drift detected" in " ".join(last.reasons)


def test_rollback_from_v1_refuses_rather_than_leaving_nothing_active(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    with pytest.raises(ValueError, match="no parent"):
        store.rollback("RT_2A")


def test_rollback_of_an_unknown_cohort_raises(store):
    with pytest.raises(KeyError, match="No active model"):
        store.rollback("NEVER_SEEN")


# ---------------------------------------------------------------------------
# Storage format
# ---------------------------------------------------------------------------

def test_stored_models_are_human_readable_json(store):
    """Artifacts a reviewer cannot read are not governance artifacts."""
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    path = next((store.versions_dir).glob("*.json"))
    payload = json.loads(path.read_text())
    assert payload["params"]["temp_coef"] == pytest.approx(0.0038)
    assert payload["cohort_id"] == "RT_2A"


def test_unserialisable_parameters_are_refused_with_an_explanation(store):
    with pytest.raises(TypeError, match="JSON-serialisable"):
        store.propose("RT_2A", {"model": object()}, _pass())


def test_the_decision_log_is_append_only(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _fail())
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0040}, _pass())

    lines = store.log_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert [json.loads(l)["outcome"] for l in lines] == ["REJECT", "PROMOTE", "PROMOTE"]


def test_a_reopened_store_sees_prior_state(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    reopened = ModelStore(store.root)
    assert reopened.active_version("RT_2A") == 1
    assert reopened.active("RT_2A").params["temp_coef"] == pytest.approx(0.0038)


def test_cohort_ids_with_awkward_characters_round_trip(store):
    """Cohort ids come from data and are not guaranteed filename-safe."""
    cohort = "MIXED_24/44C multiload"
    store.propose(cohort, {"temp_coef": 0.0038}, _pass())
    assert store.active(cohort).cohort_id == cohort


def test_summary_reports_promotions_rejections_and_rollbacks(store):
    store.propose("RT_2A", {"temp_coef": 0.0038}, _fail())
    store.propose("RT_2A", {"temp_coef": 0.0038}, _pass())
    store.propose("RT_2A", {"temp_coef": 0.0042}, _pass())
    store.rollback("RT_2A")

    row = next(r for r in store.summary() if r["cohort_id"] == "RT_2A")
    assert row["active_version"] == 1
    assert row["n_versions"] == 2
    assert (row["n_promotions"], row["n_rejections"], row["n_rollbacks"]) == (2, 1, 1)
