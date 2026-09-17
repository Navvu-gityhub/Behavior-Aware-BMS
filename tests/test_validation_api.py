"""The validation endpoint must report the record, never a plausible default.

This endpoint exists so the dashboard can state what was actually tested beside
the scores it displays. Two properties make it trustworthy, and both are pinned
here:

* every reported R2 carries the noise ceiling of its target, so a client cannot
  render one without the other by accident;
* a missing artifact produces a refusal that names the file, not a substituted
  number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.api.app import app  # noqa: E402
from src.bms.api.validation_routes import LIMITATIONS  # noqa: E402

client = TestClient(app)


def test_the_summary_is_served_from_the_tracked_artifacts():
    response = client.get("/validation/summary")
    assert response.status_code == 200

    body = response.json()
    assert body["available"] is True, body["missing_artifacts"]
    assert body["target_ceilings"], "no target ceilings were reported"
    assert body["methods"], "no method results were reported"


def test_no_r2_is_reported_without_its_target_noise_ceiling():
    """The structural rule. An R2 of 0.46 means something entirely different
    against a ceiling of 0.044 than against one of 0.907, and reporting the
    first without the second is how this project misread its own results."""
    body = client.get("/validation/summary").json()

    scored = [m for m in body["methods"] if m["loco_r2"] is not None]
    assert scored, "no scored methods to check"

    for method in scored:
        assert method["r2_ceiling"] is not None, (
            f"{method['method']} on {method['target']} reports "
            f"loco_r2={method['loco_r2']} with no ceiling beside it"
        )


def test_the_known_noise_ceiling_is_reported_as_measured():
    """0.044 on `capacity_loss` is the project's most consequential number."""
    body = client.get("/validation/summary").json()
    ceilings = {t["target"]: t for t in body["target_ceilings"]}

    assert "capacity_loss" in ceilings
    entry = ceilings["capacity_loss"]
    assert entry["max_attainable_r2"] == pytest.approx(0.044, abs=0.001)
    assert "noise" in entry["verdict"].lower()


def test_a_marginal_ceiling_is_not_described_as_well_conditioned():
    """0.569 means 43% of the variance is noise. Calling that well-conditioned
    is how a reader treats an R2 of 0.46 against it as a near-perfect fit."""
    body = client.get("/validation/summary").json()
    ceilings = {t["target"]: t for t in body["target_ceilings"]}

    soh = ceilings.get("soh")
    if soh is None:
        pytest.skip("no soh target in the tracked signal report")
    assert soh["max_attainable_r2"] < 0.75
    assert "well-conditioned" not in soh["verdict"]


def test_the_limitations_are_served_and_include_the_withdrawn_claims():
    body = client.get("/validation/summary").json()
    limitations = " ".join(body["limitations"]).lower()

    assert len(body["limitations"]) == len(LIMITATIONS)
    assert "six" in limitations and "withdrawn" in limitations
    assert "0.044" in limitations


def test_nothing_is_promoted_against_a_target_that_is_mostly_noise():
    """Methods do clear the gate - on well-conditioned targets.

    Four clear it on `soh`/`cumulative_fade`, which the 2026-08-21 addendum to
    `docs/final_report.md` records. What must never happen is a promotion
    against `capacity_loss`, whose ceiling of 0.044 means the best possible fit
    explains 4% of the variance. A promotion there would be the gate failing to
    do the one thing it exists for.
    """
    body = client.get("/validation/summary").json()
    ceilings = {t["target"]: t["max_attainable_r2"] for t in body["target_ceilings"]}

    offenders = [
        (m["method"], m["target"], ceilings.get(m["target"]))
        for m in body["methods"]
        if m.get("promoted") and ceilings.get(m["target"], 1.0) < 0.1
    ]
    assert not offenders, (
        f"promoted against a target that is mostly measurement noise: "
        f"{offenders}"
    )


def test_promotion_status_is_reported_so_a_viewer_can_see_it():
    """Which methods cleared the gate is part of the evidence, not a detail.

    The dashboard's own health index, risk score and RUL remain rule-based and
    are not produced by any of these; the promoted set describes the benchmark,
    not the product.
    """
    body = client.get("/validation/summary").json()
    assert any(m.get("promoted") is not None for m in body["methods"]), (
        "promotion status is absent, so a viewer cannot tell which methods "
        "cleared the validation gate"
    )


def test_a_missing_artifact_refuses_rather_than_substituting_a_default(
    monkeypatch, tmp_path
):
    from src.bms.api import validation_routes

    monkeypatch.setattr(validation_routes, "METRICS", tmp_path)

    response = client.get("/validation/ceilings")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "benchmark_signal_report.csv" in detail
    assert "will not substitute" in detail

    summary = client.get("/validation/summary").json()
    assert summary["available"] is False
    assert "benchmark_signal_report.csv" in summary["missing_artifacts"]
    assert summary["target_ceilings"] == []
    # The editorial limitations survive an absent artifact: they are not
    # derived from it, and a viewer with no data most needs the caveats.
    assert summary["limitations"]
