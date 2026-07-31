"""Tests for cross-dataset transfer validation.

The load-bearing test is `test_a_scale_offset_does_not_count_as_transfer`. It
constructs the exact false positive this harness exists to prevent: a target
dataset whose cells have a different nominal capacity, where a model can post a
large R-squared against the *source* mean purely because the source mean is
badly calibrated for the target. Measured against the target's own mean — which
anyone holding the target data could compute in one line — the same model has
added nothing.

`test_transfer_to_a_shifted_domain_is_flagged_as_extrapolation` covers the
other half: a transfer attempted outside the source's observed range is
reported as extrapolation rather than silently adapted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.transfer import (
    TransferValidator,
    assess_compatibility,
    measure_domain_shift,
    transfer_summary,
)

FEATURES = ("trailing_avg_temp",)


def _dataset(
    n_cells: int = 8,
    cycles: int = 50,
    temp_base: float = 25.0,
    capacity_scale: float = 1.0,
    fade_per_degree: float = 0.004,
    noise: float = 0.0005,
    seed: int = 3,
) -> pd.DataFrame:
    """A synthetic cycling dataset with a controllable fade law.

    `fade_per_degree` is the transferable physics. Two datasets sharing it
    should transfer; two with different laws should not.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_cells):
        temp = temp_base + rng.normal(0, 1.5, cycles)
        frames.append(pd.DataFrame({
            "cell_id": f"C{i:02d}",
            "cohort": f"COHORT_{i % 2}",
            "cycle": np.arange(1, cycles + 1),
            "trailing_avg_temp": temp,
            "capacity_loss": capacity_scale * (
                fade_per_degree * temp + rng.normal(0, noise, cycles)
            ),
        }))
    return pd.concat(frames, ignore_index=True)


def _temp_fit(train: pd.DataFrame):
    slope, intercept = np.polyfit(
        train["trailing_avg_temp"], train["capacity_loss"], 1
    )
    return lambda test: intercept + slope * test["trailing_avg_temp"].to_numpy(float)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

def test_compatible_frames_report_shared_features():
    report = assess_compatibility(
        _dataset(), _dataset(seed=9), FEATURES, "capacity_loss", "nasa", "calce"
    )
    assert report.usable is True
    assert bool(report) is True
    assert "trailing_avg_temp" in report.shared


def test_a_target_missing_the_label_is_incompatible():
    target = _dataset(seed=9).drop(columns=["capacity_loss"])
    report = assess_compatibility(
        _dataset(), target, FEATURES, "capacity_loss", "nasa", "calce"
    )
    assert report.usable is False
    assert "capacity_loss" in report.missing_in_target
    assert "INCOMPATIBLE" in report.render()


def test_incompatible_transfer_fails_without_producing_a_number():
    """A number computed from mismatched features would be meaningless."""
    source = _dataset()
    target = _dataset(seed=9).drop(columns=["trailing_avg_temp"])
    result = TransferValidator(source, "capacity_loss", "nasa").evaluate(
        _temp_fit, target, FEATURES, "calce"
    )
    assert result.status == "ERROR"
    assert result.transferred is False
    assert np.isnan(result.r2_vs_target_mean)


# ---------------------------------------------------------------------------
# Domain shift is measured, not corrected
# ---------------------------------------------------------------------------

def test_domain_shift_reports_per_feature_statistics():
    shift = measure_domain_shift(
        _dataset(temp_base=25), _dataset(temp_base=27, seed=9), FEATURES
    )
    stats = shift.per_feature["trailing_avg_temp"]
    assert stats["source_min"] < stats["source_max"]
    assert stats["target_median"] > stats["source_mean"]


def test_a_far_shifted_target_is_flagged_as_extrapolation():
    shift = measure_domain_shift(
        _dataset(temp_base=25), _dataset(temp_base=60, seed=9), FEATURES
    )
    assert shift.is_extrapolation is True
    assert "trailing_avg_temp" in shift.features_outside_source_range


def test_an_overlapping_target_is_not_extrapolation():
    shift = measure_domain_shift(
        _dataset(temp_base=25), _dataset(temp_base=26, seed=9), FEATURES
    )
    assert shift.is_extrapolation is False


# ---------------------------------------------------------------------------
# Transfer, and the false positive it must reject
# ---------------------------------------------------------------------------

def test_a_shared_fade_law_transfers():
    """Same physics, different cells: transfer should succeed."""
    source = _dataset(temp_base=25, seed=1)
    target = _dataset(temp_base=27, seed=99)
    result = TransferValidator(source, "capacity_loss", "nasa").evaluate(
        _temp_fit, target, FEATURES, "calce"
    )
    assert result.transferred is True
    assert result.status == "TRANSFERRED"
    assert result.r2_vs_target_mean > 0


def test_a_different_fade_law_does_not_transfer():
    source = _dataset(fade_per_degree=0.004, seed=1)
    target = _dataset(fade_per_degree=-0.006, seed=99)  # opposite direction
    result = TransferValidator(source, "capacity_loss", "nasa").evaluate(
        _temp_fit, target, FEATURES, "stanford"
    )
    assert result.transferred is False
    assert any("target's own mean" in r for r in result.reasons)


def test_a_level_only_transfer_does_not_count_as_transfer():
    """The false positive this harness exists to catch.

    The model lands the target's average almost exactly — so measured against
    the *source's* mean it posts R2 = +0.89, which reads as an excellent
    transfer. But the target has no internal temperature signal, so the model
    tracks none of its variation, and measured against the target's own mean
    it scores about -242.

    Getting the level right is not transfer. Anyone holding the target data
    could compute its mean in one line of pandas, and would beat this model by
    two orders of magnitude. This is why `r2_vs_target_mean` is the headline
    and `r2_vs_source_mean` is only a diagnostic.
    """
    source = _dataset(temp_base=25, fade_per_degree=0.004, noise=0.0005, seed=1)

    # Put the target at a level the source-fitted model predicts well...
    level = float(
        np.polyval(
            np.polyfit(source["trailing_avg_temp"], source["capacity_loss"], 1), 29.5
        )
    )
    target = _dataset(temp_base=29.5, fade_per_degree=0.0, noise=0.0004, seed=99)
    # ...but give it no temperature dependence for the model to track.
    target["capacity_loss"] = level + (
        target["capacity_loss"] - target["capacity_loss"].mean()
    )

    result = TransferValidator(source, "capacity_loss", "nasa").evaluate(
        _temp_fit, target, FEATURES, "calce"
    )

    # Against the source mean it looks like a strong success.
    assert result.r2_vs_source_mean > 0.5
    # Against the target's own mean it is far worse than a constant.
    assert result.r2_vs_target_mean < 0
    assert result.transferred is False
    assert any("capacity-scale offset" in r for r in result.reasons)


def test_extrapolation_is_reported_alongside_the_score():
    source = _dataset(temp_base=25, seed=1)
    target = _dataset(temp_base=70, seed=99)
    result = TransferValidator(source, "capacity_loss", "nasa").evaluate(
        _temp_fit, target, FEATURES, "hot_fleet"
    )
    assert result.shift.is_extrapolation is True
    assert any("extrapolation" in r.lower() for r in result.reasons)


def test_a_crashing_candidate_is_recorded_not_raised():
    def broken(train):
        raise RuntimeError("singular")

    result = TransferValidator(_dataset(), "capacity_loss").evaluate(
        broken, _dataset(seed=9), FEATURES
    )
    assert result.status == "ERROR"
    assert "singular" in (result.error or "")


def test_nothing_is_refitted_on_the_target():
    """The model must see the target only at prediction time."""
    seen: list[int] = []

    def spy_fit(train: pd.DataFrame):
        seen.append(len(train))
        return _temp_fit(train)

    source = _dataset(n_cells=8, cycles=50)
    target = _dataset(n_cells=3, cycles=20, seed=9)
    TransferValidator(source, "capacity_loss").evaluate(spy_fit, target, FEATURES)

    # Fitted exactly once, on the source only.
    assert seen == [len(source)]


# ---------------------------------------------------------------------------
# Multiple targets
# ---------------------------------------------------------------------------

def test_results_are_reported_per_target_not_averaged():
    """A model that works on one dataset and fails on another has said something specific."""
    source = _dataset(fade_per_degree=0.004, seed=1)
    targets = {
        "calce": _dataset(fade_per_degree=0.004, seed=50),
        "stanford": _dataset(fade_per_degree=-0.006, seed=51),
    }
    results = TransferValidator(source, "capacity_loss", "nasa").evaluate_many(
        _temp_fit, targets, FEATURES
    )
    assert len(results) == 2
    statuses = {r.target: r.transferred for r in results}
    assert statuses["calce"] is True
    assert statuses["stanford"] is False


def test_summary_table_has_one_row_per_target():
    source = _dataset(seed=1)
    results = TransferValidator(source, "capacity_loss", "nasa").evaluate_many(
        _temp_fit,
        {"calce": _dataset(seed=50), "stanford": _dataset(seed=51)},
        FEATURES,
    )
    table = transfer_summary(results)
    assert len(table) == 2
    assert set(table["target"]) == {"calce", "stanford"}
    assert "r2_vs_target_mean" in table.columns


def test_render_states_the_headline_metric_first():
    result = TransferValidator(_dataset(seed=1), "capacity_loss", "nasa").evaluate(
        _temp_fit, _dataset(seed=9), FEATURES, "calce"
    )
    rendered = result.render()
    assert "R2 vs TARGET mean (headline)" in rendered
    assert rendered.index("TARGET mean") < rendered.index("source mean")
