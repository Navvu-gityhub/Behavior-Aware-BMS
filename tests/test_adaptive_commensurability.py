"""Tests for feature commensurability.

The test that carries the most weight is
`test_the_nasa_stanford_orthogonality_problem`. It reconstructs the structural
obstacle to the project's headline experiment: NASA varies ambient temperature
across nine protocols while holding fast-charge duration at identically zero;
Stanford/Severson holds temperature at 30 C in a chamber while varying charge
policy from 3.6C to 6C. The axes are orthogonal, so neither dataset can carry a
model fitted on the other's variable, and this must be detected before any
transfer number is computed.

`test_celsius_scale_does_not_decide_whether_a_feature_varies` pins the fix to a
real defect in the first version of this module, which used coefficient of
variation and therefore let the arbitrary zero of the Celsius scale determine
whether temperature counted as varying.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.commensurability import (
    assess_commensurability,
    measure_variation,
)

REPO = Path(__file__).resolve().parents[1]
NASA_TRAINING = REPO / "reports/metrics/continuous_model_training_data.csv"


def _frame(n: int = 200, seed: int = 0, **columns) -> pd.DataFrame:
    """Build a frame where each keyword is (mean, std)."""
    rng = np.random.default_rng(seed)
    data = {"cell_id": [f"C{i % 8}" for i in range(n)], "cycle": np.arange(n)}
    for name, (mean, std) in columns.items():
        data[name] = mean + rng.normal(0, std, n) if std > 0 else np.full(n, mean)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def test_variation_is_measured_without_judgement():
    measurement = measure_variation(_frame(temp=(25.0, 2.0)), "temp", "nasa")
    assert measurement.mean == pytest.approx(25.0, abs=0.5)
    assert measurement.std == pytest.approx(2.0, abs=0.4)
    assert measurement.can_fit_a_slope is True


def test_an_absent_feature_measures_as_none():
    assert measure_variation(_frame(temp=(25.0, 2.0)), "voltage", "nasa") is None


def test_a_constant_channel_cannot_fit_a_slope():
    """NASA's fast_charge_duration is identically zero across all 2,682 rows."""
    measurement = measure_variation(_frame(fast_charge=(0.0, 0.0)), "fast_charge")
    assert measurement.std == 0.0
    assert measurement.can_fit_a_slope is False


def test_spread_ratio_compares_like_with_like():
    wide = measure_variation(_frame(temp=(25.0, 10.0), seed=1), "temp", "nasa")
    narrow = measure_variation(_frame(temp=(30.0, 1.0), seed=2), "temp", "stanford")
    ratio = narrow.spread_ratio_against(wide)
    assert 0.0 < ratio < 0.25


# ---------------------------------------------------------------------------
# The defect the first version had
# ---------------------------------------------------------------------------

def test_celsius_scale_does_not_decide_whether_a_feature_varies():
    """The same physical spread must be judged the same at any temperature.

    The first version of this module used coefficient of variation, which made
    +/-1.5 C count as varying at 25 C and constant at 70 C purely because
    Celsius has an arbitrary zero.
    """
    source = _frame(temp=(25.0, 1.5), seed=1)
    cold_target = _frame(temp=(4.0, 1.5), seed=2)
    hot_target = _frame(temp=(70.0, 1.5), seed=3)

    cold = assess_commensurability(source, cold_target, ["temp"], "nasa", "cold")
    hot = assess_commensurability(source, hot_target, ["temp"], "nasa", "hot")

    # Identical absolute spread, so identical verdict.
    assert cold.usable_features == ("temp",)
    assert hot.usable_features == ("temp",)


# ---------------------------------------------------------------------------
# Commensurability
# ---------------------------------------------------------------------------

def test_a_feature_varying_in_both_is_usable():
    report = assess_commensurability(
        _frame(temp=(25.0, 3.0), seed=1),
        _frame(temp=(27.0, 3.0), seed=2),
        ["temp"], "nasa", "calce",
    )
    assert report.feasible is True
    assert bool(report) is True
    assert report.status == "FEASIBLE"


def test_a_feature_constant_in_the_source_cannot_be_fitted():
    report = assess_commensurability(
        _frame(fast_charge=(0.0, 0.0), seed=1),
        _frame(fast_charge=(4.5, 1.2), seed=2),
        ["fast_charge"], "nasa", "stanford",
    )
    assert report.feasible is False
    assert report.constant_in_source == ("fast_charge",)
    assert "no coefficient can be fitted" in report.render()


def test_a_feature_constant_in_the_target_has_nothing_to_act_on():
    report = assess_commensurability(
        _frame(temp=(25.0, 12.0), seed=1),      # NASA: 4C to 43C
        _frame(temp=(30.0, 0.05), seed=2),      # tightly held chamber
        ["temp"], "nasa", "stanford",
    )
    assert report.feasible is False
    assert report.constant_in_target == ("temp",)
    assert "nothing to act on" in report.render()


def test_absent_features_are_reported_separately_from_constant_ones():
    """A missing channel and a flat channel are different problems."""
    report = assess_commensurability(
        _frame(temp=(25.0, 3.0), seed=1),
        _frame(other=(1.0, 1.0), seed=2),
        ["temp"], "nasa", "calce",
    )
    assert report.absent_in_target == ("temp",)
    assert report.constant_in_target == ()


def test_a_surviving_subset_is_reported_as_reduced():
    report = assess_commensurability(
        _frame(temp=(25.0, 3.0), fast_charge=(0.0, 0.0), seed=1),
        _frame(temp=(27.0, 3.0), fast_charge=(4.5, 1.0), seed=2),
        ["temp", "fast_charge"], "nasa", "stanford",
    )
    assert report.usable_features == ("temp",)
    assert report.constant_in_source == ("fast_charge",)
    assert report.status == "FEASIBLE_REDUCED"


def test_the_variation_table_covers_both_datasets():
    report = assess_commensurability(
        _frame(temp=(25.0, 3.0), seed=1),
        _frame(temp=(27.0, 3.0), seed=2),
        ["temp"], "nasa", "calce",
    )
    table = report.to_frame()
    assert set(table["dataset"]) == {"nasa", "calce"}
    assert "can_fit_a_slope" in table.columns


# ---------------------------------------------------------------------------
# The structural obstacle to the project's headline experiment
# ---------------------------------------------------------------------------

def test_the_nasa_stanford_orthogonality_problem():
    """NASA and Stanford vary along orthogonal axes.

    NASA: ambient temperature spans roughly 4-43 C across nine protocols, while
    fast_charge_duration is identically zero in all 2,682 observations.

    Stanford/Severson: 124 A123 LFP cells in a forced-convection chamber set to
    30 C, discharged identically at 4C, varied by charge policy from 3.6C to 6C.

    So the axis NASA varies is held constant in Stanford, and the axis Stanford
    varies does not exist in NASA. Neither dataset can carry a model fitted on
    the other's experimental variable, and no transfer metric should be produced.
    """
    nasa = _frame(
        n=2682, seed=1,
        avg_temp=(24.0, 12.0),          # varied by design across cohorts
        fast_charge_duration=(0.0, 0.0),  # identically zero
    )
    stanford = _frame(
        n=96700, seed=2,
        avg_temp=(30.0, 0.4),           # chamber-held; self-heating only
        fast_charge_duration=(4.8, 0.9),  # varied by design
    )

    report = assess_commensurability(
        nasa, stanford, ["avg_temp", "fast_charge_duration"], "nasa", "stanford",
    )

    assert report.feasible is False
    assert "fast_charge_duration" in report.constant_in_source
    assert "avg_temp" in report.constant_in_target
    assert "No feature varies in both datasets" in report.render()


@pytest.mark.skipif(not NASA_TRAINING.exists(), reason="NASA training frame not present")
def test_the_real_nasa_frame_confirms_fast_charge_is_degenerate():
    data = pd.read_csv(NASA_TRAINING)
    if "fast_charge_duration" not in data.columns:
        pytest.skip("fast_charge_duration not in this frame")
    measurement = measure_variation(data, "fast_charge_duration", "nasa")
    assert measurement.can_fit_a_slope is False, (
        "fast_charge_duration is expected to be degenerate in NASA; if it now "
        "varies, the threshold audit's conclusion needs revisiting"
    )
