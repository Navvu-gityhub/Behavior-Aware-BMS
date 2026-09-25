"""Tests for the fold bootstrap behind the benchmark table's intervals.

The benchmark table reported point estimates with no uncertainty, which is the
thing this project criticises other benchmark tables for. These tests pin the
properties that make the added interval mean something:

`test_interval_brackets_the_median` — the obvious one, and the one that would
break first if the percentile arithmetic were wrong.

`test_identical_folds_give_a_zero_width_interval` — nine folds that all agree
carry no spread, and the interval must say so rather than manufacture one.

`test_wider_spread_gives_a_wider_interval` — monotonicity. An interval that
does not respond to the data is decoration.

`test_too_few_folds_returns_nan_rather_than_a_narrow_interval` — with three
folds the resample median can only take three values, so a percentile interval
would read as precision that is not there. NaN is the honest answer and the
renderer prints "n=3 too few".

`test_the_seed_makes_it_reproducible` — a re-run must not move a published
interval. A CI that changes on every run cannot be pinned by
test_reported_numbers.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.validation import (
    MIN_FOLDS_FOR_CI,
    CrossValidationResult,
    FoldResult,
    bootstrap_median_ci,
)


def _fold(r2: float, held_out: str = "c") -> FoldResult:
    return FoldResult(
        split="LOCO",
        held_out=held_out,
        n_train=100,
        n_test=20,
        mae=0.01,
        r2_vs_global_mean=r2,
        r2_vs_own_mean_oracle=r2,
        spearman_rho=0.5,
        rmse=0.02,
    )


def _result(values: list[float]) -> CrossValidationResult:
    return CrossValidationResult(
        split="LOCO",
        group_col="cohort",
        folds=[_fold(v, f"cohort_{i}") for i, v in enumerate(values)],
    )


# ---------------------------------------------------------------------------
# The interval means what it says
# ---------------------------------------------------------------------------

def test_interval_brackets_the_median():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = bootstrap_median_ci(values)
    assert low <= np.median(values) <= high


def test_identical_folds_give_a_zero_width_interval():
    """No spread in, no spread out. Anything else is manufactured."""
    low, high = bootstrap_median_ci([0.42] * 9)
    assert low == pytest.approx(0.42)
    assert high == pytest.approx(0.42)


def test_wider_spread_gives_a_wider_interval():
    tight = bootstrap_median_ci([0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.50])
    loose = bootstrap_median_ci([0.10, 0.90, 0.20, 0.80, 0.30, 0.70, 0.40, 0.60])
    assert (loose[1] - loose[0]) > (tight[1] - tight[0])


def test_a_negative_median_is_reported_as_negative():
    """LSTM scores negative under LOCO; the interval must not clip at zero."""
    low, high = bootstrap_median_ci([-0.5, -0.3, -0.2, -0.1, -0.4, -0.6])
    assert high < 0


# ---------------------------------------------------------------------------
# Refusing to report what the data cannot support
# ---------------------------------------------------------------------------

def test_too_few_folds_returns_nan_rather_than_a_narrow_interval():
    low, high = bootstrap_median_ci([0.1, 0.5, 0.9])
    assert np.isnan(low) and np.isnan(high)


def test_the_floor_is_the_documented_constant():
    assert MIN_FOLDS_FOR_CI == 4
    ok = bootstrap_median_ci([0.1] * MIN_FOLDS_FOR_CI)
    assert np.isfinite(ok[0]) and np.isfinite(ok[1])


def test_non_finite_folds_are_dropped_not_propagated():
    """One crashed fold must not turn the whole interval into NaN."""
    values = [0.4, 0.5, float("nan"), 0.6, 0.5, float("inf"), 0.45]
    low, high = bootstrap_median_ci(values)
    assert np.isfinite(low) and np.isfinite(high)


def test_all_non_finite_gives_nan():
    low, high = bootstrap_median_ci([float("nan")] * 9)
    assert np.isnan(low) and np.isnan(high)


def test_empty_input_does_not_raise():
    low, high = bootstrap_median_ci([])
    assert np.isnan(low) and np.isnan(high)


# ---------------------------------------------------------------------------
# Reproducibility — a published interval must not move on a re-run
# ---------------------------------------------------------------------------

def test_the_seed_makes_it_reproducible():
    values = [0.1, 0.4, 0.2, 0.8, 0.3, 0.55, 0.6, 0.35]
    first = bootstrap_median_ci(values)
    second = bootstrap_median_ci(values)
    assert first == second


def test_a_different_seed_gives_a_similar_interval():
    """Seed choice must not be load-bearing.

    If two seeds disagreed materially the resample count would be too low and
    the published interval would be an artifact of the default seed.
    """
    values = [0.1, 0.4, 0.2, 0.8, 0.3, 0.55, 0.6, 0.35, 0.5, 0.45]
    a = bootstrap_median_ci(values, seed=1)
    b = bootstrap_median_ci(values, seed=999)
    assert a[0] == pytest.approx(b[0], abs=0.05)
    assert a[1] == pytest.approx(b[1], abs=0.05)


def test_confidence_level_widens_the_interval():
    values = [0.1, 0.4, 0.2, 0.8, 0.3, 0.55, 0.6, 0.35, 0.5, 0.45]
    narrow = bootstrap_median_ci(values, confidence=0.80)
    wide = bootstrap_median_ci(values, confidence=0.99)
    assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])


# ---------------------------------------------------------------------------
# Wiring into CrossValidationResult
# ---------------------------------------------------------------------------

def test_median_ci_matches_the_free_function():
    values = [0.1, 0.4, 0.2, 0.8, 0.3, 0.55, 0.6, 0.35]
    result = _result(values)
    assert result.median_ci() == bootstrap_median_ci(values)


def test_n_completed_counts_finite_folds_only():
    result = _result([0.1, 0.2, float("nan"), 0.3, 0.4])
    assert result.n_completed == 4


def test_errored_folds_are_excluded_from_the_interval():
    """A fold that raised carries no R2 and must not enter the resample."""
    folds = [_fold(v, f"c{i}") for i, v in enumerate([0.4, 0.5, 0.6, 0.5, 0.45])]
    folds.append(FoldResult(
        split="LOCO", held_out="broken", n_train=0, n_test=0,
        mae=float("nan"),
        r2_vs_global_mean=float("nan"), r2_vs_own_mean_oracle=float("nan"),
        spearman_rho=float("nan"), rmse=float("nan"), error="fit failed",
    ))
    result = CrossValidationResult(split="LOCO", group_col="cohort", folds=folds)
    assert result.n_completed == 5
    assert result.median_ci() == bootstrap_median_ci([0.4, 0.5, 0.6, 0.5, 0.45])
