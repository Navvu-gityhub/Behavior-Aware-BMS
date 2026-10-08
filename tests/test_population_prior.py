"""Tests for rul.population_prior on synthetic linear fade.

Whether the prior helps on real cells is measured on held-out datasets
(reports/metrics/population_rul/), not asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.rul.population_prior import (  # noqa: E402
    FadePrior,
    cell_fade_rate,
    fit_prior,
    posterior_rul,
)


def _cell(k: float, n: int = 400, noise: float = 0.003, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    return x, 1.0 - k * x + rng.normal(0, noise, n)


def test_fade_rate_recovers_the_slope():
    x, y = _cell(2e-4)
    assert cell_fade_rate(x, y) == pytest.approx(2e-4, rel=0.05)


def test_a_cell_that_does_not_fade_has_no_rate():
    x = np.arange(50.0)
    assert np.isnan(cell_fade_rate(x, np.ones(50)))


def test_prior_needs_a_population():
    with pytest.raises(ValueError, match="at least 3"):
        fit_prior(np.array([1e-4, 2e-4]))


def test_with_one_reading_the_answer_is_the_population():
    prior = FadePrior(mean=float(np.log(1e-3)), sd=0.3, n_cells=20)
    p = posterior_rul(np.array([0.0]), np.array([1.0]), 0.0, prior)
    assert p.median == pytest.approx(0.10 / 1e-3, rel=0.02)
    assert p.lower < p.median < p.upper


def test_with_long_history_the_answer_is_the_cell():
    prior = FadePrior(mean=float(np.log(1e-3)), sd=0.5, n_cells=20)   # population fades 5x faster
    x, y = _cell(2e-4, n=300)
    p = posterior_rul(x, y, 299, prior)
    assert p.median == pytest.approx(0.10 / 2e-4 - 299, rel=0.1)


def test_readings_after_at_cycle_are_not_used():
    prior = FadePrior.flat()
    x, y = _cell(2e-4, n=300)
    y_future_changed = y.copy()
    y_future_changed[200:] = 0.5
    a = posterior_rul(x, y, 150, prior)
    b = posterior_rul(x, y_future_changed, 150, prior)
    assert a.median == b.median


# -- v2 ----------------------------------------------------------------------

from src.bms.rul.population_prior import (  # noqa: E402
    LifePrior,
    fit_life_curve,
    fit_life_prior,
    posterior_life,
)


def _power(life: float, p: float, n: int, noise: float = 0.002, seed: int = 1):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    return x, 1.0 - 0.10 * (x / life) ** p + rng.normal(0, noise, n)


def test_life_curve_recovers_end_of_life_and_shape():
    x, y = _power(800.0, 0.6, 900)
    life, p = fit_life_curve(x, y)
    assert life == pytest.approx(800.0, rel=0.05)
    assert p == pytest.approx(0.6, abs=0.1)


def test_life_prior_needs_a_population():
    with pytest.raises(ValueError, match="at least 3"):
        fit_life_prior(np.array([500.0, 600.0]), np.array([1.0, 1.0]))


def test_v2_with_one_reading_is_the_population():
    prior = LifePrior(float(np.log(600.0)), 0.3, 1.0, 0.3, 20)
    p = posterior_life(np.array([0.0]), np.array([1.0]), 0.0, prior)
    assert p.median == pytest.approx(600.0, rel=0.05)


def test_v2_follows_a_front_loaded_cell():
    x, y = _power(800.0, 0.6, 500)
    p = posterior_life(x, y, 499, LifePrior.flat())
    assert p.lower <= 800.0 - 499 <= p.upper


def test_v2_noise_widens_for_correlated_misfit():
    x = np.arange(200, dtype=float)
    y = 1.0 - 0.10 * (x / 600.0) + 0.01 * np.sin(x / 15.0)
    p = posterior_life(x, y, 199, LifePrior.flat())
    assert p.sigma > 0.01
