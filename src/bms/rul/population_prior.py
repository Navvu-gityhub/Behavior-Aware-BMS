"""Remaining life from a population prior combined with the cell's own fade.

WHY THIS EXISTS
---------------
`fade_extrapolation.estimate_rul` uses only the cell's own history. That is
what makes it hold on cells it was never tuned for, and also why it refuses
until it has 30 points and why it is only useful near end of life: early on,
a cell's own trend says almost nothing about where it is going.

Other cells do. Across many cells, how fast capacity fades per cycle is
spread over about an order of magnitude, which is wide but far narrower than
"anything". This module learns that spread from cells of OTHER datasets and
combines it with the cell's own readings by Bayes' rule:

    early:  few readings, the likelihood is flat, the answer is the
            population's, with the population's (wide) uncertainty;
    later:  the readings dominate and the answer becomes this cell's.

Nothing is learned about the shape of any one cell. What is learned is a
two-number summary of a population, and whether it helps is tested only on
whole datasets that took no part in learning it
(scripts/run_population_rul_study.py).

THE MODEL
---------
SOH relative to the cell's own reference falls as  SOH(x) = 1 - k * x,
x = cycles since the first reading. Across cells, log k ~ Normal(mu, sd).
Readings scatter around the line with standard deviation sigma, estimated
from the cell's own residuals (floored, since a short history understates
it). The posterior over log k is computed on a grid - no sampling, no
optimiser - and remaining life to `threshold` is (1 - threshold) / k minus
the cycles already elapsed, reported as a median and an interval.

WHAT IT CANNOT DO
-----------------
A straight line does not anticipate a knee. A cell that will accelerate is
under-predicted by the population and by its own early trend alike; the
interval is the only warning, and it is wide early by construction.
Populations without a chemistry in them say nothing about that chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRID_POINTS = 801
GRID_SPAN_SD = 5.0
SIGMA_FLOOR = 0.005
SIGMA_DEFAULT = 0.01
MIN_FOR_SIGMA = 10
SD_FLOOR = 0.25


@dataclass(frozen=True)
class FadePrior:
    """log(fade per cycle) across a population of cells."""

    mean: float
    sd: float
    n_cells: int

    @classmethod
    def flat(cls) -> FadePrior:
        """A prior that says nothing: the cell's own data alone."""
        return cls(mean=float(np.log(1e-3)), sd=float("inf"), n_cells=0)


@dataclass(frozen=True)
class RULPosterior:
    at_cycle: float
    median: float
    lower: float          # interval bounds at the requested coverage
    upper: float
    n_readings: int
    sigma: float


def cell_fade_rate(cycles: np.ndarray, soh: np.ndarray, threshold: float = 0.90) -> float:
    """Fade per cycle of a whole trajectory, for learning a population.

    Least squares of 1 - SOH on cycles since the first reading, through the
    origin, over readings up to the first crossing of `threshold` (all
    readings if it never crosses). NaN when the cell did not fade.
    """
    x = np.asarray(cycles, float)
    y = np.asarray(soh, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, y = x[order] - x[order][0], y[order]
    below = np.flatnonzero(y <= threshold)
    if len(below):
        x, y = x[: below[0] + 1], y[: below[0] + 1]
    denom = float(np.sum(x * x))
    if denom <= 0:
        return float("nan")
    k = float(np.sum(x * (1.0 - y)) / denom)
    return k if k > 0 else float("nan")


def fit_prior(rates: np.ndarray) -> FadePrior:
    """Log-normal summary of fade rates from cells of other datasets."""
    r = np.asarray(rates, float)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 3:
        raise ValueError(f"fit_prior: {len(r)} usable fade rates; a population needs at least 3")
    logs = np.log(r)
    return FadePrior(mean=float(np.mean(logs)), sd=max(float(np.std(logs, ddof=1)), SD_FLOOR),
                     n_cells=len(r))


def _sigma(dx: np.ndarray, y: np.ndarray) -> float:
    if len(dx) < MIN_FOR_SIGMA:
        return SIGMA_DEFAULT
    fit = np.polyval(np.polyfit(dx, y, 1), dx)
    res = y - fit
    return max(1.4826 * float(np.median(np.abs(res - np.median(res)))), SIGMA_FLOOR)


def posterior_rul(
    cycles: np.ndarray,
    soh: np.ndarray,
    at_cycle: float,
    prior: FadePrior,
    threshold: float = 0.90,
    coverage: float = 0.80,
) -> RULPosterior:
    """Remaining cycles to `threshold`, using readings at or before `at_cycle`.

    With `FadePrior.flat()` this is the cell's own data alone; with no
    readings past the first it is the population's answer alone.
    """
    x = np.asarray(cycles, float)
    y = np.asarray(soh, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x <= at_cycle)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        raise ValueError("posterior_rul: no readings at or before at_cycle")
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x0 = x[0]
    dx = x - x0
    elapsed = float(at_cycle - x0)

    if np.isfinite(prior.sd):
        center, half = prior.mean, GRID_SPAN_SD * prior.sd
    else:
        center, half = float(np.log(1e-3)), 12.0
    log_k = np.linspace(center - half, center + half, GRID_POINTS)
    k = np.exp(log_k)

    log_post = np.zeros_like(log_k) if not np.isfinite(prior.sd) else \
        -0.5 * ((log_k - prior.mean) / prior.sd) ** 2
    sigma = _sigma(dx, y)
    if len(x) > 1:
        resid = y[None, :] - (1.0 - k[:, None] * dx[None, :])
        log_post = log_post - 0.5 * np.sum(resid ** 2, axis=1) / sigma ** 2
    w = np.exp(log_post - log_post.max())
    w /= w.sum()

    rul = np.maximum((1.0 - threshold) / k - elapsed, 0.0)
    order_r = np.argsort(rul)
    cdf = np.cumsum(w[order_r])
    tail = (1.0 - coverage) / 2.0

    def q(p: float) -> float:
        return float(rul[order_r][min(np.searchsorted(cdf, p), len(cdf) - 1)])

    return RULPosterior(at_cycle=float(at_cycle), median=q(0.5), lower=q(tail),
                        upper=q(1.0 - tail), n_readings=len(x), sigma=sigma)
