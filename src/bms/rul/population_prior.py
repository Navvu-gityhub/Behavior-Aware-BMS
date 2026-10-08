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
    if len(dx) < MIN_FOR_SIGMA or np.ptp(dx) == 0:
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


# ===========================================================================
# v2: power-law shape and honest noise (scripts/run_population_rul_study_v2.py)
# ===========================================================================
# v1 (above) failed its preset criteria on four held-out datasets: a line
# through SOH = 1 is the wrong shape - early fade ran 2x the life average on
# CALCE and M50T and slower on NASA and Oxford - and noise taken from
# detrended scatter understated the model's misfit 3x while ignoring that
# consecutive residuals are 0.77-0.97 correlated. v2 changes exactly those.
#
#     1 - SOH(x) = (1 - threshold) * (x / L) ** p
#
# L is the cycle at which the cell reaches `threshold` - the quantity wanted -
# and p the shape: p < 1 front-loaded fade, p > 1 accelerating. The
# population prior is independent normals on log L and p. The noise is the
# RMS residual of the best-fitting curve itself, inflated by
# (1 + rho) / (1 - rho) for its lag-1 autocorrelation, since n correlated
# residuals carry the information of far fewer independent ones.

P_GRID = np.linspace(0.3, 3.0, 55)
L_GRID_POINTS = 241
P_SD_FLOOR = 0.2
RHO_MAX = 0.95


@dataclass(frozen=True)
class LifePrior:
    """Population of (log end-of-life cycle, shape) for one threshold."""

    log_l_mean: float
    log_l_sd: float
    p_mean: float
    p_sd: float
    n_cells: int

    @classmethod
    def flat(cls) -> LifePrior:
        return cls(float(np.log(500.0)), float("inf"), 1.0, float("inf"), 0)


def fit_life_curve(cycles: np.ndarray, soh: np.ndarray, threshold: float = 0.90) -> tuple[float, float]:
    """(L, p) of one whole trajectory, for learning a population.

    Least squares over readings up to the first crossing of `threshold`,
    grid over p with the scale solved in closed form. NaNs if it never fades.
    """
    x = np.asarray(cycles, float)
    y = np.asarray(soh, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan"), float("nan")
    o = np.argsort(x, kind="stable")
    x, y = x[o] - x[o][0], y[o]
    below = np.flatnonzero(y <= threshold)
    if len(below):
        x, y = x[: below[0] + 1], y[: below[0] + 1]
    fade = 1.0 - y
    best = (np.inf, float("nan"), float("nan"))
    for p in P_GRID:
        xp = x ** p
        d = float(np.sum(xp * xp))
        if d <= 0:
            continue
        c = float(np.sum(xp * fade) / d)
        if c <= 0:
            continue
        sse = float(np.sum((fade - c * xp) ** 2))
        if sse < best[0]:
            best = (sse, ((1.0 - threshold) / c) ** (1.0 / p), float(p))
    return best[1], best[2]


def fit_life_prior(lives: np.ndarray, shapes: np.ndarray) -> LifePrior:
    """Population summary from cells of other datasets."""
    lv, sh = np.asarray(lives, float), np.asarray(shapes, float)
    ok = np.isfinite(lv) & (lv > 0) & np.isfinite(sh)
    if ok.sum() < 3:
        raise ValueError(f"fit_life_prior: {int(ok.sum())} usable cells; a population needs at least 3")
    logs = np.log(lv[ok])
    return LifePrior(float(np.mean(logs)), max(float(np.std(logs, ddof=1)), SD_FLOOR),
                     float(np.mean(sh[ok])), max(float(np.std(sh[ok], ddof=1)), P_SD_FLOOR),
                     int(ok.sum()))


def posterior_life(
    cycles: np.ndarray,
    soh: np.ndarray,
    at_cycle: float,
    prior: LifePrior,
    threshold: float = 0.90,
    coverage: float = 0.80,
) -> RULPosterior:
    """Remaining cycles to `threshold` under the v2 model, readings <= at_cycle."""
    x = np.asarray(cycles, float)
    y = np.asarray(soh, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x <= at_cycle)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        raise ValueError("posterior_life: no readings at or before at_cycle")
    o = np.argsort(x, kind="stable")
    x, y = x[o], y[o]
    dx = x - x[0]
    elapsed = float(at_cycle - x[0])

    flat = not np.isfinite(prior.log_l_sd)
    if flat:
        center, half = float(np.log(500.0)), 6.0
    else:
        center, half = prior.log_l_mean, GRID_SPAN_SD * prior.log_l_sd
    has_data = len(x) > 1 and np.ptp(dx) > 0

    def evaluate(log_l: np.ndarray, p_grid: np.ndarray, sigma: float | None):
        lg, pg = np.meshgrid(log_l, p_grid, indexing="ij")
        log_prior = np.zeros_like(lg)
        if not flat:
            log_prior -= 0.5 * ((lg - prior.log_l_mean) / prior.log_l_sd) ** 2
            log_prior -= 0.5 * ((pg - prior.p_mean) / prior.p_sd) ** 2
        if not has_data:
            return lg, pg, log_prior, SIGMA_DEFAULT
        pred = 1.0 - (1.0 - threshold) * (dx[None, None, :] / np.exp(lg)[..., None]) ** pg[..., None]
        sse = np.sum((y[None, None, :] - pred) ** 2, axis=2)
        if sigma is None:
            sigma = SIGMA_DEFAULT
            if len(x) >= MIN_FOR_SIGMA:
                i, j = np.unravel_index(np.argmin(sse - 2.0 * SIGMA_DEFAULT ** 2 * log_prior), sse.shape)
                res = y - pred[i, j]
                rho = float(np.corrcoef(res[:-1], res[1:])[0, 1]) if np.std(res) > 0 else 0.0
                rho = min(max(rho if np.isfinite(rho) else 0.0, 0.0), RHO_MAX)
                sigma = max(float(np.sqrt(np.mean(res ** 2))), SIGMA_FLOOR) * float(
                    np.sqrt((1 + rho) / (1 - rho)))
        return lg, pg, log_prior - 0.5 * sse / sigma ** 2, sigma

    # Coarse pass over the prior's range, then a fine pass over where the
    # posterior actually lies: with a long clean history it is narrower than
    # any fixed grid spacing, and a one-point posterior has no interval.
    log_l = np.linspace(center - half, center + half, L_GRID_POINTS)
    lg, pg, log_post, sigma = evaluate(log_l, P_GRID, None)
    w = np.exp(log_post - log_post.max())
    w /= w.sum()

    def span(values: np.ndarray, marginal: np.ndarray, step: float) -> tuple[float, float]:
        cdf = np.cumsum(marginal)
        lo = values[min(np.searchsorted(cdf, 1e-4), len(values) - 1)] - step
        hi = values[min(np.searchsorted(cdf, 1 - 1e-4), len(values) - 1)] + step
        return float(lo), float(hi)

    l_lo, l_hi = span(log_l, w.sum(axis=1), log_l[1] - log_l[0])
    p_lo, p_hi = span(P_GRID, w.sum(axis=0), P_GRID[1] - P_GRID[0])
    lg, pg, log_post, _ = evaluate(np.linspace(l_lo, l_hi, L_GRID_POINTS),
                                   np.linspace(max(p_lo, 0.05), p_hi, len(P_GRID)), sigma)
    w = np.exp(log_post - log_post.max()).ravel()
    w /= w.sum()
    rul = np.maximum(np.exp(lg).ravel() - elapsed, 0.0)
    order = np.argsort(rul)
    cdf = np.cumsum(w[order])
    tail = (1.0 - coverage) / 2.0

    def q(prob: float) -> float:
        return float(rul[order][min(np.searchsorted(cdf, prob), len(cdf) - 1)])

    return RULPosterior(float(at_cycle), q(0.5), q(tail), q(1.0 - tail), len(x), float(sigma))
