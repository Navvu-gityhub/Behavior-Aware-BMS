"""How far off each shipped estimate has been, measured - not assumed.

Every value here comes from `reports/metrics/error_bands.csv`, built by
`scripts/build_error_bands.py` from the CALCE validation studies.
`tests/test_error_bands.py` fails if the constants and the file disagree.

These are EMPIRICAL bands: how the estimator actually erred on lab cells with
known capacity. They are not model-based confidence intervals, and they
describe single LCO cells at constant current - outside that, the band is a
lower bound on the uncertainty, not a guarantee.

WHY RUL IS INDEXED BY THE PREDICTION
------------------------------------
The study's headline - 73% within +/-20 cycles - is conditioned on the TRUE
remaining life being under 25 cycles. A user never knows the true value; they
see the prediction. Conditioned on the prediction, the picture is different
and more useful: when the estimate says under 25 cycles, the cell lasted at
least that long in 95% of cases, typically 23 cycles longer. Near end of life
the figure behaves as a safe lower bound, which is the right direction for a
replacement decision.
"""

from __future__ import annotations

from dataclasses import dataclass

# 90th-percentile absolute SOH error (fraction), median across cells.
SOH_P90_ABS_FIXED = 0.0343     # 17 cells, field method, 3.90-3.60 V
SOH_P90_ABS_LEARNED = 0.0493   # 2 cells, top-of-charge partial cycling
# When the readings are INCONSISTENT (field_soh.CONSISTENCY_TOLERANCE
# exceeded), from reports/metrics/calce_sufficiency/: 2,168 scored points,
# 13 cells, median 5.1% - about 4x the consistent case.
SOH_P90_ABS_INCONSISTENT = 0.2186


@dataclass(frozen=True)
class RulBand:
    """For one range of PREDICTED remaining life: where the truth fell."""

    low: float                 # predicted-RUL range, cycles
    high: float
    later_q05: float           # true - predicted, 5th percentile
    later_q50: float
    later_q95: float
    share_at_least: float      # fraction of cases the cell lasted >= prediction
    n: int
    cells: int


RUL_BANDS: tuple[RulBand, ...] = (
    RulBand(0, 25, 1, 23, 82, 0.95, 78, 15),
    RulBand(25, 50, -14, 40, 129, 0.88, 52, 14),
    RulBand(50, 100, -40, 60, 141, 0.85, 85, 12),
    RulBand(100, 200, -86, 4, 167, 0.51, 71, 7),
    RulBand(200, 100_000, -109, 162, 179, 0.63, 30, 2),
)


def soh_band(window_mode: str, consistent: bool = True) -> float:
    """90%-of-readings half-width for an SOH estimate, as a fraction."""
    if not consistent:
        return SOH_P90_ABS_INCONSISTENT
    return SOH_P90_ABS_LEARNED if window_mode == "learned" else SOH_P90_ABS_FIXED


def rul_band(predicted_cycles: float) -> RulBand:
    """The validation band for a given predicted remaining life."""
    for band in RUL_BANDS:
        if band.low <= predicted_cycles < band.high:
            return band
    return RUL_BANDS[-1]


def describe_rul_band(predicted_cycles: float) -> str:
    """One plain sentence: what happened in validation at this prediction."""
    b = rul_band(predicted_cycles)
    lo = max(predicted_cycles + b.later_q05, 0)
    hi = predicted_cycles + b.later_q95
    thin = " (few cells at this range)" if b.cells < 5 else ""
    span = f"{b.low:.0f}+" if b.high >= 100_000 else f"{b.low:.0f}-{b.high:.0f}"
    return (f"In validation, when the estimate was {span} cycles, the true "
            f"remaining life fell between {lo:.0f} and {hi:.0f} cycles in 90% of "
            f"cases, and the cell lasted at least as long as predicted "
            f"{b.share_at_least:.0%} of the time{thin}.")
