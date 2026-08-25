"""Distribution-free uncertainty quantification.

Conformal prediction is used rather than a Bayesian or ensemble alternative
because its guarantee holds for a misspecified model, and this project's
models are known to be misspecified. See `conformal.py` for the argument and
for what the guarantee does *not* survive.
"""

from src.bms.uncertainty.conformal import (
    DEFAULT_ALPHA,
    Interval,
    IntervalRefusal,
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
    coverage_report,
    coverage_summary,
    empirical_coverage,
)

__all__ = [
    "DEFAULT_ALPHA",
    "Interval",
    "IntervalRefusal",
    "MondrianConformal",
    "SplitConformal",
    "conformal_quantile",
    "coverage_report",
    "coverage_summary",
    "empirical_coverage",
]
