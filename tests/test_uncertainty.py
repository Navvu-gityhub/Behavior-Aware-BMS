"""Tests for conformal prediction.

The load-bearing test is `test_coverage_holds_under_exchangeability`: the
whole reason for choosing conformal prediction over a Bayesian alternative is
its finite-sample guarantee, so that guarantee is checked empirically rather
than trusted. `test_coverage_breaks_under_distribution_shift` then confirms it
fails when its assumption is violated, which is the property this project
actually exploits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bms.uncertainty import (
    Interval,
    IntervalRefusal,
    MondrianConformal,
    SplitConformal,
    conformal_quantile,
    coverage_report,
    coverage_summary,
    empirical_coverage,
)

TARGET = "y"


def linear_fit_fn(features=("x",), target=TARGET):
    """A plain OLS `FitFn`, matching the harness's contract."""
    columns = list(features)

    def fit(train: pd.DataFrame):
        matrix = np.column_stack(
            [train[c].to_numpy(float) for c in columns] + [np.ones(len(train))]
        )
        coefficients, *_ = np.linalg.lstsq(
            matrix, train[target].to_numpy(float), rcond=None
        )

        def predict(test: pd.DataFrame) -> np.ndarray:
            test_matrix = np.column_stack(
                [test[c].to_numpy(float) for c in columns] + [np.ones(len(test))]
            )
            return test_matrix @ coefficients
        return predict
    return fit


def exchangeable_frame(n_cells: int = 40, n_rows: int = 60, seed: int = 0):
    """One homogeneous population — the assumption conformal needs."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        for _ in range(n_rows):
            x = rng.normal(0, 1)
            rows.append({
                "cell_id": f"C{cell:03d}",
                "cohort": f"K{cell % 5}",
                "x": x,
                TARGET: 2.0 * x + rng.normal(0, 1.0),
            })
    return pd.DataFrame(rows)


class TestConformalQuantile:
    def test_finite_sample_correction_is_applied(self):
        """The rank is ceil((n+1)(1-alpha)), not the plain empirical quantile."""
        scores = np.arange(1.0, 101.0)  # n = 100
        # ceil(101 * 0.9) = 91 -> the 91st smallest, which is 91.0
        assert conformal_quantile(scores, alpha=0.1) == 91.0

    def test_returns_infinity_when_rank_exceeds_sample(self):
        """Too few points to justify any finite interval."""
        assert conformal_quantile(np.arange(5.0), alpha=0.01) == float("inf")

    def test_empty_scores_give_infinity(self):
        assert conformal_quantile(np.array([]), alpha=0.1) == float("inf")

    def test_ignores_non_finite_scores(self):
        scores = np.array([1.0, 2.0, np.nan, 3.0, np.inf])
        assert np.isfinite(conformal_quantile(scores, alpha=0.5))


class TestSplitConformal:
    def test_coverage_holds_under_exchangeability(self):
        """The guarantee: >= 1 - alpha coverage when the assumption holds."""
        data = exchangeable_frame()
        train = data[data["cell_id"] < "C030"]
        test = data[data["cell_id"] >= "C030"]

        predictor = SplitConformal(linear_fit_fn(), target=TARGET, alpha=0.1).fit(train)
        intervals = predictor.predict(test)
        stats = empirical_coverage(intervals, test[TARGET].to_numpy())

        assert stats["coverage"] >= 0.85, stats
        assert stats["n_refused"] == 0

    def test_tighter_alpha_gives_wider_intervals(self):
        data = exchangeable_frame()
        widths = {}
        for alpha in (0.2, 0.05):
            predictor = SplitConformal(
                linear_fit_fn(), target=TARGET, alpha=alpha
            ).fit(data)
            intervals = predictor.predict(data.head(50))
            widths[alpha] = intervals[0].width
        assert widths[0.05] > widths[0.2]

    def test_coverage_breaks_under_distribution_shift(self):
        """Exchangeability is required; violating it loses the guarantee.

        This is the property the whole project exploits — a model calibrated
        on one protocol does not carry its guarantee to another.
        """
        data = exchangeable_frame()
        shifted = data.copy()
        # A genuinely different regime: different slope and much more noise.
        rng = np.random.default_rng(7)
        shifted[TARGET] = -3.0 * shifted["x"] + rng.normal(0, 4.0, len(shifted))

        predictor = SplitConformal(linear_fit_fn(), target=TARGET, alpha=0.1).fit(data)
        stats = empirical_coverage(
            predictor.predict(shifted), shifted[TARGET].to_numpy()
        )
        assert stats["coverage"] < 0.9

    def test_refuses_when_calibration_is_too_small(self):
        tiny = exchangeable_frame(n_cells=4, n_rows=3)
        predictor = SplitConformal(
            linear_fit_fn(), target=TARGET, alpha=0.1, group_col=None,
        ).fit(tiny)
        result = predictor.predict(tiny)
        assert all(isinstance(r, IntervalRefusal) for r in result)

    def test_refuses_when_interval_is_too_wide(self):
        data = exchangeable_frame()
        predictor = SplitConformal(linear_fit_fn(), target=TARGET, alpha=0.1).fit(data)
        result = predictor.predict(data.head(20), max_useful_width=0.001)
        assert all(isinstance(r, IntervalRefusal) for r in result)
        assert "too wide" in result[0].reason

    def test_refusal_is_falsy_and_interval_is_truthy(self):
        assert not IntervalRefusal(reason="x")
        assert Interval(prediction=1.0, lower=0.0, upper=2.0).contains(1.5)

    def test_predict_before_fit_raises(self):
        predictor = SplitConformal(linear_fit_fn(), target=TARGET)
        with pytest.raises(RuntimeError, match="before fit"):
            predictor.predict(exchangeable_frame().head(5))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be"):
            SplitConformal(linear_fit_fn(), target=TARGET, alpha=1.5)

    def test_split_is_by_group_not_by_row(self):
        """Rows of one cell are autocorrelated; splitting by row leaks."""
        data = exchangeable_frame(n_cells=10, n_rows=50)
        predictor = SplitConformal(linear_fit_fn(), target=TARGET, group_col="cell_id")
        proper, calibration = predictor._split(data)
        overlap = set(proper["cell_id"]) & set(calibration["cell_id"])
        assert not overlap


class TestMondrianConformal:
    def test_refuses_unseen_cohort_rather_than_pooling(self):
        data = exchangeable_frame()
        predictor = MondrianConformal(
            linear_fit_fn(), target=TARGET, alpha=0.1,
        ).fit(data)

        unseen = data.head(10).copy()
        unseen["cohort"] = "NEVER_SEEN"
        result = predictor.predict(unseen)

        assert all(isinstance(r, IntervalRefusal) for r in result)
        assert "no calibration data for this cohort" in result[0].reason

    def test_produces_a_quantile_per_cohort(self):
        data = exchangeable_frame()
        predictor = MondrianConformal(linear_fit_fn(), target=TARGET).fit(data)
        assert len(predictor.quantiles) >= 1
        assert all(np.isfinite(q) for q in predictor.quantiles.values())

    def test_missing_cohort_column_raises(self):
        data = exchangeable_frame().drop(columns=["cohort"])
        predictor = MondrianConformal(linear_fit_fn(), target=TARGET)
        with pytest.raises(ValueError, match="missing cohort column"):
            predictor.fit(data)


class TestEmpiricalCoverage:
    def test_refusals_are_excluded_from_coverage_and_counted(self):
        intervals = [
            Interval(prediction=1.0, lower=0.0, upper=2.0),
            IntervalRefusal(reason="nope"),
        ]
        stats = empirical_coverage(intervals, np.array([1.5, 99.0]))
        assert stats["coverage"] == 1.0
        assert stats["n_quoted"] == 1.0
        assert stats["n_refused"] == 1.0
        assert stats["refusal_rate"] == 0.5

    def test_all_refused_gives_nan_coverage_not_zero(self):
        stats = empirical_coverage([IntervalRefusal(reason="x")], np.array([1.0]))
        assert np.isnan(stats["coverage"])
        assert stats["refusal_rate"] == 1.0


class TestCoverageReport:
    def test_reports_both_splits(self):
        data = exchangeable_frame(n_cells=15, n_rows=40)
        report = coverage_report(data, linear_fit_fn(), target=TARGET, alpha=0.1)
        assert set(report["split"].unique()) == {"LOBO", "LOCO"}

    def test_summary_exposes_worst_group_not_just_median(self):
        """The median can look fine while one group fails badly."""
        report = pd.DataFrame([
            {"split": "LOCO", "held_out": "good1", "coverage": 1.0,
             "mean_width": 1.0, "refusal_rate": 0.0, "nominal": 0.9, "error": None},
            {"split": "LOCO", "held_out": "good2", "coverage": 0.99,
             "mean_width": 1.0, "refusal_rate": 0.0, "nominal": 0.9, "error": None},
            {"split": "LOCO", "held_out": "BAD", "coverage": 0.20,
             "mean_width": 1.0, "refusal_rate": 0.0, "nominal": 0.9, "error": None},
        ])
        summary = coverage_summary(report).iloc[0]

        assert summary["median_coverage"] == pytest.approx(0.99)
        assert summary["median_shortfall"] < 0          # median looks fine
        assert summary["min_coverage"] == pytest.approx(0.20)
        assert summary["worst_shortfall"] == pytest.approx(0.70)
        assert summary["worst_group"] == "BAD"
        assert summary["fraction_below_nominal"] == pytest.approx(1 / 3)

    def test_empty_report_returns_empty_summary(self):
        assert coverage_summary(pd.DataFrame()).empty
