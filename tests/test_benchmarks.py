"""Tests for the benchmark harness.

Two properties carry most of the weight here.

`test_train_mean_scores_exactly_zero` is a live self-test of the metric
itself: `r2_vs_global_mean` is defined against the training-fold mean, so a
method that predicts exactly that must score exactly zero. Any other value
means the metric is wired wrong, and every other number in the results table
would be wrong with it.

`test_unavailable_methods_appear_in_the_table` pins the reporting discipline:
a method that cannot run must produce a row saying so, never vanish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bms.benchmarks import (
    BenchmarkMethod,
    Family,
    add_targets,
    all_methods,
    check_availability,
    get,
    load_all,
    register,
    registry_frame,
    run_study,
    screen_cells_for_soh,
    signal_report,
    signal_to_noise,
)


@pytest.fixture(scope="module", autouse=True)
def _registry_loaded():
    load_all()


def cycling_frame(n_cells: int = 12, n_cycles: int = 60, seed: int = 0) -> pd.DataFrame:
    """A well-behaved fleet: monotone fade, several cohorts, mild noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        cohort = f"P{cell % 3}"
        rate = 0.002 * (1 + cell % 3)
        initial = 2.0
        for cycle in range(1, n_cycles + 1):
            capacity = initial * (1 - rate * cycle) + rng.normal(0, 0.005)
            rows.append({
                "cell_id": f"B{cell:03d}",
                "cohort": cohort,
                "cycle": cycle,
                "capacity_ah": capacity,
                "capacity_loss": rate * initial + rng.normal(0, 0.001),
                "avg_temp": 20.0 + 5 * (cell % 3) + rng.normal(0, 0.5),
                "max_temp": 30.0 + 5 * (cell % 3),
                "trailing_avg_temp": 20.0 + 5 * (cell % 3),
                "avg_stress": 5.0 + rng.normal(0, 1),
                "deep_discharge_duration": 300.0 + rng.normal(0, 20),
                "aggressive_discharge_count": 80.0 + rng.normal(0, 10),
                "avg_soc": 50.0 + rng.normal(0, 3),
                "ambient_temperature_c": 20 + 5 * (cell % 3),
                "fast_charge_duration": 0.0,  # degenerate, as on NASA
            })
    return pd.DataFrame(rows)


class TestRegistry:
    def test_duplicate_registration_is_refused(self):
        method = BenchmarkMethod(
            name="train_mean", family=Family.NAIVE, citation="x",
            build=lambda f, t: (lambda train: (lambda test: np.zeros(len(test)))),
        )
        with pytest.raises(ValueError, match="already registered"):
            register(method)

    def test_unknown_method_raises_with_the_known_list(self):
        with pytest.raises(KeyError, match="Unknown benchmark method"):
            get("no_such_method")

    def test_registry_frame_lists_every_method(self):
        frame = registry_frame()
        assert len(frame) == len(all_methods())
        assert {"method", "family", "citation", "requires_curves"} <= set(frame.columns)

    def test_families_are_all_populated(self):
        families = {m.family for m in all_methods()}
        assert {Family.NAIVE, Family.CLASSICAL, Family.CURVE, Family.PHYSICS} <= families


class TestAvailability:
    def test_curve_methods_are_unavailable_on_a_summary_frame(self):
        verdict = check_availability(get("severson_delta_q_variance"), cycling_frame())
        assert not verdict
        assert "curves" in verdict.reason
        assert verdict.status == "UNAVAILABLE"

    def test_missing_feature_is_reported_by_name(self):
        data = cycling_frame().drop(columns=["avg_temp"])
        verdict = check_availability(get("elasticnet"), data)
        assert not verdict
        assert "avg_temp" in verdict.missing_features

    def test_degenerate_feature_is_refused(self):
        """A constant feature cannot carry a coefficient."""
        verdict = check_availability(
            get("elasticnet"), cycling_frame(), features=["fast_charge_duration"],
        )
        assert not verdict
        assert "constant" in verdict.reason

    def test_runnable_method_passes(self):
        assert check_availability(get("elasticnet"), cycling_frame())

    def test_feature_free_method_is_not_treated_as_misconfigured(self):
        """`train_mean` uses no features by design, which is not an error."""
        method = get("train_mean")
        assert method.feature_free
        assert check_availability(method, cycling_frame())
        assert method.fit_fn(target="capacity_loss") is not None


class TestXGBoost:
    """The registered `xgboost` method must be actual XGBoost."""

    def test_is_findable_in_the_registry(self):
        method = get("xgboost")
        assert method.name == "xgboost"
        assert method.requires_package == ("xgboost",)

    def test_is_the_real_library_not_a_substitute(self):
        """Guards against relabelling HistGradientBoosting as XGBoost."""
        pytest.importorskip("xgboost")
        from xgboost import XGBRegressor

        from src.bms.benchmarks.classical import _xgboost
        assert isinstance(_xgboost(), XGBRegressor)

    def test_is_distinct_from_hist_gradient_boosting(self):
        pytest.importorskip("xgboost")
        from sklearn.ensemble import HistGradientBoostingRegressor

        from src.bms.benchmarks.classical import _hist_gradient_boosting, _xgboost
        assert isinstance(_hist_gradient_boosting(), HistGradientBoostingRegressor)
        assert not isinstance(_xgboost(), HistGradientBoostingRegressor)

    def test_seed_is_fixed(self):
        pytest.importorskip("xgboost")
        from src.bms.benchmarks.classical import RANDOM_SEED, _xgboost
        assert _xgboost().random_state == RANDOM_SEED

    def test_trains_and_predicts_with_correct_shape(self):
        pytest.importorskip("xgboost")
        data = add_targets(cycling_frame())
        train = data[data["cell_id"] != "B000"]
        test = data[data["cell_id"] == "B000"]

        predict = get("xgboost").fit_fn(target="cumulative_fade")(train)
        out = predict(test)

        assert out.shape == (len(test),)
        assert np.isfinite(out).all()

    def test_is_deterministic(self):
        pytest.importorskip("xgboost")
        data = add_targets(cycling_frame())
        train = data[data["cell_id"] != "B000"]
        test = data[data["cell_id"] == "B000"]

        method = get("xgboost")
        first = method.fit_fn(target="cumulative_fade")(train)(test)
        second = method.fit_fn(target="cumulative_fade")(train)(test)
        assert np.allclose(first, second)


class TestOptionalPackageGating:
    def test_missing_package_reports_unavailable_with_its_name(self):
        """One UNAVAILABLE row, not forty identical ImportErrors."""
        method = BenchmarkMethod(
            name="_needs_absent_package", family=Family.CLASSICAL, citation="x",
            build=lambda f, t: (lambda train: (lambda test: np.zeros(len(test)))),
            default_features=("avg_temp",),
            requires_package=("definitely_not_installed_xyz",),
        )
        verdict = check_availability(method, add_targets(cycling_frame()))
        assert not verdict
        assert "definitely_not_installed_xyz" in verdict.reason

    def test_present_package_does_not_block(self):
        method = BenchmarkMethod(
            name="_needs_numpy", family=Family.CLASSICAL, citation="x",
            build=lambda f, t: (lambda train: (lambda test: np.zeros(len(test)))),
            default_features=("avg_temp",),
            requires_package=("numpy",),
        )
        assert check_availability(method, add_targets(cycling_frame()))


class TestTargets:
    def test_add_targets_creates_the_soh_family(self):
        out = add_targets(cycling_frame())
        for column in ("soh", "cumulative_fade", "horizon_fade_10", "horizon_fade_50"):
            assert column in out.columns

    def test_soh_is_bounded_above_after_screening(self):
        out = add_targets(cycling_frame())
        assert out["soh"].max() <= 1.05

    def test_horizon_target_is_undefined_at_the_end_of_each_cell(self):
        out = add_targets(cycling_frame(n_cycles=60), horizons=(50,))
        per_cell = out.groupby("cell_id")["horizon_fade_50"].apply(lambda s: s.isna().sum())
        assert (per_cell == 50).all()

    def test_cell_with_too_few_cycles_is_excluded_by_name(self):
        data = cycling_frame()
        short = data[data["cell_id"] == "B000"].head(3).copy()
        short["cell_id"] = "SHORT"
        combined = pd.concat([data, short], ignore_index=True)

        screen = screen_cells_for_soh(combined)
        row = screen[screen["cell_id"] == "SHORT"].iloc[0]
        assert not row["admissible"]
        assert "below minimum" in row["reason"]

    def test_alternating_measurement_cell_is_excluded_by_step_size(self):
        """The NASA randomised-usage case: full and partial discharges interleaved.

        Note which criterion catches this. Normalising by a high quantile
        bounds SOH at 1.0 by construction, so this cell shows almost no
        overshoot — the overshoot test alone passes it. What gives it away is
        that capacity jumps by tens of percent between adjacent cycles, which
        no degradation process does.
        """
        data = cycling_frame()
        spiky = data[data["cell_id"] == "B001"].copy()
        spiky["cell_id"] = "MIXED"
        # Half the series at triple capacity — not scatter, a different
        # measurement.
        spiky.loc[spiky.index[::2], "capacity_ah"] *= 3.0
        combined = pd.concat([data, spiky], ignore_index=True)

        row = screen_cells_for_soh(combined).query("cell_id == 'MIXED'").iloc[0]
        assert not row["admissible"]
        assert row["overshoot_fraction"] <= 0.10  # the overshoot test alone passes it
        assert row["median_step_fraction"] > 0.05
        assert "median step between consecutive cycles" in row["reason"]

    def test_cell_normalised_by_a_multi_discharge_reference_is_excluded(self):
        """CALCE CS2 Type 3: a cycle index spanning six discharges at different rates.

        The reference quantile then lands on a summed multi-discharge cycle —
        5.4 Ah on a 1.1 Ah cell — and every ordinary cycle reads as ~99.7%
        degraded. Neither the overshoot nor the step test catches this: a
        too-large reference *deflates* readings smoothly rather than
        overshooting.
        """
        data = cycling_frame()
        odd = data[data["cell_id"] == "B001"].copy()
        odd["cell_id"] = "MULTI"
        # One cycle in eight sums six discharges. The fraction has to exceed
        # 5% for the 95th-percentile reference to land on a summed cycle, which
        # is what makes the reference wrong; at exactly 5% it sits on the
        # boundary and the pathology does not reproduce. In the real CS2_3 the
        # fraction is comfortably above it.
        odd.loc[odd.index[::8], "capacity_ah"] *= 6.0
        combined = pd.concat([data, odd], ignore_index=True)

        row = screen_cells_for_soh(combined).query("cell_id == 'MULTI'").iloc[0]
        assert not row["admissible"]
        assert row["overshoot_fraction"] <= 0.10   # overshoot test alone passes it
        assert row["median_step_fraction"] <= 0.05  # step test alone passes it too
        assert "median SOH across life" in row["reason"]

    def test_degenerate_cell_with_no_trajectory_is_excluded(self):
        """CALCE CX2_3: 91,279 rows whose 95th percentile is 0.003 of the maximum.

        Almost every row carries no discharge, so the reference lands on noise
        and SOH pins at 1.000 with an interquartile range of 0.0001. None of
        the other four criteria fire — the cell looks pristine and is empty —
        and it supplied 69% of the admissible rows.
        """
        data = cycling_frame()
        empty = data[data["cell_id"] == "B001"].copy()
        empty["cell_id"] = "DEGENERATE"
        # A handful of real readings; everything else near zero.
        empty["capacity_ah"] = 0.002
        empty.loc[empty.index[:2], "capacity_ah"] = 2.0
        combined = pd.concat([data, empty], ignore_index=True)

        row = screen_cells_for_soh(combined).query("cell_id == 'DEGENERATE'").iloc[0]
        assert not row["admissible"]
        assert "of this cell's maximum" in row["reason"]

    def test_normal_cell_passes_the_reference_ratio_check(self):
        screen = screen_cells_for_soh(cycling_frame())
        assert screen["admissible"].all()

    def test_median_soh_is_reported_for_admissible_cells(self):
        screen = screen_cells_for_soh(cycling_frame())
        assert screen["admissible"].all()
        assert (screen["median_soh"] > 0.40).all()

    def test_healthy_cell_is_not_excluded_by_the_median_floor(self):
        """A cell fading normally to 60% must survive."""
        data = cycling_frame(n_cells=4, n_cycles=60)
        screen = screen_cells_for_soh(data)
        assert screen["admissible"].all(), screen.to_string()

    def test_gradual_degradation_passes_the_step_test(self):
        """The threshold must not condemn ordinary cells."""
        screen = screen_cells_for_soh(cycling_frame())
        assert screen["admissible"].all()
        assert screen["median_step_fraction"].max() < 0.05

    def test_declining_trend_is_reported_but_not_required(self):
        """Screening on trend would select cells agreeing with the hypothesis."""
        data = cycling_frame()
        rising = data[data["cell_id"] == "B002"].copy()
        rising["cell_id"] = "RISING"
        rising["capacity_ah"] = 1.0 + 0.001 * rising["cycle"]
        combined = pd.concat([data, rising], ignore_index=True)

        row = screen_cells_for_soh(combined).query("cell_id == 'RISING'").iloc[0]
        assert row["admissible"]
        assert row["capacity_cycle_rho"] > 0

    def test_signal_fraction_is_high_for_a_clean_trend(self):
        out = add_targets(cycling_frame())
        assert signal_to_noise(out, "cumulative_fade").signal_fraction > 0.9

    def test_signal_fraction_is_near_zero_for_pure_noise(self):
        out = add_targets(cycling_frame())
        rng = np.random.default_rng(1)
        out["pure_noise"] = rng.normal(0, 1, len(out))
        assert signal_to_noise(out, "pure_noise").signal_fraction < 0.5

    def test_signal_report_covers_present_targets_only(self):
        out = add_targets(cycling_frame())
        report = signal_report(out, targets=("cumulative_fade", "not_a_column"))
        assert list(report["target"]) == ["cumulative_fade"]

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            add_targets(pd.DataFrame({"cell_id": ["a"], "cycle": [1]}))


class TestStudy:
    @pytest.fixture(scope="class")
    def study(self):
        data = add_targets(cycling_frame())
        methods = [get(n) for n in
                   ("train_mean", "age_linear", "elasticnet",
                    "severson_delta_q_variance")]
        return run_study(data, target="cumulative_fade", methods=methods)

    def test_train_mean_scores_exactly_zero(self, study):
        """Self-test of the metric: predicting the training mean is R2 = 0."""
        row = study.to_frame().query("method == 'train_mean'").iloc[0]
        assert row["lobo_r2"] == pytest.approx(0.0, abs=1e-9)
        assert row["loco_r2"] == pytest.approx(0.0, abs=1e-9)

    def test_unavailable_methods_appear_in_the_table(self, study):
        frame = study.to_frame()
        row = frame.query("method == 'severson_delta_q_variance'").iloc[0]
        assert row["status"] == "UNAVAILABLE"
        assert row["reason"]
        assert np.isnan(row["lobo_r2"])

    def test_every_method_produces_exactly_one_row(self, study):
        assert len(study.to_frame()) == 4

    def test_ceiling_is_attached_to_the_result(self, study):
        assert 0.0 < study.r2_ceiling <= 1.0
        assert (study.to_frame()["r2_ceiling"] == study.r2_ceiling).all()

    def test_render_mentions_the_ceiling(self, study):
        assert "ceiling" in study.render()

    def test_a_crashing_method_is_reported_as_error_not_runnable(self):
        """A method that passes availability then crashes must not read clean."""
        def exploding(features, target):
            def fit(train):
                raise RuntimeError("boom")
            return fit

        method = BenchmarkMethod(
            name="_exploding_test_method", family=Family.NAIVE, citation="x",
            build=exploding, default_features=("avg_temp",),
        )
        data = add_targets(cycling_frame())
        result = run_study(data, target="cumulative_fade", methods=[method])
        row = result.to_frame().iloc[0]
        assert row["status"] == "ERROR"
        assert "boom" in row["reason"]

    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="no target column"):
            run_study(add_targets(cycling_frame()), target="not_a_target")

    def test_target_mismatch_is_flagged_not_silently_scored(self):
        """A method scored against a quantity it does not model must say so.

        The Arrhenius power law predicts cumulative fade, which starts near
        zero. Scored against `soh`, a level near one, it reports a large error
        that describes a units mismatch rather than a modelling failure —
        which would be badly misread in a results table.
        """
        data = add_targets(cycling_frame())
        result = run_study(
            data, target="soh", methods=[get("arrhenius_avg_temp")],
        )
        row = result.to_frame().iloc[0]
        assert row["target_mismatch"]
        assert "cumulative_fade" in row["target_mismatch"]

    def test_matching_target_is_not_flagged(self):
        data = add_targets(cycling_frame())
        result = run_study(
            data, target="cumulative_fade", methods=[get("arrhenius_avg_temp")],
        )
        assert not result.to_frame().iloc[0]["target_mismatch"]

    def test_rows_missing_the_target_are_dropped_for_every_method_alike(self):
        """Otherwise a NaN-tolerant method is scored on an easier set."""
        data = add_targets(cycling_frame(), horizons=(50,))
        result = run_study(
            data, target="horizon_fade_50", methods=[get("train_mean")],
        )
        assert result.n_rows == int(data["horizon_fade_50"].notna().sum())
