"""Tests for the physics-grounded models.

The central test is `test_recovers_known_activation_energy`. Every other claim
this project makes about the Arrhenius fit — including the claim that its
negative activation energy on NASA data is a property of the data rather than
a bug — rests on the estimator being correct. That is established the only way
it can be: by generating data from the model with known parameters and
checking they come back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bms.physics import (
    GAS_CONSTANT,
    arrhenius_stability,
    assess_identifiability,
    confound_report,
    correct_thermal_measurement,
    fit_arrhenius,
    predict_arrhenius,
    thermal_baseline,
)

TRUE_EA = 45_000.0
TRUE_Z = 0.5
TRUE_LN_A = 5.0


def synthetic_arrhenius(
    n_cells: int = 20,
    n_cycles: int = 200,
    noise: float = 0.05,
    temp_noise: float = 0.5,
    seed: int = 0,
    temp_range: tuple[float, float] = (5.0, 55.0),
) -> pd.DataFrame:
    """Data generated from the model itself, with known parameters.

    `noise` perturbs the response; `temp_noise` perturbs the recorded
    temperature. They are separate because they bias the estimator
    differently: response noise is what least squares assumes and averages
    away, while noise in a *predictor* is errors-in-variables and attenuates
    the fitted coefficient no matter how many rows there are. Setting only
    `noise=0` therefore does not give exact recovery, and a test asserting it
    would be testing a false premise rather than the estimator.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for cell in range(n_cells):
        temperature = rng.uniform(*temp_range)
        for cycle in range(1, n_cycles + 1):
            fade = np.exp(
                TRUE_LN_A
                - (TRUE_EA / GAS_CONSTANT) / (temperature + 273.15)
                + TRUE_Z * np.log(cycle)
            )
            jitter = rng.normal(0, temp_noise) if temp_noise > 0 else 0.0
            scatter = np.exp(rng.normal(0, noise)) if noise > 0 else 1.0
            rows.append({
                "cell_id": f"S{cell:02d}",
                "cohort": f"C{cell % 4}",
                "cycle": cycle,
                "avg_temp": temperature + jitter,
                "ambient_temperature_c": round(temperature),
                "cumulative_fade": fade * scatter,
            })
    return pd.DataFrame(rows)


class TestArrheniusEstimator:
    def test_recovers_known_activation_energy(self):
        """The estimator must return the parameters it was generated from."""
        fit = fit_arrhenius(synthetic_arrhenius(), target="cumulative_fade")

        assert fit.activation_energy_j_per_mol == pytest.approx(TRUE_EA, rel=0.02)
        assert fit.exponent_z == pytest.approx(TRUE_Z, rel=0.05)
        assert fit.ln_a == pytest.approx(TRUE_LN_A, rel=0.05)
        assert fit.r2_log_space > 0.95
        assert fit.plausible
        assert fit.admissible

    def test_noise_free_recovery_is_exact(self):
        data = synthetic_arrhenius(noise=0.0, temp_noise=0.0)
        fit = fit_arrhenius(data, target="cumulative_fade")
        assert fit.activation_energy_j_per_mol == pytest.approx(TRUE_EA, rel=1e-6)
        assert fit.exponent_z == pytest.approx(TRUE_Z, rel=1e-6)
        assert fit.r2_log_space == pytest.approx(1.0, abs=1e-9)

    def test_temperature_measurement_noise_attenuates_the_estimate(self):
        """Errors-in-variables biases Ea downward; sample size does not fix it.

        Worth pinning because it is the standard reason a physically-motivated
        coefficient comes back too small, and distinguishing it from a genuine
        null is exactly the judgement this project has to make repeatedly.
        """
        clean = fit_arrhenius(
            synthetic_arrhenius(noise=0.0, temp_noise=0.0), target="cumulative_fade"
        )
        noisy = fit_arrhenius(
            synthetic_arrhenius(noise=0.0, temp_noise=3.0), target="cumulative_fade"
        )
        assert noisy.activation_energy_j_per_mol < clean.activation_energy_j_per_mol

    def test_rate_limiting_reading_tracks_exponent(self):
        fit = fit_arrhenius(synthetic_arrhenius(), target="cumulative_fade")
        assert "diffusion-limited" in fit.rate_limiting_reading

    def test_predictions_round_trip(self):
        data = synthetic_arrhenius(noise=0.0, temp_noise=0.0)
        fit = fit_arrhenius(data, target="cumulative_fade")
        predicted = predict_arrhenius(fit, data)
        assert np.allclose(predicted, data["cumulative_fade"].to_numpy(), rtol=1e-6)

    def test_nonpositive_fade_is_dropped_and_counted(self):
        data = synthetic_arrhenius(n_cells=5, n_cycles=50)
        data.loc[data.index[:37], "cumulative_fade"] = -0.01
        fit = fit_arrhenius(data, target="cumulative_fade")
        assert fit.n_dropped_nonpositive == 37
        assert fit.n_used == len(data) - 37

    def test_undefined_inputs_predict_nan_not_a_stand_in(self):
        """A cycle of zero has no logarithm; the prediction must be NaN."""
        data = synthetic_arrhenius(n_cells=5, n_cycles=50)
        fit = fit_arrhenius(data, target="cumulative_fade")

        broken = data.head(3).copy()
        broken["cycle"] = 0.0
        assert np.isnan(predict_arrhenius(fit, broken)).all()

    def test_too_few_rows_returns_nan_fit_rather_than_raising(self):
        tiny = synthetic_arrhenius(n_cells=1, n_cycles=2)
        fit = fit_arrhenius(tiny, target="cumulative_fade")
        assert np.isnan(fit.activation_energy_j_per_mol)
        assert not fit.admissible

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            fit_arrhenius(pd.DataFrame({"cycle": [1, 2]}), target="nope")

    def test_negative_activation_energy_is_inadmissible(self):
        """A fit saying degradation slows with heat must not pass as plausible.

        This is the NASA outcome, and the guard that keeps it from being
        reported as a result.
        """
        data = synthetic_arrhenius(noise=0.0, temp_noise=0.0)
        # Invert the temperature dependence.
        data["avg_temp"] = 60.0 - data["avg_temp"]
        fit = fit_arrhenius(data, target="cumulative_fade")
        assert fit.activation_energy_j_per_mol < 0
        assert not fit.plausible
        assert not fit.admissible


class TestIdentifiability:
    def test_clean_synthetic_data_is_identifiable(self):
        data = synthetic_arrhenius()
        data["soh"] = 1.0 - data["cumulative_fade"].clip(0, 0.9)
        report = assess_identifiability(data, temperature_col="avg_temp")
        assert report.identifiable, report.render()

    def test_temperature_confounded_with_cohort_is_refused(self):
        """One temperature per cohort makes the coefficient unidentifiable."""
        data = synthetic_arrhenius()
        # Collapse: cohort now determines temperature exactly.
        data["cohort"] = data["ambient_temperature_c"].astype(str)
        data["soh"] = 1.0 - data["cumulative_fade"].clip(0, 0.9)
        report = assess_identifiability(data, temperature_col="avg_temp")
        assert not report.identifiable
        assert any("within cohorts" in r for r in report.reasons)

    def test_early_soh_spread_is_refused(self):
        data = synthetic_arrhenius()
        data["soh"] = 1.0
        # Impose a large early-life offset by ambient level.
        cold = data["ambient_temperature_c"] < 20
        data.loc[cold, "soh"] = 0.75
        report = assess_identifiability(data, temperature_col="avg_temp")
        assert not report.identifiable
        assert any("common temperature" in r for r in report.reasons)

    def test_report_renders_reasons(self):
        data = synthetic_arrhenius()
        data["cohort"] = data["ambient_temperature_c"].astype(str)
        data["soh"] = 1.0 - data["cumulative_fade"].clip(0, 0.9)
        text = assess_identifiability(data, temperature_col="avg_temp").render()
        assert "NOT_IDENTIFIABLE" in text


class TestArrheniusStability:
    def test_per_cohort_fits_agree_on_clean_data(self):
        """Ea is a chemistry constant, so cohorts of one chemistry must agree."""
        data = synthetic_arrhenius(n_cells=24, noise=0.02)
        table = arrhenius_stability(data, target="cumulative_fade",
                                    temperature_col="avg_temp")
        assert len(table) == data["cohort"].nunique()
        assert table["plausible"].all(), table.to_string()
        assert table["ea_kj_per_mol"].std() < 5.0


class TestThermalConfound:
    def _frame_with_offset(self) -> pd.DataFrame:
        """Healthy cells whose measured SOH is depressed at low ambient."""
        rows = []
        for cell in range(12):
            ambient = [4, 24, 44][cell % 3]
            offset = {4: 0.80, 24: 0.92, 44: 1.00}[ambient]
            for cycle in range(1, 41):
                true_soh = 1.0 - 0.0005 * cycle
                rows.append({
                    "cell_id": f"T{cell:02d}",
                    "cohort": f"A{ambient}",
                    "cycle": cycle,
                    "ambient_temperature_c": ambient,
                    "avg_temp": float(ambient),
                    "soh": true_soh * offset,
                })
        return pd.DataFrame(rows)

    def test_baseline_recovers_the_offsets(self):
        table = thermal_baseline(self._frame_with_offset(), early_cycles=20)
        by_level = dict(zip(
            table["ambient_temperature_c"], table["baseline_soh"], strict=True,
        ))
        assert by_level[4] == pytest.approx(0.80, abs=0.01)
        assert by_level[24] == pytest.approx(0.92, abs=0.01)
        assert by_level[44] == pytest.approx(1.00, abs=0.01)

    def test_correction_removes_the_cross_level_spread(self):
        corrected = correct_thermal_measurement(self._frame_with_offset())
        early = corrected[corrected["cycle"] <= 20]
        spread = early.groupby("ambient_temperature_c")["soh_corrected"].mean()
        assert float(spread.max() - spread.min()) < 0.01

    def test_unmeasured_level_gets_nan_not_a_benign_offset(self):
        """An absent correction must not silently become 'no correction'."""
        data = self._frame_with_offset()
        extra = data[data["ambient_temperature_c"] == 24].copy()
        extra["ambient_temperature_c"] = 60  # never seen in the early window
        extra["cycle"] = extra["cycle"] + 100
        extra["cell_id"] = "T99"
        combined = pd.concat([data, extra], ignore_index=True)

        corrected = correct_thermal_measurement(combined)
        unseen = corrected[corrected["ambient_temperature_c"] == 60]
        assert unseen["thermal_offset"].isna().all()
        assert unseen["soh_corrected"].isna().all()

    def test_confound_report_flags_positive_early_correlation(self):
        report = confound_report(self._frame_with_offset())
        assert report["early_spearman_ambient_vs_soh"] > 0.5
        assert report["early_soh_spread_across_levels"] > 0.15

    def test_clean_data_shows_no_confound(self):
        data = self._frame_with_offset()
        data["soh"] = 1.0 - 0.0005 * data["cycle"]  # no offset at all
        report = confound_report(data)
        assert abs(report["early_soh_spread_across_levels"]) < 1e-9

    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required column"):
            thermal_baseline(pd.DataFrame({"soh": [1.0]}))
