"""Tests for declarative dataset specifications.

Three tests encode findings that took real investigation and must not silently
regress:

`test_nasa_to_stanford_is_predicted_infeasible` — the orthogonality result from
batch 7, now reachable from published metadata alone. NASA varies temperature
and cannot fit charge rate; Severson varies charge rate and holds temperature.

`test_nasa_to_calce_cs2_is_infeasible_on_temperature_but_feasible_on_depth` —
the CALCE equivalent. CS2 is room-temperature only, so it cannot receive a
temperature coefficient, but it does vary depth of discharge, and NASA's
`deep_discharge_duration` is far from constant. Depth of discharge, not
temperature, is where a NASA-to-CALCE transfer is well posed.

`test_cx2_4_is_the_only_thermally_admissible_calce_target` — CX2_4 is the one
CALCE cell cycled across temperatures, which makes it admissible on that axis
and simultaneously limits it to n=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bms.adaptive.dataset_specs import (
    CALCE_CS2_SPEC,
    CALCE_CX2_4_THERMAL_SPEC,
    NASA_SPEC,
    REGISTRY,
    STANFORD_SEVERSON_SPEC,
    Axis,
    DatasetSpec,
    Variation,
    VariationProfile,
    feasibility_matrix,
    get_spec,
    predict_transfer_feasibility,
)


# ---------------------------------------------------------------------------
# Variation semantics
# ---------------------------------------------------------------------------

def test_a_fixed_axis_can_neither_fit_nor_receive_a_coefficient():
    assert Variation.FIXED.can_fit_a_coefficient is False
    assert Variation.FIXED.can_receive_a_coefficient is False


def test_an_absent_axis_is_unusable_on_both_sides():
    assert Variation.ABSENT.can_fit_a_coefficient is False
    assert Variation.ABSENT.can_receive_a_coefficient is False


def test_an_incidental_axis_is_usable_but_marginal():
    """Severson's cell temperature moves from self-heating, not by design."""
    assert Variation.INCIDENTAL.can_fit_a_coefficient is True
    assert Variation.INCIDENTAL.can_receive_a_coefficient is True


def test_an_unspecified_axis_defaults_to_absent():
    profile = VariationProfile(axes={Axis.CHARGE_RATE: Variation.VARIED})
    assert profile.get(Axis.AMBIENT_TEMPERATURE) is Variation.ABSENT


# ---------------------------------------------------------------------------
# Column mapping is declarative
# ---------------------------------------------------------------------------

def test_column_mapping_renames_present_columns_only():
    raw = pd.DataFrame({
        "Cycle_Index": [1, 2],
        "Discharge_Capacity(Ah)": [1.1, 1.09],
        "SomethingElse": [0, 0],
    })
    mapped = CALCE_CS2_SPEC.rename(raw)
    assert "cycle" in mapped.columns
    assert "capacity_ah" in mapped.columns
    # Unmapped columns survive untouched rather than being dropped.
    assert "SomethingElse" in mapped.columns


def test_missing_declared_columns_are_reported():
    raw = pd.DataFrame({"Cycle_Index": [1, 2]})
    missing = CALCE_CS2_SPEC.missing_columns(raw)
    assert "Discharge_Capacity(Ah)" in missing
    assert "Cycle_Index" not in missing


def test_a_new_dataset_needs_only_a_spec():
    """Adding a dataset must be configuration, not code."""
    spec = DatasetSpec(
        name="my_fleet",
        description="internal fleet export",
        column_map={"pack_temp_degC": "avg_temp", "cyc": "cycle"},
        variation=VariationProfile(axes={Axis.AMBIENT_TEMPERATURE: Variation.VARIED}),
    )
    raw = pd.DataFrame({"pack_temp_degC": [30.0, 31.0], "cyc": [1, 2]})
    mapped = spec.rename(raw)
    assert list(mapped.columns) == ["avg_temp", "cycle"]

    prediction = predict_transfer_feasibility(
        NASA_SPEC, spec, axes=[Axis.AMBIENT_TEMPERATURE]
    )
    assert prediction.feasible is True


# ---------------------------------------------------------------------------
# The findings this module exists to preserve
# ---------------------------------------------------------------------------

def test_nasa_to_stanford_is_predicted_infeasible_on_the_designed_axes():
    """The orthogonality result, reachable from metadata alone.

    NASA varies ambient temperature and holds charge rate. Severson varies
    charge policy and holds ambient temperature in a 30 C chamber. Neither can
    carry a model fitted on the other's experimental variable.
    """
    prediction = predict_transfer_feasibility(
        NASA_SPEC, STANFORD_SEVERSON_SPEC,
        axes=[Axis.AMBIENT_TEMPERATURE, Axis.CHARGE_RATE],
    )
    assert prediction.feasible is False
    assert prediction.status == "PREDICTED_NOT_FEASIBLE"

    by_axis = {v.axis: v for v in prediction.verdicts}
    # Temperature: held in the target, so a coefficient has nothing to act on.
    assert by_axis[Axis.AMBIENT_TEMPERATURE].usable is False
    assert "nothing to act on" in by_axis[Axis.AMBIENT_TEMPERATURE].reason
    # Charge rate: NASA cannot fit it at all.
    assert by_axis[Axis.CHARGE_RATE].usable is False
    assert "no coefficient can be fitted" in by_axis[Axis.CHARGE_RATE].reason


def test_nasa_to_calce_cs2_fails_on_temperature_but_works_on_depth_of_discharge():
    """CS2 is room-temperature only, but it does vary depth of discharge.

    This is the actionable half of the CALCE finding: the axis on which a
    NASA-to-CALCE transfer is well posed is depth of discharge, not temperature.
    """
    prediction = predict_transfer_feasibility(
        NASA_SPEC, CALCE_CS2_SPEC,
        axes=[Axis.AMBIENT_TEMPERATURE, Axis.DEPTH_OF_DISCHARGE],
    )
    by_axis = {v.axis: v for v in prediction.verdicts}

    assert by_axis[Axis.AMBIENT_TEMPERATURE].usable is False
    assert by_axis[Axis.DEPTH_OF_DISCHARGE].usable is True
    # So the transfer as a whole is feasible, just not on temperature.
    assert prediction.feasible is True


def test_cx2_4_is_the_only_thermally_admissible_calce_target():
    """One CALCE cell was cycled across temperatures, and only one."""
    prediction = predict_transfer_feasibility(
        NASA_SPEC, CALCE_CX2_4_THERMAL_SPEC, axes=[Axis.AMBIENT_TEMPERATURE]
    )
    assert prediction.feasible is True

    # And the n=1 limit must travel with the spec, not live in a chat log.
    assert CALCE_CX2_4_THERMAL_SPEC.n_cells == 1
    assert any("n=1" in c for c in CALCE_CX2_4_THERMAL_SPEC.caveats)
    assert any(
        "cannot support any cell-level" in c
        for c in CALCE_CX2_4_THERMAL_SPEC.caveats
    )


def test_nasas_degenerate_charge_axis_is_recorded_on_the_spec():
    assert NASA_SPEC.variation.get(Axis.CHARGE_RATE) is Variation.FIXED
    assert any("fast_charge_duration is degenerate" in c for c in NASA_SPEC.caveats)


def test_stanford_records_temperature_as_incidental_not_varied():
    """Recorded and moving is not the same as experimentally varied."""
    assert (
        STANFORD_SEVERSON_SPEC.variation.get(Axis.AMBIENT_TEMPERATURE)
        is Variation.FIXED
    )
    assert any("order of magnitude below" in c for c in STANFORD_SEVERSON_SPEC.caveats)


# ---------------------------------------------------------------------------
# Marginal cases are labelled, not hidden
# ---------------------------------------------------------------------------

def test_a_transfer_resting_only_on_incidental_variation_is_marginal():
    incidental_target = DatasetSpec(
        name="chamber_held",
        description="temperature recorded but not varied",
        column_map={},
        variation=VariationProfile(
            axes={Axis.AMBIENT_TEMPERATURE: Variation.INCIDENTAL}
        ),
    )
    prediction = predict_transfer_feasibility(
        NASA_SPEC, incidental_target, axes=[Axis.AMBIENT_TEMPERATURE]
    )
    assert prediction.feasible is True
    assert prediction.status == "PREDICTED_MARGINAL"
    assert Axis.AMBIENT_TEMPERATURE in prediction.marginal_axes


def test_the_prediction_says_it_is_only_a_prediction():
    """It must not be mistakable for a measurement on real data."""
    rendered = predict_transfer_feasibility(
        NASA_SPEC, STANFORD_SEVERSON_SPEC
    ).render()
    assert "predicted from published metadata" in rendered
    assert "assess_commensurability" in rendered


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_every_registered_spec_carries_a_citation():
    for name, spec in REGISTRY.items():
        assert spec.citation, f"{name} has no citation"


def test_every_non_nasa_spec_declares_its_caveats():
    for name, spec in REGISTRY.items():
        assert spec.caveats, f"{name} declares no caveats"


def test_unknown_spec_raises_with_the_known_list():
    with pytest.raises(KeyError, match="Known:"):
        get_spec("does_not_exist")


def test_feasibility_matrix_covers_every_target():
    matrix = feasibility_matrix("nasa")
    assert not matrix.empty
    assert set(matrix["target"]) == set(REGISTRY) - {"nasa"}
    assert {"axis", "usable", "marginal", "reason"} <= set(matrix.columns)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------

def test_feasibility_command_runs_in_predicted_mode(capsys):
    from src.bms.adaptive.__main__ import main

    assert main(["feasibility"]) == 0
    out = capsys.readouterr().out
    assert "stanford_severson" in out
    assert "PREDICTED" in out
    # It must not be mistakable for a measurement.
    assert "Predicted from published metadata" in out


def test_feasibility_command_runs_in_measured_mode(capsys):
    from src.bms.adaptive.__main__ import main

    if not (Path(__file__).resolve().parents[1]
            / "reports/metrics/continuous_model_training_data.csv").exists():
        pytest.skip("NASA training frame not present")

    assert main(["feasibility", "--measured"]) == 0
    out = capsys.readouterr().out
    assert "source cohort:" in out


def test_feasibility_command_rejects_an_unknown_cohort(capsys):
    from src.bms.adaptive.__main__ import main

    if not (Path(__file__).resolve().parents[1]
            / "reports/metrics/continuous_model_training_data.csv").exists():
        pytest.skip("NASA training frame not present")

    assert main(["feasibility", "--measured", "--reference", "NO_SUCH"]) == 1
    assert "Unknown cohort" in capsys.readouterr().out
