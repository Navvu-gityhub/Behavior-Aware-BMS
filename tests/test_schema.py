import pandas as pd

from src.bms.preprocessing.schema import standardize_validate_bms_data


def test_standardize_validate_bms_data_passes_for_sample():
    raw = pd.DataFrame(
        {
            "Battery_ID": ["B0005", "B0005", "B0005"],
            "Cycle_Index": [1, 1, 1],
            "Voltage_measured": [4.10, 3.95, 3.80],
            "Current_measured": [-1.0, -1.1, -1.2],
            "Temperature_measured": [27.1, 27.4, 27.9],
            "Capacity": [1.86, 1.86, 1.86],
            "SOC": [95, 70, 18],
        }
    )

    clean, issues = standardize_validate_bms_data(raw, dataset="nasa", source_file="sample.csv")

    assert issues == []
    assert "voltage_v" in clean.columns
    assert "current_a" in clean.columns
    assert "temperature_c" in clean.columns
    assert "power_w" in clean.columns
    assert list(clean["soc_band"]) == ["high_90_100", "normal_20_80", "low_0_20"]
    assert list(clean["mode_guess"]) == ["discharge", "discharge", "discharge"]


def test_missing_required_column_is_reported():
    raw = pd.DataFrame(
        {
            "cell_id": ["cell_1"],
            "cycle": [1],
            "voltage_v": [3.7],
            "current_a": [0.5],
        }
    )

    _, issues = standardize_validate_bms_data(raw, dataset="simulated")

    assert any("temperature_c" in issue for issue in issues)
