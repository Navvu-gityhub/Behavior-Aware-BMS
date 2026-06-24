# V1 Modeling Scope

## Dataset

NASA Battery Dataset

## Prediction Target

stress_score

## Features

- voltage_v
- current_a
- temperature_c
- power_w
- soc
- fast_charge_event
- deep_discharge_event
- high_temp_event
- aggressive_discharge_event
- high_soc_duration

## Models

- HistGradientBoostingRegressor
- RandomForestRegressor

## Scope Freeze

CALCE and Stanford datasets are excluded from V1 because they are not yet integrated into the preprocessing pipeline.

Future versions will use CALCE and Stanford for SOH prediction and cross-dataset validation.
