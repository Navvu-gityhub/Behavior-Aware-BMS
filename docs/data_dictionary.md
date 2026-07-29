# Day 4 Data Dictionary — Unified BMS Schema

## Purpose

This document defines the unified data schema for the Behavior-Aware EV Battery Health Monitoring and Usage Optimization System.

The goal of Day 4 is to make NASA, CALCE, Stanford, and future BMS/OBD/CAN datasets follow one common structure before feature extraction, risk scoring, and model training.

A unified schema is important because BMS datasets use different names, units, formats, and missing-value styles. Without a standard schema, later modules such as degradation-risk estimation, SOC/SOH analysis, and recommendation generation will fail or produce inconsistent results.

## Scientific basis

A Battery Management System commonly monitors voltage, current, and temperature, and uses these values for protection, state estimation, charging/discharging management, communication, and data logging. Our software layer keeps these core BMS variables and adds dataset, cell, behavior, and label fields needed for user-behavior analytics.

## Canonical schema

| Column name | Type | Unit / format | Required | Null rule | Description | Example |
|---|---:|---|---:|---|---|---|
| `dataset` | string | text | Yes | Must not be null | Dataset source name such as NASA, CALCE, Stanford, simulated, or OBD. | `nasa` |
| `source_file` | string | text | No | Can be null | Original file name or path used for traceability. | `B0005.mat` |
| `cell_id` | string | text | Yes | Must not be null | Cell, module, pack, or vehicle identifier. | `B0005` |
| `timestamp` | datetime/string | ISO-8601 preferred | Conditional | Required if `cycle` is missing | Measurement timestamp. Used for time-series ordering. | `2026-06-05T09:00:00` |
| `cycle` | integer | cycle number | Conditional | Required if `timestamp` is missing | Charge/discharge cycle number. Used for battery aging progression. | `42` |
| `voltage_v` | float | V | Yes | Must be numeric; null only allowed before imputation | Cell/module/pack voltage in volts. | `3.72` |
| `current_a` | float | A | Yes | Must be numeric; null only allowed before imputation | Battery current in amperes. Positive/negative sign convention should be documented per dataset. | `-1.25` |
| `temperature_c` | float | °C | Yes | Must be numeric; null only allowed before imputation | Battery temperature in Celsius. | `31.4` |
| `capacity_ah` | float | Ah | Recommended | Can be null for real-time telemetry | Available or measured capacity. Important for SOH estimation. | `1.82` |
| `resistance_ohm` | float | Ω | Optional | Can be null | Internal resistance or DCIR if available. | `0.045` |
| `impedance_ohm` | float | Ω | Optional | Can be null | AC impedance or derived impedance indicator if available. | `0.051` |
| `soc` | float | % | Recommended | Can be null if not available; can be estimated later | State of Charge from dataset, BMS, or later estimator. Expected range: 0–100. | `76.5` |
| `soh` | float | % | Optional | Can be null | State of Health from dataset or estimated model. Expected range: 0–100. | `91.2` |
| `soc_band` | string | category | Derived | Can be derived from `soc` | SOC range label used for behavior analytics. | `high_80_100` |
| `mode_guess` | string | category | Derived | Can be derived from current sign | Estimated operating mode: charge, discharge, rest, unknown. | `discharge` |
| `power_w` | float | W | Derived | Can be derived from voltage and current | Electrical power calculated as voltage × current. | `-4.65` |
| `label` | string | category | Optional | Can be null | Target label for ML classification/regression if available. | `high_degradation_risk` |
| `notes` | string | text | Optional | Can be null | Any dataset-specific remarks. | `fast charge session` |

## Minimum required columns for Day 4 validation

At least these fields must exist after standardization:

```text
dataset
cell_id
voltage_v
current_a
temperature_c
```

In addition, at least one ordering field must exist:

```text
cycle OR timestamp
```

## Unit convention

All datasets should be converted into these standard units before entering the feature extraction pipeline:

| Measurement | Standard unit |
|---|---|
| Voltage | volts (`V`) |
| Current | amperes (`A`) |
| Temperature | Celsius (`°C`) |
| Capacity | ampere-hour (`Ah`) |
| Resistance / impedance | ohm (`Ω`) |
| SOC / SOH | percentage (`%`) |
| Power | watts (`W`) |

## SOC band rules

| SOC range | `soc_band` label | Battery usage meaning |
|---:|---|---|
| SOC < 20 | `low_0_20` | Deep-discharge region |
| 20 ≤ SOC < 80 | `normal_20_80` | Preferred operating region |
| 80 ≤ SOC < 90 | `elevated_80_90` | Higher SOC region |
| SOC ≥ 90 | `high_90_100` | High-SOC storage / calendar-aging risk region |
| Missing SOC | `unknown` | SOC not available yet |

## Mode inference rule

The `mode_guess` field is inferred from current if the dataset does not provide an operating mode.

| Current condition | `mode_guess` |
|---|---|
| `current_a > +0.02` | `charge` |
| `current_a < -0.02` | `discharge` |
| `-0.02 ≤ current_a ≤ +0.02` | `rest` |
| missing current | `unknown` |

> Note: Some datasets use the opposite sign convention for current. The loader must correct the sign convention before schema validation if required.

## Null-handling policy

1. Required identity fields (`dataset`, `cell_id`) must not be null.
2. Required sensor fields (`voltage_v`, `current_a`, `temperature_c`) must be present and numeric.
3. Either `cycle` or `timestamp` must exist for ordering.
4. Optional battery-health fields such as `capacity_ah`, `resistance_ohm`, `impedance_ohm`, `soc`, and `soh` may be null because not every public dataset exposes them.
5. Derived fields (`power_w`, `soc_band`, `mode_guess`) should be regenerated after every preprocessing run.

## Validation checks

The schema validator should check:

- column names are standardized,
- required columns exist,
- required fields are not fully empty,
- numeric columns can be converted to numbers,
- SOC and SOH values are within 0–100 when present,
- voltage, capacity, resistance, and impedance are non-negative when present,
- either cycle or timestamp exists,
- derived fields can be created safely.

## How this supports later project modules

| Later module | How the schema helps |
|---|---|
| Feature extraction | Provides consistent voltage/current/temperature/cycle/SOC fields. |
| Behavior analytics | Enables fast charging, deep discharge, high SOC storage, and high temperature detection. |
| Degradation risk scoring | Provides normalized features for rule-based and ML-based risk models. |
| SOH estimation | Keeps capacity, resistance, impedance, and cycle history in a consistent format. |
| Dashboard | Gives predictable column names for plotting and user-facing explanations. |
| Research documentation | Makes the project traceable and professional for reports, reviews, and GitHub. |
