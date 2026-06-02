# Behavior-Aware EV Battery Health Monitoring and Usage Optimization System

## Project Overview

This project proposes a software-based intelligence layer for electric vehicle battery health monitoring. The system uses simulated or BMS-derived battery telemetry such as voltage, current, temperature, State of Charge, charging status, speed, and distance to identify user charging and driving behavior patterns.

The system does not replace the existing Battery Management System. Instead, it works as an additional analytics layer above the BMS to estimate degradation risk and provide user-facing recommendations for improving battery lifespan.

## Problem Statement

Electric vehicle batteries degrade over time due to electrochemical aging, thermal stress, charge/discharge cycling, and user-dependent operating conditions. Existing Battery Management Systems mainly focus on safety, monitoring, protection, balancing, and state estimation. However, they do not usually explain how user behavior affects long-term battery degradation.

This project aims to develop a behavior-aware software layer that converts BMS telemetry into degradation risk insights and actionable recommendations.

## V1 Scope

The first version of this project focuses only on software simulation and analytics.

### Included in V1

- Simulated EV battery telemetry dataset
- Battery data ingestion using CSV
- Feature extraction from time-series telemetry
- Detection of risky usage patterns
- Rule-based battery degradation risk score
- Simple dashboard for visualization
- User recommendations based on detected behavior

### Excluded from V1

- Embedded firmware
- Real BMS hardware
- CAN/OBD integration
- Cell balancing circuit implementation
- Hardware validation
- Real-time vehicle deployment
- Cloud deployment
- Digital twin implementation

## Input Data Fields

The minimum telemetry fields required are:

- timestamp
- vehicle_id
- voltage
- current
- temperature
- soc
- charging_status
- charger_type
- speed
- distance

## Feature Extraction

The system will calculate behavior-related features such as:

- average_temperature
- maximum_temperature
- time_above_40C
- time_above_90_soc
- time_below_20_soc
- fast_charging_count
- depth_of_discharge
- high_current_discharge_events

## Risk Scoring

The system will generate a degradation risk score from 0 to 100.

Risk levels:

- 0–30: Low Risk
- 31–60: Medium Risk
- 61–100: High Risk

## Expected Output

The final output of V1 will be a working software prototype that takes battery telemetry data, identifies harmful usage patterns, calculates degradation risk, and displays recommendations to the user.

## Success Criteria

The project will be considered successful if:

1. The repository is structured clearly.
2. The Python environment is reproducible through requirements.txt.
3. Simulated battery telemetry can be loaded.
4. Behavior features can be extracted.
5. A degradation risk score can be generated.
6. Recommendations can be shown to the user.
7. A basic dashboard can display the results.
