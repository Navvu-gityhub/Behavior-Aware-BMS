# Project Scope Document

## Project Title

Behavior-Aware EV Battery Health Monitoring and Usage Optimization System

## One-Line Description

A software analytics layer that uses EV battery telemetry to identify user behavior patterns, estimate battery degradation risk, and generate battery-life-improvement recommendations.

## Background

Lithium-ion battery packs in electric vehicles are already equipped with Battery Management Systems. A BMS monitors and controls key functions such as voltage, current, temperature, protection, cell balancing, state estimation, charging/discharging management, communication, and data logging.

However, most BMS implementations are control-oriented and vehicle-internal. They are designed to keep the battery safe and operational, but they do not usually provide user-facing explanations of how charging and driving habits contribute to long-term battery degradation.

Battery degradation is influenced by factors such as temperature, charging rate, depth of discharge, time spent at high SOC, and aggressive discharge patterns. Therefore, analyzing BMS telemetry can help convert raw battery data into useful behavioral insights.

## Problem Statement

Electric vehicle users often do not know which of their charging or driving habits are damaging the battery. Existing BMS dashboards may show values such as SOC, temperature, voltage, or range, but they usually do not explain the connection between user behavior and battery health degradation.

The problem addressed in this project is:

To develop a software prototype that uses BMS-style telemetry to detect user charging and driving patterns, estimate their contribution to battery degradation risk, and provide recommendations for healthier battery usage.

## Proposed Solution

The proposed system will take battery telemetry data as input, process it, extract behavior-based features, calculate a degradation risk score, and provide actionable recommendations.

The pipeline is:

Raw battery telemetry  
→ Data preprocessing  
→ Feature extraction  
→ Behavior classification  
→ Degradation risk scoring  
→ Recommendation generation  
→ Dashboard visualization

## V1 Implementation Scope

V1 will be a software-only prototype.

### V1 Includes

1. Project repository setup
2. Python-based data processing
3. Simulated battery telemetry dataset
4. Feature extraction engine
5. Rule-based risk scoring model
6. Recommendation engine
7. Basic dashboard

### V1 Excludes

1. Embedded firmware
2. Real battery pack testing
3. BMS hardware design
4. Cell balancing circuits
5. CAN/OBD communication
6. Hardware validation
7. Cloud BMS deployment
8. Digital twin implementation

## Minimum Data Required

The prototype will use the following fields:

| Field | Meaning |
|---|---|
| timestamp | Time of reading |
| vehicle_id | Vehicle identifier |
| voltage | Battery voltage |
| current | Battery current |
| temperature | Battery temperature |
| soc | State of Charge |
| charging_status | Charging or discharging state |
| charger_type | Slow, normal, or fast charging |
| speed | Vehicle speed |
| distance | Distance travelled |

## Behavior Features

The system will extract:

| Feature | Purpose |
|---|---|
| average_temperature | Measures general thermal stress |
| maximum_temperature | Detects peak thermal exposure |
| time_above_40C | Detects high-temperature operation |
| time_above_90_soc | Detects high-SOC storage risk |
| time_below_20_soc | Detects deep-discharge behavior |
| fast_charging_count | Detects frequent fast charging |
| high_current_discharge_events | Detects aggressive driving |
| depth_of_discharge | Measures cycling stress |

## Risk Model

The first version will use a rule-based risk score:

Battery Degradation Risk Score =
temperature stress score
+ fast charging score
+ deep discharge score
+ high SOC storage score
+ aggressive discharge score

Risk categories:

| Score | Risk Level |
|---|---|
| 0–30 | Low |
| 31–60 | Medium |
| 61–100 | High |

## Recommendations

Example recommendations:

| Detected Pattern | Recommendation |
|---|---|
| SOC frequently above 90% | Avoid keeping the battery fully charged for long periods |
| SOC frequently below 20% | Avoid repeated deep discharge |
| Frequent fast charging | Prefer slow or moderate charging when possible |
| High battery temperature | Avoid charging immediately after heavy driving |
| High discharge spikes | Reduce aggressive acceleration |

## Success Criteria

The V1 prototype is successful if:

1. The project repository runs locally or in Google Colab.
2. The Python environment requirements are documented.
3. Simulated battery telemetry can be loaded.
4. Behavior-based features can be extracted.
5. Risk scores are generated from the extracted features.
6. Recommendations are generated based on detected patterns.
7. A basic dashboard or notebook output displays telemetry, risk score, and recommendations.

## Risk and Mitigation

### Risk: Scope Creep

The project may become too large if embedded firmware, real sensors, hardware circuits, digital twin, or cloud deployment are included too early.

### Mitigation

V1 will explicitly exclude embedded firmware and hardware validation. The first version will focus only on a software-based battery analytics prototype using simulated or CSV-based telemetry.
