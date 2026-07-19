# Behavior-Aware EV Battery Health Monitoring and Usage Optimization System

> **A software intelligence layer for EV Battery Management Systems.**
> Converts BMS telemetry into degradation risk insights, battery health estimates,
> remaining useful life predictions, and actionable user recommendations —
> presented on a self-contained interactive dashboard.

---

## Status

V1 is a working end-to-end pipeline that runs on **simulated telemetry** via
`python main.py`: schema validation → behavior features → rule-based risk
scoring → health index → RUL estimation → guardian reports → a self-contained
HTML dashboard.

Two things to know before using any of the numbers this produces:

1. **Every score is a hand-tuned rule-based heuristic, not a fitted model.**
   Stress score, risk score, health index, and RUL all use hand-chosen
   weights and cut points (documented in `docs/`). None have been validated
   against measured capacity-fade ground truth. Treat outputs as
   directional and auditable, not predictive.
2. **Real-dataset integration is partial.** NASA/CALCE loading
   (`src/bms/io/`) works and is tested. The feature/risk/health/RUL/guardian
   stages accept any dataframe that passes schema validation, including
   real NASA/CALCE data, but have only been exercised end-to-end on
   simulated data and small samples so far — running the full pipeline
   against the complete NASA/CALCE archives and checking the output
   distributions is the next validation step, not something already done.

Known methodological limitation, left undecided on purpose rather than
silently patched: the RUL formula (`src/bms/rul/rul_estimation.py`) reuses
temperature, deep-discharge, and fast-charge signals that are *already*
baked into the health index, so those effects get double-counted in the RUL
estimate. Fixing this requires refitting against capacity-fade ground truth,
not another guess — see the module docstring.

---

## Overview

Electric vehicle batteries degrade due to electrochemical aging, thermal stress,
charge/discharge cycling, and user-dependent behavior. Existing BMS hardware handles
safety and protection but rarely explains *how user behavior drives long-term degradation*.

This project adds a **behavior-aware analytics layer** on top of any BMS to:

1. **Ingest** battery telemetry (voltage, current, temperature, SoC, speed, distance)
2. **Extract** behavior features (fast-charging frequency, deep discharge events, thermal stress)
3. **Score** degradation risk (0–100, rule-based; optional ML scoring hook, unvalidated — see below)
4. **Compute** a Battery Health Index with an aging-budget model
5. **Estimate** Remaining Useful Life (RUL in cycles)
6. **Run** the Battery Guardian AI — explains causes and recommends actions
7. **Display** an interactive HTML dashboard (no server required)

The system does not replace the BMS. It's an analytics layer above it.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (uses simulated data by default)
python main.py

# 3. Open the dashboard
#    → dashboard.html  (open in any browser)
```

### Run with your own data

```bash
python main.py --data path/to/your_unified_schema_file.csv
```

Your CSV needs to satisfy `src/bms/preprocessing/schema.py`'s unified schema
(or its recognized aliases) — see `docs/data_dictionary.md`.

### Run tests

```bash
python -m pytest tests/
```

---

## Project Structure

```
Behavior-Aware-BMS-main/
├── main.py                          # Full pipeline entry point
├── requirements.txt
├── dashboard.html                   # Generated — open in browser
│
├── src/bms/                         # Core Python package
│   ├── simulation/
│   │   └── simulate_telemetry.py    # Synthetic telemetry generator (illustrative, not a validated battery model)
│   ├── io/
│   │   ├── loader_common.py         # Shared column normalization helpers
│   │   ├── load_nasa.py             # NASA dataset loader
│   │   └── load_calce.py            # CALCE dataset loader
│   ├── preprocessing/
│   │   └── schema.py                # Unified BMS schema, aliases, validation
│   ├── features/
│   │   └── behavior_features.py     # Flags, rolling stats, per-battery summary
│   ├── risk/
│   │   └── stress_score.py          # Row-level stress score + battery-level risk assessment (rule-based; optional ML hook)
│   ├── health/
│   │   └── health_index.py          # Battery Health Index via aging-budget model
│   ├── rul/
│   │   └── rul_estimation.py        # RUL via Equivalent Aging Factor
│   ├── guardian/
│   │   └── guardian.py              # Battery Guardian AI reports
│   └── dashboard/
│       └── dashboard.py             # Self-contained HTML dashboard generator
│
├── configs/
│   └── dataset_sources.yaml         # Dataset source configuration
├── data/
│   ├── raw/                         # Place NASA/CALCE data here
│   ├── interim/                     # Archive manifests
│   ├── processed/                   # Per-dataset normalized CSVs
│   └── features/                    # Pipeline output CSVs
├── models/
│   └── stress_score_rf_sample_v1.joblib  # sklearn HistGradientBoostingRegressor of unknown provenance (see note below) — not wired into the pipeline by default
├── notebooks/                       # Original Colab notebooks (reference/history; superseded by src/bms/)
├── reports/
│   └── metrics/                     # Distribution CSV reports, generated by main.py
├── scripts/                         # CLI utility scripts
├── tests/
│   ├── smoke_day3.py                # Day 3 data ingestion smoke test
│   ├── test_schema.py               # Unit tests for the unified schema
│   └── test_pipeline.py             # Full pipeline end-to-end tests
└── docs/                            # Documentation
```

**Note on `models/stress_score_rf_sample_v1.joblib`:** despite the filename,
this loads as a `sklearn.ensemble.HistGradientBoostingRegressor`, not a
random forest, and the repo has no record of what it was trained on, what
features it expects, or how it performs. It is not used by the default
pipeline. `src/bms/risk/stress_score.try_ml_stress_score()` will attempt to
load it and silently decline to use it unless its expected feature columns
can be verified — see that function's docstring before trusting it with
anything.

---

## Pipeline Steps

| Step | Module | Description |
|------|--------|-------------|
| 1 | `simulation/simulate_telemetry.py` | Generate synthetic fleet telemetry (or load CSV via `--data`) |
| 2 | `preprocessing/schema.py` | Normalize column names to unified BMS schema, validate |
| 3 | `features/behavior_features.compute_behavior_flags` | Row-level behavior flags |
| 4 | `risk/stress_score.compute_stress_score` | Row-level rule-based stress score (0–100) |
| 5 | `features/behavior_features.add_rolling_features` / `add_age_features` | Rolling stats, per-battery-normalized age factor |
| 6 | `features/behavior_features.summarize_batteries` | Per-battery summary table |
| 7 | `risk/stress_score.compute_risk_assessment` | Battery-level risk score, level, reason, recommendation |
| 8 | `health/health_index.compute_health_index` | Aging budget → Battery Health Index → state |
| 9 | `rul/rul_estimation.compute_rul` | Equivalent aging factor → RUL cycles → replacement policy |
| 10 | `guardian/guardian.generate_guardian_reports` | Causes, recommendation, human-readable report |
| 11 | Outputs | CSVs in `data/features/` and `reports/metrics/` |
| 12 | `dashboard/dashboard.build_dashboard` | Interactive HTML dashboard (`dashboard.html`) |

---

## Battery States (Health Index)

| State | Health Index | Meaning |
|-------|-------------|---------|
| HEALTHY | 0 – 29 | Normal operation |
| WARNING | 30 – 59 | Early degradation indicators |
| DEGRADED | 60 – 79 | Performance degradation detected |
| CRITICAL | 80 – 100 | High risk — immediate action required |

## Degradation Risk Levels (Battery-level Risk Score)

| Level | Score | Meaning |
|-------|-------|---------|
| LOW | 0 – 39 | Minimal degradation risk |
| MEDIUM | 40 – 59 | Moderate risk — monitor conditions |
| HIGH | 60 – 79 | High risk — change behavior |
| CRITICAL | 80 – 100 | Immediate action required |

This is a 4-band scale matching `docs/risk_rules.md` and the implemented
code in `src/bms/risk/stress_score.py`. An earlier version of this README
described a different 3-band scale (Low 0–30 / Medium 31–60 / High 61–100)
that did not match any implemented code — that was a documentation bug,
not an alternate design; it has been removed.

Note that Health Index and Risk Score are two **separate** scales computed
from overlapping but not identical inputs — a battery's health-index state
and risk-score level will usually track together but are not guaranteed to
agree, since they're independent rule sets. See `src/bms/risk/stress_score.py`
module docstring.

---

## Feature Catalog

### Telemetry-level features
- `stress_score` — row-level degradation stress score (0–100)
- `aggressive_discharge_event` — binary flag (current < -2A)
- `fast_charge_flag` — binary flag (current > 2A)
- `high_temp_flag` — binary flag (temperature > 40°C)
- `deep_discharge_flag` — binary flag (SoC < 20%)
- `high_soc_flag` — binary flag (SoC > 90%)
- `stress_rolling_mean` / `stress_rolling_std` — 50-step rolling stress stats (per battery)
- `temp_rolling_mean` / `temp_rolling_max` — 50-step rolling temp stats (per battery)
- `battery_age_factor` — normalised cycle position, 0→1 over *that battery's own* lifetime
- `cycle_stress_index` — stress × age factor (penalises late-life stress events)

### Per-battery summary features
- `avg_stress`, `avg_temp`, `avg_soc`
- `fast_charge_duration` — total fast-charge rows
- `deep_discharge_duration` — total deep-discharge rows
- `high_temp_duration` — total high-temp rows
- `aggressive_discharge_count` — count of aggressive discharge events

---

## Datasets

Place downloaded datasets under:

```
data/raw/nasa/          ← NASA Battery Dataset (.mat / CSV exports)
data/raw/calce/         ← CALCE Battery Dataset (Excel / CSV)
data/raw/stanford/      ← Stanford Battery Dataset
data/raw/simulated/     ← Auto-generated synthetic data
```

**NASA Battery Dataset** — available at the CALCE group website and Kaggle.
**CALCE Battery Dataset** — available at calce.umd.edu.

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
joblib>=1.3.0
matplotlib>=3.7.0
scipy>=1.11.0
openpyxl>=3.1.0
PyYAML>=6.0.0
```

---

## Scope

### Included (V1)
- Simulated EV battery telemetry
- Data ingestion (NASA, CALCE, simulated CSV)
- Behavior feature extraction from time-series telemetry
- Rule-based degradation risk score
- Battery Health Index with aging budget model
- Remaining Useful Life estimation
- Battery Guardian AI recommendations
- Interactive HTML dashboard

### Excluded (V1)
- Embedded firmware / real BMS hardware
- CAN/OBD integration
- Real-time vehicle deployment
- Cloud deployment
- Cell balancing circuits
- Model calibration/validation against measured capacity-fade ground truth (planned next step — see Status)

---

## Notebooks

The original Colab notebooks used during development are preserved in `notebooks/`
for history. They are **superseded** by `src/bms/` — the package versions fix a
few bugs found while porting (per-battery age normalization, a stale duplicate
health-index formula, inconsistent risk-band thresholds between the README and
`docs/risk_rules.md`); see each module's docstring for specifics.

| Notebook | Purpose | Ported to |
|----------|---------|-----------|
| `BMS_Project_Setup.ipynb` | Initial data exploration | `src/bms/io/`, `src/bms/preprocessing/` |
| `03_calce_preprocessing.ipynb` | CALCE data preprocessing | `src/bms/io/load_calce.py` |
| `04_feature_refinement.ipynb` | Feature engineering | `src/bms/features/behavior_features.py` |
| `n05_T1v1.ipynb` | Battery-level risk classification | `src/bms/risk/stress_score.py` |
| `06_battery_health_index.ipynb` | BHI aging budget model | `src/bms/health/health_index.py` |
| `07_remaining_useful_life.ipynb` | RUL estimation | `src/bms/rul/rul_estimation.py` |
| `08_battery_guardian.ipynb` | Guardian AI reports | `src/bms/guardian/guardian.py` |
| `09_digital_twin.ipynb`, `error_analysis.ipynb` | Exploratory, not yet ported | — |

---

## Author

**Naveen Vaidyanathan**
[GitHub: Navvu-gityhub/Behavior-Aware-BMS](https://github.com/Navvu-gityhub/Behavior-Aware-BMS)
