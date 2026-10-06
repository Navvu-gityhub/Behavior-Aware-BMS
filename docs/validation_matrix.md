# Validation matrix

Every claim BEACON makes, with the data it was tested on, the ground truth it
was scored against, the metric, the result, and its status. Claims that
failed or were never tested are listed too. That is what makes the others
believable.

All figures trace to files under `reports/metrics/`.

| # | Claim | Data | Ground truth | Metric | Result | Status | Artifact |
|---|---|---|---|---|---|---|---|
| 1 | **Capacity health** from voltage, current and time, full discharges | CALCE CS2/CX2, 22 cells, 7 protocols | Cycler-measured capacity | Per-cell MAE, median, 95% CI over cells | **1.7%** (CI 1.4–4.3%), 17 of 22 cells; worst 10.3%; 5 refused with a stated cause | **Validated** | `calce_field_soh/` |
| 2 | Capacity health from **partial discharges** (learned window) | CALCE Type 6, 2 cells of real top-of-charge partial cycling | Lab capacity checks | MAE at each check | **3.4%, 3.2%** | **Validated, 2 cells** | `calce_partial_soh/` |
| 3 | Near-empty partials are **refused** | CALCE Type 5, 2 cells | Lab capacity checks | Error had they been reported | Would have been 6.8% and 12.8%; refused (window on the discharge knee) | **Validated refusal** | `calce_partial_soh/` |
| 4 | **Resistance health** from the load step | CALCE, 20 cells | Cycler's Internal_Resistance column | Spearman over life; growth ratio | Median ρ **0.86**; same direction **19 of 20**; growth 1.26× vs 1.29× | **Validated** | `calce_resistance/` |
| 5 | **Remaining life**, near end of life | CALCE, 17 cells, threshold 90% | Observed 90% crossing | Within ±20 cycles | **73%** when truly under 25 cycles away. By prediction: under 25 predicted → lasted at least that long **95%** | **Validated near end of life** | `calce_rul_horizon_early_ref/`, `error_bands.csv` |
| 6 | Remaining life, far from end of life | same | same | Within ±20 cycles | 15% at 50–100 cycles; **0%** at 200–400 | **Not valid**: shown with a wide band | same |
| 7 | Remaining life for **partial-only** logs | CALCE, real and emulated sparse complete discharges | Observed 90% crossing | Within ±20, bias | 4 methods tested; none accurate enough | **Not available**: refused with the reason | `calce_field_rul/` |
| 8 | Error bands printed on the card | the studies above | the same ground truth | Pinned by test | Constants equal `error_bands.csv` | **Validated (test)** | `tests/test_error_bands.py` |
| 9 | Behaviour **risk score** tracks degradation | NASA, 33 cells | Measured fade | Spearman | ρ = −0.27, p = 0.12 | **Not validated**: labelled heuristic | `docs/final_report.md` §4 |
| 10 | Usage features predict SOH on unseen conditions | NASA, 31 cells, 9 cohorts | SOH | LOCO R² (XGBoost, behaviour only) | **−0.29** | **Not supported** | `ablation/` |
| 11 | Temperature is linked to faster fade | NASA, within cohort | Fade per cycle | Coefficient, cohort-controlled | Significant, 7 of 7 cells; size did not transfer to new protocols | **Direction only** | `docs/final_report.md` §4.3 |
| 12 | ML models generalise to new protocols | NASA, 12 published methods | SOH | LOCO R², bootstrap CI over folds | Best 0.459, CI [−2.80, 0.86]; all intervals overlap | **Not generalising**: none ships | `benchmark_results.csv` |
| 13 | The benchmark harness does not manufacture skill | NASA, capacity shuffled in time | No signal by construction | LOCO R² CI | All intervals at or below zero | **Validated control** | `ablation/` |
| 14 | **Telemetry pipeline** on physical hardware | ESP8266 + INA219 + LM35, one 18650 cell | Frame integrity, timing | Frames decoded, cadence | 83/83 frames, 1.000 s, no gaps | **Validated** | `data/interim/rig_*.txt` |
| 15 | Sensor **accuracy** on the rig | — | A reference meter | Measurement error | Not measured | **Not tested** | — |
| 16 | **Fault handling**: corruption, NaN, impossible value, time reversal, missing channel | Injected into a real capture | Expected rejection or refusal | Pass/fail | All rejected or refused; 999 V on a declared cell found and fixed | **Validated** | `tests/test_serial_telemetry.py`, `scripts/panel_demo.py` |
| 17 | **Data sufficiency** decisions per output | Synthetic cells with known missing inputs; CALCE | Known missingness | Correct availability per output | Tested | **Validated (tests)** | `tests/test_evidence.py` |
| 18 | Health from a capture of a real rig discharge | — | Direct capacity measurement | SOH error | No discharge has been recorded yet | **Not tested** | — |
| 19 | Cross-dataset (train NASA, test CALCE) | — | — | — | Not runnable: the NASA features need temperature, which CALCE cycling files lack | **Not tested** | — |
| 20 | Packs, other chemistries (LFP, NMC), dynamic drive cycles | — | — | — | No data | **Not tested** | — |

## How to read it

- **Validated** means scored against an independent ground truth the
  estimator never saw, with the result above.
- **Validated, 2 cells** means the mechanism is shown, not a population error.
- **Not valid**, **not supported** and **not generalising** mean tested and
  failed. These outputs are refused or labelled heuristic everywhere they
  appear.
- **Not tested** means no evidence either way. BEACON makes no claim there.

## Scope in one line

Single lithium-ion cells of one LCO family, at constant current. Within that
scope: capacity health, resistance health and near-end-of-life remaining life
are validated against lab measurements. Outside it, nothing is claimed.
