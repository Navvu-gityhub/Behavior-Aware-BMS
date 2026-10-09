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
| 17b | **Evidence sufficiency**: readings' consistency predicts error | CALCE, 19 cells, 11,161 scored points | Lab capacity checks | Error by consistency (standard error ≤ 1 point, fixed before running) | Consistent: median **1.3%**. Inconsistent: **5.1%** (p90 22%). Discharge *count* does not predict error: it grows with age | **Validated**: confidence follows consistency | `calce_sufficiency/` |
| 17c | Reference size (5 vs 10 discharges) | CALCE field study | Cycler capacity | Median per-cell MAE | 1.681% vs 1.665%: no meaningful difference, kept at 5 | **Tested, unchanged** | `calce_sufficiency/`, `calce_field_soh/` |
| 18 | Health from a capture of a real rig discharge | — | Direct capacity measurement | SOH error | No discharge has been recorded yet | **Not tested** | — |
| 19 | Cross-dataset (train NASA, test CALCE) | — | — | — | Not runnable: the NASA features need temperature, which CALCE cycling files lack | **Not tested** | — |
| 21 | **External validation**: capacity health on a second chemistry, manufacturer, format and temperature, unchanged | Oxford Battery Degradation Dataset 1: 8 Kokam NMC/LCO pouch cells, 40 °C, 519 checks | 1C capacity at each check | Per-cell MAE, median | Fixed window **3.0%** (8 of 8 cells); learned window **2.3%**; C/18 window 3.2% | **Validated** | `oxford/` |
| 22 | Reading the window on a **real drive-cycle** discharge | Oxford Cell 1, Artemis urban cycle, 1 Hz, −5.0 to +1.6 A | Same cell's 1C discharge, same window | Ratio | 0.2625 vs 0.2661 Ah: **1.4%** apart; both ~13% below C/18 (rate effect) | **Shown, 1 discharge** | `oxford/drive_cycle_report.md` |
| 23 | Remaining life at the **80% convention** | Oxford, 5 cells crossing 80% | Observed 80% crossing | Lasted at least as predicted | Predicted under 500 cycles: **97%**; median miss 385 cycles | **Conservative bound replicated; precision low** | `oxford/` |
| 24 | **Third dataset, unchanged**: NASA PCoE 18650 cells, ambient 4–44 °C | 27 of 29 screened NASA cells | NASA's computed capacity | Per-cell MAE, median | **9.2%** overall; 43 °C: **1.7%**; 4 °C and 22 °C cohorts 12–14% | **Fails outside warm conditions** | `cross_dataset/` |
| 25 | Does the system **know** when it is wrong on unseen data? CALCE-fixed consistency gate, unchanged | NASA, 1,623 checks | Same | Median per-cell error, HIGH/MEDIUM vs LOW | **4.5%** when it says HIGH/MEDIUM vs **13.2%** when it says LOW; within 3 points 49% vs 12% | **Validated: low confidence marks the failures** | `cross_dataset/D_consistency_gate.csv` |
| 26 | Does a **fitted** model hold on unseen datasets? | Gradient boosting on 3 window features; every source→target pair | Same | Median per-cell MAE | 0.7% inside Oxford → 6.7% on NASA; 1.2% inside CALCE → 5.5% on Oxford. In-dataset validation overstates by 4–10× | **Fails across datasets; kept as a negative result** | `cross_dataset/B_transfer.csv` |
| 27 | **Cold-temperature fix**: anchor the total overpotential to the cell's own early discharge (no cross-cell fitting) | CALCE, NASA, Oxford; criteria preset | Same | Median per-cell MAE vs the shipped correction | Oxford **3.0% → 0.9%**; NASA cold 13.0% → **9.7%** (target was 8.7%); CALCE 1.7% → 2.7% | **Preset criteria not met; not the default.** Oxford gain is real | `temperature_fix/` |
| 28 | **Early remaining life from population knowledge** (v1: line through SOH = 1, log-normal fade-rate prior) | Leave-one-dataset-out over CALCE, NASA, Oxford, 21700 (M50T) | Observed 90% crossing | Median relative error at 10/25% of life; 80% coverage | Best of three arms in 2 of 4 datasets (needed 3); coverage 0.22 (needed 0.65–0.95). Where it helped: Oxford at 25% of life, 8% vs 41% from the cell alone | **Not met** | `population_rul/` |
| 29 | Same, **v2** (power-law shape, learned end of life and shape, noise from model misfit and autocorrelation), deciding test on unseen drive-cycle ageing | Imperial Expt 4, 6 cells crossing 90% | Same | Same | 10% of life: 0.66 vs 0.59 cell-only; 25%: 0.64, best of three; coverage 0.38 | **Not met.** Early RUL learned from other datasets does not transfer; BEACON keeps refusing early RUL | `population_rul_v2/` |
| 30 | **Per-battery memory**: health from a stored profile, updated per telemetry piece, equals the whole-log answer | Synthetic logs fed whole, in pieces, across save/reload; duplicate pieces | `current_field_soh` on the whole log | Exact equality of SOH, uncertainty, resistances, refusals | Equal to 1e-12 in every case; repeated telemetry not double-counted | **Verified (engineering)** | `tests/test_battery_profile.py`, `tests/test_profile_api.py` |
| 31 | **Fourth dataset, held out**: LG M50T 21700 (NMC811/Si-graphite, 5 Ah), 1C, cooling plate at 10/25/40 °C - **measured cell temperature during discharge ~23 / 32 / 46 °C** (self-heating; range 14-31, 26-37, 40-50 °C); shipped vs anchored correction, criteria preset | Imperial Expt 5, 8 cells, sets 1,3,…,15; rerun after the resistance-step repair (run 1 invalid, kept in `run1_invalid/`) | Cycler charge of each full discharge (≥ 4.0 Ah) | Median per-cell MAE | Shipped, unchanged: **2.8%** at the 25 °C plate (~32 °C cell), **2.3%** at the 40 °C plate (~46 °C cell), 0.4% at the 10 °C plate (~23 °C cell; early life only: the 4.0 Ah truth rule leaves 32–90 discharges, SOH ≥ 0.88); all 8 cells measured; step resistance 28–35 mΩ. Anchored: 0.8% at the 10 °C plate (worse), no change at the others. **Not a cold test**: no cell discharged below ~14 °C, so this says nothing about NASA's 4 °C failure. Consistency gate never said LOW, so it gave no warning of the 2–3% errors | **Shipped generalises (room to warm temperatures); anchored gives no benefit at room/warm temperatures; cold remains tested only on NASA; gate silent here** | `m50t_heldout/` |
| 32 | **Confidence blind spot**: flag an estimate when two voltage windows (3.90-3.60 and 4.05-3.55 V) disagree by > 2 points; window, threshold and criteria preset | Deciding: LG M50T (8 cells); development: CALCE, NASA, Oxford | Measured capacity | Among HIGH/MEDIUM estimates: flagged vs unflagged error; share of > 3-point errors flagged | M50T: 2.8% vs 2.1%, catches 7% of large errors. Development: NASA 5.3% vs 1.6% (catches 76%), CALCE 4.8% vs 1.3% (39%), Oxford 4.7% vs 3.6% (75%) | **Not met on its deciding test; not shipped.** On M50T the windows shift together, so disagreement cannot see that bias | `window_agreement/` |
| 20 | Packs, LFP, large-scale dynamic drive-cycle validation | — | — | — | No data | **Not tested** | — |

## How to read it

- **Validated** means scored against an independent ground truth the
  estimator never saw, with the result above.
- **Validated, 2 cells** means the mechanism is shown, not a population error.
- **Not valid**, **not supported** and **not generalising** mean tested and
  failed. These outputs are refused or labelled heuristic everywhere they
  appear.
- **Not tested** means no evidence either way. BEACON makes no claim there.

## Scope in one line

Single lithium-ion cells: LCO (CALCE) for development, NMC/LCO pouch at
40 °C (Oxford) for external validation, mostly constant current with one real
drive cycle. Within that scope: capacity health, resistance health and near-end-of-life remaining life
are validated against lab measurements. Outside it, nothing is claimed.
