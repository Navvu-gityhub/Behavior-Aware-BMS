# The three open weaknesses: audit and experiment plan

Written 2026-10-09, before any of the experiments below were run. Evidence is
in `reports/metrics/audit/` (`scripts/audit_weaknesses.py`,
`scripts/audit_followup.py`); earlier results are rows 24-32 of
`docs/validation_matrix.md`. Every statement is marked **Confirmed** (shown by
data here) or **Hypothesis** (consistent with data, not yet tested).

## 1. Audit findings

### Cold temperature (NASA 4 C cohorts: 12-14% error)

| Finding | Status | Evidence |
|---|---|---|
| The error is not a truth-label artefact. NASA's stored capacity disagrees with the cells' own integrated charge (cold cohorts: median -8%, 95th percentile 29%), but scoring against a label from the cells' own current (charge to 2.7 V) gives the same error: 11.8% vs 11.9% (4 C / 1 A), 13.7% vs 13.8% (4 C / 2 A). | **Confirmed, label hypothesis rejected** | `followup_report.md` §1, `A_labels.csv` |
| The "22 C" cohort (B0042-B0044) ran at both 4 and 22 C. Discharges at a different ambient from the cell's reference discharges err by 25%; at the same ambient, 4.9%. Comparing a cold discharge with a warm reference measures temperature, not ageing. | **Confirmed** (3 cells, 192 discharges) | `followup_report.md` §2, `B_cold_mechanism.csv` (temperature drift -22 C) |
| The true 4 C cohorts never change ambient, and still err 11-12%. So there is a cold error beyond condition mismatch. | **Confirmed** | `followup_report.md` §2 |
| In 4 C / 1 A, the window would need a further ~0.26 V shift beyond the ohmic correction to read measured capacity: non-ohmic polarisation that grows with age. In 4 C / 2 A the missing shift is about zero, so that cohort's error is not explained by it. Across all NASA cells the missing shift correlates only weakly with error (Spearman 0.21). | **Hypothesis**, supported for one cohort only | `B_cold_mechanism.csv` |
| The anchored overpotential correction failed its preset criteria (row 27) and gave no benefit on the M50T cells, which were at 23-46 C (row 31), not cold. | **Confirmed**, recorded | rows 27, 31 |

### Misleading confidence

| Finding | Status | Evidence |
|---|---|---|
| The consistency uncertainty u measures the scatter of readings, not their accuracy. If it were a calibrated standard error, ~95% of estimates would lie within 2u of truth. Measured: CALCE 12.5%, NASA 21%, Oxford 0%, M50T 5.5%. Errors are typically 5-30 times u. | **Confirmed** (all 4 datasets) | `C_calibration.csv` |
| Consequence: the HIGH/MEDIUM/LOW label, the "+/- u points" in the health card and on the new `/memory` page, all present a precision as if it were an accuracy. | **Confirmed** by construction | `health/evidence.py`, `dashboard/memory_dashboard.html` |
| On M50T the error is a steady bias, not scatter: the estimate reads high by +2.3% (median, cell D, 259 full discharges). A bias produces no scatter, so u cannot see it. | **Confirmed** (cell D) | `followup_report.md` §3 |
| The two-window check failed (row 32) because both windows sit in the same voltage region and shift together. | **Confirmed** as the failure; the reason is a **hypothesis** | row 32, `window_agreement/` |
| Cause of the M50T bias: capacity in this NMC811 / graphite-SiOx cell is lost mostly at the low-voltage end, outside the 3.90-3.60 V window, so the window ratio fades less than the full capacity. | **Hypothesis**; testable with a low-voltage window | - |

### Early remaining life

| Finding | Status | Evidence |
|---|---|---|
| 72% of the variance in log(cycles to 90%) is between datasets, 28% within. A prior learned across datasets mostly encodes "which lab and protocol", so it cannot be sharp for a new one. This explains v1/v2 failing (rows 28-29). | **Confirmed** | `D_rul_predictability.csv` |
| Within a dataset, early fade predicts life in some cases: Spearman between fade in the first 10% of life and total life is 0.99 (M50T, 8 cells), 0.76 (NASA, 12), 0.39 (Oxford, 8), -0.02 (CALCE, 16, which mixes protocols). | **Confirmed**, small samples | `D_rul_predictability.csv` |
| "Cycle" means different things in each dataset (CALCE Arbin cycle, NASA discharge index, Oxford drive cycle, M50T row). A throughput-based unit (equivalent full cycles) might cut the between-dataset share. | **Hypothesis** | - |

### Leakage and fairness

- Splits have been by cell or by dataset throughout; no cell's readings appear on both sides of a split.
- The cold fix (row 27) and the window check (row 32) were designed after seeing the data they were decided on: NASA cold and M50T respectively. Both are recorded as such. Neither NASA cold nor M50T can serve as an untouched deciding set again.
- The M50T labels (10/25/40 C) are cooling-plate set points; the cells ran at about 23/32/46 C (row 31). Measured temperature is the variable to use.

## 2. Proposed experiments (at most three per weakness)

| # | Weakness | Approach | Mechanism | Data | Main risk | Baseline |
|---|---|---|---|---|---|---|
| C1 | Confidence | **Empirical error bands**: replace "+/- u" with the measured error distribution, by condition, learned leave-one-dataset-out | Calibrate to observed error, not scatter | 4 datasets + an untouched deciding set | Between-dataset bias makes bands wide or under-cover | Current u |
| C2 | Confidence | Out-of-envelope flag: LOW when a reading's conditions (cell temperature, current, resistance growth) fall outside the validated range | Refuse to extrapolate silently | Same | Too coarse to be useful | Current label |
| C3 | Confidence | Low-voltage window check for location-dependent fade | Tests the M50T bias hypothesis | M50T plus a fresh set | Overlaps the failed row 32 | Shipped window |
| T1 | Cold | **Temperature-matched reference**: compare a reading only with reference discharges within 5 C of its own cell temperature; refuse otherwise | Removes the confirmed mismatch error (25% vs 4.9%) | Needs a temperature channel | Refuses many readings; no help for steady-cold error | Shipped |
| T2 | Cold | Cold flag: readings below a measured cell temperature threshold are LOW or refused | Honest about the residual cold error | KIT 0/10 C | Threshold chosen on NASA only | Shipped |
| T3 | Cold | Temperature-dependent polarisation correction | Targets the 0.26 V missing shift | Cells cycled at several temperatures | Already failed once (row 27) | Shipped |
| R1 | RUL | Same-type early prediction: learn early-fade to life from sibling cells of the same type and protocol, leave-one-cell-out | Within-dataset signal (0.76-0.99) | M50T, NASA, a fresh fleet | 8-12 cells per dataset; useless for a new type | Population median |
| R2 | RUL | Conservative lower bound: "at least X cycles with 90% probability" from empirical quantiles | Useful even when point estimates are not | Same | Bound too low to matter | Shipped refusal |
| R3 | RUL | Equivalent-full-cycle units | Removes unit differences | Charge throughput | May not reduce variance | Cycle count |

### Which first: C1, empirical error bands

1. **It is the best-supported defect.** It is confirmed on all four datasets, by a large factor (5-30x), and it touches every number a user sees: the health card, the API and the new dashboard all show "+/- u".
2. **One subsystem, nothing else changed.** The estimator is untouched, so health accuracy cannot regress. Only how its uncertainty is reported changes.
3. **It is the honest base for T1/T2.** Once bands are measured per condition, a cold or temperature-mismatched reading shows the wide band it really has. Refusing (T1) can then be judged against an honest alternative rather than against a false "+/- 0.1 point".

T1 comes second, as its own experiment: confirmed, but on 3 cells. R2 comes third: the cheapest useful remaining-life output, given that point prediction failed twice.

## 3. Pre-registration for C1 (fixed before running)

- **What changes.** A new function returns an empirical band (5th-95th percentile of signed error, so a bias shows as an off-centre band) for a reading, given its condition bin. Bins, fixed now from the audit:
  - cell temperature during the discharge below 15 C, or not;
  - cell temperature more than 5 C away from that of the reference discharges, or not.

  Where the temperature is unknown, the reading uses the pooled band.
- **Development.** Leave-one-dataset-out over CALCE, NASA, Oxford, M50T: bands from three datasets, scored on the fourth. Reported; it decides nothing.
- **Deciding set.** KIT LG HG2 (NMC/C-SiO 18650), cells cycled at 0 and 10 C plus a 25 C control. No method has touched it. Bands from all four development datasets; nothing fitted on KIT. Truth: check-up capacity relative to each cell's first check-up, interpolated to the reading's time. If KIT cannot be obtained per cell, the test is reported UNDECIDED. No other dataset is substituted after the fact.
- **Acceptance on KIT.** Nominal 90% band.
  1. Empirical coverage 0.80-0.97 over all readings.
  2. Coverage at least 0.75 in the cold bin.
  3. Median band width under 10 SOH points outside the cold bin, so it stays useful.

  SUCCESS needs all three. Baseline for comparison: coverage of the current +/- 2u.
- **Regression.** Estimator unchanged. Test suite and pinned numbers must pass. Refusals are unchanged; a band is added only to readings that are already reported.
- **If it fails.** Record it. Keep u for its real meaning (scatter) but stop presenting it as accuracy, and run C2 (out-of-envelope flag), which needs no band to be calibrated.

## 4. Status of the C1 deciding set (checked 2026-10-09, before any C1 code)

KIT cannot be obtained per cell.
- **Log release (DOI 10.35097/1947):** a single streamed tar. Its only data member is `cell_logext.7z` at 59.9 GB. The server answers neither HEAD nor Range requests and delivered about 0.48 MB/s, so it is roughly 35 hours with no way to resume.
- **Result release (DOI 10.35097/1969, 333 MB):** check-up capacity, impedance and pulses only, with no discharge curves, so BEACON's estimator cannot run on it.

As written in section 3, the C1 deciding test on KIT is therefore **UNDECIDED**. No other dataset is substituted after the fact.

**Proposed amendment, to be written in before any of its data is opened:** the University of Michigan NMC/graphite pouch-cell set (12 cells, 5 Ah, cycled at -5, 25 and 45 C; described in arXiv:2010.07460). It is hosted on Deep Blue behind a browser check, so it cannot be fetched by script. Its contents (discharge curves, rests, temperature channel, truth) are not yet confirmed. The same acceptance criteria would apply, with "cold" meaning the -5 C cells.
