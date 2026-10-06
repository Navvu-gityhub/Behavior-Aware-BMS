# Measuring lithium-ion cell health from voltage, current and time: a validated software layer over the battery management system, and when its evidence suffices

**Target venue:** to be decided (candidates: *Journal of Energy Storage*; *Reliability Engineering & System Safety*)
**Status:** DRAFT. Not submittable until the gaps below are closed.
**Author:** Naveen Vaidyanathan
**Draft date:** 2026-10-06
**Companion draft:** `docs/manuscript/ress_draft.md` (the validation-methodology paper). This draft reuses its benchmark result as motivation (§5.1) and is otherwise independent.

---

> ## Blocking gaps: read before circulating
>
> 1. **Related work is not written from a survey.** Every `[CITE: …]` marker
>    is a claim that needs a real, verified reference. None has been filled in
>    from memory, and none may be. The reference list holds only the five
>    entries this project has verified against the publisher's record.
> 2. **Two results rest on two cells each** (partial-discharge SOH, §5.3).
>    They show a mechanism, not a population error, and the text says so.
>    More partial-cycling cells would turn them into a claim.
> 3. **No hardware discharge.** The bench rig validates the telemetry path
>    only (§5.8). A measured discharge series on the rig would let §5.2 be
>    reproduced on an independent cell.
> 4. **Author list, affiliation, funding, CRediT and conflict-of-interest are
>    unwritten** (Declarations).
> 5. **Figures exist but are not placed to a journal template**
>    (`reports/figures/results_*.png`, from `scripts/export_results.py`).

---

## Abstract

A battery management system (BMS) measures terminal voltage, current and
temperature. Whether a software layer on top of it can turn those
measurements into a trustworthy state-of-health (SOH) figure is usually
answered with a fitted model. We first show why that answer does not survive
a change of test protocol. On 31 NASA cells in 9 protocol cohorts, twelve
published methods all lose skill under leave-one-cohort-out validation. The
best point estimate (XGBoost, R² 0.459) has a 95% interval of −2.80 to 0.86
that overlaps every other method. A cycle-count-only linear baseline reaches
0.406. A shuffle control confirms the harness does not create skill.

We then build and validate the alternative: estimators that fit nothing
across cells.

- **SOH** is the ratio of charge delivered between two fixed voltages now to
  the same quantity on the cell's own first discharges. Ohmic sag is
  corrected with a resistance the system estimates from the voltage step at
  each load onset. Measured on 22 CALCE LCO cells from voltage, current and
  time only, it reaches a **1.7% median per-cell error** (95% CI 1.4–4.3%)
  on 17 cells. The other five are refused, each with a stated physical cause.
  Using the cycler's own resistance gives 1.4% on 12 cells; no correction
  gives 4.3% on 12.
- **The same step resistance tracks the cycler's measured resistance** over
  life on 19 of 20 cells (median Spearman 0.86), giving a second, validated
  health quantity: power fade.
- **Partial cycling.** For cells that never cross the standard window, a
  window learned from the cell's own early discharges measures SOH at 3.2–3.4%
  on two cells of real top-of-charge partial cycling. It refuses two
  near-empty cells, where even the flattest available window sits on the
  discharge knee.
- **Remaining useful life (RUL)**, extrapolated from each cell's own fade, is
  within ±20 cycles 73% of the time in the last 25 cycles before 90% capacity.
  Indexed by the *prediction* a user sees, an estimate under 25 cycles was
  never exceeded in 95% of cases, so near end of life it behaves as a safe
  lower bound.
- **Two corrections.** Scoring the same estimator against a reference
  capacity drawn from the cell's whole life, which uses the future, had
  inflated this to 88%. Four methods for RUL from sparse complete discharges
  all fail, and the system refuses rather than reports.
- **When is there enough evidence?** We measure it rather than assert it. The
  number of discharges observed does not predict accuracy, because error grows
  with age. The consistency of the readings does. With the ageing trend
  removed, a standard error at or below one SOH point corresponds to 1.3%
  median error against the lab; above it, to 5.1%.

The result is a software layer whose outputs are each validated, banded by
their measured error, or refused with a reason.

**Keywords:** lithium-ion; state of health; battery management system;
incremental capacity; ohmic resistance; partial discharge; remaining useful
life; evidence sufficiency; validation

---

## 1. Introduction

A BMS protects a battery pack and estimates its state of charge. It already
holds the three measurements from which ageing is visible: terminal voltage,
current and temperature. A software layer above it can, in principle, turn
those into answers an operator acts on: how much of its original capacity
the battery retains, how fast that is changing, and when it should be
replaced.

The dominant way to build that layer is to fit a model. Features are
extracted from cycling data and a regressor is trained to predict SOH
[CITE: reviews of data-driven SOH estimation], with feature sets ranging
from usage aggregates to discharge-curve descriptors such as the
capacity-voltage difference between cycles (Severson et al., 2019). Skill is
commonly reported under leave-one-cell-out validation. A held-out cell's
siblings from the same test protocol remain in training, so the test asks
whether the model recognises a protocol it has already seen. Group-wise
splitting by cell has been shown to change reported skill substantially
(Maher & Yerken, 2025), and a large gap between random and cell-wise splits
has been reported on NASA data (Le & Nguyen, 2026; read from the abstract).

A deployed BMS faces a harder question: the next battery comes from conditions
the training set never contained. §5.1 shows that under that question, on the
data available to us, no fitted method retains reliable skill and no method
can be distinguished from another. We take that as the starting point rather
than the conclusion, and ask what a software layer *can* measure reliably
from the BMS's own signals.

### 1.1 Contributions

1. **A field-realisable SOH estimator, validated from voltage, current and
   time alone.** No cycler-derived quantity reaches the estimate. The ohmic
   correction uses a resistance the system estimates from the voltage step at
   each load onset, which raises coverage from 12 to 17 of 22 cells and
   lowers median error from 4.3% to 1.7%.
2. **Resistance as a validated second health quantity.** The step resistance
   tracks the cycler's resistance column on 19 of 20 cells.
3. **Partial discharges, with a physical refusal rule.** A window learned
   from the cell's own early discharges, and refused when its steepness
   places it on the discharge knee. The steepness separates working from
   failing windows by about 9×, so the threshold is not tuned.
4. **RUL accuracy reported the way a user meets it**, indexed by the
   prediction, plus a measured correction of our own earlier figure (88% →
   73%) caused by a reference capacity that used the future.
5. **Evidence sufficiency measured, not asserted.** Confidence follows the
   readings' consistency, which predicts error, rather than a count of
   discharges, which does not.
6. **Negative results kept in the record:**
   - fitted models do not transfer across protocols (§5.1)
   - behaviour features alone score below zero under LOCO (§5.1)
   - four routes to RUL from sparse complete discharges fail (§5.6)

### 1.2 What this paper does not claim

- Anything about packs, about chemistries other than one LCO family, or
  about dynamic drive cycles. All validation is on single cells at constant
  current.
- That temperature-dependent capacity effects are corrected. The CALCE
  cycling files carry no temperature channel.
- That the bench rig validates health estimation. It validates acquisition.
- That usage behaviour predicts degradation. Temperature showed a consistent
  *direction* in 7 of 7 NASA cells within a protocol; its magnitude did not
  transfer.
- The 80% end-of-life convention. The CALCE trajectories end near 81%, so
  RUL is validated at 90% only.

---

## 2. Related work

*To be written from a documented survey. Positions to cover, each needing
verified references:*

- **Data-driven SOH estimation and its validation practice.** Feature-based
  regressors and sequence models [CITE: reviews]; the influence of the split
  on reported skill (Maher & Yerken, 2025; Le & Nguyen, 2026).
- **Discharge-curve features.** Capacity-voltage difference features
  (Severson et al., 2019); incremental capacity and differential voltage
  analysis [CITE: ICA/DVA methods].
- **Partial-charge and voltage-segment SOH estimators** [CITE: partial
  charging/discharging segment methods]. This is the family the present
  window estimator belongs to. The distinctions to establish against it: no
  cross-cell fitting, an online resistance correction from the load step,
  gates that refuse instead of extrapolating, and validation from raw
  telemetry rather than cycler summaries.
- **Online resistance identification in BMS** [CITE: pulse/step resistance
  estimation]. Here it serves both as a correction and as an output.
- **RUL prognostics and uncertainty** [CITE: RUL reviews; prognostic
  uncertainty].

---

## 3. Data

### 3.1 CALCE CS2 and CX2

Prismatic LCO cells cycled at room temperature by the CALCE group
(Liu, Saxena, Pecht et al.): CS2 cells rated 1.1 Ah (Type 1: 0.9 Ah), and CX2
cells rated 1.35 Ah. Cells are grouped by the dataset's own test types. We
use the 22 cells whose raw Arbin records and full-discharge capacity truth
are available. They span:

- **constant-current cycling** at several rates (CS2 Types 1–4, CX2 Types 1–4)
- **multi-rate cycling** (Type 3), whose discharge switches rate within a cycle
- **real partial cycling** (CS2 Types 5 and 6): thousands of ~25%-depth
  discharges in a fixed band, with a full capacity check about every 100
  cycles. Type 6 cycles at the top of charge (~4.07 → 3.78 V); Type 5 near
  empty (~3.69 → 2.70 V).

The Arbin exports contain no temperature channel. Each cell's records span
many files whose test-time restarts per file. A monotonic time base is
constructed by offsetting each restart, not by sorting timestamps, which
would interleave overlapping segments.

### 3.2 NASA PCoE

The NASA battery ageing set (Saha & Goebel, 2007): 18650 cells under nine
documented protocols (ambient ~4–43 °C, varied loads and cut-offs). It is
used only for the benchmark of §5.1: 31 cells after admissibility screening,
2,585 cycle-level rows, 9 cohorts.

### 3.3 Bench rig

An ESP8266 with an INA219 current/voltage monitor and an LM35 temperature
sensor on a single 18650 cell. It streams a checksummed line protocol at 1 Hz.
It has recorded only resting captures (§5.8).

---

## 4. Methods

### 4.1 From telemetry to discharges

The estimator receives voltage, current and time. The cycler's capacity,
resistance and cycle summaries are dropped before scoring.

- **Segmentation.** Current is segmented into charge, discharge and rest
  phases with a dead band (0.02 A for CALCE, where rest is logged as exactly
  zero) and a minimum run length, so brief sign changes do not split a phase.
- **Charge per discharge.** Each discharge's charge is the trapezoidal
  integral of |I| dt.
- **Every discharge is kept.** A partial discharge that crosses the window is
  a full measurement of the window.

### 4.2 Voltage-window SOH

For a window [V_low, V_high], the window charge Q_w(n) is the charge
delivered while terminal voltage falls from V_high to V_low on discharge n,
read from the curve by interpolation. A discharge must span at least 95% of
the window to be measured.

    SOH(n) = Q_w(n) / median(Q_w over the cell's first 5 measurable discharges)

The reported SOH is the median of the latest 5 accepted readings. The primary
window is 3.90–3.60 V, fixed before any validation run; 4.05–3.55 and
3.85–3.55 V are reported as sensitivity analyses, not chosen between.

The estimator assumes the discharge curve scales roughly uniformly with
capacity. Loss of lithium inventory shifts the curve, which a window ratio
partly absorbs. Loss of active material and resistance growth change its
shape and position, which it does not. This is the method's main
approximation, and §5.7 measures how it drifts with age.

### 4.3 Resistance from the load step, and ohmic correction

Terminal voltage under load sits below open-circuit voltage by roughly I·R.
As R grows with age, a fixed terminal-voltage window slides along the curve,
and the slide reads as extra fade. A cycler records R; a BMS must infer it.
At each discharge onset, from the last rest sample to the first loaded one:

    R_step = (V_rest − V_load) / |I_load − I_rest|

This is accepted only if the two samples are within 60 s and the current
change is at least 0.05 A.

R_step is an *apparent* resistance: ohmic drop plus whatever polarisation
builds by the first loaded sample. On CS2_35 it reads ~0.165 Ω against the
cycler's ~0.099 Ω. Each discharge uses the trailing median of the last 11
step estimates, so a correction never depends on later data. The window for
discharge n is shifted down by |I_n|·R_trailing(n). Discharges before the
first resistance estimate are left unmeasured rather than scored
uncompensated.

### 4.4 Gates

Each gate needs no ground truth, so it can run in deployment.

1. **Truncated discharge.** A discharge delivering under 50% of the reference
   discharges' total charge is not scored. A cut-short cycle crosses the
   window with little charge and would read as severe fade. This gate took
   the worst cell in an earlier study from 52% to ~10% error.
2. **Physical plausibility.** A window ratio below 0.50 is withheld. A cell
   with more than 20% of its readings below that is refused entirely. A ratio
   above 1.10 is withheld as not spanning the same part of the curve.
3. **Reference timing.** The reference must be complete within 20 equivalent
   full cycles of charge throughput. This was found by a partial-discharge
   test in which the first readable discharge came at discharge 805, so "as
   new" was set late in life. Throughput rather than a discharge count is
   used because storage-protocol records contain thousands of short pulses.

### 4.5 Learned windows for partial discharges

When the fixed window cannot be read and the rated capacity is known, a
window is learned from the cell's first 10 discharges only:

1. Find the voltage band they all cover, trimmed 15% at each end.
2. Within it, choose the flattest sub-window of up to 0.30 V.
3. Measure steepness in volts per unit state of charge.

The window is refused if even the flattest choice is steeper than 2.0 V per
unit state of charge, because it would lie on the end-of-discharge knee.
There, voltage reflects kinetics (resistance and diffusion) rather than
stored charge.

### 4.6 RUL from the cell's own fade trend

At cycle k, the cell's SOH history up to k is smoothed (11-point rolling
median), a straight line is fitted over the full history, and its crossing of
0.90 gives the end-of-life estimate. The rules:

- **Minimum history:** 30 points.
- **Refused trends:** flat or rising ones.
- **Horizon bound:** an estimate reaching more than twice the history length
  ahead is refused.
- **Past the threshold:** RUL is reported as zero.

Linear, square-root and quadratic forms, and full versus trailing windows,
were compared earlier; the full-history line was most accurate near end of
life.

### 4.7 Evidence sufficiency

Each output has its own requirements:

- **Capacity health:** voltage, current and the window minimum (5 reference
  readings plus one).
- **Resistance health:** a clean load step.
- **RUL:** 30 complete discharges.
- **Heat exposure:** a temperature channel.

Within capacity health, confidence follows **consistency**. A healthy
estimate keeps moving because the cell ages, so stability of the estimate is
the wrong test. The question is whether readings agree once that trend is
removed:

    reference SE = 1.2533 · MAD(first 5 window charges) / √5 / reference
    recent SE    = 1.2533 · MAD(residuals of the last 10 readings
                   around their own straight line) / √5
    u            = √(reference SE² + recent SE²)

1.2533·MAD/√n is the standard error of a median under Gaussian noise. The
tolerance u ≤ 0.01 (one SOH point) was fixed before the study of §5.7 ran,
chosen to sit below the validated 1.7% median error.

### 4.8 Validation protocol

**Ground truth.** Ground truth is the cycler's measured discharge capacity,
which never enters an estimate. SOH truth is capacity over the median of the
cell's first five cycles. For partial-cycling cells, whose capacity checks
are ~100 cycles apart, it is capacity over the checks within the first 10
cycles. The point of using the cell's own *early* capacity is that this is
the reference a BMS can hold. A reference drawn from the cell's whole life
uses the future, and §5.5 shows it inflates RUL accuracy.

**Pre-declared configurations.** The primary window, the arms, the
tolerance and the scoring were fixed before each study ran. Choices made
after seeing a result are labelled *exploratory* where they appear.

**Benchmark (§5.1).** It uses leave-one-cell-out (LOBO, 31 folds) and
leave-one-cohort-out (LOCO, 9 folds), with R² against the training-fold mean
and 95% bootstrap intervals resampling folds (2,000 resamples).
Hyperparameters were fixed, not tuned per fold; ElasticNet's penalty alone
was tuned by inner cross-validation on the training fold.

**Error per cell.** Errors are summarised per cell and then across cells, so
long-lived cells do not dominate.

---

## 5. Results

### 5.1 Fitted models do not survive a change of protocol

On the NASA benchmark (target SOH; isotonic signal ceiling 0.569), every one
of twelve methods loses skill from LOBO to LOCO:

| Method | LOBO R² | LOCO R² | LOCO 95% CI | LOCO MAE |
|---|---:|---:|---|---:|
| XGBoost | 0.732 | **0.459** | [−2.80, 0.86] | 7.62% |
| Age-linear baseline | 0.483 | 0.406 | [−0.43, 0.51] | 10.71% |
| ElasticNet | 0.609 | 0.355 | [−1.65, 0.76] | 10.31% |
| GPR (Matern) | 0.725 | 0.207 | [−3.72, 0.43] | 17.06% |
| Random forest | 0.773 | 0.150 | [−1.09, 0.90] | 10.67% |
| LSTM (10-cycle windows) | 0.594 | −0.185 | [−2.09, 0.66] | 17.36% |

The full table has twelve methods. LOBO rank carries almost no information
about LOCO rank (Spearman ρ = +0.084, p = 0.795). Across fourteen methods the
spread of LOCO point estimates (2.06 R²) is smaller than the width of the
best method's own interval (3.66).

**Ablation and control** (`reports/metrics/ablation/`):
- XGBoost on cycle number alone: LOCO −0.27.
- XGBoost on behaviour features alone: LOCO −0.29.
- Only the combination reaches 0.459, and a straight line on cycle number
  alone reaches 0.406.
- Shuffling each cell's capacity in time puts every LOCO interval at or
  below zero, so the harness does not manufacture skill.

On this data, usage features add about 0.05 R² over counting cycles, within
the intervals. Temperature was the one usage factor with a consistent,
correctly signed within-protocol relationship to fade (7 of 7 cells), and its
magnitude did not transfer between protocols.

### 5.2 Field SOH from voltage, current and time

At the primary window, on 22 CALCE cells (`reports/metrics/calce_field_soh/`):

| Ohmic correction | Cells measured | Median per-cell error | 95% CI | Worst |
|---|---:|---:|---|---:|
| None | 12 of 22 | 4.3% | 2.5–5.4% | 6.4% |
| Cycler's resistance column (laboratory only) | 12 of 20 | 1.4% | 1.2–2.5% | 5.6% |
| **Step resistance (field)** | **17 of 21** | **1.7%** | **1.4–4.3%** | **10.3%** |

The field correction matches the laboratory correction within its interval
and measures five more cells. Sensitivity windows give 1.2% (4.05–3.55 V,
16 cells) and 1.9% (3.85–3.55 V, 17 cells).

The three worst cells (6.4–10.3%) are all Type 3, whose discharge switches
rate within a cycle. Every refusal carries a stated cause:

- **Four storage/partial-protocol cells (Types 5 and 6):** their records
  delivered 80–155 equivalent full cycles before any constant-current
  discharge crossed the window. No as-new reference exists, so the reference
  gate refuses them at this window.
- **One cell (CS2_7):** never rests before a discharge, so no step resistance
  can be estimated.

Figure 1 (`results_soh_tracking.png`) tracks the estimate against every lab
capacity test over 600–1,000 cycles; mean errors are 0.9 and 1.1 points on
the two cells shown. Figure 2 (`results_soh_per_cell.png`) shows every cell.

### 5.3 Partial discharges

On the four cells of real partial cycling (`reports/metrics/calce_partial_soh/`).
Estimates use only partial discharges before each lab check; checks never
enter the estimate.

| Cell | Band used | Learned window | Steepness (V per unit SOC) | Result |
|---|---|---|---:|---|
| CS2_24 | top of charge | 4.00–3.82 V | 0.83 | **3.4%** error |
| CS2_25 | top of charge | 4.03–3.83 V | 0.82 | **3.2%** error |
| CS2_5 | near empty | — | 10.4 | refused |
| CS2_6 | near empty | — | 8.2 | refused |

Before the knee gate existed, the two near-empty cells were reported at 6.8%
and 12.8% error. The steepness of learned windows on the 18 full-discharge
cells lies between 0.44 and 0.88. Any threshold between about 0.9 and 8
therefore gives the same verdict on every cell, and the chosen 2.0 is not
tuned.

As a control, the learned mode on those 18 cells measured 16 at a 1.95%
median, beside 1.7% for the fixed window. It is a fallback, not a
replacement.

### 5.4 Resistance tracks the instrument

Against the Arbin cycler's internal-resistance column, over each cell's life
(`reports/metrics/calce_resistance/`, 20 cells with both values):

- median Spearman correlation **0.86**
- growth in the same direction on **19 of 20** cells
- median growth over life **1.26×** (step) vs **1.29×** (cycler)

This includes the partial-cycling cells (0.90 and 0.98). The exception is
CX2_8, a Type 3 cell. The level differs, because the step value includes
early polarisation, but the trend agrees. That is what a power-fade indicator
requires.

### 5.5 Remaining useful life

At threshold 0.90, referenced to each cell's first cycles
(`reports/metrics/calce_rul_horizon_early_ref/`, 17 cells):

| True cycles to end of life | Within ±20 | Median error |
|---|---:|---:|
| 0–25 | **73%** | 9.6 |
| 25–50 | 44% | 22.8 |
| 50–100 | 15% | 42.7 |
| 200–400 | 0% | 144 (all early) |

A user sees the prediction, not the truth. Indexed by the prediction
(`reports/metrics/error_bands.csv`):

| Predicted cycles left | Lasted at least as predicted | True − predicted, 90% range |
|---|---:|---|
| 0–25 | **95%** | +1 to +82 |
| 25–50 | 88% | −14 to +129 |
| 50–100 | 85% | −40 to +141 |

Near end of life the estimate is a safe lower bound: it warns early, almost
never late. Errors are mostly early because these cells fade fast early and
then flatten, so a full-history line is steeper than the remaining fade.

**A correction to our own earlier figure.** Scored against a reference
capacity taken as the 95th percentile of the cell's whole record, the same
estimator was within ±20 cycles 88% of the time near end of life. That
reference uses the future, which no BMS can hold. Against the cell's own
early capacity the figure is 73%. We report the latter.

### 5.6 RUL when complete discharges are rare (negative result)

The RUL above extrapolates capacity from complete discharges. A driver who
seldom fully discharges produces few: CS2_24 has 18 in 1,617. Four methods
were tested on the same sparse evidence (`reports/metrics/calce_field_rul/`).
Sparsity is real on the partial-cycling cells and emulated, as 1 complete
discharge in 50, on the constant-current cells.

| Method | Outcome |
|---|---|
| Sparse capacity, standard minimum | refuses every point (never reaches 30) |
| Field SOH trajectory | answers 62%; when predicting under 25 cycles, median miss 79 cycles |
| Field SOH scaled by the cell's own complete discharges | over-corrects; errors turn late near end of life |
| Sparse capacity, minimum lowered to 5 (labelled post hoc) | 0% within ±20 cycles; misses late |

The field-trajectory failure has a clean cause. Near 0.90 the field SOH reads
a median 1.1 points below capacity. Because fade near the threshold is slow,
that small bias moves the 0.90 crossing a median 56 cycles early. A SOH error
acceptable as a health figure becomes large as a time to threshold. None of
the four is reported. The system refuses RUL for partial-only records and
says this was tested.

### 5.7 When the evidence suffices

Three questions were fixed before the study ran, scored at every lab capacity
check on 19 cells (`reports/metrics/calce_sufficiency/`).

**Count does not predict accuracy.** Median error was 0.75% at 6–10 usable
readings and 1.97% beyond 60. Error grows with age, as the uniform-scaling
approximation of §4.2 drifts, rather than shrinking with count. A rule of
"high confidence after N discharges" has no support and is not used.

**Consistency does.**

| Readings | Points | Cells | Median error | 90th percentile |
|---|---:|---:|---:|---:|
| u ≤ 0.01 (consistent) | 8,993 | 16 | **1.3%** | 4.4% |
| u > 0.01 (inconsistent) | 2,168 | 13 | **5.1%** | 21.9% |

The pooled Spearman correlation between u and absolute error is 0.48. Within
cells it is weaker (median 0.14, positive in 14 of 19): u mainly separates
unreliable cells and periods rather than ranking readings within a good one.

**Reference size.** In the study's own scoring, larger references reduced
error (1 discharge: 3.6%; 5: 2.0%; 10: 1.3%). On the field-study metric of
§5.2, however, 10 versus 5 changed median error from 1.681% to 1.665%. The
estimator keeps 5.

Confidence on capacity health is therefore assigned as follows:

- **HIGH:** consistent, fixed window, beginning-of-life reference.
- **MEDIUM:** consistent, but a learned window or a start-of-log reference.
- **LOW:** inconsistent. Reported with the measured 22-point band.

### 5.8 The system and its refusals

The estimators run inside a single pipeline shared by three acquisition
paths: CAN with a DBC definition, a serial bench rig, and research datasets.
Each output is labelled measured, estimated or heuristic, and carries its
error band.

**Hardware.** The bench rig has streamed 83 of 83 checksummed frames at a
measured 1.000 s cadence with no gaps. The two sensors have not yet operated
in the same capture, and no discharge has been recorded. The rig validates
acquisition, not health estimation.

**Fault injection** into a real capture:
- a corrupted byte fails its checksum
- a NaN field is rejected
- a reversed timestamp refuses the capture
- a missing sensor is refused by name

Injection also exposed one defect. A 999 V reading passed the field range,
which is wide so that packs are legal. Records outside 0–5 V are now rejected
when the source declares itself a single cell.

---

## 6. Discussion

**Measurement conditioning moved accuracy; model choice did not.** Across
fourteen fitted methods, LOCO differences were smaller than the uncertainty
on any one of them. By contrast, three conditioning decisions moved error
by multiples:

- refusing truncated discharges: worst cell 52% → ~10%
- estimating resistance from the load step: median 4.3% → 1.7%, 12 → 17 cells
- refusing knee windows: two cells from 6.8–12.8% error to refusal

For a BMS software layer the leverage is in what is measured and how it is
conditioned, not in which regressor follows.

**Reference definitions change conclusions.** Two of our own results moved
when the reference was made one a BMS can hold. RUL accuracy fell from 88% to
73%. A partial-discharge reference formed at discharge 805 produced a 58%
error until the timing gate existed. Validation should use the reference the
deployed system will have; a reference computed from the whole record leaks
the future.

**Threshold crossings amplify small biases.** A 1.1-point SOH bias,
unremarkable as a health figure, shifted predicted end of life by ~56 cycles.
Any RUL built on an estimated, rather than measured, SOH inherits this, and
should be evaluated as a time-to-threshold, not via the SOH error.

**Sufficiency is a measurable property.** Treating "enough data" as a count is
intuitive and wrong here, because the estimator's error is dominated by
ageing-related drift, not by sampling. The readings' own consistency is
observable without ground truth and separates 1.3% from 5.1% median error.
This gives a BMS layer a principled way to say "not yet" for a battery it has
never seen. It needs no population model, because every estimate is relative
to the cell's own first discharges.

**Implications for a deployed layer.** Within the scope tested, a BMS that
logs voltage, current and time can report capacity health, resistance health
and near-end-of-life RUL, each with a measured error band. It should refuse:

- RUL for partial-only histories
- capacity health on knee windows
- any figure whose reference formed late

---

## 7. Limitations

- **One chemistry family, single cells, constant current.** Packs add
  cell-to-cell imbalance, thermal gradients and balancing currents. Dynamic
  drive cycles would challenge both the window read and the step-resistance
  estimate. LFP's flat plateau would challenge the window method directly.
- **Two cells per partial-cycling band.**
- **No temperature correction.** Capacity and resistance are temperature
  dependent; CALCE cycling carries no temperature channel. A cold cell would
  read low SOH and high resistance without having aged.
- **90% end of life only.**
- **Apparent, not ohmic, resistance.** R_step depends on the logger's sample
  interval and is comparable within one record, not across loggers.
- **The approximation drifts with age.** Window charge does not scale exactly
  with capacity, and error grows over life (§5.7). The bands reported are
  medians and percentiles over the validated range.
- **No hardware discharge** has been recorded on the bench rig.
- **Related work is incomplete** (blocking gap 1).

---

## 8. Conclusion

A software layer over a BMS can measure lithium-ion cell health from voltage,
current and time without any model trained on other cells. On 22 CALCE cells:

- capacity health at 1.7% median error, with five refusals each explained
- resistance that tracks the laboratory instrument on 19 of 20 cells
- partial-discharge capacity health at 3.2–3.4% on two cells, with a physical
  rule that refuses the knee
- remaining life that, near end of life, is a safe lower bound

Each output carries a measured error band. Each is withheld when its evidence
is missing or inconsistent, and the consistency criterion that decides this
was itself validated against laboratory truth. The same study that produced
these results also recorded what failed: fitted models across protocols, and
RUL from sparse complete discharges. Those outputs are refused rather than
reported.

---

## Data and code availability

All code, tests and result artifacts are at
`github.com/Navvu-gityhub/Behavior-Aware-BMS`. Every figure in this draft is
regenerated by a named script from committed artifacts (Appendix A). The raw
CALCE and NASA data are not redistributed; download locations are given in
`docs/reproducing.md`.

---

## Declarations

*To be completed before submission: author list and affiliations; funding;
CRediT author contributions; conflict-of-interest statement; use of AI tools
in analysis and writing.*

---

## References

> Verified only. Each entry has been checked against the publisher's,
> preprint server's or data repository's own record, as recorded in
> `docs/manuscript/survey/extraction.csv` and `docs/final_report.md`. Every
> `[CITE: …]` in the text is an open gap with no entry here, deliberately.

Le, H. H., & Nguyen, K.-A. (2026). Charging phase health indicators for battery
state-of-health estimation: A systematic comparison of CC, CV, and combined
approaches under cross-battery validation. *arXiv:2607.23482*. — *Read from
the abstract only; confirm in full text before submission.*

Liu, Y., Saxena, S., Pecht, M., et al. CALCE CS2 and CX2 prismatic cell
cycling series. Center for Advanced Life Cycle Engineering, University of
Maryland. — *Dataset citation; confirm the preferred citation form with the
CALCE group before submission.*

Maher, K., & Yerken, N. (2025). Comprehensive machine learning for lithium-ion
battery state-of-health estimation using group-wise cross-validation. In
*14th International Conference on Renewable Energy Research and Applications
(ICRERA 2025)* (pp. 1465–1468). IEEE.
https://doi.org/10.1109/ICRERA66237.2025.11283788

Saha, B., & Goebel, K. (2007). Battery data set. NASA Prognostics Data
Repository, NASA Ames Research Center, Moffett Field, CA.

Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen,
M. H., Aykol, M., Herring, P. K., Fraggedakis, D., Bazant, M. Z., Harris, S. J.,
Chueh, W. C., & Braatz, R. D. (2019). Data-driven prediction of battery cycle
life before capacity degradation. *Nature Energy*, 4(5), 383–391.
https://doi.org/10.1038/s41560-019-0356-8

---

## Appendix A: Where every number comes from

| Section | Figures | Artifact | Script |
|---|---|---|---|
| 5.1 | benchmark table, intervals, rank correlation | `reports/metrics/benchmark_results.csv` | `scripts/run_benchmark_study.py` |
| 5.1 | ablation, shuffle control | `reports/metrics/ablation/` | `scripts/run_ablation_study.py` |
| 5.2 | field SOH arms, refusals | `reports/metrics/calce_field_soh/` | `scripts/run_field_soh_study.py` |
| 5.3 | partial discharges, steepness | `reports/metrics/calce_partial_soh/` | `scripts/run_partial_soh_study.py` |
| 5.4 | resistance tracking | `reports/metrics/calce_resistance/` | `scripts/run_resistance_study.py` |
| 5.5 | RUL by true horizon | `reports/metrics/calce_rul_horizon_early_ref/` | `scripts/run_rul_horizon_study.py --reference early` |
| 5.5 | RUL by prediction | `reports/metrics/error_bands.csv` | `scripts/build_error_bands.py` |
| 5.6 | sparse-RUL arms | `reports/metrics/calce_field_rul/` | `scripts/run_field_rul_study.py` |
| 5.7 | sufficiency | `reports/metrics/calce_sufficiency/` | `scripts/run_sufficiency_study.py` |
| Figs 1–3 | tracking, per-cell, RUL | `reports/figures/results_*.png` | `scripts/export_results.py` |
| All claims | status of each | `docs/validation_matrix.md` | — |

## Appendix B: To do before submission

1. Run the related-work survey under its documented protocol; replace every
   `[CITE: …]` with a verified reference, or remove the claim.
2. Confirm the Le & Nguyen figure against the full text.
3. Confirm the CALCE dataset citation form.
4. Record a discharge series on the bench rig and reproduce §5.2 on it.
5. Obtain more partial-cycling cells to turn §5.3 into a population claim.
6. Format figures and tables to the chosen journal's template.
7. Complete the Declarations.
