# What a software layer over a BMS can and cannot do

A capability assessment written against this project's own measurements. Every
figure traces to an artifact under `reports/metrics/`; nothing here is quoted
from the literature as though it were our result, and nothing is asserted that
we did not measure.

The question this answers: **given a battery management system that already
measures voltage, current and temperature, what does adding a software layer
buy you — and what is it physically unable to buy?**

---

## 1. The finding that reorganises everything else

Across 14 published methods on the NASA framing, under leave-one-cohort-out:

| | |
|---|---|
| Spread between best and worst method (point estimates) | **2.06 R²** |
| 95% bootstrap interval on the *best* method alone | **3.66 R² wide** |
| Methods whose interval overlaps the best method's | **14 of 14** |

**The uncertainty on one method's score is larger than the entire difference
between fourteen methods.** At this sample size, model choice is not a
resolvable question.

Now contrast that with two *data-conditioning* decisions made in this project,
measured on the same cells against the same ground truth:

| change | effect on SOH error |
|---|---|
| Rejecting truncated discharges that were being read as capacity fade | worst cell **0.523 → 0.105** (5×) |
| Compensating terminal voltage for ohmic sag at 1.23C | **0.068 → 0.035** median on high-rate cells (2×) |
| Choosing a different model | indistinguishable from zero |

That asymmetry is the assessment. **What you measure, and how you condition it,
dominates what you fit by a margin large enough to see; the choice of estimator
is below the noise floor.** A software layer that competes on algorithms is
optimising the term that does not move.

---

## 2. The boundary: measured, derived, inferred

A BMS measures three quantities — terminal voltage, current, cell temperature —
plus time. Everything else in a battery report is computed, and the computations
divide cleanly by how much they can go wrong.

**Measured.** Voltage, current, temperature. Bounded by sensor error. Our rig
reads voltage in exact multiples of the INA219's 4 mV bus LSB, at 1.000 s with
zero dropped samples across 83/83 frames.

**Derived by integration.** Charge (coulomb counting), energy, throughput.
Exact given the measurements, accumulating sensor bias over time. The voltage
window estimator lives here, which is why it works.

**Derived by differentiation.** dQ/dV for incremental capacity analysis.
Amplifies noise; needs smoothing, and the smoothing window changes the answer,
so two ICA studies with different windows are not comparable.

**Inferred by fitting.** State of health, remaining useful life, degradation
mode attribution. This is where the cross-protocol collapse lives, and it is
the only layer where the failure is *unbounded* — a fitted model on an unseen
protocol can be arbitrarily wrong, as our −1.60 R² rows show.

A software layer should push work **down** this list wherever possible. Both
estimators this project ships were built by moving SOH and RUL from the fourth
category into the second.

---

## 3. What we established, with the evidence

### 3.1 The target may carry no signal, and you can check before you fit

Isotonic signal fraction — the fraction of a target's variance that is
monotone in cycle — bounds the attainable R² from above:

| target | max attainable R² | cells |
|---|---|---|
| `capacity_loss` (per-cycle delta) | **0.044** | 33 |
| `soh` / `cumulative_fade` | 0.569 | 31 |
| `horizon_fade_10` | 0.389 | 31 |
| `horizon_fade_20` | 0.639 | 25 |
| `horizon_fade_50` | 0.967 | 17 |

The project spent months reporting "R² ≈ 0" against a target whose ceiling was
0.044. **This is a pre-flight, and it costs one function call.** Any BMS
software layer that fits anything should run it first and report it beside the
score, because a model at R² 0.04 against a ceiling of 0.044 is at 91% of
attainable and looks like failure.

### 3.2 Generalisation fails at the protocol boundary, not the cell boundary

Leave-one-*cell*-out is what most published work reports. Leave-one-*cohort*-out
holds out a whole protocol:

| method | LOBO R² | LOCO R² | LOCO 95% CI |
|---|---|---|---|
| xgboost | 0.732 | 0.459 | [−2.803, 0.856] |
| age_linear | 0.483 | 0.406 | [−0.428, 0.506] |
| random_forest | 0.773 | 0.150 | [−1.093, 0.901] |
| lstm | 0.593 | **−0.185** | [−2.090, 0.655] |

Every method loses skill, without exception, across four dataset framings. The
simplest model has the tightest interval. **A shipped model is a model of its
training protocol**, and LOBO cannot detect that because the held-out cell's
cohort siblings remain in training.

### 3.3 Better features do not fix it

Severson's ΔQ(V) variance and Dubarry-style ICA peak features — the field's
two most-cited feature families — were added to identical rows, cells, cohorts
and gate. One of four methods improved; none beyond fold-to-fold noise; the
collapse was unchanged. So the collapse is **not** attributable to the features
being crude usage aggregates, which was the obvious objection to it.

### 3.4 Reference data transfers better than models do

Fitting a two-parameter gain and offset on **one** cell of an unseen cohort
improved 23 of 24 folds, median +0.242 R². But fitting the model *locally* on
that same one cell beat transfer-plus-calibration on median (0.888 vs 0.813) —
while failing catastrophically, reaching R² −1758 where transfer's worst fold
was −0.04.

**Transfer's value is bounded failure, not peak accuracy.** For an output
feeding a replacement decision, an estimator that is occasionally wildly wrong
is worse than one that is reliably mediocre, because you cannot tell which
reading is the bad one.

---

## 4. What the software layer is actually for

Not prediction. Four things, in descending order of how much of the value they
carry.

### 4.1 Knowing what it cannot compute, and refusing

The project exists because a NaN comparison scored an *unmeasured* temperature
channel as healthy — `NaN > threshold` is `False`, so every heat check passed
for a pack nobody had instrumented. That is the failure mode a BMS software
layer is uniquely positioned to prevent, and uniquely positioned to cause.

Every gate this project ships needs **no ground truth**, which is what lets it
run in the field:

| gate | what it catches | cost |
|---|---|---|
| Channel coverage | a quantity derived from a channel that was never measured | — |
| Rated capacity on the wire | C-rate computed against an assumed capacity (a 3.4 Ah cell scored as 2.0 Ah reads 1.7 C while drawing 1 C) | — |
| Partial discharge | a cut-short cycle read as a faded cell | measurable SOH bounded below ~50% |
| Physical plausibility | a window ratio implying a cell holding under half its charge | refuses 3 of 16 cells |

### 4.2 Making measurements comparable before comparing them

This is where the measured gains were. Constant-current run segmentation,
full-versus-partial discharge detection, ohmic compensation, capacity declared
on the wire rather than assumed. Each is unglamorous; together they moved
worst-case error 5× where model selection moved it by an amount we cannot
distinguish from zero.

### 4.3 Estimators that fit nothing across cells

| estimator | result | scope |
|---|---|---|
| SOH from charge in a fixed voltage window | ~3% median error | cells passing both gates; needs one reference measurement of that cell when new |
| RUL by extrapolating that cell's own fade trend | within ±20 cycles, 88% | inside 25 cycles of end of life; validated at 0.90, **not** at the 0.80 convention |

Both survive partial discharge, which is what a vehicle actually produces — a
car never runs 100% to 0%, which is the reason SOH is estimated rather than
measured in the field at all.

### 4.4 Scoping claims so they survive being checked

LOCO rather than LOBO. Bootstrap intervals beside every point estimate. Signal
ceiling reported with every score. Six ranking claims withdrawn when the
intervals showed them unsupported.

---

## 5. What it cannot do, and why

- **It cannot beat the target's information content.** No algorithm exceeds an
  isotonic ceiling. If the quantity you want carries 4% signal, that is the end
  of the conversation.
- **It cannot generalise across protocols it has not seen.** Measured, four
  framings, no exceptions.
- **It cannot recover a measurement the hardware did not make.** It can only
  refuse clearly.
- **It cannot rank methods at this sample size.** 18–33 cells and 7–10 cohorts
  put the interval wider than the effect.
- **It cannot make a pack-level claim from cell-level data.** All three datasets
  are single cells. Packs add cell-to-cell imbalance, module thermal gradients
  and balancing interventions, none of which appear anywhere in this evidence.
- **It cannot score a protocol whose cycles are not comparable.** CALCE Type 3
  switches discharge rate six times per cycle; no single constant-current run in
  its schedule *is* a full discharge, so capacity-referenced SOH is undefined
  for it. That is a property of the protocol, not a gap in the code.

---

## 6. What follows

Ordered by evidential value per unit of work, not by appeal.

1. **A load step on the bench rig.** Every hardware capture to date is a cell at
   rest for under 90 s. One charge/discharge transition converts "we measured a
   resting voltage" into a measured discharge, coulomb-counted, with a real
   C-rate and a segmented phase. It is an afternoon and it is the cheapest
   evidence available.
2. **Wire the two estimators into the scoring path.** Partly done. `compute_rul`
   still ships the hand-picked weighting times `base_cycle_life = 1000`, because
   the telemetry pipeline hands it a per-battery SUMMARY and fade extrapolation
   needs a per-cycle state-of-health history that a summary does not carry.
   What is closed is the silence: every RUL row now carries `rul_method` and
   `rul_validated`, and the Guardian report says "an UNVALIDATED remaining-life
   estimate of N cycles" unless the provenance says otherwise. A frame with no
   provenance is treated as unvalidated, because the safe default for an unknown
   estimator is not to vouch for it.

   **Now done.** `rul_from_cycle_capacity` reads the state-of-health trajectory
   straight off the `cycles` frame the pipeline already segments - no schema
   change was needed, the trajectory was simply being discarded by
   `summarize_batteries` before `compute_rul` saw it. The pipeline prefers the
   validated estimator and falls back to the labelled heuristic when the log
   cannot support an extrapolation. Verified on both paths: a 3-cycle bench
   capture refuses ("fewer than 30 cycles of history") and keeps the heuristic
   marked UNVALIDATED, while a 500-cycle CALCE trajectory produces a validated
   figure whose error shrinks toward end of life (-54 cycles at 128 out, -15 at
   78), exactly as the horizon study predicts.
3. **A second chemistry — currently blocked, not merely undone.** Every transfer
   statement here is within one LCO family. `data/raw/nasa` and
   `data/raw/stanford` are empty: only CALCE is downloaded, and the NASA source
   is gitignored and not redistributable. So the cross-chemistry test cannot be
   run from this checkout at all, which is a data-availability limit rather than
   an effort one. LFP (Stanford/Severson) is the test worth running, because its
   plateau sits differently and the window method's construction claims to be
   chemistry-agnostic.
4. **Pack-level data, or an explicit scope statement.** No pack data exists
   here and none can be synthesised, so the scope statement is the honest move
   - and it is now **enforced rather than written down**. The wire protocol
   carries a declared `unit` (`cell`, `module` or `pack`), defaulting to `cell`
   because that is what every dataset behind these figures actually is, and
   refusing an unrecognised value rather than defaulting. It travels to the
   scored output as `measurement_unit` and `unit_validated`, so a pack result
   cannot read as a validated cell result. Nothing stops a pack being scored -
   the stages run identically - but the answer now carries the fact that every
   coefficient behind it was established on single cells.
5. **Not: another model.** The measurement above says it would not be
   detectable.

---

## 7. The one-paragraph version

A software layer over a BMS cannot predict its way past the physics. Its value
is in knowing which numbers are computable from the measurements actually
taken, refusing the rest with a reason, conditioning the data so that cycles
are comparable before anything is fitted, and carrying the scope of each claim
alongside the claim. Measured on this project's own data, those decisions moved
accuracy by a factor of five where the choice of learning algorithm moved it by
an amount indistinguishable from zero.
