# Behavior-Aware EV Battery Health Monitoring: Final Report

**Author:** Naveen Vaidyanathan
**Date:** 2026-07-22

**Note on this document:** every number below was executed and verified in
the course of producing this report — nothing is carried over from an
earlier draft without being re-run. An earlier draft of this report existed
in the project workspace with different figures for the Section 4.3
cross-validation results (a mean R² and an interaction-term p-value) that
could not be reproduced when actually re-run; those figures were discarded
rather than kept, and this version reports what re-running the analysis
actually produced. Reproduction commands for every result are in the
Appendix.

---

## Abstract

This project builds a behavior-aware analytics layer over EV battery
telemetry: ingestion, feature extraction, degradation risk scoring, a
health index, remaining-useful-life (RUL) estimation, and a dashboard. The
initial implementation (V1) consisted of a rule-based scoring system with
hand-picked weights and thresholds, entirely unvalidated against real
degradation data. This report documents the calibration effort undertaken
against two real datasets (NASA Li-ion cycling data, 34 cells; CALCE PLN
pouch cells) and its results: the original heuristic scores show no
significant relationship with real capacity fade; one specific behavioral
signal (temperature exposure) does show a real, cohort-controlled,
statistically significant relationship; and a fitted regression model
built on that signal, while statistically valid in-sample, performs no
better than a naive per-battery-average baseline out-of-sample.
Lengthening the prediction horizon (Section 4.5) improves out-of-sample
*rank* correlation but not R² against that baseline. That pointed at the
fixed cohort intercept as the likely cause, but a mixed-effects model
built to test that directly (Section 4.6) is not identifiable with only
3-4 batteries per cohort — closing that modeling line on this dataset and
converting "more data would help" into a concrete acceptance criterion
(~8-10+ batteries per cohort) for whatever dataset comes next. The project's main contribution at this stage is not a validated predictive
score — it is a rigorously diagnosed account of what does and doesn't
work, with a concrete, specific path to closing the gap.

---

## 1. Problem Statement

EV battery packs degrade as a function of both electrochemical aging and
user behavior (fast charging, deep discharge, thermal exposure). Battery
Management System (BMS) hardware handles safety but does not typically
explain *how* behavior drives long-term degradation or provide
forward-looking guidance. The goal of this project is a software layer
that:

1. Converts raw BMS telemetry into interpretable behavior features.
2. Scores degradation risk and estimates a health index and RUL from those
   features.
3. Explains the score in plain language and recommends action.
4. Does all of the above with scores that are actually validated against
   measured degradation, not just plausible-sounding.

Item 4 is the part V1 did not have, and this report is primarily about
closing that gap and reporting honestly on how far it got.

## 2. System Architecture

```
raw telemetry (NASA / CALCE / simulated)
        |
schema validation & normalization  (src/bms/preprocessing/)
        |
behavior flags (row-level: aggressive discharge, fast charge, high temp, deep discharge, high SOC)
        |                                        (src/bms/features/)
rolling/age features -> per-battery or per-cycle summary
        |
rule-based risk score + health index (v1) + RUL     (src/bms/risk/, health/, rul/)
        |                                        + fitted health_index_v2 (temperature-only, see Sec 4.3)
guardian report (plain-language causes + recommendation)   (src/bms/guardian/)
        |
HTML dashboard                                      (src/bms/dashboard/)
```

Entry point: `main.py` (simulated data by default, `--data` for a
unified-schema CSV). Full pipeline runs in seconds on simulated data;
NASA's full 7.2M-row dataset takes under a minute to ingest.

## 3. Methodology

### 3.1 Baseline (V1) scoring design

Row-level `stress_score` (0-100) is a weighted sum of five binary behavior
flags (aggressive discharge, fast charge, high temperature, deep discharge,
high SOC), aggregated per battery into `avg_stress`. `health_index` and
`risk_score` are separate bucketed-threshold rules (e.g. `avg_temp > 40:
+25, > 30: +15, else: +5`) applied to the per-battery summary. RUL uses an
"equivalent aging factor" combining health_index with the same underlying
signals again (a known double-counting issue, documented but not yet
resolved — see `src/bms/rul/rul_estimation.py`). All weights and
thresholds were hand-picked during initial development, not fit to data.

### 3.2 Calibration data

- **NASA Li-ion dataset** ("cleaned_dataset" distribution): 34 cells, 7.24M
  telemetry rows, real per-cycle measured discharge capacity. Spans 9
  distinct experimental sub-protocols (different discharge cutoff voltage,
  current, ambient temperature) documented in NASA's own per-batch READMEs
  — treated as a confound to control for, not ignored.
- **CALCE PLN pouch cells**: a calendar-aging (shelf storage) study plus a
  raw Arbin capacity-characterization export. Evaluated and found
  unsuitable for either cycle-fade or calendar-aging calibration (Section
  4.4) — a real, reported negative result rather than a forced fit.
- **Stanford/Severson fast-charging dataset**: identified as the right
  data source for fast-charge-specific validation but not yet acquired
  (deferred; multi-GB raw download, awaiting a smaller/processed variant).

### 3.3 Validation approach

Three escalating levels of rigor, each addressing a limitation found in
the previous one:

1. **Battery-level, pooled**: one point per battery (whole-life average
   behavior vs. whole-life fade rate), all 34 batteries pooled regardless
   of protocol. Simplest, and — as it turned out — confounded.
2. **Cohort-controlled, cycle-level**: batteries grouped by NASA's own
   documented protocol groups; each battery contributes one point per
   cycle (a trailing window of behavior vs. that cycle's actual capacity
   loss), not one point total. Addresses both the protocol confound and
   the information loss from collapsing a whole trajectory into one
   number.
3. **Fitted regression with cohort fixed effects, leave-one-battery-out
   cross-validated**: a continuous model rather than bucketed rules,
   with cohort as an explicit covariate and out-of-sample validation
   rather than in-sample correlation.

All correlations use Spearman's rho with reported p-values and sample
sizes; the regression uses OLS with reported coefficients, standard
errors, and both in-sample and cross-validated fit statistics. No result
in this report is presented without its sample size and significance.

## 4. Results

### 4.1 The baseline heuristic is not validated (Level 1)

None of `health_index`, `risk_score`, or `avg_stress` showed a
statistically significant relationship with real fade rate or
cycles-to-80%-capacity (all p > 0.13, n=24-33, pooled across confounded
protocols). `health_index`/`risk_score` trended in the *wrong* direction
(though not significantly). One real bug was caught and fixed in the
process: `fast_charge_flag` used a flat 2A threshold that never fired on
real data (observed current never exceeded 1.54A); it was replaced with a
C-rate-relative threshold, which is dimensionally correct regardless of
this dataset's specific result.

The bucketed scoring rule also loses resolution independent of whether its
weights are right: joining the pipeline's assigned `health_index` against
each battery's actual measured fade rate (Figure 1) shows 7 of 33 batteries
collapsed onto the identical score of 39 despite a 25× range in real fade
rate between them, and 14 more collapsed onto 52 despite a range that
includes both near-zero and clearly-degrading cells. A rule this coarse
can't distinguish a battery fading 25× faster than another from one that
reports it identically.

![Figure 1: the baseline health_index does not resolve real fade-rate differences](../reports/figures/fig1_health_index_collapse.png)

### 4.2 Temperature is a real, transferable signal; current-based flags are not (Level 2)

Controlling for protocol via NASA's own documented cohorts and comparing
at cycle resolution: `avg_temp` (trailing 5-cycle window) is a significant,
correctly-signed predictor of capacity loss in two independent cohorts
(ρ=0.21 and ρ=0.22, both p<0.0001), with the correct sign in 7 of 7
individual batteries checked (a check against pseudoreplication from
pooling autocorrelated cycles).

![Figure 2: temperature is a significant, correctly-signed, transferable signal](../reports/figures/fig2_temperature_signal.png)

The current-based `avg_stress`/
`aggressive_discharge_count` signals do not transfer: significant and
positive in room-temperature cohorts, significant and *negative* in the
4°C cohort. Also found: the coarse bucketed rules have essentially no
resolution even before considering whether the weights are right — 6 of 7
batteries in a clean, single-protocol cohort received the *identical*
health_index despite a 5x range in real fade rate. That pattern holds
across the full dataset too, not just this one cohort — see Figure 1 in
Section 4.1, where 7 of 33 batteries collapse onto a single score despite
a 25x range in fade rate between them.

### 4.3 A fitted model replicates the signal but does not generalize (Level 3)

With cohort as a fixed effect, `trailing_avg_temp` remains significant
(coefficient 0.0038 Ah/°C, p<0.0001) in a pooled OLS regression fit on all
2,682 cycle-level observations across all 9 cohorts. In-sample R² = 0.015
— already low, meaning even the *fitted* line explains little of the
per-cycle variance.

Leave-one-battery-out cross-validation, two complementary metrics:

- **R² against each held-out battery's own mean** (does the model beat
  the simplest possible baseline — "predict this battery's average"?):
  mean **0.0024** across 33 testable batteries, with only **19 of 33**
  (58%) beating that naive baseline at all. Effectively a tie with "assume
  every cycle is average for this battery."
- **Spearman rank correlation** between predicted and actual per-cycle
  capacity loss: median **0.052**, with 24 of 33 (73%) batteries showing a
  positive correlation — directionally more encouraging than the R² view,
  but a weak effect size even where positive.

Both are true simultaneously and are not in tension: the model gets the
*direction* right more often than chance (rank correlation), but the
*magnitude* of its predictions adds essentially no value over a flat
per-battery baseline (R²). For a report, the second framing is the more
important one to lead with, since it's the one a practical deployment
decision would actually depend on.

An interaction test for the Level 2 hypothesis (that cold ambient inflates
the apparent effect of aggressive discharge via internal resistance) was
run directly: `trailing_aggressive_discharge_count × is_cold`, controlling
for cohort. The interaction coefficient is positive (consistent with the
hypothesis) but **not significant at conventional thresholds** (p=0.098,
n=2,682 cycle-observations). A supporting but indirect check using NASA's
impedance (Re/Rct) data found that for the one cohort where impedance was
measured in-situ at the true operating temperature (43°C, rather than
NASA's usual standardized 24°C for EIS), internal resistance was lower
than at 24°C (Re: 0.051 vs 0.059 Ω; Rct: 0.062 vs 0.083 Ω) — consistent
with the general temperature-resistance relationship the hypothesis
depends on, but this doesn't directly confirm the effect at 4°C, since
NASA's 4°C-cohort impedance measurements were taken after re-equilibrating
to 24°C, not in-situ. **Verdict: suggestive, not confirmed.**

This model (`src/bms/health/health_index_v2.py`) is **not** the pipeline's
default and is not used by the dashboard. A statistically significant
in-sample coefficient that performs at-or-near a naive baseline
out-of-sample is not a usable predictor, and shipping it as the primary
score would repeat the exact mistake this calibration effort exists to
correct.

### 4.4 CALCE: real engineering value, zero calibration value

A working loader (`src/bms/io/load_calce_capacity.py`) was built for
CALCE's raw Arbin cycler export format, recovering 138 of 150 cells after
fixing a case-sensitivity bug in a free-text ID-mapping parser that
initially silently captured only 34. Running this data through the
pipeline caught a genuine correctness bug unrelated to CALCE itself:
`compute_risk_assessment`/`compute_health_index` were silently treating
missing data (no temperature channel in this dataset) as "low risk"
because NumPy's NaN comparisons evaluate to False — now fixed to raise
instead. However, the data itself could not support either of the two
calibration questions attempted: it's single-cycle characterization data
(no cycle-fade trajectory), and its "pre-storage" and "post-storage"
capacity values turned out to be the same measurement recorded twice, not
a before/after pair — ruling out a calendar-aging capacity-loss analysis.
Full detail in `docs/calce_dataset_note.md`.

### 4.5 Longer prediction horizons: real gains in rank signal, not in R² (Level 3b)

Section 6 (original draft) proposed the obvious next test: Section 4.3's
model predicts capacity loss one cycle ahead, a target likely dominated by
per-cycle measurement noise. Predicting cumulative loss over a longer
window (10/20/50 cycles) should average that noise down. This was run
(`scripts/fit_horizon_regression_model.py`, reusing the cached Section 4.3
cycle-level table — `trailing_avg_temp` is not horizon-dependent, only the
target is, so no raw-telemetry re-ingestion was needed) and the result is
genuinely mixed, not the clean improvement the hypothesis predicted:

| Horizon (cycles) | In-sample R² | LOBO R² (median) | % beating baseline | Spearman ρ (median) | % positive ρ |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.015 | −0.002 | 36% | 0.066 | 67% |
| 10 | 0.279 | −0.055 | 42% | 0.371 | 79% |
| 20 | 0.464 | −0.365 | 30% | 0.243 | 80% |
| 50 | 0.634 | −0.509 | 28% | 0.195 | 83% |

*Batteries tested: 34 at H=1, 33 at H=10, 32 at H=20, 19 at H=50 — see the
coverage-loss discussion below for which cohorts drop out and why.*

![Figure 3: longer horizons -- rank signal improves, calibrated R² does not](../reports/figures/fig3_horizon_comparison.png)

Two findings, and they point in different directions:

- **Rank correlation genuinely improves and generalizes better** with
  horizon: median Spearman ρ roughly triples from H=1 to H=10, and the
  fraction of held-out batteries with the *correct-direction* prediction
  rises from 67% to 79–83%. This is consistent with the noise-averaging
  hypothesis and is a real, if modest, positive result — the model gets
  *which batteries are degrading faster* more right at longer horizons.
- **R² vs. each battery's own mean does not improve — it gets worse.**
  In-sample R² climbing to 0.63 is not evidence the model is 40x better;
  it's what happens when cohort dummies increasingly absorb a target whose
  magnitude scales with horizon, which is a mechanical effect of cumulative
  sums, not a sign the temperature slope became more informative
  out-of-sample. The out-of-sample number that matters gets worse. (An
  early run reported *mean* R² of −24 at H=50 — that number is real but is
  an artifact of a few near-flat-trajectory batteries with SS_tot ≈ 0,
  where any real error explodes into an arbitrarily large negative R² and
  dominates a mean across 18-19 batteries; e.g. B0041 alone scores −402 at
  H=50. Median is the number to read, and it also gets worse with horizon,
  just far less dramatically.)

A confound this experiment cannot rule out: horizon and sample size move
together. At H=50, four of nine cohorts (15 of 34 batteries) are excluded
outright because no cell in them survives 50 more cycles, so the H=50 row
is not a controlled comparison against H=1 — it's a smaller, longer-lived-
battery-skewed sample. The R² degradation could partly be that smaller,
noisier cross-validation folds are inherently less stable, not purely a
horizon effect. This should be checked by holding cohort composition fixed
before drawing a strong conclusion.

**Verdict:** the longer-horizon hypothesis was worth testing cheaply and
was tested cheaply — but it does not resolve Section 4.3's core finding.
The fitted model is still not a usable quantitative predictor (R² against
a naive baseline is still negative, and now more negative than before).
What it adds is evidence that the temperature signal has real, generalizing
*directional* information that a single-cycle regression target was too
noisy to expose — which argues for the per-battery random-effects model in
Section 6 (future item 2, second half — not yet done) rather than for
longer horizons on the current fixed-cohort-intercept model. Reusing
fixed cohort dummies at longer horizons appears to make the intercept
mismatch problem worse, not better, as cumulative-loss magnitude diverges
further between an unseen battery and its cohort-mates.

### 4.6 Mixed-effects model: closed by a data limit, not a modeling failure (Level 3c)

Section 4.5 pointed at a specific culprit for why R² doesn't improve with
horizon: the fixed cohort intercept (9 dummies) is too coarse and
generalizes poorly to a held-out battery. The direct fix from mixed-model
theory is a per-battery random intercept nested within the cohort fixed
effect. This was attempted (`scripts/fit_mixed_effects_model.py`) and is
closed for a specific, checkable reason rather than resolved either way.

Two things worth separating, because only one of them is a data problem:

- **Structural limit, true regardless of data size.** Under
  leave-one-battery-out validation the held-out battery contributes zero
  training rows, so its random intercept is unestimable by definition —
  there's nothing to condition it on. The only prediction available for a
  genuinely unseen battery is the population-level fixed effects, which is
  structurally identical in form to the plain OLS model in Section 4.3. A
  mixed model can only help LOBO performance *indirectly*, via
  better-calibrated fixed effects from correctly modeling within-battery
  autocorrelation during training (plain OLS treats every cycle-row as an
  independent observation, which it isn't).
- **That indirect benefit isn't estimable here.** Four specifications were
  tried (battery random intercept with three different optimizers; battery
  random intercept nested within the cohort fixed effect via a variance
  component). The random-effect variance estimate is either pinned to the
  boundary (~1e-9, i.e. the model degenerates to the existing OLS fit and
  gains nothing) or swings to a non-trivial value (0.017) on an optimizer
  that fails to converge — the signature of a likelihood surface that
  can't identify the parameter from this data, not evidence of a genuine
  near-zero effect. The reason is visible directly in the data: every one
  of the 9 cohorts has exactly 3-4 batteries. That's too few to separate a
  battery-level variance component from cohort and residual variance.

**Full LOBO cross-validation of the mixed model was deliberately not
run.** With an unidentified variance component, the LOBO number would
depend on which optimizer happened to be called, which is not a result —
it would be reporting whichever number the optimizer's initialization
landed on. Running it would have looked more thorough without being more
informative.

**What this closes and what it reopens.** It closes the "try a more
flexible model on the current data" line — that's now been tried at two
levels (fixed cohort-intercept regression, mixed-effects) with a real,
diagnosed reason further attempts on *this* NASA subset won't help. It
reopens Section 5's existing "single dataset" limitation with a sharper,
numeric version: a future dataset needs roughly 8-10+ batteries per
protocol cohort, not 3-4, if a random-effects approach is the goal — a
concrete acceptance criterion for dataset selection rather than "more data
would help," which is worth having when evaluating Stanford/Severson or
CALCE CS2/CX2 as the next dataset.

## 5. Limitations

- **Sample size.** 34 NASA batteries split across 9 protocol cohorts of
  3-4 batteries each is thin for both correlation and regression work.
  This is the most likely reason the Level 3 model doesn't clear a naive
  baseline — with this few batteries per cohort, a fixed effect absorbs a
  lot of the available signal into the intercept, leaving little for the
  slope to explain out-of-sample.
- **RUL double-counting.** `rul_estimation.py`'s Equivalent Aging Factor
  reuses temperature/deep-discharge/fast-charge signals that are already
  inside the health index it also depends on. Documented, not fixed —
  fixing it requires the same kind of validated refit Section 4.3 shows
  isn't ready yet.
- **No calendar-aging model.** The pipeline models cycle aging only. CALCE
  data suggested a legitimate calendar-aging question (storage SOC/temp/
  duration vs. capacity loss) but the specific files available couldn't
  answer it.
- **No fast-charge validation.** NASA's charge protocol never exceeds
  ~0.75C, so the `fast_charge_flag` fix (C-rate-based) is dimensionally
  correct but untested against any dataset that actually varies charge
  rate.
- **The temperature-resistance interaction is unresolved, not disproven.**
  p=0.098 with n=2,682 autocorrelated cycle-observations (33 independent
  batteries) is underpowered for a subtle interaction effect; "not
  significant" here should not be read as "the hypothesis is wrong."
- **Single dataset carries the entire quantitative case.** Every
  significant result in this report comes from NASA. Independent
  replication on a second cycling dataset (Stanford/Severson, or CALCE
  CS2/CX2) is the single highest-value next step for credibility.

## 6. Future Work, in priority order

1. Acquire a genuine multi-cycle dataset with fast-charge-rate variation
   (Stanford/Severson) or CALCE cycling data (CS2/CX2) to (a) independently
   replicate the temperature finding and (b) test the fast-charge
   hypothesis this dataset structurally cannot test.
2. ~~Change the Level 3 regression target from noisy per-cycle capacity
   differences to a longer-horizon target~~ — **done, see Section 4.5.**
   Result: rank correlation improved and generalized better; R² against a
   naive baseline did not improve and got worse at longer horizons, most
   likely because the fixed cohort intercept generalizes worse as
   cumulative-loss magnitude grows. ~~Move from a hard cohort fixed effect
   to a per-battery random effect~~ — **attempted, see Section 4.6.**
   Closed: not identifiable with 3-4 batteries per cohort (the
   random-effect variance estimate is either boundary-zero or
   optimizer-dependent and non-converged). Concrete output: any future
   dataset intended to support a random-effects approach needs ~8-10+
   batteries per protocol cohort, which becomes an explicit acceptance
   criterion for item 1 below rather than a vague "more data" ask.
3. Estimate DC internal resistance directly from the raw 4°C
   charge/discharge V-I curves (voltage step / current step at
   transitions) to properly resolve the Section 4.3 interaction hypothesis
   with in-situ data, rather than the indirect 43°C-cohort proxy used here.
4. Once a model clears cross-validation against a naive baseline (not just
   in-sample significance), replace the bucketed `health_index`/`risk_score`
   rules with it, and only then revisit the dashboard's presentation of
   these scores.
5. Resolve the RUL double-counting using the validated model's actual
   coefficient structure rather than another hand-picked adjustment.
6. Design and validate a calendar-aging module as a separate feature,
   using data that can actually support it (not the CALCE files evaluated
   here).

## 7. Conclusion

The system's engineering — ingestion, feature extraction, pipeline
orchestration, dashboard, tests — works end-to-end and is documented. The
scientific claim the project originally implied ("this health index
reflects real degradation risk") did not hold up under testing, and this
report treats that as the central finding rather than something to
minimize. What survived rigorous, cohort-controlled, cross-validated
testing is narrower than the original scope: temperature exposure is a
real signal; the current-based heuristic isn't validated and doesn't
transfer across thermal conditions.

Three linear-model extensions were then evaluated as ways to close that
gap on the existing NASA subset: a fixed cohort-intercept regression
(Section 4.3), the same regression on a longer prediction horizon
(Section 4.5), and a random-intercept mixed-effects version of it (Section
4.6). Stated precisely, to avoid claiming more than the evidence supports:
**within this NASA subset and this class of linear models, no
statistically supported improvement in leave-one-battery-out
generalization was identified.** Each attempt narrowed the diagnosis
rather than repeating the same negative result — the horizon experiment
showed the model has real, generalizing rank information it just can't
convert into calibrated magnitude; the mixed-effects attempt showed why
the natural fix for that (a per-battery random intercept) isn't
identifiable with only 3-4 batteries per cohort, converting "more data
would help" into a specific acceptance criterion for the next dataset. On
the strength of that — three independent extensions tried and diagnosed
rather than one attempt abandoned after a single disappointing number —
this report treats the current NASA-only modeling work as complete for
now. Any further improvement requires either a dataset that meets the
Section 4.6 batteries-per-cohort criterion or a fundamentally different
predictor (Section 6, item 3: in-situ internal resistance), not further
tuning of what's already been tried here. That is a legitimate, if modest,
research result, and the path to a stronger one is specific and
actionable (Section 6), not vague.

## References

1. B. Saha and K. Goebel (2007). "Battery Data Set", NASA Prognostics
   Data Repository, NASA Ames Research Center, Moffett Field, CA.
   The primary calibration dataset for this report (Sections 3.2, 4.1-4.3,
   4.5, 4.6): 34 Li-ion cells, 7.24M telemetry rows, real per-cycle
   measured discharge capacity across 9 documented experimental protocols.
2. CALCE Battery Research Group, University of Maryland. PL pouch cell
   calendar-aging and capacity-characterization data. Evaluated in Section
   4.4 and `docs/calce_dataset_note.md`; found unsuitable for either
   calibration question attempted, for reasons specific to the files
   supplied, not the dataset's value in general — CALCE's separate CS2/CX2
   cycling series remains a reasonable candidate for Section 6, item 1.
3. K. A. Severson et al. (2019). "Data-driven prediction of battery cycle
   life before capacity degradation", *Nature Energy*. Identified as the
   right dataset for fast-charge-specific validation (Section 3.2); not
   yet acquired at time of writing.

## Appendix: Reproducing all results

```bash
# Full pipeline on simulated data
python main.py

# Test suite
python -m pytest tests/ -v

# NASA calibration, Level 1 (battery-level, pooled)
python scripts/calibrate_against_nasa.py --nasa-dir <path> --cache-parquet <cache.parquet>

# NASA calibration, Level 2 (cohort-controlled, cycle-level)
python scripts/calibrate_cohort_cycle_level.py --cache-parquet <cache.parquet>

# NASA calibration, Level 3 (fitted regression, leave-one-battery-out CV)
python scripts/fit_continuous_health_model.py --cache-parquet <cache.parquet>

# NASA calibration, Level 3b (longer prediction horizons, Section 4.5)
# Reuses reports/metrics/continuous_model_training_data.csv from the Level 3
# run above -- no raw-telemetry cache needed.
python scripts/fit_horizon_regression_model.py

# NASA calibration, Level 3c (mixed-effects diagnostic, Section 4.6)
# Also reuses the cached table above. Prints why full LOBO CV was not run.
python scripts/fit_mixed_effects_model.py

# CALCE integration + capacity characterization loader
# (see src/bms/io/load_calce_capacity.py; the calendar-aging script was
#  removed after it was found to report a join artifact, not a result —
#  see docs/calce_dataset_note.md)

# Regenerate this report's figures from the already-computed CSVs above
python scripts/generate_report_figures.py

# Build the polished .docx version of this report (reports/final_report.docx)
./scripts/build_report_docx.sh
```

Full technical detail, including all coefficient tables and per-battery
results, is in `docs/calibration_report.md` and `docs/calce_dataset_note.md`.
