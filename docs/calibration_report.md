# Calibration Report: Pipeline Scores vs. Real NASA Capacity Fade

**Date:** 2026-07-20
**Data:** NASA Li-ion battery dataset ("cleaned_dataset" distribution), 34 cells,
7,241,631 telemetry rows, real measured discharge capacity per cycle.
**Script:** `scripts/calibrate_against_nasa.py`
**Status:** First calibration pass. Result: the current rule-based scores do
**not** show a validated, statistically significant relationship with real
capacity fade. This is reported as-is, not adjusted to look better.

---

## 1. What was tested

Two questions, kept separate on purpose:

1. Do the pipeline's existing battery-level scores (`health_index`,
   `risk_score`, `avg_stress`, and RUL's `estimated_total_cycles`) correlate
   with real, measured degradation (linear fade rate in Ah/cycle, and
   cycles-to-80%-capacity where observed)?
2. Independent of the current hand-picked weights, do the raw underlying
   signals (`avg_temp`, `deep_discharge_duration`, `fast_charge_duration`,
   `aggressive_discharge_count`) correlate with fade on their own?

Correlations use Spearman's rho (rank-based, appropriate for a small,
non-normally-distributed sample) with a two-sided p-value. Sample size is
reported alongside every result — several of these are underpowered and are
labeled as such rather than treated as conclusive.

## 2. Result 1: a real bug the calibration caught

`fast_charge_flag` was defined as `current_a > 2.0` (a flat Amp threshold).
Across all 7.24M rows in this dataset, observed `current_a` never exceeds
**1.54A**. The flag never fired — not once. It was calibrated against the
project's own synthetic simulator, not any real charge-rate convention, and
doesn't generalize across cells of different capacity anyway (2A means
something different for a 1Ah cell vs a 20Ah cell).

**Fix applied:** thresholds for `aggressive_discharge_event` and
`fast_charge_flag` now use C-rate (current / rated capacity) instead of a
flat Amp value — see `src/bms/features/behavior_features.py`. This is a
dimensionally correct fix, not a fit to this dataset: re-running after the
fix still shows zero variance in `fast_charge_duration` on NASA data,
because NASA's charge protocol is a fixed ~0.75C constant-current profile
across every cell in this distribution — it never varies charge rate at
all. **NASA data cannot validate the fast-charge hypothesis, at any
threshold**, because it contains no variation in that variable. This is a
dataset coverage gap, not a remaining bug — the Stanford/Severson dataset
(deferred earlier) specifically varies fast-charging protocols across
cells and is the right data source for that question.

## 3. Result 2: a data-heterogeneity confound in the "34 batteries"

The NASA distribution is not one homogeneous experiment. Per NASA's own
per-batch READMEs (`data/raw/nasa/.../extra_infos/`), the 34 cells span
**at least 9 separate sub-experiments** that differ in:

- discharge cutoff voltage (2.0V / 2.2V / 2.5V / 2.7V — this alone rescales
  the measured "capacity" for an otherwise-identical cell state)
- discharge current/profile (constant 1A/2A/4A, or a 0.05Hz square wave)
- ambient temperature (4°C / 24°C / 43°C, deliberately varied across groups)
- and for several groups, NASA's own documentation admits: *"there are
  several discharge runs where the capacity was very low. Reasons for this
  have not been fully analyzed."*

Treating all 34 as one comparable pool — which the first calibration pass
did — mixes behavioral effects with experimental-design effects (e.g. a
higher voltage cutoff mechanically produces a smaller measured capacity
regardless of cell health). Several anomalously small `initial_capacity_ah`
values in the raw results (as low as 0.06 Ah, against a ~2 Ah rated cell)
trace directly to these flagged batches, not to real physical degradation.

**Mitigation used here:** a protocol-homogeneous subset (B0005, B0006,
B0007, B0018, B0033, B0034, B0036 — all 24°C ambient, constant-current
discharge, no NASA-flagged data-quality caveat) was checked as a sensitivity
analysis. At n=7 this is too small for a reliable significance test, but it
surfaces a second, independent problem — see Result 4.

## 4. Result 3: no significant correlation, full sample (n=33, confounded)

| feature | target | n | Spearman ρ | p |
|---|---|---|---|---|
| health_index | fade_rate_ah_per_cycle | 33 | -0.269 | 0.130 |
| risk_score | fade_rate_ah_per_cycle | 33 | -0.269 | 0.130 |
| avg_stress | fade_rate_ah_per_cycle | 33 | -0.100 | 0.579 |
| estimated_total_cycles | cycles_to_eol (n=24 that reached 80% EOL) | 24 | -0.253 | 0.233 |
| avg_temp | fade_rate_ah_per_cycle | 33 | -0.214 | 0.231 |
| deep_discharge_duration | fade_rate_ah_per_cycle | 33 | -0.106 | 0.559 |
| aggressive_discharge_count | fade_rate_ah_per_cycle | 33 | 0.016 | 0.928 |

No result clears conventional significance (p < 0.05). `health_index` and
`risk_score` (identical here because they're built from the same summary
table — see `src/bms/risk/stress_score.py` docstring on that overlap) trend
*negative*: higher scored risk mildly associates with *slower* measured
fade in this sample. Given p=0.13 and the protocol confound in Result 2,
this is much more likely to be noise/confound than a real inverted
relationship — but "not statistically distinguishable from zero, and not
even pointing the intended direction" is the honest summary, not "roughly
works."

## 5. Result 4: the scores have almost no resolution, even before asking if they're right

On the clean 7-battery subset (Result 2), **6 of the 7 batteries received
the exact same `health_index` (52) and `risk_score` (52)**, despite their
measured fade rates spanning a 5x range (0.0009 to 0.0051 Ah/cycle). Their
`avg_temp` only varies 25.9-27.4°C — all falling inside the same coarse
"else" bucket in the current rule (`avg_temp > 40: +25`, `> 30: +15`,
`else: +5`). Bucketed hard-cutoff rules can't discriminate between cells
whose behavior differs moderately rather than crossing a threshold. This is
a design problem independent of Result 3's null correlation: even with
correctly-fit weights, a step-function scoring rule with only ~4 buckets
per input can't produce fine-grained differentiation across a fleet that
mostly behaves similarly. A continuous function of the inputs (e.g. a
fitted linear or logistic model) would not have this failure mode.

## 6. What this does and doesn't mean

**Does not mean:** the underlying idea (behavior drives degradation) is
wrong — that's well established in the battery-aging literature generally.
It also doesn't mean NASA data is useless — Result 1 was a genuine, fixed
bug, and Result 2 is a legitimate, documented dataset property, not a
failure of this analysis.

**Does mean:** the specific hand-picked weights and cutoffs currently in
`health_index.py`, `stress_score.py`, and `rul_estimation.py` have no
empirical support from this test, should not be described as validated
anywhere in this project (README already states this), and the bucketed
rule structure itself needs to change to a continuous function before
recalibration would even be able to succeed.

## 7. Recommended next steps, in order

1. **Rebuild the protocol-homogeneous subset properly.** Pull in the
   remaining groups (B0025-32 at 43°C, B0041-56 at 4°C) as *separate*,
   explicitly-labeled ambient-temperature cohorts rather than pooling
   them — this turns the confound in Result 2 into a real controlled
   comparison (does high-ambient-temp cohort fade faster than room-temp
   cohort?) instead of noise to route around.
2. **Replace bucketed rules with a fitted continuous model** (start with
   linear regression of fade rate on avg_temp, deep_discharge_duration,
   fast_charge_duration, aggressive_discharge_count — interpretable, and
   directly diagnoses Result 4). Only move to a black-box model if a
   linear fit is clearly inadequate.
3. **Re-run with Stanford/Severson data** once available, since it's the
   only current or planned data source with actual fast-charge-protocol
   variation.
4. Until 1-3 land, every score in this pipeline should keep the "heuristic,
   not validated" framing already in the README and dashboard warning
   banner — that framing is now backed by a specific negative result, not
   just caution.

## 8. Addendum: cohort-controlled, cycle-level re-analysis (Stage 1+2)

The analysis in Sections 3-6 pooled all 34 batteries and compared one
whole-life average per battery. Two follow-up changes (`scripts/calibrate_cohort_cycle_level.py`):
grouping batteries into NASA's own documented experimental cohorts, and
comparing a trailing 5-cycle window of behavior against that cycle's actual
capacity loss (not a single whole-life point). This surfaced a real result
the earlier pass was hiding.

**`avg_temp` is a robust, transferable predictor.** Positive and
significant in two independent cohorts — RT_CC_mixed_current (ρ=0.21,
p<0.0001, n=585 cycles) and COLD4C_multiload (ρ=0.22, p<0.0001, n=392
cycles) — with the correct sign in **7 of 7** individual batteries across
those cohorts (checked per-battery, not just pooled, since pooling cycles
within a battery isn't statistically independent). Higher recent
temperature exposure predicts more capacity loss next cycle. This is the
first result in this project with both statistical significance and
physical direction that holds up under a pseudoreplication check.

**The current `avg_stress` composite does not transfer across thermal
regimes.** Positive and significant in the two room-temperature cohorts
(3/4 and 3/3 batteries with the expected sign), but significantly
*negative* and consistent (0/4 batteries positive) in the 4°C cohort. Same
flip, same magnitude of consistency, for `aggressive_discharge_count`
specifically. Working hypothesis, not yet verified: at 4°C internal
resistance rises sharply, so the same current-based "aggressive discharge"
threshold may be detecting the cell straining against cold-weather
resistance rather than behavior that actually accelerates aging — i.e. the
flag likely needs to be conditioned on temperature, not just on C-rate.

**Revised takeaway, replacing the blanket "scores don't work":** the
current `health_index`/`risk_score`/`avg_stress` composites are not
validated and one of their two dominant inputs (temperature) has now been
shown to carry a real, transferable signal on its own. The other dominant
input (current-based aggressive-discharge/fast-charge flags) does not
transfer across ambient conditions in its current form and is actively
diluting the temperature signal when combined into one score. This
sharpens next-steps item #2 in Section 7: the continuous replacement model
should treat temperature and current-based flags as separate terms
(possibly with a temperature-conditioned current threshold), not assume
they combine linearly with fixed weights the way the current heuristic
does.

Caveat maintained from Section 8's own methodology: cohorts have only 3-4
batteries each, so between-cohort comparisons (e.g. "does the sign flip
specifically because of temperature, or because of some other unmeasured
difference in the 4°C protocol") are not yet isolated by this analysis —
that requires either more cohorts at intermediate temperatures or a direct
interaction term in the eventual regression model.

### Reproducing this addendum

```bash
python scripts/calibrate_cohort_cycle_level.py --cache-parquet path/to/cache.parquet
```

Outputs `reports/metrics/cohort_cycle_level_results.csv`.

```bash
python scripts/calibrate_against_nasa.py \
    --nasa-dir path/to/nasa/cleaned_dataset \
    --cache-parquet data/interim/nasa_telemetry_cache.parquet
```

Outputs `reports/metrics/nasa_ground_truth_fade.csv`,
`nasa_pipeline_scores.csv`, `calibration_merged.csv`, and
`calibration_results.csv`.

## 9. CALCE: closed out, zero calibration value, one real bug caught

Full detail in `docs/calce_dataset_note.md`. Summary: the previously-missing
`Capacity Characterization_Initialization.zip` was supplied and integrated
(`src/bms/io/load_calce_capacity.py`, a real loader for a messy raw Arbin
export). It turned out to be a single-cycle characterization, not cycling
data, and — critically — its capacity values are bit-for-bit identical to
the "post-storage" values already in the calendar-aging file, meaning it's
the same measurement counted twice, not a before/after pair. A planned
calendar-aging (storage SOC/temp/duration → capacity loss) analysis was
therefore not possible and its misleading output was deleted rather than
reported.

Real value recovered anyway: running this data through the existing
pipeline exposed a genuine correctness bug — `compute_risk_assessment`/
`compute_health_index` were silently treating missing data (this dataset
has no temperature channel) as "temperature is fine," because NumPy
comparisons against NaN evaluate to False and fell through to the
lowest-risk bucket instead of raising. Fixed to fail loudly instead
(regression test added). CALCE is now closed out for this project unless a
genuinely different file (a real pre-storage baseline, or a multi-cycle
CALCE cycling dataset like CS2/CX2) becomes available.

## 10. Stage 3: fitted continuous model — real signal, not yet a usable predictor

`scripts/fit_continuous_health_model.py` fits capacity_loss (per-cycle, same
target as Section 8) against trailing_avg_temp, trailing behavioral
features, and a cohort fixed effect (to avoid re-introducing the Section 3
protocol confound). Two results, and they point in different directions:

**In-sample, the temperature signal replicates cleanly.**
`trailing_avg_temp` is significant at p<0.0001 (coef +0.0043) even after
cohort fixed effects absorb the between-cohort confound. The overall model
is significant (F-test p=2.5e-5) though R²=0.017 — real but small.
Critically, the `ambient x aggressive_discharge` interaction term that
motivated Section 8's "cold ambient inflates internal resistance"
hypothesis is **not significant** here (p=0.346) once cohort is properly
controlled — that hypothesis doesn't survive stricter testing and should be
treated as unresolved, not confirmed, walking back the framing in Section
8/README.

**Out-of-sample, via leave-one-battery-out CV, the model does not
generalize.** Mean R² across the 33 held-out batteries: **-0.071** (median
-0.017). Only 11 of 33 beat the trivial baseline of predicting the
training-set mean for every cycle. This is a genuine negative result, not
a bug — it means: a real, statistically robust population-level pattern
(temperature relates to fade) does not translate into a model that can
usefully score an individual unseen battery's next-cycle capacity loss with
the features and data available here.

**Why, most likely (not yet verified further):** per-cycle capacity_loss is
a noisy target — single-cycle capacity differences carry substantial
measurement noise relative to the signal. A 5-cycle trailing window and 6
features can't average that out, and 33-34 batteries isn't enough for the
model to learn battery-specific idiosyncrasies a fixed cohort effect
doesn't capture.

**Decision: this model is not wired into the pipeline or dashboard.**
Shipping it would repeat the original problem — presenting an unvalidated
number as if it were reliable — just with a regression instead of hand-set
weights behind it. The heuristic `health_index`/`risk_score` remain what
they were: illustrative and explicitly labeled as unvalidated. This is the
honest state of the project's predictive modeling as of this analysis.

**What would actually close this gap**, in priority order:
1. Predict a longer-horizon target (e.g. cumulative capacity loss over the
   next 20-50 cycles, or cycles-to-EOL directly) instead of noisy
   per-cycle differences — averages out measurement noise the current
   target doesn't.
2. A hierarchical/mixed-effects model (per-battery random intercept) rather
   than a hard cohort fixed effect, so the model can partially pool
   battery-level idiosyncrasy instead of ignoring it.
3. More batteries. 33-34 is thin for 9 cohorts; Stanford/Severson (124
   cells, deferred earlier) would help here independent of its
   fast-charge-variation value.

### Reproducing this analysis

```bash
python scripts/fit_continuous_health_model.py --cache-parquet path/to/cache.parquet
```

## 10. Stage 3: fitted continuous model (health_index_v2)

`scripts/fit_continuous_health_model.py` fits capacity_loss ~
trailing_avg_temp + C(cohort) by OLS on all 2,682 cycle-level observations
across all 9 NASA cohorts (not just the two that showed significance in
isolation — pooling with cohort fixed effects uses the full sample while
still controlling for the Section 2 confound). Deliberately excludes
avg_stress and current-based flags, since Section 8 showed they don't
transfer across cohorts; including a sign-flipping predictor would make
the fitted model *worse* than the bucketed heuristic it's replacing, not
better.

**Pooled fit:** `trailing_avg_temp` coefficient = 0.0038 Ah/°C, p<0.0001,
correct sign, consistent with Section 8. R-squared = 0.015.

**Leave-one-battery-out cross-validation** (the number that actually
matters — a random train/test split would leak cycles from the same
battery between train and test, repeating Stage 1's pseudoreplication
mistake): median per-battery Spearman correlation between predicted and
actual capacity loss = 0.052; 24 of 33 held-out batteries positive
(better than chance, not by much); overall out-of-sample MAE = 0.052 Ah.

**Honest interpretation:** this is a real methodological improvement — a
fitted, cross-validated model with quantified coefficients and uncertainty,
replacing a hand-picked bucketed heuristic with zero empirical grounding.
It is NOT an accurate predictive model. R-squared of 0.015 and median
out-of-sample rho of 0.05 mean single-cycle capacity loss is dominated by
noise that one temperature feature cannot explain. Two most likely paths
to a materially better model, neither attempted here: (a) predict
cumulative fade over a longer horizon (e.g. 20+ cycles) rather than
single-cycle increments, which would average out per-cycle measurement
noise; (b) resolve the current-flag sign-flip (Section 8's open item) so
it can be safely added back as a second predictor.

Implemented as `src/bms/health/health_index_v2.py`, kept alongside (not
replacing) the original `health_index.py` — the original is still the one
`main.py` calls by default, since v2's narrow validated scope (temperature
only, NASA-cohort-specific intercepts) doesn't yet cover what the main
pipeline needs to score arbitrary fleets. Swapping the default should wait
until at least (a) or (b) above closes the accuracy gap.

## 11. Temperature-conditioned-current hypothesis: inconclusive, not confirmed

Section 8 hypothesized that the current-based aggressive-discharge flag's
sign flip at 4°C reflects rising internal resistance under cold ambient,
not real behavioral stress. Tested against NASA's impedance (Re/Rct) data
and found **inconclusive**: NASA measures EIS impedance at a standardized
24°C regardless of the battery's actual cycling ambient for the 4°C
cohorts specifically, so in-situ resistance at 4°C isn't directly
available. Partial, indirect support: for the one cohort where impedance
*was* measured in-situ at true operating temperature (43°C), resistance
was lower than the 24°C cohort's (Re: 0.051 vs 0.059 Ω; Rct: 0.062 vs
0.083 Ω) — consistent with the general temperature-resistance relationship
the hypothesis depends on, extrapolated rather than directly confirmed at
4°C. Resolving this properly would require estimating DC internal
resistance directly from the raw 4°C charge/discharge V-I curves
(voltage step / current step at transitions) — not done here; flagged as
the concrete next step before reintroducing current-based features into
health_index_v2.

## 12. Correction: Level 3 cross-validation, properly framed

Section 10 reported LOOCV using MAE and median Spearman rho. A more
decision-relevant metric was added afterward: R² against each held-out
battery's own mean (does the model beat "just predict this battery's
average"?). Result: mean R² = 0.0024 across 33 testable batteries, with
only 19/33 (58%) beating that naive baseline at all — essentially a tie.
This is a more honest headline number than the rho-based framing in
Section 10, which undersold how weak the out-of-sample performance is by
leading with a metric (rank correlation) that looks more favorable than
the magnitude-sensitive one. Both numbers are true and both are now in
docs/final_report.md Section 4.3; the R²-vs-naive-baseline framing should
be treated as the primary one going forward.

The temperature x cold-ambient interaction hypothesis (Section 11) was
also tested directly (not just via the indirect Re/Rct comparison):
`trailing_aggressive_discharge_count x is_cold`, controlling for cohort.
Coefficient positive (consistent with the hypothesis), p=0.098 — not
significant at conventional thresholds, but with n=2,682 autocorrelated
cycle-observations from only 33 independent batteries, this is
underpowered, not a clean disconfirmation. Verdict unchanged from Section
11: suggestive, not confirmed.
