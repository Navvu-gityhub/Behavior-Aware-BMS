# ADR 0005: The gate is the product, not the model

**Status:** Accepted
**Date:** 2026-07-29

## Context

The request was for "a dynamic system that can accept available datasets and
keep improving the model's efficiency."

Taken literally, that framing has a problem. It presumes a working model to
improve. ADR 0002 records that there isn't one: the fitted health model scores
Spearman rho = 0.841 ranking an unseen cell inside a known protocol and
rho = -0.295 on an unseen protocol. Its apparent skill lives in fitted
per-cohort intercepts.

An unguarded retraining loop on top of that would get steadily better at
memorising protocols while its reported metrics improved. It would automate
precisely the failure this project spent its calibration effort diagnosing,
and it would do so behind a dashboard that looked increasingly green.

## Decision

**Build the system with the validation gate as the primary artifact. The
default answer is REJECT.**

Five components, each owning exactly one decision:

| Module | Owns | Refuses when |
|---|---|---|
| `datasets.py` | Can this data answer the question? | No fade target, or a fade rate that cannot be estimated |
| `cohort.py` | Have we seen these conditions? | Operating point outside every observed envelope |
| `validation.py` | Does this candidate generalise? | Fails LOBO, LOCO, or the confound baseline |
| `store.py` | Is this fit stable enough to deploy? | Coefficient flips sign or moves more than 50% |
| `calibrator.py` | Orchestration only | Delegates every judgement above |

The orchestrator adds no criteria of its own. An orchestrator quietly applying
extra thresholds would be a fifth rulebook nobody reviewed — the pattern that
produced a Guardian explaining scores with cut points that existed nowhere
else (ADR 0003).

## Three design choices worth defending

### LOCO is mandatory, not optional

`Validator.gate` rejects a candidate supplied without leave-one-cohort-out
evidence. Not "warns" — rejects.

Leave-one-battery-out always leaves the held-out cell's cohort siblings in the
training set, so it structurally cannot detect a model that has memorised
protocols. A candidate validated only under LOBO has not been shown to
generalise; it has been shown to interpolate. Treating missing LOCO evidence
as a pass would make the gate ornamental.

### R-squared is measured against the training mean, then against age

Textbook R-squared compares against the mean of the *test* fold, which no
deployed model could know. That flatters a candidate by crediting it with
information it would not have. `r2_vs_global_mean` compares against the
training-fold mean, which is what a naive deployed predictor would emit.

That turned out not to be enough. Every behavioural feature in this dataset
drifts with cycle count, because a degrading cell genuinely does run hotter
and discharge deeper:

| Feature | Median within-cell abs(rho) vs cycle | Cells above 0.5 |
|---|---|---|
| `trailing_avg_temp` | 0.407 | 12/32 |
| `avg_temp` | 0.335 | 9/32 |
| `max_temp` | 0.535 | 18/32 |
| `avg_stress` | 0.588 | 20/32 |
| `avg_soc` | 0.739 | 27/32 |
| `deep_discharge_duration` | 0.872 | 30/32 |

Beating a constant is therefore a weak claim: a model can clear it by learning
"later cycle, more loss" and nothing else. Every candidate is now also scored
against a baseline that predicts the target from cycle count alone.

This is not clean target leakage, and calling it that would be imprecise. It
is **age confounding**: the features and the target both increase with cycle
number for real physical reasons. The fix is not to exclude the features but
to raise the bar they must clear.

### Rejections are stored, and models are stored as JSON

The decision log is append-only and records rejections alongside promotions. A
store keeping only successes would present a history of unbroken progress; the
rejections are the record of what the gate caught, and they are the more
informative half.

Model parameters are stored as inspectable JSON rather than pickles. A
governance artifact a reviewer cannot read is not a governance artifact. The
cost is that a candidate must express itself as a flat coefficient mapping,
and one that cannot is — for this project's purposes — too opaque to govern.

## What happened when it was first run

The confound baseline was not in the original design. It was added because the
system, as first built, **promoted a model it should not have.**

`temp_stress_soc` (trailing temperature, average stress, average SOC) cleared
the original gate on the real NASA frame with LOCO R2 = +0.0125 against the
training mean. Investigating rather than accepting it showed `avg_soc` has a
median within-cell correlation of -0.73 with cycle index. Against a cycle-only
baseline the same candidate scores **-0.0053**: all of its apparent skill was
age.

`avg_soc` was removed from the default candidate list and the confound
baseline was added to every candidate.
`test_a_candidate_whose_skill_is_age_is_rejected` pins the case.

This is the strongest available argument that the gate does something. It
caught a false positive that the person who wrote it did not anticipate.

## Current state

Nothing is promoted. On the real NASA frame:

```
temp_only        REJECTED  (LOBO R2=-0.0016, LOCO R2=-0.0012, vs-age R2=-0.0009)
temp_and_stress  REJECTED  (LOBO R2=-0.0147, LOCO R2=-0.0168, vs-age R2=-0.0133)
No candidate was promoted. No cohort has an active model.
```

`temp_and_stress` scoring *below* `temp_only` on every split is consistent
with the threshold reachability audit: `avg_stress` spans 1.71-33.38 against
bands at 50 and 70, so it is near-constant and contributes noise rather than
signal. Three independent lines — the threshold audit, the SHAP skill gate,
and this — now converge on the same conclusion.

Consequently `AdaptiveCalibrator.score` refuses. It does not fall back to the
most recent rejected candidate, and it does not fall back to the v1 rule-based
index, which ADR 0002 measured at rho = -0.269 against real fade. A refusal
carrying a reason is more useful than a number carrying none.

The rule-based pipeline in `main.py` is unaffected and still emits its
severity score. That remains appropriate because it is labelled throughout as
triage rather than measurement. What this system declines to do is dress that
score up as a calibrated fade prediction.

## Reopening condition

The binding constraint is data, not code. Every verdict above would be worth
re-running on a second **multi-cycle** dataset with charge-rate variation —
CALCE CS2/CX2, or Stanford/Severson. Two things would change at once: a
genuine cross-dataset test becomes possible (ADR 0001), and the fast-charge
penalty becomes constrainable for the first time, since `fast_charge_duration`
is identically zero across all 2,682 NASA observations.

The harness is written against a generic
`(cell_id, cohort, cycle, capacity_loss, features...)` frame, so a loader is
the only missing piece.

## Alternatives rejected

- **Auto-promote whatever fits best.** The literal request. It would automate
  cohort memorisation and report it as improvement.
- **Warn on failed validation but promote anyway.** A gate that can be ignored
  is documentation, not a control.
- **Fall back to the v1 index when nothing is promoted.** Substitutes a score
  measured at rho = -0.269 for an honest refusal, and hides the fact that
  nothing passed.
- **Exclude every age-correlated feature.** Would remove temperature, the one
  transferable signal this project found. The confounding is physical, so the
  answer is a harder baseline rather than a smaller feature set.
- **A density model for distribution membership.** With 3-4 cells per cohort
  there is not enough data to estimate tails whose values could be trusted. An
  observed-envelope band is cruder, honest about being crude, and fails
  predictably.
