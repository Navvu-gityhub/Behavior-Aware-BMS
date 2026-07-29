# ADR 0002: health_index v1 stays the default; v2 is exposed conditionally

**Status:** Accepted
**Date:** 2026-07-29
**Closes:** the open question "should `health_index_v2.py` replace or augment the pipeline default?"

## Context

`src/bms/health/health_index_v2.py` was built, fitted against NASA, honestly
documented — and then left unwired. Neither promoted nor removed.

An orphaned module is a credibility problem in its own right. A reader cannot
tell whether it is abandoned, pending, or quietly believed to be better than
what ships. This ADR closes that, and `scripts/validate_health_index_versions.py`
generates the evidence.

## The two candidates are not interchangeable

A naive "which has the better R-squared?" comparison is meaningless here,
because the candidates emit different kinds of quantity:

- **v1** emits a 0–100 severity index per battery. It is a **ranking and
  triage** instrument. It does not predict a physical quantity, so R-squared
  against capacity loss is undefined for it.
- **v2** emits a predicted per-cycle capacity loss in **amp-hours**. It is a
  **regression** instrument, and R-squared is the right metric.

So each was evaluated on the task it actually claims to perform.

## Evidence

### Task A — ranking batteries by measured fade rate (n = 33 NASA cells)

Spearman correlation against `fade_rate_ah_per_cycle`, permutation-tested
(5,000 resamples) because n = 33 with a heavy-tailed target makes the
analytic p-value untrustworthy.

| Candidate | rho | p | Distinct output values |
|---|---|---|---|
| v1 (rule-based) | **−0.269** | 0.124 | **6** of 33 |
| v2, in-sample | 0.870 | 0.0002 | 33 |
| v2, LOBO-refit | **0.841** | 0.0002 | 33 |
| v2, LOCO-refit | **−0.295** | 0.097 | 33 |

The in-sample row is inadmissible: v2's shipped coefficients include a fitted
intercept per NASA cohort, estimated on these same cells, and fade rate varies
strongly by cohort. Refitting per fold is what makes the other two rows
meaningful.

### Task B — predicting per-cycle capacity loss (v2's specification, refit per fold)

| Split | Median R² vs global mean | Folds beating baseline |
|---|---|---|
| LOBO | +0.008 | 73% |
| LOCO | **−0.167** | **11%** |

## Findings

1. **v1 does not rank batteries by real degradation.** rho = −0.269 is
   non-significant *and negative*. v1 is not merely unvalidated; the best
   available estimate of its ranking ability points slightly backwards.
2. **v1 has almost no resolution.** Six distinct values across 33 cells. ADR
   0003 and `scripts/audit_threshold_reachability.py` explain the mechanism:
   61% of the mean risk score comes from terms that are *identical for every
   battery in the dataset*, because their thresholds sit outside the range
   real data occupies.
3. **v2's ranking ability is real but conditional.** It survives holding out a
   cell (rho = 0.841) and collapses when the cell's entire protocol is held
   out (rho = −0.295). The ability lives in the fitted cohort intercepts, not
   in the temperature coefficient.

## Decision

**v1 remains the pipeline default. v2 is exposed as a conditional,
explicitly-labelled signal. Neither is presented as a validated fade predictor.**

Concretely:

1. `main.py` continues to call `compute_health_index` (v1). It requires no
   cohort label, so it runs on any input — which is the only property that
   currently justifies its position.
2. v1's output is **relabelled** wherever it surfaces. The dashboard states
   that it is a severity score, not a fade measurement, and the evidence panel
   carries rho = −0.27 on the face of the UI.
3. `predict_capacity_loss_per_cycle` stays available and documented for use
   **only where the operating protocol is known and represented in training**.
   Its module docstring and `MODEL_QUALITY` record the LOCO collapse.
4. v2 is **not** promoted to default, because the pipeline cannot in general
   know a new battery's cohort, and the unknown-cohort fallback is exactly the
   case that fails.

## Why not the obvious alternatives

- **Promote v2 unconditionally.** Its advantage exists only when the cohort is
  known. Deploying it against an unseen protocol substitutes a
  confidently-wrong signal for a weakly-wrong one.
- **Delete v2.** It is the project's clearest positive result — a real,
  cross-validated within-protocol ranking signal — and the mechanism of its
  failure is itself the main finding. Deleting it would discard both.
- **Blend v1 and v2.** A weighted combination of an instrument that ranks
  backwards and one that only works conditionally has no defensible weighting
  and no data to fit one with.
- **Keep v1 unchanged and unlabelled.** This was the status quo, and it is the
  option this ADR exists to reject. Shipping a score whose measured ranking
  correlation is negative, without saying so, is the credibility risk.

## Consequence

The pipeline's headline number is now accompanied, in the product itself, by
the evidence that it is not validated. That is a worse-looking dashboard and a
more defensible project. The path to a genuinely better default is stated in
ADR 0001: a multi-cycle second dataset, which would let the cohort-intercept
question be tested rather than argued.
