# ADR 0010: The LOCO estimate is noisier than the effect it is used to measure

**Status:** Accepted
**Date:** 2026-08-21

## Context

Five claims were proposed and withdrawn across ADR 0008 and ADR 0009. Every
one died the same way: a method ranking read off one frame did not survive the
next.

- "High-capacity models transfer worse" — XGBoost was a counterexample.
- "Regularisation, not capacity, predicts transfer" — contradicted on CALCE.
- "Selecting by LOBO picks the worst method" — rested on one cell, B0041.
- "The LSTM is the worst transferring method on both datasets" — reversed when
  CX2 was added.
- "Behaviour adds ~0.05 R² over cycle count" — NASA-specific.

ADR 0009 proposed an explanation: the LOBO-to-LOCO gap is a property of
**cohort coverage**, not of the method, because a held-out protocol is barely
unseen when eight others remain in training. It was to be the paper's thesis.

This ADR reports the experiment that tested it.

## The experiment

`scripts/run_coverage_sweep.py` holds the method, target, harness and features
fixed and varies only the number of cohorts available. Each coverage level is
sampled with 12 random cohort subsets, on NASA and CALCE separately — never
pooled, since NASA carries temperature and SOC while CALCE records neither.
220 measurements.

The controls that mattered:

- Cohorts holding a single cell are excluded. With one cell, holding out the
  cohort *is* holding out the cell, LOCO and LOBO become the same split, and
  the gap is 0.0000 by construction. An earlier draft produced exactly that.
- Cell count is recorded per subframe and regressed out, because cohort count
  and cell count rise together.

## The hypothesis is refuted

| Dataset | n | Raw ρ(cohorts, gap) | Adjusted for cell count | ρ(cells, gap) |
|---|---:|---:|---:|---:|
| NASA | 146 | −0.029 (p = 0.73) | +0.007 (p = 0.94) | −0.031 |
| CALCE | 74 | **+0.257 (p = 0.027)** | **+0.024 (p = 0.84)** | +0.272 |
| Pooled | 220 | +0.034 (p = 0.62) | +0.025 (p = 0.71) | +0.028 |

On NASA there is no relationship at all, before or after adjustment. The
median gap is flat from three cohorts to nine: −0.38, −0.43, −0.36, −0.38,
−0.30, −0.38, −0.32.

**On CALCE the raw correlation is significant and the medians look
convincingly monotone** — −0.668, −0.347, −0.206, −0.125 as coverage grows
from three cohorts to six. Presented on its own it reads as clean
confirmation.

It is not. Adjusted for cell count the correlation collapses to +0.024
(p = 0.84), and the gap's correlation with cell count alone (+0.272) accounts
for essentially all of the raw figure. **The apparent coverage effect is a
training-set-size effect.**

Without the partial-correlation control this ADR would have reported
ρ = +0.257, p = 0.027 as support for the hypothesis. The control was the whole
experiment.

## What actually explains the five withdrawn claims

The sweep answers it directly. At a *fixed* coverage level, across 24 draws
that differ only in which cohorts were chosen:

| | Value |
|---|---:|
| Median within-level IQR of the LOCO gap | **0.724 R²** |
| Median difference, `svr_rbf` − `age_linear`, paired on identical subframes | **−0.351 R²** |
| IQR of that paired difference | 0.807 R² |
| Fraction of draws where `svr_rbf` beats `age_linear` | **0.25** |

**The spread of the LOCO estimate between cohort draws is roughly twice the
median difference between a nonlinear model and a straight line on cycle
count.** The measurement noise exceeds the effect being measured.

That is why five rankings failed to replicate. Each was a single draw from a
distribution whose interquartile range is wider than the gaps between the
methods being ranked. Nothing was wrong with any individual run; the estimator
simply does not resolve what it was being asked to resolve.

Note also the paired comparison: on identical subframes the nonlinear model
*loses* to the age baseline in three draws out of four, with a median deficit
of 0.351. Full-frame runs where `svr_rbf` or the tree ensembles "won" were
draw-specific outcomes, not stable orderings.

## Decision

**Report leave-one-cohort-out as an interval over cohort draws, never as a
point estimate, and do not rank methods on a single LOCO run.**

Three consequences:

1. `benchmarks/coverage.py` ships as the tool that produces the interval.
   Any claim in this project that one method transfers better than another
   must be accompanied by the paired difference across draws, not a
   single-frame comparison.

2. **The cohort-coverage thesis is withdrawn before publication rather than
   after.** ADR 0009's addendum proposing it is superseded by this document.

3. The claim the evidence supports is narrower and about methodology rather
   than about batteries: *at the sample sizes standard battery datasets
   provide, a single leave-one-cohort-out estimate cannot distinguish between
   candidate methods, and the interval is wide enough to reverse orderings.*

## What survives everything

**Every method loses skill under protocol shift.** The gap is negative in
every one of the 36 method-frame combinations in ADR 0008 and ADR 0009, and
its median is negative at every coverage level in all 220 sweep measurements
here. Not one configuration transferred without loss.

That claim has now survived a screen revision, a dataset addition, a cell
exclusion that reversed a headline, and a designed experiment aimed at
explaining it away. It is the one result this project should stand on.

## Limitations

- Two methods in the sweep (`age_linear`, `svr_rbf`). `elasticnet` and
  `random_forest` were excluded on runtime — 300 trees refitted across every
  fold of every subframe did not finish a dataset in eight minutes. A wider
  method set might narrow or widen the paired-difference distribution.
- Cells are thinned to at most 200 cycles each to bound compute. Thinning is
  uniform in cycle index and applied identically at every coverage level, so
  it cannot bias the trend, but it does reduce within-cell resolution.
- Cohort subsets are sampled, not enumerated. With 12 draws per level the
  interval estimates are themselves uncertain.
- The partial correlation adjusts for cell count linearly in rank space. A
  nonlinear confound would not be fully removed.

## References

- `scripts/run_coverage_sweep.py`, `src/bms/benchmarks/coverage.py`
- `reports/metrics/coverage_sweep.csv` — the 220 measurements
- `reports/metrics/coverage_sweep_summary.csv`
- ADR 0008, ADR 0009 (the withdrawn claims this explains)
