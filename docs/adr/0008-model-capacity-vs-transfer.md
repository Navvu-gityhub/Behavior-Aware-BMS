# ADR 0008: Model capacity buys within-protocol skill and spends it on transfer

**Status:** Accepted
**Date:** 2026-08-21

## Context

ADR 0007 established that `capacity_loss` has an R² ceiling of 0.044 and that
`cumulative_fade` is the well-conditioned target. That made a question
answerable that previously was not: on a target with real signal, how do
standard published methods behave under leave-one-cohort-out?

`scripts/run_benchmark_study.py` runs twelve methods through the project's own
`Validator` — same gate, same mandatory LOCO, same age-confound baseline.

## The result

Target `cumulative_fade`, 2,648 rows, 32 cells, 9 cohorts, R² ceiling 0.612.
Sorted by LOBO, which is what the field's standard protocol would report:

| Method | LOBO R² | LOCO R² | Δ |
|---|---|---|---|
| `random_forest` | **0.785** | **−0.515** | −1.300 |
| `mlp` | 0.761 | −0.264 | −1.026 |
| `gpr_matern` | 0.758 | −0.106 | −0.864 |
| `hist_gradient_boosting` | 0.754 | −0.059 | −0.813 |
| `svr_rbf` | 0.669 | 0.182 | −0.487 |
| `elasticnet` | 0.651 | **0.295** | −0.357 |
| `arrhenius_avg_temp` | 0.560 | 0.069 | −0.491 |
| `arrhenius_trailing_temp` | 0.558 | 0.052 | −0.506 |
| `age_linear` | 0.383 | 0.274 | **−0.108** |
| `age_isotonic` | 0.212 | −0.011 | −0.223 |
| `age_quadratic` | 0.203 | 0.035 | −0.167 |
| `train_mean` | 0.000 | 0.000 | 0.000 |

Two readings, one robust and one suggestive.

### The selection failure is concrete and needs no statistics

**Choosing the best method by leave-one-cell-out selects `random_forest`,
which is the worst method in the table under protocol shift** — LOCO
R² = −0.515, meaningfully worse than emitting the training mean. The method
that actually transfers best, `elasticnet` at LOCO R² = 0.295, ranks *sixth*
on LOBO and would not be chosen.

That is a single concrete instance of the failure this project has been
arguing about in the abstract, on a real dataset, with the field's own
standard protocol doing the choosing.

### The capacity-versus-transfer ordering is suggestive but underpowered

Across the twelve methods, Spearman correlation between LOBO rank and LOCO
rank is **−0.490 (p = 0.106, n = 12)**. Negative, in the direction that the
most flexible models transfer worst — but *not significant at n = 12*, and it
is reported that way rather than as an established relationship. Twelve
methods is a small sample and the choice of which methods to include is itself
a researcher degree of freedom.

What is not in doubt is the magnitude ordering at the extremes: the four
highest-capacity methods (random forest, neural network, Gaussian process,
gradient boosting) occupy the four worst Δ positions, ranging from −0.81 to
−1.30, while the simplest non-trivial method in the table, `age_linear`, has
the smallest Δ at −0.108.

The mechanism is the one ADR 0002 identified for the fitted v2 health model:
capacity is spent learning per-cohort structure. On a held-out cell of a known
protocol that structure is present and helps enormously. On a held-out
protocol it is absent and actively misleads.

### On the degenerate target, the pattern vanishes

On `capacity_loss`, Spearman(LOBO, LOCO) is **+0.818 (p = 0.001)** and every
method sits within ±0.01 of zero on both splits. When a target has no signal
there is nothing to overfit to, so the two splits agree — on nothing. This is
a useful control: it shows the divergence on `cumulative_fade` is a property
of a target with real structure, not an artefact of the harness.

> ## ⚠ SUPERSEDED IN PART — read this first
>
> The table and the selection-failure claim below were computed **before** the
> fourth SOH screen criterion (ADR 0007) excluded cell **B0041**. Re-running
> with 31 cells instead of 32 changes them materially, and the dramatic
> version of this ADR's headline does not survive.
>
> The corrected numbers, the withdrawn claims, and what actually replicates
> are in the second addendum at the end of this document. The original text is
> kept because the revision is the point: a single pathological cell was
> carrying a conclusion, and the control that found it is the same kind of
> control this project argues for.

## Addendum, 2026-08-21: XGBoost is a counterexample, and it sharpens the claim

Two models were added after this ADR was written — real XGBoost and a real
LSTM (`docs/benchmark_models.md`). On the `soh` target, 2,648 rows, 32 cells,
9 cohorts:

| Method | LOBO MAE | LOCO MAE | LOBO R² | LOCO R² |
|---|---|---|---|---|
| **`xgboost`** | 4.89% | **7.89%** | **0.807** | **0.391** |
| `elasticnet` | 7.11% | 10.28% | 0.651 | 0.295 |
| `random_forest` | 4.50% | 11.12% | 0.784 | −0.536 |
| `hist_gradient_boosting` | 4.35% | 15.97% | 0.754 | −0.059 |
| `lstm` | 6.15% | **20.37%** | 0.646 | −0.065 |

**XGBoost is the best method on both splits simultaneously.** It is a
high-capacity tree ensemble, so the "suggestive" ordering recorded above —
that more flexible models transfer worse — has a direct counterexample.

That ordering was labelled underpowered (ρ = −0.490, p = 0.106, n = 12) when
first reported, and this is what an underpowered claim failing looks like. It
is withdrawn as a general statement.

**What survives is more precise and more useful: it is regularisation, not
capacity.** XGBoost differs from the two ensembles it beats by carrying an
explicit L2 penalty on leaf weights plus row and column subsampling. Random
forest and sklearn's histogram booster have neither in the configurations
tested. Ranked by LOCO, the regularised models — XGBoost, ElasticNet, and the
age baselines — occupy the top of the table, and the unregularised
high-capacity ones occupy the bottom.

**The headline finding is untouched.** Selecting by LOBO still picks
`random_forest` (LOBO R² 0.784, second only to XGBoost's 0.807 on this
target), which still lands at LOCO −0.536. And every method still loses
substantial skill under protocol shift, XGBoost included (Δ = −0.416).

## Decision

**Report LOBO and LOCO together, always, and select on LOCO.**

Three consequences encoded in the code rather than in guidance:

1. `Validator.gate` already requires LOCO and rejects a candidate supplied
   without it (ADR 0005). This result is the empirical justification for
   that rule rather than an argument for it.

2. `StudyResult.to_frame` emits `loco_minus_lobo` as a first-class column. The
   gap is the diagnostic, not a derived quantity a reader should have to
   compute.

3. `benchmark_results.csv` is tracked, so the table above is checkable rather
   than quoted.

## The Arrhenius model passes the gate and remains physically inadmissible

`arrhenius_avg_temp` scores LOBO 0.560 / LOCO 0.069 and is marked PROMOTED.
It is nonetheless the same fit that ADR 0007 records as returning
**Ea = −31 kJ/mol** — an activation energy saying degradation slows with heat.

These are not in conflict, and the pair is worth keeping in view. The gate
measures out-of-sample predictive skill. It does not, and cannot, check
whether a fitted parameter means what its name says. A model can predict
adequately while its coefficients are physically nonsense, because a wrong
sign on one term can be compensated by the rest of the fit.

So `physics/arrhenius.assess_identifiability` remains a separate check that
runs *before* fitting and refuses independently of the gate's verdict. Passing
a predictive gate is not a licence to quote a coefficient as a physical
constant, and this project now has a worked example of exactly that
distinction.

## Consequences

**The project's "nothing is promoted" state is no longer accurate, and the
documentation has been updated.** On `cumulative_fade`, four candidates clear
the gate: `elasticnet`, both Arrhenius variants, and `age_quadratic`. Three of
those four beat the training mean by a margin that is small in absolute terms
(LOCO R² between 0.035 and 0.295 against a ceiling of 0.612).

**`elasticnet` is the strongest available candidate**, and it is a penalised
linear model on six behavioural features. It is not promoted into the
dashboard: ADR 0005's rule that the gate is the product still holds, and one
dataset is not enough evidence to deploy on. It is the candidate to carry into
the CALCE replication.

**`age_linear` remains the baseline to beat, and beating it is harder than it
looks.** At LOCO R² = 0.274 versus ElasticNet's 0.295, cycle count alone
recovers 93% of the best behavioural model's cross-protocol skill. Whatever
behaviour contributes beyond age on this dataset, it is small.

## Second addendum, 2026-08-21: one cell was carrying the headline

Re-run on 31 cells after the fourth screen criterion removed B0041 — a cell
whose capacity reference was already known to be unreliable, and which reads
as 4.5% state of health for most of its life. Target `soh`, 2,585 rows, 9
cohorts, ceiling **0.569**.

| Method | LOBO MAE | LOCO MAE | LOBO R² | LOCO R² | Δ |
|---|---:|---:|---:|---:|---:|
| `xgboost` | **4.21%** | **7.62%** | 0.732 | **0.459** | −0.273 |
| `age_linear` | 8.54% | 10.71% | 0.483 | 0.406 | **−0.076** |
| `elasticnet` | 7.61% | 10.31% | 0.609 | 0.355 | −0.254 |
| `gpr_matern` | 4.48% | 17.06% | 0.725 | 0.207 | −0.518 |
| `random_forest` | 4.90% | 10.67% | **0.773** | 0.150 | −0.623 |
| `svr_rbf` | 6.91% | 14.29% | 0.651 | 0.087 | −0.564 |
| `age_quadratic` | 6.62% | 8.31% | 0.336 | 0.034 | −0.301 |
| `age_isotonic` | 6.99% | 8.50% | 0.278 | 0.018 | −0.260 |
| `train_mean` | 11.17% | 11.92% | 0.000 | 0.000 | 0.000 |
| `lstm` | 5.62% | 17.36% | 0.594 | −0.185 | −0.778 |
| `hist_gradient_boosting` | 4.12% | 10.27% | 0.770 | −0.265 | −1.035 |
| `mlp` | 4.15% | 14.19% | 0.731 | −0.623 | −1.354 |

### What is withdrawn

**"Selecting by LOBO picks the worst method in the table."** On 32 cells,
LOBO selected `random_forest` at LOCO −0.536, last of twelve. On 31 cells the
same procedure still selects `random_forest`, but its LOCO is **+0.150** —
fifth of twelve, and positive. One cell was supplying most of that collapse.

**The rank correlation.** ρ(LOBO, LOCO) was −0.490 (p = 0.106) on 32 cells and
is **+0.084 (p = 0.795)** on 31. Neither is significant; the sign flipped on
one cell. Every ordering claim built on it is gone, in both directions.

### What survives, and is now cleaner

**LOBO rank carries essentially no information about LOCO rank.** ρ ≈ 0 is a
more defensible statement than the inversion originally claimed, and it is
the one that matters for practice: a leaderboard sorted by leave-one-cell-out
tells you close to nothing about which model to deploy across protocols.

**Choosing by LOBO still costs 0.309 R².** `random_forest` (best LOBO) reaches
LOCO 0.150; `xgboost` (best LOCO) reaches 0.459. The failure is real, just not
catastrophic.

**Every method still loses skill**, from −0.076 to −1.354. Three still go
negative — `mlp`, `hist_gradient_boosting`, `lstm`.

**`age_linear` is the most robust method in the table**, and by a wide margin:
Δ = −0.076 against the next-best −0.254. A straight line on cycle count reaches
LOCO R² 0.406, against XGBoost's 0.459. **All the behavioural modelling buys
about 0.05 R² over counting cycles.** That is consistent with Section 4.7's
threshold audit and with the CALCE result, and it is arguably the most
important number in this project.

### Why this correction is kept rather than quietly applied

The dramatic version of this ADR rested on one cell. The control that found it
— a screen criterion asking whether a cell's reference capacity means what it
says — is exactly the kind of check this project argues the field should run,
and applying it to my own headline weakened it. Reporting that is cheaper than
having a reviewer find it.

## References

- `scripts/run_benchmark_study.py` — reproduces the table
- `reports/metrics/benchmark_results.csv` — the tracked result (31-cell run)
- `reports/metrics/calce/benchmark_results.csv` — the CALCE replication
- `src/bms/adaptive/validation.py` — the gate
- ADR 0002 (cohort intercepts), ADR 0005 (the gate is the product),
  ADR 0007 (target definition), ADR 0009 (CALCE replication)
