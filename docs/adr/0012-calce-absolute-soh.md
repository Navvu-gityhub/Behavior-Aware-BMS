# ADR 0012: An absolute SOH target on CALCE recovers four cells and refutes ADR 0009's "learned models add enormously"

**Status:** Accepted
**Date:** 2026-08-26

## Context

ADR 0009 replicated the LOBO-to-LOCO collapse on CALCE CS2+CX2 and left one
item open, which `docs/roadmap.md` called the largest open quality item:

> Types 5 and 6 pass, and their SOH is measured against a **partial-cycle**
> reference of ~0.26–0.42 Ah. That tracks relative fade of a repeated partial
> cycle, which is a real signal, but it is not absolute state of health.

The cause is that the cycle-level target groups on Arbin's `Cycle_Index` and
takes the within-group span of `Discharge_Capacity`. That is one discharge only
for Types 1 and 2. Type 3 switches discharge rate six times within a cycle;
Types 5 and 6 cycle partially by design.

`src/bms/io/calce_full_discharge.py` replaces that with contiguous
negative-current-run segmentation on sample-level telemetry, delegating to
`telemetry/cycles.segment_phases` — the same function the CAN and serial
telemetry paths use, so a discharge means one thing across the project.

## Three things had to be got right, and each was got wrong first

**A cycle-level fix cannot work.** Selecting the `Cycle_Index` groups that look
full — reaching cutoff, with a near-maximal span — returns references of
1.40–1.75 Ah for the Type 5 and 6 cells, against 1.1 Ah nominal, while the same
rule on Types 1 and 2 returns a correct 1.12–1.13 Ah. The inflation is the
diagnosis: a group holding several discharges has a span covering all of them,
so selecting the largest spans preferentially selects the groups containing the
most discharges. Anything still trusting `Cycle_Index` inherits the defect it is
trying to remove.

**Arbin's test clock restarts at every file boundary.** CS2_33 resets 21 times,
the largest being `819352.3 -> 30.0`. Sorting the concatenated frame by that
column interleaves twenty-two unrelated files and shatters every discharge into
one-sample runs — detected capability 0.002 Ah on a 1.1 Ah cell, and 12,018
"discharges" for a cell with 4,990 cycles. Row order is the true temporal order;
the time column is not.

**The cutoff cannot be a quantile of terminal voltages.** The roadmap proposed
detecting "contiguous negative-current runs terminating at the voltage cutoff".
The cutoff half is necessary and nowhere near sufficient:

| Cell | Type | discharges reaching cutoff | median charge moved |
|---|---|---:|---:|
| CS2_24 | 6 | 53 / 4,990 | 0.288 Ah |
| CS2_5 | 5 | 6,951 / 6,962 | 0.237 Ah |

Type 6 partials stop at 3.78 V and separate cleanly. Type 5 partials *do* reach
the 2.7 V cutoff, discharging to it from a partially charged state, so 99.8% of
them pass a cutoff test while delivering a fifth of the cell's capacity.

Worse, a quantile of terminal voltages is set by whichever discharge type is
most common, so for CS2_24 even a 2nd-percentile cutoff returns **3.78 V** — the
partials define the "cutoff", every partial then trivially reaches it, and the
detector certifies precisely the cycles it exists to reject. The cutoff is
therefore estimated from the deepest discharges only, and paired with a
charge-capability criterion.

## What the detector recovers

Per-cell references move from partial to physical. 22 of 23 processed cells
return a cutoff of 2.694–2.700 V.

| Cell | Type | old reference | new capability | nominal |
|---|---|---:|---:|---:|
| CS2_5 | 5 | 0.177 Ah | **1.055 Ah** | 1.1 |
| CS2_6 | 5 | 0.265 Ah | 0.747 Ah | 1.1 |
| CS2_24 | 6 | 0.367 Ah | **1.101 Ah** | 1.1 |
| CS2_25 | 6 | 0.432 Ah | **1.103 Ah** | 1.1 |
| CS2_33 | 1 | 1.120 Ah | 1.143 Ah | 1.1 |
| CX2_35 | 1 | 1.304 Ah | 1.344 Ah | 1.35 |

**Four cells the screen had excluded are recovered**, because their exclusions
were consequences of the `Cycle_Index` target rather than properties of the
cells: CS2_3 and CS2_9 (Type 3, references computing to 5.4 and 6.5 Ah on 1.1 Ah
cells), CX2_3 (reference 0.003 of maximum), CX2_8 (27.8% median step). All four
now pass every screen criterion.

| | rows | cells | cohorts | admissible | SOH > 1.05 masked | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| `Cycle_Index` target | 136,661 | 23 | 11 | 19/23 | 505 | 0.870 |
| Full-discharge target | 11,529 | 22 | 10 | **22/22** | 14 | **0.907** |

One cell is lost. **CX2_32 is refused** by a new guard: the detector returned a
capability of 0.0397 Ah, 2.9% of that cell's own largest discharge (1.357 Ah),
on an estimated cutoff of 1.999 V where every other cell gives 2.70 V. Nothing
downstream catches this — every retained row is equally wrong, so the reference
equals the maximum, the median SOH reads 1.0, and all five screen criteria pass.
A detector that can be confidently wrong must refuse for itself.

**The cost is most of the rows.** A Type 5 or 6 cell yields 36–73 full
discharges out of 5,000–7,000 cycles. That is a property of the protocol, not a
defect, and it is reported rather than engineered around.

## The result

Target `soh`, identical explicit feature set on both frames (`cycle`,
`mean_voltage_v`, `min_voltage_v`, `mean_current_a`, `resistance_ohm`,
`cycle_duration_s`), sorted by LOCO.

**Full-discharge target** — 22 cells, 10 cohorts, 11,529 rows, ceiling 0.907:

| Method | LOBO R² | LOCO R² | Δ |
|---|---:|---:|---:|
| `random_forest` | 0.9061 | **0.6696** | −0.2365 |
| `age_isotonic` | 0.7781 | **0.5238** | −0.2543 |
| `age_quadratic` | 0.7786 | **0.5149** | −0.2638 |
| `age_linear` | 0.7264 | **0.4780** | −0.2484 |
| `xgboost` | 0.8730 | 0.4082 | −0.4649 |
| `elasticnet` | 0.7177 | 0.3883 | −0.3294 |
| `hist_gradient_boosting` | 0.8827 | 0.2932 | −0.5895 |
| `svr_rbf` | 0.3707 | 0.2345 | −0.1362 |
| `lstm` | 0.6570 | 0.0566 | −0.6004 |
| `train_mean` | 0.0000 | 0.0000 | 0.0000 |
| `mlp` | 0.7061 | −1.4613 | −2.1674 |
| `gpr_matern` | 0.6584 | −2.0874 | −2.7458 |

**Controlled baseline**, same features, `Cycle_Index` target — 19 cells, 8
cohorts, 43,832 rows, ceiling 0.870:

| Method | LOBO R² | LOCO R² | Δ |
|---|---:|---:|---:|
| `xgboost` | 0.9320 | 0.6971 | −0.2348 |
| `hist_gradient_boosting` | 0.9264 | 0.6751 | −0.2513 |
| `random_forest` | 0.9302 | 0.6647 | −0.2655 |
| `mlp` | 0.9086 | 0.6061 | −0.3025 |
| `gpr_matern` | 0.8184 | 0.4375 | −0.3809 |
| `svr_rbf` | 0.8229 | 0.3089 | −0.5140 |
| `lstm` | 0.9362 | 0.2586 | −0.6776 |
| `age_isotonic` | 0.3387 | 0.1980 | −0.1407 |
| `elasticnet` | 0.1921 | 0.1209 | −0.0712 |
| `age_quadratic` | 0.1425 | 0.0171 | −0.1254 |
| `age_linear` | 0.0956 | 0.0159 | −0.0796 |
| `train_mean` | 0.0000 | 0.0000 | 0.0000 |

This baseline does **not** reproduce ADR 0009's published table — `lstm` scores
LOCO 0.259 here against 0.526 there, and the LOCO ordering differs — because
ADR 0009 did not record the feature set it used. That is a reproducibility
defect in ADR 0009, noted here rather than papered over, and it is why the
comparison above is against a re-run baseline rather than the published numbers.

## What this changes

**1. ADR 0009's claim that "the learned models add enormously" is refuted.**

ADR 0009 wrote:

> **"Behaviour adds about 0.05 R² over counting cycles."** NASA-specific. Here
> `age_linear` reaches LOCO 0.016 while `random_forest` reaches 0.693. The
> learned models add enormously.

On an absolute SOH target the age baselines are transformed:

| Method | LOCO, `Cycle_Index` | LOCO, full-discharge |
|---|---:|---:|
| `age_isotonic` | 0.198 | **0.524** |
| `age_quadratic` | 0.017 | **0.515** |
| `age_linear` | 0.016 | **0.478** |

Cycle count alone now reaches LOCO 0.48–0.52 and **beats every learned method
except `random_forest`**. The gap between counting cycles and the best model
falls from 0.68 R² to 0.19 R².

The apparent enormous advantage of learned models was substantially an artifact
of the partial-cycle reference. Against a target that alternates between full
and partial discharge capacities, a smooth function of cycle number cannot
track the alternation while a flexible model can partially fit it — so the
flexible models were rewarded for fitting a measurement artifact, not
degradation.

**2. A fourth mutually inconsistent method ranking.** `xgboost` falls from LOCO
0.697 to 0.408; `hist_gradient_boosting` from 0.675 to 0.293; `mlp` and
`gpr_matern` go from healthy positives to −1.46 and −2.09. `random_forest` is
now the only learned method above the age baselines.

This project has now produced four incompatible rankings by varying only the
target derivation, the cohort coverage, and the feature set — none of them the
model. ADR 0009's decision stands and is strengthened: **report the collapse,
never a ranking.**

**3. The collapse itself replicates a third time.** All twelve deltas are
negative, on a cleaner target with more cells and more cohorts. This is the only
claim that has survived every revision in this project.

**4. Cohort coverage is not sufficient to explain the gap.** ADR 0009 found
LOCO gets easier as cohort coverage grows. Coverage rises here from 8 to 10
cohorts, and most methods got *worse*. Coverage is one factor, not the factor.

## Decision

**Adopt the full-discharge target as the CALCE target of record**, and keep the
`Cycle_Index` frame tracked alongside it as the baseline this ADR compares
against.

**Withdraw ADR 0009's "learned models add enormously" claim.** It joins the
capacity and regularisation claims withdrawn from ADR 0008 — the third ranking
claim this project has made and retracted.

**Do not report the two frames' numbers in one table without the caveat below.**

## Limitations, stated because they bound the claim

- **The comparison is not a clean controlled experiment.** The two frames differ
  in rows (43,832 vs 11,529), cells (19 vs 22) and cohorts (8 vs 10). The cell
  set changed *because* the target changed, which is the point of the exercise
  and also confounds it. Attributing the whole difference to target derivation
  would overstate what was measured.
- **Absolute SOH on partial-cycling protocols is sparse.** CS2_24 contributes 36
  points and CS2_25 contributes 45. Cohort-level conclusions about Types 5 and 6
  rest on tens of observations.
- **CS2_6's capability is 0.747 Ah against 1.1 Ah nominal.** It passes the guard
  and every screen, but it is the one retained cell whose reference is not
  clearly a fresh-capacity figure. It may be a genuinely degraded cell or a
  protocol whose deepest discharge is not full. Not resolved here.
- **`arrhenius_*` still ERROR and the curve methods still UNAVAILABLE.** CALCE
  records no temperature channel, and this reduction carries cycle-level
  aggregates rather than per-cycle voltage traces.

## References

- `src/bms/io/calce_full_discharge.py` — the detector, and why each criterion exists
- `scripts/build_calce_full_discharge_frame.py` — `make calce-full-discharge`
- `tests/test_calce_full_discharge.py` — the three defects above, asserted
- `reports/metrics/calce_full_discharge.csv` — the target of record
- `reports/metrics/calce_full_discharge_yield.csv` — per-cell yield and refusals
- `reports/metrics/calce_full_discharge/benchmark_results.csv` — the first table
- `reports/metrics/calce_baseline_controlled/benchmark_results.csv` — the second
- ADR 0007 (target definition), ADR 0009 (CALCE replication), ADR 0010 (LOCO variance)
