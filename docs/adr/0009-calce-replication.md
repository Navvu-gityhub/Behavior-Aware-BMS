# ADR 0009: The protocol-shift collapse replicates on CALCE, but its shape is dataset-specific

**Status:** Accepted
**Date:** 2026-08-21

## Context

Every quantitative result in this project came from NASA until now. ADR 0008
reported that leave-one-cell-out cross-validation selects a model that fails
under leave-one-cohort-out, and `docs/paper_outline.md` named replication on a
second dataset family as the blocking item — a reviewer would reject on that
ground alone, and would be right to.

CALCE CS2 was the download the commensurability screen said was worth making:
FEASIBLE on four axes, and — critically — its cohort structure varies **depth
of discharge and discharge rate**, not temperature. If the collapse reproduces
there, it is a property of protocol shift rather than of NASA.

15 cells downloaded, 13 loadable (two are CADEX-format and refused), 11
admitted by the SOH screen, across 5 cohorts. 31,311 cycle rows.

## The result

Target `soh`, CALCE CS2, 11 cells, 5 cohorts, noise ceiling **0.888**. Sorted
by LOCO, which is the column that matters:

| Method | LOBO MAE | LOCO MAE | LOBO R² | LOCO R² | Δ R² |
|---|---:|---:|---:|---:|---:|
| `random_forest` | 5.13% | **15.20%** | 0.871 | **0.636** | −0.235 |
| `xgboost` | **3.97%** | 15.89% | **0.919** | 0.615 | −0.303 |
| `hist_gradient_boosting` | 4.80% | 18.79% | 0.907 | 0.480 | −0.428 |
| `elasticnet` | 14.92% | 20.02% | 0.133 | 0.009 | −0.124 |
| `train_mean` | 19.03% | 20.89% | 0.000 | 0.000 | 0.000 |
| `age_isotonic` | 15.32% | 19.90% | 0.153 | −0.005 | −0.158 |
| `gpr_matern` | 5.76% | 28.64% | 0.866 | −0.263 | −1.129 |
| `mlp` | 8.19% | 29.36% | 0.830 | −0.643 | −1.473 |
| `svr_rbf` | 7.69% | 34.36% | 0.774 | −2.156 | −2.930 |
| **`lstm`** | 6.69% | **62.16%** | 0.906 | **−4.669** | **−5.574** |

Curve-based methods remain UNAVAILABLE. Both Arrhenius variants report ERROR:
they require `avg_temp`, and **CALCE records no temperature channel at all** —
correctly surfaced rather than silently scored.

## What replicates

**The collapse itself, unambiguously.** Every method loses skill moving from a
held-out cell to a held-out protocol, on a dataset from a different laboratory
with a different cohort axis. Deltas run from −0.12 to −5.57 R². Six of twelve
methods go from strongly positive LOBO to *negative* LOCO — worse than
predicting a constant.

**The LSTM's failure mode, emphatically.** It posts the third-best LOBO R²
(0.906) and by far the worst LOCO (−4.669, MAE 62%). It was also the
worst-transferring method on NASA. A sequence model over cycle-level
aggregates learns cell-specific trajectory shape, and that shape does not
survive a protocol change. This is now a two-dataset finding.

## What does not replicate, and this matters

**Which family transfers best is dataset-dependent.** ADR 0008 recorded, on
NASA, that the tree ensembles occupied the four worst Δ positions and that
`elasticnet` transferred best. On CALCE the ordering **inverts**: the three
tree ensembles take the top three LOCO positions, and `elasticnet` is barely
better than the training mean.

The mechanism is visible in the LOBO column. `elasticnet` scores LOBO
R² = 0.133 on CALCE — it cannot fit the data *in sample*. CALCE carries no
temperature and no SOC, so the only available features are electrical
(voltage, current, internal resistance, cycle duration) and their relationship
to state of health is strongly nonlinear. A penalised linear model has nothing
to work with; trees do.

**So ADR 0008's revised claim — "it is regularisation, not capacity" — does
not survive either.** On CALCE, `random_forest` carries no explicit
regularisation and transfers best of all. That claim is withdrawn. It is the
second time this ordering has been claimed and then contradicted by new data,
which is itself the useful lesson: with 12 methods on one dataset, the ranking
*among* methods is not identifiable.

**And the selection failure is milder here.** Choosing by LOBO picks
`xgboost`, which ranks second on LOCO. On NASA the same procedure picked
`random_forest`, the worst method in that table. The failure is real on both
datasets but its severity is not predictable.

## Decision

**Report the collapse as the finding. Do not report a method ranking as one.**

Three consequences:

1. **The paper's claim narrows to what two datasets support**: leave-one-cell-out
   overstates deployable skill, on both datasets, for every method tested. That
   is robust. *Which* method to prefer is not, and any table ordering methods
   must carry the dataset it was measured on.

2. **`docs/adr/0008` capacity and regularisation claims are withdrawn**, not
   merely qualified. Both were labelled suggestive when first made; both have
   now been contradicted.

3. **CALCE and NASA results are not feature-comparable and must not be placed
   in one table without saying so.** NASA uses six behavioural features plus
   cycle; CALCE uses four electrical features plus cycle, because it records
   nothing else. The comparison is of a *phenomenon*, not of a model.

## The `<3%` target, on the best available evidence

`xgboost` reaches **LOBO MAE 3.97%** on CALCE — the lowest SOH error anywhere
in this project, on its cleanest target (ceiling 0.888 against NASA's 0.569).
It is still above 3%, and its LOCO error is 15.89%.

So the target remains unmet on the more favourable split of the more
favourable dataset, and is off by a factor of four on the honest split.

## Loading CALCE surfaced four defects

Worth recording, because three were in code that predated this work and all
four were invisible without real data:

1. **Arbin workbooks carry two sheets.** `pd.read_excel` takes the first,
   which is a metadata block named `Info`, not the channel log.
2. **`Discharge_Capacity(Ah)` does not reset between cycles.** The loader took
   a per-cycle maximum on the documented assumption that it does; it
   accumulates, so CS2_33 reported capacity rising 1.16 → 4.45 Ah and SOH of
   383% on a 1.1 Ah cell. The correct quantity is the within-cycle span.
   **The regression fixture encoded the same false assumption**, so the suite
   confirmed the bug rather than catching it.
3. **`_sort_key` called `Path.stat()`** on archive members that have no
   filesystem entry, losing an entire cell over one oddly-named file.
4. **CADEX exports were misread rather than refused.** CS2_8 and CS2_21 are
   tab-separated mV/mA exports; through the Arbin path CS2_21 read as
   "capacity 97.0 → 86.0 Ah" on a 1.1 Ah cell. Now refused with a reason,
   costing no cohort (both are Type 1, which retains CS2_33 and CS2_34).

## Open: Types 3, 5 and 6

`Cycle_Index` equals one full discharge only for Types 1 and 2. Type 3
switches discharge rate six times within a cycle; Types 5 and 6 cycle
partially by design.

The screen's fourth criterion (ADR 0007) excludes Type 3, whose reference
computes to 5.4 Ah on a 1.1 Ah cell. Types 5 and 6 pass, and their SOH is
measured against a **partial-cycle** reference of ~0.26–0.42 Ah. That tracks
relative fade of a repeated partial cycle, which is a real signal, but it is
not absolute state of health.

Recovering absolute SOH for those four cells needs full-discharge segment
detection — contiguous negative-current runs terminating at the voltage
cutoff — which is precisely what `telemetry/cycles.py` already does for CAN
logs. Until then, three of the five CALCE cohorts rest on a relative target,
and this ADR's numbers should be read with that attached.

## Addendum: adding CX2 changes the answer, and that is the finding

The table above used CS2 alone — 11 cells, 5 cohorts, one cell family. Adding
CALCE CX2 gives **19 cells across 8 cohorts and two cell families** (1.1 Ah
CS2 prismatic and 1.35 Ah CX2), 43,832 rows, ceiling **0.870**.

Cohort labels are namespaced `CS2_Type1` … `CX2_Type6`: both families number
their experiment types 1–6, and a bare `Type1` would merge four CS2 cells with
four CX2 cells. Holding out that merged cohort would leave half its members in
training — flattering LOCO without it noticing.

| Method | LOBO MAE | LOCO MAE | LOBO R² | LOCO R² | Δ |
|---|---:|---:|---:|---:|---:|
| `random_forest` | 3.04% | **8.59%** | 0.938 | **0.693** | −0.245 |
| `hist_gradient_boosting` | 3.22% | 10.00% | 0.922 | 0.642 | −0.280 |
| `gpr_matern` | **2.93%** | 10.06% | 0.911 | 0.617 | −0.294 |
| `xgboost` | 3.38% | 16.07% | 0.912 | 0.534 | −0.378 |
| `lstm` | **2.26%** | 15.68% | **0.971** | 0.526 | −0.445 |
| `mlp` | 2.74% | 15.88% | 0.940 | 0.507 | −0.433 |
| `svr_rbf` | 5.59% | 22.37% | 0.878 | 0.301 | −0.577 |
| `age_isotonic` | 9.79% | 12.07% | 0.339 | 0.198 | −0.141 |
| `age_quadratic` | 14.02% | 14.97% | 0.143 | 0.017 | −0.125 |
| `age_linear` | 14.52% | 15.43% | 0.096 | 0.016 | −0.080 |
| `elasticnet` | 14.87% | 15.39% | 0.131 | 0.000 | −0.130 |
| `train_mean` | 17.53% | 18.16% | 0.000 | 0.000 | 0.000 |

### Three earlier claims die here

**"The LSTM is the worst transferring method on both datasets."** Withdrawn.
On CS2 alone it scored LOCO −4.669; here it has the **best LOBO of any method
(0.971)** and a healthy positive LOCO of 0.526. The catastrophic figure was an
artifact of thin cohort coverage, not a property of sequence models.

**"Behaviour adds about 0.05 R² over counting cycles."** NASA-specific. Here
`age_linear` reaches LOCO 0.016 while `random_forest` reaches 0.693. The
learned models add enormously, because CALCE's electrical features carry real
information that cycle count does not.

**"LOBO rank carries no information about LOCO rank."** Not general.
ρ(LOBO, LOCO) is +0.084 on NASA and **+0.818 (p = 0.001)** here. Choosing by
LOBO costs 0.168 R² rather than 0.309.

### What actually generalises: the gap depends on cohort coverage

Three runs, same harness, same code:

| Frame | Cells | Cohorts | Families | ρ(LOBO,LOCO) | Worst Δ |
|---|---:|---:|---:|---:|---:|
| NASA | 31 | 9 | 1 | +0.084 | −1.354 |
| CALCE CS2 | 11 | 5 | 1 | — | −5.574 |
| CALCE CS2+CX2 | 19 | 8 | 2 | +0.818 | −0.577 |

**Leave-one-cohort-out gets easier as cohort coverage grows.** When
`CX2_Type1` is held out, `CX2_Type2`, `CX2_Type6` and all six CS2 cohorts
remain in training — the "unseen protocol" is barely unseen. With five
cohorts in one family, holding one out removes a fifth of the design space and
the collapse is severe.

So the size of the LOBO-to-LOCO gap is **not a fixed property of a method**.
It is a property of how much of the experimental design space survives in
training. Any paper reporting a LOCO result must report its cohort coverage
alongside, or the number is uninterpretable — and this project has now
produced three mutually inconsistent method rankings by varying nothing but
that.

### What survives all three runs

**Every method loses skill under protocol shift, every time.** Deltas are
negative in all 36 method-frame combinations, ranging −0.08 to −5.57. Not one
method transferred without loss on any frame. That is the claim the evidence
supports, and it is the only one that has survived every revision.

### The `<3%` target is met — on the easy split only

`lstm` reaches **LOBO MAE 2.26%** and `gpr_matern` **2.93%**, the first
sub-3% figures in this project. Both are leave-one-*cell*-out on the
best-conditioned frame available.

Under leave-one-cohort-out the same models score 15.68% and 10.06%; the best
is `random_forest` at 8.59%. So the target is reachable exactly where the
project has spent two ADRs arguing the measurement is optimistic, and is off
by roughly 3× where it is not.

### Cells excluded, and why

23 loaded, 19 admitted. Four CADEX cells were refused at load
(CS2_8, CS2_21, CX2_31, and **CX2_4** — the only cell with a within-protocol
thermal axis, which is a real loss for the Arrhenius line). Of the loaded
cells the screen excluded CS2_3 and CS2_9 (Type 3, multi-discharge cycle
index), CX2_8 (27.8% median step), and **CX2_3** — 91,279 rows whose 95th
percentile capacity is 0.003 of the cell's maximum, SOH pinned at 1.000 with
an interquartile range of 0.0001. That last one required a fifth screen
criterion; it supplied 69% of the admissible rows before exclusion.

## References

- `reports/metrics/calce/benchmark_results.csv` — the table above
- `reports/metrics/calce_cycle_level.csv` — the cached cycle-level frame
- `scripts/build_calce_frame.py`, `scripts/run_benchmark_study.py`
- ADR 0007 (target definition), ADR 0008 (withdrawn rankings)
