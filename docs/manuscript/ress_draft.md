# Leave-one-cell-out cross-validation overstates deployable skill in battery prognostics — and the protocol-shift estimate that would correct it does not resolve method differences

**Target venue:** *Reliability Engineering & System Safety*
**Status:** DRAFT — not submittable. See "Blocking gaps" below.
**Author:** Naveen Vaidyanathan
**Draft date:** 2026-08-26

---

> ## Blocking gaps — read before circulating
>
> 1. **The §2 survey is 3 papers into a declared sample of 40.** The protocol,
>    extraction schema and screening gate are complete and reproducible
>    (`scripts/render_survey_table.py --check`, which exits non-zero and says
>    why). Three references are verified; two were read in full, one from its
>    abstract only and is marked as such. Every remaining `[CITE]` marker is a
>    claim needing a real reference. **None has been invented, and none may be
>    filled in from memory** — this is the premise of the paper, so a fabricated
>    row here would discredit the whole argument.
> 2. **Numbers are current as of ADR 0013.** Every figure traces to a tracked
>    artifact under `reports/metrics/`; the Appendix maps each one. Do not
>    re-quote ADR 0009's tables — its feature set was not recorded and its
>    orderings do not reproduce (§6.4).
> 3. **§5.2's headline figures pool a derivation §5.4 refutes.** This is now
>    stated and quantified in §5.2.1 rather than left for a reviewer to find,
>    and the limitation is recorded in §7. The ratio survives the correction;
>    the magnitudes do not.
> 4. **Author list, affiliation, funding, CRediT and conflict-of-interest are
>    unwritten** (§ Declarations, below).
> 5. **No figures are drawn.** Candidates listed in Appendix C.

---

## Abstract

Data-driven state-of-health (SOH) estimation for lithium-ion cells is
conventionally validated by holding out cells — leave-one-cell-out (LOBO). We
show on two public datasets that this systematically overstates the skill a
model retains when the operating protocol changes, and that the obvious
correction is itself unreliable. Across five evaluation frames spanning two
laboratories and two distinct cohort axes, **every method carrying non-zero
skill loses it under leave-one-cohort-out (LOCO) relative to LOBO, without a
single exception**. We then
test whether LOCO can be used to select among methods. Resampling cohort subsets
at fixed coverage (220 measurements, cell count regressed out), the
interquartile range of the LOCO estimate across draws exceeds the median paired
difference between a nonlinear model and a straight line on cycle count by a
factor of **2.7**. We replicate this ratio on two different derivations of the
same CALCE cells — including one derivation this paper's own §5.4 shows to be a
measurement artifact — and obtain 2.66 and 2.70 respectively, while the
underlying magnitudes differ roughly threefold (IQR 0.892 vs 0.296 R²). The
noise-to-effect ratio is therefore a property of the estimator; the magnitudes
are properties of a particular frame, and we report both rather than the larger.
We report six ranking claims that we ourselves made from single-frame LOCO runs
and subsequently withdrew, and identify the estimator's variance as their common
cause. We further show that two upstream choices
usually left implicit — the noise ceiling of the target, and how the target is
derived from raw cycler records — can change conclusions more than the choice of
model: on a corrected absolute-SOH target, cycle count alone outperforms every
learned method but one. We recommend reporting the target's attainable-R² ceiling,
reporting LOCO as an interval over cohort draws rather than a point estimate, and
reporting worst-group conformal coverage rather than the aggregate.

**Keywords:** battery prognostics; state of health; cross-validation; covariate
shift; conformal prediction; validation methodology

---

## 1. Introduction

Lithium-ion state-of-health estimation is a mature applied machine-learning
problem with a large literature and a small number of shared benchmark datasets.
A typical contribution proposes a feature set or an architecture, evaluates it
by holding out cells from one of those datasets, and reports an R² or a mean
absolute error `[CITE — survey needed, see §2]`.

The deployment question is different from the question that evaluation answers.
A model that will run in a vehicle or a grid installation must generalise to
duty cycles the laboratory did not run: different depths of discharge,
different rates, different ambient conditions. Cells within one dataset cohort
share a protocol, so holding out a cell measures interpolation within a
protocol, not transfer across protocols.

This paper is a validation audit rather than a new estimator. We ask three
questions:

**Q1.** How much does leave-one-cell-out overstate skill relative to a
protocol-level split, and does the answer replicate across laboratories?

**Q2.** Can the protocol-level split be used in its place — specifically, can a
single leave-one-cohort-out estimate distinguish between candidate methods?

**Q3.** What upstream choices, made before any model is fitted, change the
answer by more than the choice of model?

Our answers are, respectively: consistently and substantially; **no**; and at
least two — the noise ceiling of the target, and the derivation of the target
from raw cycler records.

### 1.1 Contributions

1. A replication of the LOBO→LOCO collapse across two dataset families, two
   laboratories, and two distinct cohort axes (temperature; depth of discharge
   and rate), across five evaluation frames with no exceptions (§5.1).
2. A designed experiment showing the LOCO estimate's between-draw spread exceeds
   the between-method differences it is used to measure by a factor of 2.7
   (§5.2), replicated across two derivations of the same cells that differ
   threefold in magnitude (§5.2.1), with a partial-correlation control that
   refuted our own prior explanation on both (§5.2.2).
3. A signal-to-noise estimator giving a per-target ceiling on attainable R²,
   and evidence that a widely used target is 96% measurement noise (§5.3).
4. Evidence that target derivation from raw records changes method ordering more
   than model choice does, including a case where cycle count alone beats every
   learned method but one (§5.4).
5. A worst-group conformal coverage analysis showing aggregate coverage
   certifies a system that fails on the hardest cohorts (§5.5).
6. Six ranking claims made and withdrawn during this work, reported as evidence
   for the paper's thesis rather than omitted (§6.1).

### 1.2 What this paper does not claim

We do not propose a better SOH estimator, and we deliberately do not recommend
one of the benchmarked methods. §5.2 is precisely the argument that our
evaluation cannot support such a recommendation, and §6.1 records what happened
each time we tried.

---

## 2. Related work

> **Completion status.** The survey protocol (§2.1), the extraction schema and
> the screening infrastructure are complete and reproducible. The extraction
> table is **partially populated**: rows marked `verified` in
> `docs/manuscript/survey/extraction.csv` were read against the paper's own
> methods section; the rest are identified but not extracted. The proportions
> in §2.2 are computed from that file by `scripts/render_survey_table.py` and
> will change as it fills. **Do not submit while any row is unverified**, and do
> not quote a proportion from a partial table.

The premise of this paper — that holding out cells is the field's de facto
validation standard, and that what it measures is not what deployment requires —
is an empirical claim about the literature. We therefore establish it by
systematic survey rather than assertion.

### 2.1 Survey protocol

We follow PRISMA 2020 conventions for the identification and screening stages.

**Sources.** Scopus, IEEE Xplore, and arXiv (cs.LG, eess.SY), 2019–2026. The
lower bound is 2019 because Severson et al. established the benchmark dataset
and evaluation framing that much subsequent work adopts.

**Search string**, applied to title, abstract and keywords:

```
("state of health" OR "SOH" OR "remaining useful life" OR "RUL")
AND ("lithium-ion" OR "li-ion" OR "battery" OR "cell")
AND ("machine learning" OR "deep learning" OR "neural" OR "data-driven"
     OR "regression" OR "gaussian process")
```

**Inclusion.** Peer-reviewed or preprint; proposes or evaluates a data-driven
SOH or RUL estimator; evaluates on at least one public cycling dataset; reports
a quantitative held-out figure.

**Exclusion.** Electrochemical or equivalent-circuit modelling with no learned
component; evaluation exclusively on proprietary data, which cannot be checked;
review articles, which are recorded in §2.5 and excluded from the split-type
proportions because a review reports others' protocols rather than adopting one.

**Sampling.** Records are ordered by citation count within year and sampled
proportionally across years, so that the well-cited early work — which is least
representative of current routine practice — does not dominate. Target n = 40.

**Extraction schema.** Eight fields per paper, one row in
`docs/manuscript/survey/extraction.csv`:

| Field | Values |
|---|---|
| `dataset` | NASA · CALCE · Oxford · MIT-Stanford · other · proprietary |
| `split_type` | random · k-fold · leave-one-cell-out · leave-one-protocol-out · temporal · cross-dataset · unclear |
| `split_stated` | explicit · inferable · unclear |
| `ceiling_reported` | yes · no |
| `target_derivation` | specified · partial · unspecified |
| `uncertainty` | none · aggregate · per-group · conformal |
| `n_cells` | integer |
| `verified` | full · abstract · pending |

`split_stated` is deliberately separate from `split_type`. A paper whose split
cannot be determined from its text is itself a finding, and folding "unclear"
into a split category would conceal it.

### 2.2 Validation practice: what is held out

*Proportions pending completion of the table. The claims below are those the
verified subset already supports.*

Three positions appear, and they form a ladder: each rung holds out a larger
unit than the last, and each exposes inflation the rung below it concealed.

**Random or k-fold splits over cycles.** The weakest, because cycles from one
cell fall on both sides of the split. Maher and Yerken (2025) quantify this
directly: across seven models, random splitting reaches R² = 0.9999 while
cell-wise `GroupKFold` yields R² ≈ 0.91 on the same data. They characterise the
difference as optimistic bias and recommend group-wise validation as standard.

**Cell-wise splits (LOBO).** The position Maher and Yerken argue for, and this
paper's *starting* point rather than its conclusion. Le and Nguyen (2026)
report a 119% performance gap between 5-fold cross-validation and LOBO on the
NASA dataset, independently corroborating the direction and rough scale of
§5.1's first step.

**Protocol-wise splits (LOCO).** Rare, and the gap this paper addresses.

### 2.3 Positioning against the closest prior work

Maher and Yerken (2025) is the nearest neighbour and the paper must be read
against it. The distinction is one rung of the ladder above, and stating it
precisely matters because a reviewer will ask.

| | Maher & Yerken (2025) | This paper |
|---|---|---|
| Comparison | Random → cell-wise split | Cell-wise → **protocol-wise** split |
| Finding | Random splitting is optimistically biased | Cell-wise splitting is *also* biased, for a different reason |
| Remedy | Adopt group-wise CV | **The remedy does not work**: the protocol-level estimate is too noisy to rank methods (§5.2) |
| Target treatment | Not analysed | Noise ceiling (§5.3) and derivation (§5.4) dominate model choice |

The two results are consecutive rather than competing: they show that pooling
cycles within a cell inflates skill; we show that pooling protocols within a
cohort inflates it again. The contribution here is the second step *and* the
negative result about the correction. A paper stopping at "use LOCO instead"
would be recommending an estimator whose interquartile range exceeds the
differences it is used to adjudicate.

### 2.4 Feature families for SOH

Severson et al. (2019) established the dominant framing: features from the
discharge voltage curve — specifically the variance of ΔQ(V) between early
cycles — predict cycle life before capacity degradation is measurable, at 9.1%
test error from the first 100 cycles across 124 LFP/graphite cells. Their
dataset and feature construction anchor a large literature on curve-based
features, incremental capacity analysis (ICA) and differential voltage analysis
(DVA) among them. `[CITE — ICA/DVA representatives, pending extraction]`

The behavioural and usage-aggregate family this project began with — mean
temperature, C-rate exposure, depth-of-discharge statistics — is less
represented, and is not the subject of this paper's positive claims.
`[CITE — pending extraction]`

### 2.5 Reviews consulted

Recorded separately and excluded from the split-type proportions.
`[CITE — pending extraction]`

---

---

## 3. Methods

### 3.1 Splits

Let a *cell* be one physical cell and a *cohort* be the set of cells sharing an
experimental protocol. We compare:

- **LOBO** (leave-one-cell-out): hold out one cell; train on all others,
  including cells from the same cohort.
- **LOCO** (leave-one-cohort-out): hold out every cell of one cohort.

LOBO is the field's common practice. LOCO is the protocol-shift analogue. Both
are implemented in one validator (`src/bms/adaptive/validation.py`) so the two
differ only in the grouping column.

**Degenerate-cohort control.** A cohort containing a single cell makes LOCO
identical to LOBO by construction, and the measured gap is then exactly zero.
An early version of our sweep produced precisely that artifact. Single-cell
cohorts are excluded from all LOCO analyses.

### 3.2 Baselines that must be beaten

Two baselines are included because a large reported R² against a monotone
target can be obtained without learning anything about degradation:

- `train_mean` — predicts the training mean; defines R² = 0.
- `age_linear`, `age_quadratic`, `age_isotonic` — regress the target on cycle
  index alone. Any method that does not beat these has not shown that its
  features carry information beyond counting cycles.

The age baselines are load-bearing in §5.4.

### 3.3 Target noise ceiling

For each cell we decompose the within-cell variance of a target into a monotone
trend component and a residual, using isotonic regression on cycle index. The
trend fraction,

    signal_fraction = Var(trend) / (Var(trend) + Var(residual)),

is an estimate of the largest R² any predictor can attain against that target,
under the assumption that the non-monotone residual is unpredictable. We report
it alongside every R² (`src/bms/benchmarks/targets.py`).

This assumption is conservative in one direction and not the other: genuinely
non-monotone degradation (e.g. capacity recovery after rest) is scored as noise,
so the ceiling is a lower bound on attainable skill for such targets. §7 returns
to this.

### 3.4 Cell admissibility screen

SOH is defined relative to a fresh-capacity reference, and that reference is not
always well defined for a given cell. We screen cells on five criteria before
any model is fitted: a minimum cycle count; a bound on the fraction of the
series exceeding physically plausible SOH; a bound on the median step between
consecutive cycles; a floor on median SOH across life; and a floor on the
reference as a fraction of the cell's own maximum. Each criterion was added in
response to a specific observed pathology, documented in the Appendix.

The screen is reported, not silent: excluded cells and reasons are emitted as a
tracked artifact.

### 3.5 Target derivation from raw cycler records

Arbin exports index rows by a `Cycle_Index` counter. Taking the within-index
span of discharge capacity as the cycle's capacity is correct **only when one
index contains exactly one full discharge**. For protocols that cycle partially
by design, or that switch discharge rate within an index, it is not.

We therefore also derive the target by segmenting sample-level telemetry into
contiguous discharge runs and grading each run as full or partial. A run is full
if it terminates at the cell's own voltage cutoff **and** moves charge
comparable to that cell's most capable single discharge. Both criteria are
necessary; §5.4.1 shows what fails when either is dropped.

§5.4 compares the two derivations directly.

### 3.6 Conformal coverage

We use split conformal prediction to produce nominal 90% intervals, and evaluate
coverage **per held-out group** rather than in aggregate, reporting the
worst-group coverage and the fraction of groups falling below nominal
(`src/bms/uncertainty/`).

### 3.7 Cohort-coverage sweep

To test whether a single LOCO estimate can rank methods, we hold method, target,
harness and features fixed and vary only which cohorts are available, sampling
12 random cohort subsets at each coverage level, on each dataset separately.
Cell count is recorded per subframe and regressed out by partial correlation,
because cohort count and cell count rise together. 220 measurements
(`src/bms/benchmarks/coverage.py`).

### 3.8 Reproducibility

Every number in §5 is produced by a tracked script and written to a tracked
artifact under `reports/metrics/`. The Appendix maps each reported figure to its
generating command and output file.

---

## 4. Datasets

| Dataset | Cells | Cohort axis | Temperature | SOC | Reference |
|---|---:|---|---|---|---|
| NASA PCoE | 31 admitted | Ambient temperature, load | yes | yes | NASA Ames Prognostics Center of Excellence battery data repository |
| CALCE CS2 / CX2 | 22 admitted | Depth of discharge, discharge rate, cutoff voltage | **no** | **no** | CALCE Battery Group, University of Maryland, doi:10.21227/w9rg-7173 |

The two are **not feature-comparable**. NASA carries temperature and
state-of-charge channels and supports behavioural features; CALCE records
neither, so its usable features are electrical (voltage, current, internal
resistance, cycle duration) plus cycle index. We therefore never pool them, and
we compare the *phenomenon* across them rather than model performance.

The cohort axes differ deliberately. NASA's cohorts vary ambient temperature;
CALCE's vary depth of discharge and rate. A collapse appearing on both is a
property of protocol shift rather than of temperature.

**Datasets screened out.** Oxford (Birkl, Oxford Battery Degradation Dataset 1,
University of Oxford Research Archive, 2017) and the Severson et al. dataset
(Severson et al., *Nature Energy* 4, 383–391, 2019) were assessed by a
pre-registered commensurability screen and found MARGINAL for this comparison —
they share only internal resistance as a usable axis with the frames above. The
screen is run before download and is reported in full (Appendix).

---

## 5. Results

### 5.1 The collapse replicates without exception

Five evaluation frames have been run: NASA, CALCE CS2 alone, and CALCE CS2+CX2
in earlier analyses (36 method-frame combinations), plus the `Cycle_Index` and
full-discharge derivations of CALCE CS2+CX2 in §5.4 (12 methods each).

The `train_mean` baseline is excluded from the claim below: its R² is 0 under
both splits by construction, so it cannot lose skill. **Every remaining
method-frame combination shows R² falling from LOBO to LOCO, with no
exception.** Deltas range from −0.08 to −5.57 R². In the 220-measurement sweep
of §5.2 the median gap is negative at every coverage level on both datasets.

> *Draft note:* state the exact combination count here once the three earlier
> frames' method lists are re-tabulated from
> `reports/metrics/*/benchmark_results.csv`. The documented figure of 36 is from
> ADR 0010 and includes degenerate baselines; do not publish an exact total that
> has not been recounted from the artifacts.

Six of twelve methods on one CALCE frame move from strongly positive LOBO to
*negative* LOCO — worse than predicting a constant.

This is the only claim in this project that has survived every revision: a
screen revision, a dataset addition, a single-cell exclusion that reversed a
headline, a target re-derivation, and a designed experiment aimed at explaining
it away.

### 5.2 A single LOCO estimate cannot rank methods

At a fixed coverage level, across 24 draws differing only in which cohorts were
selected:

| Quantity | Value |
|---|---:|
| Median within-level IQR of the LOCO gap | **0.724 R²** |
| Median paired difference, `svr_rbf` − `age_linear`, identical subframes | **−0.351 R²** |
| IQR of that paired difference | 0.807 R² |
| Fraction of draws where `svr_rbf` beats `age_linear` | **0.25** |

**The spread of the LOCO estimate between cohort draws is roughly twice the
median difference between a nonlinear model and a straight line on cycle
count.** The measurement noise exceeds the effect being measured.

Note the paired comparison: on identical subframes the nonlinear model *loses*
to the age baseline in three draws out of four. Full-frame runs in which a
nonlinear model "won" were draw-specific outcomes, not stable orderings.

#### 5.2.1 The same measurement on a corrected target

The figures above pool two frames, and one of them is the `Cycle_Index`
derivation that §5.4 of this paper shows to be a measurement artifact. That is a
problem for §5.2 specifically, not merely an inconsistency of presentation: §5.4
argues that this derivation alternates between full and partial discharge
capacities, so it cannot be tracked by a smooth function of cycle number but
*can* be partially fitted by a flexible one. The sweep's two arms are a flexible
model (`svr_rbf`) and a smooth function of cycle number (`age_linear`). The
experiment was therefore measuring exactly the contrast the target distorts.

We re-ran the sweep on the ADR-corrected full-discharge derivation of the same
CALCE cells — same cells, same feature set, same two methods, same sweep
machinery, differing only in how the target is derived from the raw records.
Seven of ten cohorts carry the two cells LOCO requires, spanning 19 cells and
11,515 rows; k = 3…7 with 12 draws each, 98 sweep points. Compared on the three
coverage levels both runs draw fully (k = 3, 4, 5), CALCE against CALCE:

| | `Cycle_Index` target | Full-discharge target |
|---|---:|---:|
| LOCO IQR of the gap | 0.892 | **0.296** |
| Median paired `svr_rbf` − `age_linear` | −0.336 | **−0.110** |
| Draws where `svr_rbf` wins | 13.9% | 38.9% |
| **Noise-to-effect ratio** | **2.66** | **2.70** |

The IQR reduction is 0.596 R², bootstrap 95% CI [0.335, 0.936] over 4,000
resamples of draws within levels; the corrected estimate is smaller in every
resample.

Three conclusions follow, and they are not the same conclusion.

**The claim survives, as a ratio.** The spread of the LOCO estimate is 2.70× the
between-method difference on the corrected target against 2.66× on the artifact
target. The agreement to three significant figures is closer than this design
warrants and should not be over-read, but the qualitative claim — that the
measurement is several times noisier than the effect it is used to detect — is
not an artifact of the target derivation.

**The magnitudes were inflated roughly threefold.** Both the noise and the
effect shrink by a factor near 2.7–3. Any statement quoting 0.724 or 0.351 as a
property of leave-one-cohort-out *in general* overstates it by that factor.
Those are properties of that estimator on that frame, and we state them as such
here and in the abstract.

**The mechanism does not survive, and this is the part that matters.** On the
`Cycle_Index` target the flexible model collapsed hardest: `svr_rbf` fell from a
median LOBO of 0.835 to a LOCO of −0.196, a gap of −1.037, against `age_linear`'s
−0.096. On the corrected target the ordering **inverts** — `age_linear` has the
higher LOBO (0.695 → 0.231, gap −0.460) and `svr_rbf` the smaller collapse
(0.357 → 0.090, gap −0.203). "The flexible method transfers worst" was the
artifact of §5.4 seen from a second direction. We had stated it; we withdraw it,
and it is the fifth entry in §6.1.

This subsection is itself an instance of the paper's thesis. A single frame
produced a mechanism story that was clean, plausible, physically interpretable,
and false — and only a second derivation of the same cells exposed it.

#### 5.2.2 A control that refuted our own hypothesis

We had proposed that the LOBO→LOCO gap is governed by cohort coverage: a
held-out protocol is barely unseen when many others remain in training. This was
to be the paper's thesis.

| Dataset | n | Raw ρ(cohorts, gap) | Adjusted for cell count | ρ(cells, gap) |
|---|---:|---:|---:|---:|
| NASA | 146 | −0.029 (p = 0.73) | +0.007 (p = 0.94) | −0.031 |
| CALCE | 74 | **+0.257 (p = 0.027)** | **+0.024 (p = 0.84)** | +0.272 |
| Pooled | 220 | +0.034 (p = 0.62) | +0.025 (p = 0.71) | +0.028 |

On CALCE the raw correlation is significant and the medians look convincingly
monotone (−0.668, −0.347, −0.206, −0.125 as coverage grows from three cohorts to
six). Presented alone it reads as clean confirmation.

Adjusted for cell count it collapses to +0.024 (p = 0.84), and the gap's
correlation with cell count alone (+0.272) accounts for essentially all of the
raw figure. **The apparent coverage effect is a training-set-size effect.**

Without the partial-correlation control we would have reported ρ = +0.257,
p = 0.027 as support. We report this because it is the same failure mode the
paper is about, occurring in our own hands.

The refutation replicates on the corrected derivation of §5.2.1. On the
full-discharge frame (n = 98) the raw ρ(cohorts, gap) is +0.126 (p = 0.217),
adjusted for cell count −0.002 (p = 0.982), and the gap's correlation with cell
count alone +0.125. The raw relationship is weaker there and was never
significant; in both derivations it vanishes once cell count is regressed out.
The coverage hypothesis is refuted on two independent target derivations of the
same cells.

### 5.3 Targets are often mostly noise, and the ceiling is rarely reported

On the NASA frame, per-cycle capacity delta has an attainable-R² ceiling of
**0.044** — it is 96% measurement noise. Every earlier null result in this
project against that target was competing for a band of 0.04.

Ceilings for the targets used here:

| Frame | Target | Ceiling |
|---|---|---:|
| NASA | per-cycle capacity delta | 0.044 |
| NASA | cumulative fade | 0.612 |
| CALCE CS2+CX2, `Cycle_Index` | SOH | 0.870 |
| CALCE, full-discharge | SOH | **0.907** |

An R² of 0.03 means something very different against a ceiling of 0.04 than
against 0.9. We recommend reporting the ceiling alongside any R² in this domain;
it costs one isotonic fit.

### 5.4 Target derivation changes the ordering more than model choice does

CALCE CS2 Types 5 and 6 cycle partially by design. Under the `Cycle_Index`
derivation their SOH is measured against a **partial-cycle reference** of
0.18–0.43 Ah on 1.1 Ah cells — relative fade of a repeated partial cycle, not
absolute state of health. Segmenting discharges directly recovers physical
references:

| Cell | Protocol | `Cycle_Index` reference | Full-discharge capability | Nominal |
|---|---|---:|---:|---:|
| CS2_5 | Type 5 | 0.177 Ah | **1.055 Ah** | 1.1 |
| CS2_24 | Type 6 | 0.367 Ah | **1.101 Ah** | 1.1 |
| CS2_25 | Type 6 | 0.432 Ah | **1.103 Ah** | 1.1 |
| CS2_33 | Type 1 | 1.120 Ah | 1.143 Ah | 1.1 |

Correcting the derivation also recovers four cells the admissibility screen had
excluded, because those exclusions were consequences of the derivation rather
than properties of the cells:

| | rows | cells | cohorts | admissible | ceiling |
|---|---:|---:|---:|---:|---:|
| `Cycle_Index` | 136,661 | 23 | 11 | 19/23 | 0.870 |
| Full-discharge | 11,529 | 22 | 10 | **22/22** | **0.907** |

The effect on conclusions is larger than any model comparison in this paper.
With identical features and harness, only the derivation changing:

| Method | LOCO, `Cycle_Index` | LOCO, full-discharge |
|---|---:|---:|
| `age_isotonic` | 0.198 | **0.524** |
| `age_quadratic` | 0.017 | **0.515** |
| `age_linear` | 0.016 | **0.478** |
| `random_forest` | 0.665 | 0.670 |
| `xgboost` | **0.697** | 0.408 |
| `hist_gradient_boosting` | 0.675 | 0.293 |
| `mlp` | 0.606 | **−1.461** |
| `gpr_matern` | 0.438 | **−2.087** |

On the corrected target, **cycle count alone reaches LOCO 0.478–0.524 and beats
every learned method except `random_forest`**. The gap between counting cycles
and the best model falls from 0.68 R² to 0.19 R².

The interpretation is mechanical. A target alternating between full and partial
discharge capacities cannot be tracked by a smooth function of cycle number, but
*can* be partially fitted by a flexible model. The flexible models were being
rewarded for fitting a measurement artifact. We had previously concluded from
the left-hand column that learned models add substantially over cycle counting;
that conclusion does not survive the corrected target.

#### 5.4.1 Both detection criteria are necessary

Detecting full discharges as "runs terminating at the voltage cutoff" — the
natural first formulation — is insufficient:

| Cell | Protocol | Runs reaching cutoff | Median charge moved |
|---|---|---:|---:|
| CS2_24 | Type 6 | 53 / 4,990 | 0.288 Ah |
| CS2_5 | Type 5 | 6,951 / 6,962 | 0.237 Ah |

Type 6 partial cycles stop well above cutoff and separate cleanly. Type 5
partial cycles discharge *to* the cutoff from a partially charged state, so
99.8% satisfy a cutoff test while delivering a fifth of the cell's capacity.

Further, the cutoff cannot be estimated as a quantile of terminal voltages: the
partial cycles then define it (3.78 V for CS2_24, against a true 2.70 V), every
partial trivially satisfies it, and the detector certifies exactly the cycles it
exists to reject. We estimate the cutoff from the deepest discharges only.

**Cost.** A Type 5 or 6 cell yields 36–73 full discharges out of 5,000–7,000
cycles. Absolute SOH on partial-cycling protocols is inherently sparse, and we
report the yield rather than scaling partial cycles up to look complete.

### 5.5 Aggregate conformal coverage certifies a system that fails per cohort

Nominal 90% intervals, target `cumulative_fade`:

| Method | Split | Median coverage | Worst-group coverage | Fraction of groups below nominal | Median width |
|---|---|---:|---:|---:|---:|
| `elasticnet` | LOCO | 0.824 | **0.262** | **0.556** | 0.399 |
| `elasticnet` | LOBO | 0.970 | **0.000** | 0.188 | 0.387 |
| `age_linear` | LOCO | 1.000 | 0.541 | 0.111 | 0.505 |
| `age_linear` | LOBO | 1.000 | 0.365 | 0.125 | 1.551 |

Under LOCO, `elasticnet`'s median coverage of 0.824 is not far below nominal,
but **56% of held-out cohorts fall below nominal and the worst reaches 0.262**.
Under LOBO there is a cell (`B0045`) whose 90% interval contained the truth
**zero times** while median coverage read 0.970.

The simpler `age_linear` puts only 11% of cohorts below nominal under LOCO, but
requires intervals 27% wider to do it — an honest width for the information
available.

This is the safety-case form of the paper's argument, and the form a
certification body would care about: a fleet-average coverage statistic
certifies a system that fails specifically on cells in the hardest conditions.

---

## 6. Discussion

### 6.1 Six withdrawn claims, and their common cause

During this work we made and then withdrew six ranking claims, each from a
single-frame LOCO run:

| # | Claim | What varied | How it died |
|---|---|---|---|
| 1 | "High-capacity models transfer worse" | Method set | XGBoost was a counterexample |
| 2 | "Regularisation, not capacity, predicts transfer" | Dataset | Contradicted on CALCE |
| 3 | "Selecting by LOBO picks the worst method" | Cell set | Rested on one cell (B0041) |
| 4 | "The LSTM transfers worst on both datasets" | Cohort coverage | Reversed when CX2 was added |
| 5 | "Learned models add enormously over cycle count" | Target derivation | Reversed by the corrected target (§5.4) |
| 6 | "The flexible method transfers worst" | Target derivation | Ordering inverted on the corrected target (§5.2.1) |

The third column is the paper's argument in one place. **In no case did the
model change.** Each claim was overturned by varying something the field
routinely treats as a fixed preliminary — which methods were compared, which
dataset, which cells were admitted, how many cohorts, and how the target was
derived from the raw records.

§5.2 identifies the common cause. Each was a single draw from a distribution
whose interquartile range is wider than the differences between the methods
being ranked. Nothing was wrong with any individual run; the estimator does not
resolve what it was being asked to resolve.

Claims 5 and 6 deserve separate mention because they share a cause and were
found nine days apart. Both died to the same target correction, in opposite
directions: correcting the derivation removed a large advantage the learned
methods appeared to hold, *and* removed the flexible model's apparent fragility
under shift. The derivation was simultaneously inflating one contrast and
manufacturing another.

We report these rather than omitting them because they are the paper's most
direct evidence, produced under controlled conditions with tracked artifacts.

### 6.2 Recommendations for practice

1. **Report the target's attainable-R² ceiling.** One isotonic fit. Without it,
   a reported R² is uninterpretable.
2. **Specify how the target was derived from raw records**, and verify that the
   reference is physically plausible for the cell. §5.4 shows this can dominate
   model choice.
3. **Split by protocol, not only by cell** — and report LOCO as an interval over
   cohort draws, never as a point estimate.
4. **Do not rank methods on a single LOCO run.** If a ranking is claimed, report
   the paired difference across draws.
5. **Report worst-group coverage**, not aggregate coverage.
6. **Include a cycle-count baseline.** On our corrected target it beats all but
   one learned method.

### 6.3 Two specification targets that cannot be met as written

This project began with an engineering specification requiring SOH error < 3%
and RUL within ±20 cycles. Both are ill-posed:

- **SOH < 3%** is met only under LOBO, on the best-conditioned frame. Under
  protocol shift the best figure is roughly 3× the target. The target is
  reachable precisely where the measurement is optimistic.
- **RUL ±20 cycles** is achievable on lab-accelerated cells and impossible on a
  production pack. NASA cells fade at ≈0.251% SOH/cycle, so ±18 cycles follows
  from 4.5% SOH error. A production pack fading ≈0.013%/cycle would require
  0.27% SOH accuracy — below coulomb-counting drift, typically 1–2%.

"±20 cycles on NASA" and "±20 cycles on a car" are not the same claim. We suggest
RUL be expressed as a percentage of remaining life on production packs.

### 6.4 A reproducibility failure in our own record

An earlier internal report of the CALCE results did not record the feature set
used. Re-running the same harness with an explicit feature set does not
reproduce its orderings — `lstm` scores LOCO 0.259 against a recorded 0.526.
The collapse replicates; the ordering does not. We note this because it is the
same class of problem as the paper's subject, and because §5.4's comparison is
consequently made against a re-run baseline rather than the published one.

---

## 7. Limitations

- **The §5.4 comparison is not a fully controlled experiment.** The two frames
  differ in rows (136,661 vs 11,529), cells (19 vs 22) and cohorts (8 vs 10).
  The admitted cell set changed *because* the derivation changed — which is the
  finding, and also a confound. Attributing the entire difference to derivation
  would overstate what was measured.
- **Absolute SOH on partial-cycling protocols is sparse.** Two cohorts rest on
  36 and 45 observations respectively.
- **The noise-ceiling estimator scores non-monotone degradation as noise.**
  Capacity recovery after rest is real and would be counted against the ceiling,
  making it a lower bound for such targets.
- **The headline sweep figures pool a refuted derivation.** The 0.724 / −0.351 /
  0.25 figures in §5.2 pool NASA with the CALCE `Cycle_Index` frame that §5.4
  shows to be an artifact. §5.2.1 reports the corrected re-run and finds the
  ratio stable at 2.7 while the magnitudes fall roughly threefold, so we retain
  the original figures as a description of that frame and report the corrected
  ones alongside. We do not have a corrected NASA counterpart: NASA's target
  derivation was not in question, so no second derivation of it exists, and the
  pooled figures therefore mix one contested frame with one uncontested one.
- **The sweep uses two methods** (`age_linear`, `svr_rbf`); tree ensembles were
  excluded on runtime. A wider method set might narrow or widen the
  paired-difference distribution.
- **Cells are thinned to at most 200 cycles** in the sweep to bound compute.
  Thinning is uniform in cycle index and applied identically at every coverage
  level, so it cannot bias the trend, but it reduces within-cell resolution.
- **Cohort subsets are sampled, not enumerated**, with 12 draws per level, so
  the interval estimates are themselves uncertain.
- **Two chemistries, two laboratories.** Both datasets are small-format cells
  under laboratory cycling. We make no claim about production packs, and §6.3
  argues explicitly against transferring a cycle-count accuracy target to one.
- **One retained cell (CS2_6) has a capability of 0.747 Ah against 1.1 Ah
  nominal.** It passes every screen but its reference may not be a
  fresh-capacity figure. Unresolved.
- **No temperature axis on CALCE.** Arrhenius parameterisation is not
  identifiable on either frame — on NASA ambient temperature is collinear with
  protocol, and CALCE records no temperature at all.

---

## 8. Conclusion

Leave-one-cell-out cross-validation overstates the skill a battery SOH model
retains under protocol shift, consistently and across laboratories: across five
evaluation frames we found no exception. The natural correction —
holding out whole protocols — does not support the use it would be put to. At
the sample sizes standard datasets provide, the between-draw spread of the
leave-one-cohort-out estimate is roughly twice the between-method differences it
would be used to adjudicate, and on identical subframes a nonlinear model loses
to a straight line on cycle count three times in four.

The practical consequence is not that battery prognostics is impossible, but
that the reporting conventions are insufficient to tell when it has worked. Two
choices made before any model is fitted — the target's noise ceiling and its
derivation from raw records — changed our conclusions more than the choice of
model did, and neither is routinely reported.

---

## References

> **Verified only.** Every entry below has been checked against the publisher's
> or preprint server's own record. Entries are added here *only* when the
> corresponding row in `docs/manuscript/survey/extraction.csv` is marked
> `verified`, so this list and that table cannot drift apart. Items still marked
> `[CITE]` in the text have no entry here yet, deliberately.

Le, H. H., & Nguyen, K.-A. (2026). Charging phase health indicators for battery
state-of-health estimation: A systematic comparison of CC, CV, and combined
approaches under cross-battery validation. *arXiv:2607.23482*.
— *Cited for the 119% 5-fold-vs-LOBO gap. Read from abstract only; confirm in
full text before submission.*

Maher, K., & Yerken, N. (2025). Comprehensive machine learning for lithium-ion
battery state-of-health estimation using group-wise cross-validation. In
*14th International Conference on Renewable Energy Research and Applications
(ICRERA 2025)* (pp. 1465–1468). IEEE.
https://doi.org/10.1109/ICRERA66237.2025.11283788

Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen,
M. H., Aykol, M., Herring, P. K., Fraggedakis, D., Bazant, M. Z., Harris, S. J.,
Chueh, W. C., & Braatz, R. D. (2019). Data-driven prediction of battery cycle
life before capacity degradation. *Nature Energy*, 4(5), 383–391.
https://doi.org/10.1038/s41560-019-0356-8

---

## Appendix A — Reproducibility

Every figure in §5 maps to a tracked artifact.

| Figure | Command | Artifact |
|---|---|---|
| §5.1 collapse, NASA + CALCE | `make study` | `reports/metrics/benchmark_results.csv` |
| §5.1, §5.4 CALCE full-discharge | `make calce-full-discharge`, `make calce-study-full-discharge` | `reports/metrics/calce_full_discharge/benchmark_results.csv` |
| §5.4 controlled baseline | `make calce-study-baseline` | `reports/metrics/calce_baseline_controlled/benchmark_results.csv` |
| §5.2 sweep, 220 measurements | `make coverage-sweep` | `reports/metrics/coverage_sweep.csv`, `coverage_sweep_summary.csv` |
| §5.3 noise ceilings | `make study` | `reports/metrics/benchmark_signal_report.csv` |
| §5.4 per-cell yield | `make calce-full-discharge` | `reports/metrics/calce_full_discharge_yield.csv` |
| §5.5 conformal coverage | `make coverage-study` | `reports/metrics/coverage_summary.csv`, `coverage_summary.md` |
| §3.4 screen decisions | `make study` | `reports/metrics/benchmark_cell_screen.csv` |

Decision records ADR 0001–0012 hold the reasoning, including superseded and
withdrawn claims, and are the audit trail for §6.1.

## Appendix B — Screen criteria and their provenance

| Criterion | Threshold | Pathology that motivated it |
|---|---|---|
| Minimum cycles | 10 | Cells with too few points for a trend |
| Overshoot fraction | ≤10% above SOH 1.05 | Series mixing measurement types |
| Median step | ≤5% of reference | Alternating full and partial discharges |
| Median SOH across life | ≥0.40 | Reference physically impossible (5.4 Ah on a 1.1 Ah cell) |
| Reference / maximum | ≥0.02 | A cell whose rows carry almost no discharge; 69% of the training set from one degenerate cell |
| Capability / max discharge | ≥0.20 | Detector returning a 0.04 Ah reference on a 1.35 Ah cell, which every downstream screen passes |

## Declarations

**Author contributions (CRediT).** *To be completed before submission.* The
roles applicable to this work are: Conceptualization, Methodology, Software,
Formal analysis, Data curation, Writing – original draft, Writing – review &
editing, Visualization.

**Funding.** *To be stated.* If none: "This research received no specific grant
from any funding agency in the public, commercial, or not-for-profit sectors."

**Declaration of competing interest.** *To be stated.*

**Data availability.** This work uses only publicly available datasets and
introduces no new experimental data. The NASA PCoE battery aging dataset and the
CALCE CS2/CX2 cycling datasets are available from their respective repositories;
accession details and the exact file manifests used are given in Appendix A. All
derived artifacts quoted in this paper — per-fold metrics, sweep outputs, signal
reports — are tracked in the accompanying code repository under
`reports/metrics/`, and each figure quoted in the text is bound to its source
artifact by an automated test (`tests/test_reported_numbers.py`), so a quoted
number that no longer matches its artifact fails the build rather than
persisting in the prose.

**Code availability.** *Repository URL and archived DOI to be added at
submission.* The analysis is reproducible with the commands in Appendix A.

---

## Appendix C — To do before submission

**Blocking**

1. **§2 survey: 37 of 40 papers remain.** Protocol, schema and gate are done.
   Run `python scripts/render_survey_table.py --check`; it exits non-zero and
   names what is missing. Extract each paper against its own methods section —
   `le2026` is currently abstract-only and its 119% figure must be confirmed in
   the full text before it is cited.
2. Resolve every remaining `[CITE]` marker with a verified reference.
3. Complete the Declarations section above.
4. **Figures: none drawn.** Candidates, in order of argumentative weight:
   (a) the LOCO gap distribution across draws, both derivations overlaid, which
   is §5.2's whole argument in one panel; (b) per-cohort conformal coverage
   against nominal (§5.5); (c) the two target derivations for one CS2 cell
   (§5.4).

**Non-blocking but strengthening**

5. A controlled re-run of §5.4 holding the cell set fixed, to separate
   derivation from admissibility. §7 currently concedes this confound; removing
   it would convert the paper's fourth contribution from suggestive to
   demonstrated. This is the single highest-value remaining experiment.
6. Gate model promotion on an interval over cohort draws rather than a point
   estimate — the practical recommendation §6.2 makes, implemented in the
   accompanying software. Unblocked by §5.2.1, which measures the distribution
   on a defensible target.
7. Confirm RESS formatting, length limits, and structured-abstract requirements.
8. Resolve CS2_6 (§7).
