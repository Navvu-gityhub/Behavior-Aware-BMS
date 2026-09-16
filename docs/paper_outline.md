# Publication outline

An honest assessment of what in this project is publishable, where, and what
is still missing. Written to be argued with rather than followed.

## The claim worth making

> Leave-one-cell-out cross-validation, the de facto standard in data-driven
> battery prognostics, systematically overstates out-of-sample skill. But the
> leave-one-protocol-out estimate that would correct it is, at the sample
> sizes standard datasets provide, **noisier than the differences it is used
> to measure** — so a single LOCO number cannot rank candidate methods, and
> must be reported as an interval over cohort draws.

This is a methodological audit, not a new predictor. That framing is a
strength, not a fallback: there are hundreds of papers proposing another
SOH regressor and very few testing whether the field's validation protocol
supports the claims made on it.

**The paper's own strongest evidence is its history.** Five ranking claims
were made from single-frame LOCO runs during this work and all five were
withdrawn — one reversed by excluding a single pathological cell, one by
adding a second cell family, one by a control added to test the authors' own
hypothesis. That is not an embarrassment to hide; it is the phenomenon, and
it is reproducible from the tracked artifacts.

Supporting results below give the paper substance beyond the headline.

### Result 1 — the target is often mostly noise, and nobody reports it

`benchmarks/targets.signal_to_noise` decomposes a target's within-cell
variance into monotone trend and residual, yielding a ceiling on attainable
R². On this project's NASA frame, per-cycle capacity delta has a ceiling of
**0.044**. Papers reporting R² against similar targets are competing for a
band they do not disclose.

*Why it is publishable:* it is a general, cheap diagnostic any prognostics
paper could run, and it reframes an entire class of null results. The claim
"our model achieves R² = 0.03" means something very different against a
ceiling of 0.04 than against 0.9.

### Result 2 — measured capacity carries a reversible thermal offset that inverts the temperature signal

At cycle ≤ 20, apparent SOH is 0.918 at 4 °C and 0.999 at 44 °C. This is
kinetic, not degradation, and it is *anti*-correlated with true thermal
aging. It mechanically explains a result this project had documented but not
understood: temperature predicts fade with the correct sign within cohorts and
the wrong sign across them.

*Why it is publishable:* the NASA dataset is one of the two most-used
benchmarks in the field. A demonstration that a naive SOH target built from it
inverts the temperature relationship has direct consequences for a large
existing literature.

### Result 3 — conformal coverage fails per-cohort while the aggregate looks fine

Nominal 90% intervals, ElasticNet, leave-one-cohort-out: median coverage 0.824
but **56% of held-out cohorts fall below nominal**, worst 0.262. Under LOBO
there is a cell whose 90% interval contained the truth zero times. A simpler
model (`age_linear`) puts only 11% of cohorts below nominal but needs intervals
27% wider to do it.

*Why it is publishable:* this is the safety-case version of the headline, and
it is the form a certification body would care about. A fleet-average coverage
statistic certifies a system that fails specifically on cells in the hardest
conditions.

## What is still missing, in order of how badly

### 1. Replication on a second dataset family — **done** (ADR 0009)

This section previously read "blocking"; it was not updated when the
replication landed, and said so for some time after it was false. Corrected
here.

CALCE CS2+CX2 was loaded and run: 19 cells, 8 cohorts, two cell families,
43,832 rows, noise ceiling 0.870. **The LOBO-to-LOCO collapse reproduces** on a
different laboratory's data with cohort structure on depth of discharge and
discharge rate rather than NASA's temperature. Deltas are negative in all 36
method-frame combinations across the three frames run.

What did *not* replicate is any method ranking, and ADR 0009 makes that the
finding rather than an embarrassment: three mutually inconsistent rankings were
produced by varying nothing but cohort coverage.

Oxford and Severson remain MARGINAL as transfer targets (internal resistance
only) and are not worth the download for this claim.

**Caveat attached to those numbers, and now addressed** — see item 1b.

### 1b. Absolute SOH on the partial-cycling protocols — **done** (ADR 0012)

The rebuilt study is complete. Headline for the paper: on an absolute SOH
target, **counting cycles reaches LOCO R² 0.478–0.524 and beats every learned
method except `random_forest`** (0.670). Against the partial-cycle target the
same baselines scored 0.016–0.198, which is where ADR 0009's "the learned models
add enormously" came from. That claim is withdrawn — the advantage was largely
the flexible models fitting a measurement artifact.

This is a fifth withdrawn ranking claim, and it belongs in the paper's history
section as one: the project has now produced four incompatible method rankings
by varying only target derivation, cohort coverage and feature set.

Admissibility 19/23 → 22/22 cells, cohort coverage 8 → 10, ceiling 0.870 →
0.907. See ADR 0012 for the limitations that bound the comparison.

#### Original entry, retained for context

ADR 0009's own open section records that CS2 Types 5 and 6 cycle partially by
design, so their SOH was measured against a **partial-cycle reference** of
roughly 0.26–0.42 Ah on a 1.1 Ah cell. That is relative fade of a repeated
partial cycle, not absolute state of health, and it applies to three of the
eight admitted cohorts.

`src/bms/io/calce_full_discharge.py` recovers an absolute reference by
segmenting sample-level telemetry into contiguous discharge runs and grading
each against the cell's own voltage cutoff *and* charge capability. Measured
per-cell references move from 0.177/0.367 Ah (partial) to 1.055/1.101 Ah, which
is the 1.1 Ah nominal capacity.

It costs most of the rows: a Type 5 or 6 cell yields tens of full discharges out
of thousands of cycles. That is a property of the protocol, and the paper should
report the yield rather than hide it.

Until the rebuilt study is complete, ADR 0009's tables stand with the
partial-reference caveat attached.

### 2. Curve-based methods — strongly wanted

Severson's ΔQ(V) variance and ICA/DVA are the two feature families a reviewer
will ask about. Both are implemented and tested; both report UNAVAILABLE on
cycle-level aggregates. CALCE's Arbin exports carry the traces.

### 3. Statistical treatment of the headline gap — **done**

`src/bms/benchmarks/coverage.py` + `scripts/run_coverage_sweep.py`. 220
measurements resampling cohort subsets, with cell count regressed out. This
became the paper's central result rather than a supporting one (ADR 0010):
IQR of the gap across draws is 0.724 R² against a median between-method
difference of 0.351.

It also refuted the authors' own prior hypothesis — that the gap reflects
cohort coverage — which is worth reporting in the paper as a worked example of
why the interval is needed.

### 4. A positive result — optional but valuable

The paper is stronger with one. The Arrhenius parameterisation was the
intended candidate and **failed** on NASA (ADR 0007) — not because the model
is wrong but because Ea is not identifiable there. CX2_4, cycled at
25/35/45/55 °C, is the one available frame where the identifiability gate can
pass. It is n=1, which characterises the relationship without supporting a
generalisation claim, and the paper must say so.

Publishing without a positive result is viable for a methods-and-audit paper;
it just needs to be framed as such from the abstract onward.

## Venues

| Venue | Fit | Note |
|---|---|---|
| *Reliability Engineering & System Safety* | **Best fit** | Publishes validation-methodology and negative results; the conformal-coverage result is squarely in scope |
| *Journal of Power Sources* | Good | High visibility in the field; will want the CALCE replication and probably the curve-based methods |
| *Journal of Energy Storage* | Good | Broader scope, faster review, somewhat lower bar |
| *Applied Energy* | Possible | Prefers a demonstrated application or system-level impact |
| *eTransportation* | Possible | Would want the EV-behaviour framing foregrounded |
| *Nature Energy / Joule* | No | Needs a new capability, not an audit |

Recommendation: target **RESS**, with *Journal of Power Sources* as the
fallback. RESS reviewers are the audience most likely to see the validation
argument as the contribution rather than as an absence of one.

## Suggested structure

1. **Introduction** — validation practice in battery prognostics; the gap
   between reported and deployable skill.
2. **Related work** — how existing SOH/RUL papers split their data. This
   section needs an actual survey: sample ~40 recent papers and tabulate the
   split used. That survey does not yet exist and is a day of work; it is what
   converts "the field does this" from an assertion into evidence.
3. **Methods** — the harness; LOBO vs LOCO; training-mean and age-confound
   baselines; the noise-ceiling estimator; conformal coverage.
4. **Datasets** — NASA and CALCE, with the commensurability screen as a
   principled pre-registration of which comparisons are admissible.
5. **Results** — the three results above, on both datasets.
6. **Discussion** — what should change in practice: report the target ceiling,
   split by protocol, report worst-group coverage.
7. **Limitations** — chemistry coverage, cell counts, the CALCE thermal axis.

## What to reuse directly

| Paper element | Source |
|---|---|
| Methods §3 | `src/bms/adaptive/validation.py` docstring |
| Noise ceiling | `src/bms/benchmarks/targets.py` docstring, ADR 0007 |
| Thermal confound | `src/bms/physics/thermal_confound.py` docstring |
| Conformal | `src/bms/uncertainty/conformal.py` docstring |
| Dataset admissibility | `src/bms/adaptive/dataset_specs.py`, ADR 0006 |
| Reproducibility appendix | `docs/final_report.md` Appendix |

## Folding back into `final_report.md`

`docs/final_report.md` remains the project's authoritative internal account
and currently predates ADR 0007. It needs:

- Abstract updated: the null results were against a target with a 0.044
  ceiling.
- New Section 4.10 (noise ceiling) and 4.11 (thermal confound).
- Section 4.2's temperature finding annotated with the confound explanation.
- Appendix extended with `run_benchmark_study.py` and `run_coverage_study.py`.

Until that is done, ADR 0007 is the authoritative statement of these results
and the report is out of date on them.
