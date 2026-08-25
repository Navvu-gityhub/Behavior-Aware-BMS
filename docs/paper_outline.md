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

### 1. Replication on a second dataset family — blocking

Every result above is currently established on NASA alone. A reviewer will
reject on this ground alone, and would be right to. **CALCE CS2/CX2 is the
required download**; the commensurability screen already confirms it is well
posed on depth of discharge, discharge rate and cutoff voltage, giving cohort
structure on a *different axis* than NASA's temperature. If the LOBO-to-LOCO
collapse reproduces there, the claim is about protocol shift rather than about
NASA.

Oxford and Severson are both MARGINAL as transfer targets (internal resistance
only) and are not worth the download for this claim.

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
