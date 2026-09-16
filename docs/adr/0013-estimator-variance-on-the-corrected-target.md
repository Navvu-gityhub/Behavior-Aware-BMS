# ADR 0013: The estimator-variance result survives the target correction as a ratio, not as a magnitude

**Status:** Accepted
**Date:** 2026-09-03

## Context

The project's one novel claim is that a leave-one-cohort-out R2 is too noisy to
rank methods with: the spread of the LOCO estimate across cohort draws is
larger than the between-method difference it is being used to adjudicate.
The published figures are

    LOCO interquartile range                    0.724 R2
    median paired svr_rbf - age_linear          -0.351 R2
    draws in which svr_rbf beats age_linear     25%

and they come from `reports/metrics/coverage_sweep.csv`, produced by
`scripts/run_coverage_sweep.py` over two frames: NASA, and
`reports/metrics/calce_cycle_level.csv`.

That second frame is the `Cycle_Index` derivation that **ADR 0012 refutes as a
measurement artifact**. ADR 0012's own words: a target alternating between full
and partial discharge capacities cannot be tracked by a smooth function of
cycle number, but can be partially fitted by a flexible model, so the flexible
models were being rewarded for fitting a measurement artifact.

The sweep's nonlinear arm is `svr_rbf`; its baseline `age_linear` is a smooth
function of cycle number. The sweep was therefore measuring precisely the
flexible-versus-smooth contrast that ADR 0012 shows this target distorts. The
repository contained the argument for why its own central measurement needed
repeating, and had not repeated it.

## Decision

Re-run the sweep on `reports/metrics/calce_full_discharge.csv` — the ADR 0012
derivation of the *same* CALCE cells, with the *same* feature set, the same two
methods and the same sweep machinery. Only the target derivation differs.

**The new run writes a separate artifact.** All three published figures above
are recomputed by `tests/test_reported_numbers.py` by pooling every row of
`coverage_sweep.csv`, so a third frame written into that file would have
redefined three quoted numbers without any test failing on the substance.
`run_coverage_sweep.py` now takes `--frames` and `--out-prefix`, and refuses to
write a non-default frame set into the default prefix.

    python scripts/run_coverage_sweep.py --frames calce_full_discharge \
        --out-prefix coverage_sweep_full_discharge

## What the corrected target says

Seven of ten cohorts carry the two cells LOCO needs, spanning 19 cells and
11,515 rows; k = 3..7 with 12 draws each, 98 sweep points.

Compared on the three coverage levels both runs draw fully (k = 3, 4, 5), CALCE
against CALCE:

| | `Cycle_Index` target | full-discharge target |
|---|---:|---:|
| LOCO IQR of the gap | **0.892** | **0.296** |
| median paired `svr_rbf - age_linear` | -0.336 | -0.110 |
| draws where `svr_rbf` wins | 13.9% | 38.9% |
| **noise-to-effect ratio** | **2.66** | **2.70** |

Every figure in that table is computed on k = 3, 4, 5 only. Pooled over all
fully-drawn levels the corrected IQR is 0.291 and the win rate 32.7%; the
matched-level figures are the ones that compare like with like.

The IQR reduction is 0.596, bootstrap 95% CI [0.335, 0.936] over 4,000
resamples of draws within levels; the corrected estimate is smaller in every
one of them.

**Three conclusions, and they are not the same conclusion.**

1. **The claim survives, as a ratio.** The spread of the LOCO estimate is 2.7x
   the between-method difference on the corrected target, against 2.66x on the
   artifact target. The two numbers agree to two significant figures, which is
   more agreement than this project has any right to expect and should not be
   over-read — but the qualitative claim, that the measurement is several times
   noisier than the effect it is used to detect, is **not** an artifact of the
   target derivation.

2. **The magnitudes were inflated roughly threefold.** Both the noise and the
   effect shrink by about 2.7-3x. Any sentence quoting 0.724 or 0.351 as a
   property of leave-one-cohort-out in general is overstating it by that
   factor; those are properties of that estimate on that frame.

3. **The mechanism does not survive, and this is the part to report.** On the
   `Cycle_Index` target the flexible model collapsed hardest: `svr_rbf` had
   median LOBO 0.835 falling to LOCO -0.196 (gap -1.037), against `age_linear`'s
   gap of -0.096. On the corrected target the ordering **inverts**: `age_linear`
   has the higher LOBO (0.695 -> 0.231, gap -0.460) and `svr_rbf` the smaller
   collapse (0.357 -> 0.090, gap -0.203). "The nonlinear method loses the most
   skill under protocol shift" was the artifact ADR 0012 identified, seen from
   a second direction. It must not be stated as a finding.

**ADR 0010 replicates.** Raw Spearman(n_cohorts, gap) = +0.126 (p = 0.217, n =
98); adjusted for cell count = -0.002 (p = 0.982); gap versus cell count alone
= +0.125. On the `Cycle_Index` frame the same three figures were +0.257
(p = 0.027), +0.024 (p = 0.837), +0.272. The raw relationship is weaker here
and was never significant, and in both frames it vanishes once cell count is
regressed out. The coverage hypothesis stays refuted on a second derivation.

## Consequences

- `coverage_sweep.csv` and the three figures pinned to it are **unchanged and
  still correct for what they measure**. Nothing published is withdrawn.
- The manuscript's Section 5.2 needs the corrected-target measurement beside
  the original, and Section 7 needs the limitation that Section 5.4 already
  implies: the sweep was run on the derivation Section 5.4 refutes. A reviewer
  who reads 5.4 before 5.2 finds this unaided.
- The abstract's magnitude claim becomes target-specific. The ratio claim does
  not.
- Do not restate "the flexible model transfers worst". That is the fifth
  withdrawn ranking claim in this project's history, and it died the same way
  as the other four (see `bms-key-findings`: the ranking moved by varying the
  target derivation, never the model).
- Candidate C from the 2026-09-02 audit — gating promotion on an interval over
  cohort draws rather than a point estimate — now has its distribution measured
  on a defensible target and is unblocked.

## Artifacts

    reports/metrics/coverage_sweep_full_discharge.csv          98 rows
    reports/metrics/coverage_sweep_full_discharge_summary.csv
    reports/metrics/coverage_sweep_full_discharge.md

The figures in the table above are pinned in `tests/test_reported_numbers.py`
against these files, with the recipes stated, exactly as the originals are.
