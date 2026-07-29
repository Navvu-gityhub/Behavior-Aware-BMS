# ADR 0001: Cross-dataset validation is not possible; leave-one-cohort-out replaces it

**Status:** Accepted
**Date:** 2026-07-29
**Supersedes:** the "NASA → CALCE cross-dataset validation" item in the project backlog

## Context

Leave-one-battery-out (LOBO) cross-validation was already in place. The
natural next test — and the one the reference literature (papers #3, #4, #7)
uses to justify generalisation claims — is cross-dataset: fit on NASA, test
on CALCE. Both loaders exist, so this looked like a build task rather than a
research question.

It is not buildable, and the reason is a property of the supplied data rather
than of the code.

## The blocking fact

Verified by direct inspection, not assumed (see `docs/calce_dataset_note.md`):

- The supplied CALCE files are `Capacity_Characterization_Initialization` —
  17 Arbin workbooks covering 138 of 150 PLN pouch cells.
- **Every channel in every workbook contains exactly one `Cycle_Index`.**

A cycle-based capacity-fade model needs a cycle-indexed fade target. A
single-cycle baseline characterisation has no fade to predict. There is no
model error to compute because there is no target, so the test cannot return
a result of any kind — not a poor one, none.

A calendar-aging variant was also attempted and closed: the per-cell capacity
in the characterisation files is bit-for-bit identical to the "post-storage"
`Discharge Capacity` column for all 132 matched cells, because both record
the same measurement event. There is no independent pre-storage baseline in
this upload.

## Decision

1. **Do not run, simulate, or approximate a NASA → CALCE test.** Reporting a
   number from a test that cannot be performed would be fabrication, and the
   absence of a target makes it fabrication that no reviewer could detect
   from the output alone.
2. **Substitute leave-one-cohort-out (LOCO) across NASA's 9 experimental
   protocols,** implemented in `scripts/validate_health_index_versions.py`
   and `scripts/fit_shap_attribution_model.py`.
3. **Label the substitution explicitly** everywhere it is reported, including
   in `docs/final_report.md`. It is not a cross-dataset result.

## Why LOCO is a defensible substitute

Cross-dataset validation probes one property: does the model survive a
distribution shift it was not fitted on? NASA's 34 cells span 9 protocols
that differ in ambient temperature (4 °C to 43 °C), discharge current
(1 A to 4 A), load profile (constant-current, square-wave, multi-load) and
cutoff voltage. Holding out an entire protocol is a real distribution shift
in exactly the variables that drive degradation.

It is a **weaker** claim than cross-dataset, and the write-up says so: same
laboratory, same cell chemistry, same instrumentation, same measurement
convention. It rules out protocol-specific overfitting. It does not rule out
NASA-specific overfitting.

## Consequence, and why this ADR earned its place

LOCO was not a consolation prize. It produced the single most important
finding in the calibration work:

| Test | Spearman rho vs measured fade | Verdict |
|---|---|---|
| v2 fitted model, in-sample | 0.870 | inadmissible (fitted on these cells) |
| v2, unseen cell in a **known** protocol (LOBO) | 0.841 (p < 0.001) | strong |
| v2, unseen **protocol** (LOCO) | −0.295 (p = 0.10) | collapses, wrong sign |

The model's apparent ranking ability is almost entirely carried by its fitted
per-cohort intercepts. Within a known protocol it works; on an unseen
protocol it is no better than noise and points the wrong way. LOBO alone
could never have shown this, because LOBO always leaves the held-out cell's
cohort siblings in the training set.

## Reopening condition

This ADR should be revisited when a **multi-cycle** CALCE dataset is
obtained — the CS2 or CX2 cycling series, which CALCE publishes separately
and which does carry cycle-indexed capacity. The validation harness in
`scripts/validate_health_index_versions.py` is written against a generic
`(cell_id, cohort, capacity_loss, trailing_avg_temp)` frame, so a working
loader for that data is the only missing piece.

## Alternatives rejected

- **Treat the CALCE single-cycle capacities as a one-point fade target.**
  One point per cell defines no fade trajectory, and the resulting "test"
  would measure between-cell manufacturing spread, not degradation.
- **Use the CALCE EIS impedance spectra as a proxy target.** Impedance growth
  correlates with fade but is a different quantity on a different scale; a
  model fitted to predict capacity loss cannot be scored against it without
  an additional, itself-uncalibrated, transfer assumption.
- **Downgrade the claim and report LOBO as if it were general.** This is what
  the literature review flagged as the field-wide failure mode, and the LOCO
  result above shows precisely why it would have been wrong here.
