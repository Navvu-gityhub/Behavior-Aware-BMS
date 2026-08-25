# ADR 0007: The target was the binding constraint, not the model

**Status:** Accepted
**Date:** 2026-08-21

## Context

Every calibration result this project has published — ADR 0002, ADR 0005,
`docs/final_report.md` Sections 4.1 through 4.8 — measures candidate models
against `capacity_loss`, the per-cycle change in measured discharge capacity.
The consistent finding was that nothing generalises: R² near or below zero for
the rule-based index, the fitted v2 model, and both adaptive candidates.

That finding was interpreted as a statement about the models. Building the
benchmark suite (`src/bms/benchmarks/`) required running *published* methods
against the same target, and doing so surfaced a prior question nobody had
asked: how much signal does the target contain at all?

Two measurements answer it, and both invalidate the earlier interpretation.

### Finding 1: `capacity_loss` is 96% noise

`benchmarks/targets.signal_to_noise` decomposes a target's within-cell
variance into monotone trend and residual by isotonic regression against cycle
index. The trend fraction is a ceiling on attainable R², since nothing can
predict the residual.

| Target | Signal fraction (max attainable R²) | Cells | Rows |
|---|---|---|---|
| `capacity_loss` | **0.044** | 33 | 2,680 |
| `soh` / `cumulative_fade` | 0.569 | 31 | 2,585 |
| `horizon_fade_10` | 0.389 | 31 | 2,268 |
| `horizon_fade_20` | 0.639 | 25 | 1,933 |
| `horizon_fade_50` | 0.967 | 17 | 1,246 |

*(Recomputed 2026-08-21 after the fourth screen criterion below. `soh` moved
0.612 → 0.569 as B0041 left the admissible set. `capacity_loss` is unaffected
because it is a within-cell difference that needs no fresh-capacity
reference — so the headline figure, and the conclusion resting on it, is
unchanged.)*

An R² of 0.044 was the *best possible score* on the target every previous
result was measured against. A per-cycle capacity delta is a difference of two
noisy measurements: it inherits both errors and cancels most of the trend.
Its mean is 0.003 Ah against a standard deviation of 0.132, and its minimum is
−2.61 Ah — capacity does not un-degrade by 2.6 Ah in one cycle.

So "no method beat a constant predictor" was substantially a property of the
target. On `cumulative_fade`, a plain `ElasticNetCV` scores LOBO R² = 0.651
and LOCO R² = 0.295 and **passes the promotion gate**. Three further
candidates clear it too (ADR 0008), so this project's "nothing is promoted"
state holds only against `capacity_loss`.

### Finding 2: measured capacity carries a large reversible thermal offset

Switching to an SOH-family target exposed a second problem, in the opposite
direction. Restricting to cycle ≤ 20, before meaningful degradation is
possible:

| Ambient | Mean SOH at cycle ≤ 20 | Cells |
|---|---|---|
| 4 °C | 0.918 | 10 |
| 22 °C | 0.992 | 3 |
| 24 °C | 0.909 | 14 |
| 43 °C | 0.979 | 4 |
| 44 °C | 0.999 | 3 |

A cell does not lose 8% of its capacity in twenty cycles. This is the
reversible temperature dependence of measured capacity — at low temperature,
higher internal resistance and slower lithium diffusion mean the cell reaches
its voltage cutoff sooner. It is not degradation, and it is *anti*-correlated
with real thermal degradation.

This supplies a mechanism for the project's longest-standing open puzzle.
Section 4.2 reports trailing temperature as a real, correctly-signed predictor
*within* cohorts; ADR 0002 reports the fitted model's ranking collapsing to
ρ = −0.295 *across* cohorts. Those are one phenomenon. Within a cohort ambient
temperature is fixed, the offset is constant, and residual temperature
variation carries genuine signal. Across cohorts the offset switches on and,
being anti-correlated with true thermal aging, does not add noise — it
reverses the sign.

## Decision

**Treat the target definition as an experimental variable, report its noise
ceiling alongside every result, and refuse to fit physics models whose
identifying assumptions the data violates.**

Four concrete consequences:

1. **`benchmarks/targets.py` constructs multiple targets** and
   `signal_report` publishes each one's ceiling. No R² is reported in this
   project without the ceiling it is measured against.

2. **SOH is screened at two levels.** `screen_cells_for_soh` excludes cells
   whose capacity series mixes measurement types — NASA's randomised-usage
   cells interleave reference and partial discharges, and normalising B0041 by
   its earliest cycles yields SOH = 22.6. Individual implausible readings are
   masked per observation. Both counts are reported.

   The cell screen applies three criteria, and the third is the one that does
   the work. Normalising by a high quantile of the cell's own capacity bounds
   SOH at 1.0 by construction, so an alternating full/partial series shows
   almost no *overshoot* — it simply reads as implausibly sudden loss and
   recovery. What identifies it is step size: degradation is gradual, so
   consecutive cycles should differ by a fraction of a percent. On NASA the
   separation is clean and needs no tuning — admissible cells have a median
   step of 0.5% of reference capacity (maximum 2.2%), while B0050 sits at
   28.3% and is excluded.

   A declining trend is *reported* but deliberately not screened on.
   Requiring the target to trend downward before admitting a cell would
   select the cells that agree with the hypothesis under test.

   **A fourth criterion was added on 2026-08-21, after loading CALCE.** The
   three above admitted all 13 CS2 cells, including two that are plainly
   wrong. CALCE Types 3, 5 and 6 do not perform one full discharge per
   `Cycle_Index` — Type 3 switches discharge rate six times *within* a cycle,
   and Types 5 and 6 cycle partially by design. Normalising by a high quantile
   of the cell's own capacity then picks a reference that is physically
   impossible: CS2_3 computes a **5.4 Ah reference on a 1.1 Ah cell**, and
   every ordinary cycle reads as 99.7% degraded.

   Neither existing test catches it, and the reason is instructive: a
   *too-large* reference deflates every reading smoothly, so there is no
   overshoot and no large step. The symptom is an absurdly low median.
   `MIN_MEDIAN_SOH = 0.40` therefore requires a cell to spend most of its life
   above 40% of its own reference — cells are retired near 70–80%, so a median
   below 40% means the denominator is not fresh capacity.

   The separation is wide rather than tuned: excluded cells sit at 0.003 and
   0.062, retained ones at 0.659 and above. On NASA it excludes exactly one
   additional cell, B0041 — the cell whose reference was already known to be
   unreliable (it is the SOH-22.6 case above). NASA's admissible count moves
   32 → 31 and its `soh` ceiling 0.612 → 0.569.

   **Types 5 and 6 still pass, and that needs stating.** Their reference is
   ~0.26–0.42 Ah — roughly a quarter of nominal — because their cycles are
   deliberately partial. SOH measured against that tracks *relative* fade of a
   repeated partial cycle, which is a real signal, but it is not absolute state
   of health and must not be reported as such. Recovering absolute SOH for
   those cohorts needs full-discharge segment detection, of the kind
   `telemetry/cycles.py` already performs for CAN logs.

3. **`physics/thermal_confound.py` measures and corrects the offset**, and
   documents that the correction is first-order and does not rescue the
   Arrhenius fit.

4. **`physics/arrhenius.assess_identifiability` refuses before fitting.** On
   this frame it returns NOT_IDENTIFIABLE for two quantified reasons:
   temperature varies only 0.171× as much within cohorts as between them, and
   apparent SOH differs by 0.090 across ambient levels at cycle ≤ 20.

## Why the Arrhenius model is kept but refused

`physics/arrhenius.py` fits `fade = A·exp(−Ea/RT)·N^z`, recovering an
activation energy with physical units and published reference values
(20–80 kJ/mol). The hypothesis was that `Ea`, being a property of the
chemistry rather than the experiment, would transfer across protocols where a
fitted linear temperature coefficient provably does not.

**On NASA data it returns Ea = −31 kJ/mol** — negative, meaning degradation
slows with heat. Per-cohort fits scatter from −568 to +183 kJ/mol, with 1 of 9
inside the published band. Correcting the thermal offset moves it to
−25.4 kJ/mol and discards 927 of 2,585 observations.

The estimator is not at fault: `tests/test_physics.py` recovers a known Ea to
within 0.12% from synthetic data, and to machine precision without noise. The
hypothesis is simply not testable on this frame.

The module ships anyway, gated, for the same reason `health_index_v2` ships
unpromoted (ADR 0002): a correct implementation with a documented refusal is a
usable asset the moment adequate data arrives, and deleting it would discard
the diagnosis along with the code.

## Consequences

**Earlier conclusions need restating, not retracting.** Every published number
remains correct as a statement about `capacity_loss`. What changes is the
scope of the claim: "these scores do not predict per-cycle capacity delta,
which is 96% measurement noise" rather than "these scores do not predict
degradation."

**The headline finding survives, and is now cleanly separated from the
artifact.** On a well-conditioned target the LOBO-to-LOCO gap is still large
(−0.357 R² for ElasticNet). Protocol shift genuinely degrades generalisation.
That was previously indistinguishable from the target's noise floor; it no
longer is.

**A concrete data acceptance criterion follows**, joining the "8–10+ batteries
per cohort" criterion from Section 4.6. The next dataset must supply reference
discharges at a common temperature, and must vary temperature *within*
protocol rather than confounding the two. Neither NASA nor the supplied CALCE
data does.

## Alternatives rejected

**Keep reporting against `capacity_loss` only.** Reproducible, and wrong. It
reports a target artifact as a modelling conclusion.

**Silently switch to `cumulative_fade`.** Would make every previously
published number incomparable with no visible reason, and would smuggle in the
thermal confound undetected.

**Tune the Arrhenius model until Ea is positive.** The available knobs —
temperature column, correction window, cohort subset, fade floor — are
sufficient to produce a plausible number. Doing so would fabricate a physical
constant by search, which is the exact failure this project exists to prevent.

## References

- `src/bms/benchmarks/targets.py` — target construction, screening, noise ceiling
- `src/bms/physics/thermal_confound.py` — offset measurement and correction
- `src/bms/physics/arrhenius.py` — model and identifiability gate
- `tests/test_physics.py` — synthetic recovery of a known activation energy
- `scripts/run_benchmark_study.py` — reproduces every table above
- ADR 0002 (health index version), ADR 0005 (the gate is the product)
