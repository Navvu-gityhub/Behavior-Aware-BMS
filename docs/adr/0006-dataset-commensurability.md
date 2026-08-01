# ADR 0006: Transfer targets are screened on commensurability, not availability

**Status:** Accepted
**Date:** 2026-07-30

## Context

The project's headline experiment is "train on NASA, validate on CALCE and
Stanford." The obvious implementation order is: write loaders, run the transfer,
report the metric.

That order is wrong, and following it would have produced a number with no
meaning.

A transfer requires a feature that varies in **both** datasets. If a feature is
constant in the source, no coefficient can be fitted for it. If it is constant
in the target, the fitted coefficient has nothing to act on — every target row
receives the same contribution, which is an intercept, not a prediction. Either
way the model looks fitted and predicts nothing, the same defect class as the
degenerate risk terms in ADR 0005.

## Decision

**Screen every proposed transfer target on commensurability before loading it,
and refuse to compute a transfer metric when no feature varies on both sides.**

Two layers, in order of cost:

1. `dataset_specs.py` — `predict_transfer_feasibility` compares *published
   metadata*. Runs in milliseconds, before any download.
2. `commensurability.py` — `assess_commensurability` measures *loaded data*, and
   runs as a hard precondition inside `TransferValidator.evaluate`.

The first is a screening aid and says so in its own output. It reflects what
dataset authors documented, which is not always what the files contain, so it
never substitutes for the second.

## What the screen found

### NASA and Stanford/Severson vary along orthogonal axes

| | NASA | Severson |
|---|---|---|
| Ambient temperature | **varied**, ~4-43 C over 9 protocols | **fixed**, 30 C chamber |
| Charge rate | **fixed** (`fast_charge_duration` identically zero, all 2,682 rows) | **varied**, 3.6C-6C |
| Discharge rate | varied | fixed (all cells 4C to 2.0 V) |
| Depth of discharge | varied | fixed |

The axis NASA varies is held in Severson; the axis Severson varies cannot be
fitted on NASA. Predicted status: `PREDICTED_MARGINAL`, with internal
resistance the only surviving feature and incidental on both sides.

Severson does record cell temperature, and it does move — by up to about 10 C
from self-heating. But that is roughly an order of magnitude below NASA's
designed range, so a coefficient fitted across 4-43 C acting on a ~2 C spread
produces almost no prediction variance. Recorded and moving is not the same as
experimentally varied, which is why `Variation.INCIDENTAL` exists as a distinct
state from `VARIED`.

### CALCE is feasible, but not on temperature

CS2's 15 LCO prismatic cells were cycled at room temperature, about 23 C, all
on the same 0.5C CC-CV charge profile. So CS2 cannot receive a NASA temperature
coefficient either.

What CS2 *does* vary is depth of discharge, discharge rate and cutoff voltage —
and NASA varies all three, with `deep_discharge_duration` spanning 77 to 3,768
(std 430). **Depth of discharge, not temperature, is where a NASA-to-CALCE
transfer is well posed.** That reframes the experiment rather than cancelling
it.

### One CALCE cell is thermally admissible, and it is n=1

CX2_4 was cycled across 25, 35, 45 and 55 C with separate thermocouple data. It
is the only CALCE unit with a thermal axis comparable to NASA's, and therefore
the only scientifically admissible CALCE target for a temperature model.

It is also one cell. A transfer test there can characterise the temperature
relationship on a single unit but cannot support a cell-level generalisation
claim, because there is no between-cell variance to estimate. Both facts are
recorded in `CALCE_CX2_4_THERMAL_SPEC.caveats` so neither is lost, and
`test_cx2_4_is_the_only_thermally_admissible_calce_target` asserts the caveat
survives.

## A defect this produced, and the fix

The first implementation judged variation by coefficient of variation
(std / |mean|) against a fixed floor. That is wrong for temperature.

Celsius is an interval scale with an arbitrary zero: 70 C is not "2.8 times more
temperature" than 25 C. A fixed CV floor therefore made the same physical spread
of +/-1.5 C count as varying at 25 C (CV 0.06) and constant at 70 C (CV 0.02).
On the real datasets it inverted the truth — NASA's 4 C cohort scores CV 0.375
and looks highly variable, while Severson's 30 C chamber scores about 0.08 and
looks flat despite a larger absolute excursion.

Variation is now judged by **spread relative to the source's own spread**, which
is the range the coefficient was fitted over. That is scale-free in the way that
matters because it compares like with like on one channel.
`test_celsius_scale_does_not_decide_whether_a_feature_varies` pins it.

## Consequences

- A transfer to a dataset with no commensurable feature returns no metric. The
  `TransferResult` carries `status="ERROR"` with the commensurability report
  attached, rather than a number.
- The NASA-to-CALCE experiment proceeds, reframed onto depth of discharge and
  discharge rate.
- The NASA-to-Stanford temperature experiment does not proceed. Reporting it
  would mean reporting a coefficient acting on a near-constant.
- Adding a dataset is configuration: a `DatasetSpec` with a column map and a
  variation profile, no new code.

## What would change this

A dataset with **both** thermal and charge-rate variation would make the
original experiment well posed. Neither NASA, CS2, CX2 (excepting CX2_4) nor
Severson has both. Candidates worth screening: the CALCE pouch-cell set (16 LCO
cells across depths of discharge in a semi-controlled 25 +/- 2 C room), and
Attia et al. 2020, the 45-cell Severson follow-up.

Screen with `predict_transfer_feasibility` before downloading either.

## Alternatives rejected

- **Write the loaders first, screen later.** The order that motivated this ADR.
  It spends the effort before learning whether the question is answerable.
- **Standardise features across datasets before transfer.** Rescaling a
  near-constant target feature to match the source's spread manufactures
  variance that is not in the data, turning a failed transfer into a
  successful-looking one.
- **Report the Stanford temperature transfer with a caveat.** A caveat does not
  repair a coefficient acting on a constant. Refusing is the honest output.
- **Treat `INCIDENTAL` as equivalent to `VARIED`.** Would have passed the
  Stanford temperature transfer on self-heating noise.
