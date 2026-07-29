# ADR 0003: Exact Shapley attribution for the rule scores; SHAP only where it is the right tool

**Status:** Accepted
**Date:** 2026-07-29

## Context

The project claimed a "Battery Guardian AI" that "explains causes". The
implementation was a set of if/else thresholds in `guardian.py`. The reference
literature (papers #8, #10, #12) treats SHAP or comparable feature attribution
as the expected standard for that claim, so this was the largest gap between
what the project asserted and what it did.

Auditing the old implementation turned up something worse than an absent
method. `_primary_causes` used cut points — `avg_temp > 35`,
`fast_charge_duration > 50`, `deep_discharge_duration > 50` — that **appear
nowhere else in the codebase**. The risk score's bands sit at 30/40, 20/100
and 20/100. Guardian was explaining a score using a different rulebook from
the one that produced it, which permitted two concrete failures:

- **Naming a cause that contributed nothing.** A cell at 36 °C was told "high
  temperature exposure" while the temperature term had placed it in its
  *lowest* band.
- **Omitting the dominant term.** `avg_stress` contributes up to 30 points and
  `aggressive_discharge_count` up to 15. Neither appeared in the cause list.

## Decision

**Two explainers, chosen by whether the thing being explained is additive.**

### 1. Exact closed-form Shapley values for the rule-based scores

The risk score and aging budget are sums of independent per-feature terms.
For an additive model `f(x) = Σᵢ fᵢ(xᵢ)`, writing the coalition value as
`v(S) = E[f(X) | X_S = x_S]` gives a marginal contribution
`v(S ∪ {i}) − v(S) = fᵢ(xᵢ) − E[fᵢ(Xᵢ)]` that **does not depend on S**.
Every term of the Shapley average is identical, so

    φᵢ = fᵢ(xᵢ) − E[fᵢ(Xᵢ)]

This is implemented in `src/bms/explain/attribution.py`. Three consequences
made it the right choice over calling KernelSHAP on the same function:

- **No approximation.** KernelSHAP samples coalitions and returns an estimate;
  the closed form is exact, so there is no sampling error and nothing to tune.
- **No feature-independence assumption.** KernelSHAP's interventional estimate
  is faithful only when features are roughly independent — false for battery
  telemetry, where temperature, C-rate and SOC exposure are strongly
  correlated. The derivation above never factorises a joint distribution:
  `v(S)` decomposes because `f` is additive, not because the features are
  independent. The correlation is real and large, and does not bias these
  attributions.
- **Cheap enough to run inline.** One vectorised pass, so every battery is
  attributed in the pipeline rather than in an offline analysis step.

### 2. `shap` on a gradient-boosted model, for the question the rules cannot answer

Attribution of the rules is circular: it can only ever restate what the rules
assert. It cannot say whether the rules weight the *right* things.
`scripts/fit_shap_attribution_model.py` fits a non-additive booster to
**measured** capacity fade and uses TreeSHAP — the correct tool there, because
that model genuinely is not additive.

### 3. Term definitions are shared, and the sharing is enforced

`RISK_TERMS` and `HEALTH_TERMS` are defined once and consumed by both the
scorer and the explainer. `verify_efficiency` asserts `Σφᵢ = f(x) − E[f(X)]`
on every call. Because that identity is guaranteed by the maths, any failure
means the specs have drifted from the scorer — so it doubles as a drift test,
and `tests/test_explain.py` exercises it with a deliberately-mutated term.

The refactor that extracted the terms was verified **bit-identical on all 33
real NASA batteries** against the committed pre-refactor outputs
(`test_refactor_is_score_identical_on_real_nasa_data`).

## What the SHAP analysis found, and why it is reported as negative

The booster **fails its own validation gate**. LOBO median R² = −0.096 (42% of
folds beat the baseline); LOCO median R² = −1.61 (11%). The script therefore
reports its SHAP ranking as describing the model's internal fitting behaviour
only, and **not** as evidence about physical degradation drivers.

This gate is the methodological point. SHAP attributes a model's predictions
to its inputs and says nothing about whether those predictions are any good.
Running it on a model with no out-of-sample skill yields a confident-looking
importance ranking that describes how the model fit noise — which is worse
than no analysis, because it looks like evidence. The gate is computed, not
chosen in advance, and the ranking is still published rather than suppressed.

Rank correlation between the rules' weighting and the model's importance
ranking: **−0.103**. The honest reading is not "the rules are wrong and SHAP
is right" — it is that this dataset supports *neither* weighting.

## The finding that came out of building this

Sharing the term definitions made the scores auditable, which made
`scripts/audit_threshold_reachability.py` possible. It shows that several
hand-chosen cut points sit outside the range real data ever occupies:

| Term | Feature range in NASA | Status |
|---|---|---|
| `stress` | 1.71 – 33.38 (bands at 50, 70) | **DEGENERATE** — never leaves its lowest band |
| `fast_charge` | 0.00 – 0.00 (bands at 20, 100) | **DEGENERATE** — feature is identically zero |
| `deep_discharge` | 77 – 3768 | PARTIAL — 99.9% in one band |
| `temperature` | 5.18 – 46.78 | ACTIVE |
| `aggressive_discharge` | 0 – 546 | ACTIVE |
| `soc_extremes` | 0.43 – 68.78 | ACTIVE (only the low branch fires) |

**61% of the mean risk score comes from terms identical for every battery in
the dataset.** NASA contains no fast-charge events at all, so the 15-point
fast-charge penalty is not merely unvalidated — it is *unvalidatable* on this
data.

This reframes the earlier "no significant correlation with fade" result.
The scores did not fail because the behavioural hypothesis was wrong; they
failed because most of the score was a constant, and a constant cannot
correlate with anything. Those are different diagnoses with different fixes.

## Boundary on what Guardian may claim

Guardian explains **why a battery scored what it scored**. That is now exact.
Guardian does **not** explain **why a battery is degrading** — the v1 index it
attributes correlates with measured fade at rho = −0.27 (ADR 0002).
Attributing a score that does not track degradation gives a faithful account
of the rules and no account of the physics.

`guardian_caveat` carries this into the output table itself, so it travels with
the data rather than living only in a document beside the dashboard.

## Known defect, logged rather than silently fixed

`RISK_TERMS` uses `>=` where the structurally identical `HEALTH_TERMS` uses
`>`. This predates the refactor and only changes behaviour for a battery
sitting exactly on a cut point. It is **preserved**, because harmonising it
would alter scores already published in `docs/calibration_report.md`, and the
refactor's stated purpose was to change no score. Proposed fix: harmonise to
`>=` and regenerate the affected calibration artifacts in one commit whose
diff is reviewable as a deliberate numerical change.

## Alternatives rejected

- **KernelSHAP on the rule scores.** Adds a dependency, sampling noise, and a
  false independence assumption, in exchange for an approximation of a value
  available exactly.
- **Keep the if/else causes and document the mismatch.** The mismatch is a
  correctness bug, not a limitation.
- **Report SHAP importance without the skill gate.** This is the failure mode
  the gate exists to prevent, and it would have produced a confident,
  publishable-looking, meaningless ranking.
