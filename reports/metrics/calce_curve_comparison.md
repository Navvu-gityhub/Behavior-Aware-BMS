# Do curve-derived features survive the cohort boundary?

Arm A: usage aggregates. Arm B: the same, plus `delta_q_variance`,
`ica_peak_height`, `ica_peak_voltage` and `ica_area`. Identical rows
(9,950), identical cells (18), identical cohorts (7), identical gate.
Only the feature set differs.

## Verdict

- Fitted methods whose LOCO R2 improved: **1 of 4**
- Methods whose A and B intervals are separated: **0 of 4**
- Mean change in LOCO R2: **-0.1240**
- LOBO-to-LOCO skill loss still present in every fitted method under arm B: **True**

**Adding the literature's curve-derived features did not restore
cross-cohort skill.** No method's interval separates between arms, so
the per-method changes below are within fold-to-fold noise and none of
them supports a claim in either direction.

What does survive is the direction. Every fitted method still loses
skill crossing the cohort boundary with curve features in the model.
That makes the original finding harder to dismiss rather than easier:
it can no longer be attributed to the features being mere usage
aggregates, because the field's canonical electrochemical features
are in arm B and the collapse is unchanged.

## Per-method

| method | LOCO A | LOCO B | change | A 95% CI | B 95% CI | CIs overlap |
|---|---|---|---|---|---|---|
| `random_forest` | 0.7292 | 0.3604 | -0.3688 | [-0.668, 0.795] | [-0.536, 0.639] | yes |
| `elasticnet` | 0.5793 | 0.4058 | -0.1735 | [-0.171, 0.651] | [0.010, 0.747] | yes |
| `xgboost` | 0.7501 | 0.6313 | -0.1188 | [-1.031, 0.788] | [-1.661, 0.836] | yes |
| `age_linear` | 0.6599 | 0.6599 | +0.0000 | [-0.147, 0.726] | [-0.147, 0.726] | yes |
| `age_quadratic` | 0.5839 | 0.5839 | +0.0000 | [0.209, 0.822] | [0.209, 0.822] | yes |
| `age_isotonic` | 0.5576 | 0.5576 | +0.0000 | [0.212, 0.810] | [0.212, 0.810] | yes |
| `train_mean` | 0.0000 | 0.0000 | +0.0000 | [0.000, 0.000] | [0.000, 0.000] | yes |
| `hist_gradient_boosting` | 0.6072 | 0.7726 | +0.1653 | [-0.951, 0.826] | [-0.828, 0.901] | yes |

## Harness check

`train_mean` and the three `age_*` baselines regress on cycle number
alone and cannot see the curve columns. Identical across arms to 1e-9: **True**. That is the check that the two runs differed in nothing but the feature set.

## What this comparison does not establish

- **Not a refutation of Severson et al.** That result predicts *cycle
  life* from early cycles on 124 LFP cells varying charge policy.
  This predicts *per-cycle SOH* on 18 LCO cells varying depth of
  discharge and rate. The feature is the same; the target, the
  chemistry and the experimental axis are not.
- **Not a test at adequate power.** Seven cohorts, two of them a
  single cell. The curve features are per-cell constants, so their
  effective sample size is 18, not 9,950.
- **Not a full method table.** Excluded from both arms for runtime: `gpr_matern`, `svr_rbf`, `mlp`, `lstm`, `arrhenius_avg_temp`, `arrhenius_trailing_temp`. Their scores on the full CALCE run are in
  `reports/metrics/calce_full_discharge/`.
- **Not a statement about an unseen cohort.** The intervals cover
  spread across the seven cohorts present, which is a different
  quantity from transfer to an eighth.