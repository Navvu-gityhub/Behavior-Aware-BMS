# Benchmark model catalogue

What each registered method is, what it needs, and what it may be claimed to
show. Generated results live in `reports/metrics/benchmark_results.csv`;
reproduce with `make study`.

Every method here is evaluated by `src.bms.adaptive.validation.Validator` —
the same gate the project applies to its own candidates, with leave-one-cohort-out
mandatory. A method does not get a friendlier evaluation for being well known.

## Families

| Family | Methods |
|---|---|
| `naive` | `train_mean`, `age_linear`, `age_quadratic`, `age_isotonic` |
| `classical` | `elasticnet`, `svr_rbf`, `gpr_matern`, `random_forest`, `hist_gradient_boosting`, **`xgboost`**, `mlp`, **`lstm`** |
| `physics` | `arrhenius_avg_temp`, `arrhenius_trailing_temp` |
| `curve` | `severson_delta_q_variance`, `ica_peak_features`, `sequence_model` |

## Optional dependencies

`xgboost` and `lstm` need packages the core install does not carry:

```bash
pip install "behavior-aware-bms[benchmarks]"   # xgboost + torch
```

Without them, `check_availability` returns one UNAVAILABLE row naming the
missing package. It does not crash a fold, and it does not silently drop the
method from the results table — an absent row and an absent method are
indistinguishable to a reader, which is why neither is allowed.

---

## XGBoost

**Status:** experimental benchmark. Not wired into the pipeline, the API, the
dashboard or the digital twin.

Implemented in `src/bms/benchmarks/classical.py::_xgboost` using the actual
`xgboost` package (`XGBRegressor`).

| | |
|---|---|
| Library | `xgboost` (verified by `tests/test_benchmarks.py::TestXGBoost`) |
| Features | `DEFAULT_FEATURES` — same six behavioural columns + `cycle` as every other classical method |
| Target | whichever the study runs (`soh`, `cumulative_fade`, `capacity_loss`) |
| Preprocessing | shared `impute → standardise → fit` pipeline |
| Hyperparameters | 300 rounds, lr 0.05, depth 4, subsample 0.8, colsample 0.8, L2 1.0, `tree_method="hist"` |
| Seed | `RANDOM_SEED = 20260821`, `n_jobs=1` |
| Tuning | none per fold — see below |

### Why it is registered separately from `hist_gradient_boosting`

They are different implementations of the same idea and the distinction is
not cosmetic: XGBoost grows level-wise by default and applies an explicit
L1/L2 penalty to leaf weights, where sklearn's histogram booster (LightGBM
lineage) grows leaf-wise. Registering one and labelling it "XGBoost" would
misreport which algorithm produced the number.

`tests/test_benchmarks.py` asserts the estimator really is an `XGBRegressor`
and really is not a `HistGradientBoostingRegressor`, so the substitution
cannot creep back in.

### Hyperparameters are fixed, not searched

Per-fold tuning would be more favourable to XGBoost and is the right thing to
do for a paper claiming a method is best. This benchmark makes the opposite
claim — that even reasonable defaults fail to transfer — so untuned defaults
are the conservative choice *against* the thesis. The honest reading of the
reported score is that it is a **lower bound** on what a tuned XGBoost would
achieve, and the gap this study reports is correspondingly an upper bound.

---

## LSTM

**Status:** experimental benchmark. Not wired into anything, and **not
production-ready** merely because it runs.

Implemented in `src/bms/benchmarks/sequence.py`. This is a genuine sequence
model — shuffling the cycles destroys its input, which `tests/test_sequence.py`
checks directly.

| | |
|---|---|
| Sequence length | **10 consecutive cycles** (`DEFAULT_WINDOW`) |
| Features per timestep | `avg_temp`, `max_temp`, `avg_stress`, `deep_discharge_duration`, `aggressive_discharge_count`, `avg_soc`, `cycle` — identical to every classical method |
| Prediction target | the target value at the **last cycle of the window** |
| Target alignment | window covers cycles *k−9 … k*, predicts at *k*. Never reads *k+1* |
| Architecture | 1 LSTM layer, hidden 32, linear head on the final timestep |
| Training | Adam, lr 1e-3, batch 64, ≤60 epochs, early stopping (patience 8) on a **cell-disjoint** 20% split of the training fold |
| Seeds | `torch.manual_seed` + `numpy` + a seeded shuffle generator; `use_deterministic_algorithms(warn_only=True)` |
| Library | `torch` (CPU) |

### Why a sequence formulation is defensible here

Checked before the module was written, because the answer might have been no:

* Cycle index is **strictly increasing in all 32 cells** — zero violations.
* **99.4% of consecutive-cycle gaps are exactly 1**; the largest is 2. A
  window of 10 rows is a window of ~10 cycles, not an arbitrary span.
* Shortest cell 21 cycles, median 67, longest 193.

### Why the window is 10

| Window | Cells retained | Windows |
|---|---|---|
| 5 | 32/32 | 2,520 |
| **10** | **32/32** | **2,360** |
| 20 | 32/32 | 2,040 |
| 30 | 26/32 | 1,748 |

At 30 six cells drop out entirely. Ten keeps every cell **and** gives the LSTM
twice the temporal context of the hand-engineered `trailing_*` features (a
5-cycle trailing window) it competes against — so if it loses, it does not
lose for want of history.

### Leakage controls

| Risk | Control |
|---|---|
| Across the fold boundary | Windows built separately per frame; never span train and test. Held-out cell/cohort contributes no training window |
| Through the scaler | `StandardScaler` statistics computed on training windows only |
| From the future | Window for cycle *k* spans *k−9 … k*; asserted by a fixture whose feature value equals its own cycle index |
| Early-stopping split | Split by **cell**, not by window — overlapping windows from one cell are near-duplicates, and a random split would measure memorisation |

**Not leakage:** using a held-out cell's *earlier* cycles as input when
predicting its later ones. That is the deployment situation, and no held-out
target is ever read.

### Known limitations

1. **Timesteps are cycle-level aggregates, not raw telemetry.** The frame
   carries one pre-aggregated row per cycle. A sequence model over the
   within-cycle voltage/current trace is a different and probably better
   experiment; it is registered as `sequence_model` in `curves.py` and
   correctly reports UNAVAILABLE, because this repository does not retain
   those traces.
2. **The axis is cycle count, not wall-clock time.** Calendar ageing is
   invisible.
3. **The first 9 cycles of each cell are left-padded** by repeating that
   cell's earliest observation. Dropping them would score the LSTM on an
   easier subset than every other method. `predict` reports the padded count.
4. **2,360 windows across 32 cells is small for a neural network.** The model
   is deliberately tiny (hidden 32, one layer) so the LOBO/LOCO gap is a
   statement about protocol shift rather than about model capacity.
5. **No per-fold hyperparameter search**, for the same reason as XGBoost.

---

## What these two may not be claimed to show

Neither is promoted, and adding them does not change the project's position:

- A score on this dataset is a statement about **NASA's nine protocols**, not
  about batteries in general. Replication on CALCE is the outstanding item
  (`docs/roadmap.md`).
- Both are measured against targets whose **noise ceiling is published
  alongside** (ADR 0007). An R² is meaningless here without it.
- Passing the promotion gate is **not** a licence to deploy: ADR 0005's rule
  that the gate is the product still holds, and one dataset is not deployment
  evidence.
