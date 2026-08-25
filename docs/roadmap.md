# Roadmap

Where this project stands, and what each remaining step actually requires. The
ordering is by dependency, not by appeal.

## Done

| Capability | Where | Status |
|---|---|---|
| End-to-end pipeline on simulated telemetry | `main.py` | Working |
| Unified schema, NASA/CALCE loaders, CAN/DBC replay | `src/bms/io/`, `src/bms/telemetry/` | Working |
| Exact Shapley attribution over the rule scores | `src/bms/explain/attribution.py` | Working, exact |
| Validation gate with mandatory LOCO | `src/bms/adaptive/validation.py` | Working |
| Pre-download commensurability screen | `src/bms/adaptive/dataset_specs.py` | Working |
| Benchmark suite of published methods | `src/bms/benchmarks/` | Working |
| Target noise-ceiling estimation | `src/bms/benchmarks/targets.py` | Working |
| Thermal measurement confound diagnosis | `src/bms/physics/thermal_confound.py` | Working |
| Arrhenius model + identifiability gate | `src/bms/physics/arrhenius.py` | Correct, **refused on available data** |
| Conformal prediction + coverage under shift | `src/bms/uncertainty/` | Working |
| Real XGBoost + real LSTM benchmarks | `src/bms/benchmarks/{classical,sequence}.py` | Working |
| CALCE CS2+CX2 loader (zip, nested, cohort-from-directory) | `src/bms/io/load_calce_cycling.py` | Working |
| Cohort-coverage sweep with size control | `src/bms/benchmarks/coverage.py` | Working — refuted its own hypothesis (ADR 0010) |
| Container build, CI with lint/type/coverage | `Dockerfile`, `.github/workflows/` | Working |

## Next, in dependency order

### 0. Full-discharge segment detection — now the largest open quality item

Three of eight admitted CALCE cohorts (Types 5 and 6) rest on a **partial-cycle
reference** of ~0.26–0.42 Ah, roughly a quarter of nominal, because those
protocols cycle partially by design. Their SOH tracks relative fade of a
repeated partial cycle — a real signal, but not absolute state of health.
Type 3 is excluded outright, and CX2_3 needed a fifth screen criterion.

The fix is to detect full discharges directly — contiguous negative-current
runs terminating at the voltage cutoff — rather than trusting `Cycle_Index`.
`telemetry/cycles.py` already does exactly this for CAN logs; the work is
applying it at CALCE's scale and re-deriving the target from it.

This would recover four cells, remove a caveat currently attached to three
cohorts, and is the only route to an *absolute* SOH target on those protocols.

### 0b. CADEX loader — the only route to testing the Arrhenius model

Four cells are refused at load as CADEX-format exports (tab-separated, mV/mA,
`Pgm cycle` rather than `Cycle_Index`). One of them is **CX2_4**, the single
cell in either archive cycled across 25/35/45/55 °C — the only within-protocol
thermal axis available anywhere in this project's data.

ADR 0007 records that an activation energy is not identifiable on NASA because
ambient temperature is collinear with protocol. CX2_4 is the one frame where
`assess_identifiability` could plausibly pass. Supporting CADEX means mapping
its schema against the cycler's documentation and confirming the capacity
column's units — the unit question is exactly why it is currently refused
rather than guessed.

### 1. Load CALCE CS2/CX2 as a fade target — **done**

Everything else in the scientific track is gated on this. The project's central
claim is currently established on one dataset family from one lab, and a
reviewer's first question will be whether it replicates.

The commensurability screen already says which targets are worth the download
(`python -m src.bms.adaptive feasibility`):

| Target | Usable axes | Verdict |
|---|---|---|
| `calce_cs2` / `calce_cx2` | cutoff voltage, depth of discharge, discharge rate | **FEASIBLE — do this one** |
| `calce_cx2_4_thermal` | + ambient temperature | Feasible but n=1 cell |
| `oxford_degradation` | internal resistance only | MARGINAL |
| `stanford_severson` | internal resistance only | MARGINAL |

The loader exists (`src/bms/io/load_calce_cycling.py`). What is missing is
wiring it into `adaptive.datasets.DatasetRegistry` as a fade target and running
the benchmark study against it. CS2's Type-1..Type-6 groupings supply the
cohort structure LOCO needs, on the depth-of-discharge axis rather than the
temperature axis — which is the point: **if the LOBO-to-LOCO collapse
reproduces on a different axis, in a different lab, it is a property of
protocol shift and not of NASA.**

### 2. Curve-level features

`severson_delta_q_variance` and `ica_peak_features` are implemented and tested
but report UNAVAILABLE, because the retained NASA frame carries cycle-level
aggregates rather than per-cycle voltage/capacity traces. CALCE's Arbin
exports do carry them. Loading (1) makes these runnable with no further
modelling work, and they are the two most-cited feature families in the field.

### 3. Re-run the Arrhenius test where it is identifiable

ADR 0007 records why an activation energy cannot be estimated from NASA:
ambient temperature is collinear with protocol, and capacity is not measured
at a common temperature. `assess_identifiability` encodes both checks.

CX2_4 was cycled at 25/35/45/55 C with separate thermocouple files, so the
thermal axis is genuinely varied *within* one cell. That makes it the only
available frame where the check can pass — on one cell, which characterises
the relationship without supporting a cell-level generalisation claim. Worth
running, worth reporting with that caveat attached.

### 4. Persistence

The API's fleet store is in-memory and non-persistent, which is a deliberate
choice (`src/bms/api/store.py`) rather than an oversight: with nothing
promoted through the gate, there was no model state worth durably storing.

That changes if (1) promotes something. The work is a repository layer behind
the existing store interface plus a schema for telemetry, twin snapshots, and
the model-version log that `adaptive/store.py` already maintains in memory. A
time-series store is the natural fit for the telemetry table specifically.

Doing this *before* (1) would build persistence for a system whose only
durable state is a decision log saying nothing was promoted.

### 5. Streaming ingest and horizontal scale

`src/bms/telemetry/pipeline.py` processes a log or a live bus for one vehicle.
Fleet scale means a queue, partitioning by vehicle, and windowed cycle
segmentation across message batches. Gated on (4), since streaming with
nowhere to write is a demo.

### 6. Observability

`/healthz` exists. Structured logging, request metrics, and drift monitoring
in production are not built. The drift detection needed already exists as a
domain component (`adaptive/cohort.py` classifies an operating point against
learned envelopes) — the missing piece is exporting it as a live signal rather
than a batch report, which is a genuinely small job once (4) and (5) land.

## Specification defects — not missing code

Two of the architecture diagram's stated targets cannot be met as written. No
amount of implementation fixes them; the specification has to change.

### `SOH Prediction Error < 3%`

**Met only under leave-one-cell-out.** Best figures anywhere in the project,
on CALCE CS2+CX2 (19 cells, 8 cohorts, ceiling 0.870):

| | LOBO | LOCO |
|---|---:|---:|
| `lstm` | **2.26%** | 15.68% |
| `gpr_matern` | **2.93%** | 10.06% |
| `random_forest` | 3.04% | **8.59%** |

Under the honest protocol-shift split the best is 8.59%, roughly 3× the
target. The target is reachable precisely where ADR 0008 and ADR 0010 show the
measurement is optimistic.

**Proposed restatement:** *SOH MAE ≤ 5% under leave-one-cell-out; ≤ 10% under
leave-one-cohort-out, reported as an interval over cohort draws (ADR 0010).*

### `RUL Estimation Accuracy within ± 20 cycles`

**Achievable on lab cells, impossible on a production pack.** NASA cells fade
at 0.251% SOH/cycle, so ±18 cycles follows from a 4.5% SOH error. A real EV
pack fades ~19× slower (roughly 20% over 1,500 cycles ≈ 0.013%/cycle), so
±20 cycles there requires **0.27% SOH accuracy** — below coulomb-counting
drift, which is typically 1–2%.

"±20 cycles on NASA" and "±20 cycles on a car" are not the same claim.

**Proposed restatement:** *RUL within ±20 cycles on lab-accelerated cells;
on production packs, expressed as a percentage of remaining life rather than
an absolute cycle count.*

### `3.1 Behavior Analysis → Usage Patterns (Daily/Weekly Trends)`

**Not implementable on any dataset held here.** Every frame is cycle-indexed;
there is no wall-clock axis, so daily and weekly aggregation has nothing to
aggregate over. NASA's timestamps cover test-rig time, not calendar usage, and
CALCE's Arbin exports carry test time only.

This needs telemetry with real timestamps — a fleet feed or the instrumented
rig — not a modelling change. Until then the box should be marked
*not applicable to lab-cycled data*.

## Explicitly not planned

**Wiring a model into the dashboard before it passes the gate.** The gate is
the product (ADR 0005). A dashboard that shows a number the gate rejected is
the failure mode this project was built to prevent.

**Replacing the rule-based severity score with a fitted one for its own sake.**
The rule score is labelled throughout as triage rather than measurement, and
that labelling is accurate. Swapping in a fitted model that has not cleared
LOCO would change what is displayed without changing what is known.
