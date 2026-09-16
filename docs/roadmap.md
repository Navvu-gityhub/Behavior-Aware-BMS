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
| Serial telemetry ingestion + emulated rig | `src/bms/telemetry/serial_*.py` | Working — untested against a physical board |
| Rated capacity declared on the wire; live capture recording | `serial_schema.py`, `serial_source.py` | Working (ADR 0014) — closes the four defects blocking a bring-up |

## Next, in dependency order

### 0aa. Remove the C-rate default from the CAN and batch paths — **open**

ADR 0014 removed the silent 2.0 Ah C-rate denominator from the serial path: a
rig declares `capacity_ah` in HELLO, or the caller supplies it, or the capture
is measured but not scored. `DEFAULT_RATED_CAPACITY_AH` survives for the CAN
path and the batch dataset path, which still take it silently.

It is defensible there and not merely unfinished — the dataset loaders describe
cells the constant is correct for, and a DBC carries no capacity declaration to
read — but it is the same defect waiting for the first pack that is not 2 Ah. A
DBC has no standard capacity signal, so this needs a per-vehicle registry entry
rather than a wire change; `src/bms/io/can_vehicle_registry.py` is the place.

Do this before the CAN path is pointed at any real vehicle.

### 0. Full-discharge segment detection — **done** (ADR 0012)

`src/bms/io/calce_full_discharge.py`, `make calce-full-discharge`.

Segments sample-level telemetry into contiguous discharge runs via
`telemetry/cycles.segment_phases` and grades each against the cell's own voltage
cutoff *and* charge capability. References move from partial to physical
(CS2_5: 0.177 → 1.055 Ah; CS2_24: 0.367 → 1.101 Ah), all four previously
excluded cells are recovered, admissibility goes 19/23 → 22/22, and the target
noise ceiling rises 0.870 → 0.907.

Two things this item's original description got wrong, both corrected in
ADR 0012 and worth keeping visible:

- **"Terminating at the voltage cutoff" is not sufficient.** CS2_5's partial
  discharges reach the 2.7 V cutoff from a partially charged state, so 99.8% of
  them pass a cutoff test while moving a fifth of the cell's capacity.
- **The cutoff cannot be estimated as a quantile of terminal voltages**, because
  the partials then define it — 3.78 V for CS2_24, which certifies exactly the
  cycles the detector exists to reject.

It also refuted ADR 0009's "the learned models add enormously": on the absolute
target the age baselines reach LOCO 0.478–0.524 and beat every learned method
except `random_forest`.

Cost: a Type 5 or 6 cell yields 36–73 full discharges out of 5,000–7,000 cycles.
Absolute SOH on those protocols is inherently sparse.

### 0a. Estimator variance on the corrected target — **done** (ADR 0013)

The cohort-coverage sweep behind the project's one novel claim was run on
`calce_cycle_level.csv` — the `Cycle_Index` derivation that ADR 0012 refutes as
a measurement artifact. The sweep's nonlinear arm is `svr_rbf` and its baseline
is a smooth function of cycle number, so it was measuring exactly the
flexible-versus-smooth contrast that target distorts.

Re-run on the ADR 0012 full-discharge frame — same cells, same features, same
methods, only the target derivation different — via
`run_coverage_sweep.py --frames calce_full_discharge --out-prefix
coverage_sweep_full_discharge`.

On the coverage levels both runs draw fully, the LOCO interquartile range falls
from 0.892 to **0.296** and the between-method difference from -0.336 to
-0.110, while the **ratio between them holds at 2.66 versus 2.70**. The claim
that survives is the ratio: the LOCO estimate is several times noisier than the
difference it is used to adjudicate, on both derivations. The magnitudes were
inflated about threefold by the artifact.

What does **not** survive is the mechanism. On the artifact target `svr_rbf`
collapsed hardest (gap -1.037 against `age_linear`'s -0.096); on the corrected
target the ordering inverts (-0.203 against -0.460). "The flexible method
transfers worst" is the fifth ranking claim this project has had to withdraw.

ADR 0010 replicates: raw Spearman +0.126 (p = 0.217), **-0.002 (p = 0.982)**
once cell count is regressed out.

`coverage_sweep.csv` and the three figures pinned to it are unchanged.

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

**Partially unblocked.** The serial ingestion path
(`docs/hardware_integration.md`) is the instrumented-rig half of that. A
microcontroller rig streams samples at wall-clock intervals, so a capture from
one carries the axis every dataset here lacks.

The host-side timestamp is **now built**: `run_serial_pipeline(...,
captured_at=...)` adds a `timestamp` column as the supplied start instant plus
each record's elapsed `test_time_s`. It is not defaulted to "now" — replaying
last week's capture would then stamp it with today's date, inventing a calendar
rather than recording one — and a naive datetime is refused, because the same
local hour is two different instants across a DST change and daily aggregation
would mis-bucket silently. `test_time_s` remains what the scoring stages
integrate over, so a wrong `captured_at` shifts only the dates; a test asserts
capacity and health are unchanged by it.

Two things are still missing before the box can be ticked: a physical rig
actually logging (the software is untested against hardware), and a capture long
enough for daily and weekly aggregation to mean anything — which is days of
continuous logging, not a lab session.

## Explicitly not planned

**Wiring a model into the dashboard before it passes the gate.** The gate is
the product (ADR 0005). A dashboard that shows a number the gate rejected is
the failure mode this project was built to prevent.

**Replacing the rule-based severity score with a fitted one for its own sake.**
The rule score is labelled throughout as triage rather than measurement, and
that labelling is accurate. Swapping in a fitted model that has not cleared
LOCO would change what is displayed without changing what is known.
