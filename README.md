# BEACON — Behavior-Aware EV Battery Monitoring

A three-tier telemetry system that ingests battery data from a **CAN bus**, a
**USB serial rig**, or **research datasets**, converges all three on one schema,
and scores them through one shared pipeline into health, risk and
remaining-useful-life estimates — served over a REST API to a React dashboard.

Written in Python (FastAPI, pandas, scikit-learn), Node (Express), React (Vite)
and C++ (Arduino firmware). ~27,500 lines, 39 test modules, 9-job CI, containerised.

It ships two estimators that **fit nothing across cells** — SOH from charge in a
fixed voltage window, measured from **voltage, current and time alone** (1.7%
median error on 17 of 22 CALCE cells), and RUL from a cell's own fade trend
(within ±20 cycles 73% of the time near end of life) — because the fitted
models measurably do not transfer across protocols. Both run inside the main
pipeline and feed a plain-language **health report card**. See
[The research half](#the-research-half).

## Demo

A recorded walkthrough of the running system — pipeline, dashboard and the
serial rig.

<video src="https://github.com/Navvu-gityhub/Behavior-Aware-BMS/raw/main/docs/media/beacon_demo.mp4" controls muted playsinline width="100%"></video>

*(If the player does not load, [download or view `docs/media/beacon_demo.mp4`](https://github.com/Navvu-gityhub/Behavior-Aware-BMS/raw/main/docs/media/beacon_demo.mp4).)*

What the rig captures in [`data/interim/`](data/interim/) actually contain —
every figure below is computed from the committed files, not quoted from memory:

| Capture | Channels the board declared | Records | Cadence | Measured |
|---|---|---|---|---|
| [`rig_stage_b_voltage_verified.txt`](data/interim/rig_stage_b_voltage_verified.txt) | `t`, `v`, `i`, `soc` — INA219 up, LM35 refused | 83 over 82 s | 1.000 s, 0 gaps, monotonic | 3.9871 V (sd 1.9 mV); current at the ±0.4 mA noise floor |
| [`rig_demo.txt`](data/interim/rig_demo.txt) | `t`, `tc` — LM35 up, INA219 refused | 85 over 84 s | 1.000 s, 0 gaps, monotonic | 32.11 °C (sd 0.086 °C) |

Note the second column. The two sensors have **never been up at the same
time**, and neither capture pretends otherwise: the firmware probes each sensor
at boot, emits a status line naming what failed, and declares in its `HELLO`
only the channels it can actually supply. The host then scores the channels
present and refuses the rest by name. That is the refusal design of this
codebase running on real hardware against its own missing sensor — which is a
better test of it than a capture where everything worked.

The same records, decoded into a spreadsheet with per-channel summary statistics:
**[`BEACON_Hardware_Telemetry_Log.xlsx`](BEACON_Hardware_Telemetry_Log.xlsx)**.

---

## Quickstart

```bash
git clone https://github.com/Navvu-gityhub/Behavior-Aware-BMS.git
cd Behavior-Aware-BMS
pip install -r requirements-dev.txt

python main.py                        # end-to-end run → dashboard.html
python scripts/run_serial_demo.py     # rig → parser → gate → Guardian
python scripts/health_report.py --serial data/interim/rig_stage_b_voltage_verified.txt
python -m pytest tests/ -q            # the suite
```

With CALCE downloaded, `python scripts/health_report.py --calce
data/raw/calce/CS2/Type2/CS2_35.zip --upto-cycle 90` prints a real cell's report
card as it read at cycle 90, then what the lab measured afterwards — see
[A real report card](#a-real-report-card).

No hardware, no database, no API keys, no dataset download. Every command above
runs on a fresh clone. On Linux/macOS `make pipeline`, `make serial-demo`,
`make check` are shorthands for the same things; `docker compose up` brings all
three services (API `:8000`, gateway `:5000`, client `:5173`).

---

## The idea that shaped the codebase

Most monitoring systems, asked for a number they cannot compute, return a
plausible one. This one refuses, and says why:

```
$ python scripts/run_serial_demo.py --replay capture.txt

serial_rig: REFUSED
  frames read: 246, decoded: 244
  stages completed: ['read', 'parse', 'segment_cycles']
  2/2 discharges usable for SOH (100%); 0 partial. Largest observed discharge 1.976 Ah.
  REFUSED: Scoring skipped: no rated capacity is known for this cell, so
  C-rate cannot be computed. `aggressive_discharge_event` and
  `fast_charge_flag` are current divided by rated capacity, and the risk,
  health and RUL figures are all derived from them. A 3.4 Ah cell scored
  against an assumed 2.0 Ah would read 1.7 C while drawing 1 C and would
  trip both flags on every row of the capture. Declare `capacity_ah` in the
  rig's HELLO line, or pass `rated_capacity_ah`.
  244/244 data records accepted (100%); 1 status record(s).
```

*(Line-wrapped for width; otherwise verbatim.)*

Note what still ran. Segmentation and coulomb counting do not divide by
capacity, so they are reported; only the part that would be wrong is withheld.
Refusals are values, not exceptions — they are returned in the result, rendered,
and reflected in the process exit code so a bench script can branch on them.

This is not defensive coding as a style preference. It is the direct response to
a shipped defect: a missing temperature channel was being scored as *"temperature
is fine"*, because a NumPy comparison against `NaN` evaluates to `False`, so
every heat check silently passed for a pack nobody had measured. Nine of the
system's gates exist to make a specific version of that impossible.

**→ [`docs/architecture.md`](docs/architecture.md)** traces a request end to end
and explains the seams.

---

## What's in here

| | |
|---|---|
| **`src/bms/`** | 20 packages: telemetry ingest, feature extraction, risk, health, RUL, digital twin, conformal prediction, explainability, benchmarks |
| **`src/bms/api/`** | FastAPI service — scoring, telemetry ingest, fleet store |
| **`mern/`** | Express gateway + React/Vite client (fleet table, twin panel, telemetry replay, thermal map) |
| **`firmware/beacon_rig/`** | Board-agnostic Arduino sketch — ESP32 / ESP8266 / AVR, sensors behind four swappable stubs |
| **`tests/`** | 33 modules, including prose-to-evidence pinning and cross-artifact structural tests |
| **`docs/adr/`** | 14 architecture decision records, including the ones that were reversed |

### Engineering worth a look

- **[`docs/architecture.md`](docs/architecture.md)** — the transport→schema→scoring design, and why the scoring tail was *extracted* rather than copied.
- **[`.github/workflows/tests.yml`](.github/workflows/tests.yml)** — one job uninstalls the optional CAN dependency and re-runs the suite, because "it's optional" is otherwise an untested claim. Another reproduces the headline study end to end.
- **[`tests/test_reported_numbers.py`](tests/test_reported_numbers.py)** — every figure quoted in the docs is tied to the artifact that produced it. Re-run an experiment, forget to update a paragraph, fail the build.
- **[`docs/adr/0014-rated-capacity-on-the-wire.md`](docs/adr/0014-rated-capacity-on-the-wire.md)** — a silent 2.0 Ah default found by audit before any hardware was connected, and what removing it cost.
- **[`Dockerfile`](Dockerfile)** — multi-stage, compiler confined to the build stage, non-root UID, healthcheck that is explicitly *service* liveness and not battery health.

---

## Running it

| What | Plain Python (works anywhere) | `make` shorthand |
|---|---|---|
| Dependencies | `pip install -r requirements-dev.txt` | `make install-dev` |
| End-to-end run → `dashboard.html` | `python main.py` | `make pipeline` |
| API on `:8000`, docs at `/docs` | `python -m uvicorn src.bms.api.app:app --reload` | `make api` |
| Emulated rig → Guardian | `python scripts/run_serial_demo.py` | `make serial-demo` |
| Record a session while scoring it | `python scripts/run_serial_demo.py --capture s.txt` | `make serial-capture` |
| Test suite | `python -m pytest tests/ -q` | `make test` |
| Everything CI runs | `ruff check src tests scripts main.py && mypy && pytest tests/` | `make check` |
| All three services | `docker compose up` | `make docker-up` |

The serial demo needs no hardware and no `pyserial`: the emulator generates
byte-for-byte wire output — including an ESP32 boot banner — through the same
parser, checksum and coverage gate a physical board's output would take.
Swapping in real hardware changes one constructor call.

**→ [`docs/protocol_design.md`](docs/protocol_design.md)** — the three protocol
stacks, the wire format, error-detection analysis, and measured channel
utilisation.
**→ [`docs/hardware_design.md`](docs/hardware_design.md)** — the instrumentation
design: component selection, shunt sizing, pin assignment, power budget, and a
measurement error budget carried through to capacity accuracy.
**→ [`docs/hardware_integration.md`](docs/hardware_integration.md)** — wire
protocol, the bring-up record with measured figures, what the first real board
exposed, and what remains untested.

---

## A real report card

What the project set out to give a user — how healthy the battery is, how long
it has left, what usage is doing to it, what to do — for CALCE cell CS2_35 as it
would have read at cycle 90. The pipeline saw voltage, current and time only.
Abridged from `scripts/health_report.py`:

```
1. HOW HEALTHY IT IS
   91.0% state of health  [MEASURED]
   State: WARNING
2. HOW LONG IT HAS LEFT
   About 5 cycles until it falls to 90% of its early capacity.  [ESTIMATE]
3. WHAT YOUR USAGE IS DOING TO IT
   No temperature was measured, so heat exposure ... cannot be assessed.
4. WHAT TO DO
   Capacity fade is measurable. Keep monitoring; replacement is not yet indicated.
   General industry guidance, NOT confirmed by this project's data: ...
5. WHAT THIS REPORT CANNOT TELL YOU
   - Scoring skipped: the feature layer requires channels this source does not supply ...

CHECK AGAINST THE LAB  (the card above was not shown any of this)
   Cycler-measured capacity at cycle 90: 90.6% of initial; the card said 91.0% (+0.5 points)
   The cell actually crossed 90% at cycle 146: 56 cycles after cycle 90; the card said 5
```

The health figure is within half a point. The remaining-life figure is 51
cycles early — the conservative bias the RUL study measured at that distance
(median −32 cycles at 50–100 out), which is why the card prints its own
accuracy beside the number rather than the number alone.

Only temperature is offered as a usage finding, because it is the one
behaviour→ageing link that survived testing (7 of 7 NASA cells, direction
only). Fast-charge and state-of-charge advice is printed, and labelled as not
confirmed by this project's data.

---

## The research half

BEACON started as a battery-degradation study. The fitted models did not
transfer across protocols, that was measured rather than assumed, and the two
estimators that now work were built to route around it.

### What does not work, and how thoroughly

- The obvious target, `capacity_loss`, is **96% measurement noise** — an
  isotonic signal fraction of 0.044 is the *maximum attainable* R², which
  retroactively explains every "R² ≈ 0" result the project had recorded.
- **Every method loses skill under leave-one-cohort-out**, across four dataset
  framings, without exception. Selecting a model by leave-one-*cell*-out
  — what most papers report — picks a worse model under protocol shift.
- **No LOCO ranking is supportable at all.** Bootstrapping over folds, every
  interval spans zero and every interval overlaps every other one: XGBoost's
  LOCO R² is 0.459 with a 95% CI of [−2.803, 0.856]. Six ranking claims were
  withdrawn on that basis. The benchmark table now ships its intervals, because
  a project that diagnoses overclaiming from point estimates should not print
  them.
- **Curve features do not rescue it.** Severson's ΔQ(V) variance and ICA peak
  features were added to the same rows, same cells, same gate — one of four
  methods improved, none by more than fold-to-fold noise, and the collapse is
  unchanged. So it is not attributable to the features being usage aggregates.

### What works

Both estimators fit **nothing across cells**, which is why neither can suffer
the collapse above — there is no training cohort to transfer from.

| | result | scope |
|---|---|---|
| **SOH**, field method: voltage, current, time only | **1.7% median error** (95% CI 1.4–4.3%), worst 10.3% | 17 of 22 CALCE cells; 5 refused with a stated reason |
| **RUL** by extrapolating a cell's own fade trend | **within ±20 cycles, 73%** of the time | inside 25 cycles of end of life, against the reference a BMS can hold |

**The field method** ([`calce_field_soh/`](reports/metrics/calce_field_soh/field_soh_report.md))
gives the estimator only what a vehicle BMS has. It segments the discharges
itself, and it estimates the cell's resistance from the voltage drop when load
switches on, instead of reading the cycler's resistance column. On the same
cells:

| ohmic correction | cells measured | median error |
|---|---|---|
| none | 12 of 22 | 4.3% |
| cycler's resistance column (lab only) | 12 of 20 | 1.4% |
| **resistance from the load step (field)** | **17 of 21** | **1.7%** |

The step estimate reads higher than the cycler's (~0.165 vs ~0.099 Ω on CS2_35)
because it captures polarisation as well as pure ohmic drop — which is closer
to the sag the window actually sees, and why it measures more cells.

The five refusals are each explained by the code: four dynamic-profile cells
(CALCE Types 5/6) did 80–155 equivalent full cycles before any constant-current
discharge crossed the window, so no "as new" reference exists; one (CS2_7)
never rests before a discharge, so its resistance cannot be estimated. The
three worst measured cells (6.4–10.3%) are all Type 3, which switches rate six
times per cycle.

**Two figures were corrected downward by this work.** The earlier 3.3% SOH and
88% RUL were scored against a reference capacity taken from each cell's whole
life — the future, which no BMS has. Against the cell's own first cycles, RUL
near end of life is 73% within ±20; the earlier figures stay in their
artifacts as what they were.

**What it does not do yet:** partial discharges. Cut to 85%→15% state of
charge, the primary window can no longer be read at ~1C once the ohmic shift
is applied, and the gates refuse 18 of 21 cells rather than report. Before a
reference-timing gate was added, the same arm reported a confident 58% error
on CS2_38. A window placed inside the used range (4.00–3.80 V) measured 8
cells at 4.0% — labelled exploratory, because it was chosen after seeing the
first result.

SOH is the ratio of charge delivered between two fixed terminal voltages now
to the same window early in that cell's life. It survives partial discharge,
which is what a vehicle actually produces, and needs one reference measurement
of the same cell — which production BMS firmware already stores at manufacture.

**Two gates, and the first one was a bug I had misdiagnosed.** The estimator's
worst cell once reported 52% error while looking confident, and I attributed it
to loss of active material breaking the uniform-scaling assumption. **That was
wrong.** CS2_9's late cycles traverse the full 4.07–2.70 V range in ~100 samples
delivering 0.03 Ah, against 1.13 Ah in 3,690 samples early — they are *truncated
discharges*, not a faded cell, and my extractor was reading them as full ones. A
window cannot tell fade from a cut-short cycle, so each cycle is now required to
deliver at least 50% of the reference discharge. That alone took CS2_9 from 52%
to 9% error and CS2_3 from 23% to 10%, and both are now usable rather than
refused. Worst case across all cells: **52% → 10.5%.**

The second gate remains physical: a window ratio below 0.50 means a cell holding
under half its charge, which is scrap rather than degraded, so the reading is
withheld — never clipped into the plausible range. It refuses 3 of 16 cells.
Both checks need no ground truth, which is what lets them run in deployment.

Two literature-backed hypotheses were tested first and **both failed**:
constant-current *charge* curves (worse on four of five cells here, because
CALCE logs the charge leg sparsely) and rate-induced voltage depression
(refuted — the failing and working cells run at the same 0.50C).

**→ [`docs/capability_assessment.md`](docs/capability_assessment.md)** is the
synthesis: what a software layer over a BMS can and cannot do, argued from
these measurements. Its central number is that the 95% interval on one method's
LOCO score is 3.66 R² wide while the entire spread between the best and worst
of 14 methods is 2.06 — so model choice is unresolvable here, while two
data-conditioning fixes moved worst-case error 5×.

**→ [`reports/metrics/calce_voltage_window/`](reports/metrics/calce_voltage_window/)**
and **[`calce_rul_horizon/`](reports/metrics/calce_rul_horizon/)** carry the
per-fold numbers.

Fourteen ADRs record how those conclusions were reached and reversed. Two
document experiments that refuted their own premise.

**→ [`docs/final_report.md`](docs/final_report.md)** is the authoritative
account. **→ [`docs/project_history.md`](docs/project_history.md)** is this
README's previous life, with the full calibration narrative and every figure.

---

## Status

The pipeline, API, gateway, client, container build and CI are working.

The serial path now runs against a **physical board** — a NodeMCU ESP8266 with
an INA219 current/voltage monitor and an LM35 temperature sensor on a single
HONGLI ICR-18650 cell. `SerialPortSource.lines()`, previously the one untested
surface, has carried three captures end to end: board → wire protocol → parser
→ checksum → coverage gate → scoring → dashboard. Timing held at 1.000 s with
no dropped samples in any capture.

Read that result narrowly. It is **one cell, at rest, for under 90 seconds**,
with no load step and no charge/discharge cycle. It demonstrates that the
transport, the protocol, the gate and the refusal path work against real
silicon. It is *not* a validation of the health or RUL models, which were fitted
on cycled research cells and cannot be confirmed or refuted by a cell that was
never cycled. [`docs/hardware_integration.md`](docs/hardware_integration.md)
carries the full bring-up record and the remaining gaps.

**Estimators.** Field SOH and fade-extrapolation RUL run inside
`score_telemetry_frame` on every transport, before and independently of the
behaviour scoring, so a log with no temperature channel still gets its fade
measured. Where measured SOH exists and the log starts at beginning of life it
sets `battery_state`, with `state_basis` recording that it did; otherwise the
heuristic index is used and labelled as such. `compute_rul`'s hand-picked
weighting remains only as a labelled fallback for logs too short to extrapolate.

The conventional 80% end-of-life threshold is **not** validated: CALCE's
full-discharge trajectories end near 0.81 and one cell of 22 crosses it, so
every RUL figure here is measured at 0.90.

Not done, in dependency order: persistence (the fleet store is in-memory by
design), structured logging and metrics, a load-test harness, async serial
ingest with backpressure. See [`docs/roadmap.md`](docs/roadmap.md).
