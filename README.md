# BEACON — Behavior-Aware EV Battery Monitoring

A three-tier telemetry system that ingests battery data from a **CAN bus**, a
**USB serial rig**, or **research datasets**, converges all three on one schema,
and scores them through one shared pipeline into health, risk and
remaining-useful-life estimates — served over a REST API to a React dashboard.

Written in Python (FastAPI, pandas, scikit-learn), Node (Express), React (Vite)
and C++ (Arduino firmware). ~27,500 lines, 33 test modules, 7-job CI, containerised.

```bash
git clone https://github.com/Navvu-gityhub/Behavior-Aware-BMS.git
cd Behavior-Aware-BMS
pip install -r requirements-dev.txt

python main.py                        # end-to-end run → dashboard.html
python scripts/run_serial_demo.py     # rig → parser → gate → Guardian
python -m pytest tests/ -q            # the suite
```

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
protocol, bring-up procedure, and an honest account of what has never been run
against a board.

---

## The research half

BEACON started as a battery-degradation study and the modelling work is real,
but the results are largely **negative**, and that is the interesting part.

- The obvious target, `capacity_loss`, turned out to be **96% measurement
  noise** — an isotonic signal fraction of 0.044 is the *maximum attainable* R²,
  which retroactively explains every "R² ≈ 0" result the project had recorded.
- **Selecting a model by leave-one-cell-out cross-validation picks a worse model
  under protocol shift.** This is the project's one durable claim: across four
  dataset framings, every method loses skill under leave-one-cohort-out, without
  exception.
- **Every method-ranking claim has been withdrawn, five times over.** Rankings
  moved by varying only the target derivation, the cohort coverage and the
  feature set — never the model. The instability is the finding.

Fourteen ADRs record how those conclusions were reached and reversed. Two
document experiments that refuted their own premise.

**→ [`docs/final_report.md`](docs/final_report.md)** is the authoritative
account. **→ [`docs/project_history.md`](docs/project_history.md)** is this
README's previous life, with the full calibration narrative and every figure.

---

## Status

The pipeline, API, gateway, client, container build and CI are working. The
serial path is exercised end to end by an emulator and **has never run against a
physical board** — `SerialPortSource.lines()` is the one untested surface, and
[`docs/hardware_integration.md`](docs/hardware_integration.md) says so rather
than implying otherwise.

Not done, in dependency order: persistence (the fleet store is in-memory by
design), structured logging and metrics, a load-test harness, async serial
ingest with backpressure. See [`docs/roadmap.md`](docs/roadmap.md).
