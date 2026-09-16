# Architecture

BEACON ingests battery telemetry from three transports, converges them on one
schema, scores them through one shared path, and serves the result over an HTTP
API to a React client. This document traces a request end to end and explains
the decisions that shaped the seams.

The organising constraint is stated once and applies everywhere below: **the
system refuses to produce a number it cannot justify.** Most of the design falls
out of that.

---

## The shape of it

```
                   ┌─────────────────────────────────────────┐
   USB serial ───▶ │  serial_schema  ·  parse + checksum      │
   (bench rig)     │                    + range validation    │
                   └──────────────────┬──────────────────────┘
                                      │
   CAN bus     ───▶ ┌─────────────────▼──────────────────────┐
   (vehicle)        │  sources  ·  DBC decode + coverage gate │
                    └─────────────────┬──────────────────────┘
                                      │
   CSV / MAT   ───▶ ┌─────────────────▼──────────────────────┐
   (NASA, CALCE)    │      unified telemetry frame            │
                    │   cell_id · test_time_s · voltage_v     │
                    │   current_a · temperature_c · soc       │
                    └─────────────────┬──────────────────────┘
                                      │
                    ┌─────────────────▼──────────────────────┐
                    │  score_telemetry_frame                  │
                    │  ─────────────────────────────────────  │
                    │  segment cycles → coulomb count →       │
                    │  behaviour flags → stress → risk →      │
                    │  health index → RUL → Guardian → twin   │
                    └─────────────────┬──────────────────────┘
                                      │
   ┌──────────────────────────────────▼───────────────────────────────┐
   │  FastAPI  :8000   ──▶   Express gateway  :5000   ──▶   React  :5173 │
   └──────────────────────────────────────────────────────────────────┘
```

Three transports, one frame, one scoring function. That is the whole system in a
sentence, and the next section is why it is arranged that way.

---

## One scoring path, extracted rather than copied

`telemetry/pipeline.score_telemetry_frame` is everything downstream of
acquisition: segmentation, coulomb counting, features, risk, health, RUL,
Guardian and the digital twin. All three transports call it.

It was extracted from the CAN pipeline when serial ingestion was added, and the
choice to extract rather than copy was the binding design constraint on the
serial module. A second segmentation or a second feature extractor would mean a
disagreement between a CAN run and a serial run over the same battery could not
be traced to either the data or the code — you would be debugging two
implementations of the same idea. With the tail shared, **a difference in output
is a difference in the telemetry**, which is a debuggable statement.

The cost is real and worth naming: every transport is coupled to one signature,
so a change that suits serial has to suit CAN. That coupling is the point.

### The seam is narrower than it looks

A serial line source is a Protocol with one method:

```python
class LineSource(Protocol):
    name: str
    def lines(self) -> Iterator[str]: ...
```

A pyserial port, a recorded `.txt`, a list in a test, and the deterministic
emulator are interchangeable. The demo path and the hardware path differ in
**one constructor call**.

This is why the emulator generates *wire-format text* rather than record
objects. Emitting objects directly would have been less code and would have
proved nothing: the parser, the checksum, the schema validation and the coverage
gate would all be bypassed in exactly the path the tests exercise, and the first
real board would be the first thing that ever tested them. The emulator even
emits an ESP32 boot banner, so the sentinel filter is exercised on every run.

Recording composes at the same seam:

```python
source = RecordingLineSource(inner=SerialPortSource(port="COM5"), path=out)
```

One decorator, and a live port, a replay and the emulator are all recordable —
and the file on disk is exactly the bytes that produced the result.

---

## Tracing a request

`POST /telemetry/serial/replay` with a capture file:

1. **`api/telemetry_routes.py`** validates the request against a Pydantic schema
   and calls `replay_serial_capture`.
2. **`serial_source.LogFileLineSource`** yields lines. Decode errors use
   `errors="replace"`, so one corrupt byte from a real cable becomes one
   rejected line in the statistics rather than an exception that discards the
   session.
3. **`serial_schema.parse_stream`** classifies each line — noise, hello, status,
   data — verifying the XOR checksum and validating every field against its
   declared range. Bad lines are **counted, not raised**: serial is lossy, and
   aborting on the first dropped byte would make the pipeline unusable on real
   hardware for a reason unrelated to the battery.
4. **`serial_pipeline.run_serial_pipeline`** applies the gates below, assembles
   the unified frame, and resolves the C-rate basis.
5. **`pipeline.score_telemetry_frame`** runs the shared tail.
6. The result — including every refusal — is serialised back through the
   gateway to the client.

`SerialDecodeStats` rides along the whole way. A capture that silently dropped
90% of its lines and scored the rest is, from its output alone, indistinguishable
from a clean one. The counts are what make that visible, and the pipeline
refuses below a 50% accepted fraction rather than scoring the remainder.

---

## Where it refuses, and why that is the architecture

Every gate below returns a *result carrying a refusal*, not an exception. A
refusal is a correct outcome — it is what the system knows — so it is data, it
is rendered, and the CLI's exit code distinguishes it for a bench check.

| Gate | What it prevents |
|---|---|
| Missing required channel | A NumPy comparison against NaN is `False`, so an absent temperature reads as "not hot" for every row and yields a healthy score for a pack nobody measured. This defect shipped once and is what the gate exists for. |
| Unimplemented schema id | Identical field names can mean different things across a schema change. |
| SOC that is a 0–1 fraction | Passes the range check and means the wrong thing. Caught at capture level, not per record — a fraction lies *inside* [0, 100]. |
| Reversed current sign | A rig reporting discharge as positive yields no discharge phases and no SOH while appearing to stream perfectly. Also invisible per record. |
| Time running backwards | A brown-out restarts `millis()` at zero. Coulomb counting integrates over time, so the second session's charge would cancel against the first's. Sorting the frame would *hide* it by interleaving two unrelated sessions. |
| <50% of records parsed | A wrong baud rate or mismatched firmware, not cable noise. |
| No rated capacity declared | C-rate is current ÷ rated capacity. A 3.4 Ah cell scored against an assumed 2.0 Ah reads 1.7 C while drawing 1 C and trips both behaviour flags on every row (ADR 0014). |
| No complete discharge cycle | Capacity is comparable only across equal depths of discharge, so partial cycles are excluded rather than scaled. |
| Nothing has passed the promotion gate | `AdaptiveCalibrator.score` refuses to predict fade. No transport routes around this. |

Two properties of this table are load-bearing:

**Range checks reject rather than clamp.** An out-of-range value is *evidence
that the reading is not what the schema says it is* — a disconnected thermistor
floats, an unpowered INA219 reads zero. Clamping produces a value that looks
measured and is not.

**Some errors are not expressible per record.** A 0–1 SOC fraction and a
reversed shunt both produce perfectly valid records that mean the wrong thing.
They are caught one level up, by looking at the shape of a whole capture. The
layering exists because the checks genuinely need different scopes.

### Refusing is not the same as withholding everything

Worth its own note, because the first implementation of the capacity gate got it
wrong. Segmentation and coulomb counting do not divide by capacity, so refusing
early discarded cycle measurements that were perfectly valid. The gate now sits
at the scoring stage: the capture is **measured and reported**, and only the
behaviour scoring is withheld. Refuse the narrowest thing that is actually
unknown.

---

## Service topology

| Service | Stack | Responsibility |
|---|---|---|
| `api` | FastAPI + uvicorn, :8000 | Scoring, telemetry ingest, fleet store |
| `gateway` | Express, :5000 | Client-facing aggregation and shaping |
| `client` | React + Vite, :5173 | Fleet table, detail, twin, replay |

`docker compose up` brings all three, with the gateway held on
`depends_on: condition: service_healthy` against the API's `/healthz`.

Two deliberate absences:

**No database.** The fleet store is in-memory (`api/store.py`). Adding Postgres
to the compose file would imply a persistence layer that does not exist; the
compose file says so in a comment rather than leaving a reader to discover it.
It is the next substantial piece of work.

**`/healthz` is service liveness and is emphatically not battery health.** The
container `HEALTHCHECK` polls it. The naming collision in a battery-monitoring
system is a real hazard and is called out at the definition.

---

## Testing strategy

33 test modules, and three kinds of test that are not the usual kind:

**Prose-to-evidence pinning.** `tests/test_reported_numbers.py` ties every
figure quoted in the documentation to the artifact that produced it. Re-running
an experiment and forgetting to update a paragraph fails the build. This is the
mechanism that makes the documentation trustworthy rather than aspirational, and
it caught a stale figure that had already been published.

**Cross-artifact structural tests.** The firmware sketch's declared field set is
compared as a *set* against the schema's `FIELDS`; the documented schema table
must equal `schema_table()`'s rendered output. Substring assertions were not
enough and had already let a defect through: a renamed field left the schema id
untouched, so the sketch would keep announcing a version it no longer spoke.

**Known-defect pins.** Some tests assert behaviour that is *not* desirable, with
a comment saying so — for instance that the risk score is too coarse to see a
7× change in average stress, because the step boundaries sit at 50 and 100.
Changing that would silently re-score every published figure, so it is pinned
rather than fixed, and a future change to it has to be deliberate.

CI runs seven jobs, of which three are unusual:

- **`no-optional-extras`** uninstalls `cantools` and re-runs the suite. Without
  it, "the CAN dependency is optional" is an untested claim that breaks the day
  someone moves an import to module scope.
- **`reproduce-study`** runs the headline benchmark end to end, so the project's
  central result cannot rot.
- **A Python-version floor matrix (3.10 and 3.12).** Testing only the newest
  version lets a declared floor rot silently.

---

## Decision records

`docs/adr/0001`–`0014` record the decisions that were expensive to reach,
including the ones that were reversed. Several document a hypothesis this
project *refuted* — ADR 0010's cohort-coverage sweep refuted its own premise,
and ADR 0012 withdrew a modelling claim after finding the target it rested on
was a measurement artifact.

They are worth reading in the opposite order to how they were written: the later
ones explain why the earlier ones were not enough.

---

## See also

- `docs/hardware_integration.md` — the rig, the wire protocol, the bring-up
- `docs/telemetry_pipeline.md` — the CAN path in detail
- `docs/api.md` — endpoint reference
- `docs/mern.md` — gateway and client
- `docs/roadmap.md` — what is not done, in dependency order
