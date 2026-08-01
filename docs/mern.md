# MERN layer (React + Express gateway)

`mern/` — a React + Express layer in front of the existing, unmodified
Python pipeline. This is deliberately **not** a rewrite of the pipeline
into JavaScript. See "Why not full MERN" below before assuming Mongo
belongs here.

## Architecture

```
Browser
  │
  ▼
React (Vite dev server, :5173)
  │  fetch('/api/...')  -- same-origin in dev via Vite's proxy
  ▼
Express gateway (mern/server, :5000)
  │  forwards to the Python service, unchanged
  ▼
FastAPI (src/bms/api/app.py, :8000)
  │
  ▼
The existing, tested pipeline (health_index, risk_score, rul_estimation,
guardian, digital_twin) -- zero lines of this changed for this layer.
```

The gateway (`mern/server/src/`) has exactly one job: translate HTTP
calls from the React client into calls to the Python service and back.
`pythonClient.js` is the only place that talks to Python; every route in
`routes/batteries.js` is a thin forward. If a route's behavior is wrong,
the fix almost certainly belongs in `src/bms/api/app.py` or the pipeline
modules it calls, not here — see `pythonClient.js`'s docstring.

## Why not full MERN (no MongoDB, pipeline logic stays in Python)

Two things this project has that don't come along for free with a
JS rewrite:

- **The research/calibration scripts** (`scripts/fit_horizon_regression_model.py`,
  `scripts/fit_mixed_effects_model.py`) use `statsmodels`' `MixedLM` (REML
  optimization, variance-component diagnostics) and `scipy.stats`. Node's
  statistics ecosystem has no comparably mature equivalent for mixed-effects
  modeling — reimplementing Section 4.6 of `docs/final_report.md` in JS
  would mean either hand-rolling MLE optimization or trusting a much
  less-tested library. These are also offline, run-once scripts, not part
  of the live service, so there's no actual reason they need to be in the
  same language as the web app.
- **The live pipeline's state** already lives in Python
  (`src/bms/api/store.py`'s `FleetStore`). With Python still owning that
  state, there's no honest role for MongoDB — adding it here would mean
  picking a database because the acronym asked for one, not because
  anything in this architecture needs a document store. If persistence
  across restarts is ever needed, the right question is "should the
  *Python* service persist to something" (Postgres, SQLite, Mongo —
  whichever fits), not "does the gateway need its own database."

What *did* move to JS: routing, request validation, and response shaping
(`mern/server/`), and the entire UI (`mern/client/`) — the parts that are
genuinely presentation/transport concerns, where React and Express are a
reasonable, ordinary choice.

## Running it

Three processes. One command for all of them:

```bash
cd mern
npm run install-all   # first time only
npm run dev
```

This starts, in one terminal with labeled/colored output: the Python
FastAPI service (`:8000`, via `uvicorn --reload`), the Express gateway
(`:5000`), and the Vite dev server (`:5173`). Open `http://localhost:5173`.

Or run each separately (useful for debugging one layer in isolation):

```bash
# Terminal 1 -- from the repo root
uvicorn src.bms.api.app:app --reload

# Terminal 2
cd mern/server && npm run dev

# Terminal 3
cd mern/client && npm run dev
```

## Testing

```bash
cd mern/server && npm test
```

7 tests, all against a fake Python backend (an in-process `http.Server`
started in the test file) rather than the real one — these tests verify
the gateway's *own* logic (forwarding, error-status mapping: Python's 404
passes through as 404 with its real message, an unreachable Python service
becomes a 502, unknown gateway routes are 404 without ever reaching
Python). Re-testing `health_index`/`risk_score`/twin-state correctness
here would duplicate `tests/test_api.py` and `tests/test_digital_twin.py`
in the Python package for no benefit.

The React client has no component test suite yet — `npm run build` and
`npm run lint` (via `oxlint`) both pass clean, and the full three-service
chain has been exercised end-to-end with real HTTP calls (see this
project's development history), but no automated browser-level test
exists. Worth adding (Vitest + React Testing Library) before treating the
UI layer as verified the way the gateway and pipeline are.

## What's the same as the Python-only dashboard, what's different

Visually, deliberately similar — same dark instrument-panel design tokens,
same layout, same fleet table / detail panel / oscilloscope-style traces —
so it reads as "same product, different stack," not a different app.
Functionally identical, since both are clients of the same pipeline output;
this one just goes through Express instead of talking to FastAPI directly.
See `docs/api.md` for what `src/bms/dashboard/live_dashboard.html` (the
Python-served version) does differently.

---

## Telemetry, twin and transfer views (batch 11)

The React client gained five views. The fleet view is unchanged; the other four
are new and every one of them renders values the Python pipeline computed.

| View | Reads | Component |
|---|---|---|
| Fleet | `/batteries`, `/pipeline/simulate` | unchanged |
| CAN Replay | `/telemetry/replay`, `/telemetry/coverage` | `TelemetryReplay.jsx` |
| Digital Twin | `/telemetry/twin/{id}` | `TwinPanel.jsx` |
| Thermal | `/telemetry/thermal/{id}` | `ThermalMap.jsx` |
| Transfer Validation | `/transfer/feasibility`, `/datasets` | `TransferPanel.jsx` |

`GuardianPanel.jsx` renders inside the CAN Replay view rather than as its own
tab, because a Guardian explanation is only meaningful next to the run it
explains.

### The gateway had to be extended

The client reaches the backend only through `/api/*`, and `app.js` returns 404
for anything that does not match a route. Each Python endpoint is therefore
invisible to the UI until it is forwarded.

`mern/server/src/routes/telemetry.js` adds that forwarding. It is a second
router mounted alongside `batteriesRouter` on the same prefix, so no existing
route is modified; `test_mounting_a_second_router_does_not_shadow_the_fleet_routes`
asserts the fleet routes still resolve.

A blanket proxy would have been less code, and was rejected: it would expose
every future Python route to the browser by default, which is a wider surface
than this gateway should have. Query parameters are named per route for the
same reason.

### Why there is no router

Views switch on local state. The app had no router, and adding one to reach
four panels would introduce a dependency and a URL scheme for no functional
gain. If deep links become a requirement, `App.jsx`'s `view` state is the
single place that changes.

### `telemetryId` is deliberately separate from `selectedId`

`selectedId` tracks the simulated fleet; `telemetryId` tracks the battery a CAN
replay produced. Conflating them would make a fleet-table click silently
repoint the twin and thermal views at a battery that has no CAN run behind it.

### The pack visualisation shows what is measured

The pack figure draws 24 cells greyed out, with a legend splitting channels into
instrumented and not instrumented. This is ADR 0004 applied to the UI: the
unified schema carries **one** aggregate `temperature_c`, so colouring cells
individually would mean interpolating dozens of values from one measurement.

The same constraint shapes the thermal view. Its axes are cycle and progress
through the discharge, both measured. There is no per-cell axis, and the
backend's `resolution_note` is rendered verbatim rather than paraphrased so the
limitation travels with the view.

Channels listed as never instrumented — individual cell voltages, cell
temperatures, balancing currents — appear in the legend precisely because a
reader would otherwise assume a pack diagram implies them.

### Refusals are results, not errors

A replay that decodes 40,000 frames and then declines to compute SOH has
produced useful information. `TelemetryReplay` renders `refusals` as a
first-class block, and `fade_prediction` is shown as explicitly unavailable with
the backend's reason attached, so a viewer cannot mistake the rule-based
severity index for a calibrated forecast.

### Running the stack

```bash
# 1. Python API
python -m uvicorn src.bms.api.app:app --port 8000

# 2. Express gateway (proxies /api to the Python service)
cd mern/server && npm install && npm start

# 3. React client
cd mern/client && npm install && npm run dev
```

Verified end to end: 11/11 calls succeed through React → Express → FastAPI,
including a replay producing four cycles, a thermal timeline with 160 points,
twin history, and the 404 path.
