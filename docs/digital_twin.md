# Battery Digital Twin

Implemented in `src/bms/digital_twin/twin.py`, exposed over HTTP by
`src/bms/api/`. This document describes what's actually there, not a
plan — the previous version of this file was a components/states sketch
written before any of it existed; see `docs/final_report.md` Section 4
for why that distinction matters in this project specifically (the
"Digital Twin" and "backend/API" tiles in the original architecture
diagram were aspirational, and an earlier external review of this repo
correctly flagged that gap — this closes it).

## What it is

A rule-based state machine layered over the existing pipeline outputs
(`health.health_index`, `rul.rul_estimation`, `guardian.guardian`). It
adds structure and a queryable API — it does **not** add a new predictive
model. Every number it reports (health_index, rul_cycles,
failure_likelihood) is the same hand-tuned, not-validated-against-measured-
degradation heuristic documented in `docs/final_report.md` Sections 4-6.
Treat this the same way the rest of the project asks you to treat the
health index and risk score: a transparent, auditable starting point, not
a scientific result.

## Twin states

```
NORMAL | MODERATE_RISK | HIGH_RISK | FAILURE_IMMINENT
```

This is a 1:1 relabel of `health_index`'s existing
`HEALTHY / WARNING / DEGRADED / CRITICAL` categories (same thresholds:
30/60/80), not a fifth independently-tuned threshold system. The codebase
already has two rule-based scoring systems with overlapping-but-different
thresholds (`risk.stress_score`'s row-level and battery-level scores — see
that module's docstring); adding a third for "twin state" would make the
system harder to reason about without adding real capability. If
validated calibration work ever replaces `health_index`'s formula, this
state machine updates automatically, since it derives from that same
field.

## What it tracks

- **Twin state** — current NORMAL/MODERATE_RISK/HIGH_RISK/FAILURE_IMMINENT
  classification for a battery, from the most recent pipeline run.
- **State transitions** — every time a battery's twin state changes
  between pipeline runs (or is observed for the first time), the fleet
  store records it. `GET /batteries/{id}` returns the full transition
  history for that battery.
- **Failure likelihood** — a monotonic transform of `health_index`
  (0-100 → 0-1). Named `failure_likelihood`, not `failure_probability`
  (this file's original draft used "Failure Probability") — it is not a
  calibrated statistical probability, and `docs/final_report.md` Section
  4 is specifically about why this project is careful about that
  distinction after the Guardian "Explainable AI" naming issue an earlier
  review caught.
- **Health timeline** — per-cycle stress score, SOC, and temperature for
  one battery from the most recent pipeline run's telemetry, for plotting.
  Ordered by `cycle` (the time axis every other module in this codebase
  already uses), not a wall-clock timestamp.

## What it does not do

- No persistence across API process restarts (see `src/bms/api/store.py`)
  — this is a demonstration fleet store, not a database.
- No prediction beyond what `health_index`/`rul_estimation` already
  compute. "Predictive Maintenance" in the original sketch describes the
  `replacement_policy` field (`REPLACE`/`PLAN_SERVICE`/`MONITOR`/`NORMAL`,
  from `rul.rul_estimation`), which was already implemented before this
  module existed — the digital twin surfaces it, it doesn't add it.
- No support for the applications listed in the original sketch (Cloud
  BMS, MQTT/ROS/CAN gateways, etc.) — the API layer (`docs/api.md`) is
  REST/HTTP only. The domain module (`src/bms/digital_twin/`) has no
  transport dependency, so adding another front end means writing a new
  thin adapter, not touching this module.

## Reproducing

```bash
python -m pytest tests/test_digital_twin.py -v
```
