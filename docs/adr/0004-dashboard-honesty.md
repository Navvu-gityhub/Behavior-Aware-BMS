# ADR 0004: The BEACON dashboard renders only computed values

**Status:** Accepted
**Date:** 2026-07-29

## Context

The approved BEACON visual design was mocked up with placeholder readings:
state of health 92.4%, state of charge 78%, remaining useful life 1,245
cycles, internal resistance 24.6 mΩ, voltage 3.92 V, and "up 1.3% vs last
week".

Several of those are quantities this pipeline **cannot produce**:

| Mockup value | Why it cannot be computed |
|---|---|
| Internal resistance | Not in the unified schema; no dataset in use records it per battery |
| Voltage / current (instantaneous) | Row-level telemetry only; there is no meaningful per-battery scalar |
| "up 1.3% vs last week" | Requires timestamped history the pipeline does not retain |
| State of health | Requires measured per-cycle capacity — NASA has it, the simulator does not |

Implementing the design literally would have meant either inventing these
numbers or wiring them to whatever nearby quantity was on the right scale.
The second is more dangerous than the first, because it produces a value that
is plausible, stable, and wrong.

## Decision

**Reproduce the layout faithfully. Bind every tile to a quantity the pipeline
actually produces. Render anything unsupported as an explicit unavailable
state rather than a substitute.**

Specifically:

1. **State of health is computed only from measured capacity**, relative to
   the cell's own initial capacity. Where capacity is absent, the tile reads
   "Not available" with the reason. It is **never** derived from
   `remaining_health` — that is a heuristic severity score on an unrelated
   scale, and presenting it as SOH would be a false measurement claim. This
   is the single easiest way this dashboard could state something untrue, so
   it is pinned by `test_state_of_health_unavailable_without_measured_capacity`.
2. **Absent series render as "no data", never as a flat line at zero.** A flat
   line reads as "measured and constant", which is a different and false claim.
3. **Provenance is displayed prominently.** A simulated run carries a banner
   stating the values are not measurements of any real battery. The label is
   derived from how `main.py` was invoked rather than left to the caller to
   remember, because mislabelling a simulated run as real is the most
   consequential error this tool could make in a presentation.
4. **The evidence panel is part of the product, not an appendix.** It shows
   rho = −0.27 for the health index against measured fade, the LOCO collapse,
   the 61% constant-term share, and this fleet's distinct-value count — on
   the same page as the headline numbers those caveats qualify.
5. **The four KPI tiles are relabelled** to what the pipeline computes:
   health index (severity, 0–100), degradation risk, remaining useful life,
   and state of health where available.

## Implementation notes

- Data preparation (`beacon_data.py`) is separated from rendering
  (`beacon.py`). Correctness lives in the mapping from pipeline output to
  displayed value, and separating it makes that testable without parsing HTML.
- Charts are inline SVG rather than the base64 matplotlib PNGs the previous
  dashboard embedded. Not cosmetic: SVG stays crisp at any zoom, the file is
  roughly an order of magnitude smaller, and series can be redrawn
  client-side when the user switches battery, which a baked PNG cannot do.
- The file remains fully self-contained with no external requests, per the V1
  "no server required" constraint. `test_dashboard_is_self_contained` asserts
  no `http://` or `https://` string survives into the output.

## Consequence

The dashboard looks emptier than the mockup on simulated data, and carries a
negative validation result on its own front page. That is the intended
outcome. A demo that hides the limitation is worth less than one that
survives a reviewer opening the report next to it.

## Alternatives rejected

- **Populate the mockup fields with the nearest available quantity.** Produces
  plausible, stable, wrong numbers — the hardest kind for a reviewer to catch.
- **Hide unsupported tiles entirely.** Loses the design, and silently omits
  the fact that the pipeline does not measure these things, which is itself
  information a reader needs.
- **Show sample/demo values with a footnote.** Screenshots outlive footnotes.
