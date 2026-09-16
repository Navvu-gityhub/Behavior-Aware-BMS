# ADR 0014 — Rated capacity is declared, not defaulted

**Date:** 2026-09-09
**Status:** Accepted
**Supersedes:** nothing. Closes the four defects the Part 4 hardware audit
recorded against the serial path.

## Context

The serial ingestion path was audited before any board was connected, on the
premise that a defect found by reading is cheaper than a defect found with a
cell on the bench. Four findings blocked a physical bring-up. Three were small.
One was not.

**The one that mattered.** `features.behavior_features.compute_behavior_flags`
took `rated_capacity_ah: float = 2.0`, and `telemetry/pipeline.py` called it
with no argument. That 2.0 is the denominator of C-rate, and C-rate is the
definition of both `aggressive_discharge_event` and `fast_charge_flag`, which
feed `stress_score` → `health_index` → `rul_cycles`. Nothing on the wire carried
a capacity; nothing in the output named the one being used.

It was correct only by coincidence: the NASA cells this project calibrated
against are 2 Ah, and so is the bench emulator's profile. Connect a 3.4 Ah
18650 — the common cell for a rig like this — and a 1 C discharge is scored as
1.7 C. Both flags fire on every row of the capture. The resulting risk score,
health index and RUL are not noisy; they are confidently wrong, and there is
nothing in the output to suggest it.

This was the only place on the serial path that silently substituted a value for
one it did not have. Every other gate on that path refuses instead, which is
what makes the substitution notable rather than merely unfortunate: the pattern
the module documents at length was violated in the one spot nobody was looking.

The other three:

- **Status records were checksummed by the firmware and never verified by the
  parser.** `parse_line` returned the status body directly while data and hello
  both split and verified. Operator-facing fault text arrived with a trailing
  `*1C` — visible in this project's own documented example — and a status line
  corrupted on the cable was repeated back as though the rig had said it.
- **`--capture` could only record the emulator.** It short-circuited before
  `--port` was examined, so a live capture could not be written to a file at
  all. A hardware result could therefore not be replayed, committed as a
  fixture, or checked by anyone who was not in the room.
- **The firmware/schema agreement test asserted two string literals.** Renaming
  a wire field left the schema id untouched, so the sketch would keep announcing
  `beacon.telemetry.v1` while sending channels the parser silently drops.

## Decision

**Capacity is declared on the wire, or supplied by the caller, or the capture is
not scored.**

`capacity_ah` is now an optional HELLO attribute, alongside `cell_id`,
`period_ms`, `device` and `firmware`. No schema version bump: it is the same
kind of additive optional metadata those are, and bumping would have made every
existing rig incompatible for a field none of them send.

Resolution order is **caller → rig → refuse**. Caller wins because the caller is
the one who can see the cell; a rig's `NOMINAL_CAPACITY_AH` is whatever was
compiled into the sketch, and a bench that swaps cells without reflashing is the
ordinary case. When an override disagrees with a declaration, the result records
both.

**The refusal lives at the scoring stage, not at the gate.** This was got wrong
once during implementation and is worth recording. Refusing early — beside the
schema and coverage gates — discarded the cycle measurements and the capacity
yield summary, neither of which divides by capacity. On a bring-up where the
metadata is the only thing missing, those are exactly the numbers you want to
see. So the refusal sits beside the incomplete-coverage refusal in
`score_telemetry_frame`, and everything upstream of behaviour scoring still
runs and is still reported.

**The declared value is bounded at 0.01–500 Ah.** The bound exists to catch
milliamp-hours declared as amp-hours, which would make every C-rate 1000× too
small so that nothing is ever flagged and the capture scores as a model of
gentle usage. The first draft of this bound was 10,000 Ah, which admitted every
mAh rating in circulation and caught nothing; the test written against the
stated intent is what exposed that.

The CAN path and the batch dataset path still take the module default, now named
`DEFAULT_RATED_CAPACITY_AH`. A DBC carries no capacity declaration and giving it
one is separate work. The asymmetry is recorded rather than papered over.

**Recording is a source decorator.** `RecordingLineSource` wraps any
`LineSource` and writes each line as it passes through, so a live port, a replay
and the emulator are all recordable by one mechanism, and the file on disk is
exactly the bytes the run scored — not a second session generated separately.
Lines are flushed per line, so a capture interrupted by a brown-out keeps
everything received up to that point.

## Consequences

**A rig with old firmware needs one argument.** `rated_capacity_ah=` on the
Python call or `--capacity-ah` on the CLI. This is deliberate friction: it makes
the assumption appear at the call site, in the diff, where a reviewer sees it.

**Existing captures on disk that declare no capacity now measure but do not
score.** They still report cycles, capacities and the yield summary. This is a
behaviour change to replay, and the correct one — those captures never had a
capacity, and the numbers they used to produce were computed against a 2.0 Ah
assumption nobody made deliberately.

**The C-rate basis is printed on every serial result**, with its provenance:

```
C-rate basis: 3.4 Ah (caller override; the rig declared 2 Ah and was overridden)
```

A reader of a stress score can now see the denominator it was divided by. This
is the point of the change; the refusal is only what happens when there isn't
one.

**A pre-existing insensitivity was found and pinned, not fixed.** Scoring the
same capture against 1.0 Ah instead of 2.0 Ah moves `avg_stress` from 4.5 to
33.1 and `aggressive_discharge_count` from 0 to 80, and leaves the battery-level
`risk_score` at exactly 29.0. The `RISK_TERMS` step functions cut at 50 and 100
respectively, so both changes happen inside a single step. That is a property of
the rule-based scorer, not of this change, and altering the cut points would
silently re-score every figure in `docs/calibration_report.md`. It is now
asserted by `test_the_risk_score_is_too_coarse_to_see_that_change`, so a future
change to those boundaries is a deliberate one. Whether the terms should be
continuous is open — see ADR 0003.

**What this does not do.** It does not make the serial path hardware-verified.
`SerialPortSource.lines()` has still never run against a physical board, and the
stub firmware still produces a complete Guardian report on a bare board with no
sensors attached, because its synthetic discharge phases register as complete
cycles. That remains the largest hazard in a first bring-up, and the only
defence is procedural: any claim of hardware verification must name which of the
four `read*()` functions were real.

## See also

- `src/bms/telemetry/serial_schema.py` — the wire protocol and `capacity_ah`
- `src/bms/telemetry/serial_pipeline.py` — resolution, and why enforcement moved
- `src/bms/features/behavior_features.py` — `DEFAULT_RATED_CAPACITY_AH`
- `tests/test_serial_telemetry.py` — every claim above, asserted
- `docs/hardware_integration.md` — the bring-up procedure
- ADR 0003 — the explainability layer, and the step-function question
