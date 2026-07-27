# CAN/DBC adapter

`src/bms/io/load_can_dbc.py` — decodes real CAN bus frames (described by
a DBC file) into the unified BMS schema, the same role
`load_nasa.py`/`load_calce.py` play for their datasets. This is the
concrete answer to "can this connect to a real EV's BMS" — see below for
exactly what it does and doesn't prove.

## What's real here, and what's an example

- **The decoder is general-purpose.** It uses `cantools` to parse any
  standard DBC file and decode any message it describes — it is not
  hardcoded to one vehicle.
- **The bundled example is real, not fabricated.** `dbc_examples/
  twizy_bms_1.dbc` is transcribed from the [Open Vehicles (OVMS) project's
  public documentation](https://docs.openvehicles.com/en/latest/components/vehicle_dbc/docs/dbc-primer.html)
  — an open-source project with real reverse-engineered CAN decoding for
  30+ production EVs. This specific message (CAN ID 0x155, a Renault
  Twizy's primary BMS status frame) is documented there as real vehicle
  protocol, not a hypothetical. See `dbc_examples/SOURCE.md` for full
  provenance.
- **Correctness is verified against that source, not just internally.**
  `tests/test_can_dbc.py::test_decode_matches_ovms_documented_example`
  decodes the exact raw bytes OVMS's docs use as their worked example and
  asserts the output matches OVMS's own stated result (58.25A, 69.98%)
  to the millidigit. That's a real correctness check against an
  independent, published source — not a test that just confirms the code
  agrees with itself.

## Why cantools instead of a hand-rolled decoder

DBC's big-endian ("Motorola") bit-numbering is a well-documented source
of subtle bugs — the bit-position semantics genuinely aren't intuitive
(see OVMS's own primer, which spends several paragraphs on it). Getting
it *almost* right would produce plausible-looking, silently wrong
decoded values — precisely the failure mode this project has caught and
fixed before (the NaN-as-healthy bug). `cantools` is an established,
actively maintained library built for exactly this; using it is the same
call as using `pandas`/`scikit-learn`/`statsmodels` instead of
reimplementing statistics from scratch.

## Honest evaluation: what this does and doesn't get you

**Proven:** the adapter pattern — CAN frame → decoded signals → unified
schema — works correctly end to end, against real vehicle protocol data,
with the same rigor as the rest of this codebase.

**Not proven, and not claimed:**

- **Broad OEM coverage.** This is one message type from one vehicle.
  IEEE DataPort's public multi-manufacturer EV CAN dataset makes the
  general problem concrete: it ships real raw CAN dumps from five EV
  manufacturers with *no* DBC files, because manufacturer DBCs are
  proprietary and not publicly released. Covering another vehicle means
  either an OEM data-sharing relationship or the same kind of
  vehicle-by-vehicle reverse engineering OVMS's contributors have
  done — there's no general fix for "the DBC problem."
- **A complete-enough signal set to run the pipeline.** This message
  only carries current and SoC — no voltage, no temperature. The unified
  schema marks both as required, non-nullable fields.
  `tests/test_can_dbc.py::test_honest_integration_with_schema_validator_flags_missing_voltage_and_temp`
  confirms the *existing* validator correctly catches this and reports
  it, rather than the adapter silently proceeding on incomplete data or
  fabricating plausible-looking defaults. A real vehicle almost
  certainly transmits voltage and temperature on other CAN IDs — this
  primer just doesn't document them — so this is a gap in available
  public documentation for this one example, not a structural limit of
  the adapter itself.
- **Validated numbers for whatever vehicle you connect this to.** Even
  with a complete signal set, `docs/final_report.md` Sections 4.3–4.6
  already show the calibration hasn't generalized on NASA's lab cells —
  the easiest case tested so far. A real EV pack (hundreds of cells,
  active thermal management, real driving cycles) is a harder version of
  a problem not yet solved on an easier one. Wiring up real telemetry
  doesn't change that; recalibration is still the deeper open problem
  (see `docs/final_report.md` Section 6, item 1).
- **Live bus reading.** This decodes frames handed to it (e.g. from a
  logged `.dbc`+data pair, or a CSV export). It doesn't listen to a
  physical CAN interface — that would be `python-can`'s job, layered on
  top of this decoding logic, and hasn't been built.

## Reproducing

```bash
python -m pytest tests/test_can_dbc.py -v
```
