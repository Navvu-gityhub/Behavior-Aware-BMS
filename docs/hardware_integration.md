# Hardware integration

How to connect a microcontroller rig to BEACON, and what the software already
does while you have no hardware at all.

**Status: the software path is complete and tested end to end against an
emulated rig. It has never been run against a physical board**, because none is
attached to this environment. That limit is stated here rather than implied to
be verified; the section [What is genuinely untested](#what-is-genuinely-untested)
says exactly which surface it covers.

---

## Two transports, one pipeline

BEACON ingests telemetry over two transports. They converge on one unified
schema and then share every scoring stage.

```
  PRODUCTION PATH                        BENCH / DEMO PATH
  ---------------                        -----------------

  Vehicle BMS                            ESP32 / Arduino / any MCU
       |                                          |
       | CAN frames                               | USB serial, text lines
       v                                          v
  LiveBusSource                            SerialPortSource
  LogFileSource       (python-can)         LogFileLineSource
  MemorySource                             EmulatedRigSource  <-- no hardware
       |                                   MemoryLineSource
       | (timestamp, id, bytes)                   |
       v                                          | "BEACON1 D {...}"
  decode_frames(dbc, signal_map)                  v
       |                                   parse_stream  (serial_schema)
       |                                          |  validate units, ranges,
       |                                          |  checksum, required fields
       v                                          v
  check_signal_coverage(dbc)              coverage_from_channels(HELLO)
       |                                          |
       +--------------------+---------------------+
                            |
                            v
                 +---------------------------+
                 |   COVERAGE GATE           |   one gate, both transports
                 |   missing channel ->      |   refuses and names the
                 |   REFUSE                  |   consumer that breaks
                 +---------------------------+
                            |
                            v
                 unified schema DataFrame
                 cell_id, test_time_s, voltage_v,
                 current_a, temperature_c, soc
                            |
                            v
       +--------------------------------------------------+
       |  pipeline.score_telemetry_frame                   |
       |  ---------------------------------                |
       |   segment_phases    charge / discharge / rest     |
       |   measure_cycles    coulomb counting -> capacity  |
       |   behaviour features, stress, rolling, age        |
       |   risk -> health index -> RUL                     |
       |   Guardian report -> digital twin                 |
       +--------------------------------------------------+
                            |
                            v
                  API  /  dashboard  /  twin
```

The shared block is literally one function. `serial_pipeline` and `pipeline`
both call `pipeline.score_telemetry_frame`, and the test suite asserts they are
the same object, because a second segmentation or feature extractor would make
a disagreement between a CAN run and a serial run untraceable.

**CAN is the production-oriented path. Serial is the bench and demo adapter.**
A vehicle BMS publishes on a bus; a breadboard publishes lines over USB. Serial
does not replace the CAN path and is not intended to.

---

## The telemetry schema

This table is generated from `serial_schema.FIELDS`. A test fails if this file
and that definition drift apart.

| Wire field | Unified channel | Unit | Range | Required |
|---|---|---|---|---|
| `t` | `test_time_s` | s | 0 to 1e+09 | yes |
| `v` | `voltage_v` | V | 0 to 1000 | yes |
| `i` | `current_a` | A | -2000 to 2000 | yes |
| `tc` | `temperature_c` | degC | -40 to 150 | yes |
| `soc` | `soc` | % | 0 to 100 | yes |

### The two conventions that actually break integrations

Both produce records that pass every range check and mean the wrong thing.
Neither is catchable per record; BEACON catches both at capture level.

**Current sign — negative is discharge.** A rig with the opposite convention
streams flawlessly, yields no discharge phase, and therefore no capacity and no
state of health, while appearing to work perfectly. BEACON detects a capture
with charge phases and no discharge phases and names the reversed shunt as the
likely cause.

**SOC scale — percent, 0–100, not a 0–1 fraction.** The feature layer compares
SOC against 20.0 and 90.0, so a fraction would set `deep_discharge_flag` on
every row and `high_soc_flag` on none — a confident, wrong risk profile. A
fraction lies *inside* the declared 0–100 range, so the field check cannot catch
it; BEACON detects an SOC channel that never exceeds 1.0 while sweeping a real
span, and refuses.

Fix both in firmware. The unit belongs to the device that measured it.

### Line format

```
BEACON1 HELLO {"schema":"beacon.telemetry.v1","fields":["t","v","i","tc","soc"],"cell_id":"RIG_01","device":"esp32","period_ms":1000}*4A
BEACON1 S rig started*1C
BEACON1 D {"t":0.000,"v":4.1500,"i":-1.5000,"tc":25.000,"soc":100.00}*06
BEACON1 D {"t":1.000,"v":4.1404,"i":-1.4991,"tc":25.050,"soc":99.98}*3F
```

| Element | Meaning |
|---|---|
| `BEACON1` | Sentinel. Lines without it are ignored as noise — boot banners, bootloader chatter, leftover `Serial.print` debugging. You do not need to suppress those. |
| `HELLO` | Schema declaration, sent once at start-up. Lets coverage be checked *before* decoding, the way a DBC does. Optional but preferred; without it coverage is inferred from what arrives, and the result says so. |
| `S` | Free-text device status. Reported, never scored. |
| `D` | One telemetry sample. Must carry every required field. |
| `*HH` | Optional XOR checksum over the record body, NMEA-style. Optional to compute, but validated strictly whenever present. |

A compact form is accepted for boards without room for JSON. Both decode to the
same record, and the test suite asserts that rather than assuming it:

```
BEACON1 D t=0.000 v=4.1500 i=-1.5000 tc=25.000 soc=100.00*2B
```

---

## Running the whole path today, with no hardware

```python
from src.bms.telemetry import EmulatedRigSource, RigProfile, run_serial_pipeline

source = EmulatedRigSource(profile=RigProfile(n_cycles=3, sample_period_s=60.0))
print(run_serial_pipeline(source, cell_id="RIG_01").render())
```

```
emulated_rig: SCORED
  frames read: 742, decoded: 732
  stages completed: ['read', 'parse', 'segment_cycles', 'score', 'twin']
  3/3 discharges usable for SOH (100%); 0 partial. Largest observed discharge 1.978 Ah.
  twin: 1 snapshot(s)
    RIG_01: NORMAL (health 29.0, failure likelihood 0.290, RUL 1541)
  rig: schema=beacon.telemetry.v1  fields=['t', 'v', 'i', 'tc', 'soc']  device=emulated-rig ...
  732/732 data records accepted (100%); 8 non-BEACON line(s) ignored; 1 status record(s).
```

Or from the command line:

```
python scripts/run_serial_demo.py                    # emulated rig, end to end
python scripts/run_serial_demo.py --ports            # list serial ports
python scripts/run_serial_demo.py --port COM5        # a real board
python scripts/run_serial_demo.py --capture out.txt  # record a session
python scripts/run_serial_demo.py --replay out.txt   # replay it
```

The emulator emits **wire-format text**, parsed by the same parser a real board's
output goes through — including a reproduction of an ESP32 boot banner, so the
sentinel filter is exercised on every run. It would have been less code to emit
records directly; that would also have meant the parser, the checksum, the schema
validation and the coverage gate were bypassed in exactly the path the tests
exercise, and the first real board would be the first thing that ever tested
them. Replacing the emulator with hardware **removes** a code path rather than
adding one.

---

## Connecting an ESP32 or Arduino later

Nothing in BEACON changes. The steps are:

**1. Flash the reference firmware.** `firmware/beacon_rig/beacon_rig.ino`
compiles as-is on ESP32, ESP8266 and AVR and emits a synthetic profile, so you
can verify the wire format on a bare board with no sensors attached.

**2. Confirm the format.** Open a serial monitor at 115200. You should see
`BEACON1 HELLO ...` followed by `BEACON1 D ...` lines. Close the monitor — only
one program can hold the port.

**3. Point BEACON at it.** This is the entire integration:

```python
from src.bms.telemetry import SerialPortSource, run_serial_pipeline

source = SerialPortSource(
    name="bench_rig", port="COM5", baudrate=115200, duration_s=120,
)
print(run_serial_pipeline(source).render())
```

`SerialPortSource` needs `pyserial` (`pip install pyserial`). The emulated path
does not, so the demo runs without it.

**Record when the capture started.** The rig's `t` is seconds since boot, so the
wire carries elapsed time only. Pass the host's wall-clock start instant to get
a calendar axis — the thing every dataset in this project lacks, and the thing
daily and weekly usage aggregation needs:

```python
from datetime import datetime, timezone

result = run_serial_pipeline(
    source, captured_at=datetime.now(timezone.utc),
)
result.telemetry["timestamp"]   # start instant + each record's test_time_s
result.has_calendar_axis        # True
```

Three deliberate constraints:

- **It is never defaulted to "now".** Replaying a capture recorded last week
  would otherwise stamp it with today's date and place a week-old session on
  the wrong days. Without the argument the frame carries elapsed time only and
  the rendered result says so.
- **A naive datetime is refused.** Across a DST change the same local hour is
  two different instants, so "usage by hour of day" over a multi-day capture
  would mis-bucket an hour of samples with nothing to indicate it. Pass an
  aware datetime.
- **It labels, it does not measure.** `test_time_s` remains what coulomb
  counting integrates over. A wrong `captured_at` shifts the dates and leaves
  capacity and state of health untouched, which the test suite asserts.

For a replay, pass the instant the *original* session ran, not the instant of
the replay. A capture file does not record it, so it is omitted rather than
guessed.

**There is no default port, and no auto-detect.** On a laptop with a Bluetooth
serial device or two boards attached, auto-selecting would silently read the
wrong one and report telemetry from a device nobody chose. `available_ports()`
lists candidates; it never chooses.

**Declare the cell's rated capacity.** `NOMINAL_CAPACITY_AH` in the sketch is
sent in the HELLO line and becomes the denominator of every C-rate BEACON
computes, which is how `aggressive_discharge_event` and `fast_charge_flag` are
defined. A rig that declares nothing is **refused** rather than assumed to be
2.0 Ah:

```
Scoring skipped: no rated capacity is known for this cell, so C-rate cannot
be computed. ... Declare `capacity_ah` in the rig's HELLO line, or pass
`rated_capacity_ah`.
```

The refusal catches an omission. It cannot catch a *wrong* value, so this is the
one number in the sketch worth checking twice: a 3.4 Ah 18650 left at the
default 2.0 reads 1.7 C while drawing 1 C, sets both flags on every row, and
produces a confident, wrong stress score, health index and RUL. A 3400 mAh cell
is `3.4f`, not `3400.0f` — the latter is rejected as a unit mistake.

To override a rig's declaration without reflashing — the normal case on a bench
that swaps cells — pass `rated_capacity_ah=3.4` (or `--capacity-ah 3.4`). The
result records which value won and where it came from:

```
C-rate basis: 3.4 Ah (caller override; the rig declared 2 Ah and was overridden)
```

**Record the session while you score it.** `--capture` wraps whatever source the
run is using, so the file on disk is exactly the bytes that produced the result:

```
python scripts/run_serial_demo.py --port COM5 --capacity-ah 3.4 \
    --duration 600 --capture data/interim/rig_bringup.txt
python scripts/run_serial_demo.py --replay data/interim/rig_bringup.txt
```

The two runs must agree. A bring-up that cannot be replayed is a result only the
people in the room can check, so committing the capture is what turns a hardware
session into evidence. Lines are flushed as they arrive, so a capture cut short
by a brown-out still holds everything received up to that point — which is
exactly what a post-mortem needs.

**4. Replace the sensor stubs, one at a time.** In the sketch, only
`initSensors`, `readVoltageV`, `readCurrentA`, `readTemperatureC` and
`readSocPercent` touch hardware. No pin number, sensor part number or I²C
address appears anywhere above them. Everything else stays untouched.

### A rig that would work

Nothing in the software depends on these parts — this is a worked example, not a
requirement.

> **For the full instrumentation design** — component selection and its
> justification, shunt sizing, pin assignment, power budget, and a measurement
> error budget carried through to capacity accuracy — see
> [`hardware_design.md`](hardware_design.md). The table below is the shopping
> list; that document is the engineering.

| Measurement | Part | Note |
|---|---|---|
| Current | INA219 / INA226 (I²C, ≤26 V) or ACS712 (hall-effect) | Gives you voltage too, on the INA parts. Verify the sign under a known load. |
| Voltage | INA219 bus voltage, or a resistor divider on an ADC pin | ESP32 GPIO is 3.3 V and **not** 5 V tolerant. Size the divider so the pin sees ≤3.0 V at maximum pack voltage, and check with a multimeter before connecting. |
| Temperature | DS18B20 (1-Wire) or 10 kΩ NTC + divider | Tape it to the cell body, not to the tab. A disconnected DS18B20 reads −127 °C, which falls outside the schema range and is rejected rather than averaged in. |
| Cell | One **protected** 18650 (TP4056+DW01 or equivalent) | One cell, not a series pack, for a first build. |

### Safety

The microcontroller is a **monitor**. It is never the protection device.

- Put real protection in hardware, in series with the cell — over-discharge,
  over-charge, short circuit. If the firmware crashes, the cell must still be
  safe.
- Never connect pack voltage to a GPIO pin directly.
- Lithium cells fail dangerously when shorted, over-discharged or over-charged.
  Charge on a bench, attended, on a non-flammable surface.
- Leave WiFi and BLE off for USB-serial operation. Their transmit spikes can
  brown out a weakly-supplied board, and a brown-out restarts `millis()` at
  zero — which BEACON refuses (see below) rather than silently mis-integrating.

---

## First bring-up: the exact procedure

Two stages, and **they must not be combined.** Stage A proves the transport with
nothing at risk; Stage B adds a cell. Doing both at once means a wiring fault and
a protocol fault are indistinguishable, with a lithium cell already connected.

Before either, verify the firmware compiles for your board:

```bash
arduino-cli core update-index --additional-urls \
  https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls \
  https://espressif.github.io/arduino-esp32/package_esp32_index.json
make firmware-compile          # arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all
```

CI compiles `esp32:esp32:esp32`, `esp8266:esp8266:nodemcuv2` and
`arduino:avr:uno` on every push, so a target that stops compiling fails the
build rather than the bench.

### Compilation record

The sketch was first compiled on **2026-09-16**. Before that date "compiles on
ESP32, ESP8266 and AVR" was a claim, not a tested fact. Toolchain:
`arduino-cli 1.5.1` (commit `01f3d4f2b`).

| Target | FQBN | Core | Flash | RAM | Result |
|---|---|---|---|---|---|
| Arduino Uno | `arduino:avr:uno` | `arduino:avr` 1.8.8 | 8,990 B / 32,256 (27%) | 460 B / 2,048 (22%) | ✅ |
| NodeMCU v2 | `esp8266:esp8266:nodemcuv2` | `esp8266:esp8266` 3.1.2 | 240,340 B / 1,048,576 (22%) | 28,372 B / 80,192 (35%) | ✅ |
| ESP32 Dev Module | `esp32:esp32:esp32` | `esp32:esp32` 3.3.11 | 272,324 B / 1,310,720 (20%) | 22,140 B / 327,680 (6%) | ✅ |

Compiled with `--warnings all`. **No warning originates in
`beacon_rig.ino`** — the only warnings emitted come from the Arduino AVR core's
own `new.cpp` (`-Wunused-parameter`), which is outside this project.

All three targets the sketch claims are verified. The ESP32 build leaves
**305,540 bytes of RAM free**, so the Stage B sensor libraries
(`Adafruit_INA219`, `OneWire`, `DallasTemperature`) fit with room to spare.

Two figures worth noting for the AVR target, because they settle a question the
documentation had previously only speculated about:

- **1,588 bytes of SRAM remain free for locals** on an Uno. The 195-byte HELLO
  `String` and the ~64-byte data records fit comfortably, so the Uno is usable
  with the default JSON codec. Setting `EMIT_COMPACT 1` still reduces heap churn
  over a long capture and remains the safer choice for an all-day run.
- **27% of flash used**, so there is ample room for the sensor libraries Stage B
  adds (`Adafruit_INA219`, `OneWire`, `DallasTemperature`).

### Stage A — transport only, no cell

Nothing is connected but the board and a USB cable. **No battery, no sensors.**

1. **Flash** `firmware/beacon_rig/beacon_rig.ino` unmodified.
2. **Connect USB** and let the board enumerate.
3. **Open a serial monitor at 115200 baud.** Confirm you see the boot banner,
   then `BEACON1 HELLO {...}`, then `BEACON1 S rig started`, then a stream of
   `BEACON1 D {...}`. **Close the monitor** — only one program can hold the port.
4. **Find the port:** `python scripts/run_serial_demo.py --ports`. It is never
   chosen automatically.
5. **Ingest live, and record while doing it:**

   ```bash
   python scripts/run_serial_demo.py --port COM5 --duration 600 \
       --capture data/interim/rig_bringup.txt
   ```

6. **Verify** in the rendered result:
   - `accepted_fraction` ≥ 0.99
   - a HELLO was seen (`coverage_inferred` is False)
   - zero time reversals
   - the `C-rate basis` line names the capacity the sketch declared
7. **Replay the capture** and confirm it reproduces the live numbers:

   ```bash
   python scripts/run_serial_demo.py --replay data/interim/rig_bringup.txt
   ```

8. **Commit the capture.** A bring-up that cannot be replayed is a result only
   the people in the room can check.

**Stage A passes when steps 6–7 hold.** It proves the transport, the framing,
the checksum, the schema handshake and the whole scoring path. It proves nothing
about measurement — see the trap below.

### Stage B — sensor integration

Only after Stage A passes. Wire per [`hardware_design.md`](hardware_design.md)
§2 and §5.3.

> **The monitoring firmware is not a substitute for hardware protection.** The
> MCU is a monitor. Protection must be in hardware, in series with the cell
> (TP4056 + DW01 or equivalent), so that the cell stays safe if the firmware
> hangs, crashes or is mid-flash. Use one **protected** cell, never a series
> pack, and never leave a charge unattended.

1. Replace the four `read*()` stubs **one at a time**, verifying each in a
   serial monitor before adding the next.
2. **Set `NOMINAL_CAPACITY_AH` to the cell's actual rating** — `3.4f` for a
   3400 mAh cell, not `3400.0f`. The host refuses a rig that declares nothing,
   but it cannot detect a value that is merely wrong.
3. **Confirm current reads negative under load** before any capture. This is
   the single most common integration fault and the cheapest to check.
4. Obtain the reference capacity from an independent charger/discharger
   **first**, before looking at BEACON's figure.
5. Run one full constant-current discharge to cutoff, then a charge, supervised.

---

## Physical acceptance checklist

The first real-cell test passes only if **every** applicable row passes. This is
the boundary between "the software is ready" and "the hardware works" — the test
suite establishes the former and cannot establish the latter.

| # | Criterion | How to check | Pass |
|---|---|---|---|
| 1 | Current sign correct | Under load, `current_a` is negative | ☐ |
| 2 | SOC in percent | `soc` spans well above 1.0 over a discharge | ☐ |
| 3 | Temperature responds to load | `temperature_c` rises under current, not a stuck constant | ☐ |
| 4 | Capacity explicitly transmitted | `C-rate basis: <X> Ah (declared by the rig...)` and X matches the cell | ☐ |
| 5 | At least one discharge detected | `result.cycles` is non-empty | ☐ |
| 6 | `is_complete` behaves | The full discharge is marked complete; a deliberate partial one is not | ☐ |
| 7 | Capacity vs independent reference | Compare BEACON's `capacity_ah` against the bench reference | ☐ |
| 8 | **Capacity error ≤ 5%** | `abs(beacon - ref) / ref ≤ 0.05` | ☐ |
| 9 | Accepted fraction ≥ 0.99 | From the rendered stats, over a ≥10 min capture | ☐ |
| 10 | Zero time reversals | No time-reversal refusal in the result | ☐ |
| 11 | Raw capture saved | `--capture` file committed | ☐ |
| 12 | Replay reproduces the result | `--replay` gives identical cycles and health index | ☐ |
| 13 | Real sensors named | Record which of the four `read*()` were real | ☐ |

Row 8's threshold is set against the error budget in
[`hardware_design.md`](hardware_design.md) §8, which predicts ~1.2% capacity
accuracy dominated by shunt tolerance. A result outside 5% means something is
wrong beyond the predicted instrument error — start with the shunt sizing (§4)
and the `V_bus` sense point.

Row 13 is not optional bookkeeping. It is the only defence against the trap
below.

**What a passing run licenses you to say:**

> The serial ingestion path has been run against physical hardware (board X,
> sensors Y) and recovered discharge capacity within Z% of an independent
> reference, on one cell, in one session, over N cycles.

Not "hardware validated", and nothing about the CAN path. One cell in one
session is a bring-up, not a validation study.

---

## Where the serial path refuses

Refusals are the product, not a rough edge (ADR 0005). Each of these is a case
where producing a number would mean producing a wrong one.

| Refusal | Why |
|---|---|
| Rig does not supply a required channel | Same gate as a DBC without a temperature signal. A NumPy comparison against NaN is False, so an absent temperature would read as "not hot" for every row and yield a healthy score for a pack nobody measured. |
| No rated capacity is declared | C-rate is current divided by rated capacity, and both behaviour flags are defined by it. Assuming a cell size does not make the flags uncertain; it makes them confidently wrong for every cell that is not that size. Segmentation and capacity measurement still run and are still reported — they do not divide by it. |
| A capacity outside 0.01–500 Ah | Milliamp-hours declared as amp-hours. Every C-rate would come out 1000× too small, nothing would ever be flagged, and the capture would score as a model of gentle usage. |
| A status line whose checksum does not match | Status text is shown to an operator. A line mangled on the cable would otherwise be repeated back as though the rig had said it. |
| Declared schema id is not implemented | Identical field names can mean different things across a schema change. |
| SOC is really a 0–1 fraction | Passes the range check, means the wrong thing, would flag deep discharge on every row. |
| Time went backwards | A brown-out reset restarts the clock. Coulomb counting integrates over time, so the second session's charge would cancel against the first's. Sorting would hide it by interleaving two unrelated sessions. |
| Fewer than 50% of data records parsed | A wrong baud rate or mismatched firmware, not cable noise. Scoring the survivors would compute a health index from an arbitrary subsample. |
| No complete discharge cycle | Capacity is only comparable across equal depths of discharge. Inherited from the shared stages, not specific to serial. |
| Fade prediction | `AdaptiveCalibrator.score` refuses while nothing has passed the promotion gate. The serial path does not route around that any more than the CAN path does. |

Individual bad lines are **counted, not fatal** — serial is a lossy transport,
and aborting a capture on the first dropped byte would make the pipeline
unusable on real hardware for a reason unrelated to the battery. The counts are
reported, and the accepted fraction is what the refusal above is based on.

---

## What is genuinely untested

`SerialPortSource.lines()` — the `pyserial` port read itself — has never run
against a physical board. Nothing downstream of it is in that category: the
emulator drives the identical parser, gate, schema check and scoring stages, so
the untested surface is the port read, not the pipeline.

The software side of readiness *is* tested: `tests/test_hardware_readiness.py`
(`make hardware-check`) covers the wire contract, checksum integrity on both
data and status records, capacity propagation from HELLO to the C-rate
calculation, replay determinism against a committed fixture, and the
firmware↔schema contract. Those tests establish that the software is ready for a
bring-up. **They do not establish that any hardware works**, and none of them
may be cited as evidence for a row in the acceptance checklist above.

### The trap: a bare board produces a full report

The stub firmware **scores**. Its synthetic profile toggles between −1.5 A and
+0.75 A every 120 s, and `COMPLETE_CYCLE_FRACTION` is relative to the largest
discharge observed *for that cell*, so those phases register as complete cycles.
A board with no sensors attached therefore yields cycle measurements, a health
index and an RUL — a full Guardian report that looks exactly like success.

There is no way for BEACON to detect this: the records are well-formed, the
channels are all present, the values are all in range. The only defence is
procedural, so state it in any claim you make: **which of `readVoltageV`,
`readCurrentA`, `readTemperatureC` and `readSocPercent` were real.** A result
from a rig with three stubs and one sensor is not a hardware result.

What a first bring-up would most plausibly expose, in order:

1. **DTR reset behaviour.** Most USB-serial boards reset when the port opens, so
   the first capture may begin mid-line. The sentinel filter handles the boot
   banner that follows; `reset_input_buffer()` handles the partial line. Neither
   is verified against real timing.
2. **Baud mismatch.** Produces garbage that fails the sentinel check, so it
   surfaces as "no usable telemetry records" — which names the baud rate as a
   cause. Correct behaviour, unverified in the field.
3. **Sign and scale.** The two conventions above. The refusals exist; whether
   their wording is enough to fix a real rig quickly is not yet known.

---

## See also

- `src/bms/telemetry/serial_schema.py` — wire protocol and the authoritative field table
- `src/bms/telemetry/serial_source.py` — line sources, including the emulator
- `src/bms/telemetry/serial_pipeline.py` — gate, plausibility checks, and wiring to the shared stages
- `tests/test_serial_telemetry.py` — the refusals above, asserted
- `docs/telemetry_pipeline.md` — the CAN path
- `docs/can_dbc_adapter.md` — DBC signal mapping
