# Hardware design

The instrumentation design for the BEACON bench rig: component selection and
its justification, signal-conditioning calculations, pin assignment, power
budget, and a measurement error budget carried through to the accuracy of the
state-of-health figure the rig produces.

> **Status.** This is a *design*, not a *build report*. Every calculation below
> is worked from datasheet figures and stated assumptions, and none has been
> measured on assembled hardware. Values marked **[verify]** must be checked
> against the datasheet revision of the part actually purchased before they are
> quoted in a submitted report — datasheet figures vary between revisions and
> between vendors of nominally identical breakouts. §9 lists what changes once
> the rig is built.

Companion documents: `hardware_integration.md` (wire protocol and bring-up
procedure), `architecture.md` (where the rig sits in the software system).

---

## 1. Requirement

The rig must measure one lithium-ion cell during charge and discharge, and
supply the five channels the BEACON schema requires, at a rate and accuracy
sufficient to recover discharge capacity.

| Channel | Range required | Why this range |
|---|---|---|
| Terminal voltage | 2.5 – 4.3 V | 18650 cutoff to full charge, plus margin |
| Current, signed | −3.5 – +2.0 A | 1 C discharge of a 3.4 Ah cell; 0.5 C charge |
| Cell temperature | 0 – 60 °C | Ambient to the upper limit of safe operation |
| State of charge | 0 – 100 % | Derived, not measured — see §6 |
| Elapsed time | monotonic, ms | Coulomb counting integrates over it |

The binding accuracy requirement is on **capacity**, not on any single channel.
`hardware_integration.md` sets the acceptance criterion at **within 5% of an
independent reference**, and §8 shows which component errors that budget
actually spends.

---

## 2. System block diagram

```
        ┌───────────────── cell under test ─────────────────┐
        │                                                   │
     ╔══╧══╗      ┌──────────┐        ┌─────────┐        ╔══╧══╗
     ║  +  ║──────┤  R_shunt ├────────┤ TP4056  ├────────║  −  ║
     ║18650║      │  0.1 Ω   │        │ + DW01  │        ║     ║
     ╚══╤══╝      └────┬─────┘        │protection│       ╚═════╝
        │              │              └─────────┘
        │         ┌────┴──────┐
        │         │  INA219   │   V_shunt → current
        └─────────┤  I²C 0x40 │   V_bus   → terminal voltage
                  └────┬──────┘
                       │ SDA / SCL
                  ┌────┴───────────────┐
   DS18B20 ───────┤   ESP32-WROOM-32   ├────── USB (5 V, CDC serial)
   1-Wire, GPIO4  │  radios disabled   │       115200 baud
   on cell body   └────────────────────┘       BEACON1 lines → host
```

Measurement is **high-side**: the shunt sits in the positive leg, so the cell's
negative terminal, the protection module and the ESP32 share one ground. A
low-side shunt would be simpler to level-shift but would put the cell's negative
terminal above system ground by the shunt drop, so the terminal-voltage
measurement would silently read low by exactly that amount — an error that grows
with current and therefore looks like extra internal resistance.

---

## 3. Component selection

### 3.1 Current and voltage — INA219 over the alternatives

| Option | Resolution | Verdict |
|---|---|---|
| **INA219** (I²C, 12-bit) | 10 µV shunt LSB → 100 µA at 0.1 Ω **[verify]** | **Selected.** Digital output, no ADC calibration, measures bus voltage on the same part |
| INA226 (I²C, 16-bit) | 2.5 µV shunt LSB **[verify]** | Better, and a drop-in upgrade if §8 proves insufficient |
| ACS712 (hall, analog) | ~66–185 mV/A, analog | **Rejected** — output goes to the ESP32 ADC, inheriting the nonlinearity in §3.2, and its zero-current offset drifts with temperature, which is fatal for coulomb counting (§8.3) |
| Shunt + op-amp | arbitrary | **Rejected** — requires offset trim and a precision reference; more error sources than the integrated part, for no gain at this scale |

The INA219 also removes the need for a separate voltage divider, which is the
decisive argument and is worth stating plainly:

### 3.2 Why no voltage divider

The obvious design reads terminal voltage on an ESP32 ADC pin through a
divider. A single cell reaches 4.2 V and the ESP32 ADC accepts 3.3 V, so a
divider of R1 = 10 kΩ, R2 = 22 kΩ gives

```
V_adc = 4.2 × 22/(10+22) = 4.2 × 0.6875 = 2.89 V      ✓ within range
I_divider = 4.2 / 32 kΩ = 131 µA                      ✗ permanent cell drain
```

That is rejected for three reasons, in increasing order of importance:

1. **It drains the cell continuously** — 131 µA is 3.1 mAh per day. Negligible
   against 3400 mAh, but it is a current the shunt does not see, so it is
   uncounted charge and therefore a small systematic capacity error.
2. **The ESP32 ADC is nonlinear**, appreciably so near the rails, and requires
   per-chip calibration to reach even ±1% **[verify against the ESP32
   calibration application note]**. The INA219's bus channel is a calibrated
   12-bit converter with a 4 mV LSB and needs none.
3. **It adds a resistor-tolerance error to voltage** that does not exist
   otherwise. 1% resistors give a divider ratio uncertain to ~1.4%, which is
   larger than the whole INA219 bus error.

The INA219 measures bus voltage up to 26 V directly. The divider is designed
above only to show it was considered and why it lost.

### 3.3 Temperature — DS18B20 over an NTC

| | DS18B20 | 10 kΩ NTC + divider |
|---|---|---|
| Output | Digital, 1-Wire | Analog → ESP32 ADC |
| Accuracy | ±0.5 °C, −10…+85 °C **[verify]** | Depends on divider tolerance, ADC linearity, and β accuracy |
| Linearisation | None needed | β equation or Steinhart–Hart in firmware |
| Failure mode | Reads −127 °C — **outside the schema range, so the host rejects the record** | Floats to a plausible mid-scale value that passes every check |

The failure mode is the deciding argument and it is a systems argument, not a
component one. `serial_schema.py` declares temperature valid over −40…150 °C and
rejects records outside it. A disconnected DS18B20 returns −127 °C and is
therefore *caught*. A disconnected NTC leaves the ADC pin floating, which reads
an arbitrary value inside the valid range, and the pipeline scores it as a real
measurement. Choosing the part whose failure is detectable is the same principle
the software applies throughout.

If an NTC is used anyway, the β equation is

```
1/T = 1/T₀ + (1/β)·ln(R/R₀)      T₀ = 298.15 K, R₀ = 10 kΩ, β ≈ 3950 K
```

with T in kelvin. Note the sensor must be linearised **in firmware**, not on the
host: the schema declares °C, and shipping raw counts would put a unit
conversion in a layer that cannot know the divider values.

### 3.4 Protection — TP4056 + DW01

Mandatory, and separate from the microcontroller. The MCU is a monitor; if its
firmware hangs, the cell must still be safe. The DW01 provides over-discharge
(~2.4 V), over-charge (~4.3 V) and over-current cutoff in hardware, in series
with the cell, with no software in the path.

---

## 4. Signal conditioning: shunt sizing

The shunt converts current to a voltage the INA219 can resolve. Sizing it is the
one genuine analog design decision in the rig, and it is a three-way trade
between range, resolution and self-heating.

The INA219's shunt channel is ±320 mV full scale at 12 bits, giving a 10 µV LSB
**[verify]**. With shunt resistance R_s:

```
I_max = 320 mV / R_s          I_LSB = 10 µV / R_s          P_shunt = I² · R_s
```

| R_s | I_max | I_LSB | P at 2 A | Verdict |
|---|---:|---:|---:|---|
| 0.1 Ω | 3.20 A | 100 µA | 0.40 W | **Selected** for ≤3 A |
| 0.05 Ω | 6.40 A | 200 µA | 0.20 W | For 1 C on a 3.4 Ah cell |
| 0.01 Ω | 32.0 A | 1.00 mA | 0.04 W | Wasteful here — 1 mA LSB for no needed range |

**Selected: 0.1 Ω, 2 W, 1%** — the value fitted to most INA219 breakouts, so no
rework is needed.

Three consequences worth stating, because each is a real constraint rather than
a formality:

**Discharge must stay below 3.2 A.** A 3.4 Ah cell at a full 1 C draws 3.4 A and
would saturate the shunt channel. Either discharge at 0.5 C (1.7 A, comfortably
inside range) or fit the 0.05 Ω shunt. **Saturation is not a graceful failure**:
the reading clips, coulomb counting under-integrates, and the measured capacity
comes out low with nothing to indicate why. Choose the rate before the run.

**Self-heating shifts the shunt value.** At 2 A the shunt dissipates 0.4 W. A
2 W resistor survives it, but its resistance drifts with temperature, and a
shunt reading high makes current read *low* — a systematic capacity error in one
direction. This is why §8 treats the shunt tolerance as a gain error rather than
noise.

**The shunt drop is subtracted from the cell's available voltage.** At 2 A,
0.1 Ω drops 200 mV, so the load sees 200 mV less than the cell terminal. The
INA219 measures bus voltage on the load side of the shunt, so what is recorded
is terminal-minus-shunt, not terminal. At 2 A this is a 5% error on a 4 V
reading, and it is the one place in this design where the schematic and the
schema disagree about what "terminal voltage" means. **Resolve it before the
build** — either wire V_bus to the cell side of the shunt, or correct in
firmware by adding I·R_s.

---

## 5. Digital interfaces and pin assignment

### 5.1 I²C

| Parameter | Value |
|---|---|
| Bus | I²C, 100 kHz standard mode |
| INA219 address | 0x40 (A0, A1 tied low) **[verify]** |
| Pull-ups | 10 kΩ, usually fitted on the breakout **[verify]** — do not add a second pair |

Duplicate pull-ups are the classic first-build I²C fault: two 10 kΩ pairs in
parallel give 5 kΩ, which over-loads the bus drivers and produces intermittent
NACKs that look like a flaky sensor.

### 5.2 1-Wire

DS18B20 data requires a **4.7 kΩ pull-up to 3.3 V**. This one *is* required —
unlike I²C, breakout boards frequently omit it, and without it the bus reads all
ones and the driver reports 85 °C, the DS18B20 power-on default. 85 °C is inside
the schema's valid range, so the host would accept it as a real reading.

**Conversion time is a design constraint, not a detail.** At 12-bit resolution
the DS18B20 takes up to 750 ms to convert **[verify]**, against a
`SAMPLE_PERIOD_MS` of 1000. A blocking read therefore consumes three-quarters of
every sample period. Two acceptable resolutions:

- Request the conversion at the end of sample *n* and read it at the start of
  sample *n+1* — one period of latency, full resolution, no blocking.
- Drop to 11-bit (0.125 °C, ~375 ms **[verify]**). Still far finer than the
  ±0.5 °C sensor accuracy, so nothing real is lost.

The first is preferred: cell temperature moves slowly relative to 1 s, so one
period of latency is immaterial, whereas losing 750 ms of every second is not.

### 5.3 Pin assignment — ESP32-WROOM-32 DevKit v1

| Signal | GPIO | Notes |
|---|---|---|
| I²C SDA | 21 | Default `Wire` mapping |
| I²C SCL | 22 | Default `Wire` mapping |
| DS18B20 DATA | 4 | 4.7 kΩ pull-up to 3.3 V |
| USB serial TX/RX | 1 / 3 | UART0, used by the USB bridge — **do not reuse** |
| 3V3 | — | Sensor supply |
| GND | — | Common with cell negative |

Pins to avoid, and why:

- **GPIO 34–39 are input-only** and have no internal pull-ups — unusable for
  1-Wire, which needs a bidirectional pin.
- **GPIO 6–11** connect to the on-board SPI flash. Using them prevents boot.
- **GPIO 0, 2, 12, 15** are strapping pins sampled at reset. A sensor holding
  one at the wrong level puts the board into the wrong boot mode, which
  presents as "the sketch does not run" with no other symptom.

> These assignments do not yet appear in `beacon_rig.ino`. The sketch is
> deliberately board-agnostic — its four `read*()` stubs contain no pin numbers —
> so this table is the specification a builder implements *into* those stubs.
> Keeping the numbers here rather than scattering them through the sketch is
> intentional; §9 covers what must be written back once the build is done.

---

## 6. State of charge is derived, not measured

No sensor reports SOC. The firmware estimates it by coulomb counting:

```
SOC(t) = 100 · Q(t) / Q_nominal        Q(t) = Q(t−1) + I·Δt/3600
```

with `Q_nominal = NOMINAL_CAPACITY_AH`. Three properties follow, and the third
is the one that matters:

1. **It drifts.** Integration accumulates the current error of §8 without bound,
   and there is no absolute reference to correct against between full charges.
2. **It must be reported as percent, 0–100**, not as a 0–1 fraction. The feature
   layer compares against 20.0 and 90.0, so a fraction would set
   `deep_discharge_flag` on every row. The host's capture-level plausibility
   check catches this, but the unit belongs to the rig that measured it.
3. **Capacity does not depend on it.** Discharge capacity is integrated from
   current directly. SOC drift therefore degrades the behaviour flags and leaves
   the state-of-health figure — the project's actual output — untouched. This is
   why an imperfect SOC estimate is acceptable here and would not be in a
   production BMS.

---

## 7. Power budget

Supply is USB 5 V through the DevKit's regulator; the cell powers only the load,
never the electronics.

| Element | Typical | Peak | Note |
|---|---:|---:|---|
| ESP32, radios disabled | ~45 mA | ~90 mA | WiFi/BLE off in `setup()` |
| ESP32, WiFi active | — | ~250 mA | **Avoided by design** |
| INA219 | 1 mA | 1 mA | **[verify]** |
| DS18B20 | 1.5 mA converting | 1.5 mA | ~1 µA standby **[verify]** |
| **Total** | **~48 mA** | **~93 mA** | Against 500 mA available from USB 2.0 |

Margin is roughly 5×, and the reason for the largest line item is worth
recording: the firmware disables the radios not to save power but because their
transmit current spikes can brown out a weakly-supplied board — and a brown-out
restarts `millis()` at zero, which the host refuses as a time reversal rather
than integrating two sessions against each other.

---

## 8. Measurement error budget

This is the section that decides whether the rig can meet its acceptance
criterion, and it connects the hardware directly to the project's methodology.

### 8.1 Current

| Source | Magnitude | Type |
|---|---|---|
| INA219 gain error | ±0.5% of reading **[verify]** | Systematic (gain) |
| INA219 offset | ±100 µV shunt → ±1 mA at 0.1 Ω **[verify]** | Systematic (offset) |
| Shunt tolerance | ±1% of value | Systematic (gain) |
| Shunt tempco self-heating | ~±0.2% at 0.4 W **[verify]** | Systematic (gain) |
| Quantisation | ±100 µA | Random |

Gain terms combine in quadrature for an uncorrelated worst case:
√(0.5² + 1.0² + 0.2²) ≈ **1.14% of reading**.

### 8.2 Capacity

Capacity is `∫I dt`, so a gain error on current is a gain error on capacity:

```
Δ(capacity)/capacity ≈ Δ(current)/current = 1.14%
```

Offset contributes separately: ±1 mA integrated over a 2-hour discharge is
±2 mAh, or ±0.06% of a 3400 mAh cell — negligible. Timing error is negligible
too: an ESP32 crystal at ±10 ppm **[verify]** contributes ±0.001%.

**Expected capacity accuracy: ~1.2%, dominated by shunt tolerance.**

Against the 5% acceptance criterion this leaves comfortable margin, and it
identifies the one upgrade that would matter: a **0.1% shunt** would take the
combined gain error to ~0.55%, roughly halving it. Nothing else in the chain is
worth improving first — replacing the INA219 with a 16-bit INA226 improves
quantisation, which is not the limiting term.

### 8.3 Why this matters to the research half

The project's central methodological contribution is that a target's **noise
ceiling** bounds the R² any model can attain against it (`docs/final_report.md`
§4.10: a signal fraction of 0.044 caps attainable R² at 0.044).

The rig's error budget sets that ceiling for any SOH target it produces. A cell
fades roughly 20% over its usable life; the rig measures capacity to ~1.2%. The
signal-to-noise ratio is therefore about 17:1, which is a well-conditioned
target — unlike `capacity_loss`, the per-cycle delta whose ceiling of 0.044
invalidated a year of modelling on this project.

This is the argument for measuring **cumulative fade** rather than per-cycle
delta: differencing two capacity measurements adds their errors while
subtracting most of the signal. At 1.2% measurement error and a per-cycle fade
of ~0.02%, a per-cycle delta is essentially all noise — which is the same
conclusion the software reached from the data, arrived at here from the
instrumentation. The hardware error budget and the target noise ceiling are two
views of one quantity.

*This is the strongest link between the two halves of the project and should be
its own section in the report.*

---

## 9. What changes once the rig is built

This document is a design. On assembly:

1. **Write the §5.3 pin numbers into the four `read*()` stubs** in
   `beacon_rig.ino`, and set `NOMINAL_CAPACITY_AH` to the cell's actual rating.
2. **Resolve the §4 V_bus question** — cell side of the shunt, or firmware
   correction. Record which, because the choice is invisible in the data.
3. **Confirm current is negative under load** before any capture. The schema
   requires negative for discharge, and a reversed shunt yields no discharge
   phases while appearing to stream perfectly.
4. **Replace every [verify] marker** with the figure from the datasheet revision
   of the part actually purchased, and re-run §8.
5. **Measure, don't assume, the error budget** — one full discharge against a
   bench reference gives the real number, and §8 becomes a prediction that was
   tested rather than a calculation that was presented.
6. **Record which `read*()` functions are real.** The stub firmware produces a
   complete Guardian report on a bare board with no sensors attached, because
   its synthetic phases register as complete cycles. It looks exactly like
   success. See `hardware_integration.md`.

---

## 10. Bill of materials

| # | Item | Spec | Qty |
|---|---|---|---:|
| 1 | ESP32-WROOM-32 DevKit v1 | 30-pin, USB micro-B | 1 |
| 2 | INA219 breakout | 0.1 Ω 2 W shunt, I²C 0x40 | 1 |
| 3 | DS18B20 | Waterproof probe or TO-92 | 1 |
| 4 | Resistor | 4.7 kΩ, ¼ W (1-Wire pull-up) | 1 |
| 5 | 18650 cell | Protected, capacity known | 1 |
| 6 | TP4056 + DW01 module | With protection, micro-B | 1 |
| 7 | 18650 holder | Soldered tabs preferred | 1 |
| 8 | Breadboard + jumpers | — | 1 |
| 9 | Constant-current load | Adjustable, ≥2 A | 1 |
| 10 | Reference charger/discharger | With mAh readout | 1 |

Item 10 is not optional. Without an independent reference there is no way to
state the accuracy in §8, and "the rig agrees with itself" is not a result.

---

## 11. Safety

The microcontroller is a **monitor**. It is never the protection device.

- Protection in hardware, in series, always (item 6). If the firmware crashes,
  the cell must still be safe.
- Never connect cell voltage to a GPIO pin directly — GPIO is 3.3 V and **not**
  5 V tolerant.
- One cell, never a series pack, for a first build.
- Charge attended, on a non-flammable surface, never overnight.
- Verify polarity with a multimeter before applying power. A reversed 18650 into
  a TP4056 destroys the module and can vent the cell.
- Lithium cells fail dangerously when shorted, over-discharged or over-charged.
