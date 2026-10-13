# Bench test on the rig: step by step

This guide is for the parts on hand: NodeMCU ESP8266, INA219, LM35, one 18650
cell, a TP4056 charger board and a load resistor. One command records and
judges each session: `scripts/bench_session.py`. Background and the full
acceptance list are in [`hardware_integration.md`](hardware_integration.md).

## What this proves

| Stage | What you get | Time |
|---|---|---|
| 0. Both sensors working together | First capture with voltage, current **and** temperature at once (never achieved so far) | 30 min |
| 1. One discharge | First real discharge on BEACON's own hardware; capacity counted and cross-checked against Ohm's law | 3-4 h |
| 2. Six or more cycles | Enough discharges for BEACON's health and resistance readings on your own cell | 2-4 days, by hand |
| 3. Charges recorded | Charge curves BEACON has never seen: a fresh test for health-while-charging | comes with stage 2 |

## Parts and safety

- **Load: a 4.7-10 ohm resistor rated 5-10 W** (cement or wire-wound). At
  4.7 ohm the cell drives about 0.8 A and the resistor sheds about 3 W, so it
  gets hot; mount it in free air.
  - A small 1/4 W resistor will burn.
  - An LED is far too small a load (~20 mA, about 100 hours per discharge).
- **Measure the resistor with a multimeter**, including its leads. Write the value down: it is the cross-check.
- **Cell:** use a protected 18650 if you can. Never discharge below 3.0 V under
  load; the script warns you, but you disconnect the load.
- **Work on a non-flammable surface. Stay near the cell** while it discharges,
  and don't leave a charge running overnight unattended.
- **Wires** must carry 1 A: no thin jumper wires in the load path.

## Wiring

```
   Cell +  ──► INA219 VIN+          INA219 VCC ── NodeMCU 3V3
               INA219 VIN- ──┬──►  load resistor ──┐      INA219 GND ── NodeMCU GND
                             │                     │      INA219 SDA ── D2
                             └──►  TP4056 B+       │      INA219 SCL ── D1
                                   TP4056 B- ──────┤      LM35 OUT   ── A0
   Cell -  ────────────────────────────────────────┘      LM35 VCC/GND from 3V3/GND
   (NodeMCU GND joined to Cell -)                         LM35 taped to the cell
```

Connect **either** the load **or** the TP4056 to VIN-, never both at once.
The current sign is handled by the firmware: BEACON expects discharge to be
negative and refuses a capture whose sign is reversed.

## Stage 0: both sensors at once

1. In `firmware/beacon_rig/beacon_rig.ino` set `USE_REAL_SENSORS 1` and
   `SERIAL_BAUD 9600`, then flash it (`python scripts/flash_rig.py --port COM5`).
2. Run:
   ```
   python scripts/bench_session.py --port COM5 --hours 0.1
   ```
3. The first lines must read `board declares: t, v, i, tc, soc`.
   - If `tc` or `i` is missing, that sensor failed its start-up check. Fix the
     wiring and repeat until all five appear.

## Stage 1: one discharge

1. Charge the cell fully with the TP4056 (its LED turns from red to blue/green).
   Disconnect it.
2. Let the cell **rest 10 minutes with nothing on VIN-**. The quiet samples
   before the load are what BEACON measures the resistance step from.
3. Start recording, then connect the resistor firmly:
   ```
   python scripts/bench_session.py --port COM5 --hours 4 --load-ohm 4.7
   ```
   Use your measured resistance for `--load-ohm`.
4. When the alarm says the voltage is below 3.0 V, disconnect the load. Leave
   it recording for 10 minutes more, then press Ctrl+C.
5. The report is written beside the capture in `data/interim/bench/`. Look for:
   - **PASS** that current was negative under load;
   - the **Ohm's-law check** within 5%;
   - the discharge in mAh, against the cell's 2,200 mAh rating (an old cell reads lower).

## Stage 2 and 3: six or more cycles

Repeat, recording each step:

1. **Rest** 10 minutes, nothing connected.
2. **Discharge** through the resistor to 3.0 V.
3. **Rest** 10 minutes.
4. **Charge** with the TP4056 until it shows full.

Record each session (one per discharge or charge is fine). After six
discharges, BEACON can form the cell's reference and report health and
resistance. The charges are new data for the health-while-charging method,
which has never seen this cell. Write its test criteria before running it.

## What to claim afterwards

Name the channels that were real (the report lists them), the number of
cycles, and the cross-check result. For example:

> "The serial path recorded N discharges of one HONGLI 18650 on a NodeMCU with
> INA219 and LM35. Counted capacity agreed with V/R to X%."

Not "hardware validated". One cell is one cell.
