# Real BMS telemetry pipeline

Runs recorded CAN logs and live CAN buses through the same scoring stages the
dataset pipeline uses:

```
source -> decode (DBC) -> unified schema -> cycle segmentation
       -> behaviour features -> risk -> health -> RUL -> Guardian
```

No stage is reimplemented here. Live and batch runs share one code path, so a
disagreement between them is a data difference rather than an untraceable
divergence between two implementations.

## Quick start

```python
import cantools
from src.bms.telemetry import replay_log

dbc = cantools.database.load_file("src/bms/io/dbc_examples/beacon_reference_pack.dbc")
signal_map = {
    "pack_voltage": "voltage_v",
    "pack_current": "current_a",
    "pack_soc": "soc",
    "pack_temp_mean": "temperature_c",
    "pack_temp_max": "max_temp",
}

result = replay_log("logs/drive_2026_07.asc", dbc, signal_map, cell_id="VEH_01")
print(result.render())
```

Live capture differs only in the source:

```python
from src.bms.telemetry import LiveBusSource, run_telemetry_pipeline

source = LiveBusSource(name="vehicle", channel="can0",
                       interface="socketcan", duration_s=60.0)
result = run_telemetry_pipeline(source, dbc, signal_map, cell_id="VEH_01")
```

## State of health cannot be read off the bus

This is the part most likely to surprise.

`dashboard/beacon_data.py` derives SOH from `capacity_ah` — each cycle's measured
discharge capacity over the cell's initial capacity. **A CAN bus does not carry
per-cycle discharge capacity.** It carries instantaneous current, voltage,
temperature and the BMS's own SOC estimate. No schema mapping produces capacity
from those.

What it does carry is enough to integrate it. Charge moved during a discharge is
the time integral of current:

```
Q = integral(|i| dt) / 3600      amp-hours
```

`cycles.py` segments the stream into charge, discharge and rest phases, then
integrates each discharge trapezoidally. That is where `capacity_ah` comes from,
and therefore where SOH becomes possible at all.

### Partial cycles are excluded, not scaled

The integral is only comparable across cycles at equal depth of discharge. A pack
taken from 90% to 40% moves roughly half the charge of one taken from 100% to 0%.
Reporting the first as "capacity" would show a healthy cell as severely degraded.

The tempting repair is to normalise by the observed SOC swing. This pipeline does
not, for two reasons:

1. The BMS's reported SOC is itself derived from a capacity estimate, so scaling
   capacity by it makes the measurement depend on the quantity being measured.
2. SOC is least accurate at the extremes and under load, which is exactly where a
   partial cycle ends.

So partial discharges are marked `is_complete=False` and excluded from SOH.
`depth_of_discharge` is still reported, for diagnosis only.

**Consequence for real fleets:** ordinary driving produces mostly partial cycles.
A log with 400 discharges may support six capacity points. That is a property of
the data, not a defect, and `TelemetryResult.yield_summary` reports the count so a
thin result is visible rather than assumed:

```
6/412 discharges usable for SOH (1%); 406 partial.
Largest observed discharge 41.2 Ah.
```

## Where the pipeline refuses

Three refusals, each tracing to a decision made earlier in the project.

### 1. Missing signal channels

Checked before any frame is decoded. `check_signal_coverage` compares a DBC's
decodable signals against what the downstream stages need.

The example DBC shipped with this repository fails that check:

```
twizy_bms_1.dbc: INCOMPLETE
  decodable signals: ['v_b_current', 'v_b_soc', 'v_c_climit']
  mapped channels: ['current_a', 'soc']
  MISSING temperature_c -> needed by behaviour features (high_temp_flag, temp_rolling_*)
  MISSING voltage_v -> needed by unified schema validation
```

`compute_behavior_flags` computes `high_temp_flag` from `temperature_c`. The
NaN-as-healthy fix established why an absent channel must refuse rather than
proceed: a NumPy comparison against NaN evaluates False, so a missing temperature
would silently produce "not hot" for every row and a healthy-looking score for a
pack nobody measured.

`beacon_reference_pack.dbc` is included as a DBC with complete coverage, for
benching the full path.

### 2. No complete discharge cycle

Covered above. The run reports how many usable capacity points it found.

### 3. Fade prediction

`AdaptiveCalibrator.score` refuses while nothing has passed the promotion gate,
which is the current state (ADR 0005). **This pipeline does not route around
that.** It emits the rule-based severity index, labelled throughout as triage
rather than measurement, and does not present it as a calibrated fade prediction.

Wiring a model the gate rejected into a live dashboard, where it would look
authoritative, is precisely the failure this project exists to prevent.

## Segmentation details

| Constant | Default | Why |
|---|---|---|
| `REST_THRESHOLD_A` | 0.5 A | Dead band around zero. Key-off parasitic draw is well under an amp; any real load is several. |
| `MIN_PHASE_SAMPLES` | 3 | A regenerative-braking spike inside a discharge is not a charge phase. |
| `COMPLETE_CYCLE_FRACTION` | 0.80 | Relative to the largest discharge seen for that cell, because nominal capacity is usually unknown from telemetry alone. |

All three are arguments, not hard-coded. A pack's noise floor is a property of its
instrumentation.

Two guards worth knowing about:

- **Non-monotonic time raises.** Integrating over unsorted time silently cancels
  charge against itself.
- **Timestamps are never forward-filled.** A CAN bus interleaves messages, so each
  frame carries a subset of channels; rows are assembled per timestamp. Filling
  across timestamps would invent measurements that were not taken.

## Supported sources

| Source | Use | Tested |
|---|---|---|
| `MemorySource` | fixtures, in-process replay | yes |
| `LogFileSource` | `.blf`, `.asc`, `.trc`, `.csv`, `.log` via python-can | yes (`.asc`) |
| `LiveBusSource` | any python-can interface | virtual bus only |

`LiveBusSource` wraps `can.Bus`, so socketcan, pcan, kvaser and vector all work.
**Real hardware is untested** — none is attached to the development environment.
The virtual-bus path is covered, including the duration bound, and that limit is
stated rather than implied to be verified.

Error and remote frames are skipped: they carry no signal payload, so decoding
them would produce values from undefined bytes.

## Tests

`tests/test_telemetry_pipeline.py`, 23 tests. The four that matter most:

- `test_the_shipped_twizy_dbc_is_refused_for_lacking_temperature`
- `test_partial_discharges_alone_yield_no_soh`
- `test_a_partial_cycle_is_never_scaled_up_to_look_complete`
- `test_replay_and_live_capture_agree_frame_for_frame`
