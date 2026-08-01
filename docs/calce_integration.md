# CALCE integration

Ingests CALCE CS2/CX2 cycling data and runs it through the same scoring stages
as NASA, by delegating to those modules unchanged. Nothing here reimplements a
stage: a difference between a CALCE run and a NASA run should be a property of
the data, not of two implementations.

## What existed before, and why it was not enough

| Module | Reads | Usable for fade? |
|---|---|---|
| `io/load_calce.py` | one Excel file | no cycle reconciliation |
| `io/load_calce_capacity.py` | capacity-characterisation workbooks | **no** — single-cycle (ADR 0001) |
| `io/load_calce_cycling.py` | **CS2/CX2 cycling archives** | **yes** |

## The multi-file problem

NASA ships one `metadata.csv` indexing every test. CALCE ships **many files per
cell**, named by recording date, and each file's `Cycle_Index` restarts at 1.

Concatenating them naively produces fifteen rows labelled "cycle 1", which the
cycle-level layer collapses into a single cycle spanning the cell's entire life.
The resulting frame is structurally valid and the trajectory is destroyed —
which is why `_reconcile_cycle_index` offsets each file by the running maximum
rather than leaving it to the caller.

### Filename dates must be anchored

Files are ordered by the date in their name, because lexical order is wrong:
`9_20_10` sorts after `10_04_10` as text while preceding it by three months.

A first implementation used an unanchored regex, which consumed the cell number:
`CS2_33_10_04_10` parsed as month=2, **day=33**, year=2010. Ordering was
arbitrary and the fade trajectory came out scrambled — valid-looking output,
completely wrong sequence. The pattern is now anchored to the end of the stem,
and `test_files_are_ordered_by_date_not_lexically` pins it.

## Column mapping

`build_rename_map` resolves aliases and then keeps only names in
`CANONICAL_COLUMNS`. That filter is correct — the unified schema is a shared
contract — but it drops `Test_Time(s)`, `Step_Index` and the charge columns,
and it does not know `Discharge_Capacity(Ah)`, which **is the fade target**.

So the loader does two renames: canonical names through the shared map, and
CALCE-specific names locally. Widening `CANONICAL_COLUMNS` so CALCE could carry
`step_index` would push an Arbin implementation detail into every other
dataset's frames.

## What CALCE does not record, and what is not invented

CS2/CX2 Arbin exports carry seventeen columns. **None is temperature**, and
none is state of charge. The cells were cycled at room temperature, about 23 C,
and the ambient was a property of the room rather than a recorded channel.

Three options existed. Only one is honest:

1. **Fill `temperature_c` with 23.0.** The schema is satisfied, `high_temp_flag`
   is computed, and a thermal stress score appears for a quantity nobody
   measured. This is the NaN-as-healthy defect with extra steps — a constant
   cannot raise a flag, so every CALCE cell would score as thermally
   unstressed regardless of what happened to it.
2. **Derive `soc` by integrating current.** Defensible for NASA, where the
   loader integrates against a known per-test capacity. Circular for CALCE:
   the capacity being integrated toward is the fade target.
3. **Refuse, and name the missing channel.** What this module does.

`analyze_calce_cell` runs every stage the instrumentation supports and reports
the rest as unavailable:

```
CS2_36: MEASURED_ONLY
  CS2_36: 12,000 rows across 200 cycles from 8 file(s)
    not recorded by this dataset: ['temperature_c', 'soc']
  stages completed: ['load', 'capacity_fade']
  measured SOH: 100.4% at cycle 1 -> 64.1% at cycle 200
  measured capacity fade: 0.3940 Ah over 200 cycles
  UNAVAILABLE: behavior_flags, risk_assessment, health_index, guardian
    CALCE does not record 'temperature_c' (needed by high_temp_flag,
    temp_rolling_mean, thermal stress term), 'soc' (needed by
    deep_discharge_flag, high_soc_flag, depth-of-discharge term)...
```

A CS2 cell therefore yields **measured SOH and capacity fade** — which is what
CALCE is genuinely good for — and no behavioural risk score, which it cannot
support.

### CX2_4 is the exception

CX2_4 was cycled across 25, 35, 45 and 55 C with separate thermocouple files.
`load_calce_cell(temperature_dir=...)` joins them on test time by nearest
match, because the thermocouple logger and the cycler sample on their own
clocks and an exact join would drop nearly every row.

With temperature joined, the temperature requirement clears. The pipeline
follows the instrumentation, not the dataset label.

## Per-cycle reduction

`capacity_ah` is the **maximum** within a cycle. Arbin's `Discharge_Capacity`
accumulates monotonically through a discharge and resets between cycles, so its
per-cycle maximum is the charge that cycle delivered. The mean reports roughly
half of it; the last value is unreliable when a file ends mid-cycle.

Initial capacity is the **median of the first five cycles**, not cycle one: the
first Arbin cycle often includes a formation or conditioning step whose
capacity is not representative.

## Measured feasibility

`measured_feasibility` runs `assess_commensurability` against loaded files
rather than published metadata. ADR 0006 requires this before citing any
transfer result.

```
PREDICTED (metadata):    PREDICTED_FEASIBLE
  ambient_temperature    BLOCKED
  depth_of_discharge     usable

MEASURED (loaded files): FEASIBLE_REDUCED
  usable:                ['capacity_loss']
  absent in target:      ['avg_temp', 'avg_soc']
```

The measured screen confirms the prediction: CALCE cannot receive a temperature
coefficient because it never recorded temperature.

**One caveat worth stating.** A short CALCE run is correctly judged
*incommensurable*: a 15-cycle cell fades by ~0.03 Ah against NASA's ~0.4 Ah
range, a spread ratio below the 10% floor. That rejection is the check working.
`test_a_short_run_is_correctly_judged_incommensurable` pins it, and the
realistic fixture uses 200 cycles rather than relaxing the threshold.

## Expected layout

```
data/raw/calce/
  CS2_33/  CS2_33_9_04_10.csv  CS2_33_10_12_10.csv  ...
  CS2_34/  ...
  Temperature/  CX2_4_1_12_11.csv        # CX2_4 only
```

`data/raw/calce/` is currently empty. Place the archives there and the loader
reads them; no code change is required.

## Tests

`tests/test_calce_cycling.py`, 32 tests. Fixtures reproduce the seventeen-column
Arbin schedule export exactly, including the per-file `Cycle_Index` restart.

**The fixtures are not CALCE measurements.** Any number derived from them is a
property of the fixture, and none may be reported as a CALCE result.
