# Day 3 Data Contract

## Purpose
This document defines the minimum data format expected after Day 3 dataset loading. It helps the team understand what output the NASA and CALCE loaders should produce before feature extraction starts.

## Input formats supported

| Format | Supported |
|---|---|
| `.zip` | Yes, through archive discovery |
| `.tar`, `.tar.gz`, `.tgz`, `.gz` | Yes, through archive discovery |
| `.csv` | Yes |
| `.xlsx`, `.xls` | Yes |
| `.txt` | Yes, best-effort delimiter detection |
| `.mat` | Not in Day 3 scope |

## Required minimum useful input

A file must contain at least one meaningful battery column after normalization:

- voltage
- current
- temperature
- capacity

If none of these are detected, the loader returns a clear error instead of silently producing useless output.

## Canonical columns

| Canonical column | Accepted examples from raw data |
|---|---|
| `timestamp` | `timestamp`, `time`, `datetime`, `test_time`, `time_s` |
| `cycle` | `cycle`, `cycle_index`, `cycle_number`, `cycle_no` |
| `voltage_v` | `voltage`, `voltage_measured`, `terminal_voltage`, `Ewe/V` |
| `current_a` | `current`, `current_measured`, `current_load`, `I/A` |
| `temperature_c` | `temperature`, `temp`, `temperature_measured`, `temp_c` |
| `capacity_ah` | `capacity`, `discharge_capacity`, `charge_capacity`, `qdischarge` |
| `soc` | `soc`, `state_of_charge` |
| `soh` | `soh`, `state_of_health` |

## Derived columns

| Column | Formula / Logic |
|---|---|
| `source` | Hardcoded as `nasa` or `calce` by loader |
| `source_file` | Original input path |
| `power_w` | `voltage_v × current_a` |
| `mode_guess` | `charge` if current > 0, `discharge` if current < 0, otherwise `unknown` |

## Output locations

```text
data/processed/nasa/nasa_sample_processed.csv
data/processed/calce/calce_sample_processed.csv
```

## Quality expectations

- No manual clicking should be needed.
- Unknown extra columns should not be deleted.
- Missing canonical columns should be created with empty values.
- Numeric columns should be converted safely.
- Bad files should fail with a readable error message.
