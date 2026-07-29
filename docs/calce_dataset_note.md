# CALCE Dataset Note (updated 2026-07-21, after receiving Capacity Characterization_Initialization.zip)

## What was uploaded and checked

1. `PLN_Number_SOC_Temp_StoragePeriod.xlsx` — 150 pouch cells, each assigned
   a storage SOC/temperature and duration (3 weeks / 3 months / 6 months),
   with one recorded `Discharge Capacity` value per cell.
2. `Impedance_Characterization_Initialization.zip` — 150 EIS spectra, one
   snapshot per cell.
3. `Capacity_Characterization_Initialization.zip` — 17 raw Arbin cycler
   workbooks (17 test-day batches), one sheet per physical channel, one
   physical channel per PLN cell. **This is the source data underlying the
   `Discharge Capacity` column in file (1).**

## Loader built

`src/bms/io/load_calce_capacity.py` parses (3) into unified-schema-shaped
telemetry: real per-row voltage/current/time from an actual Arbin export.
Two real format issues were found and fixed while building it:

- Files are saved with a `.xls` extension but are OOXML (xlsx) content;
  `openpyxl` rejects them by extension alone. Worked around by re-copying
  to a `.xlsx`-suffixed temp path before opening.
- The physical-channel-to-PLN-cell mapping lives only as free text in an
  `Info` sheet `Comments` field, in **inconsistent case** across files
  (`Ch -` in some, `ch -` in others) — an initial case-sensitive parse
  silently mapped only 34 of 150 cells before this was caught and fixed;
  the corrected parser recovers 138 of 150 (92%). The remaining 12
  (PLN 30-31, 98-106) trace to an actual gap in the files supplied — no
  workbook in this upload covers that PLN range — not a parsing failure.

**No temperature channel is recorded in this export at all** — a genuine
gap in the source data. `compute_risk_assessment`/`compute_health_index`
were found to silently treat this missing data as "temperature is fine"
(NumPy comparisons against NaN evaluate to False, falling through to the
lowest-risk bucket) rather than failing — a real correctness bug, now
fixed to raise instead of silently defaulting (see
`src/bms/risk/stress_score.py`, `src/bms/health/health_index.py`, and the
regression test in `tests/test_pipeline.py`). This bug would not have
been caught without actually running the pipeline against this data.

## What this data cannot do: capacity-fade or calendar-aging calibration

Confirmed by direct inspection, not assumption: every channel in every one
of the 17 workbooks has exactly one `Cycle_Index`. This is a **single-cycle
baseline characterization**, not multi-cycle aging data — it cannot
calibrate the pipeline's cycle-based RUL/health-index model at all.

A calendar-aging analysis (does storage SOC/temperature/duration predict
capacity loss?) was attempted by joining this file's per-cell capacity to
file (1)'s post-storage capacity. **The two values are bit-for-bit
identical for every one of the 132 matched cells.** This isn't evidence of
zero capacity loss — it's evidence that `Capacity Characterization_
Initialization` and file (1)'s `Discharge Capacity` column record the
*same measurement event*. "Initialization" refers to initializing the
Arbin test procedure, not a pre-storage baseline distinct from the
post-storage reading. There is no independent "before" capacity in the
files supplied, so capacity loss during storage cannot be computed from
this data. The calendar-aging analysis script and its (misleading, since
it was really reporting a join artifact as "0% loss for everyone") output
were removed rather than kept as a result.

## Bottom line

CALCE contributed real engineering value (a working loader for a messy,
real-world raw-export format, and a genuine correctness bug caught by
running it) but **zero calibration value** for either the main
cycle-aging pipeline or a calendar-aging model, given the specific files
supplied. If CALCE calibration is wanted, what's actually needed is data
this upload doesn't contain: either a genuine pre-storage baseline capacity
distinct from the post-storage reading, or (preferably) a multi-cycle
CALCE cycling dataset (e.g. the CS2/CX2 series CALCE also publishes),
which would provide the same kind of cycle-indexed capacity-fade ground
truth NASA did.
