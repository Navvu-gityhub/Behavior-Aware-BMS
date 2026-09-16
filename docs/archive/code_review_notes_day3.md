# Day 3 Code Review Notes

## Design choices

### 1. Separate CLI scripts and package modules
The `scripts/` files are kept as simple command-line entry points. The reusable logic is kept under `src/bms/io/`. This makes the project easier to test and extend.

### 2. Manifest-based discovery
The project should not depend on manually opening folders and selecting files. `discovered_files.csv` acts as a structured file inventory for the team.

### 3. Safe extraction
The archive extractor checks extracted paths before writing files. This reduces the risk of unsafe paths inside ZIP/TAR archives.

### 4. Canonical column names
Battery datasets often use different names for the same quantity. The loader converts common variations into names such as `voltage_v`, `current_a`, `temperature_c`, and `capacity_ah`.

### 5. Small smoke test first
Day 3 validates the pipeline using generated sample files. This is faster and safer than testing only on large external datasets.

## Known limitations

- `.mat` files are not handled yet.
- Real NASA/CALCE datasets may need extra parsing rules depending on their exact file structure.
- Current mode detection assumes positive current means charging and negative current means discharging; some datasets may use the opposite convention.
- Source detection is based on file path text such as `nasa`, `calce`, or `stanford`.

## Suggested improvements

- Add `load_stanford.py`.
- Add `.mat` file support using `scipy.io` or `h5py`.
- Add dataset-specific validation reports.
- Add automated feature extraction for degradation-risk scoring.
- Add unit tests for column normalization.
