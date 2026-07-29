# Day 3 BMS Deliverables

## Task
Write archive-unpack and file-discovery scripts for ZIP/CSV/XLSX sources, and test on one NASA sample and one CALCE sample.

## Files added
- `scripts/unpack_archives.py`
- `scripts/load_nasa.py`
- `scripts/load_calce.py`
- `src/bms/io/loader_common.py`
- `src/bms/io/load_nasa.py`
- `src/bms/io/load_calce.py`
- `tests/smoke_day3.py`

## Install requirement
```bash
pip install pandas openpyxl
```

## Run Day 3 smoke test
```bash
python tests/smoke_day3.py
```

## Run on real raw data
```bash
python scripts/unpack_archives.py --raw data/raw --out data/interim --copy-loose-files
```

Then choose one NASA file path from:
```bash
cat data/interim/discovered_files.csv
```

Load one NASA sample:
```bash
python scripts/load_nasa.py --input "PATH_FROM_MANIFEST" --out data/processed/nasa/nasa_sample_processed.csv --max-rows 500
```

Load one CALCE sample:
```bash
python scripts/load_calce.py --input "PATH_FROM_MANIFEST" --out data/processed/calce/calce_sample_processed.csv --max-rows 500
```

## Completion evidence
After running the smoke test, these files should exist:
- `data/interim/discovered_files.csv`
- `data/interim/failed_files.csv`
- `data/processed/nasa/nasa_sample_processed.csv`
- `data/processed/calce/calce_sample_processed.csv`

## Git commands
```bash
git add scripts/unpack_archives.py scripts/load_nasa.py scripts/load_calce.py src/bms/io tests/smoke_day3.py
git commit -m "Day 3: add archive discovery and dataset loaders"
git push
```
