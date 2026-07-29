# Day 3 Professional Deliverables — BMS Data Ingestion Foundation

## Objective
Build the Day 3 foundation for the Behavior-Aware BMS project by creating reliable data-ingestion utilities for ZIP/CSV/XLSX battery datasets and proving that one NASA-style sample and one CALCE-style sample can be discovered, loaded, normalized, and exported without manual file clicking.

## Deliverables included

| Area | Deliverable | Purpose |
|---|---|---|
| Archive handling | `scripts/unpack_archives.py` | Recursively scans `data/raw`, safely extracts archives, discovers loadable files, creates manifest, logs failures. |
| Dataset loaders | `scripts/load_nasa.py`, `scripts/load_calce.py` | CLI entry points for loading one NASA/CALCE file into a normalized processed CSV. |
| Reusable package code | `src/bms/io/*.py` | Shared loading, column cleaning, feature creation, NASA/CALCE loader functions. |
| Validation | `tests/smoke_day3.py` | Generates small test samples, runs archive discovery, loads NASA and CALCE samples, verifies outputs. |
| Documentation | `docs/*.md` | Delivery report, data contract, acceptance checklist, team update, code review notes. |
| Configuration | `configs/dataset_sources.yaml` | Dataset-source placeholders and expected folder layout. |
| Environment | `requirements.txt` | Minimal Python dependencies. |
| Commands | `Makefile` | One-command smoke test and cleanup helpers. |

## Folder structure after copying

```text
scripts/
  unpack_archives.py
  load_nasa.py
  load_calce.py
src/bms/io/
  loader_common.py
  load_nasa.py
  load_calce.py
tests/
  smoke_day3.py
docs/
  day3_delivery_report.md
  data_contract_day3.md
  acceptance_checklist_day3.md
  team_update_day3.md
  code_review_notes_day3.md
configs/
  dataset_sources.yaml
requirements.txt
Makefile
README_DAY3.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run smoke test

```bash
python tests/smoke_day3.py
```

or:

```bash
make smoke-day3
```

Expected result:

```text
DAY 3 SMOKE TEST PASSED ✅
```

Expected generated files:

```text
data/interim/discovered_files.csv
data/interim/failed_files.csv
data/processed/nasa/nasa_sample_processed.csv
data/processed/calce/calce_sample_processed.csv
```

## Run on real raw datasets

Place downloaded or provided dataset archives/files under:

```text
data/raw/nasa/
data/raw/calce/
data/raw/stanford/
```

Run discovery:

```bash
python scripts/unpack_archives.py --raw data/raw --out data/interim --copy-loose-files
```

Load one NASA file:

```bash
python scripts/load_nasa.py --input "PATH_FROM_MANIFEST" --out data/processed/nasa/nasa_sample_processed.csv --max-rows 500
```

Load one CALCE file:

```bash
python scripts/load_calce.py --input "PATH_FROM_MANIFEST" --out data/processed/calce/calce_sample_processed.csv --max-rows 500
```

## Git commit

```bash
git add README_DAY3.md requirements.txt Makefile configs scripts src tests docs
git commit -m "Day 3: add professional data ingestion deliverables"
git push
```
