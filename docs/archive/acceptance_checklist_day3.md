# Day 3 Acceptance Checklist

Use this before showing Day 3 completion to a mentor or team lead.

## Code deliverables

- [ ] `scripts/unpack_archives.py` exists.
- [ ] `scripts/load_nasa.py` exists.
- [ ] `scripts/load_calce.py` exists.
- [ ] `src/bms/io/loader_common.py` exists.
- [ ] `src/bms/io/load_nasa.py` exists.
- [ ] `src/bms/io/load_calce.py` exists.
- [ ] `tests/smoke_day3.py` exists.

## Functional checks

- [ ] Archive discovery runs without manual file opening.
- [ ] Manifest file is created.
- [ ] Failure log file is created.
- [ ] NASA sample is converted to processed CSV.
- [ ] CALCE sample is converted to processed CSV.
- [ ] Output contains canonical columns.
- [ ] Smoke test prints `DAY 3 SMOKE TEST PASSED ✅`.

## Documentation checks

- [ ] `README_DAY3.md` explains setup and usage.
- [ ] `docs/day3_delivery_report.md` explains what was completed.
- [ ] `docs/data_contract_day3.md` defines the expected data format.
- [ ] `docs/team_update_day3.md` gives a simple team-facing update.
- [ ] `docs/code_review_notes_day3.md` explains design choices and limitations.

## Git checks

- [ ] Files are added with `git add`.
- [ ] Commit is created with a clear message.
- [ ] Changes are pushed to GitHub.

Recommended commit message:

```bash
git commit -m "Day 3: add professional data ingestion deliverables"
```
