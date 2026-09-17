# Validation suite

Generated 2026-09-16T07:41:31+00:00 by `scripts/run_validation_suite.py` --quick.

Every stage below is a command from `docs/final_report.md` Appendix A.
This runner orchestrates them and defines no protocol of its own.

| Stage | Report section | Status | Seconds |
|---|---|---|---:|
| `pipeline-smoke` | S2 / S3 end-to-end pipeline | **PASS** | 3.19 |
| `threshold-audit` | S4.7 threshold reachability | **PASS** | 2.17 |
| `health-index-versions` | S4.8 health index v1 vs v2, LOBO + LOCO | **PASS** | 60.89 |
| `horizon-regression` | S4.5 longer prediction horizons | **PASS** | 8.03 |
| `mixed-effects` | S4.6 mixed-effects identifiability | **PASS** | 9.44 |
| `conformal-coverage` | S4.12 / S5.5 coverage within vs across protocol | **PASS** | 23.91 |
| `benchmark-study-quick` | S4.10-S4.12 (reference methods only) | **PASS** | 96.78 |
| `report-figures` | final_report.md figures 1-3 | **PASS** | 4.73 |
| `manuscript-figures` | ress_draft.md figures 1-3 | **PASS** | 6.2 |

## Artifacts rewritten with different content

A changed artifact is not automatically wrong, but it is also not automatically fine: `tests/test_reported_numbers.py` pins quoted figures to these files, so run it before accepting any change.

- `coverage_folds.csv`
- `coverage_summary.csv`
- `health_version_task_b_cv.csv`
- `mixed_effects_diagnostics.csv`
- `risk_term_variability.csv`
