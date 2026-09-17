# Reproducing this project

One path from a clean clone to a running dashboard, with every `make` target's
plain-Python equivalent beside it. `make` and `docker` are **not** installed on
the author's machine, so the Python column is the one that is actually
exercised; the Makefile is a convenience, not the contract.

---

## Requirements

| | Version | Notes |
|---|---|---|
| **Python** | **≥ 3.10** | Declared in `pyproject.toml`. CI tests 3.10 and 3.12 |
| Node.js | 22 | Only for the gateway and dashboard client |
| arduino-cli | 1.5+ | Only to compile the rig firmware |
| pandoc | any recent | Only to rebuild `reports/final_report.docx` |
| Docker | optional | Nothing requires it; every service runs directly |

No API keys, no cloud service, no database, no authenticated endpoint. Nothing
in this project reads a credential — see `docs/security.md`.

---

## 1. Clone and install

```bash
git clone https://github.com/Navvu-gityhub/Behavior-Aware-BMS.git
cd Behavior-Aware-BMS
pip install -r requirements-dev.txt
```

> **Installing into an existing conda environment?** `shap` 0.52+ requires
> numpy ≥ 2, and pip will upgrade numpy underneath a conda-installed matplotlib
> built against numpy 1.x. The result is an `ImportError` from
> `matplotlib._path` that looks like a matplotlib fault and is not. Pin
> `numpy==1.26.4` and `shap==0.46.0`, or install into a clean virtualenv.

---

## 2. Run it

| What | Plain Python | `make` |
|---|---|---|
| End-to-end pipeline → `dashboard.html` | `python main.py` | `make pipeline` |
| Test suite | `python -m pytest tests/ -q` | `make test` |
| Lint + types + tests | `python -m ruff check src tests scripts main.py && python -m mypy && python -m pytest tests/` | `make check` |
| Validation suite | `python scripts/run_validation_suite.py` | `make validate` |
| Validation, CI-sized | `python scripts/run_validation_suite.py --quick` | `make validate-quick` |
| Hardware-readiness gate | `python -m pytest tests/test_hardware_readiness.py -v` | `make hardware-check` |
| Emulated rig → Guardian | `python scripts/run_serial_demo.py` | `make serial-demo` |
| List serial ports | `python scripts/run_serial_demo.py --ports` | `make serial-ports` |
| API on `:8000` | `python -m uvicorn src.bms.api.app:app --reload` | `make api` |
| Benchmark study | `python scripts/run_benchmark_study.py --targets capacity_loss cumulative_fade` | `make study` |
| Manuscript figures | `python scripts/generate_manuscript_figures.py` | — |
| Firmware compile | `arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all firmware/beacon_rig` | `make firmware-compile` |
| Rebuild report `.docx` | `powershell -File scripts/build_report_docx.ps1` | — |

On Linux/macOS the `.sh` variant of the report build is `scripts/build_report_docx.sh`.

---

## 3. The dashboard (optional)

Three processes. The Python service is the only one that computes anything; the
gateway proxies and the client renders.

```bash
# terminal 1 — Python service
python -m uvicorn src.bms.api.app:app --port 8000

# terminal 2 — Express gateway
cd mern/server && npm ci && npm start

# terminal 3 — React client
cd mern/client && npm ci && npm run dev
```

Then open `http://localhost:5173`. With Docker available, `docker compose up`
does the same three.

**Environment variables**, all optional:

| Variable | Default | Effect |
|---|---|---|
| `PYTHON_API_BASE_URL` | `http://127.0.0.1:8000` | Where the gateway forwards |
| `PORT` | `5000` | Gateway port |
| `CORS_ORIGIN` | `*` | Origins the browser may call the gateway from |
| `BEACON_ALLOWED_DATA_ROOTS` | repo data dirs + temp | Directories the API may read a request-supplied path from |
| `LOKY_MAX_CPU_COUNT` | — | Set to `1` in constrained environments; joblib's core probe fails in some sandboxes |

---

## 4. Datasets

**None are redistributed, and none are needed to run the pipeline, the tests or
the dashboard.** The simulator generates telemetry, and the tracked CSVs under
`reports/metrics/` carry every figure the reports quote.

Raw data is needed only to re-derive those CSVs from source:

| Dataset | Used by | Where it goes |
|---|---|---|
| NASA PCoE battery ageing | §4.1–4.12 calibration | `data/raw/nasa/` |
| CALCE CS2 / CX2 cycling | §4.13, ADR 0012 | `data/raw/calce/` |
| Oxford battery degradation | loader only | `data/raw/oxford/` |

Each carries its own terms; see the dataset's own repository. Raw archives are
gitignored. `scripts/run_validation_suite.py` detects their absence and
**skips** the stages that need them with a stated reason, rather than failing.

---

## 5. What reproduces, and what does not

The validation suite snapshots every tracked artifact under `reports/` and
**restores** any that a stage rewrites, so running it reports reproducibility
drift instead of causing it. Pass `--accept-changes` only after deciding new
values are correct.

Two known limits, stated rather than discovered:

- **The conformal coverage artifacts do not currently reproduce.** Re-running
  `scripts/run_coverage_study.py` on unchanged input yields 31 LOBO folds where
  the committed artifact has 32, moving `min_coverage` and `worst_group`. The
  result is deterministic across repeat runs, so this is code/artifact drift
  rather than randomness. The committed values remain the published record;
  which set is correct is unresolved.
- **`tests/test_reported_numbers.py` is the authority.** It pins every quoted
  figure to its artifact. If it fails, the prose is wrong — do not update a
  pinned number to match a new result without demonstrating the old one was
  invalid.

---

## 6. Firmware

```bash
arduino-cli core update-index --additional-urls \
  https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli core install esp32:esp32 --additional-urls \
  https://espressif.github.io/arduino-esp32/package_esp32_index.json
arduino-cli compile --fqbn esp32:esp32:esp32 --warnings all firmware/beacon_rig
```

The sketch has no `#include` of any kind — pure Arduino core, so no library
installation is required until you replace the sensor stubs. Verified compiling
on `arduino:avr:uno`, `esp8266:esp8266:nodemcuv2` and `esp32:esp32:esp32`; CI
compiles all three on every push. See `docs/hardware_integration.md`.

---

## See also

- `docs/architecture.md` — how the system fits together
- `docs/security.md` — the audit, and what this service does and does not defend
- `docs/final_report.md` Appendix A — the per-result command sequence
- `docs/hardware_integration.md` — bring-up procedure and acceptance criteria
