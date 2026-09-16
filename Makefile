.PHONY: install install-dev pipeline api test test-fast lint format typecheck check \
        study coverage-study audit docker docker-up smoke-day3 clean-day3 \
        serial-demo serial-ports serial-capture serial-fixture \
        hardware-check firmware-compile \
        calce-full-discharge calce-study-full-discharge calce-study-baseline

# joblib probes the physical core count by shelling out, which fails in some
# sandboxes. Every estimator here is n_jobs=1, so the answer is unused anyway.
export LOKY_MAX_CPU_COUNT ?= 1

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Optional benchmark-only models (xgboost, torch/LSTM). Not needed to run the
# pipeline, the API or any calibration script — without them those two methods
# report UNAVAILABLE in the results table rather than crashing.
install-benchmarks:
	pip install "xgboost>=2.0.0" "torch>=2.0.0"

pipeline:
	python main.py

# --- hardware integration --------------------------------------------------
# The serial rig path, end to end. `serial-demo` needs no hardware and no
# pyserial: it drives the deterministic emulator through the identical parser,
# coverage gate and scoring stages a physical board would.
# See docs/hardware_integration.md.

serial-demo:
	python scripts/run_serial_demo.py

serial-ports:
	python scripts/run_serial_demo.py --ports

# Record a session while scoring it, then score the recording back. The two
# runs must agree: that is what makes a capture a fixture rather than a
# souvenir. `--capture` records whatever source is in use, so the same target
# works against a real board by adding `--port`.
serial-capture:
	python scripts/run_serial_demo.py --capture data/interim/rig_session.txt
	python scripts/run_serial_demo.py --replay data/interim/rig_session.txt

# Regenerate the committed replay fixture. Rarely needed - only when the wire
# format changes - and the diff should be inspected when it is, because it
# rewrites the bytes every determinism test compares against.
serial-fixture:
	python tests/fixtures/make_serial_fixture.py

# The software-side gate that must pass before a cell is connected. It does NOT
# validate hardware and cannot: see docs/hardware_integration.md for the
# physical acceptance criteria, which need a board, sensors and a reference.
hardware-check:
	python -m pytest tests/test_hardware_readiness.py tests/test_serial_telemetry.py -v

# Compile the reference firmware for the primary target. Needs arduino-cli:
#   https://arduino.github.io/arduino-cli/latest/installation/
# then, once:
#   arduino-cli core update-index --additional-urls $(ESP32_INDEX)
#   arduino-cli core install esp32:esp32 --additional-urls $(ESP32_INDEX)
# CI compiles esp32, esp8266 and avr-uno; this target does the primary one.
ESP32_INDEX = https://espressif.github.io/arduino-esp32/package_esp32_index.json
FIRMWARE_FQBN ?= esp32:esp32:esp32

firmware-compile:
	arduino-cli compile --fqbn $(FIRMWARE_FQBN) --warnings all firmware/beacon_rig

api:
	uvicorn src.bms.api.app:app --reload

test:
	python -m pytest tests/ -v

test-fast:
	python -m pytest tests/ -v -m "not slow"

coverage:
	python -m pytest tests/ --cov --cov-report=term-missing

lint:
	ruff check src tests scripts main.py

format:
	ruff format src tests scripts main.py

typecheck:
	mypy

# What CI runs. Run this before pushing.
check: lint typecheck test

# --- research reproduction -------------------------------------------------
# Full sweep refits six estimators across 42 folds per target (~14 min each).
# `study-quick` restricts to the reference methods and finishes in seconds.

study:
	python scripts/run_benchmark_study.py --targets capacity_loss cumulative_fade

study-quick:
	python scripts/run_benchmark_study.py --quick

coverage-study:
	python scripts/run_coverage_study.py

# --- CALCE absolute-SOH target (roadmap item 0, ADR 0009 open item) ---------
# Rebuilds the CALCE cycle-level frame from detected full discharges, so Types 5
# and 6 are scored against an absolute reference rather than a partial-cycle
# one. Reads the raw archives: expect roughly an hour.
calce-full-discharge:
	python scripts/build_calce_full_discharge_frame.py

# The rebuilt study, and the controlled baseline it must be compared against.
# Both use the same explicit feature set, so the only difference between them is
# how the target was derived.
CALCE_FEATURES = cycle mean_voltage_v min_voltage_v mean_current_a \
                 resistance_ohm cycle_duration_s

calce-study-full-discharge:
	python scripts/run_benchmark_study.py \
	  --data reports/metrics/calce_full_discharge.csv --targets soh \
	  --features $(CALCE_FEATURES) \
	  --out reports/metrics/calce_full_discharge

calce-study-baseline:
	python scripts/run_benchmark_study.py \
	  --data reports/metrics/calce_cycle_level.csv --targets soh \
	  --features $(CALCE_FEATURES) \
	  --out reports/metrics/calce_baseline_controlled

# Cohort-coverage sweep (ADR 0010) and unsupervised detection (ADR 0011).
coverage-sweep:
	python scripts/run_coverage_sweep.py --repeats 12

detection-study:
	python scripts/run_detection_study.py

audit:
	python scripts/audit_threshold_reachability.py
	python scripts/validate_health_index_versions.py

# --- containers ------------------------------------------------------------

docker:
	docker build -t behavior-aware-bms:local .

docker-up:
	docker compose up

smoke-day3:
	python tests/smoke_day3.py

clean-day3:
	rm -rf data/raw data/interim data/processed logs
