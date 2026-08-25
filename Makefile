.PHONY: install install-dev pipeline api test test-fast lint format typecheck check \
        study coverage-study audit docker docker-up smoke-day3 clean-day3

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
