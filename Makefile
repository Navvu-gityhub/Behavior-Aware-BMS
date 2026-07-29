.PHONY: install install-dev pipeline api test smoke-day3 clean-day3

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

pipeline:
	python main.py

api:
	uvicorn src.bms.api.app:app --reload

test:
	python -m pytest tests/ -v

smoke-day3:
	python tests/smoke_day3.py

clean-day3:
	rm -rf data/raw data/interim data/processed logs
