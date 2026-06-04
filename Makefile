.PHONY: install smoke-day3 clean-day3

install:
	pip install -r requirements.txt

smoke-day3:
	python tests/smoke_day3.py

clean-day3:
	rm -rf data/raw data/interim data/processed logs
