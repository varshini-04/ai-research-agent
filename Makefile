.PHONY: install test lint ingest eval
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
ingest:
	research-agent ingest
eval:
	python evals/run_evals.py
