.PHONY: fmt lint typecheck security deadcode check install test

install:
	pip install -r requirements-dev.txt
	pre-commit install

fmt:
	ruff format app/ training/

lint:
	ruff check app/ training/

typecheck:
	mypy app/ training/ --strict

security:
	bandit -r app/ training/ -ll

deadcode:
	vulture app/ training/ --min-confidence 80

check: lint typecheck security deadcode

test:
	pytest tests/ -v
