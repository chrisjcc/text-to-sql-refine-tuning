.PHONY: help install install-dev test lint format clean docker-build docker-run prepare-data train inference benchmark publish

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  install-dev   - Install dependencies including dev tools"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting checks (requires install-dev)"
	@echo "  format        - Format code (requires install-dev)"
	@echo "  clean         - Clean generated files"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  prepare-data  - Download and prepare training data"
	@echo "  train         - Run training (automatically prepares data if needed)"
	@echo "  inference     - Run inference"
	@echo "  benchmark     - Run benchmark tests"
	@echo "  publish       - Publish model to HuggingFace Hub"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src --cov-report=html

lint:
	flake8 src scripts
	black --check src scripts
	isort --check-only src scripts
	mypy src

format:
	black src scripts
	isort src scripts

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf build dist *.egg-info

docker-build:
	docker build -t text-to-sql-grpo:latest .

docker-run:
	docker-compose up -d api

train: install prepare-data
	python scripts/train.py

inference: install
	python scripts/inference.py

prepare-data:
	python scripts/prepare_data.py

benchmark: install
	python scripts/benchmark.py

publish:
	python scripts/publish_to_hub.py \
		--model-path ./outputs/final_model \
		--repo-name $(REPO_NAME)
