.PHONY: help install test lint format clean docker-build docker-run train inference

help:
	@echo "Available commands:"
	@echo "  install       - Install dependencies"
	@echo "  test          - Run tests"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code"
	@echo "  clean         - Clean generated files"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-run    - Run Docker container"
	@echo "  train         - Run training"
	@echo "  inference     - Run inference"

install:
	pip install -r requirements.txt
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

train:
	python scripts/train.py

inference:
	python scripts/inference.py

prepare-data:
	python scripts/prepare_data.py

benchmark:
	python scripts/benchmark.py

publish:
	python scripts/publish_to_hub.py \
		--model-path ./outputs/final_model \
		--repo-name $(REPO_NAME)
