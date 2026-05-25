.PHONY: help install install-dev lint test train evaluate serve docker-up docker-down clean mlflow

PYTHON = python
PIP = pip
CONFIG = configs/training.yaml
MODEL = runs/train/arabic_id_detector/weights/best.pt
DATA = Egyptain-Person-ID-1/data.yaml

help:
	@echo "Arabic ID OCR - MLOps Pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install all dependencies including dev tools"
	@echo ""
	@echo "Data:"
	@echo "  make validate-data  Validate dataset integrity"
	@echo "  make dvc-repro      Reproduce full DVC pipeline"
	@echo ""
	@echo "Training:"
	@echo "  make train          Train YOLOv11 detector"
	@echo "  make evaluate       Evaluate on test set"
	@echo "  make mlflow         Open MLflow UI"
	@echo ""
	@echo "Testing:"
	@echo "  make test           Run test suite"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "Serving:"
	@echo "  make serve          Start FastAPI server locally"
	@echo "  make docker-up      Start full stack (API + MLflow + Prometheus + Grafana)"
	@echo "  make docker-down    Stop all services"

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black isort ruff

validate-data:
	$(PYTHON) -c "\
from src.data.validate import DatasetValidator; \
v = DatasetValidator('Egyptain-Person-ID-1'); \
v.run(); \
import json; print(json.dumps(v.report(), indent=2))"

train:
	$(PYTHON) src/training/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) src/training/evaluate.py --model $(MODEL) --data $(DATA)

dvc-repro:
	dvc repro

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html:coverage_html

serve:
	$(PYTHON) -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "Services started:"
	@echo "  OCR API:    http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Grafana:    http://localhost:3000  (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Prefect:    http://localhost:4200"

docker-down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf runs/ mlruns/ mlartifacts/ __pycache__ .pytest_cache coverage_html
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
