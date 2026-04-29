.PHONY: install start stop clean test lint help add-model retrain eval health

help:
	@echo "Available commands:"
	@echo ""
	@echo "  Core:"
	@echo "    make install     - Build all Docker images"
	@echo "    make start       - Start all services"
	@echo "    make stop        - Stop all services"
	@echo "    make clean       - Remove containers and artifacts"
	@echo "    make health      - Check health of all model services"
	@echo ""
	@echo "  Testing:"
	@echo "    make test        - Run system tests"
	@echo "    make lint        - Run linting (black + flake8)"
	@echo ""
	@echo "  Model Management:"
	@echo "    make add-model NAME=my_model MEDIA_TYPE=image PORT=5008"
	@echo "                     - Register a new model (config + compose + scaffold)"
	@echo ""
	@echo "  Ensemble Training:"
	@echo "    make retrain MEDIA_TYPE=image"
	@echo "                     - Retrain ensemble (generate features + train meta-learner)"
	@echo "    make eval MEDIA_TYPE=image"
	@echo "                     - Re-train from existing features (skip generation)"

install:
	@echo "Building Docker images..."
	docker compose build

start:
	@echo "Starting DeepSafe..."
	docker compose up -d
	@echo "DeepSafe is running at http://localhost:8888"

stop:
	@echo "Stopping DeepSafe..."
	docker compose down

clean:
	@echo "Cleaning up..."
	docker compose down -v
	rm -rf __pycache__
	rm -rf .pytest_cache

test:
	@echo "Running system tests..."
	docker compose up -d api
	docker cp test_system.py deepsafe-api:/app/
	docker cp test_samples deepsafe-api:/app/
	docker exec deepsafe-api python test_system.py

lint:
	@echo "Running linters..."
	black --check api/main.py api/database.py
	flake8 api/main.py api/database.py --select=E9,F63,F7,F82,F401,F811,F841 --max-line-length 120

health:
	@python3 scripts/health_check.py

add-model:
	@test -n "$(NAME)" || (echo "ERROR: NAME required. Usage: make add-model NAME=my_model MEDIA_TYPE=image PORT=5008" && exit 1)
	@test -n "$(MEDIA_TYPE)" || (echo "ERROR: MEDIA_TYPE required (image|video|audio)" && exit 1)
	@test -n "$(PORT)" || (echo "ERROR: PORT required" && exit 1)
	python3 scripts/add_model.py --name $(NAME) --media-type $(MEDIA_TYPE) --port $(PORT)

retrain:
	@test -n "$(MEDIA_TYPE)" || (echo "ERROR: MEDIA_TYPE required. Usage: make retrain MEDIA_TYPE=image" && exit 1)
	python3 scripts/retrain_pipeline.py --media-type $(MEDIA_TYPE) $(if $(DATASET_DIR),--dataset-dir $(DATASET_DIR),) $(if $(OPTIMIZER),--optimizer $(OPTIMIZER),) $(if $(TRIALS),--optuna-trials $(TRIALS),)

eval:
	@test -n "$(MEDIA_TYPE)" || (echo "ERROR: MEDIA_TYPE required. Usage: make eval MEDIA_TYPE=image" && exit 1)
	python3 scripts/retrain_pipeline.py --media-type $(MEDIA_TYPE) --skip-generate $(if $(META_CSV),--meta-csv $(META_CSV),)
