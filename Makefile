# WattCast — common tasks. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help install data train eval predict pipeline demo web dev deploy lint test clean

PY ?= python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python deps (runtime + dev)
	pip install -r requirements-dev.txt

data: ## Ingest + build features (synthetic, offline)
	$(PY) -m ml.ingest --synthetic --start 2022-01-01 --end 2024-12-31
	$(PY) -m ml.features

train: ## Train the production model (XGBoost)
	$(PY) -m ml.train

eval: ## Walk-forward bake-off (XGBoost vs Prophet)
	$(PY) -m ml.evaluate

predict: ## Generate the J+1 forecast + dashboard JSON
	$(PY) -m ml.predict

pipeline: train eval predict ## Train → evaluate → predict (assumes `make data` ran)

demo: data pipeline ## Full offline run from scratch (synthetic data)

web: ## Build the static dashboard (bun)
	cd web && bun install && bun run build

dev: ## Serve the dashboard locally (bun)
	cd web && bun run dev

deploy: ## Build + deploy to Cloudflare (needs CLOUDFLARE_* secrets)
	cd web && bun run deploy

lint: ## Lint the ML package (ruff)
	ruff check ml

test: ## Run the test suite (pytest)
	pytest

clean: ## Remove generated artefacts
	rm -rf data model web/dist web/.astro
