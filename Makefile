.PHONY: help install install-dev test lint format docker-up docker-down train backtest dashboard live clean

help:
	@echo "AegisQuant - common dev tasks"
	@echo ""
	@echo "  make install       Install runtime dependencies"
	@echo "  make install-dev   Install runtime + dev dependencies (pre-commit, ruff)"
	@echo "  make test          Run the pytest suite (mocked, no API keys needed)"
	@echo "  make lint          Lint with ruff (no auto-fix)"
	@echo "  make format        Auto-fix style with ruff format + ruff check --fix"
	@echo "  make docker-up     Start the full docker-compose stack (trader + dashboard)"
	@echo "  make docker-down   Stop the docker-compose stack"
	@echo "  make train         Train PPO via curriculum walk-forward on real Nifty50 data"
	@echo "  make backtest      Run the multi-fold walk-forward backtester"
	@echo "  make dashboard     Launch the Streamlit command center on http://localhost:8501"
	@echo "  make live          Run a single live trading cycle now (--now flag)"
	@echo "  make clean         Remove Python caches, pytest cache, and stale model zips"

install:
	pip install -r requirements.txt

install-dev: install
	pip install pre-commit ruff
	pre-commit install

test:
	python -m pytest tests/ -q

lint:
	ruff check .

format:
	ruff format .
	ruff check --fix .

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

train:
	python train_rl.py

backtest:
	python src/backtest/walk_forward.py --algo PPO --mc-sims 10000

dashboard:
	streamlit run src/ui/dashboard.py

live:
	python main.py --now

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -f pytest_*.txt
