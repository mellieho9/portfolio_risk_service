# Setup & Operations

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (for TimescaleDB)
- AWS account with S3 bucket for model artifacts
- Plaid developer account
- Alpha Vantage API key

## Installation

```bash
pip install -r requirements.txt
pre-commit install
cp .env.example .env  # fill in values below
```

## Environment Variables

| Variable | Purpose |
|---|---|
| `PLAID_ACCESS_TOKEN` | Plaid Investments API |
| `ALPHA_VANTAGE_API_KEY` | Market data (25 req/day free tier) |
| `DB_URL` | PostgreSQL connection string (TimescaleDB) |
| `S3_MODEL_URI` | S3 path to `model.pt` + `scaler.pkl` |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | S3 access for model artifacts |
| `DB_PASSWORD` | Postgres password (used by Docker Compose) |

## Key Commands

```bash
# Quality checks
make check        # run all: lint, types, security, dead code
make fmt          # ruff format
make lint         # ruff check
make typecheck    # mypy --strict
make security     # bandit -ll
make deadcode     # vulture --min-confidence 80

# Run locally
docker compose up

# Nightly realization job (also runs via cron 06:00 Mon-Fri)
python app/jobs/realize_pnl.py
```

## EC2 Spot Training Run (Phase 2)

1. Launch a `g4dn.xlarge` Spot instance with a Deep Learning AMI (CUDA + PyTorch pre-installed)
2. Clone repo, `pip install -r requirements.txt`
3. Export `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_MODEL_URI`
4. Run in sequence:
   ```bash
   python training/generate_scenarios.py
   python training/train.py
   python training/evaluate.py
   ```
5. Artifacts (`model.pt`, `scaler.pkl`) are pushed to S3 automatically — instance can be terminated after
