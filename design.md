
# Portfolio Risk Service — Design Document

**Status:** Draft  
**Date:** 2026-04-12  
**Scope:** Internal single-user service for personal portfolio risk monitoring

---

## 1. Goals

Extend the existing GPU option pricing engine (CUDA C++, Monte Carlo, Black-Scholes, CRR binomial, autograd Greeks) into a personal risk dashboard:

- Replace expensive Monte Carlo at inference time with a trained neural network surrogate (same approach as the 231x option pricing speedup).
- Expose risk metrics via a small HTTP API.
- Log every forecast and realized P&L to TimescaleDB for backtesting and breach monitoring.
- Pull live portfolio and market data from Plaid and Alpha Vantage.

**Design principle:** local-first, open source everything. AWS only for GPU training (EC2 Spot, spin up → train → terminate) and model artifact storage (S3). No always-on cloud infra.

---

## 2. Stack

| Layer | Tool | Type |
|---|---|---|
| API | FastAPI + Uvicorn | Open source |
| ML | PyTorch (TorchScript) | Open source |
| Simulation engine | CUDA C++ (existing) | Open source |
| Database | TimescaleDB (Docker) | Open source |
| Market data cache | DuckDB | Open source |
| Containerization | Docker + Docker Compose | Open source |
| Scheduler | Linux cron | Open source |
| Linter + formatter | Ruff | Open source |
| Type checker | mypy (strict) | Open source |
| Security analysis | bandit | Open source |
| Dead code detection | vulture | Open source |
| Commit gates | pre-commit | Open source |
| Secrets | python-dotenv | Open source |
| **Model storage** | S3 | AWS |
| **Training GPU** | EC2 Spot g4dn.xlarge (~$0.16/hr) | AWS |

**Optional:** Ray Serve in place of plain Uvicorn for zero-downtime model hot-reloading. Not included by default.

**Monthly cost:** ~$0 at rest. EC2 Spot training run (~2–4 hrs) ≈ $0.65 per retrain.

---

## 3. High-Level Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Data Layer                            │
│  Plaid API ──► holdings          Alpha Vantage ──► OHLCV │
│                    │                      │               │
│                    └──────────┬───────────┘               │
│                          Feature Builder                   │
│                         (features.py)                     │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│                  NN Emulator (PyTorch)                    │
│  Input:  portfolio state + market state + risk config     │
│  Output: VaR, ES, mean P&L, vol P&L, quantiles           │
│  Loaded in-process by FastAPI on startup (pulled from S3) │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│              Risk API  (FastAPI + Uvicorn)                │
│  POST /risk/portfolio        → risk snapshot              │
│  GET  /risk-monitor/summary  → breach rate + health       │
│  GET  /risk-monitor/timeseries → VaR vs realized P&L      │
└──────────────────────────────┬───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│         TimescaleDB (Postgres + TimescaleDB ext)          │
│  risk_forecasts hypertable                                │
│  nightly cron fills realized_pnl + breach flag            │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Components

### 4.1 MC Simulation Engine (existing — offline only)

Used only to generate training data. Never called at inference time.

| Model | Usage |
|---|---|
| Monte Carlo (32M paths) | Training data generation — equity + options |
| Black-Scholes | Analytical baseline; fast Greeks for option positions |
| CRR Binomial | American options, path-dependent payoffs |
| Autograd Greeks | Delta/Gamma/Vega per position for feature construction |

**Training data generation pipeline:**

```
For each synthetic portfolio scenario:
  1. Sample portfolio weights (equity, option strikes/expiries)
  2. Sample market state (spot, vol surface, rates)
  3. Sample risk config (horizon ∈ {21, 63} days, confidence ∈ {0.95, 0.99})
  4. Run GPU MC → compute VaR, ES, mean P&L, vol P&L, p05/p50/p95
  5. Save (features, targets) as a Parquet row
```

Run on **EC2 Spot g4dn.xlarge**. Output ~1–5M scenarios as Parquet shards, stored in S3. Terminate instance when done.

**Feature vector (~35 dims):**
- Portfolio: notional per asset, delta, gamma, vega, theta (Greeks via autograd)
- Market: 21d rolling return, 21d realized vol, ATM implied vol, skew proxy, risk-free rate
- Config: horizon (log-scaled), confidence level

**Target vector (6 values):** `VaR, ES, mean_pnl, vol_pnl, q05_pnl, q95_pnl`

---

### 4.2 Neural Network Surrogate (PyTorch)

**Architecture:**

```
Input(N_features)
  → BatchNorm1d
  → Linear(256) → SiLU
  → Linear(256) → SiLU  ─┐ residual
  → Linear(256) → SiLU  ←┘
  → Linear(128) → SiLU
  → Linear(6)             [VaR, ES, mean, vol, q05, q95]
```

**Training:**

```python
loss = MSE(pred, target) + 0.1 * quantile_consistency_penalty(pred)
# penalty enforces: q05 < mean < q95, VaR <= ES
```

- Optimizer: AdamW, cosine LR schedule
- Batch size: 2048, ~50 epochs with early stopping
- Run on EC2 Spot g4dn.xlarge, terminate after

**Artifacts pushed to S3 after training:**
- `model.pt` — TorchScript compiled
- `scaler.pkl` — StandardScaler for input normalization
- `model_version` — hash of config + training data fingerprint

On API startup: pull artifacts from S3 if not already in `./models/`.

---

### 4.3 Risk API (FastAPI + Uvicorn)

Model loaded once into memory at startup. Inference is in-process — no separate model server.

#### `POST /risk/portfolio`

**Request:**
```json
{
  "portfolio_id": "personal_main",
  "positions": [
    {"ticker": "AAPL", "shares": 10, "asset_type": "equity"},
    {"ticker": "SPY",  "shares": 5,  "asset_type": "equity"},
    {
      "ticker": "SPY", "strike": 480, "expiry": "2026-06-20",
      "option_type": "put", "contracts": 2, "asset_type": "option"
    }
  ],
  "risk_config": {
    "horizon_days": 21,
    "confidence": 0.95
  }
}
```

**Response:**
```json
{
  "forecast_id": "f_20260412_143022_a3f1",
  "timestamp": "2026-04-12T14:30:22Z",
  "model_version": "v1.2_abc123",
  "var": -1842.50,
  "es": -2310.00,
  "mean_pnl": 120.30,
  "vol_pnl": 980.00,
  "quantiles": { "p05": -2100.00, "p50": 115.00, "p95": 2300.00 }
}
```

**Internal flow per request:**
1. Fetch current prices from Alpha Vantage (DuckDB cache, TTL = 5 min for quotes, daily batch refresh for OHLCV).
2. Fetch holdings from Plaid (cached 15 min); request body positions override/supplement.
3. Compute Greeks for option positions via Black-Scholes (analytical).
4. Build feature vector.
5. Run NN emulator (< 5ms).
6. Write forecast row to TimescaleDB.
7. Return response.

#### `GET /risk-monitor/summary`

```json
{
  "window_days": 90,
  "total_forecasts": 87,
  "breaches": 4,
  "breach_rate": 0.046,
  "target_breach_rate": 0.05,
  "status": "ok"
}
```

#### `GET /risk-monitor/timeseries?days=30&portfolio_id=personal_main`

```json
{
  "series": [
    { "ts": "2026-03-01T00:00:00Z", "var": -1800.00, "realized_pnl": -950.00, "breach": false },
    { "ts": "2026-03-02T00:00:00Z", "var": -1750.00, "realized_pnl": null,    "breach": null }
  ]
}
```

---

### 4.4 TimescaleDB Schema

```sql
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE risk_forecasts (
    forecast_id     TEXT        NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    portfolio_id    TEXT        NOT NULL,

    horizon_days    INT         NOT NULL,
    confidence      FLOAT       NOT NULL,

    var             FLOAT       NOT NULL,
    es              FLOAT       NOT NULL,
    pred_mean_pnl   FLOAT       NOT NULL,
    pred_vol_pnl    FLOAT       NOT NULL,
    q05_pnl         FLOAT,
    q95_pnl         FLOAT,

    realized_pnl    FLOAT,       -- filled by nightly cron
    breach          BOOLEAN,     -- NULL until realized_pnl is set

    model_version   TEXT        NOT NULL,
    config_hash     TEXT        NOT NULL,

    PRIMARY KEY (forecast_id, ts)
);

SELECT create_hypertable('risk_forecasts', 'ts');
CREATE INDEX ON risk_forecasts (portfolio_id, ts DESC);
```

**Nightly realization cron** (`jobs/realize_pnl.py`):

```
0 6 * * 1-5  python /app/jobs/realize_pnl.py
```

```
For each forecast where horizon has elapsed and realized_pnl IS NULL:
  1. Pull prices at forecast_ts and (forecast_ts + horizon_days) from DuckDB cache
  2. Compute realized portfolio P&L
  3. UPDATE risk_forecasts SET realized_pnl = ..., breach = (realized_pnl < var)
```

---

### 4.5 Market Data Cache (DuckDB)

DuckDB runs embedded — no server process, no extra container.

```
market_cache.duckdb
├── daily_ohlcv(ticker, date, open, high, low, close, volume)  -- refreshed nightly
└── rt_quotes(ticker, price, fetched_at)                       -- TTL 5 min, replaced on read
```

Alpha Vantage free tier: 25 req/day. Strategy:
- Batch-fetch all portfolio tickers in one nightly cron run → store in `daily_ohlcv`.
- Real-time quotes fetched lazily per request, cached 5 min in `rt_quotes`.

---

### 4.6 Data Sources

**Plaid**
- Product: Investments — holdings, shares, cost basis.
- Refresh: cached 15 min in memory.
- Auth: `PLAID_ACCESS_TOKEN` in `.env`.

**Alpha Vantage**
- `TIME_SERIES_DAILY` for OHLCV, `GLOBAL_QUOTE` for real-time prices.
- Auth: `ALPHA_VANTAGE_API_KEY` in `.env`.

---

## 5. Code Quality Harness

All checks run as pre-commit hooks — bad code never lands.

### 5.1 Ruff (lint + format + complexity)

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes (unused imports, undefined names)
    "I",    # isort
    "N",    # pep8 naming
    "UP",   # pyupgrade (modernize syntax)
    "B",    # flake8-bugbear (likely bugs)
    "SIM",  # flake8-simplify
    "C90",  # McCabe complexity
]

[tool.ruff.mccabe]
max-complexity = 10  # hard cap — forces decomposition of complex logic
```

### 5.2 mypy (strict type checking)

```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
disallow_any_generics = true
warn_return_any = true
warn_unused_ignores = true  # no silent type: ignore escapes
```

Catches the most common class of AI slop: plausible-looking but type-unsafe code.

### 5.3 bandit (security analysis)

```toml
[tool.bandit]
targets = ["app"]
severity = "medium"   # fail on medium+ (catches hardcoded secrets, SQL injection, unsafe pickle)
```

### 5.4 vulture (dead code detection)

```bash
vulture app/ --min-confidence 80
```

Catches AI-generated functions and classes that were never wired up.

### 5.5 pre-commit config

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.10.0
    hooks:
      - id: mypy
        args: [--strict]
        additional_dependencies: [types-all]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.8
    hooks:
      - id: bandit
        args: [-ll, -r, app/]

  - repo: https://github.com/jendrikseipp/vulture
    rev: v2.11
    hooks:
      - id: vulture
        args: [app/, --min-confidence, "80"]
```

Install once with: `pre-commit install`

### 5.6 Makefile

```makefile
.PHONY: lint typecheck security deadcode check fmt

fmt:
	ruff format app/ training/

lint:
	ruff check app/ training/

typecheck:
	mypy app/ training/ --strict

security:
	bandit -r app/ -ll

deadcode:
	vulture app/ --min-confidence 80

check: lint typecheck security deadcode
```

Run `make check` before any PR or model retrain. CI equivalent: pre-commit run --all-files.

---

## 6. Repository Structure

```
portfolio-risk-service/
├── cuda_engine/              # existing CUDA C++ engine
│   ├── monte_carlo.cu
│   ├── black_scholes.cu
│   └── binomial.cu
├── training/
│   ├── generate_scenarios.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── main.py               # FastAPI app, loads model on startup
│   ├── emulator.py           # NN wrapper (pull from S3, TorchScript inference)
│   ├── features.py           # feature construction (Greeks, rolling windows)
│   ├── data/
│   │   ├── plaid.py
│   │   └── market.py         # Alpha Vantage client + DuckDB cache
│   ├── db/
│   │   ├── schema.sql
│   │   └── queries.py
│   └── jobs/
│       └── realize_pnl.py
├── models/                   # pulled from S3 on first run, gitignored
│   ├── model.pt
│   └── scaler.pkl
├── market_cache.duckdb       # local only, gitignored
├── .pre-commit-config.yaml
├── pyproject.toml            # ruff + mypy + bandit config
├── Makefile
├── docker-compose.yml
├── Dockerfile
├── .env.example
└── requirements.txt
```

---

## 7. Docker Compose

```yaml
services:
  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: riskdb
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  api:
    build: .
    environment:
      - DB_URL=postgresql://postgres:${DB_PASSWORD}@db:5432/riskdb
      - PLAID_ACCESS_TOKEN=${PLAID_ACCESS_TOKEN}
      - ALPHA_VANTAGE_API_KEY=${ALPHA_VANTAGE_API_KEY}
      - S3_MODEL_URI=${S3_MODEL_URI}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    ports:
      - "8000:8000"
    depends_on:
      - db
    volumes:
      - ./models:/app/models
      - ./market_cache.duckdb:/app/market_cache.duckdb

volumes:
  pgdata:
```

---

## 9. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Local-first, no always-on cloud | Docker Compose locally | Personal tool; no need for 24/7 uptime |
| AWS only for GPU training | EC2 Spot g4dn.xlarge | Cheapest GPU access; ~$0.65/run |
| S3 for model artifacts | S3 | Required to transfer from EC2 Spot to local before termination |
| NN in-process, no model server | Load in FastAPI on startup | Single user, one model — Ray Serve overhead not justified |
| Ray Serve (optional) | Add only if hot-reloading needed | Worth adding if retraining frequently |
| DuckDB for market cache | DuckDB | Embedded, zero-infra, fast columnar rolling window queries |
| TimescaleDB over plain Postgres | TimescaleDB | Rolling breach rate + P&L timeseries are the core queries |
| Analytical BS Greeks at inference | Exact for vanilla options | Autograd only needed during training feature generation |
| Ruff over black + flake8 | Ruff | Single tool, 10-100x faster, same rules |
| mypy strict | Strict | Catches type-unsafe AI-generated code that passes casual review |
| pre-commit gates | pre-commit | Blocks bad code at commit time, not PR review time |

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| NN poorly calibrated for tail risk | Quantile consistency loss; validate ES breach rate on holdout set |
| Alpha Vantage rate limits (25 req/day free) | Batch nightly OHLCV refresh; real-time quotes cached 5 min |
| Plaid token expiry | Handle 400 with re-auth prompt; document refresh flow |
| Model drift over time | Log `model_version` + `config_hash` per forecast; retrain quarterly |
| Path-dependent / American options | Use CRR binomial output as training feature; flag in API response |
| AI-generated dead code / type errors | vulture + mypy strict block at commit via pre-commit |
| Hardcoded secrets or unsafe pickle in model loading | bandit -ll catches both; model loaded via TorchScript not pickle |
