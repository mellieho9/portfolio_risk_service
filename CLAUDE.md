# CLAUDE.md — Portfolio Risk Service

## Project

Personal portfolio risk monitoring service. Replaces expensive Monte Carlo simulation at inference time with a trained PyTorch neural network surrogate. Exposes VaR/ES/P&L forecasts via a FastAPI HTTP API, logs every forecast to TimescaleDB, and runs a nightly cron to fill in realized P&L and flag VaR breaches. Live portfolio data comes from Plaid; market data from Alpha Vantage (DuckDB-cached). AWS is used only for GPU training (EC2 Spot g4dn.xlarge) and model artifact storage (S3).

---

## Key Documents

| File | Contents |
|---|---|
| [design.md](design.md) | Full architecture and build spec: five-phase plan, training pipeline, NN surrogate design, API endpoints, TimescaleDB schema, DuckDB cache, Docker Compose setup, and code quality harness. |
| [progress.md](progress.md) | Build state and per-phase checklists. Phases 1–2 complete; Phase 2 unvalidated (no EC2 Spot run yet). Phases 3–5 not started. |
| [setup.md](setup.md) | Prerequisites, env vars, key commands, and step-by-step EC2 Spot training run instructions. |

---

## Repository Structure

```
portfolio-risk-service/
├── cuda_engine/          # CUDA C++ engine (MC, Black-Scholes, CRR binomial)
├── training/
│   ├── generate_scenarios.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── main.py           # FastAPI
│   ├── emulator.py       # inference wrapper
│   ├── features.py       # 35-dim feature vector
│   ├── data/             # plaid.py, market.py
│   ├── db/               # schema.sql, queries.py
│   └── jobs/realize_pnl.py
├── models/               # gitignored, pulled from S3 on first run
├── market_cache.duckdb   # gitignored
├── docker-compose.yml
├── pyproject.toml
└── Makefile
```
