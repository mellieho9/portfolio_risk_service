# Portfolio Risk Service — Progress

_Last updated: 2026-05-03_

---

## Phase 1 — Harness Setup ✅ (2026-04-12, commit c45683a)

- [x] `pyproject.toml` — ruff + mypy + bandit config
- [x] `.pre-commit-config.yaml` + `pre-commit install`
- [x] `Makefile` — `make check`
- [x] `requirements.txt` + `requirements-dev.txt`
- [x] `.env.example`
- [x] Project skeleton (`app/`, `training/`, `tests/`)

---

## Phase 2 — Offline Training ⚠️ Code written (2026-04-12, commit 221f831), not yet validated

Files written:
- [x] `training/generate_scenarios.py` — CuPy RawKernel GBM, Parquet shards to S3
- [x] `app/features.py` — 35-dim feature vector, analytical BS Greeks
- [x] `training/train.py` — RiskEmulator MLP (residual + SiLU), quantile consistency loss, TorchScript export
- [x] `training/evaluate.py` — MAE by target, VaR R², inference speedup report

Assessment checklist (requires EC2 Spot run — none completed yet):

**Data quality**
- [ ] Feature distributions: no NaN/Inf
- [ ] Target distributions: VaR < 0 for most; ES ≤ VaR; q05 < mean < q95
- [ ] Scenario diversity across all (horizon, confidence) pairs

**Training convergence**
- [ ] Val loss decreases monotonically; best checkpoint selected by early stopping
- [ ] No significant train/val gap (> 2x)
- [ ] Quantile consistency penalty → 0 by end of training

**NN accuracy vs MC ground truth**
- [ ] VaR R² ≥ 0.90 on holdout
- [ ] ES R² ≥ 0.85 on holdout
- [ ] MAE(VaR) < 2% of mean portfolio notional
- [ ] VaR over-estimation rate < 10%

**Tail calibration**
- [ ] 95% confidence: fraction where `pred_var > mc_var` ≈ 0.05 ± 0.02
- [ ] 99% confidence: same check, tolerance ± 0.03
- [ ] ES error concentrated in body, not tails

**Inference speedup**
- [ ] NN batch (10K) < 50ms CPU, < 10ms GPU
- [ ] Single-scenario latency < 5ms
- [ ] Speedup ratio vs MC recorded

**Regime stress tests**
- [ ] High-vol (vol > 0.4): MAE(VaR) within 5% of notional
- [ ] Long-horizon (63-day): errors not significantly larger than 21-day
- [ ] Options-heavy (option_weight > 0.5): Greeks features well-scaled

---

## Phase 3 — API + DB ❌ Not started

- [ ] `docker-compose.yml` + `Dockerfile`
- [ ] `app/db/schema.sql` + TimescaleDB up
- [ ] `app/emulator.py` — pull from S3, TorchScript inference
- [ ] `app/data/plaid.py` + `app/data/market.py` — DuckDB caching
- [ ] `app/main.py` — FastAPI app, `POST /risk/portfolio`

---

## Phase 4 — Monitoring + Realization ❌ Not started

- [ ] `app/jobs/realize_pnl.py` — nightly cron
- [ ] `GET /risk-monitor/summary`
- [ ] `GET /risk-monitor/timeseries`

---

## Phase 5 — Validation ❌ Not started (needs Phases 3+4 live)

- [ ] Track breach rate over 1–3 months vs target confidence level
- [ ] Retrain quarterly: EC2 Spot → S3 → restart API
