"""Generate portfolio risk training scenarios on GPU and upload to S3 as Parquet.

Uses a CuPy RawKernel (GBM) to simulate equity portfolio paths — same pattern
as the existing barrier option engine (JIT-compiled, no nvcc step).

Usage:
    python training/generate_scenarios.py --n-scenarios 1000000 --out-prefix s3://bucket/data/
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import time
from dataclasses import dataclass

import boto3
import cupy as cp
import numpy as np
import pandas as pd

N_PATHS = 32_000
N_STEPS_PER_DAY = 1

_GBM_KERNEL = cp.RawKernel(
    r"""
extern "C" __global__
void gbm_portfolio_paths(
    const float* __restrict__ weights,   // [N_ASSETS]
    const float* __restrict__ vols,      // [N_ASSETS]
    const float* __restrict__ randoms,   // [N_PATHS * N_STEPS * N_ASSETS]
    float*       __restrict__ pnl,       // [N_PATHS]  output
    float mu, float r, float dt,
    int N_PATHS, int N_STEPS, int N_ASSETS
) {
    int path = blockIdx.x * blockDim.x + threadIdx.x;
    if (path >= N_PATHS) return;

    float portfolio_pnl = 0.0f;
    for (int a = 0; a < N_ASSETS; a++) {
        float log_s = 0.0f;
        float drift = (mu - 0.5f * vols[a] * vols[a]) * dt;
        for (int t = 0; t < N_STEPS; t++) {
            int idx = path * N_STEPS * N_ASSETS + t * N_ASSETS + a;
            log_s += drift + vols[a] * sqrtf(dt) * randoms[idx];
        }
        portfolio_pnl += weights[a] * (expf(log_s) - 1.0f);
    }
    pnl[path] = portfolio_pnl;
}
""",
    "gbm_portfolio_paths",
)


@dataclass
class Scenario:
    weights: list[float]
    vols: list[float]
    mu: float
    r: float
    horizon_days: int
    confidence: float


def _run_mc(scenario: Scenario) -> dict[str, float]:
    n_assets = len(scenario.weights)
    n_steps = scenario.horizon_days * N_STEPS_PER_DAY
    dt = 1.0 / 252.0

    weights_gpu = cp.array(scenario.weights, dtype=cp.float32)
    vols_gpu = cp.array(scenario.vols, dtype=cp.float32)
    randoms_gpu = cp.random.normal(0, 1, N_PATHS * n_steps * n_assets, dtype=cp.float32)
    pnl_gpu = cp.zeros(N_PATHS, dtype=cp.float32)

    threads = 256
    blocks = math.ceil(N_PATHS / threads)
    _GBM_KERNEL(
        (blocks,),
        (threads,),
        (
            weights_gpu,
            vols_gpu,
            randoms_gpu,
            pnl_gpu,
            np.float32(scenario.mu),
            np.float32(scenario.r),
            np.float32(dt),
            np.int32(N_PATHS),
            np.int32(n_steps),
            np.int32(n_assets),
        ),
    )
    cp.cuda.Device().synchronize()

    pnl = pnl_gpu.get()
    pnl_sorted = np.sort(pnl)
    var_idx = int((1.0 - scenario.confidence) * N_PATHS)

    var = float(pnl_sorted[var_idx])
    es = float(pnl_sorted[:var_idx].mean()) if var_idx > 0 else var

    return {
        "var": var,
        "es": es,
        "mean_pnl": float(pnl.mean()),
        "vol_pnl": float(pnl.std()),
        "q05_pnl": float(np.percentile(pnl, 5)),
        "q95_pnl": float(np.percentile(pnl, 95)),
    }


def _sample_scenario(rng: np.random.Generator) -> Scenario:
    n_assets = int(rng.integers(1, 6))
    raw_weights = rng.uniform(0.0, 1.0, n_assets)
    weights = (raw_weights / raw_weights.sum()).tolist()
    vols = rng.uniform(0.10, 0.60, n_assets).tolist()
    mu = float(rng.uniform(0.02, 0.15))
    r = float(rng.uniform(0.01, 0.06))
    horizon_days = int(rng.choice([21, 63]))
    confidence = float(rng.choice([0.95, 0.99]))
    return Scenario(weights=weights, vols=vols, mu=mu, r=r,
                    horizon_days=horizon_days, confidence=confidence)


def _scenario_to_features(s: Scenario, targets: dict[str, float]) -> dict[str, object]:
    n_assets = len(s.weights)
    avg_vol = float(np.mean(s.vols))
    port_vol = float(np.sqrt(np.sum(np.array(s.weights) ** 2 * np.array(s.vols) ** 2)))
    return {
        "total_notional_log": math.log(1.0 + n_assets),
        "equity_weight": 1.0,
        "option_weight": 0.0,
        "port_delta": 1.0,
        "port_gamma": 0.0,
        "port_vega": 0.0,
        "port_theta": 0.0,
        "return_21d": float(s.mu * 21 / 252),
        "realized_vol_21d": port_vol * math.sqrt(21 / 252),
        "atm_implied_vol": avg_vol,
        "skew_proxy": 0.0,
        "risk_free_rate": s.r,
        "vol_of_vol": 0.0,
        "log_horizon": math.log(s.horizon_days),
        "confidence": s.confidence,
        **{f"pad_{i}": 0.0 for i in range(20)},
        **targets,
    }


def _upload_parquet(df: pd.DataFrame, s3_uri: str, shard_id: int) -> None:
    bucket, prefix = s3_uri.replace("s3://", "").split("/", 1)
    key = f"{prefix}shard_{shard_id:04d}.parquet"
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    boto3.client("s3").upload_fileobj(buf, bucket, key)
    print(f"  uploaded s3://{bucket}/{key}  ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-scenarios", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--out-prefix", type=str, default=os.environ.get("S3_DATA_URI", ""))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    shard_id = 0
    t0 = time.time()

    for i in range(args.n_scenarios):
        scenario = _sample_scenario(rng)
        cp.random.seed(i)
        targets = _run_mc(scenario)
        rows.append(_scenario_to_features(scenario, targets))

        if len(rows) >= args.shard_size:
            df = pd.DataFrame(rows)
            config_hash = hashlib.md5(str(args.seed).encode()).hexdigest()[:8]
            df["config_hash"] = config_hash
            if args.out_prefix:
                _upload_parquet(df, args.out_prefix, shard_id)
            else:
                df.to_parquet(f"shard_{shard_id:04d}.parquet", index=False)
            rows = []
            shard_id += 1
            elapsed = time.time() - t0
            print(f"[{i+1}/{args.n_scenarios}] {elapsed:.1f}s elapsed")

    if rows:
        df = pd.DataFrame(rows)
        if args.out_prefix:
            _upload_parquet(df, args.out_prefix, shard_id)
        else:
            df.to_parquet(f"shard_{shard_id:04d}.parquet", index=False)


if __name__ == "__main__":
    main()
