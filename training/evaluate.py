"""Validate NN emulator accuracy vs MC ground truth and report speedup.

Usage:
    python training/evaluate.py --model-dir models/ --data-prefix s3://bucket/data/
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import time

import boto3
import numpy as np
import pandas as pd
import torch

from training.train import FEATURE_COLS, TARGET_COLS, RiskEmulator, _load_parquet_from_s3

N_EVAL = 10_000


def _load_model(model_dir: str) -> tuple[torch.jit.ScriptModule, object]:
    if model_dir.startswith("s3://"):
        bucket, prefix = model_dir.replace("s3://", "").split("/", 1)
        s3 = boto3.client("s3")

        model_buf = io.BytesIO()
        s3.download_fileobj(bucket, f"{prefix}model.pt", model_buf)
        model_buf.seek(0)
        model = torch.jit.load(model_buf)

        scaler_buf = io.BytesIO()
        s3.download_fileobj(bucket, f"{prefix}scaler.pkl", scaler_buf)
        scaler_buf.seek(0)
        scaler = pickle.load(scaler_buf)  # noqa: S301
    else:
        model = torch.jit.load(f"{model_dir}/model.pt")
        with open(f"{model_dir}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)  # noqa: S301

    return model, scaler


def _mae_by_target(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    return {col: float(np.abs(pred[:, i] - true[:, i]).mean())
            for i, col in enumerate(TARGET_COLS)}


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading data...")
    if args.data_prefix.startswith("s3://"):
        df = _load_parquet_from_s3(args.data_prefix)
    else:
        import glob
        df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{args.data_prefix}*.parquet")])

    df = df.sample(min(N_EVAL, len(df)), random_state=0)
    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_true = df[TARGET_COLS].values.astype(np.float32)

    model, scaler = _load_model(args.model_dir)
    model = model.to(device)  # type: ignore[assignment]
    model.eval()

    X = scaler.transform(X_raw).astype(np.float32)
    X_t = torch.tensor(X).to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred_t = model(X_t)
    nn_ms = (time.perf_counter() - t0) * 1000

    pred = pred_t.cpu().numpy()

    mae = _mae_by_target(pred, y_true)
    var_true = y_true[:, 0]
    var_pred = pred[:, 0]
    r2_var = float(1 - np.var(var_pred - var_true) / (np.var(var_true) + 1e-8))

    print(f"\nNN inference: {nn_ms:.1f}ms for {N_EVAL} scenarios ({nn_ms/N_EVAL*1000:.1f}µs each)")
    print(f"\nMAE by target:")
    for col, err in mae.items():
        print(f"  {col:<14} {err:.4f}")
    print(f"\nVaR R²: {r2_var:.4f}")

    breach_rate = float((pred[:, 0] > y_true[:, 0]).mean())
    print(f"VaR over-estimation rate: {breach_rate:.3f} (want < 0.10)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("S3_MODEL_URI", "models/"))
    parser.add_argument("--data-prefix", default=os.environ.get("S3_DATA_URI", ""))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
