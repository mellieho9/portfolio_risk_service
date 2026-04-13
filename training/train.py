"""Train the portfolio risk NN emulator and export TorchScript + scaler to S3.

Usage:
    python training/train.py --data-prefix s3://bucket/data/ --out-prefix s3://bucket/models/v1/
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import pickle
import time

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

FEATURE_COLS = [
    "total_notional_log", "equity_weight", "option_weight",
    "port_delta", "port_gamma", "port_vega", "port_theta",
    "return_21d", "realized_vol_21d", "atm_implied_vol",
    "skew_proxy", "risk_free_rate", "vol_of_vol", "log_horizon", "confidence",
    *[f"pad_{i}" for i in range(20)],
]
TARGET_COLS = ["var", "es", "mean_pnl", "vol_pnl", "q05_pnl", "q95_pnl"]


class RiskDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class RiskEmulator(nn.Module):
    def __init__(self, in_dim: int = 35, hidden: int = 256, out_dim: int = 6) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 128)
        self.fc5 = nn.Linear(128, out_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.act(self.fc1(x))
        residual = x
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x)) + residual
        x = self.act(self.fc4(x))
        return self.fc5(x)  # type: ignore[no-any-return]


def quantile_consistency_loss(pred: torch.Tensor) -> torch.Tensor:
    q05 = pred[:, 4]
    mean = pred[:, 2]
    q95 = pred[:, 5]
    var = pred[:, 0]
    es = pred[:, 1]
    penalty = (
        torch.relu(q05 - mean).mean()
        + torch.relu(mean - q95).mean()
        + torch.relu(var - es).mean()
    )
    return penalty


def _load_parquet_from_s3(prefix: str) -> pd.DataFrame:
    bucket, key_prefix = prefix.replace("s3://", "").split("/", 1)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    frames: list[pd.DataFrame] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key_prefix):
        for obj in page.get("Contents", []):
            buf = io.BytesIO()
            s3.download_fileobj(bucket, obj["Key"], buf)
            buf.seek(0)
            frames.append(pd.read_parquet(buf))
    return pd.concat(frames, ignore_index=True)


def _upload_bytes(buf: io.BytesIO, s3_uri: str) -> None:
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    buf.seek(0)
    boto3.client("s3").upload_fileobj(buf, bucket, key)


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading data...")
    if args.data_prefix.startswith("s3://"):
        df = _load_parquet_from_s3(args.data_prefix)
    else:
        import glob
        df = pd.concat([pd.read_parquet(f) for f in glob.glob(f"{args.data_prefix}*.parquet")])

    X_raw = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[TARGET_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    dataset = RiskDataset(X, y_raw)
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 4)

    model = RiskEmulator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = mse(pred, y_b) + 0.1 * quantile_consistency_loss(pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pred = model(X_b)
                val_loss += mse(pred, y_b).item() * len(X_b)
        val_loss /= len(val_ds)
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    scripted = torch.jit.script(model)
    model_buf = io.BytesIO()
    torch.jit.save(scripted, model_buf)

    scaler_buf = io.BytesIO()
    pickle.dump(scaler, scaler_buf)

    config_hash = hashlib.md5(
        f"{args.epochs}{args.batch_size}{args.lr}".encode()
    ).hexdigest()[:8]
    version = f"v{int(time.time())}_{config_hash}"
    print(f"model version: {version}  best_val={best_val:.4f}")

    if args.out_prefix.startswith("s3://"):
        _upload_bytes(model_buf, f"{args.out_prefix}model.pt")
        _upload_bytes(scaler_buf, f"{args.out_prefix}scaler.pkl")
        version_buf = io.BytesIO(version.encode())
        _upload_bytes(version_buf, f"{args.out_prefix}model_version")
        print(f"uploaded to {args.out_prefix}")
    else:
        os.makedirs(args.out_prefix, exist_ok=True)
        with open(f"{args.out_prefix}/model.pt", "wb") as f:
            f.write(model_buf.getvalue())
        with open(f"{args.out_prefix}/scaler.pkl", "wb") as f:
            f.write(scaler_buf.getvalue())
        with open(f"{args.out_prefix}/model_version", "w") as f:
            f.write(version)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-prefix", default=os.environ.get("S3_DATA_URI", ""))
    parser.add_argument("--out-prefix", default=os.environ.get("S3_MODEL_URI", "models/"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
