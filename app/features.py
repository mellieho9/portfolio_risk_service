from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

FEATURE_DIM = 35


@dataclass
class Position:
    ticker: str
    asset_type: str
    notional: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0


@dataclass
class MarketState:
    return_21d: float
    realized_vol_21d: float
    atm_implied_vol: float
    skew_proxy: float
    risk_free_rate: float


@dataclass
class RiskConfig:
    horizon_days: int
    confidence: float


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    contracts: int,
    contract_multiplier: int = 100,
) -> tuple[float, float, float, float]:
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0, 0.0, 0.0, 0.0

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    pdf_d1 = _norm_pdf(d1)
    scale = float(contracts * contract_multiplier)
    sign = 1.0 if option_type == "call" else -1.0

    delta = sign * _norm_cdf(sign * d1) * scale
    gamma = pdf_d1 / (S * sigma * math.sqrt(T)) * scale
    vega = S * pdf_d1 * math.sqrt(T) * scale / 100.0
    theta = (
        -(S * pdf_d1 * sigma) / (2.0 * math.sqrt(T))
        - sign * r * K * math.exp(-r * T) * _norm_cdf(sign * d2)
    ) * scale / 365.0

    return delta, gamma, vega, theta


def build_feature_vector(
    positions: list[Position],
    market: MarketState,
    config: RiskConfig,
) -> npt.NDArray[np.float32]:
    total_notional = sum(abs(p.notional) for p in positions) + 1e-8
    equity_notional = sum(abs(p.notional) for p in positions if p.asset_type == "equity")
    option_notional = sum(abs(p.notional) for p in positions if p.asset_type == "option")

    port_delta = sum(p.delta for p in positions)
    port_gamma = sum(p.gamma for p in positions)
    port_vega = sum(p.vega for p in positions)
    port_theta = sum(p.theta for p in positions)

    vol_of_vol = (
        abs(market.skew_proxy) / market.atm_implied_vol
        if market.atm_implied_vol > 0
        else 0.0
    )

    feat: list[float] = [
        math.log(total_notional + 1.0),
        equity_notional / total_notional,
        option_notional / total_notional,
        port_delta / total_notional,
        port_gamma / total_notional,
        port_vega / total_notional,
        port_theta / total_notional,
        market.return_21d,
        market.realized_vol_21d,
        market.atm_implied_vol,
        market.skew_proxy,
        market.risk_free_rate,
        vol_of_vol,
        math.log(max(config.horizon_days, 1)),
        config.confidence,
    ]

    feat += [0.0] * (FEATURE_DIM - len(feat))
    return np.array(feat, dtype=np.float32)
