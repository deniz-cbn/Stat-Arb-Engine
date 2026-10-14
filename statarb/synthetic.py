"""Synthetic universe with *known* cointegrated pairs, for testing and the demo (no data download needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ou import simulate_ou


def make_universe(n_days: int = 1500, n_pairs: int = 3, n_noise: int = 6, seed: int = 0,
                  kappa_range=(0.03, 0.10), spread_sigma_eq: float = 0.03,
                  market_vol: float = 0.009, idio_vol: float = 0.011, start="2018-01-01"):
    """
    Log-price construction
        trend_k(t) = beta_k * M(t) + idiosyncratic random walk      (I(1), driven partly by market M)
        B_k = trend_k                                               (leg 2)
        A_k = gamma_k * trend_k + S_k(t) + c_k                      (leg 1),  S_k ~ OU
    so  log A_k - gamma_k log B_k = S_k + c_k  is stationary  =>  cointegration vector (1, -gamma_k).
    Noise assets are independent I(1) processes (never cointegrated).
    Returns (prices DataFrame, truth dict).
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    M = np.cumsum(rng.normal(0.0003, market_vol, n_days))
    cols, truth = {}, {}

    def trend():
        beta_m = rng.uniform(0.6, 1.4)
        return beta_m * M + np.cumsum(rng.normal(0.0001, idio_vol, n_days)) + rng.uniform(3.5, 5.0)

    for k in range(n_pairs):
        t = trend()
        gamma = rng.uniform(0.7, 1.3)
        kappa = rng.uniform(*kappa_range)
        sigma = spread_sigma_eq * np.sqrt(2 * kappa)
        S = simulate_ou(kappa, 0.0, sigma, n_days, rng=rng)
        cols[f"A{k}"] = gamma * t + S + rng.uniform(-0.5, 0.5)
        cols[f"B{k}"] = t
        truth[(f"A{k}", f"B{k}")] = dict(gamma=gamma, kappa=kappa, half_life=np.log(2) / kappa,
                                         sigma_eq=spread_sigma_eq)
    for k in range(n_noise):
        cols[f"N{k}"] = trend()
    prices = np.exp(pd.DataFrame(cols, index=idx))
    return prices, truth
