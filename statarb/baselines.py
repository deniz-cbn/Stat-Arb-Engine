"""Static (fit-once) pairs strategy -- the classic notebook approach, with the known pitfalls fixed.

Fixes relative to a naive implementation: log prices, dollar-weighted legs (w = beta / sum|beta|),
hedge ratio / mean / std estimated on the TRAINING period only, one-bar execution lag, costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import generate_positions


def static_pair_backtest(prices: pd.DataFrame, a: str, b: str, train_end: str, test_start: str,
                         test_end: str | None = None, entry: float = 2.0, exit_: float = 0.5,
                         stop: float | None = None, cost_bps: float = 5.0) -> dict:
    logp = np.log(prices[[a, b]])
    train = logp.loc[:train_end]
    X = np.column_stack([np.ones(len(train)), train[b].to_numpy()])
    (_, gamma), *_ = np.linalg.lstsq(X, train[a].to_numpy(), rcond=None)
    beta = np.array([1.0, -gamma])
    w = beta / np.abs(beta).sum()
    s_train = train.to_numpy() @ beta
    mu, sd = s_train.mean(), s_train.std(ddof=1)

    test = logp.loc[test_start:test_end]
    z = (test.to_numpy() @ beta - mu) / sd                      # frozen training parameters
    sig = generate_positions(z, entry, exit_, np.inf if stop is None else stop)
    pos_prev = np.concatenate([[0.0], sig[:-1]])
    rets = prices[[a, b]].pct_change().loc[test.index].fillna(0.0).to_numpy() @ w
    pnl = pos_prev * rets - np.abs(sig - pos_prev) * cost_bps / 1e4
    return {"returns": pd.Series(pnl, index=test.index), "z": pd.Series(z, index=test.index),
            "beta": beta, "mu": mu, "sd": sd, "positions": pd.Series(sig, index=test.index)}
