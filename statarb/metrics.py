from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_summary(returns: pd.Series, market: pd.Series | None = None) -> dict:
    r = returns.dropna()
    mu, sd = r.mean(), r.std(ddof=1)
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    years = len(r) / TRADING_DAYS
    out = {
        "total_return": equity.iloc[-1] - 1,
        "cagr": equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan,
        "ann_vol": sd * np.sqrt(TRADING_DAYS),
        "sharpe": mu / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan,
        "max_drawdown": dd.min(),
        "hit_rate_days": (r[r != 0] > 0).mean() if (r != 0).any() else np.nan,
    }
    if market is not None:
        m = market.reindex(r.index).fillna(0.0)
        cov = np.cov(r, m, ddof=1)
        out["market_beta"] = cov[0, 1] / cov[1, 1]
        out["market_corr"] = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return out
