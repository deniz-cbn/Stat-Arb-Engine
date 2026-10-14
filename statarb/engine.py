"""
Walk-forward statistical-arbitrage engine.

For each rolling window:
  1. FORMATION (past data only)  screen all pairs with the Johansen test -> hedge vector beta,
                                 fit an OU process to the spread -> kappa, mu, sigma_eq, half-life,
                                 choose the cost-optimal z-entry threshold.
  2. TRADING   (next `trading_days`, parameters frozen)  trade z_t = (S_t - mu)/sigma_eq.
No quantity estimated on data after t is used at time t; signals are executed one bar later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
import pandas as pd

from .johansen import johansen_test
from .ou import fit_ou, optimal_entry_threshold
from .metrics import performance_summary


@dataclass
class EngineConfig:
    formation_days: int = 252
    trading_days: int = 63
    top_n_pairs: int = 5
    min_half_life: float = 2.0
    max_half_life: float = 40.0
    cost_bps: float = 5.0            # one-way cost per unit of traded gross notional (commission + slippage)
    z_entry: float | None = None     # None -> cost-optimal threshold from OU theory
    z_exit: float = 0.0
    z_stop: float = 4.0
    max_lags: int = 5
    lag_criterion: str = "bic"
    backend: str = "auto"            # "auto" | "statsmodels" | "native"
    max_adf_t: float | None = -3.0   # confirm with a Dickey-Fuller t-stat on the spread (EG 5% cv ~ -3.37)
    require_long_short: bool = True


@dataclass
class PairModel:
    names: tuple
    beta: np.ndarray
    weights: np.ndarray              # beta / sum|beta|  (gross exposure 1)
    mu: float
    sigma_eq: float
    kappa: float
    half_life: float
    trace_stat: float
    crit_value: float
    z_entry: float
    adf_t: float = float("nan")


@dataclass
class BacktestResult:
    returns: pd.Series
    trades: pd.DataFrame
    models: list                     # list of (window_start, window_end, [PairModel])
    market: pd.Series
    config: EngineConfig

    @property
    def equity(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    def selection_table(self) -> pd.DataFrame:
        """One row per (window, selected pair): the audit trail of what the engine believed and when."""
        rows = []
        for t0, t1, ms in self.models:
            for m in ms:
                rows.append(dict(window_start=t0, window_end=t1, pair="/".join(m.names),
                                 hedge_ratio=-m.beta[1] / m.beta[0], half_life=m.half_life,
                                 z_entry=m.z_entry, trace_over_cv=m.trace_stat / m.crit_value,
                                 adf_t=m.adf_t))
        return pd.DataFrame(rows)

    def summary(self) -> dict:
        s = performance_summary(self.returns, self.market)
        s["n_trades"] = len(self.trades)
        s["avg_holding_days"] = self.trades["days"].mean() if len(self.trades) else np.nan
        s["win_rate_trades"] = (self.trades["pnl"] > 0).mean() if len(self.trades) else np.nan
        return s


def generate_positions(z, entry: float, exit_: float, stop: float) -> np.ndarray:
    """Z-score state machine.  +1 = long spread, -1 = short spread, 0 = flat.  After a stop-out the
    pair stays locked flat until z has come back inside the entry band."""
    z = np.asarray(z, float)
    pos = np.zeros(len(z))
    cur, locked = 0, False
    for i, zi in enumerate(z):
        if np.isnan(zi):
            pos[i] = cur
            continue
        if cur == 0:
            if locked:
                if abs(zi) < entry:
                    locked = False
            elif -stop < zi <= -entry:
                cur = 1
            elif entry <= zi < stop:
                cur = -1
        else:
            if abs(zi) >= stop:
                cur, locked = 0, True
            elif cur == 1 and zi >= -exit_:
                cur = 0
            elif cur == -1 and zi <= exit_:
                cur = 0
        pos[i] = cur
    return pos


class StatArbEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.cfg = config or EngineConfig()

    # ---------------------------------------------------------------- formation
    def fit_pair(self, logp: pd.DataFrame, a: str, b: str) -> PairModel | None:
        c = self.cfg
        Y = logp[[a, b]].to_numpy()
        jo = johansen_test(Y, backend=c.backend, max_lags=c.max_lags, criterion=c.lag_criterion)
        if jo.rank < 1:
            return None
        beta = jo.eigenvectors[:, 0]
        if c.require_long_short and beta[0] * beta[1] > 0:
            return None
        spread = Y @ beta
        ou = fit_ou(spread)
        if not ou.mean_reverting or not (c.min_half_life <= ou.half_life <= c.max_half_life):
            return None
        adf_t = (ou.b - 1) / ou.b_se                           # DF t-stat for H0: unit root in the spread
        if c.max_adf_t is not None and adf_t > c.max_adf_t:
            return None
        gross = np.abs(beta).sum()
        if c.z_entry is not None:
            z_in = c.z_entry
        else:
            round_trip_cost = 2 * c.cost_bps / 1e4 * gross       # in spread (log-price) units
            z_in = optimal_entry_threshold(round_trip_cost / ou.sigma_eq)
        return PairModel((a, b), beta, beta / gross, ou.mu, ou.sigma_eq, ou.kappa, ou.half_life,
                         jo.trace_stats[0], jo.crit_values[0], z_in, adf_t)

    def screen(self, logp: pd.DataFrame) -> list:
        models = [m for a, b in combinations(logp.columns, 2)
                  if (m := self.fit_pair(logp, a, b)) is not None]
        models.sort(key=lambda m: m.trace_stat / m.crit_value, reverse=True)
        return models[: self.cfg.top_n_pairs]

    # ---------------------------------------------------------------- trading
    def _trade_window(self, model: PairModel, logp: pd.DataFrame, rets: pd.DataFrame, lo: int, hi: int):
        c = self.cfg
        a, b = model.names
        spread = logp[[a, b]].iloc[lo:hi].to_numpy() @ model.beta
        z = (spread - model.mu) / model.sigma_eq
        sig = generate_positions(z, model.z_entry, c.z_exit, c.z_stop)
        sig[-1] = 0.0                                          # flatten at window end (params change)
        pos_prev = np.concatenate([[0.0], sig[:-1]])           # signal at close t -> held over t+1
        r = rets[[a, b]].iloc[lo:hi].to_numpy() @ model.weights
        cost = np.abs(sig - pos_prev) * c.cost_bps / 1e4       # trade at close t, gross exposure = 1
        pnl = pos_prev * r - cost                              # position from close t-1 earns return t
        return pd.Series(pnl, index=logp.index[lo:hi]), sig

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        c = self.cfg
        logp, rets = np.log(prices), prices.pct_change().fillna(0.0)
        T = len(prices)
        port = pd.Series(0.0, index=prices.index)
        trades, history = [], []
        start = 0
        while start + c.formation_days + 2 <= T:
            f_end = start + c.formation_days
            t_end = min(f_end + c.trading_days, T)
            models = self.screen(logp.iloc[start:f_end])
            history.append((prices.index[f_end], prices.index[t_end - 1], models))
            for m in models:
                pnl, sig = self._trade_window(m, logp, rets, f_end, t_end)
                port.iloc[f_end:t_end] += pnl.to_numpy() / c.top_n_pairs
                trades += self._log_trades(m, pnl, sig)
            start += c.trading_days
        first = history[0][0] if history else prices.index[0]
        port = port.loc[first:]
        market = rets.mean(axis=1).loc[first:]
        cols = ["pair", "entry", "exit", "side", "days", "pnl"]
        return BacktestResult(port, pd.DataFrame(trades, columns=cols), history, market, c)

    @staticmethod
    def _log_trades(m: PairModel, pnl: pd.Series, sig: np.ndarray) -> list:
        out, i, n = [], 0, len(sig)
        while i < n:
            if sig[i] != 0:
                j = i
                while j + 1 < n and sig[j + 1] == sig[i]:
                    j += 1
                seg = pnl.iloc[i + 1: j + 2].sum() + pnl.iloc[i]   # costs at entry + P&L while held + exit cost
                out.append(("/".join(m.names), pnl.index[i], pnl.index[min(j + 1, n - 1)],
                            "long" if sig[i] > 0 else "short", j - i + 1, float(seg)))
                i = j + 1
            else:
                i += 1
        return out
