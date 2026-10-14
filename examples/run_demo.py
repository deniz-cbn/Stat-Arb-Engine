"""End-to-end demo on a synthetic universe with planted cointegrated pairs.

    python examples/run_demo.py            # prints metrics, writes figures to docs/img/

To use real data, replace `make_universe` with a DataFrame of adjusted close prices
(index = dates, columns = tickers), e.g. from yfinance:  yf.download(tickers)["Close"].dropna()
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statarb import (StatArbEngine, EngineConfig, make_universe, johansen_test, fit_ou,
                     expected_profit_rate, optimal_entry_threshold, simulate_ou)

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)

prices, truth = make_universe(n_days=1500, n_pairs=3, n_noise=5, seed=2)
print("Planted pairs:", {k: round(v["half_life"], 1) for k, v in truth.items()}, "(half-lives, days)")

cfg = EngineConfig(cost_bps=5.0)
res = StatArbEngine(cfg).run(prices)
print("\nBacktest summary")
for k, v in res.summary().items():
    print(f"  {k:>18s}: {v:8.3f}")

# --- Figure 1: one pair, spread, OU bands, positions --------------------------------------
logp = np.log(prices)
Y = logp[["A0", "B0"]].to_numpy()
jo = johansen_test(Y[:500])
beta = jo.eigenvectors[:, 0]
spread = Y @ beta
ou = fit_ou(spread[:500])
z = (spread - ou.mu) / ou.sigma_eq
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(spread, lw=.8); ax[0].axhline(ou.mu, c="k", ls="--")
for k in (-2, 2): ax[0].axhline(ou.mu + k * ou.sigma_eq, c="r", ls=":")
ax[0].set_title(f"Johansen spread  S = log A0 {beta[1]:+.3f} log B0   (kappa={ou.kappa:.3f}/day, half-life={ou.half_life:.1f}d)")
ax[1].plot(z, lw=.8); ax[1].axhline(0, c="k"); ax[1].set_title("z-score  (S - mu) / sigma_eq   [parameters fitted on first 500 days]")
ax[1].axvline(500, c="g"); ax[1].text(505, ax[1].get_ylim()[1] * .8, "out-of-sample ->")
plt.tight_layout(); plt.savefig(OUT / "spread_ou.png", dpi=130); plt.close()

# --- Figure 2: equity curve ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.8))
res.equity.plot(ax=ax, label="Strategy (5 bps one-way)")
(1 + res.market).cumprod().plot(ax=ax, label="Equal-weight universe", alpha=.7)
ax.set_title("Walk-forward out-of-sample equity (synthetic data)"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "equity.png", dpi=130); plt.close()

# --- Figure 3: expected profit rate vs entry threshold for several cost levels -------------
zz = np.linspace(0.3, 3.5, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for c in (0.1, 0.3, 0.6):
    r = np.array([expected_profit_rate(x, 0.05, 1.0, c) for x in zz])
    ax.plot(zz, r, label=f"cost = {c} sigma_eq"); zs = optimal_entry_threshold(c)
    ax.plot(zs, expected_profit_rate(zs, 0.05, 1.0, c), "ko")
ax.set_xlabel("entry threshold z"); ax.set_ylabel("expected profit / day (units of sigma_eq)")
ax.set_title("Optimal z* (dots) from OU first-passage theory"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "optimal_threshold.png", dpi=130); plt.close()
print("\nFigures written to", OUT)
