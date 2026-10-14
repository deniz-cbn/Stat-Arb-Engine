"""Run the walk-forward engine on real data and compare it with the static single-pair strategy.

    python examples/run_real.py                       # downloads 16 US stocks via yfinance (needs internet)
    python examples/run_real.py --synthetic --pair A0 B0 --holdout-start 2022-06-01   # offline smoke test

What it does
  1. Loads (and caches) adjusted closes.
  2. ENGINE: rolling Johansen screen -> OU -> cost-optimal z*; every window is out of sample.
  3. BASELINE: the static V/MA strategy (fit on data before --holdout-start, frozen thereafter).
  4. Reports metrics for the development period and the HOLDOUT, writes CSVs and figures to results/.
The configuration is fixed a priori. Do NOT tune it on the holdout: that turns it into a second training set.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statarb import StatArbEngine, EngineConfig, performance_summary, static_pair_backtest, make_universe

TICKERS = ["KO", "PEP", "JPM", "BAC", "XOM", "CVX", "MSFT", "AAPL",
           "GOOGL", "META", "V", "MA", "HD", "LOW", "MCD", "SBUX"]


def load_prices(args) -> pd.DataFrame:
    if args.synthetic:
        return make_universe(n_days=1500, n_pairs=3, n_noise=5, seed=2)[0]
    cache = Path(args.cache)
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    import yfinance as yf
    px = yf.download(TICKERS, start=args.start, end=args.end, auto_adjust=True)["Close"].dropna()
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px


def fmt(s: dict) -> dict:
    keep = ["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "market_beta"]
    return {k: round(float(s[k]), 3) for k in keep if k in s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--holdout-start", default="2025-01-01")
    ap.add_argument("--pair", nargs=2, default=["V", "MA"])
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--cache", default="data/prices.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    prices = load_prices(args)
    print(f"Universe: {prices.shape[1]} assets, {len(prices)} days, {prices.index[0].date()} -> {prices.index[-1].date()}")
    ho = pd.Timestamp(args.holdout_start)
    dev_end = prices.index[prices.index < ho][-1]

    # ------------------------------------------------------------------ ENGINE (walk-forward)
    cfg = EngineConfig(formation_days=252, trading_days=63, top_n_pairs=5, cost_bps=args.cost_bps)
    res = StatArbEngine(cfg).run(prices)
    mkt = res.market
    dev, hold = res.returns.loc[:dev_end], res.returns.loc[ho:]
    sel = res.selection_table()
    sel.to_csv(out / "selected_pairs_by_window.csv", index=False)

    # ------------------------------------------------------------------ BASELINE (static pair)
    a, b = args.pair
    base = static_pair_backtest(prices, a, b, train_end=str(dev_end.date()), test_start=str(ho.date()),
                                entry=2.0, exit_=0.5, stop=None, cost_bps=args.cost_bps)
    base_stop = static_pair_backtest(prices, a, b, train_end=str(dev_end.date()), test_start=str(ho.date()),
                                     entry=2.0, exit_=0.5, stop=4.0, cost_bps=args.cost_bps)

    table = pd.DataFrame({
        f"ENGINE dev (to {dev_end.date()})": fmt(performance_summary(dev, mkt)),
        f"ENGINE holdout (from {ho.date()})": fmt(performance_summary(hold, mkt)),
        f"STATIC {a}/{b} holdout": fmt(performance_summary(base["returns"], mkt)),
        f"STATIC {a}/{b} + stop |z|=4": fmt(performance_summary(base_stop["returns"], mkt)),
    })
    print("\n", table.to_string())
    table.to_csv(out / "comparison.csv")

    # ------------------------------------------------------------------ diagnostics
    print("\nWindows traded:", len(res.models), "| mean pairs selected per window:",
          round(len(sel) / max(len(res.models), 1), 2))
    if len(sel):
        freq = sel["pair"].value_counts().head(8)
        print("Most frequently selected pairs:\n", freq.to_string())
        pair_name = f"{a}/{b}"
        in_pair = sel[sel["pair"] == pair_name]
        print(f"\n{pair_name} selected in {len(in_pair)} of {len(res.models)} windows")
        if len(in_pair):
            print(in_pair[["window_start", "hedge_ratio", "half_life", "adf_t"]].round({"hedge_ratio": 3, "half_life": 2, "adf_t": 2}).to_string(index=False))
    n_pairs = prices.shape[1] * (prices.shape[1] - 1) // 2
    print(f"\nMultiple testing: {n_pairs} pairs screened per window -> expect ~{0.05 * n_pairs:.0f} false positives "
          f"at 5% before the Dickey-Fuller and half-life filters.")
    print(f"Static baseline: frozen z ranged {base['z'].min():.2f} .. {base['z'].max():.2f} in the holdout; "
          f"days in position: {int((base['positions'] != 0).sum())}/{len(base['positions'])}")
    print("Trades:", len(res.trades), "| avg holding days:", round(res.trades['days'].mean(), 1) if len(res.trades) else "n/a")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7))
    (1 + res.returns).cumprod().plot(ax=ax[0], label="Engine (walk-forward)")
    ax[0].axvline(ho, c="k", ls="--"); ax[0].set_title("Engine equity (dashed = holdout start)"); ax[0].legend()
    (1 + base["returns"]).cumprod().plot(ax=ax[1], label=f"Static {a}/{b}")
    (1 + base_stop["returns"]).cumprod().plot(ax=ax[1], label=f"Static {a}/{b} + stop")
    (1 + res.returns.loc[ho:]).cumprod().plot(ax=ax[1], label="Engine")
    ax[1].set_title("Holdout comparison"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out / "comparison.png", dpi=130)
    print("\nWrote", out / "comparison.csv", out / "selected_pairs_by_window.csv", out / "comparison.png")


if __name__ == "__main__":
    main()
