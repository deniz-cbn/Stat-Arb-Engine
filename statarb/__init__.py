"""Statistical arbitrage backtesting engine: Johansen cointegration + Ornstein-Uhlenbeck mean reversion."""
from .johansen import johansen_test, select_lag_order, JohansenResult
from .ou import (fit_ou, simulate_ou, OUFit, expected_cycle_time,
                 optimal_entry_threshold, expected_profit_rate)
from .engine import StatArbEngine, EngineConfig, BacktestResult, generate_positions
from .metrics import performance_summary
from .synthetic import make_universe
from .baselines import static_pair_backtest

__all__ = [
    "johansen_test", "select_lag_order", "JohansenResult",
    "fit_ou", "simulate_ou", "OUFit", "expected_cycle_time",
    "optimal_entry_threshold", "expected_profit_rate",
    "StatArbEngine", "EngineConfig", "BacktestResult", "generate_positions",
    "performance_summary", "make_universe", "static_pair_backtest",
]
__version__ = "0.1.0"
