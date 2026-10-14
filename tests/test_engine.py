import numpy as np
from statarb import StatArbEngine, EngineConfig, generate_positions, make_universe


def test_state_machine_entries_exits_and_stop():
    z = [0, -2.5, -1, 0.1, 0, 2.2, 1, -0.2, 2.5, 4.5, 3, 1.0, 2.5]
    p = generate_positions(z, entry=2.0, exit_=0.0, stop=4.0)
    assert list(p) == [0, 1, 1, 0, 0, -1, -1, 0, -1, 0, 0, 0, -1]


def test_no_lookahead_future_data_does_not_change_past_pnl():
    prices, _ = make_universe(n_days=900, seed=11)
    cfg = EngineConfig(backend="native")
    full = StatArbEngine(cfg).run(prices).returns
    cut = StatArbEngine(cfg).run(prices.iloc[:700]).returns
    common = cut.index[:-1]                                  # last bar of `cut` is force-flattened
    np.testing.assert_allclose(full.loc[common[:-70]], cut.loc[common[:-70]], atol=1e-12)


def test_engine_finds_planted_pairs_and_is_market_neutral():
    prices, truth = make_universe(n_days=1500, n_pairs=3, n_noise=5, seed=2)
    res = StatArbEngine(EngineConfig(backend="native")).run(prices)
    picked = {m.names for _, _, ms in res.models for m in ms}
    planted = set(truth)
    assert len(picked & planted) >= 2
    s = res.summary()
    assert abs(s["market_beta"]) < 0.15
    assert s["n_trades"] > 10
