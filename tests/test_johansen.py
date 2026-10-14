import numpy as np
import pytest
from statarb import johansen_test
from statarb.synthetic import make_universe
from statarb.johansen import _sm_coint_johansen


def test_detects_cointegrated_pair_and_hedge_ratio():
    prices, truth = make_universe(n_days=800, n_pairs=1, n_noise=0, seed=3)
    jo = johansen_test(np.log(prices.to_numpy()), backend="native")
    assert jo.rank == 1
    gamma = truth[("A0", "B0")]["gamma"]
    assert abs(jo.eigenvectors[1, 0] + gamma) < 0.08       # beta = (1, -gamma)


def test_empirical_size_near_nominal_with_drift():
    """The tabulated critical values (statsmodels det_order=0) assume drifting series: size ~5% then.
    (With ~zero drift the test over-rejects, which is why the engine adds a Dickey-Fuller confirmation.)"""
    rng = np.random.default_rng(0)
    rej = sum(johansen_test(np.cumsum(rng.normal(0.3, 1, size=(500, 2)), axis=0),
                            backend="native", k_ar_diff=1).rank > 0 for _ in range(300))
    assert rej / 300 < 0.09


@pytest.mark.skipif(_sm_coint_johansen is None, reason="statsmodels not installed")
def test_native_matches_statsmodels():
    prices, _ = make_universe(n_days=600, n_pairs=1, n_noise=1, seed=5)
    Y = np.log(prices.to_numpy())[:, :2]
    a = johansen_test(Y, k_ar_diff=1, backend="native")
    b = johansen_test(Y, k_ar_diff=1, backend="statsmodels")
    np.testing.assert_allclose(a.eigenvalues, b.eigenvalues, rtol=1e-6)
    np.testing.assert_allclose(a.trace_stats, b.trace_stats, rtol=1e-6)
