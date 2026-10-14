import numpy as np
from statarb.ou import (fit_ou, simulate_ou, expected_cycle_time, expected_time_numeric,
                        optimal_entry_threshold, expected_profit_rate)


def test_ou_parameter_recovery():
    kappa, mu, sigma = 0.08, 0.5, 0.02
    ests = [fit_ou(simulate_ou(kappa, mu, sigma, 3000, rng=s)) for s in range(30)]
    assert abs(np.mean([e.kappa for e in ests]) - kappa) < 0.015
    assert abs(np.mean([e.mu for e in ests]) - mu) < 0.01
    assert abs(np.mean([e.sigma for e in ests]) - sigma) / sigma < 0.05
    assert abs(np.mean([e.half_life for e in ests]) - np.log(2) / kappa) < 1.5


def test_random_walk_is_not_mean_reverting_t_stat():
    x = np.cumsum(np.random.default_rng(1).normal(size=2000))
    e = fit_ou(x)
    assert (1 - e.b) / e.b_se < 3.5      # far from "significantly < 1" at DF critical values


def test_cycle_time_closed_form_matches_quadrature():
    for z in (0.5, 1.0, 2.0, 3.0):
        assert np.isclose(expected_cycle_time(z, 0.07), expected_time_numeric(z, 0.07), rtol=1e-6)


def test_cycle_time_monte_carlo():
    """Simulate a fine-grained OU, run enter-at-+/-a / exit-at-mu, compare mean cycle time to the formula."""
    kappa, sigma, dt, z = 1.0, 1.0, 0.002, 1.0
    s_eq = sigma / np.sqrt(2 * kappa)
    x = simulate_ou(kappa, 0.0, sigma, 2_000_000, dt=dt, rng=7)
    state, t_start, times = 0, 0, []
    a = z * s_eq
    for i, xi in enumerate(x):
        if state == 0 and abs(xi) >= a:
            state = 1 if xi > 0 else -1
        elif state == 1 and xi <= 0 or state == -1 and xi >= 0:
            times.append(i - t_start); t_start, state = i, 0
    # first cycle starts at 0 only approximately; discard it
    mc = np.mean(times[1:]) * dt
    assert abs(mc - expected_cycle_time(z, kappa)) / expected_cycle_time(z, kappa) < 0.08


def test_optimal_threshold_increases_with_cost_and_maximises_rate():
    zs = [optimal_entry_threshold(c) for c in (0.05, 0.2, 0.5, 1.0)]
    assert all(a < b for a, b in zip(zs, zs[1:]))
    c, z_star = 0.4, optimal_entry_threshold(0.4)
    r = lambda z: expected_profit_rate(z, 0.05, 1.0, c)
    assert r(z_star) >= r(z_star - 0.15) and r(z_star) >= r(z_star + 0.15)
