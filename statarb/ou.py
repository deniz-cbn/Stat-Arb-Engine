"""
Ornstein-Uhlenbeck mean-reversion model for a cointegration spread.

    dX_t = kappa (mu - X_t) dt + sigma dW_t

Exact solution (integrating factor e^{kappa t}, Ito's lemma):
    X_{t+dt} = mu + (X_t - mu) e^{-kappa dt} + sigma sqrt((1 - e^{-2 kappa dt}) / (2 kappa)) * N(0,1)
which is an AR(1):  X_{t+1} = a + b X_t + eps,  b = e^{-kappa dt},  a = mu (1 - b).
Time unit = 1 trading day (dt = 1).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import integrate, optimize, special


@dataclass
class OUFit:
    kappa: float          # mean-reversion speed (1/day);  nan if not mean-reverting
    mu: float             # long-run mean
    sigma: float          # instantaneous volatility (spread units / sqrt(day))
    sigma_eq: float       # stationary std  sigma / sqrt(2 kappa)
    half_life: float      # ln 2 / kappa  (days)
    b: float              # AR(1) coefficient
    b_se: float           # OLS standard error of b
    n: int

    @property
    def mean_reverting(self) -> bool:
        return bool(np.isfinite(self.kappa) and self.kappa > 0)


def fit_ou(x, dt: float = 1.0) -> OUFit:
    """Exact-discretisation (conditional MLE == OLS) estimate of OU parameters from a spread series."""
    x = np.asarray(x, float)
    x0, x1 = x[:-1], x[1:]
    X = np.column_stack([np.ones_like(x0), x0])
    (a, b), *_ = np.linalg.lstsq(X, x1, rcond=None)
    resid = x1 - X @ np.array([a, b])
    n = len(x1)
    s2 = resid @ resid / (n - 2)
    b_se = np.sqrt(s2 / np.sum((x0 - x0.mean()) ** 2))
    if not (0 < b < 1):
        return OUFit(np.nan, x.mean(), np.nan, x.std(ddof=1), np.inf, b, b_se, n)
    kappa = -np.log(b) / dt
    mu = a / (1 - b)
    sigma = np.sqrt(s2 * 2 * kappa / (1 - b ** 2))
    sigma_eq = np.sqrt(s2 / (1 - b ** 2))          # == sigma / sqrt(2 kappa)
    return OUFit(kappa, mu, sigma, sigma_eq, np.log(2) / kappa, b, b_se, n)


def simulate_ou(kappa, mu, sigma, n, x0=None, dt=1.0, rng=None) -> np.ndarray:
    """Exact simulation (no Euler discretisation error)."""
    rng = np.random.default_rng(rng)
    b = np.exp(-kappa * dt)
    sd = sigma * np.sqrt((1 - b ** 2) / (2 * kappa))
    x = np.empty(n)
    x[0] = mu if x0 is None else x0
    eps = rng.standard_normal(n - 1) * sd
    for t in range(1, n):
        x[t] = mu + (x[t - 1] - mu) * b + eps[t - 1]
    return x


# --------------------------------------------------------------------------- optimal thresholds
def _u(z: float) -> float:
    """Dimensionless barrier u = a sqrt(kappa)/sigma for a barrier at z stationary std's (a = z sigma/sqrt(2 kappa))."""
    return z / np.sqrt(2.0)


def expected_cycle_time(z: float, kappa: float) -> float:
    """
    E[time] of one trade cycle: from X = mu wait until |X - mu| = a (a = z * sigma_eq), then hold
    until X returns to mu.  Closed form from the hitting-time ODE  L T = -1 :

        E[cycle] = (pi / (2 kappa)) * erfi(z / sqrt 2)        (days)
    """
    return np.pi / (2 * kappa) * special.erfi(_u(z))


def expected_time_numeric(z: float, kappa: float) -> float:
    """Same quantity via the two quadratures (wait + hold) -- used to verify the closed form."""
    u = _u(z)
    wait = integrate.quad(lambda v: np.exp(v * v) * special.erf(v), 0, u)[0]
    hold = integrate.quad(lambda v: special.erfcx(v), 0, u)[0]
    return np.sqrt(np.pi) / kappa * (wait + hold)


def expected_profit_rate(z: float, kappa: float, sigma_eq: float, cost: float) -> float:
    """Long-run expected profit per day (spread units) of entry at +/- z*sigma_eq, exit at mu."""
    a = z * sigma_eq
    return (a - cost) / expected_cycle_time(z, kappa)


def optimal_entry_threshold(cost_in_sigma_eq: float, z_min: float = 0.3, z_max: float = 4.0) -> float:
    """
    Maximise  (z - c) / erfi(z / sqrt 2)  over z, where c = round-trip cost / sigma_eq.
    The optimum depends only on the cost-to-volatility ratio, not on kappa (kappa only scales the
    profit rate).  With c = 0 the sup is approached as z -> 0 (infinitely many infinitesimal trades),
    hence the z_min floor and the importance of a realistic cost.
    """
    c = max(cost_in_sigma_eq, 0.0)
    f = lambda z: -(z - c) / special.erfi(_u(z))
    grid = np.linspace(max(z_min, c + 1e-3), z_max, 400)
    z0 = grid[np.argmin([f(z) for z in grid])]
    lo, hi = max(z_min, c + 1e-6, z0 - 0.05), min(z_max, z0 + 0.05)
    return float(optimize.minimize_scalar(f, bounds=(lo, hi), method="bounded").x)
