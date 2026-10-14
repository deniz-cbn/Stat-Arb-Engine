"""
Johansen cointegration test (trace statistic) with two interchangeable backends:

* ``statsmodels`` : ``statsmodels.tsa.vector_ar.vecm.coint_johansen`` (reference implementation)
* ``native``      : a from-scratch NumPy implementation of the same reduced-rank regression, so every
                    step of the maths is visible and testable.

Deterministic-term case: an *unrestricted constant* in the VECM (statsmodels ``det_order=0``).
This is the standard choice for log prices that drift but whose equilibrium spread is stationary.

VECM:   dY_t = c + Pi Y_{t-1} + sum_{j=1}^{k} Gamma_j dY_{t-j} + e_t,     Pi = alpha beta'
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:  # optional reference backend
    from statsmodels.tsa.vector_ar.vecm import coint_johansen as _sm_coint_johansen
except Exception:  # pragma: no cover
    _sm_coint_johansen = None

# 95% asymptotic critical values of the trace statistic, indexed by n - r (number of
# non-cointegrated directions under H0).  MacKinnon-Haug-Michelis (1999), constant in VECM.
_CV95_TRACE = {1: 3.8415, 2: 15.4947, 3: 29.7961, 4: 47.8545, 5: 69.8189}


@dataclass
class JohansenResult:
    eigenvalues: np.ndarray      # lambda_1 >= ... >= lambda_n  (squared canonical correlations)
    eigenvectors: np.ndarray     # columns = cointegrating vectors beta_i (first coef normalised to 1)
    trace_stats: np.ndarray      # trace_stats[r] tests H0: rank <= r
    crit_values: np.ndarray      # 95% critical value for each H0
    rank: int                    # sequentially determined cointegration rank
    k_ar_diff: int
    n_obs: int
    backend: str

    def spread(self, Y: np.ndarray, which: int = 0) -> np.ndarray:
        return np.asarray(Y) @ self.eigenvectors[:, which]


def select_lag_order(Y: np.ndarray, max_lags: int = 5, criterion: str = "bic") -> int:
    """Choose VAR(p) order in *levels* by AIC/BIC on a common sample; returns k_ar_diff = p - 1."""
    Y = np.asarray(Y, float)
    T, n = Y.shape
    best_p, best_ic = 1, np.inf
    for p in range(1, max_lags + 1):
        dep = Y[max_lags:]
        X = np.column_stack([np.ones(T - max_lags)] +
                            [Y[max_lags - j: T - j] for j in range(1, p + 1)])
        coef, *_ = np.linalg.lstsq(X, dep, rcond=None)
        resid = dep - X @ coef
        m = len(dep)
        sigma = resid.T @ resid / m
        logdet = np.linalg.slogdet(sigma)[1]
        k_params = p * n * n + n
        pen = 2 * k_params / m if criterion == "aic" else k_params * np.log(m) / m
        ic = logdet + pen
        if ic < best_ic:
            best_p, best_ic = p, ic
    return best_p - 1


def _residualise(Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    return Y - Z @ coef


def _johansen_native(Y: np.ndarray, k: int):
    """Johansen (1991) reduced-rank regression, step by step.  Returns (lambda, beta, trace, n_obs)."""
    T, n = Y.shape
    dY = np.diff(Y, axis=0)                      # (T-1, n)
    N = T - 1 - k                                # effective sample
    dep = dY[k:]                                 # dY_t
    lev = Y[k: T - 1]                            # Y_{t-1}
    Z = [np.ones((N, 1))] + [dY[k - j: T - 1 - j] for j in range(1, k + 1)]
    Z = np.hstack(Z)                             # constant + lagged differences (nuisance regressors)

    R0 = _residualise(dep, Z)                    # step 1: partial out nuisance from dY_t
    R1 = _residualise(lev, Z)                    #         ... and from Y_{t-1}
    S00, S11, S01 = R0.T @ R0 / N, R1.T @ R1 / N, R0.T @ R1 / N
    S10 = S01.T

    # step 2: solve |lambda S11 - S10 S00^{-1} S01| = 0 via the symmetric (Cholesky) form
    L = np.linalg.cholesky(S11)
    Linv = np.linalg.inv(L)
    M = Linv @ S10 @ np.linalg.solve(S00, S01) @ Linv.T
    M = (M + M.T) / 2
    lam, W = np.linalg.eigh(M)
    order = np.argsort(lam)[::-1]
    lam, W = lam[order], W[:, order]
    beta = Linv.T @ W                            # satisfies beta' S11 beta = I

    # step 3: trace statistics  -N * sum_{i>r} ln(1 - lambda_i)
    log_terms = np.log(1 - np.clip(lam, 0, 1 - 1e-12))
    trace = np.array([-N * log_terms[r:].sum() for r in range(n)])
    return lam, beta, trace, N


def _normalise(beta: np.ndarray) -> np.ndarray:
    out = beta.copy()
    for i in range(out.shape[1]):
        if abs(out[0, i]) > 1e-12:
            out[:, i] = out[:, i] / out[0, i]
    return out


def johansen_test(Y, k_ar_diff: int | None = None, backend: str = "auto",
                  max_lags: int = 5, criterion: str = "bic") -> JohansenResult:
    """Johansen trace test on the columns of Y (use *log* prices)."""
    Y = np.asarray(Y, float)
    T, n = Y.shape
    if n > 5:
        raise ValueError("critical-value table implemented for up to 5 series")
    if k_ar_diff is None:
        k_ar_diff = select_lag_order(Y, max_lags=max_lags, criterion=criterion)

    use_sm = backend == "statsmodels" or (backend == "auto" and _sm_coint_johansen is not None)
    if use_sm:
        if _sm_coint_johansen is None:
            raise ImportError("statsmodels not installed")
        res = _sm_coint_johansen(Y, det_order=0, k_ar_diff=k_ar_diff)
        lam, trace, cv = np.asarray(res.eig), np.asarray(res.lr1), np.asarray(res.cvt)[:, 1]
        beta, N, name = _normalise(np.asarray(res.evec)), T - 1 - k_ar_diff, "statsmodels"
    else:
        lam, beta, trace, N = _johansen_native(Y, k_ar_diff)
        cv = np.array([_CV95_TRACE[n - r] for r in range(n)])
        beta, name = _normalise(beta), "native"

    rank = 0
    for r in range(n):                           # sequential testing: r=0, then r<=1, ...
        if trace[r] > cv[r]:
            rank += 1
        else:
            break
    return JohansenResult(lam, beta, trace, cv, rank, k_ar_diff, N, name)
