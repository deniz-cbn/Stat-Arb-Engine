# The Mathematics Behind the Engine

This document derives every formula the code uses, in the order the pipeline uses them.
Code references point to `statarb/johansen.py`, `statarb/ou.py` and `statarb/engine.py`.

**Contents**
1. [Why cointegration, not correlation](#1-why-cointegration-not-correlation)
2. [Johansen test: from VAR to an eigenvalue problem](#2-johansen-test-from-var-to-an-eigenvalue-problem)
3. [From cointegrating vector to a tradeable, beta-neutral spread](#3-from-cointegrating-vector-to-a-tradeable-beta-neutral-spread)
4. [The Ornstein–Uhlenbeck process](#4-the-ornsteinuhlenbeck-process)
5. [Estimating OU parameters](#5-estimating-ou-parameters)
6. [Optimal, volatility-adjusted z-score thresholds](#6-optimal-volatility-adjusted-z-score-thresholds)
7. [P&L, costs, and walk-forward discipline](#7-pl-costs-and-walk-forward-discipline)
8. [Statistical pitfalls](#8-statistical-pitfalls)
9. [References](#9-references)

---

## 1. Why cointegration, not correlation

Log prices $y_t = \ln P_t$ are well modelled as **integrated of order one**, $I(1)$: they contain a stochastic trend, and only the differences $\Delta y_t$ (returns) are stationary.

*Correlation* describes co-movement of **returns**. Two assets can have highly correlated returns and still drift apart without bound in price space, so a long–short position in them is not mean-reverting. Regressing one $I(1)$ series on another also produces *spurious regression*: $R^2$ close to 1 and large $t$-statistics even for independent random walks.

**Definition.** An $n$-vector $Y_t$ of $I(1)$ series is *cointegrated* if there is a non-zero $\beta\in\mathbb R^n$ such that
$$S_t = \beta^\top Y_t \sim I(0)$$
is stationary. The stochastic trends cancel in the linear combination $\beta^\top Y_t$. $S_t$ is the **spread**. Mean reversion of $S_t$ is a statement about *levels* and is what a pairs trader needs.

The number of linearly independent such vectors is the **cointegration rank** $r$, with $0\le r<n$.

**Engle–Granger vs. Johansen.** Engle–Granger regresses $y^{(1)}$ on $y^{(2)}$ and tests the residual for a unit root. It is a two-step procedure whose answer depends on which series is the regressand, and it can only find one relation. Johansen estimates all $r$ vectors jointly by maximum likelihood, treats all series symmetrically, and has an exact LR test for $r$. That is why the engine uses it.

---

## 2. Johansen test: from VAR to an eigenvalue problem

### 2.1 VAR → VECM

Start with a VAR($p$) in levels:
$$Y_t = c + A_1Y_{t-1}+\dots+A_pY_{t-p}+\varepsilon_t,\qquad \varepsilon_t\sim N(0,\Omega).$$
Subtract $Y_{t-1}$ and regroup (add and subtract $A_jY_{t-j+1}$ terms):
$$\boxed{\;\Delta Y_t = c + \Pi\,Y_{t-1} + \sum_{j=1}^{k}\Gamma_j\,\Delta Y_{t-j}+\varepsilon_t\;},\qquad k=p-1,$$
$$\Pi=-\Big(I-\sum_{i=1}^pA_i\Big),\qquad \Gamma_j=-\sum_{i=j+1}^{p}A_i .$$
This is the **vector error-correction model (VECM)**. In the code, $k$ is `k_ar_diff`, chosen by BIC/AIC over VAR($p$) fits (`select_lag_order`).

### 2.2 Rank of $\Pi$ is the cointegration rank

* $\operatorname{rank}\Pi=0$: $\Pi=0$, the system is a VAR in differences, with no long-run relation.
* $\operatorname{rank}\Pi=n$: $Y_t$ is stationary in levels.
* $0<\operatorname{rank}\Pi=r<n$: $\Pi=\alpha\beta^\top$ with $\alpha,\beta\in\mathbb R^{n\times r}$ (Granger representation theorem).

Then $\beta^\top Y_{t-1}$ are the $r$ stationary spreads and $\alpha$ are the **adjustment (speed-of-correction) coefficients**: how strongly each series reacts to last period's disequilibrium. The left-hand side $\Delta Y_t$ and everything on the right other than $\Pi Y_{t-1}$ is $I(0)$, so $\Pi Y_{t-1}$ must be $I(0)$ too, which forces $\beta^\top Y\sim I(0)$.

Testing $H_0:\operatorname{rank}\Pi\le r$ is therefore a test of rank. Note that $\beta$ is only identified up to a basis of its column space, so the code normalises each vector so its first coefficient equals 1:
$\beta=(1,-\gamma)^\top$ for a pair, meaning *long 1 unit of A, short $\gamma$ units of B (in log terms)*.

### 2.3 Concentrating out nuisance parameters (Frisch–Waugh)

Let $Z_t=(1,\Delta Y_{t-1},\dots,\Delta Y_{t-k})$ be the "short-run" regressors, and $N$ the effective sample size. Regress $\Delta Y_t$ and $Y_{t-1}$ on $Z_t$ and keep the residuals:
$$R_{0t}=\Delta Y_t-\hat{\mathcal P}_Z\Delta Y_t,\qquad R_{1t}=Y_{t-1}-\hat{\mathcal P}_ZY_{t-1}.$$
(`_residualise` in the code.) By the Frisch–Waugh theorem the VECM becomes the **reduced-rank regression**
$$R_{0t}=\alpha\beta^\top R_{1t}+e_t .$$
Define the moment matrices
$$S_{ij}=\frac1N\sum_t R_{it}R_{jt}^\top,\qquad i,j\in\{0,1\}.$$

### 2.4 Likelihood and the eigenvalue problem

For fixed $\beta$ the Gaussian likelihood is maximised by OLS in $\alpha$:
$$\hat\alpha(\beta)=S_{01}\beta(\beta^\top S_{11}\beta)^{-1},\qquad
\hat\Omega(\beta)=S_{00}-S_{01}\beta(\beta^\top S_{11}\beta)^{-1}\beta^\top S_{10}.$$
The maximised likelihood satisfies $L_{\max}^{-2/N}\propto|\hat\Omega(\beta)|$, and with the identity $|A - BD^{-1}C| = |A||D - CA^{-1}B|/|D|$,
$$|\hat\Omega(\beta)|=|S_{00}|\;\frac{|\beta^\top(S_{11}-S_{10}S_{00}^{-1}S_{01})\beta|}{|\beta^\top S_{11}\beta|}.$$
Minimising this over $\beta$ is a generalised eigenvalue problem:
$$\boxed{\;\big|\lambda S_{11}-S_{10}S_{00}^{-1}S_{01}\big|=0\;}$$
with ordered solutions $1>\hat\lambda_1\ge\dots\ge\hat\lambda_n\ge0$ and eigenvectors $\hat v_1,\dots,\hat v_n$ normalised by $\hat v_i^\top S_{11}\hat v_j=\delta_{ij}$. The $\hat\lambda_i$ are the **squared canonical correlations** between $R_{0t}$ and $R_{1t}$: they measure how strongly each linear combination of the levels predicts the changes. The ML estimate of the cointegrating space is $\hat\beta=(\hat v_1,\dots,\hat v_r)$, and the maximised likelihood is
$$L_{\max}^{-2/N}\propto|S_{00}|\prod_{i=1}^r(1-\hat\lambda_i).$$

**Numerics.** With $S_{11}=LL^\top$ (Cholesky), the problem is equivalent to the *symmetric* eigenproblem for $M=L^{-1}S_{10}S_{00}^{-1}S_{01}L^{-\top}$ with $\beta=L^{-\top}w$. Symmetry gives real eigenvalues and orthonormal $w$, which is more stable than `eig` on a non-symmetric matrix. This is exactly `_johansen_native`.

### 2.5 The trace statistic

The likelihood-ratio statistic for $H_0:\operatorname{rank}\le r$ against $H_1:\operatorname{rank}=n$:
$$\boxed{\;\lambda_{\rm trace}(r)=-N\sum_{i=r+1}^{n}\ln\big(1-\hat\lambda_i\big)\;}$$
Intuition: if the true rank is $r$, the $n-r$ smallest canonical correlations are zero, so each $\ln(1-\hat\lambda_i)\approx0$ and the statistic is small.

**Null distribution.** Not $\chi^2$, because $Y_{t-1}$ is non-stationary. Asymptotically
$$\lambda_{\rm trace}(r)\;\Rightarrow\;\operatorname{tr}\Big\{\int_0^1 dB\,F^\top\Big(\int_0^1FF^\top du\Big)^{-1}\int_0^1F\,dB^\top\Big\},$$
where $B$ is an $(n-r)$-dimensional standard Brownian motion and $F$ a function of $B$ that depends on the deterministic terms. It is tabulated by simulation (MacKinnon–Haug–Michelis 1999; the 95% values `3.84, 15.49, 29.80, 47.85, 69.82` for $n-r=1,\dots,5$ are in the code).

**Sequential procedure.** Test $r=0$; if rejected, test $r\le1$; and so on. Stop at the first non-rejection. The rank is the number of rejections.

**Deterministic terms, and an empirical caveat we measured.** The engine uses an unrestricted constant in the VECM (`det_order=0` in statsmodels). Those critical values assume the common trends actually drift. In simulation with strong drift the native implementation has empirical size $\approx5.5\%$ at $T=500$ (nominal 5%), but with near-zero drift it over-rejects (about 9–13%). Real log-price drifts are small relative to volatility, so the engine adds a **second, residual-based confirmation** (§5.3) rather than relying on Johansen alone.

---

## 3. From cointegrating vector to a tradeable, beta-neutral spread

### 3.1 Spread and weights

For a pair with $\hat\beta=(1,-\hat\gamma)^\top$,
$$S_t=\ln P^A_t-\hat\gamma\ln P^B_t .$$
Since $d\ln P\approx$ simple return $r$, one unit of the spread position has dollar exposures $(\beta_1,\beta_2)$ and a P&L of $\beta^\top r_{t+1}\approx\Delta S_{t+1}$. We scale to **unit gross exposure**
$$w=\frac{\beta}{\sum_i|\beta_i|},\qquad \text{position return}=\text{pos}_t\cdot w^\top r_{t+1},\quad \text{pos}_t\in\{-1,0,1\}.$$
The engine requires $\beta_1\beta_2<0$ (a genuine long–short) by default.

### 3.2 Why the spread is isolated from market beta

Suppose log prices load on common $I(1)$ stochastic trends $F_t$ (market, sector, …) plus stationary noise:
$$Y_t=\Lambda F_t+u_t,\qquad F_t\ \text{is }I(1),\ u_t\sim I(0),\ \Lambda\in\mathbb R^{n\times m}.$$
Then
$$\beta^\top Y_t=(\Lambda^\top\beta)^\top F_t+\beta^\top u_t .$$
For this to be stationary the $I(1)$ part must vanish:
$$\boxed{\Lambda^\top\beta=0}$$
i.e. **$\beta$ is orthogonal to the loadings on every common stochastic trend**. Hence the spread has zero long-run exposure to those factors. In return space, with $r_t=\Lambda f_t+\Delta u_t$ ($f_t=\Delta F_t$),
$$\operatorname{Cov}\!\big(\beta^\top r_t,\;f_t\big)=\beta^\top\Lambda\operatorname{Var}(f_t)=0 .$$
So cointegration does not just *hope* for neutrality. It is *equivalent* to the hedge ratio cancelling the common-trend exposure, and this is the precise sense in which the strategy "isolates structural mispricing from broad market beta".

> **Caveat (tested, not assumed).** This is exact only for the common $I(1)$ factors. Short-run, time-varying betas can leak through, so the backtest *reports* the realised regression beta of strategy returns on the equal-weight market (`market_beta`) instead of assuming it is zero.

---

## 4. The Ornstein–Uhlenbeck process

Model the spread as the mean-reverting diffusion
$$\boxed{dX_t=\kappa(\mu-X_t)\,dt+\sigma\,dW_t},\qquad \kappa>0,\ \sigma>0 .$$
$\kappa$ = speed, $\mu$ = long-run mean, $\sigma$ = instantaneous volatility. The drift pulls $X$ toward $\mu$ proportionally to the displacement.

### 4.1 Solving the SDE (integrating factor)

Apply Itô's lemma to $f(t,X_t)=e^{\kappa t}X_t$. Since $f_t=\kappa e^{\kappa t}X$, $f_x=e^{\kappa t}$ and $f_{xx}=0$ (no Itô correction):
$$d\big(e^{\kappa t}X_t\big)=\kappa e^{\kappa t}X_tdt+e^{\kappa t}dX_t=\kappa\mu e^{\kappa t}dt+\sigma e^{\kappa t}dW_t .$$
Integrate from $0$ to $t$ and divide by $e^{\kappa t}$:
$$\boxed{X_t=\mu+(X_0-\mu)e^{-\kappa t}+\sigma\int_0^te^{-\kappa(t-s)}dW_s}$$

### 4.2 Moments

The stochastic integral is a zero-mean Gaussian, and by the Itô isometry
$$\mathbb E[X_t]=\mu+(X_0-\mu)e^{-\kappa t},\qquad
\operatorname{Var}[X_t]=\sigma^2\!\int_0^te^{-2\kappa(t-s)}ds=\frac{\sigma^2}{2\kappa}\big(1-e^{-2\kappa t}\big).$$
As $t\to\infty$ the law converges to the **stationary distribution**
$$X_\infty\sim N\!\Big(\mu,\ \sigma_{\rm eq}^2\Big),\qquad \boxed{\sigma_{\rm eq}=\frac{\sigma}{\sqrt{2\kappa}}}$$
with autocovariance $\operatorname{Cov}(X_t,X_{t+\tau})=\sigma_{\rm eq}^2e^{-\kappa\tau}$.

### 4.3 Half-life

The expected displacement decays as $e^{-\kappa t}$. Solving $e^{-\kappa t_{1/2}}=\tfrac12$:
$$\boxed{t_{1/2}=\frac{\ln 2}{\kappa}}$$
Interpretation: the expected time for a deviation to close half-way. The engine uses it to (i) reject spreads that revert too slowly (capital tied up, regime risk) or implausibly fast (noise/bid–ask bounce), and (ii) sanity-check holding periods: with $\kappa=0.07$/day, $t_{1/2}\approx10$ days.

### 4.4 Exact discretisation → AR(1)

Sampling at $\Delta t=1$ day,
$$X_{t+1}=\underbrace{\mu(1-b)}_{a}+b\,X_t+\varepsilon_{t+1},\quad b=e^{-\kappa},\quad \varepsilon\sim N\!\Big(0,\ \underbrace{\sigma^2\tfrac{1-e^{-2\kappa}}{2\kappa}}_{s_\varepsilon^2}\Big).$$
This is **exact**, with no Euler error (it is also how `simulate_ou` generates data). Inverting:
$$\kappa=-\ln b,\qquad \mu=\frac{a}{1-b},\qquad \sigma^2=\frac{2\kappa\,s_\varepsilon^2}{1-b^2},\qquad \sigma_{\rm eq}^2=\frac{s_\varepsilon^2}{1-b^2}.$$
Mean reversion requires $0<b<1$; $b\ge1$ means a unit root or explosive behaviour, and $b\le0$ means over-correction. The code returns "not mean-reverting" in those cases.

---

## 5. Estimating OU parameters

### 5.1 OLS = conditional MLE

Because the transition density is Gaussian and exact, the conditional log-likelihood is
$$\ell(a,b,s_\varepsilon^2)=-\tfrac N2\ln(2\pi s_\varepsilon^2)-\frac1{2s_\varepsilon^2}\sum_t(X_{t+1}-a-bX_t)^2 ,$$
whose maximiser in $(a,b)$ is ordinary least squares of $X_{t+1}$ on $(1,X_t)$ (`fit_ou`). The plug-in maps of §4.4 give $\hat\kappa,\hat\mu,\hat\sigma,\hat\sigma_{\rm eq}$ (by invariance of MLE).

### 5.2 Finite-sample bias

$\hat b$ is biased downward in finite samples, roughly $\mathbb E[\hat b]-b\approx-\frac{1+3b}{T}$ (Kendall, Marriott–Pope). So $\hat\kappa$ is biased **upward**: spreads look slightly faster than they are. With $T=252$ formation days and half-lives of 5–30 days the bias is small but real, which is one reason the engine uses a half-life *band* rather than trusting the point value.

### 5.3 A second stationarity check

The OLS standard error of $\hat b$ is $\widehat{\rm se}(\hat b)=\sqrt{\hat s_\varepsilon^2/\sum_t(X_t-\bar X)^2}$. The statistic $(\hat b-1)/\widehat{\rm se}(\hat b)$ is a **Dickey–Fuller $t$-statistic** for the unit-root null $b=1$. Its distribution is non-standard; because $\beta$ was also estimated, the relevant critical value is the Engle–Granger one ($\approx-3.37$ at 5% for two series). The engine requires $t<-3.0$ (`max_adf_t`), a conservative residual-based confirmation on top of Johansen (see the drift caveat in §2.5).

---

## 6. Optimal, volatility-adjusted z-score thresholds

### 6.1 Z-score

Standardise with the *stationary* volatility estimated in the formation window:
$$z_t=\frac{S_t-\hat\mu}{\hat\sigma_{\rm eq}} .$$
This makes thresholds **volatility-adjusted**: "$z=2$" means a 2-stationary-standard-deviation dislocation whatever the pair's scale, and the same $z$ rule is comparable across pairs with different spread volatilities.

### 6.2 The trading rule and the optimisation problem

Enter when $|z|\ge z^*$ (short the spread if $z>0$, long if $z<0$) and exit when $z$ crosses 0. Let $a=z^*\sigma_{\rm eq}$ and let $c$ be the **round-trip cost in spread units** ($c=2\cdot\text{bps}\cdot\sum_i|\beta_i|$). Each completed cycle earns $a-c$ in expectation (we capture the distance from $a$ back to $\mu$). A **cycle** is: wait at $\mu$ until $|X-\mu|=a$, then hold until $X$ returns to $\mu$.

By the **renewal–reward theorem**, the long-run profit per unit time is
$$\text{Rate}(a)=\frac{\mathbb E[\text{profit per cycle}]}{\mathbb E[\text{cycle duration}]}=\frac{a-c}{\mathbb E[\tau_{\rm wait}]+\mathbb E[\tau_{\rm hold}]} .$$
Larger $a$ means more profit per trade but fewer trades. Costs penalise small $a$. The optimum balances these.

### 6.3 Expected first-passage times

Centre $X$ at $\mu$. Expected hitting times $T(x)$ solve $\mathcal L T=-1$ with the OU generator $\mathcal L=-\kappa x\,\partial_x+\tfrac{\sigma^2}2\partial_{xx}$:
$$\tfrac{\sigma^2}{2}T''-\kappa xT'=-1\iff\Big(e^{-\kappa x^2/\sigma^2}T'\Big)'=-\frac2{\sigma^2}e^{-\kappa x^2/\sigma^2}.$$
Use the dimensionless coordinate $u=x\sqrt\kappa/\sigma$ and the identity $\int_0^xe^{-\kappa y^2/\sigma^2}dy=\frac{\sigma}{\sqrt\kappa}\frac{\sqrt\pi}2\operatorname{erf}(u)$.

*Hold phase*, from $a$ back to $0$ (boundary $T(0)=0$, natural at $+\infty$):
$$\mathbb E[\tau_{\rm hold}]=\frac{\sqrt\pi}{\kappa}\int_0^{u_a}e^{u^2}\operatorname{erfc}(u)\,du .$$
*Wait phase*, from $0$ until exit of $(-a,a)$ (boundary $T(\pm a)=0$, symmetry $T'(0)=0$):
$$\mathbb E[\tau_{\rm wait}]=\frac{\sqrt\pi}{\kappa}\int_0^{u_a}e^{u^2}\operatorname{erf}(u)\,du .$$
**They add up to something remarkably clean**: since $\operatorname{erf}+\operatorname{erfc}=1$,
$$\mathbb E[\tau_{\rm cycle}]=\frac{\sqrt\pi}{\kappa}\int_0^{u_a}e^{u^2}du=\boxed{\frac{\pi}{2\kappa}\,\operatorname{erfi}\!\Big(\frac{z^*}{\sqrt2}\Big)}$$
using $u_a=a\sqrt\kappa/\sigma=z^*/\sqrt2$ and $\operatorname{erfi}(w)=\frac2{\sqrt\pi}\int_0^we^{t^2}dt$.
(`expected_cycle_time`; the two-quadrature version `expected_time_numeric` is unit-tested against it, and a 2-million-step Monte-Carlo simulation agrees within 8%.)

### 6.4 The optimal threshold

Writing $\tilde c=c/\sigma_{\rm eq}$,
$$\text{Rate}(z)=\frac{2\kappa\,\sigma_{\rm eq}}{\pi}\cdot\frac{z-\tilde c}{\operatorname{erfi}(z/\sqrt2)},\qquad
z^*=\arg\max_{z>\tilde c}\frac{z-\tilde c}{\operatorname{erfi}(z/\sqrt2)} .$$
Setting the derivative to zero, using $\frac d{dz}\operatorname{erfi}(z/\sqrt2)=\sqrt{2/\pi}\,e^{z^2/2}$, gives the first-order condition
$$\operatorname{erfi}\!\Big(\frac{z^*}{\sqrt2}\Big)=\sqrt{\frac2\pi}\,(z^*-\tilde c)\,e^{z^{*2}/2},$$
solved numerically by `optimal_entry_threshold`. Properties, all visible in `docs/img/optimal_threshold.png`:

* $z^*$ depends **only on the cost-to-volatility ratio** $\tilde c$, not on $\kappa$. Speed scales the *rate* (profit per day) but not the *location* of the optimum.
* $z^*$ is increasing in $\tilde c$. With $\tilde c\to0$, $z^*\to0$ (infinitely many infinitesimal trades), which is why realistic costs matter and the code floors at $z_{\min}=0.3$.
* Expected trades per year $=252/\mathbb E[\tau_{\rm cycle}]$, a direct sanity check against the backtest's trade count.

### 6.5 Stop-loss

The OU model has no jumps and an unbounded Gaussian support, so a very large $|z|$ is evidence the model is *wrong* (structural break), not an even better entry. The engine stops out at $|z|\ge z_{\rm stop}=4$ and stays flat until $|z|$ falls back inside the entry band.

---

## 7. P&L, costs, and walk-forward discipline

**Windows.** For window $m$: use days $[s,s+F)$ ("formation") to estimate $\beta,\hat\mu,\hat\sigma_{\rm eq},\hat\kappa,z^*$, then trade days $[s+F,s+F+H)$ ("trading") with all parameters **frozen**; then roll forward by $H$. Parameters at time $t$ only ever use data $\le t$.

**Timing.** A signal observed at close $t$ becomes position $\text{pos}_t$, which earns the return of day $t+1$:
$$\text{pnl}_{t+1}=\text{pos}_t\,w^\top r_{t+1}-\underbrace{\text{bps}\cdot10^{-4}\,|\text{pos}_{t+1}-\text{pos}_t|}_{\text{cost on gross notional}} .$$
Unit tests verify that appending future data does not change earlier P&L (`test_no_lookahead...`).

**Portfolio.** The top-$N$ pairs (ranked by trace statistic over its critical value) each get capital $1/N$; unfilled slots stay in cash.

**Metrics** (`metrics.py`): annualised Sharpe $=\sqrt{252}\,\bar r/s_r$, CAGR, max drawdown, and the regression beta $\beta_M=\operatorname{Cov}(r,r_M)/\operatorname{Var}(r_M)$ against the equal-weight universe.

---

## 8. Statistical pitfalls

| Pitfall | Why it matters | What the engine does |
|---|---|---|
| **Multiple testing** | With $n$ assets there are $n(n-1)/2$ pairs; at 5% size about 1 in 20 independent pairs "pass". | Rank by trace stat, require the Dickey–Fuller confirmation and a half-life band, trade only the top $N$; always evaluate **out of sample**. |
| **Critical values assume drift** | Weak drift ⇒ over-rejection (measured ≈ 9–13%). | Second residual-based test (§5.3). |
| **Estimation error in $\kappa$ and $\hat\beta$** | AR(1) bias (§5.2); $\hat\beta$ is super-consistent but noisy. | Half-life band, rolling re-estimation, frozen parameters in the trade window. |
| **Cointegration breaks** | The OU model has no regime change. | Stop-loss at $|z|=4$, re-test every window. |
| **Gaussian, constant-vol OU** | Real spreads have fat tails and volatility clustering. | Thresholds are standardised by $\hat\sigma_{\rm eq}$; stress-test with costs and stops. |
| **Costs, borrow, shorting limits** | Edge per trade can be a few bps. | Cost enters the *threshold optimisation itself*, not just the P&L. Borrow fees and short-sale constraints are *not* modelled. |

---

## 9. References

* Johansen, S. (1991). *Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models.* Econometrica 59(6).
* Johansen, S. (1995). *Likelihood-Based Inference in Cointegrated Vector Autoregressive Models.* Oxford University Press.
* MacKinnon, J., Haug, A., Michelis, L. (1999). *Numerical distribution functions of likelihood ratio tests for cointegration.* J. Applied Econometrics 14.
* Engle, R., Granger, C. (1987). *Co-integration and error correction.* Econometrica 55(2).
* Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis.* Wiley.
* Bertram, W. (2010). *Analytic solutions for optimal statistical arbitrage trading.* Physica A 389.
* Leung, T., Li, X. (2015). *Optimal mean reversion trading with transaction costs and stop-loss exit.* Int. J. Theoretical and Applied Finance 18(3).
* Avellaneda, M., Lee, J.-H. (2010). *Statistical arbitrage in the US equities market.* Quantitative Finance 10(7).
* Ornstein, L., Uhlenbeck, G. (1930). *On the theory of the Brownian motion.* Physical Review 36.
