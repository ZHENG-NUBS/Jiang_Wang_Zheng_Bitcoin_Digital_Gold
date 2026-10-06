"""
23_portfolio_robust.py — Supplementary portfolio-optimization analysis

over a short and unusual sample period, mean–variance optimization using the
sample mean and sample covariance is very unstable; the objective of the "aggressive"
portfolio is not precise enough. Please give the exact optimization program, the expected-
return target or constraint, the annualization method, and the turnover assumption;
consider robust optimization, resampling optimization, Bayesian or shrinkage estimation,
CVaR optimization, and out-of-sample performance.

This script fills in, on top of script 17 (rolling out-of-sample 9 strategies, costs,
Ledoit–Wolf Sharpe test):

  1. Fifteen strategies, all written as exact programs (see Section 0 of the output),
     grouped in three tiers by whether they need expected returns:
     A Do not need expected returns: 1/N, 60/40, minimum variance (sample / LW shrinkage),
       risk parity (LW), minimum CVaR95
     B Need expected returns, plugging in sample estimates directly: maximum Sharpe,
       mean–variance utility (γ = 5), minimum variance subject to a target return,
       minimum CVaR subject to a target return (mean–CVaR), aggressive (maximize expected
       return subject to annualized volatility ≤ 18%)
     C Need expected returns, explicitly handling estimation error: mean–variance utility
       (Bayes–Stein mean + LW covariance; Jorion 1986; DeMiguel et al. 2009), resampled
       maximum Sharpe (Michaud 1998), robust mean–variance (ellipsoidal uncertainty set;
       Ceria & Stubbs 2006), aggressive (Bayes–Stein + LW)
     Target return r* = the 1/N portfolio's sample expected return over the estimation
     window, which is always feasible.
  2. Annualization and turnover conventions unified and stated (Section 0); costs per unit
     of traded notional, reporting one-way turnover, annualized turnover, and the
     break-even cost.
  3. Out-of-sample metrics: annualized arithmetic return, geometric return, volatility,
     Sharpe, certainty-equivalent return CER (γ = 5), ES95, maximum drawdown, turnover,
     IBIT weight. Tests: Sharpe difference against 1/N (Ledoit–Wolf 2008, HAC), block
     bootstrap of the Sharpe difference and the CER difference, Holm correction; paired
     comparison of group C against group B.
  4. Two samples: the main sample (IBIT, 607 days) and the long sample (BTC-USD spot minus
     a 0.25% annual fee to mimic a spot ETF, 2952 days from 2014-09, about 2450
     out-of-sample days, covering the three declines of 2018, 2020, 2022).
  5. Robustness grid: estimation window {125 or 250, 250 or 500, expanding window} ×
     rebalancing {5, 21, 63} days; the long sample split into three subperiods.

Risk-free rate: the main sample uses the paper's 4.3% (^IRX sample mean 4.32%). The long
sample needs a daily rate: if data/raw/ contains IDX_IRX__*L.csv or DTB3__*L.csv (fetched
by 01e_fetch_rf_long.py), it is used automatically (lagged one day); otherwise 0 is used
temporarily and this is noted in the output (the CER difference is unaffected by rf; the
Sharpe and maximum-Sharpe weights are affected).

Dependencies: numpy, pandas, scipy, matplotlib. Can be run standalone.
Usage:
    python 23_portfolio_robust.py
    python 23_portfolio_robust.py --data data/processed/returns__20260921.csv --raw data/raw --outdir out/23
    python 23_portfolio_robust.py --quick          # fewer grid points and bootstrap draws, for a trial run
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats, sparse
from scipy.optimize import minimize, linprog

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260928
RF_MAIN = 0.043
GAMMA = 5.0
SIG_STAR = 0.18
CVAR_A = 0.05
KAPPA = math.sqrt(stats.chi2.ppf(0.95, 4))
BTC_FEE = 0.0025
COSTS_BP = (0, 5, 10, 25, 50)
MAIN_BP = 10
HAC_LAG = 5
L_BOOT = 5
B_RES_OOS = 300
B_RES_IS = 2000
B_W_MAIN, B_W_LONG = 500, 300
B_TEST = 2000
N_ASSET = 4
SUBSETS = [np.array(s) for k in range(1, N_ASSET + 1) for s in itertools.combinations(range(N_ASSET), k)]

GROUP = {"A": "A no expected returns", "B": "B needs expected returns (sample estimates plugged in)",
         "C": "C needs expected returns (handles estimation error)"}
STRATS = [  # code, group, display name
    ("EW", "A", "1/N"),
    ("6040", "A", "60/40 SPY–TLT"),
    ("GMV-S", "A", "Min variance (sample)"),
    ("GMV-LW", "A", "Min variance (LW)"),
    ("RP", "A", "Risk parity (LW)"),
    ("MinCVaR", "A", "Min CVaR95"),
    ("MSR-S", "B", "Max Sharpe (sample)"),
    ("MV-S", "B", "Mean–variance γ=5 (sample)"),
    ("TR-MV", "B", "Target return · min variance"),
    ("TR-CVaR", "B", "Target return · min CVaR"),
    ("AGG-S", "B", "Aggressive σ≤18% (sample)"),
    ("MV-BS", "C", "Mean–variance γ=5 (Bayes–Stein+LW)"),
    ("MSR-RS", "C", "Max Sharpe (resampled)"),
    ("RMV", "C", "Robust mean–variance"),
    ("AGG-BS", "C", "Aggressive σ≤18% (BS+LW)"),
]
CODES = [s[0] for s in STRATS]
NAME = {s[0]: s[2] for s in STRATS}
GRP = {s[0]: s[1] for s in STRATS}
PAIRS = [("MV-BS", "MV-S"), ("MSR-RS", "MSR-S"), ("RMV", "MV-S"), ("AGG-BS", "AGG-S"), ("GMV-LW", "GMV-S")]

LINES: list[str] = []
CHECKS: list[str] = []


def w(s=""):
    LINES.append(s)
    print(s, flush=True)


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    CHECKS.append(f"[{tag}] {label} {detail}")
    w(f"   CHECK [{tag}] {label} {detail}")
    return bool(ok)


def pf(p):
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def pct(a, q=(2.5, 97.5)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, q)] if len(a) else [np.nan, np.nan]


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    nb = rng.random((n, R)) < p
    jp = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(nb[t], jp[t], (idx[t - 1] + 1) % n)
    return idx


# ============================================================ Estimators
def ledoit_wolf_cc(X):
    """Ledoit–Wolf shrinkage with a constant-correlation target (Ledoit & Wolf 2004);
    line-by-line identical to script 17."""
    T, N = X.shape
    Xc = X - X.mean(0)
    S = Xc.T @ Xc / T
    var = np.diag(S); sd = np.sqrt(var)
    R = S / np.outer(sd, sd)
    rbar = (R.sum() - N) / (N * (N - 1))
    F = rbar * np.outer(sd, sd); np.fill_diagonal(F, var)
    Y = Xc ** 2
    pi_mat = Y.T @ Y / T - S ** 2
    pi = pi_mat.sum()
    term = ((Xc ** 3).T @ Xc) / T - var[:, None] * S
    rho = np.diag(pi_mat).sum()
    for i in range(N):
        for j in range(N):
            if i != j:
                rho += rbar / 2 * (math.sqrt(var[j] / var[i]) * term[i, j] + math.sqrt(var[i] / var[j]) * term[j, i])
    gamma = ((F - S) ** 2).sum()
    kappa = (pi - rho) / gamma if gamma > 0 else 0.0
    delta = max(0.0, min(1.0, kappa / T))
    return delta * F + (1 - delta) * S, delta


def bayes_stein(X):
    """Jorion (1986) Bayes–Stein mean: shrink toward the minimum-variance portfolio's mean.
    Returns (μ_BS, shrinkage coefficient φ)."""
    T, N = X.shape
    mu = X.mean(0)
    S = np.cov(X, rowvar=False) * (T - 1) / (T - N - 2)
    Si = np.linalg.inv(S); one = np.ones(N)
    mu0 = float(one @ Si @ mu / (one @ Si @ one))
    d = mu - mu0
    phi = (N + 2) / ((N + 2) + T * float(d @ Si @ d))
    return (1 - phi) * mu + phi * mu0, float(phi)


# ============================================================ Optimizers (all: w ≥ 0, 1′w = 1)
def _norm(v):
    v = np.clip(v, 0, None); s = v.sum()
    return v / s if s > 0 else np.ones_like(v) / len(v)


def gmv(Sig):
    """Minimum variance: exact solution by enumerating support sets (a convex problem;
    take the smallest-variance stationary point w ∝ Σ_S⁻¹1 over all faces)."""
    N = len(Sig); best, bw = np.inf, None
    for S in SUBSETS:
        y = np.linalg.solve(Sig[np.ix_(S, S)], np.ones(len(S)))
        wS = y / y.sum()
        if np.any(wS < -1e-12):
            continue
        v = float(wS @ Sig[np.ix_(S, S)] @ wS)
        if v < best:
            best = v; bw = np.zeros(N); bw[S] = np.clip(wS, 0, None)
    return bw / bw.sum()


def max_sharpe_raw(mu, Sig, rf):
    """Long-only maximum Sharpe: the Sharpe ratio is homogeneous of degree 0, and the
    stationary point on any face satisfies w_S ∝ Σ_S⁻¹(μ_S − rf); enumerate and take the
    maximum (exact). Returns (w, Sharpe)."""
    N = len(mu); e = mu - rf; best, bw = -np.inf, None
    for S in SUBSETS:
        y = np.linalg.solve(Sig[np.ix_(S, S)], e[S])
        s = y.sum()
        if abs(s) < 1e-15:
            continue
        wS = y / s
        if np.any(wS < -1e-12):
            continue
        v = np.zeros(N); v[S] = np.clip(wS, 0, None); v /= v.sum()
        sr = float(v @ e) / math.sqrt(float(v @ Sig @ v))
        if sr > best:
            best, bw = sr, v
    return bw, best


def max_sharpe(mu, Sig, rf):
    """Maximum Sharpe; if the attainable maximum Sharpe ≤ 0 (no portfolio in the window
    has a positive expected excess return), the tangency portfolio is meaningless
    (in that case "maximum Sharpe" equals putting everything into the highest-volatility
    asset), so we take the minimum-variance portfolio instead."""
    v, sr = max_sharpe_raw(mu, Sig, rf)
    return v if sr > 0 else gmv(Sig)


def max_sharpe_batch(MU, SIG, rf):
    B, N = MU.shape; E = MU - rf
    best = np.full(B, -np.inf); W = np.tile(np.ones(N) / N, (B, 1))
    for S in SUBSETS:
        sub = SIG[:, S][:, :, S]
        y = np.linalg.solve(sub, E[:, S][..., None])[..., 0]
        s = y.sum(1)
        ok = np.abs(s) > 1e-15
        wS = y / np.where(ok, s, 1.0)[:, None]
        ok &= (wS >= -1e-12).all(1)
        v = np.zeros((B, N)); v[:, S] = np.clip(wS, 0, None)
        sv = v.sum(1); v /= np.where(sv > 0, sv, 1.0)[:, None]
        num = (v * E).sum(1); var = np.einsum("bi,bij,bj->b", v, SIG, v)
        sr = np.where(ok, num / np.sqrt(np.maximum(var, 1e-300)), -np.inf)
        upd = sr > best
        best[upd] = sr[upd]; W[upd] = v[upd]
    for b in np.where(best <= 0)[0]:                   # same as max_sharpe: when the max Sharpe ≤ 0, take the min variance
        W[b] = gmv(SIG[b])
    return W


def resampled_ms(mu, Sig, T, rf, B, rng):
    """Michaud (1998) resampling: μ_b ~ N(μ̂, Σ̂/T), Σ_b ~ Wishart(T−1, Σ̂/(T−1)) (the
    sampling distribution of the sample mean and covariance under normality);
    compute the maximum-Sharpe weights for each (μ_b, Σ_b) and average."""
    MU = rng.multivariate_normal(mu, Sig / T, size=B)
    SIG = stats.wishart.rvs(df=T - 1, scale=Sig / (T - 1), size=B, random_state=rng)
    return max_sharpe_batch(MU, SIG, rf).mean(0)


def risk_parity(Sig):
    """Risk parity: equalize each asset's risk contribution w_i(Σw)_i/σ_p; same as script 17."""
    N = len(Sig)

    def obj(v):
        v = np.clip(v, 1e-8, None)
        pv = math.sqrt(max(v @ Sig @ v, 1e-18))
        rc = v * (Sig @ v) / pv
        return ((rc - rc.mean()) ** 2).sum()
    res = minimize(obj, np.ones(N) / N, method="SLSQP", bounds=[(1e-6, 1)] * N,
                   constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}], options={"maxiter": 1000, "ftol": 1e-14})
    return _norm(res.x)


def min_cvar(X, mu=None, r_star=None, alpha=CVAR_A):
    """Rockafellar–Uryasev: min ζ + 1/(αT) Σ u_t, s.t. u_t ≥ −x_t′w − ζ, u ≥ 0,
    1′w = 1, w ≥ 0 [, μ′w ≥ r*]."""
    T, N = X.shape
    c = np.concatenate([np.zeros(N), [1.0], np.full(T, 1 / (alpha * T))])
    A = sparse.hstack([sparse.csr_matrix(-X), sparse.csr_matrix(-np.ones((T, 1))), -sparse.identity(T)]).tocsr()
    b = np.zeros(T)
    if r_star is not None:
        A = sparse.vstack([A, sparse.csr_matrix(np.concatenate([-mu, [0.0], np.zeros(T)])[None, :])]).tocsr()
        b = np.concatenate([b, [-r_star]])
    Aeq = np.concatenate([np.ones(N), [0.0], np.zeros(T)])[None, :]
    res = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=[1.0], bounds=[(0, 1)] * N + [(None, None)] + [(0, None)] * T,
                  method="highs")
    return _norm(res.x[:N]) if res.success else np.ones(N) / N


def cvar_of(r, alpha=CVAR_A):
    """Historical ES/CVaR: the average loss of the worst ⌈αn⌉ returns (a fixed count,
    avoiding jumpiness near the quantile when there are ties)."""
    k = int(math.ceil(alpha * len(r)))
    return float(-np.sort(r)[:k].mean())


def _slsqp(obj, jac, N, starts, ineq=()):
    cons = [{"type": "eq", "fun": lambda v: v.sum() - 1, "jac": lambda v: np.ones(N)}] + list(ineq)
    best = None
    for x0 in starts:
        r = minimize(obj, x0, jac=jac, method="SLSQP", bounds=[(0, 1)] * N, constraints=cons,
                     options={"maxiter": 1000, "ftol": 1e-13})
        viol = abs(r.x.sum() - 1) + sum(max(0.0, -c["fun"](r.x)) for c in ineq)
        if viol < 1e-7 and (best is None or r.fun < best.fun):
            best = r
    return _norm(best.x) if best is not None else None


# All the following SLSQP programs use annualized inputs (μ×252, Σ×252), so the objective
# is of order 0.1 and numerically more stable; the solution is the same as with daily inputs.
def mv_utility(mu, Sig, gamma=GAMMA):
    """max w′μ − (γ/2) w′Σw."""
    N = len(mu)
    return _slsqp(lambda v: -(v @ mu - gamma / 2 * v @ Sig @ v), lambda v: -(mu - gamma * Sig @ v), N,
                  [np.ones(N) / N, gmv(Sig)])


def mv_target(mu, Sig, r_star):
    """min w′Σw  s.t. μ′w ≥ r* (when r* ≤ the minimum-variance portfolio's expected return,
    the solution is the minimum-variance portfolio)."""
    N = len(mu); g = gmv(Sig)
    if g @ mu >= r_star:
        return g
    ineq = ({"type": "ineq", "fun": lambda v: v @ mu - r_star, "jac": lambda v: mu},)
    starts = [np.ones(N) / N] + ([np.eye(N)[int(np.argmax(mu))]])
    out = _slsqp(lambda v: v @ Sig @ v, lambda v: 2 * Sig @ v, N, starts, ineq)
    return out if out is not None else np.eye(N)[int(np.argmax(mu))]


def aggressive(mu, Sig, sig_star=SIG_STAR):
    """Aggressive: max μ′w  s.t. √(w′Σw) ≤ σ* (annualized). If σ* is below the
    minimum-variance portfolio's volatility the problem is infeasible, so take the
    minimum-variance portfolio."""
    N = len(mu); g = gmv(Sig)
    if g @ Sig @ g > sig_star ** 2:
        return g
    ineq = ({"type": "ineq", "fun": lambda v: sig_star ** 2 - v @ Sig @ v, "jac": lambda v: -2 * Sig @ v},)
    ew = np.ones(N) / N
    starts = [g] + ([ew] if ew @ Sig @ ew <= sig_star ** 2 else [])
    out = _slsqp(lambda v: -(v @ mu), lambda v: -mu, N, starts, ineq)
    return out if out is not None else g


def robust_mv(mu, Sig, theta, gamma=GAMMA, kappa=KAPPA):
    """Robust mean–variance: max w′μ̂ − κ √(w′Θw) − (γ/2) w′Σw; Θ = diag(σ̂²_i)/T is the
    covariance of the mean estimation error, κ = √χ²_N(0.95), i.e. optimizing over the
    worst case for μ inside the 95% ellipsoid (Ceria & Stubbs 2006; Goldfarb & Iyengar 2003)."""
    N = len(mu)

    def f(v):
        return -(v @ mu - kappa * math.sqrt(max(v @ (theta * v), 1e-300)) - gamma / 2 * v @ Sig @ v)

    def g(v):
        s = math.sqrt(max(v @ (theta * v), 1e-300))
        return -(mu - kappa * theta * v / s - gamma * Sig @ v)
    return _slsqp(f, g, N, [np.ones(N) / N, gmv(Sig)])


def all_weights(X, rf_d, rng, B_res, i_spy, i_tlt):
    """Compute the weights of the 15 strategies on one estimation window X (T×4 daily returns)."""
    T, N = X.shape
    a = 252.0
    mu = X.mean(0); S = np.cov(X, rowvar=False)
    LW, delta = ledoit_wolf_cc(X)
    mu_bs, phi = bayes_stein(X)
    r_star = float(mu.mean())                          # the 1/N portfolio's sample expected return (daily)
    W = {}
    W["EW"] = np.ones(N) / N
    v = np.zeros(N); v[i_spy] = 0.6; v[i_tlt] = 0.4; W["6040"] = v
    W["GMV-S"] = gmv(S)
    W["GMV-LW"] = gmv(LW)
    W["RP"] = risk_parity(LW)
    W["MinCVaR"] = min_cvar(X)
    W["MSR-S"] = max_sharpe(mu, S, rf_d)
    neg_s = max_sharpe_raw(mu, S, rf_d)[1] <= 0
    W["MV-S"] = mv_utility(a * mu, a * S)
    W["TR-MV"] = mv_target(a * mu, a * S, a * r_star)
    W["TR-CVaR"] = min_cvar(X, mu=mu, r_star=r_star)
    W["AGG-S"] = aggressive(a * mu, a * S)
    W["MV-BS"] = mv_utility(a * mu_bs, a * LW)
    W["MSR-RS"] = resampled_ms(mu, S, T, rf_d, B_res, rng)
    W["RMV"] = robust_mv(a * mu, a * LW, a * a * np.diag(S) / T)
    W["AGG-BS"] = aggressive(a * mu_bs, a * LW)
    return W, dict(delta=delta, phi=phi, neg_s=bool(neg_s))


# ============================================================ Backtest and metrics
def run_setting(R, rf, start, win, rebal, rng, B_res, i_spy, i_tlt):
    """Rolling (win an integer) or expanding (win=None) window; the weights on day t use
    only data up to t−1. Intra-portfolio weights drift with prices between rebalances;
    the traded notional on a rebalance day is Σ|w_new − w_drift| (1 at initial
    positioning)."""
    T, N = R.shape
    reb = list(range(start, T, rebal))
    WL = {c: [] for c in CODES}; info = []
    for t in reb:
        X = R[t - win:t] if win else R[:t]
        Wt, inf_ = all_weights(X, rf[t], rng, B_res, i_spy, i_tlt)
        info.append(inf_)
        for c in CODES:
            WL[c].append(Wt[c])
    n = T - start
    out = {}
    reb_pos = np.array(reb) - start
    for c in CODES:
        pr = np.empty(n); trade = np.zeros(n); w_cur = None; k = 0
        for i in range(n):
            t = start + i
            if k < len(reb) and t == reb[k]:
                w_new = WL[c][k]
                trade[i] = float(np.abs(w_new - (w_cur if w_cur is not None else 0)).sum())
                w_cur = w_new; k += 1
            pr[i] = float(w_cur @ R[t])
            gr = w_cur * (1 + R[t]); w_cur = gr / gr.sum()
        out[c] = dict(gross=pr, trade=trade, w_reb=np.array(WL[c]), reb_pos=reb_pos)
    return out, info


def net(o, bp):
    return o["gross"] - bp / 1e4 * o["trade"]


def metrics(r, rf, o, asset0=0):
    ex = r - rf; n = len(r); yrs = n / 252
    cum = np.cumprod(1 + r)
    to = o["trade"][o["reb_pos"][1:]] / 2                 # one-way turnover, excluding initial positioning
    wb = o["w_reb"][:, asset0]
    return dict(ret_a=float(252 * r.mean()), cagr=float(cum[-1] ** (1 / yrs) - 1), vol=float(r.std(ddof=1) * math.sqrt(252)),
                sharpe=float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)),
                cer=float(252 * (ex.mean() - GAMMA / 2 * ex.var(ddof=1))),
                es95=float(cvar_of(r)), mdd=float((cum / np.maximum.accumulate(cum) - 1).min()),
                to_reb=float(to.mean()) if len(to) else 0.0, to_ann=float(to.sum() / yrs),
                w0_mean=float(wb.mean()), w0_sd=float(wb.std()), w0_min=float(wb.min()), w0_max=float(wb.max()),
                w0_lt1=int((wb < 0.01).sum()), n_reb=int(len(wb)))


def cer_of(ex):
    return 252 * (ex.mean(0) - GAMMA / 2 * ex.var(0, ddof=1))


def sr_of(ex):
    return ex.mean(0) / ex.std(0, ddof=1) * math.sqrt(252)


def hac_cov(Z, lag=HAC_LAG):
    T = len(Z); Zc = Z - Z.mean(0)
    S = Zc.T @ Zc / T
    for l in range(1, lag + 1):
        A = Zc[l:].T @ Zc[:-l] / T
        S += (1 - l / (lag + 1)) * (A + A.T)
    return S / T


def sr_diff_test(r1, r2, rf, lag=HAC_LAG):
    """Ledoit & Wolf (2008): apply HAC to (μ1, μ2, E r1², E r2²), delta method gives the
    standard error of the Sharpe difference; same as script 17."""
    a, b = r1 - rf, r2 - rf
    Z = np.column_stack([a, b, a ** 2, b ** 2]); m = Z.mean(0); V = hac_cov(Z, lag)
    m1, m2, g1, g2 = m; s1, s2 = g1 - m1 ** 2, g2 - m2 ** 2
    grad = np.array([g1 / s1 ** 1.5, -g2 / s2 ** 1.5, -m1 / (2 * s1 ** 1.5), m2 / (2 * s2 ** 1.5)])
    se = math.sqrt(max(grad @ V @ grad, 1e-24)); d = m1 / math.sqrt(s1) - m2 / math.sqrt(s2)
    return d * math.sqrt(252), 2 * stats.norm.sf(abs(d / se))


def compare(r1, r2, rf, idx):
    e1, e2 = r1 - rf, r2 - rf
    dsr, p_hac = sr_diff_test(r1, r2, rf)
    E1, E2 = e1[idx], e2[idx]
    dsr_b = sr_of(E1) - sr_of(E2); dcer_b = cer_of(E1) - cer_of(E2)
    dcer = float(cer_of(e1) - cer_of(e2))
    return dict(dsr=float(dsr), p_hac=float(p_hac), p_sr_boot=float(np.mean(np.abs(dsr_b - dsr) >= abs(dsr))),
                dcer=dcer, dcer_ci=pct(dcer_b), p_cer=float(np.mean(np.abs(dcer_b - dcer) >= abs(dcer))))


def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def breakeven(o_s, o_ew, rf, cap_bp=500.0):
    """The cost c (bp per unit traded notional) that makes CER(strategy, c) = CER(1/N, c);
    if the strategy already fails to beat 1/N at zero cost, return None."""
    def f(bp):
        return cer_of(net(o_s, bp) - rf) - cer_of(net(o_ew, bp) - rf)
    if f(0.0) <= 0:
        return None
    if f(cap_bp) > 0:
        return np.inf
    lo, hi = 0.0, cap_bp
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


# ============================================================ Data
def find_main():
    c = []
    for base in (HERE, os.getcwd()):
        c += [os.path.abspath(p) for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv"))
              if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p))]
    return sorted(set(c), key=os.path.basename)[-1] if c else None


def find_raw():
    for base in (HERE, os.getcwd()):
        d = os.path.join(base, "data", "raw")
        if glob.glob(os.path.join(d, "BTC-USD__*L.csv")):
            return d
    return None


def load_long(raw):
    S = {}
    for t, f in (("BTC", "BTC-USD"), ("GLD", "GLD"), ("SPY", "SPY"), ("TLT", "TLT")):
        p = sorted(glob.glob(os.path.join(raw, f"{f}__*L.csv")))[-1]
        d = pd.read_csv(p, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
        S[t] = d["AdjClose"]
    cal = S["SPY"].index
    P = pd.DataFrame({k: v.reindex(cal) for k, v in S.items()})
    P = P[P.index >= S["BTC"].index.min()]
    R = P.pct_change().dropna()
    R = R[R.index <= pd.Timestamp("2026-06-15")]
    R["BTC"] = R["BTC"] - BTC_FEE / 252                 # simulate a spot ETF's annual fee
    # Risk-free rate
    rf, src = None, "No long-sample rate file found; using rf = 0 for now (the CER difference is unaffected; the Sharpe and max-Sharpe weights are affected)"
    for pat, col in (("IDX_IRX__*L.csv", "Close"), ("DTB3__*L.csv", "DTB3")):
        fs = sorted(glob.glob(os.path.join(raw, pat)))
        if fs:
            d = pd.read_csv(fs[-1], parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
            y = pd.to_numeric(d[col], errors="coerce").reindex(R.index.union(d.index)).ffill().shift(1).reindex(R.index)
            if y.isna().mean() < 0.01:
                rf = (y.bfill() / 100 / 252).values
                src = f"{os.path.basename(fs[-1])} (lagged one day, annualized % ÷ 100 ÷ 252)"
                break
    if rf is None:
        rf = np.zeros(len(R))
    return R, rf, src


# ============================================================ Self-test
def selftest(Rm):
    ok = True
    g = np.random.default_rng(1)
    worst_ms, worst_gmv = 0.0, 0.0
    for k in range(300):
        A = g.standard_normal((4, 4)); Sig = A @ A.T / 4 + 0.05 * np.eye(4)
        mu = g.normal(0.05 if k % 3 else -0.05, 0.1, 4); rf = 0.02
        wa, _ = max_sharpe_raw(mu, Sig, rf)
        best = -np.inf
        for x0 in list(np.eye(4)) + [np.ones(4) / 4] + list(g.dirichlet(np.ones(4), 5)):
            r = minimize(lambda v: -((v @ mu - rf) / math.sqrt(v @ Sig @ v)), x0, method="SLSQP", bounds=[(0, 1)] * 4,
                         constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}], options={"ftol": 1e-14, "maxiter": 1000})
            if abs(r.x.sum() - 1) < 1e-6 and r.x.min() > -1e-8:
                best = max(best, -r.fun)
        worst_ms = max(worst_ms, best - (wa @ mu - rf) / math.sqrt(wa @ Sig @ wa))
        wg = gmv(Sig)
        r = minimize(lambda v: v @ Sig @ v, np.ones(4) / 4, method="SLSQP", bounds=[(0, 1)] * 4,
                     constraints=[{"type": "eq", "fun": lambda v: v.sum() - 1}], options={"ftol": 1e-15})
        worst_gmv = max(worst_gmv, wg @ Sig @ wg - r.fun)
    ok &= check("Max-Sharpe enumeration solution ≥ multi-start SLSQP (300 random cases, including negative excess returns)", worst_ms < 1e-7, f"max diff {worst_ms:.1e}")
    ok &= check("Min-variance enumeration solution ≤ SLSQP", worst_gmv < 1e-10, f"max diff {worst_gmv:.1e}")
    Sg = np.diag([0.01, 0.04, 0.09, 0.16]); mneg = np.full(4, -0.01)
    ok &= check("When max Sharpe ≤ 0 take min variance (otherwise everything goes into the highest-volatility asset)",
                np.abs(max_sharpe(mneg, Sg, 0.0) - gmv(Sg)).max() < 1e-12 and max_sharpe_raw(mneg, Sg, 0.0)[0][3] > 0.99,
                f"unhandled solution: {' / '.join(f'{x:.2f}' for x in max_sharpe_raw(mneg, Sg, 0.0)[0])}")
    # Batch version agrees with the single version
    MU = g.normal(0.05, 0.1, (50, 4)); SIG = stats.wishart.rvs(df=10, scale=np.eye(4) / 10, size=50, random_state=g)
    Wb = max_sharpe_batch(MU, SIG, 0.01)
    d = max(np.abs(Wb[i] - max_sharpe(MU[i], SIG[i], 0.01)).max() for i in range(50))
    ok &= check("Max-Sharpe batch version = single version", d < 1e-9, f"{d:.1e}")
    # Resampling: converges to the plug-in solution as T grows large
    X = Rm[:250]; mu, S = X.mean(0), np.cov(X, rowvar=False)
    wr = resampled_ms(mu, S, 10 ** 7, 0.043 / 252, 200, g); wp = max_sharpe(mu, S, 0.043 / 252)
    ok &= check("Resampling: equals the plug-in solution as T → ∞", np.abs(wr - wp).max() < 0.01, f"max diff {np.abs(wr - wp).max():.4f}")
    # Bayes–Stein: φ ∈ (0,1); lowers MSE when the means are equal
    mse_s, mse_b = 0.0, 0.0
    for _ in range(300):
        Z = g.multivariate_normal(np.full(4, 3e-4), S, 250)
        mb, ph = bayes_stein(Z)
        mse_s += ((Z.mean(0) - 3e-4) ** 2).sum(); mse_b += ((mb - 3e-4) ** 2).sum()
    ok &= check("Bayes–Stein: MSE below the sample mean when true means are equal", mse_b < mse_s, f"MSE ratio {mse_b / mse_s:.2f}")
    # Min CVaR: no worse than 1/N and 2000 random portfolios
    Xc = Rm[:400]; wc = min_cvar(Xc); cv = cvar_of(Xc @ wc)
    rnd = min(cvar_of(Xc @ v) for v in g.dirichlet(np.ones(4), 2000))
    ok &= check("Min CVaR (LP) ≤ min CVaR over random portfolios", cv <= rnd + 1e-12, f"{cv * 100:.4f}% vs {rnd * 100:.4f}%")
    # Mean–CVaR: constraint satisfied; equals min CVaR when r* is very low
    mu4 = Xc.mean(0); rs = float(np.quantile(mu4, 0.8))
    wt = min_cvar(Xc, mu=mu4, r_star=rs)
    wlow = min_cvar(Xc, mu=mu4, r_star=float(mu4.min()) - 1)
    ok &= check("Mean–CVaR: return constraint satisfied; equals min CVaR when the constraint is slack", wt @ mu4 >= rs - 1e-10
                and np.abs(wlow - wc).max() < 1e-8, f"μ′w − r* = {(wt @ mu4 - rs) * 252:.2e}")
    # Target-return min variance: constraint satisfied, equals GMV when r* is low
    a = 252; Sa = a * np.cov(Xc, rowvar=False); ma = a * mu4
    wtm = mv_target(ma, Sa, float(np.quantile(ma, 0.8)))
    ok &= check("Target-return min variance: constraint satisfied; equals min variance when r* is low", wtm @ ma >= np.quantile(ma, 0.8) - 1e-7
                and np.abs(mv_target(ma, Sa, -1.0) - gmv(Sa)).max() < 1e-9, f"{wtm @ ma - np.quantile(ma, 0.8):.1e}")
    # Robust: with κ = 0 it equals mean–variance utility
    d = np.abs(robust_mv(ma, Sa, np.diag(Sa) / 400, kappa=0.0) - mv_utility(ma, Sa)).max()
    ok &= check("Robust mean–variance: equals mean–variance utility when κ = 0", d < 1e-4, f"{d:.1e}")
    # Aggressive: volatility constraint satisfied
    wa = aggressive(ma, Sa)
    ok &= check("Aggressive: annualized volatility ≤ 18%", math.sqrt(wa @ Sa @ wa) <= SIG_STAR + 1e-6, f"{math.sqrt(wa @ Sa @ wa) * 100:.3f}%")
    # Turnover and cost accounting
    Rt = Rm[:300]; rft = np.zeros(300)
    o, _ = run_setting(Rt, rft, 250, 250, 21, g, 50, 2, 3)
    e = o["EW"]; man = 0.0; wc_ = np.ones(4) / 4
    for t in range(250, 271):
        gr = wc_ * (1 + Rt[t]); wc_ = gr / gr.sum()
    man = np.abs(np.ones(4) / 4 - wc_).sum()
    ok &= check("Turnover: the traded notional at the second 1/N rebalance matches a hand calculation", abs(e["trade"][21] - man) < 1e-12, f"{e['trade'][21]:.6f}")
    ok &= check("Cost: net return = gross return − c × traded notional", abs((net(e, 10) - e["gross"]).sum() + 10 / 1e4 * e["trade"].sum()) < 1e-14,
                f"initial positioning traded notional {e['trade'][0]:.3f}")
    return ok


# ============================================================ Output helpers
def f2(x, d=2, pctg=False):
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x * 100:.{d}f}" if pctg else f"{x:.{d}f}"


def weight_table(W, assets, extra):
    rows = []
    for c in CODES:
        v = W[c]
        rows.append([GRP[c], NAME[c]] + [f"{x * 100:.1f}" for x in v] + extra(c, v))
    return rows


def in_sample(label, R, assets, rf_d, rng, B_w, i_spy, i_tlt, rf_note):
    T, N = R.shape
    w(f"\n## {label}: in-sample (full sample, {T} days)\n")
    w(f"Full-sample estimation; expected return and volatility annualized as μ×252 and √(252·w′Σw) (arithmetic); rf = {rf_d * 252 * 100:.2f}% ({rf_note}). "
      f"Brackets give the 95% interval of the {assets[0]} weight under {B_w} stationary block bootstrap draws (expected block length {L_BOOT}).\n")
    W, inf_ = all_weights(R, rf_d, np.random.default_rng(SEED + 1), B_RES_IS, i_spy, i_tlt)
    idx = stationary_boot(T, B_w, L_BOOT, rng)
    WB = {c: [] for c in CODES}
    for b in range(B_w):
        Wb, _ = all_weights(R[idx[:, b]], rf_d, rng, 200, i_spy, i_tlt)
        for c in CODES:
            WB[c].append(Wb[c])
    WB = {c: np.array(v) for c, v in WB.items()}
    mu, S = R.mean(0) * 252, np.cov(R, rowvar=False) * 252

    def ex(c, v):
        lo, hi = np.percentile(WB[c][:, 0], [2.5, 97.5])
        return [f"[{lo * 100:.1f}, {hi * 100:.1f}]", f"{v @ mu * 100:.1f}", f"{math.sqrt(v @ S @ v) * 100:.1f}",
                f"{cvar_of(R @ v) * 100:.2f}"]
    w(md(weight_table(W, assets, ex), ["Group", "Strategy"] + [f"{a} %" for a in assets] +
         [f"{assets[0]} weight 95% CI", "E[r] % (ann.)", "σ % (ann.)", "Daily CVaR95 %"]))
    w(f"LW shrinkage intensity δ = {inf_['delta']:.3f}; Bayes–Stein shrinkage coefficient φ = {inf_['phi']:.3f} (φ = 1 means all assets' expected returns "
      f"are shrunk to the minimum-variance portfolio's mean).\n")
    wid = {c: float(np.percentile(WB[c], 97.5, axis=0).max() - 0) for c in CODES}
    width = {c: [float(np.percentile(WB[c][:, j], 97.5) - np.percentile(WB[c][:, j], 2.5)) for j in range(N)] for c in CODES}
    w("Width of the block-bootstrap 95% interval for each strategy's four weights (largest): " + "; ".join(f"{NAME[c]} {max(width[c]) * 100:.0f}" for c in CODES) + " (percentage points)\n")
    # Frontier
    gv = gmv(S / 252 * 252)
    r_lo, r_hi = float(gv @ mu), float(mu.max())
    grid = np.linspace(r_lo, r_hi, 7)
    rows = []
    for rs in grid:
        v1 = mv_target(mu, S, rs)
        v2 = min_cvar(R, mu=R.mean(0), r_star=rs / 252)
        rows.append([f"{rs * 100:.1f}", " / ".join(f"{x * 100:.0f}" for x in v1), f"{math.sqrt(v1 @ S @ v1) * 100:.1f}",
                     " / ".join(f"{x * 100:.0f}" for x in v2), f"{cvar_of(R @ v2) * 100:.2f}"])
    w(f"Efficient frontier (target return r* from the minimum-variance portfolio's {r_lo * 100:.1f}% to the single highest asset's {r_hi * 100:.1f}%; weight order "
      f"{' / '.join(assets)}, %)\n")
    w(md(rows, ["r* (ann. %)", "Mean–variance frontier weights", "σ %", "Mean–CVaR frontier weights", "Daily CVaR95 %"]))
    rows = []
    for s_ in (0.10, 0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.20, 0.25, 0.30):
        v = aggressive(mu, S, s_)
        rows.append([f"{s_ * 100:.0f}", " / ".join(f"{x * 100:.1f}" for x in v), f"{v @ mu * 100:.1f}",
                     f"{math.sqrt(v @ S @ v) * 100:.1f}"])
    w("Aggressive portfolio as the volatility cap σ* varies (sample estimates)\n")
    w(md(rows, ["σ* (%)", f"Weights {' / '.join(assets)} (%)", "E[r] %", "σ %"]))
    return dict(W={c: W[c].tolist() for c in CODES}, w0_ci={c: pct(WB[c][:, 0]) for c in CODES},
                width={c: width[c] for c in CODES}, delta=inf_["delta"], phi=inf_["phi"])


def oos_block(label, R, rf, dates, assets, start, win, rebal, rng, i_spy, i_tlt, B_res, tests=True):
    o, info = run_setting(R, rf, start, win, rebal, rng, B_res, i_spy, i_tlt)
    rfo = rf[start:]
    M = {c: metrics(net(o[c], MAIN_BP), rfo, o[c]) for c in CODES}
    res = dict(M=M, o=o, info=info)
    if not tests:
        res["p_hac"] = {c: sr_diff_test(net(o[c], MAIN_BP), net(o["EW"], MAIN_BP), rfo)[1] for c in CODES if c != "EW"}
        return res
    n = len(rfo)
    w(f"\n## {label}: out-of-sample ({'expanding' if win is None else f'{win}-day rolling'} window, rebalanced every {rebal} days; "
      f"{n} days, {dates[start].date()} → {dates[-1].date()}; {len(o['EW']['reb_pos'])} rebalances)\n")
    w(f"Net returns after deducting a one-way {MAIN_BP} bp per unit traded notional. Return = daily mean × 252 (arithmetic); geometric = compounded annualized; Sharpe = daily excess mean/sd × √252; "
      f"CER = 252 × (daily excess mean − γ/2 · daily variance), γ = {GAMMA:g}; ES95 is the daily historical value (average loss over the worst 5% of days); turnover is one-way (excluding initial positioning).\n")
    rows = []
    for c in CODES:
        m = M[c]
        rows.append([GRP[c], NAME[c], f2(m["w0_mean"], 1, True), f2(m["ret_a"], 1, True), f2(m["cagr"], 1, True),
                     f2(m["vol"], 1, True), f2(m["sharpe"], 2), f2(m["cer"], 1, True), f2(m["es95"], 2, True),
                     f2(m["mdd"], 1, True), f2(m["to_reb"], 1, True), f2(m["to_ann"], 0, True)])
    w(md(rows, ["Group", "Strategy", f"{assets[0]} mean weight %", "Return %", "Geometric %", "Vol %", "Sharpe", "CER %", "ES95 %",
                "Max drawdown %", "One-way turnover/rebalance %", "Annualized turnover %"]))
    idx = stationary_boot(n, B_TEST, L_BOOT, rng)
    ew = net(o["EW"], MAIN_BP)
    C = {c: compare(net(o[c], MAIN_BP), ew, rfo, idx) for c in CODES if c != "EW"}
    cs = list(C)
    h_sr = holm([C[c]["p_hac"] for c in cs]); h_cer = holm([C[c]["p_cer"] for c in cs])
    for k, c in enumerate(cs):
        C[c]["p_hac_holm"] = float(h_sr[k]); C[c]["p_cer_holm"] = float(h_cer[k])
    BE = {c: breakeven(o[c], o["EW"], rfo) for c in cs}
    rows = [[NAME[c], f"{C[c]['dsr']:+.2f}", pf(C[c]["p_hac"]), pf(C[c]["p_sr_boot"]), pf(C[c]["p_hac_holm"]),
             f"{C[c]['dcer'] * 100:+.1f} [{C[c]['dcer_ci'][0] * 100:+.1f}, {C[c]['dcer_ci'][1] * 100:+.1f}]",
             pf(C[c]["p_cer"]), pf(C[c]["p_cer_holm"]),
             "does not beat 1/N" if BE[c] is None else (">500" if not np.isfinite(BE[c]) else f"{BE[c]:.0f}")] for c in cs]
    w("Comparison against 1/N (net 10 bp; Sharpe difference by HAC and block bootstrap, two conventions; CER difference by block bootstrap; Holm correction over 14 comparisons)\n")
    w(md(rows, ["Strategy", "ΔSharpe", "p (HAC)", "p (bootstrap)", "p (Holm)", "ΔCER % (95% CI)", "p", "p (Holm)",
                "Break-even cost bp"]))
    ns_ = sum(i["neg_s"] for i in info)
    w(f"Rebalances where the attainable max Sharpe ≤ 0 and Max Sharpe (sample) falls back to min variance: {ns_}/{len(info)}; "
      f"mean LW shrinkage intensity δ {np.mean([i['delta'] for i in info]):.3f}, mean Bayes–Stein φ {np.mean([i['phi'] for i in info]):.3f}"
      f" (range {min(i['phi'] for i in info):.2f}–{max(i['phi'] for i in info):.2f}).\n")
    nsig = sum(C[c]["p_hac_holm"] < 0.05 for c in cs); nsig_raw = sum(C[c]["p_hac"] < 0.05 for c in cs)
    nsig_c = sum(C[c]["p_cer_holm"] < 0.05 for c in cs)
    w(f"Sharpe difference significant (5%): {nsig_raw}/14 uncorrected, {nsig}/14 after Holm; CER difference significant after Holm: {nsig_c}/14.\n")
    P = {}
    rows = []
    for a_, b_ in PAIRS:
        P[f"{a_}|{b_}"] = compare(net(o[a_], MAIN_BP), net(o[b_], MAIN_BP), rfo, idx)
        q = P[f"{a_}|{b_}"]
        rows.append([f"{NAME[a_]} − {NAME[b_]}", f"{q['dsr']:+.2f}", pf(q["p_hac"]),
                     f"{q['dcer'] * 100:+.1f} [{q['dcer_ci'][0] * 100:+.1f}, {q['dcer_ci'][1] * 100:+.1f}]", pf(q["p_cer"]),
                     f"{M[a_]['to_ann'] * 100:.0f} vs {M[b_]['to_ann'] * 100:.0f}"])
    w("Does handling estimation error help: paired comparisons (net 10 bp)\n")
    w(md(rows, ["Comparison", "ΔSharpe", "p (HAC)", "ΔCER % (95% CI)", "p", "Annualized turnover %"]))
    rows = []
    for c in CODES:
        rows.append([NAME[c]] + [f2(metrics(net(o[c], bp), rfo, o[c])["sharpe"], 2) for bp in COSTS_BP])
    w("Net Sharpe under different costs (bp per unit traded notional)\n")
    w(md(rows, ["Strategy"] + [f"{b} bp" for b in COSTS_BP]))
    res.update(C=C, P=P, BE={c: (None if v is None else (float(v) if np.isfinite(v) else "inf")) for c, v in BE.items()},
               cost_sr={c: {b: metrics(net(o[c], b), rfo, o[c])["sharpe"] for b in COSTS_BP} for c in CODES})
    return res


# ============================================================ Main program
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--outdir", default="out/23")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.data = a.data or find_main(); a.raw = a.raw or find_raw()
    if a.data is None:
        sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; please specify with --data.")
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(HERE, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    global B_TEST, B_W_MAIN, B_W_LONG, B_RES_OOS
    if a.quick:
        B_TEST, B_W_MAIN, B_W_LONG, B_RES_OOS = 500, 100, 60, 100
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT = {}

    d = pd.read_csv(a.data, parse_dates=["Date"]).set_index("Date").sort_index()
    A_M = ["IBIT", "GLD", "SPY", "TLT"]
    Rm = d[A_M].values; dm = d.index; rfm = np.full(len(Rm), RF_MAIN / 252)
    have_long = a.raw is not None
    if have_long:
        RL, rfl, rf_src = load_long(a.raw)
        A_L = ["BTC", "GLD", "SPY", "TLT"]
        Rl = RL[A_L].values; dl = RL.index
    w("# Supplementary portfolio-optimization analysis (script 23)\n")
    w(f"Seed {SEED}; γ = {GAMMA:g}; aggressive σ* = {SIG_STAR * 100:.0f}%; CVaR confidence level 95%; robust optimization κ = √χ²₄(0.95) = {KAPPA:.3f}; "
      f"resampling B = {B_RES_OOS} (per rebalance out-of-sample) / {B_RES_IS} (in-sample); test block bootstrap {B_TEST} draws, expected block length {L_BOOT}; "
      f"HAC lag {HAC_LAG}.\n")
    w("## 0. Programs and conventions (can go straight into §2.8)\n")
    w("Let μ̂, Σ̂ be the sample mean and sample covariance (ddof = 1) of daily returns in the estimation window, and Σ̂_LW the Ledoit–Wolf constant-correlation shrinkage covariance; "
      "Δ = {w ∈ ℝ⁴: w ≥ 0, 1′w = 1} (long-only, fully invested). All strategies are solved over Δ:\n")
    rows = [
        ["1/N", "w = 1/4", "—"],
        ["60/40", "w_SPY = 0.6, w_TLT = 0.4", "—"],
        ["Min variance", "min w′Σw, Σ = Σ̂ or Σ̂_LW", "—"],
        ["Risk parity", "w_i(Σ̂_LW w)_i equal for all i", "—"],
        ["Min CVaR95", "min ζ + (1/0.05T) Σ_t max(0, −r_t′w − ζ) (Rockafellar–Uryasev linear program)", "—"],
        ["Max Sharpe", "max (w′μ − r_f)/√(w′Σw); when the attainable max ≤ 0, take the min-variance portfolio", "μ̂, Σ̂ | resampling"],
        ["Mean–variance utility", "max w′μ − (γ/2) w′Σw, γ = 5", "μ̂, Σ̂ | μ_BS, Σ̂_LW"],
        ["Target return · min variance", "min w′Σw s.t. w′μ ≥ r*, r* = 1′μ̂/4 (the 1/N expected return)", "μ̂, Σ̂"],
        ["Target return · min CVaR", "min CVaR95(w) s.t. w′μ ≥ r*, r* as above", "μ̂"],
        ["Aggressive", "max w′μ s.t. √(252·w′Σw) ≤ 18%; infeasible → min variance", "μ̂, Σ̂ | μ_BS, Σ̂_LW"],
        ["Robust mean–variance", "max w′μ̂ − κ√(w′Θw) − (γ/2) w′Σ̂_LW w, Θ = diag(σ̂²_i)/T, κ = √χ²₄(0.95) = 3.08", "μ̂, Σ̂_LW"],
        ["Bayes–Stein mean", "μ_BS = (1−φ)μ̂ + φ μ₀1, μ₀ the min-variance portfolio's mean, φ = (N+2)/[(N+2) + T(μ̂−μ₀1)′S⁻¹(μ̂−μ₀1)]", "Jorion (1986)"],
        ["Resampling", "μ_b ~ N(μ̂, Σ̂/T), Σ_b ~ W(T−1, Σ̂/(T−1)), b = 1…B; w = B⁻¹Σ_b w_MSR(μ_b, Σ_b)", "Michaud (1998)"],
    ]
    w(md(rows, ["Strategy", "Program", "Inputs"]))
    w("- **Annualization**: expected return μ×252 (arithmetic), volatility √252·σ_daily; out-of-sample also reports the geometric return (Π(1+r_t))^{252/n} − 1; "
      "Sharpe = daily excess return mean/sd × √252; CER = 252 × (daily excess mean − γ/2 × daily variance).")
    w("- **Timing and turnover**: the weights on day t use only data up to t−1; weights drift with prices between rebalances; the traded notional on a rebalance day "
      "is TV = Σ_i |w_i,new − w_i,drift| (1 at initial positioning), one-way turnover = TV/2; cost = c × TV, deducted from that day's return; c ∈ {0, 5, 10, 25, 50} bp, "
      "with 10 bp for the main results. Break-even cost: the c that makes the strategy's net CER equal to 1/N's.")
    w(f"- **Risk-free rate**: main sample 4.3% (constant, consistent with the paper); long sample {rf_src if have_long else '—'}.")
    w("- **Long sample**: BTC-USD (00:00 UTC close, re-indexed to the SPY trading calendar) daily returns minus 0.25%/252 to mimic a spot ETF's annual fee; before 2024 no spot ETF existed, "
      "so this is a hypothetical investment. The closing time is 3–4 hours later than the US close, which pushes down Bitcoin's same-day correlation with stocks and makes the optimizer more willing to hold Bitcoin — this bias is conservative "
      "for the conclusion that 'the optimizer does not give Bitcoin weight'.\n")

    ok_self = selftest(Rm)
    if have_long and rfl.any():
        fs = sorted(glob.glob(os.path.join(a.raw, "IDX_IRX__*S.csv")))
        fl = sorted(glob.glob(os.path.join(a.raw, "IDX_IRX__*L.csv")))
        if fs and fl:
            s_ = pd.read_csv(fs[-1], parse_dates=["Date"]).drop_duplicates("Date").set_index("Date")["Close"]
            l_ = pd.read_csv(fl[-1], parse_dates=["Date"]).drop_duplicates("Date").set_index("Date")["Close"].reindex(s_.index)
            dmax = float((l_ - s_).abs().max())
            check("Long-sample ^IRX agrees day by day with the S snapshot over the overlap", dmax < 1e-6, f"max diff {dmax:.2e} ({len(s_)} days)")
        yr = pd.Series(rfl * 252 * 100, index=dl).groupby(dl.year).mean()
        w("Long-sample daily risk-free rate, annual average (%): " + "; ".join(f"{y} {v:.2f}" for y, v in yr.items()) + "\n")

    # ---------------- Reconciliation with existing results
    w("\n### Reconciliation with Table 9 and Table 10 of the paper (scripts 09, 17)\n")
    mu_a, S_a = Rm.mean(0) * 252, np.cov(Rm, rowvar=False) * 252
    ms = max_sharpe(mu_a, S_a, RF_MAIN); gv = gmv(S_a); ag = aggressive(mu_a, S_a, 0.18)
    check("Table 9: max Sharpe GLD 49.77, SPY 50.23", abs(ms[1] - 0.4977) < 6e-4 and abs(ms[2] - 0.5023) < 6e-4,
          " / ".join(f"{x * 100:.2f}" for x in ms))
    check("Table 9: min variance TLT 56.34, SPY 30.75, GLD 12.91", abs(gv[3] - 0.5634) < 6e-4 and abs(gv[2] - 0.3075) < 6e-4
          and abs(gv[1] - 0.1291) < 6e-4, " / ".join(f"{x * 100:.2f}" for x in gv))
    old9 = np.array([0.0179, 0.7583, 0.2237, 0.0])          # Table 9 aggressive (script 09 obtained it by a λ grid approximation)
    check("Aggressive σ* = 18%: exact solution's expected return ≥ Table 9's approximate solution, and the volatility constraint is exactly tight",
          ag @ mu_a >= old9 @ mu_a - 1e-9 and abs(math.sqrt(ag @ S_a @ ag) - 0.18) < 1e-6,
          f"exact solution {' / '.join(f'{x * 100:.2f}' for x in ag)} (E[r] {ag @ mu_a * 100:.2f}%, σ {math.sqrt(ag @ S_a @ ag) * 100:.2f}%)"
          f" vs Table 9 1.79 / 75.83 / 22.37 / 0.00 (E[r] {old9 @ mu_a * 100:.2f}%, σ {math.sqrt(old9 @ S_a @ old9) * 100:.2f}%)")
    w("   Note: Table 9's aggressive column comes from script 09's λ grid approximation (volatility 17.82%, constraint not tight) and should be replaced by the exact solution above.")
    o17, _ = run_setting(Rm, rfm, 250, 250, 21, np.random.default_rng(0), 50, 2, 3)
    ref = {"EW": 0.472, "GMV-S": 1.107, "GMV-LW": 1.110, "RP": 0.952, "MinCVaR": 0.950, "MSR-S": 1.508, "6040": 0.848}
    got = {c: metrics(o17[c]["gross"], rfm[250:], o17[c])["sharpe"] for c in ref}
    check("Table 10: the seven common strategies' gross Sharpe matches script 17 (±0.002)", all(abs(got[c] - ref[c]) < 0.0021 for c in ref),
          "; ".join(f"{c} {got[c]:.3f}" for c in ref))

    # ---------------- In-sample
    w("\n# 1. In-sample weights (the expansion of Table 9 of the paper)")
    OUT["is_main"] = in_sample("main sample", Rm, A_M, RF_MAIN / 252, rng, B_W_MAIN, 2, 3, "paper value")
    if have_long:
        OUT["is_long"] = in_sample("long sample", Rl, A_L, float(rfl.mean()), rng, B_W_LONG, 2, 3,
                                   "sample-period average" if rfl.any() else "using 0 for now")

    # ---------------- Out-of-sample main setting
    w("\n# 2. Out-of-sample main setting (250-day rolling window, rebalanced every 21 days)")
    OUT["oos_main"] = oos_block("main sample", Rm, rfm, dm, A_M, 250, 250, 21, rng, 2, 3, B_RES_OOS)
    if have_long:
        OUT["oos_long"] = oos_block("long sample", Rl, rfl, dl, A_L, 500, 250, 21, rng, 2, 3, B_RES_OOS)

    # ---------------- Robustness grid
    w("\n# 3. Robustness grid: estimation window × rebalancing frequency (net 10 bp)\n")
    grids = [("main sample", Rm, rfm, dm, 250, (125, 250, None))]
    if have_long:
        grids.append(("long sample", Rl, rfl, dl, 500, (250, 500, None)))
    rebs = (5, 21, 63) if not a.quick else (21, 63)
    OUT["grid"] = {}
    for lab, R_, rf_, d_, st, wins in grids:
        G = {}
        for win in wins:
            for rb in rebs:
                key = f"{'exp' if win is None else win}/{rb}"
                if lab == "main sample" and win == 250 and rb == 21:
                    G[key] = dict(M=OUT["oos_main"]["M"], p_hac={c: OUT["oos_main"]["C"][c]["p_hac"] for c in CODES if c != "EW"})
                elif lab == "long sample" and win == 250 and rb == 21:
                    G[key] = dict(M=OUT["oos_long"]["M"], p_hac={c: OUT["oos_long"]["C"][c]["p_hac"] for c in CODES if c != "EW"})
                else:
                    G[key] = oos_block(lab, R_, rf_, d_, A_M, st, win, rb, rng, 2, 3, B_RES_OOS, tests=False)
                print(f"   {lab} {key} done ({time.time() - t0:.0f}s)", flush=True)
        keys = list(G)
        w(f"## {lab} (out-of-sample start at day {st} = {d_[st].date()}, {len(R_) - st} days; columns are window/rebalance days, exp = expanding window)\n")
        for mname, lab2, fmt in (("sharpe", "Net Sharpe", lambda x: f"{x:.2f}"), ("cer", "CER %", lambda x: f"{x * 100:.1f}"),
                                 ("w0_mean", f"{'IBIT' if lab == 'main sample' else 'BTC'} mean weight %", lambda x: f"{x * 100:.1f}"),
                                 ("to_ann", "Annualized one-way turnover %", lambda x: f"{x * 100:.0f}")):
            rows = [[NAME[c]] + [fmt(G[k]["M"][c][mname]) for k in keys] for c in CODES]
            w(f"**{lab2}**\n")
            w(md(rows, ["Strategy"] + keys))
        cnt = {k: sum(G[k]["p_hac"][c] < 0.05 for c in CODES if c != "EW") for k in keys}
        best = {k: max(CODES, key=lambda c: G[k]["M"][c]["cer"]) for k in keys}
        w("Number of strategies with a Sharpe difference from 1/N significant (HAC, uncorrected 5%): " + "; ".join(f"{k}: {cnt[k]}/14" for k in keys))
        w("Strategy with the highest CER: " + "; ".join(f"{k}: {NAME[best[k]]}" for k in keys) + "\n")
        OUT["grid"][lab] = {k: dict(M=G[k]["M"], p_hac=G[k]["p_hac"]) for k in keys}

    # ---------------- Long-sample subperiods
    if have_long:
        w("\n# 4. Long-sample subperiods (main setting 250/21, net 10 bp)\n")
        o = OUT["oos_long"]["o"]; st = 500; dd = dl[st:]
        cuts = [("2016-09 → 2019-12", dd <= pd.Timestamp("2019-12-31")),
                ("2020-01 → 2022-12", (dd > pd.Timestamp("2019-12-31")) & (dd <= pd.Timestamp("2022-12-31"))),
                ("2023-01 → 2026-06", dd > pd.Timestamp("2022-12-31"))]
        rfo = rfl[st:]
        rows = []; SUB = {}
        for c in CODES:
            r = net(o[c], MAIN_BP); row = [NAME[c]]
            for nm_, m in cuts:
                ex = r[m] - rfo[m]
                sr = ex.mean() / ex.std(ddof=1) * math.sqrt(252); ce = cer_of(ex)
                row += [f"{sr:.2f}", f"{ce * 100:.1f}"]
                SUB.setdefault(c, {})[nm_] = dict(sharpe=float(sr), cer=float(ce))
            reb_d = dd[o[c]["reb_pos"]]; wb = o[c]["w_reb"][:, 0]
            row += [f"{wb[reb_d <= pd.Timestamp('2019-12-31')].mean() * 100:.1f} / "
                    f"{wb[(reb_d > pd.Timestamp('2019-12-31')) & (reb_d <= pd.Timestamp('2022-12-31'))].mean() * 100:.1f} / "
                    f"{wb[reb_d > pd.Timestamp('2022-12-31')].mean() * 100:.1f}"]
            rows.append(row)
        head = ["Strategy"] + sum([[f"{n_} Sharpe", f"{n_} CER %"] for n_, _ in cuts], []) + ["BTC mean weight % (three periods)"]
        w(md(rows, head))
        OUT["sub_long"] = SUB

    # ---------------- H4 summary
    w("\n# 5. Summary: how much weight the optimizer gives Bitcoin (H4)\n")
    rows = []
    for c in CODES:
        r_ = [GRP[c], NAME[c], f"{OUT['is_main']['W'][c][0] * 100:.1f} [{OUT['is_main']['w0_ci'][c][0] * 100:.1f}, "
                               f"{OUT['is_main']['w0_ci'][c][1] * 100:.1f}]",
              f"{OUT['oos_main']['M'][c]['w0_mean'] * 100:.1f}"]
        if have_long:
            r_ += [f"{OUT['is_long']['W'][c][0] * 100:.1f} [{OUT['is_long']['w0_ci'][c][0] * 100:.1f}, "
                   f"{OUT['is_long']['w0_ci'][c][1] * 100:.1f}]", f"{OUT['oos_long']['M'][c]['w0_mean'] * 100:.1f}",
                   f"{min(OUT['grid']['long sample'][k]['M'][c]['w0_mean'] for k in OUT['grid']['long sample']) * 100:.1f}–"
                   f"{max(OUT['grid']['long sample'][k]['M'][c]['w0_mean'] for k in OUT['grid']['long sample']) * 100:.1f}"]
        r_.insert(4, f"{min(OUT['grid']['main sample'][k]['M'][c]['w0_mean'] for k in OUT['grid']['main sample']) * 100:.1f}–"
                     f"{max(OUT['grid']['main sample'][k]['M'][c]['w0_mean'] for k in OUT['grid']['main sample']) * 100:.1f}")
        rows.append(r_)
    head = ["Group", "Strategy", "Main sample · in-sample IBIT % [CI]", "Main sample · out-of-sample mean %", "Main sample · grid range %"]
    if have_long:
        head += ["Long sample · in-sample BTC % [CI]", "Long sample · out-of-sample mean %", "Long sample · grid range %"]
    w(md(rows, head))

    # ---------------- Figures
    if have_long:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
            SER = [("GMV-LW", "#2a78d6"), ("MSR-S", "#eb6834"), ("MV-BS", "#1baf7a"), ("MSR-RS", "#eda100"), ("RMV", "#e87ba4")]
            EN = {"GMV-LW": "Min variance (LW)", "MSR-S": "Max Sharpe (sample)", "MV-BS": "Mean–variance (Bayes–Stein)",
                  "MSR-RS": "Max Sharpe (resampled)", "RMV": "Robust mean–variance", "EW": "1/N", "6040": "60/40"}
            plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                                 "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                                 "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                                 "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7})
            o = OUT["oos_long"]["o"]; st = 500; dd = dl[st:]
            fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.4), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
            ax = axs[0]
            ax.plot(dd, np.cumprod(1 + net(o["EW"], MAIN_BP)), color=INK2, lw=1.0, ls=(0, (4, 2)), label=EN["EW"])
            ax.plot(dd, np.cumprod(1 + net(o["6040"], MAIN_BP)), color="#9d9b95", lw=1.0, ls=(0, (1, 1.5)), label=EN["6040"])
            for c, col in SER:
                ax.plot(dd, np.cumprod(1 + net(o[c], MAIN_BP)), color=col, lw=1.1, label=EN[c])
            from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
            ax.set_yscale("log"); ax.set_ylabel("Wealth (log scale, net of 10 bp)")
            ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 5, 7, 10, 15]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
            ax.legend(loc="upper left", ncol=2)
            ax.set_title("a. Out-of-sample wealth, 250-day window, monthly rebalancing", loc="left", fontsize=8)
            ax = axs[1]
            ax.axhline(25, color=INK2, lw=0.9, ls=(0, (4, 2)))
            ax.text(dd[-1], 26.5, "1/N = 25%", color=INK2, fontsize=7, ha="right", va="bottom")
            for c, col in SER:
                rp = o[c]["reb_pos"]
                ax.step(dd[rp], o[c]["w_reb"][:, 0] * 100, where="post", color=col, lw=1.0)
            ax.set_ylabel("Bitcoin weight (%)"); ax.set_ylim(-2, 72)
            ax.set_title("b. Weight assigned to Bitcoin at each rebalancing", loc="left", fontsize=8)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(os.path.join(a.outdir, f"fig_portfolio_oos_long.{ext}"), dpi=300)
            plt.close(fig)
            w("\nFigure saved: fig_portfolio_oos_long.png / .pdf")
        except Exception as e:                                   # noqa: BLE001
            w(f"\n(plotting failed: {e})")

    # ---------------- Outputs
    w("\n## Self-test summary\n")
    for c in CHECKS:
        w("  " + c)
    nf = sum(c.startswith("[FAIL]") for c in CHECKS)
    w(f"\n{len(CHECKS)} checks, {nf} failures. Elapsed {time.time() - t0:.0f} seconds.")
    with open(os.path.join(a.outdir, "portfolio_robust.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items() if k not in ("o", "info")}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    meta = dict(seed=SEED, gamma=GAMMA, sig_star=SIG_STAR, kappa=KAPPA, btc_fee=BTC_FEE, costs_bp=COSTS_BP, main_bp=MAIN_BP,
                rf_main=RF_MAIN, rf_long=rf_src if have_long else None, names=NAME, groups=GRP)
    with open(os.path.join(a.outdir, "portfolio_robust.json"), "w", encoding="utf8") as fh:
        json.dump(clean(dict(meta=meta, **OUT)), fh, indent=1, ensure_ascii=False)
    # Out-of-sample net returns (main setting)
    for key, lab, dts, st in (("oos_main", "main", dm, 250), ("oos_long", "long", dl if have_long else None, 500)):
        if key in OUT:
            pd.DataFrame({c: net(OUT[key]["o"][c], MAIN_BP) for c in CODES}, index=dts[st:]).to_csv(
                os.path.join(a.outdir, f"oos_net_returns_{lab}.csv"))
    return 0 if (nf == 0 and ok_self) else 1


if __name__ == "__main__":
    sys.exit(main())