"""
22_safe_haven_dependence.py — Testing safe-haven properties with conditional dependence methods

VIX-regime correlations and average returns over the worst 20 days are only descriptive
statistics. Quantile regression, lower-tail dependence measures, copulas, DCC-GARCH, regime
switching, or time-varying-parameter methods answer: precisely when the market falls
extremely, are Bitcoin and stocks negatively correlated, or at least uncorrelated?

This script, for IBIT, GLD, TLT (main sample, 607 days) and for Bitcoin spot BTC-USD, GLD,
TLT (long sample from 2014-09), each against SPY:
  A. Quantile regression β(τ): the slope of the asset return's τ conditional quantile on
     SPY (Baur 2013), τ = 0.01…0.95; for τ = 0.05, 0.10 also do a block bootstrap interval.
  B. Non-parametric lower-tail dependence λ_L(u) = P(U ≤ u, V ≤ u)/u (u = 5%, 10%),
     benchmarked against independence (= u) and the same-rank-correlation Gaussian copula;
     the upper tail λ_U(u) to check symmetry.
  C. Copula-GARCH: GARCH(1,1)-t marginals → probability integral transform → Gaussian,
     Student-t, Clayton, survival Gumbel, Frank, rotated Clayton (negative dependence) —
     six families by MLE, selected by AIC; report the lower-tail dependence coefficient λ_L
     with a block bootstrap interval.
  D. Cross-quantilogram (Han et al. 2016) contemporaneous values at τ = 1%…50% and intervals.
  E. Exceedance correlation (Longin & Solnik 2001; Ang & Chen 2002): the correlation when
     both fall below −c standard deviations, benchmarked against a bivariate normal with the
     same correlation; upper/lower tail symmetry Wald test (following Hong, Tu & Zhou 2007,
     with a bootstrap covariance). Also report the correlation conditional on the SPY lower
     tail and its normal benchmark (truncation bias, Boyer, Gibson & Loretan 1999).
  F. DCC-GARCH(1,1) (Engle 2002, two-step): dynamic correlation path; regression of ρ_t on
     SPY 10%/5%/1% lower-tail-day dummies (the safe-haven criterion of Ratner & Chiu 2013;
     Bouri et al. 2017), HAC standard errors.
  G. Regime switching: two-state bivariate Gaussian hidden Markov model (EM), states ordered
     by SPY variance, report each state's ρ, β, duration, with a parameter-bootstrap
     interval; cross-checked with statsmodels MarkovRegression (β and variance switching).
  H. Time-varying parameters: Kalman-filter random-walk β_t (measurement variance scaled by
     the GARCH conditional variance), mean of the smoothed β_t on lower-tail days; likelihood
     ratio test for constant β.
  I. Summary: overall table of "the direction of dependence during extreme market declines".
  J. Figure: main-sample DCC correlation and TVP β paths.

Long sample: BTC-USD's daily bar closes at 00:00 UTC, 3–4 hours after the US close, which
biases same-day dependence toward zero (script 20: in the main sample, BTC-USD's ρ with SPY
from 16:00 prices is 0.39 vs 0.41). This bias is in the direction [favorable] to the
safe-haven conclusion, so if the long sample still shows positive dependence, the conclusion
is only more robust; non-parametric measures are also re-checked with non-overlapping
two-day returns.

Dependencies: numpy, pandas, scipy, statsmodels, arch, matplotlib. Can be run standalone.
Usage:
    python 22_safe_haven_dependence.py
    python 22_safe_haven_dependence.py --data data/processed/returns__20260921.csv --raw data/raw --outdir out/22
    python 22_safe_haven_dependence.py --quick        # halve the bootstrap draws
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats, optimize, special

try:
    import statsmodels.api as sm
    from arch.univariate import ConstantMean, GARCH, StudentsT
except ImportError as e:
    sys.exit(f"Missing dependencies: {e}. Please run python -m pip install statsmodels arch")
warnings.filterwarnings("ignore")                       # placed after imports: statsmodels/arch reset some filters when imported

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260927
B_QR, B_LAM, B_COP, B_CQ, B_EXC, B_HMM = 500, 2000, 500, 2000, 1000, 200
L_BOOT = 5
HAC_DCC = 20
LINES: list[str] = []
CHECKS: list[str] = []
KEY: dict = {}


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
    return ok


def fci(pt, lo, hi, d=2):
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"
    return f"{f(pt)} [{f(lo)}, {f(hi)}]"


def pf(p):
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def pct(a, q=(2.5, 97.5)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, q)] if len(a) else [np.nan, np.nan]


def rank_u(x):
    return stats.rankdata(x) / (len(x) + 1.0)


# ====================================================================== A. Quantile regression
def qr_beta(y, x, tau):
    res = sm.QuantReg(y, sm.add_constant(x)).fit(q=tau, max_iter=5000, p_tol=1e-8)
    return float(res.params[1]), float(res.bse[1])


# ====================================================================== B. Non-parametric tail dependence
def lam_emp(x, y, u, lower=True):
    U, V = rank_u(x), rank_u(y)
    if lower:
        return float(np.mean((U <= u) & (V <= u)) / u)
    return float(np.mean((U > 1 - u) & (V > 1 - u)) / u)


def lam_gauss_bench(rho_s, u, n_sim=2_000_000, seed=1):
    """Benchmark λ(u) at finite u for the Gaussian copula with the same sample Spearman
    rank correlation."""
    r = 2 * math.sin(math.pi * rho_s / 6)          # Spearman → Pearson (Gaussian copula)
    g = np.random.default_rng(seed)
    z1 = g.standard_normal(n_sim); z2 = r * z1 + math.sqrt(1 - r * r) * g.standard_normal(n_sim)
    q = stats.norm.ppf(u)
    return float(np.mean((z1 <= q) & (z2 <= q)) / u)


# ====================================================================== C. Copulas
def _clip(u):
    return np.clip(u, 1e-10, 1 - 1e-10)


def ll_gauss(p, u, v):
    r = math.tanh(p[0]); x, y = stats.norm.ppf(_clip(u)), stats.norm.ppf(_clip(v))
    return float(np.sum(-0.5 * math.log(1 - r * r) - (r * r * (x * x + y * y) - 2 * r * x * y) / (2 * (1 - r * r))))


def ll_t(p, u, v):
    r = math.tanh(p[0]); nu = 2.05 + math.exp(p[1])
    x, y = stats.t.ppf(_clip(u), nu), stats.t.ppf(_clip(v), nu)
    c = (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
         - 0.5 * math.log(1 - r * r))
    q = (x * x + y * y - 2 * r * x * y) / (nu * (1 - r * r))
    return float(np.sum(c - (nu + 2) / 2 * np.log1p(q) + (nu + 1) / 2 * (np.log1p(x * x / nu) + np.log1p(y * y / nu))))


def _ll_clayton_raw(th, u, v):
    u, v = _clip(u), _clip(v)
    return np.sum(np.log1p(th) - (1 + th) * (np.log(u) + np.log(v))
                  - (2 + 1 / th) * np.log(u ** -th + v ** -th - 1))


def ll_clayton(p, u, v):
    return float(_ll_clayton_raw(1e-4 + math.exp(p[0]), u, v))


def ll_clayton90(p, u, v):
    return float(_ll_clayton_raw(1e-4 + math.exp(p[0]), 1 - u, v))


def _ll_gumbel_raw(th, u, v):
    u, v = _clip(u), _clip(v)
    x, y = -np.log(u), -np.log(v)
    A = (x ** th + y ** th) ** (1 / th)
    return np.sum(-A - np.log(u) - np.log(v) + (th - 1) * (np.log(x) + np.log(y)) + (1 - 2 * th) * np.log(A)
                  + np.log(A + th - 1))


def ll_sgumbel(p, u, v):
    return float(_ll_gumbel_raw(1 + math.exp(p[0]), 1 - u, 1 - v))


def ll_frank(p, u, v):
    th = p[0]
    if abs(th) < 1e-6:
        return 0.0
    u, v = _clip(u), _clip(v)
    num = math.log(abs(th)) + np.log(abs(1 - math.exp(-th))) - th * (u + v)
    den = 2 * np.log(np.abs((1 - math.exp(-th)) - (1 - np.exp(-th * u)) * (1 - np.exp(-th * v))))
    return float(np.sum(num - den))


COPULAS = {
    "Gaussian": (ll_gauss, [0.3], lambda p: 0.0),
    "Student-t": (ll_t, [0.3, math.log(6.0)],
                  lambda p: float(2 * stats.t.cdf(-math.sqrt((2.05 + math.exp(p[1]) + 1) * (1 - math.tanh(p[0]))
                                                           / (1 + math.tanh(p[0]))), 2.05 + math.exp(p[1]) + 1))),
    "Clayton": (ll_clayton, [math.log(0.5)], lambda p: float(2 ** (-1 / (1e-4 + math.exp(p[0]))))),
    "Survival Gumbel": (ll_sgumbel, [math.log(0.3)], lambda p: float(2 - 2 ** (1 / (1 + math.exp(p[0]))))),
    "Frank": (ll_frank, [2.0], lambda p: 0.0),
    "Clayton 90° (negative)": (ll_clayton90, [math.log(0.2)], lambda p: 0.0),
}
NPAR = {"Gaussian": 1, "Student-t": 2, "Clayton": 1, "Survival Gumbel": 1, "Frank": 1, "Clayton 90° (negative)": 1}


def _clayton_C(a, b, th):
    return (a ** -th + b ** -th - 1) ** (-1 / th)


def copula_lam_u(name, par, u=0.05):
    """The λ_L(u) = C(u,u)/u = P(V ≤ u | U ≤ u) implied by the fitted copula at finite
    quantile u; equals u under independence."""
    if name == "Gaussian":
        r = math.tanh(par[0]); q = stats.norm.ppf(u)
        C = stats.multivariate_normal.cdf([q, q], cov=[[1, r], [r, 1]])
    elif name == "Student-t":
        r = math.tanh(par[0]); nu = 2.05 + math.exp(par[1]); q = stats.t.ppf(u, nu)
        C = stats.multivariate_t.cdf([q, q], shape=[[1, r], [r, 1]], df=nu, random_state=0)
    elif name == "Clayton":
        C = _clayton_C(u, u, 1e-4 + math.exp(par[0]))
    elif name == "Survival Gumbel":
        th = 1 + math.exp(par[0]); a = 1 - u
        C = 2 * u - 1 + math.exp(-(2 * (-math.log(a)) ** th) ** (1 / th))
    elif name == "Frank":
        th = par[0]
        C = u * u if abs(th) < 1e-6 else -1 / th * math.log1p((math.exp(-th * u) - 1) ** 2 / (math.exp(-th) - 1))
    else:                                                    # Clayton 90°: (1−U, V) ~ Clayton
        C = u - _clayton_C(1 - u, u, 1e-4 + math.exp(par[0]))
    return float(C / u)


def fit_copula(name, u, v, x0=None):
    f, p0, lam = COPULAS[name]
    obj = lambda p: -f(p, u, v)
    best = None
    for start in ([x0] if x0 is not None else []) + [p0]:
        try:
            r = optimize.minimize(obj, np.array(start, float), method="Nelder-Mead",
                                  options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:                                   # noqa: BLE001
            pass
    ll = -best.fun
    return dict(par=best.x.tolist(), ll=float(ll), aic=float(2 * NPAR[name] - 2 * ll), lam_L=lam(best.x))


def natural_params(name, par):
    if name == "Gaussian":
        return f"ρ = {math.tanh(par[0]):.3f}"
    if name == "Student-t":
        return f"ρ = {math.tanh(par[0]):.3f}, ν = {2.05 + math.exp(par[1]):.2f}"
    if name.startswith("Clayton"):
        return f"θ = {1e-4 + math.exp(par[0]):.3f}"
    if name == "Survival Gumbel":
        return f"θ = {1 + math.exp(par[0]):.3f}"
    return f"θ = {par[0]:.3f}"


def garch_pit(r):
    am = ConstantMean(r * 100)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = StudentsT()
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 2000})
    z = np.asarray(res.std_resid, float)
    nu = float(res.params["nu"])
    u = stats.t.cdf(z * math.sqrt(nu / (nu - 2)), nu)
    return u, z, np.asarray(res.conditional_volatility, float) / 100


# ====================================================================== D. Cross-quantilogram
def cq(x, y, tau):
    """Contemporaneous cross-quantilogram; x, y may be n×B."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    qx = np.quantile(x, tau, axis=0); qy = np.quantile(y, tau, axis=0)
    px = (x < qx).astype(float) - tau; py = (y < qy).astype(float) - tau
    return (px * py).sum(0) / np.sqrt((px ** 2).sum(0) * (py ** 2).sum(0))


# ====================================================================== E. Exceedance correlation
def exc_corr(zx, zy, c, side=-1, min_n=10):
    m = (zx < -c) & (zy < -c) if side < 0 else (zx > c) & (zy > c)
    if m.sum() < min_n:
        return np.nan, int(m.sum())
    return float(np.corrcoef(zx[m], zy[m])[0, 1]), int(m.sum())


def exc_bench(rho, cs, n_sim=2_000_000, seed=2):
    g = np.random.default_rng(seed)
    z1 = g.standard_normal(n_sim); z2 = rho * z1 + math.sqrt(1 - rho * rho) * g.standard_normal(n_sim)
    return {c: exc_corr(z1, z2, c, -1, 50)[0] for c in cs}


def cond_corr_bench(rho, p, n_sim=2_000_000, seed=3):
    g = np.random.default_rng(seed)
    z1 = g.standard_normal(n_sim); z2 = rho * z1 + math.sqrt(1 - rho * rho) * g.standard_normal(n_sim)
    m = z2 <= np.quantile(z2, p)
    return float(np.corrcoef(z1[m], z2[m])[0, 1])


# ====================================================================== F. DCC
def dcc_nll(p, Z, Qbar):
    a, b = p
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    n = len(Z); Q = Qbar.copy(); ll = 0.0
    q11 = np.empty(n); q22 = np.empty(n); q12 = np.empty(n)
    for t in range(n):
        if t > 0:
            z = Z[t - 1]
            Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
        q11[t], q22[t], q12[t] = Q[0, 0], Q[1, 1], Q[0, 1]
    rho = q12 / np.sqrt(q11 * q22)
    x, y = Z[:, 0], Z[:, 1]
    ll = np.sum(np.log(1 - rho ** 2) + (x * x + y * y - 2 * rho * x * y) / (1 - rho ** 2) - (x * x + y * y))
    return 0.5 * ll


def dcc_path(p, Z, Qbar):
    a, b = p; n = len(Z); Q = Qbar.copy(); rho = np.empty(n)
    for t in range(n):
        if t > 0:
            z = Z[t - 1]; Q = (1 - a - b) * Qbar + a * np.outer(z, z) + b * Q
        rho[t] = Q[0, 1] / math.sqrt(Q[0, 0] * Q[1, 1])
    return rho


def dcc_fit(Z):
    Qbar = np.corrcoef(Z.T)
    best = None
    for a0, b0 in ((0.03, 0.95), (0.05, 0.90), (0.10, 0.80), (0.02, 0.97)):
        r = optimize.minimize(dcc_nll, [a0, b0], args=(Z, Qbar), method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    return dict(a=float(a), b=float(b), rho=dcc_path(best.x, Z, Qbar), nll=float(best.fun), Qbar=Qbar)


def nw_ols(y, X, lag):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b; n = len(y)
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lag + 1)) * (A + A.T)
    return b, XtXi @ S @ XtXi


# ====================================================================== G. Bivariate HMM
def mvn_logpdf(X, mu, S):
    d = X - mu
    Si = np.linalg.inv(S); ld = np.linalg.slogdet(S)[1]
    return -0.5 * (np.einsum("ij,jk,ik->i", d, Si, d) + ld + 2 * math.log(2 * math.pi))


def _fwd_bwd(B, P, pi):
    """Two-state forward–backward (scalar recursion, an order of magnitude faster than
    row-wise numpy)."""
    n = len(B)
    b0, b1 = B[:, 0].tolist(), B[:, 1].tolist()
    p00, p01, p10, p11 = float(P[0, 0]), float(P[0, 1]), float(P[1, 0]), float(P[1, 1])
    a0 = [0.0] * n; a1 = [0.0] * n; c = [0.0] * n
    x0, x1 = pi[0] * b0[0], pi[1] * b1[0]; s = x0 + x1
    a0[0], a1[0], c[0] = x0 / s, x1 / s, s
    for t in range(1, n):
        x0 = (a0[t - 1] * p00 + a1[t - 1] * p10) * b0[t]
        x1 = (a0[t - 1] * p01 + a1[t - 1] * p11) * b1[t]
        s = x0 + x1
        a0[t], a1[t], c[t] = x0 / s, x1 / s, s
    e0 = [1.0] * n; e1 = [1.0] * n
    for t in range(n - 2, -1, -1):
        y0, y1 = b0[t + 1] * e0[t + 1], b1[t + 1] * e1[t + 1]
        e0[t] = (p00 * y0 + p01 * y1) / c[t + 1]
        e1[t] = (p10 * y0 + p11 * y1) / c[t + 1]
    al = np.column_stack([a0, a1]); be = np.column_stack([e0, e1]); c = np.asarray(c)
    return al, be, c


def hmm_fit(X, n_start=6, iters=400, tol=1e-8, rng=None):
    n = len(X); best = None
    rng = rng or np.random.default_rng(0)
    for s in range(n_start):
        # Initial values: split into two groups by the magnitude of SPY
        thr = np.quantile(np.abs(X[:, 1]), 0.6 + 0.3 * rng.random() if s else 0.75)
        g = (np.abs(X[:, 1]) > thr).astype(int)
        mu = np.array([X[g == k].mean(0) for k in range(2)])
        S = np.array([np.cov(X[g == k].T) + 1e-10 * np.eye(2) for k in range(2)])
        P = np.array([[0.95, 0.05], [0.10, 0.90]]); pi = np.array([0.5, 0.5])
        ll_old = -np.inf
        for it in range(iters):
            L = np.column_stack([mvn_logpdf(X, mu[k], S[k]) for k in range(2)])
            Lm = L.max(1, keepdims=True)
            B = np.exp(L - Lm)                                  # row-wise scaling to avoid overflow/underflow
            al, be, c = _fwd_bwd(B, P, pi)
            gam = al * be; gam /= gam.sum(1, keepdims=True)
            xi = (al[:-1, :, None] * P[None] * (B[1:] * be[1:])[:, None, :] / c[1:, None, None]).sum(0)
            ll = float(np.log(c).sum() + Lm.sum())
            pi = gam[0]
            P = xi / xi.sum(1, keepdims=True)
            for k in range(2):
                wk = gam[:, k]; sw = wk.sum()
                mu[k] = (wk[:, None] * X).sum(0) / sw
                d = X - mu[k]
                S[k] = (wk[:, None, None] * np.einsum("ij,ik->ijk", d, d)).sum(0) / sw + 1e-12 * np.eye(2)
            if abs(ll - ll_old) < tol * abs(ll):
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = dict(ll=ll, mu=mu.copy(), S=S.copy(), P=P.copy(), pi=pi.copy(), gam=gam.copy())
    # Order states: the one with smaller SPY variance is state 0 (calm), the larger is state 1 (turbulent)
    if best["S"][0][1, 1] > best["S"][1][1, 1]:
        o = [1, 0]
        best = dict(ll=best["ll"], mu=best["mu"][o], S=best["S"][o], P=best["P"][np.ix_(o, o)], pi=best["pi"][o],
                    gam=best["gam"][:, o])
    return best


def hmm_summary(h):
    out = []
    for k in range(2):
        S = h["S"][k]
        out.append(dict(rho=float(S[0, 1] / math.sqrt(S[0, 0] * S[1, 1])), beta=float(S[0, 1] / S[1, 1]),
                        vol_asset=float(math.sqrt(S[0, 0] * 252) * 100), vol_spy=float(math.sqrt(S[1, 1] * 252) * 100),
                        dur=float(1 / (1 - h["P"][k, k])), share=float(h["gam"][:, k].mean())))
    return out


def hmm_sim(h, n, rng):
    s = np.empty(n, int); X = np.empty((n, 2))
    s[0] = rng.choice(2, p=h["pi"] / h["pi"].sum())
    for t in range(1, n):
        s[t] = rng.choice(2, p=h["P"][s[t - 1]])
    for k in range(2):
        m = s == k
        X[m] = rng.multivariate_normal(h["mu"][k], h["S"][k], m.sum())
    return X


# ====================================================================== H. TVP β (Kalman)
def tvp_nll(p, y, x, h):
    alpha, ls, lq = p
    s2, q = math.exp(ls), math.exp(lq)
    b, P = 0.0, 10.0; nll = 0.0
    for t in range(len(y)):
        P = P + q
        e = y[t] - alpha - b * x[t]; F = x[t] ** 2 * P + s2 * h[t]
        K = P * x[t] / F
        b = b + K * e; P = P - K * x[t] * P
        if t >= 20:                                   # first 20 days as a burn-in for the diffuse initialization
            nll += 0.5 * (math.log(2 * math.pi * F) + e * e / F)
    return nll


def tvp_fit(y, x, h):
    b0 = np.polyfit(x, y, 1)
    best = None
    for lq in (-8.0, -6.0, -4.0, -2.0):
        r = optimize.minimize(tvp_nll, [b0[1], math.log(np.var(y) / np.mean(h)), lq], args=(y, x, h),
                              method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    alpha, ls, lq = best.x
    s2, q = math.exp(ls), math.exp(lq)
    n = len(y); bf = np.empty(n); Pf = np.empty(n); bp = np.empty(n); Pp = np.empty(n)
    b, P = 0.0, 10.0
    for t in range(n):
        Pp[t] = P + q; bp[t] = b
        e = y[t] - alpha - b * x[t]; F = x[t] ** 2 * Pp[t] + s2 * h[t]
        K = Pp[t] * x[t] / F
        b = b + K * e; P = Pp[t] - K * x[t] * Pp[t]
        bf[t], Pf[t] = b, P
    bs, Ps = bf.copy(), Pf.copy()
    for t in range(n - 2, -1, -1):
        J = Pf[t] / Pp[t + 1]
        bs[t] = bf[t] + J * (bs[t + 1] - bp[t + 1])
        Ps[t] = Pf[t] + J * J * (Ps[t + 1] - Pp[t + 1])
    # Restricted model with constant β (q → 0)
    r0 = optimize.minimize(lambda p: tvp_nll([p[0], p[1], -30.0], y, x, h), [alpha, ls], method="Nelder-Mead",
                           options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 2000})
    LR = max(0.0, 2 * (r0.fun - best.fun))
    return dict(alpha=float(alpha), s2=float(s2), q=float(q), beta_s=bs, P_s=Ps, LR=float(LR),
                p_const=float(0.5 * stats.chi2.sf(LR, 1)))


# ====================================================================== Data
def find_main():
    c = []
    for base in (HERE, os.getcwd()):
        for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv")):
            if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p)):
                c.append(os.path.abspath(p))
    return sorted(set(c), key=os.path.basename)[-1] if c else None


def find_raw_L():
    for base in (HERE, os.getcwd()):
        d = os.path.join(base, "data", "raw")
        if glob.glob(os.path.join(d, "BTC-USD__*L.csv")):
            return d
    return None


def load_long(raw):
    S = {}
    for t, f in (("BTC", "BTC-USD"), ("GLD", "GLD"), ("TLT", "TLT"), ("SPY", "SPY"), ("VIX", "IDX_VIX")):
        p = sorted(glob.glob(os.path.join(raw, f"{f}__*L.csv")))[-1]
        d = pd.read_csv(p, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
        S[t] = d["AdjClose"] if t != "VIX" else d["Close"]
    cal = S["SPY"].index
    P = pd.DataFrame({k: v.reindex(cal) for k, v in S.items() if k != "VIX"})
    first = S["BTC"].index.min()
    P = P[P.index >= first]
    R = P.pct_change().dropna()
    R = R[R.index <= pd.Timestamp("2026-06-15")]
    R["VIX"] = S["VIX"].reindex(R.index).values
    return R


# ====================================================================== Self-test
def selftest():
    ok = True
    rng = np.random.default_rng(0)
    # Copula: simulate Clayton with θ = 2 and recover it; λ_L = 2^{-1/2}
    n = 4000; th = 2.0
    u = rng.random(n); w_ = rng.random(n)
    v = ((w_ ** (-th / (1 + th)) - 1) * u ** (-th) + 1) ** (-1 / th)
    f = fit_copula("Clayton", u, v)
    ok &= check("Copula: Clayton θ = 2 is recoverable", abs((1e-4 + math.exp(f["par"][0])) - th) < 0.25,
                f"θ̂ = {1e-4 + math.exp(f['par'][0]):.3f}, λ_L = {f['lam_L']:.3f} (true 0.707)")
    # t copula is recoverable
    r_, nu_ = 0.5, 5.0
    z = rng.multivariate_normal([0, 0], [[1, r_], [r_, 1]], n) / np.sqrt(rng.chisquare(nu_, n) / nu_)[:, None]
    ft = fit_copula("Student-t", stats.t.cdf(z[:, 0], nu_), stats.t.cdf(z[:, 1], nu_))
    ok &= check("Copula: t(ρ = 0.5, ν = 5) is recoverable", abs(math.tanh(ft["par"][0]) - r_) < 0.05
                and abs(2.05 + math.exp(ft["par"][1]) - nu_) < 2.0,
                f"ρ̂ = {math.tanh(ft['par'][0]):.3f}, ν̂ = {2.05 + math.exp(ft['par'][1]):.2f}")
    # Survival Gumbel density integrates to 1 (numerically)
    g = np.linspace(0.0005, 0.9995, 400)
    U, V = np.meshgrid(g, g)
    dens = np.exp(np.array([[_ll_gumbel_raw(1.7, 1 - a, 1 - b) for a, b in zip(ru, rv)] for ru, rv in zip(U, V)]))
    integ = dens.mean()
    ok &= check("Copula: Gumbel density integrates to ≈ 1", abs(integ - 1) < 0.03, f"{integ:.3f}")
    # DCC parameters are recoverable
    n = 3000; a0, b0 = 0.05, 0.90; Qb = np.array([[1, 0.3], [0.3, 1]]); Q = Qb.copy(); Z = np.empty((n, 2))
    for t in range(n):
        if t:
            Q = (1 - a0 - b0) * Qb + a0 * np.outer(Z[t - 1], Z[t - 1]) + b0 * Q
        d = np.sqrt(np.diag(Q)); R_ = Q / np.outer(d, d)
        Z[t] = np.linalg.cholesky(R_) @ rng.standard_normal(2)
    fd = dcc_fit(Z)
    ok &= check("DCC: a = 0.05, b = 0.90 are recoverable", abs(fd["a"] - a0) < 0.03 and abs(fd["b"] - b0) < 0.06,
                f"â = {fd['a']:.3f}, b̂ = {fd['b']:.3f}")
    # HMM: two states are recoverable
    h0 = dict(pi=np.array([0.5, 0.5]), P=np.array([[0.97, 0.03], [0.1, 0.9]]),
              mu=np.zeros((2, 2)), S=np.array([[[1, 0.1], [0.1, 1]], [[4, 3.2], [3.2, 4]]]) * 1e-4)
    Xs = hmm_sim(h0, 3000, rng)
    hf = hmm_summary(hmm_fit(Xs, rng=rng))
    ok &= check("HMM: two-state correlations (0.1 / 0.8) are recoverable", abs(hf[0]["rho"] - 0.1) < 0.1 and abs(hf[1]["rho"] - 0.8) < 0.08,
                f"{hf[0]['rho']:.3f} / {hf[1]['rho']:.3f}")
    # TVP: state variance and β path are recoverable; LR does not reject under constant β
    g2 = np.random.default_rng(1)
    n = 1500; x = g2.standard_normal(n) * 0.01; bt = 1 + np.cumsum(g2.normal(0, 0.03, n))
    y = 0.0 + bt * x + g2.normal(0, 0.01, n)
    tv = tvp_fit(y, x, np.ones(n))
    cc = np.corrcoef(tv["beta_s"][100:], bt[100:])[0, 1]
    ok &= check("TVP: q and the β_t path are recoverable", 0.5 < tv["q"] / 0.03 ** 2 < 2 and cc > 0.85,
                f"q̂/q = {tv['q'] / 0.03 ** 2:.2f}, corr = {cc:.3f}")
    y0 = 1.0 * x + g2.normal(0, 0.01, n)
    t0 = tvp_fit(y0, x, np.ones(n))
    ok &= check("TVP: does not reject under constant β", t0["p_const"] > 0.05, f"LR = {t0['LR']:.2f}, p = {t0['p_const']:.3f}")
    # Copula tail probability at finite u: Clayton analytic form agrees with simulation;
    # Gaussian benchmark agrees with the simulation in Section B
    th = 2.0; u = rng.random(200000); w_ = rng.random(200000)
    v = ((w_ ** (-th / (1 + th)) - 1) * u ** (-th) + 1) ** (-1 / th)
    emp = float(np.mean((u <= 0.05) & (v <= 0.05)) / 0.05)
    ana = copula_lam_u("Clayton", [math.log(th - 1e-4)])
    gb = copula_lam_u("Gaussian", [math.atanh(0.4)]); gs = lam_gauss_bench(6 / math.pi * math.asin(0.2), 0.05)
    ok &= check("Copula: analytic tail probability at finite u is correct", abs(emp - ana) < 0.02 and abs(gb - gs) < 0.01,
                f"Clayton {ana:.3f} vs simulation {emp:.3f}; Gaussian {gb:.3f} vs {gs:.3f}")
    # Cross-quantilogram: ≈ 0 for independent series
    xx, yy = rng.standard_normal(20000), rng.standard_normal(20000)
    ok &= check("Cross-quantilogram: ≈ 0 for independent series", abs(cq(xx, yy, 0.05)) < 0.03, f"{float(cq(xx, yy, 0.05)):.4f}")
    return ok


# ====================================================================== Full analysis for one sample
def analyze(label, R, assets, rng, B_scale=1.0, two_day=False):
    out = {}
    spy = R["SPY"].values
    n = len(spy)
    w(f"\n# {label} ({n} trading days, {R.index[0].date()} → {R.index[-1].date()})\n")
    IDX = stationary_boot(n, int(B_LAM * B_scale), L_BOOT, rng)
    q05, q10, q01 = np.quantile(spy, [0.05, 0.10, 0.01])
    tail5 = spy <= q05
    for a in assets:
        y = R[a].values
        o = {}
        w(f"\n## {label}: {a} vs SPY\n")
        # ---------------- A
        taus = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        qr = {}
        for tau in taus:
            b, se = qr_beta(y, spy, tau)
            qr[tau] = dict(beta=b, lo=b - 1.96 * se, hi=b + 1.96 * se)
        ols = float(np.polyfit(spy, y, 1)[0])
        Bq = int(B_QR * B_scale)
        idxq = stationary_boot(n, Bq, L_BOOT, rng)
        for tau in (0.05, 0.10):
            bs = [qr_beta(y[i], spy[i], tau)[0] for i in idxq.T]
            qr[tau]["boot_ci"] = pct(bs)
        o["qr"] = {str(k): v for k, v in qr.items()}; o["ols_beta"] = ols
        w("A. Quantile regression β(τ) (kernel-robust 95% CI; for τ = 0.05/0.10 also a block "
          f"bootstrap interval): OLS β = {ols:.3f}\n")
        w(md([[f"{tau:.2f}", fci(v["beta"], v["lo"], v["hi"], 3),
               fci(v["beta"], *v["boot_ci"], 3) if "boot_ci" in v else "—"] for tau, v in qr.items()],
             ["τ", "β(τ) [kernel CI]", "β(τ) [block bootstrap CI]"]))
        # ---------------- B
        lam = {}
        rs = stats.spearmanr(y, spy).correlation
        for u in (0.05, 0.10):
            ptL = lam_emp(y, spy, u); ptU = lam_emp(y, spy, u, lower=False)
            bsL = [lam_emp(y[i], spy[i], u) for i in IDX.T]
            bsU = [lam_emp(y[i], spy[i], u, False) for i in IDX.T]
            diff = np.array(bsL) - np.array(bsU)
            lam[str(u)] = dict(L=ptL, L_ci=pct(bsL), U=ptU, U_ci=pct(bsU), indep=u, gauss=lam_gauss_bench(rs, u),
                               diff=ptL - ptU, diff_ci=pct(diff), n_joint=int(round(ptL * u * n)))
        o["lam_np"] = lam; o["spearman"] = float(rs)
        w(f"B. Non-parametric tail dependence (Spearman ρ_s = {rs:.3f})\n")
        w(md([[u_, v["n_joint"], fci(v["L"], *v["L_ci"]), f"{v['indep']:.2f}", f"{v['gauss']:.3f}",
               fci(v["U"], *v["U_ci"]), fci(v["diff"], *v["diff_ci"])] for u_, v in lam.items()],
             ["u", "Days in both lower tails", "λ_L(u) [CI]", "Independence benchmark", "Same-rank-correlation Gaussian benchmark", "λ_U(u) [CI]", "λ_L − λ_U [CI]"]))
        # ---------------- C
        uy, _, _ = garch_pit(y); us, _, _ = garch_pit(spy)
        fits = {nm: fit_copula(nm, uy, us) for nm in COPULAS}
        best = min(fits, key=lambda k: fits[k]["aic"])
        Bc = int(B_COP * B_scale)
        idxc = stationary_boot(n, Bc, L_BOOT, rng)
        for nm in fits:
            fits[nm]["lam_u05"] = copula_lam_u(nm, fits[nm]["par"], 0.05)
        try:                                                 # optional: cross-check against pyvinecopulib's MLE
            import pyvinecopulib as pv
            U2 = np.column_stack([uy, us]); ctl = pv.FitControlsBicop(parametric_method="mle"); dmax = 0.0
            for nm, fam, rot in (("Gaussian", pv.BicopFamily.gaussian, 0), ("Student-t", pv.BicopFamily.student, 0),
                                 ("Clayton", pv.BicopFamily.clayton, 0), ("Survival Gumbel", pv.BicopFamily.gumbel, 180),
                                 ("Frank", pv.BicopFamily.frank, 0)):
                cb = pv.Bicop(family=fam, rotation=rot); cb.fit(U2, ctl)
                dmax = max(dmax, abs(cb.loglik(U2) - fits[nm]["ll"]))
            check(f"Copula log-likelihood agrees with pyvinecopulib ({label} {a})", dmax < 0.05, f"max diff {dmax:.3f}")
        except ImportError:
            pass
        boot_fams = list(dict.fromkeys([best, "Gaussian", "Student-t", "Clayton", "Survival Gumbel"]))
        lam_bs = {nm: [] for nm in boot_fams}; lamu_bs = {nm: [] for nm in boot_fams}
        for i in idxc.T:
            for nm in boot_fams:
                fb = fit_copula(nm, uy[i], us[i], x0=fits[nm]["par"])
                lam_bs[nm].append(fb["lam_L"]); lamu_bs[nm].append(copula_lam_u(nm, fb["par"], 0.05))
        for nm in boot_fams:
            fits[nm]["lam_L_ci"] = pct(lam_bs[nm]); fits[nm]["lam_u05_ci"] = pct(lamu_bs[nm])
        tau_k = float(stats.kendalltau(uy, us).correlation)
        o["copula"] = dict(fits={k: dict(v, params=natural_params(k, v["par"])) for k, v in fits.items()}, best=best,
                           kendall=tau_k)
        w(f"C. Copula-GARCH (marginal GARCH(1,1)-t, parameter PIT; Kendall τ = {tau_k:.3f}): AIC best = {best}\n")
        w("λ_L is the asymptotic lower-tail dependence coefficient; λ_L(5%) = C(0.05, 0.05)/0.05 = P(asset PIT ≤ 5% | SPY PIT ≤ 5%), equal to 0.05 under independence. "
          "Intervals are block bootstraps on the PIT (marginal estimation error not included).\n")

        def _c(v_, k):
            return fci(v_[k], *v_[k + "_ci"], 3) if k + "_ci" in v_ else f"{v_[k]:.3f}"
        w(md([[nm + (" ★" if nm == best else ""), v_["params"], f"{v_['ll']:.2f}", f"{v_['aic']:.2f}",
               _c(v_, "lam_L"), _c(v_, "lam_u05")]
              for nm, v_ in o["copula"]["fits"].items()],
             ["Copula", "Parameters", "Log-likelihood", "AIC", "λ_L [CI]", "λ_L(5%) [CI]"]))
        # ---------------- D
        cqs = {}
        idxq2 = IDX
        for tau in (0.01, 0.025, 0.05, 0.10, 0.20, 0.50):
            pt = float(cq(y, spy, tau))
            bs = cq(y[idxq2], spy[idxq2], tau)
            cqs[str(tau)] = dict(pt=pt, ci=pct(bs), n_tail=int(round(tau * n)))
        o["cq"] = cqs
        w("D. Cross-quantilogram ρ_τ (contemporaneous)\n")
        w(md([[t_, v["n_tail"], fci(v["pt"], *v["ci"], 3)] for t_, v in cqs.items()], ["τ", "Days in each tail", "ρ_τ [CI]"]))
        # ---------------- E
        zx = (y - y.mean()) / y.std(ddof=1); zy = (spy - spy.mean()) / spy.std(ddof=1)
        rho = float(np.corrcoef(y, spy)[0, 1])
        cs = [0.0, 0.5, 1.0, 1.5]
        bench = exc_bench(rho, cs)
        Be = int(B_EXC * B_scale)
        idxe = IDX[:, :Be]
        exc = {}
        dvec = []
        for c in cs:
            m_, nn = exc_corr(zx, zy, c, -1); p_, np_ = exc_corr(zx, zy, c, +1)
            bsm = [exc_corr(zx[i], zy[i], c, -1)[0] for i in idxe.T]
            bsp = [exc_corr(zx[i], zy[i], c, +1)[0] for i in idxe.T]
            exc[str(c)] = dict(neg=m_, neg_n=nn, neg_ci=pct(bsm) if np.isfinite(m_) else [np.nan, np.nan],
                               pos=p_, pos_n=np_, pos_ci=pct(bsp) if np.isfinite(p_) else [np.nan, np.nan], bench=bench[c])
            if c <= 1.0:
                dvec.append(np.array(bsp) - np.array(bsm))
        D_ = np.array(dvec).T; D_ = D_[np.all(np.isfinite(D_), 1)]
        dpt = np.array([exc[str(c)]["pos"] - exc[str(c)]["neg"] for c in cs if c <= 1.0])
        if len(D_) > 50 and np.all(np.isfinite(dpt)):
            Om = np.cov(D_.T)
            J = float(dpt @ np.linalg.pinv(Om) @ dpt)
            o["exc_sym"] = dict(J=J, p=float(stats.chi2.sf(J, len(dpt))))
        else:
            o["exc_sym"] = dict(J=np.nan, p=np.nan)
        cc_ = {}
        for p_ in (0.10, 0.05):
            m = spy <= np.quantile(spy, p_)
            pt = float(np.corrcoef(y[m], spy[m])[0, 1])
            bs = []
            for i in idxe.T:
                mm = spy[i] <= np.quantile(spy[i], p_)
                bs.append(np.corrcoef(y[i][mm], spy[i][mm])[0, 1])
            cc_[str(p_)] = dict(pt=pt, ci=pct(bs), bench=cond_corr_bench(rho, p_), n=int(m.sum()))
        o["exc"] = exc; o["cond_corr"] = cc_; o["rho"] = rho
        w(f"E. Exceedance correlation (standardized returns; full-sample ρ = {rho:.3f}; the normal benchmark is a bivariate normal with the same ρ)\n")
        w(md([[c, exc[str(c)]["neg_n"], fci(exc[str(c)]["neg"], *exc[str(c)]["neg_ci"]), f"{exc[str(c)]['bench']:.3f}",
               exc[str(c)]["pos_n"], fci(exc[str(c)]["pos"], *exc[str(c)]["pos_ci"])] for c in cs],
             ["c (std dev)", "Days both < −c", "Lower exceedance correlation [CI]", "Normal benchmark", "Days both > c", "Upper exceedance correlation [CI]"]))
        w(f"Upper/lower exceedance correlation symmetry (c = 0, 0.5, 1): J = {o['exc_sym']['J']:.2f}, p = {pf(o['exc_sym']['p'])}\n")
        w(md([[f"SPY lowest {int(float(k) * 100)}%", v["n"], fci(v["pt"], *v["ci"]), f"{v['bench']:.3f}"] for k, v in cc_.items()],
             ["Condition", "Days", "Conditional correlation [CI]", "Truncation benchmark under the same-ρ normal"]))
        # ---------------- F
        _, zy_, _ = garch_pit(y); _, zs_, _ = garch_pit(spy)
        Z = np.column_stack([zy_, zs_])
        dc = dcc_fit(Z)
        rt = dc["rho"]
        D10, D5, D1 = (spy <= q10).astype(float), (spy <= q05).astype(float), (spy <= q01).astype(float)
        X = np.column_stack([np.ones(n), D10, D5, D1])
        b, V = nw_ols(rt, X, HAC_DCC)
        se = np.sqrt(np.diag(V))
        cvec5 = np.array([1, 1, 1, 0.0]); cvec1 = np.array([1, 1, 1, 1.0])
        s5, s1 = float(cvec5 @ b), float(cvec1 @ b)
        se5, se1 = float(math.sqrt(cvec5 @ V @ cvec5)), float(math.sqrt(cvec1 @ V @ cvec1))
        o["dcc"] = dict(a=dc["a"], b=dc["b"], mean=float(rt.mean()), min=float(rt.min()), max=float(rt.max()),
                        c=b.tolist(), se=se.tolist(), tail5=s5, tail5_ci=[s5 - 1.96 * se5, s5 + 1.96 * se5],
                        tail1=s1, tail1_ci=[s1 - 1.96 * se1, s1 + 1.96 * se1],
                        p_c1=float(2 * stats.norm.sf(abs(b[1] / se[1]))), p_c2=float(2 * stats.norm.sf(abs(b[2] / se[2]))),
                        p_c3=float(2 * stats.norm.sf(abs(b[3] / se[3]))),
                        share_neg_tail=float(np.mean(rt[tail5] < 0)))
        vix = R["VIX"].values
        bv, Vv = nw_ols(rt, np.column_stack([np.ones(n), vix]), HAC_DCC)
        # DCC's ρ_t is determined by t−1 information (predetermined variable); also check
        # whether the correlation ρ_{t+1} the day after a lower-tail shock rises
        bl, Vl = nw_ols(rt[1:], X[:-1], HAC_DCC)
        s5l = float(cvec5 @ bl); se5l = float(math.sqrt(cvec5 @ Vl @ cvec5))
        o["dcc"]["lead_c"] = bl.tolist(); o["dcc"]["lead_tail5"] = s5l
        o["dcc"]["lead_tail5_ci"] = [s5l - 1.96 * se5l, s5l + 1.96 * se5l]
        o["dcc"]["lead_p_c123"] = float(2 * stats.norm.sf(abs((s5l - bl[0]) / math.sqrt(
            np.array([0, 1, 1, 0.]) @ Vl @ np.array([0, 1, 1, 0.])))))
        o["dcc"]["vix_slope"] = float(bv[1]); o["dcc"]["vix_p"] = float(2 * stats.norm.sf(abs(bv[1] / math.sqrt(Vv[1, 1]))))
        o["dcc_path"] = rt
        dd = o["dcc"]
        w(f"F. DCC(1,1): a = {dd['a']:.4f}, b = {dd['b']:.4f}; ρ_t mean {dd['mean']:.3f}, range {dd['min']:.3f}–{dd['max']:.3f}\n")
        w(md([["Constant c0", fci(b[0], b[0] - 1.96 * se[0], b[0] + 1.96 * se[0], 3), "—"],
              ["D(SPY ≤ 10% quantile) c1", fci(b[1], b[1] - 1.96 * se[1], b[1] + 1.96 * se[1], 3), pf(dd["p_c1"])],
              ["D(≤ 5%) c2", fci(b[2], b[2] - 1.96 * se[2], b[2] + 1.96 * se[2], 3), pf(dd["p_c2"])],
              ["D(≤ 1%) c3", fci(b[3], b[3] - 1.96 * se[3], b[3] + 1.96 * se[3], 3), pf(dd["p_c3"])],
              ["Conditional correlation on 5% tail days c0+c1+c2", fci(s5, *dd["tail5_ci"], 3), "—"],
              ["Conditional correlation on 1% tail days c0+c1+c2+c3", fci(s1, *dd["tail1_ci"], 3), "—"]],
             ["DCC regression (Newey–West, lag 20)", "Estimate [95% CI]", "p"]))
        w(f"Note: in DCC, ρ_t is determined by t−1 information (the literature practice is to regress on same-day dummies); ρ_(t+1) on the day after a lower-tail day = "
          f"{fci(dd['lead_tail5'], *dd['lead_tail5_ci'], 3)} (p relative to the constant = {pf(dd['lead_p_c123'])}). "
          "HAC intervals take the estimated ρ_t path as given and do not include DCC parameter estimation error.")
        w(f"Share of 5% tail days with ρ_t < 0: {dd['share_neg_tail'] * 100:.0f}%; slope of ρ_t on VIX {dd['vix_slope']:+.4f} (p = {pf(dd['vix_p'])})")
        # ---------------- G
        Xh = np.column_stack([y, spy])
        h = hmm_fit(Xh, rng=np.random.default_rng(SEED + 11))
        hs = hmm_summary(h)
        bsr = {k: [] for k in ("rho0", "rho1", "beta0", "beta1", "drho")}
        rng_h = np.random.default_rng(SEED + 12)
        for _ in range(int(B_HMM * B_scale)):
            Xs = hmm_sim(h, n, rng_h)
            try:
                s_ = hmm_summary(hmm_fit(Xs, n_start=2, rng=rng_h))
            except Exception:                              # noqa: BLE001
                continue
            bsr["rho0"].append(s_[0]["rho"]); bsr["rho1"].append(s_[1]["rho"])
            bsr["beta0"].append(s_[0]["beta"]); bsr["beta1"].append(s_[1]["beta"])
            bsr["drho"].append(s_[1]["rho"] - s_[0]["rho"])
        tail_p = float(h["gam"][tail5, 1].mean())
        o["hmm"] = dict(states=hs, ci={k: pct(v) for k, v in bsr.items()}, ll=h["ll"], p_turb_on_tail5=tail_p)
        w("\nG. Two-state bivariate Gaussian HMM (state 1 = high SPY variance; parameter-bootstrap intervals)\n")
        w(md([[k_, f"{s['vol_spy']:.1f}", f"{s['vol_asset']:.1f}",
               fci(s["rho"], *o["hmm"]["ci"][f"rho{k_}"], 3), fci(s["beta"], *o["hmm"]["ci"][f"beta{k_}"], 3),
               f"{s['dur']:.1f}", f"{s['share'] * 100:.0f}%"] for k_, s in enumerate(hs)],
             ["State", "SPY ann. vol %", "Asset ann. vol %", "ρ [CI]", "β [CI]", "Expected duration (days)", "Share"]))
        w(f"Difference in ρ between the turbulent and calm states: {fci(hs[1]['rho'] - hs[0]['rho'], *o['hmm']['ci']['drho'], 3)}; "
          f"mean smoothed probability of being in the turbulent state on SPY 5% tail days: {tail_p:.2f}")
        try:
            msr = sm.tsa.MarkovRegression(y, k_regimes=2, exog=spy, switching_variance=True).fit(search_reps=20, disp=False)
            par = msr.params
            s2 = [par[f"sigma2[{k}]"] for k in range(2)]
            bx = [par[f"x1[{k}]"] for k in range(2)]
            k_hi = int(np.argmax(s2))
            o["msreg"] = dict(beta_low=float(bx[1 - k_hi]), beta_high=float(bx[k_hi]),
                              vol_low=float(math.sqrt(s2[1 - k_hi] * 252) * 100), vol_high=float(math.sqrt(s2[k_hi] * 252) * 100),
                              se_beta_high=float(msr.bse[f"x1[{k_hi}]"]), se_beta_low=float(msr.bse[f"x1[{1 - k_hi}]"]))
            mr = o["msreg"]
            w(f"statsmodels MarkovRegression (asset on SPY, β and residual variance switching): high-residual-variance regime β = {mr['beta_high']:.3f}"
              f" (SE {mr['se_beta_high']:.3f}), low-variance regime β = {mr['beta_low']:.3f} (SE {mr['se_beta_low']:.3f})")
        except Exception as e:                              # noqa: BLE001
            o["msreg"] = dict(error=str(e))
        # ---------------- H
        _, _, cv = garch_pit(y)
        hh = (cv ** 2) / np.mean(cv ** 2)
        tv = tvp_fit(y, spy, hh)
        bsm, Psm = tv["beta_s"], tv["P_s"]
        ok_ = np.arange(n) >= 20
        m_t, m_o = bsm[tail5 & ok_].mean(), bsm[~tail5 & ok_].mean()
        vix = R["VIX"].values
        m_v = bsm[(vix > 25) & ok_].mean() if ((vix > 25) & ok_).sum() else np.nan
        o["tvp"] = dict(q=tv["q"], LR=tv["LR"], p_const=tv["p_const"], mean_tail5=float(m_t), mean_other=float(m_o),
                        mean_vix25=float(m_v), min=float(bsm[ok_].min()), max=float(bsm[ok_].max()),
                        share_neg_tail=float(np.mean(bsm[tail5 & ok_] < 0)),
                        band_tail5=[float(np.mean(bsm[tail5 & ok_] - 1.96 * np.sqrt(Psm[tail5 & ok_]))),
                                    float(np.mean(bsm[tail5 & ok_] + 1.96 * np.sqrt(Psm[tail5 & ok_])))])
        o["tvp_path"] = (bsm, Psm)
        tt = o["tvp"]
        w(f"\nH. TVP β_t (random walk, Kalman smoother): state variance q = {tt['q']:.2e}; LR for constant β = {tt['LR']:.2f} (p = {pf(tt['p_const'])}); "
          f"mean smoothed β_t on SPY 5% tail days {tt['mean_tail5']:.3f} (mean daily 95% band [{tt['band_tail5'][0]:.3f}, {tt['band_tail5'][1]:.3f}]), "
          f"other days {tt['mean_other']:.3f}, VIX > 25 days {tt['mean_vix25']:.3f}; range {tt['min']:.3f}–{tt['max']:.3f}; "
          f"share of tail days with β_t < 0: {tt['share_neg_tail'] * 100:.0f}%")
        out[a] = o
    # ---------------- Two-day return re-check (long sample)
    if two_day:
        w(f"\n## {label}: non-overlapping two-day return re-check (to mitigate the BTC-USD 00:00 UTC closing-time mismatch)\n")
        lr_ = np.log1p(R[["BTC", "GLD", "SPY"]])
        g2 = np.arange(len(lr_)) // 2
        R2 = np.expm1(lr_.groupby(g2).sum())
        R2 = R2.iloc[:-1] if len(lr_) % 2 else R2
        n2 = len(R2)
        idx2 = stationary_boot(n2, int(B_LAM * B_scale), L_BOOT, rng)
        rows = []
        for a in ("BTC", "GLD"):
            y2, s2_ = R2[a].values, R2["SPY"].values
            lam5 = lam_emp(y2, s2_, 0.05); lam10 = lam_emp(y2, s2_, 0.10)
            b5 = pct([lam_emp(y2[i], s2_[i], 0.05) for i in idx2.T]); b10 = pct([lam_emp(y2[i], s2_[i], 0.10) for i in idx2.T])
            c5 = float(cq(y2, s2_, 0.05)); cb = pct(cq(y2[idx2], s2_[idx2], 0.05))
            rho2 = float(np.corrcoef(y2, s2_)[0, 1])
            out.setdefault("two_day", {})[a] = dict(n=n2, rho=rho2, lam5=lam5, lam5_ci=b5, lam10=lam10, lam10_ci=b10,
                                                    cq5=c5, cq5_ci=cb)
            rows.append([a, n2, f"{rho2:.3f}", fci(lam5, *b5), fci(lam10, *b10), fci(c5, *cb, 3)])
        w(md(rows, ["Asset", "Two-day observations", "ρ", "λ_L(5%) [CI]", "λ_L(10%) [CI]", "CQ(5%) [CI]"]))
    return out


# ====================================================================== Main program
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--outdir", default="out/22")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.data = a.data or find_main()
    a.raw = a.raw or find_raw_L()
    if a.data is None:
        sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; please specify with --data.")
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(HERE, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    scale = 0.5 if a.quick else 1.0
    w("# Conditional dependence tests of safe-haven properties (script 22)\n")
    w(f"Seed {SEED}; stationary block bootstrap expected block length {L_BOOT}; bootstrap draws: quantile regression {int(B_QR * scale)}, non-parametric tail dependence and cross-quantilogram "
      f"{int(B_LAM * scale)}, copula {int(B_COP * scale)}, exceedance correlation {int(B_EXC * scale)}, HMM parameter bootstrap {int(B_HMM * scale)}; "
      f"DCC regression Newey–West lag {HAC_DCC}.\n")
    ok_self = selftest()
    rng = np.random.default_rng(SEED)

    d = pd.read_csv(a.data, parse_dates=["Date"]).set_index("Date").sort_index()
    Rm = d[["IBIT", "GLD", "TLT", "SPY"]].copy(); Rm["VIX"] = d["VIX_Close"]
    RES = {"main": analyze("main sample", Rm, ["IBIT", "GLD", "TLT"], rng, scale)}
    # Cross-check against existing results
    ib = RES["main"]["IBIT"]
    check("CQ(5%) matches script 19 (IBIT 0.218, GLD 0.116)",
          abs(ib["cq"]["0.05"]["pt"] - 0.218) < 6e-4 and abs(RES["main"]["GLD"]["cq"]["0.05"]["pt"] - 0.116) < 6e-4,
          f"{ib['cq']['0.05']['pt']:.3f} / {RES['main']['GLD']['cq']['0.05']['pt']:.3f}")
    check("Full-sample correlations match the paper (IBIT 0.411, GLD 0.173)", abs(ib["rho"] - 0.411) < 0.001
          and abs(RES["main"]["GLD"]["rho"] - 0.173) < 0.001, f"{ib['rho']:.3f} / {RES['main']['GLD']['rho']:.3f}")

    if a.raw:
        RL = load_long(a.raw)
        RES["long"] = analyze("long sample (BTC-USD spot)", RL, ["BTC", "GLD", "TLT"], rng, 0.5 * scale, two_day=True)
        check("Long-sample trading-day count is of the same order as script 12 (≈ 2950)", 2900 <= len(RL) <= 3000, str(len(RL)))
    else:
        w("\n(No long-sample snapshot data/raw/*__…L.csv found; skipping the long-sample part)")

    # ---------------------------------------------------------------- I. Summary
    w("\n# I. Summary: the direction of dependence during extreme market declines\n")
    w("Interpretation: an interval entirely above 0 = positive dependence (the opposite of safe haven); containing 0 = cannot reject uncorrelated; entirely below 0 = negative dependence (strong safe haven). "
      "The independence benchmark for λ_L is u, not 0.\n")
    rows = []

    def verdict(lo, hi, null=0.0):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return "—"
        return "Positive dependence" if lo > null else ("Negative dependence" if hi < null else "Cannot reject")
    for samp, lab in (("main", "Main sample"), ("long", "Long sample")):
        if samp not in RES:
            continue
        for asset in ("IBIT", "BTC", "GLD", "TLT"):
            if asset not in RES[samp]:
                continue
            o = RES[samp][asset]
            items = [
                ("Quantile regression β(0.05)", o["qr"]["0.05"]["beta"], *o["qr"]["0.05"]["boot_ci"], 0.0),
                ("λ_L(5%), non-parametric (independence = 0.05)", o["lam_np"]["0.05"]["L"], *o["lam_np"]["0.05"]["L_ci"], 0.05),
                (f"Copula (AIC best: {o['copula']['best']}) implied λ_L(5%) (independence = 0.05)",
                 o["copula"]["fits"][o["copula"]["best"]]["lam_u05"], *o["copula"]["fits"][o["copula"]["best"]]["lam_u05_ci"], 0.05),
                ("Cross-quantilogram ρ(5%)", o["cq"]["0.05"]["pt"], *o["cq"]["0.05"]["ci"], 0.0),
                ("Lower exceedance correlation (c = 1)", o["exc"]["1.0"]["neg"], *o["exc"]["1.0"]["neg_ci"], 0.0),
                ("DCC: conditional correlation on SPY 5% tail days", o["dcc"]["tail5"], *o["dcc"]["tail5_ci"], 0.0),
                ("HMM: turbulent-state ρ", o["hmm"]["states"][1]["rho"], *o["hmm"]["ci"]["rho1"], 0.0),
            ]
            for nm, pt, lo, hi, null in items:
                rows.append([lab, asset, nm, fci(pt, lo, hi, 3), verdict(lo, hi, null)])
            rows.append([lab, asset, "TVP: mean smoothed β_t on SPY 5% tail days (mean daily 95% band)",
                         fci(o["tvp"]["mean_tail5"], *o["tvp"]["band_tail5"], 3), verdict(*o["tvp"]["band_tail5"])])
    w(md(rows, ["Sample", "Asset", "Measure", "Estimate [95% CI]", "Interpretation"]))
    KEY["summary_rows"] = rows

    # ---------------------------------------------------------------- J. Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        C_IBIT, C_GLD, INK, INK2, SHADE, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d4", "#e6e5e1"
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                             "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                             "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                             "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7})
        dates = Rm.index
        spy = Rm["SPY"].values; tail = spy <= np.quantile(spy, 0.05)
        hv = Rm["VIX"].values > 25
        fig, axs = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True)
        from matplotlib.patches import Patch
        for ax in axs:
            for i in np.where(hv)[0]:
                ax.axvspan(dates[i] - pd.Timedelta(hours=12), dates[i] + pd.Timedelta(hours=12), color=SHADE, lw=0, zorder=0)
        ax = axs[0]
        for asset, c, lab in (("IBIT", C_IBIT, "IBIT"), ("GLD", C_GLD, "GLD")):
            ax.plot(dates, RES["main"][asset]["dcc_path"], color=c, lw=1.1, label=lab)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.plot(dates[tail], np.full(tail.sum(), -0.55), "|", color=INK, ms=6, mew=0.9, label="SPY 5% tail day")
        ax.set_ylabel("DCC correlation with SPY"); ax.set_ylim(-0.65, 1.0)
        h_, l_ = ax.get_legend_handles_labels()
        ax.legend(h_ + [Patch(color=SHADE, lw=0)], l_ + ["VIX > 25"], loc="upper left", ncol=4)
        ax.set_title("a. Dynamic conditional correlation, DCC(1,1)", loc="left", fontsize=8)
        ax = axs[1]
        for asset, c, lab in (("IBIT", C_IBIT, "IBIT"), ("GLD", C_GLD, "GLD")):
            b_, P_ = RES["main"][asset]["tvp_path"]
            ax.plot(dates[20:], b_[20:], color=c, lw=1.1, label=lab)
            ax.fill_between(dates[20:], (b_ - 1.96 * np.sqrt(P_))[20:], (b_ + 1.96 * np.sqrt(P_))[20:], color=c, alpha=0.15, lw=0)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.plot(dates[tail], np.full(tail.sum(), -1.15), "|", color=INK, ms=6, mew=0.9)
        ax.set_ylabel("Time-varying beta on SPY")
        ax.legend(loc="upper left", ncol=2)
        ax.set_title("b. Time-varying beta (Kalman smoother, 95% band)", loc="left", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.outdir, f"fig_safe_haven_dynamics.{ext}"), dpi=300)
        plt.close(fig)
        w("\nFigure saved: fig_safe_haven_dynamics.png / .pdf (gray = VIX > 25)")
    except Exception as e:                                  # noqa: BLE001
        w(f"\n(plotting failed: {e})")

    # ---------------------------------------------------------------- Outputs
    w("\n## Self-test summary\n")
    for c in CHECKS:
        w("  " + c)
    nf = sum(c.startswith("[FAIL]") for c in CHECKS)
    w(f"\n{len(CHECKS)} checks, {nf} failures. Elapsed {time.time() - t0:.0f} seconds.")
    with open(os.path.join(a.outdir, "safe_haven_dependence.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items() if k not in ("dcc_path", "tvp_path")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return None
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    KEY["results"] = clean(RES)
    with open(os.path.join(a.outdir, "safe_haven_dependence.json"), "w", encoding="utf8") as fh:
        json.dump(clean(KEY), fh, indent=1, ensure_ascii=False)
    return 0 if (nf == 0 and ok_self) else 1


if __name__ == "__main__":
    sys.exit(main())