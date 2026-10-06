"""
21_forecast_design.py — Forecast design for VaR/ES and forecast-accuracy tests

  (1) State clearly whether the risk measures are in-sample estimates or rolling / expanding
      out-of-sample forecasts;
  (2) State the estimation window, refit frequency, number of simulations, random seed,
      innovation distribution, and all GARCH parameter estimates;
  (3) The comparison between normal, Student-t, historical simulation, and GARCH-t must use
      a formal forecast-accuracy test, not description.

This script (does not modify the paper, only produces results):
  0. Setup overview: window, refit frequency, number of forecasts, simulations, seed,
     innovation distribution, mean equation.
  A. Full-sample GARCH(1,1) parameters (Student-t and normal innovations; 4 portfolios +
     IBIT/GLD/SPY/TLT single assets), robust standard errors (Bollerslev–Woodridge),
     log-likelihood, persistence, half-life, unconditional volatility, Ljung–Box test on
     standardized residuals; compared against the self-implemented estimates in script 09.
  B. Summary of the rolling re-estimation parameters (every re-estimation's parameters are
     written to CSV).
  C. Strict out-of-sample one-step forecasts for six models: static normal, static
     Student-t, historical simulation, EWMA-normal, GARCH-normal, GARCH-t. Main design:
     250-day moving window; GARCH re-estimated every 20 days, other models updated daily.
  D. Backtests: Kupiec, Christoffersen, Engle–Manganelli dynamic quantile (DQ) tests;
     Acerbi–Szekely Z1/Z2 (all six models simulated under their own predictive
     distribution); McNeil–Frey.
  E. Forecast-accuracy comparison: unconditional (DM-type) and conditional (GW) forecast
     ability tests for all 15 model pairs, FZ0 loss (VaR and ES jointly) and pinball loss
     (VaR); model confidence set (Hansen, Lunde & Nason 2011).
  F. Out-of-sample ES ratio: the ratio of each model's average ES forecast for the IBIT
     portfolio to that for the GLD portfolio (block bootstrap interval), compared against
     the in-sample 1.63.
  G. Robustness: expanding window; GARCH re-estimated daily.
  H. Item-by-item reconciliation with script 04 (Table 3 of the paper).

Dependencies: numpy, pandas, scipy, arch (>= 6). Can be run standalone.
Usage:
    python 21_forecast_design.py                    # auto-locate data/processed/returns__YYYYMMDD.csv
    python 21_forecast_design.py --data data/processed/returns__20260921.csv --outdir out/21
    python 21_forecast_design.py --quick            # skip the daily-refit robustness (saves ~3 minutes)
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
from scipy import stats, integrate

warnings.filterwarnings("ignore")
try:
    from arch.univariate import ConstantMean, GARCH, StudentsT, Normal
    from arch.bootstrap import MCS
except ImportError:
    sys.exit("Missing the arch package. Please first run: python -m pip install arch")

HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ Setup ----
ALPHAS = (0.05, 0.01)
N_START = 250            # estimation window (moving-window length / expanding-window initial length)
REFIT = 20               # GARCH refit interval (trading days)
EWMA_LAMBDA = 0.94       # RiskMetrics
SEED = 20260926          # main seed for this script; each task derives an independent substream via SeedSequence
B_AS = 2000              # simulation draws for the Acerbi–Szekely p-value
B_MF = 5000              # McNeil–Frey bootstrap draws
B_MCS = 5000             # model confidence set bootstrap draws
MCS_BLOCK = 7            # expected block length for the MCS stationary bootstrap (≈ n^{1/3})
MCS_SIZE = 0.10          # 90% model confidence set
B_RATIO = 2000           # block bootstrap draws for the out-of-sample ES ratio
L_RATIO = 20             # its expected block length (ES forecasts are highly persistent)
HAC_LAG = 5
DQ_LAGS = 4
MODELS = ["Normal", "Student-t", "HS", "EWMA", "GARCH-n", "GARCH-t"]
MODEL_EN = {"Normal": "Normal", "Student-t": "Student-t", "HS": "HS", "EWMA": "EWMA",
            "GARCH-n": "GARCH-N", "GARCH-t": "GARCH-t"}

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


def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ====================================================== Distributions and VaR/ES ====
def var_z_t(a, nu):
    """The a quantile (negative) of a unit-variance standardized t; same convention as
    arch StudentsT and rugarch's 'std'."""
    return stats.t.ppf(a, nu) / np.sqrt(nu / (nu - 2.0))


def es_z_t(a, nu):
    q = stats.t.ppf(a, nu)
    return -stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1.0) * a) / np.sqrt(nu / (nu - 2.0))


def es_raw_t(a, df):
    """ES (negative) of an unstandardized t (scale parameter = 1)."""
    q = stats.t.ppf(a, df)
    return -stats.t.pdf(q, df) * (df + q * q) / ((df - 1.0) * a)


Z_N = {a: stats.norm.ppf(a) for a in ALPHAS}
ES_N = {a: -stats.norm.pdf(stats.norm.ppf(a)) / a for a in ALPHAS}


# ====================================================== Forecasters ====
class ModelSpecError(RuntimeError):
    """The model does not match the recursion (e.g. built as GJR-GARCH by mistake). This is a code
    error: it must be raised, not swallowed as a failed fit."""


RECURSION_DEV: list[float] = []     # max deviation between this script's recursion and arch's volatility, per rolling fit


def fit_arch(y, dist):
    """ConstantMean + GARCH(p=1, o=0, q=1); y is in percentage returns. Returns the arch result object.

    Parameter set does not match -> ModelSpecError (code error, raised); optimizer did not converge ->
    ArithmeticError (in the rolling forecasts counted as a failure and the previous parameters are kept;
    in the full-sample estimates it stops the script)."""
    am = ConstantMean(y)
    # keywords are required: arch's signature is GARCH(p, o, q), so GARCH(1, 1) would build GJR-GARCH(1,1,1)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = StudentsT() if dist == "t" else Normal()
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 2000})
    exp = {"mu", "omega", "alpha[1]", "beta[1]"} | ({"nu"} if dist == "t" else set())
    if set(res.params.index) != exp:
        raise ModelSpecError(f"parameter set does not match GARCH(1,1): {sorted(res.params.index)}")
    # show_warning=False silences non-convergence, so check it explicitly
    if res.convergence_flag != 0:
        raise ArithmeticError(f"optimizer did not converge (flag={res.convergence_flag})")
    return res


def roll_garch(r, dist, window, refit, log, fails):
    """Strict out-of-sample one-step forecasts. σ_t on day t uses only information up to
    t−1; between refits the recursion advances with fixed parameters."""
    n = len(r); x = r * 100.0
    n_out = n - N_START
    mu = np.full(n_out, np.nan); sd = np.full(n_out, np.nan); nu = np.full(n_out, np.nan)
    par = None; s2p = ep = None
    for k in range(n_out):
        t = N_START + k
        if k % refit == 0:
            lo = t - N_START if window == "moving" else 0
            y = x[lo:t]
            try:
                res = fit_arch(y, dist)
                p = res.params
                par = dict(mu=p["mu"], omega=p["omega"], alpha=p["alpha[1]"], beta=p["beta[1]"],
                           nu=p["nu"] if dist == "t" else np.nan)
                cv = np.asarray(res.conditional_volatility, float)
                # recursion check: reproduce arch's conditional volatility with this script's recursion
                s2c = cv[0] ** 2; dev = 0.0
                for j in range(1, len(y)):
                    s2c = par["omega"] + par["alpha"] * (y[j - 1] - par["mu"]) ** 2 + par["beta"] * s2c
                    dev = max(dev, abs(math.sqrt(s2c) - cv[j]))
                RECURSION_DEV.append(dev)
                s2p = cv[-1] ** 2; ep = y[-1] - par["mu"]
                log.append(dict(k=k, first=lo, last=t - 1, n_obs=len(y), **{kk: float(v) for kk, v in par.items()},
                                llf=float(res.loglikelihood)))
            except ModelSpecError:
                raise
            except Exception:                       # noqa: BLE001  failed or not converged: keep the previous parameters
                fails.append(k)
        if par is None:
            continue
        s2 = par["omega"] + par["alpha"] * ep ** 2 + par["beta"] * s2p
        mu[k] = par["mu"] / 100; sd[k] = math.sqrt(s2) / 100; nu[k] = par["nu"]
        s2p = s2; ep = x[t] - par["mu"]
    return mu, sd, nu


def windows(r, window):
    n = len(r)
    for k in range(n - N_START):
        t = N_START + k
        yield k, r[(t - N_START if window == "moving" else 0):t]


def build_forecasts(r, window, garch_refit, logs):
    """Returns {model: dict(kind, parameter arrays, VaR{a}, ES{a}, scale)}, all corresponding
    to r[N_START:]."""
    n_out = len(r) - N_START
    F = {}
    # Static normal (daily, using the window's sample mean and standard deviation)
    mu = np.empty(n_out); sd = np.empty(n_out)
    for k, y in windows(r, window):
        mu[k] = y.mean(); sd[k] = y.std(ddof=1)
    F["Normal"] = dict(kind="norm", mu=mu, sd=sd)
    # Static Student-t (daily MLE on the window: location, scale, degrees of freedom;
    # df restricted to (2.05, 500])
    loc = np.empty(n_out); sc = np.empty(n_out); df = np.empty(n_out)
    for k, y in windows(r, window):
        d0, l0, s0 = stats.t.fit(y)
        if not (2.05 < d0 <= 500):
            d0 = float(np.clip(d0, 2.05, 500)); d0, l0, s0 = stats.t.fit(y, fdf=d0)
        df[k], loc[k], sc[k] = d0, l0, s0
    F["Student-t"] = dict(kind="t_raw", loc=loc, scale=sc, df=df, sd=sc * np.sqrt(df / (df - 2)))
    # Historical simulation (daily, using the window's empirical distribution)
    W = [y.copy() for _, y in windows(r, window)]
    F["HS"] = dict(kind="emp", win=W, sd=np.array([y.std(ddof=1) for y in W]))
    # EWMA-normal (λ = 0.94; independent of the window design)
    s2 = np.var(r[:N_START], ddof=1)
    for t in range(1, N_START):
        s2 = EWMA_LAMBDA * s2 + (1 - EWMA_LAMBDA) * r[t - 1] ** 2
    sde = np.empty(n_out)
    for k in range(n_out):
        t = N_START + k
        s2 = EWMA_LAMBDA * s2 + (1 - EWMA_LAMBDA) * r[t - 1] ** 2
        sde[k] = math.sqrt(s2)
    F["EWMA"] = dict(kind="norm", mu=np.zeros(n_out), sd=sde)
    # GARCH
    for dist, name in (("normal", "GARCH-n"), ("t", "GARCH-t")):
        lg, fl = [], []
        mu, sd, nu = roll_garch(r, dist, window, garch_refit, lg, fl)
        logs[name] = lg
        logs.setdefault("fails", {})[name] = len(fl)
        F[name] = dict(kind="norm", mu=mu, sd=sd) if dist == "normal" else dict(kind="t_std", mu=mu, sd=sd, nu=nu)
    # VaR / ES (return-form, negative)
    for m, f in F.items():
        f["VaR"], f["ES"] = {}, {}
        for a in ALPHAS:
            if f["kind"] == "norm":
                f["VaR"][a] = f["mu"] + f["sd"] * Z_N[a]; f["ES"][a] = f["mu"] + f["sd"] * ES_N[a]
            elif f["kind"] == "t_std":
                f["VaR"][a] = f["mu"] + f["sd"] * var_z_t(a, f["nu"]); f["ES"][a] = f["mu"] + f["sd"] * es_z_t(a, f["nu"])
            elif f["kind"] == "t_raw":
                f["VaR"][a] = f["loc"] + f["scale"] * stats.t.ppf(a, f["df"])
                f["ES"][a] = f["loc"] + f["scale"] * es_raw_t(a, f["df"])
            else:
                v = np.empty(n_out); e = np.empty(n_out)
                for k, y in enumerate(f["win"]):
                    q = np.quantile(y, a); v[k] = q; e[k] = y[y <= q].mean()
                f["VaR"][a], f["ES"][a] = v, e
    return F


def simulate(f, B, rng):
    """Simulate n×B out-of-sample returns under each model's predictive distribution
    (used for the AS-test p-values)."""
    if f["kind"] == "norm":
        n = len(f["mu"]); return f["mu"][:, None] + f["sd"][:, None] * rng.standard_normal((n, B))
    if f["kind"] == "t_std":
        nu = f["nu"][:, None]; n = len(nu)
        return f["mu"][:, None] + f["sd"][:, None] * rng.standard_t(np.repeat(nu, B, 1)) / np.sqrt(nu / (nu - 2))
    if f["kind"] == "t_raw":
        dfm = f["df"][:, None]; n = len(dfm)
        return f["loc"][:, None] + f["scale"][:, None] * rng.standard_t(np.repeat(dfm, B, 1))
    out = np.empty((len(f["win"]), B))
    for k, y in enumerate(f["win"]):
        out[k] = y[rng.integers(0, len(y), B)]
    return out


# ====================================================== Backtest statistics ====
def lr_uc(x, n, a):
    if x == 0:
        return -2.0 * n * math.log(1 - a)
    pi = x / n
    return -2.0 * ((n - x) * math.log(1 - a) + x * math.log(a) - (n - x) * math.log(1 - pi) - x * math.log(pi))


def lr_ind(br):
    b = br.astype(int)
    n00 = np.sum((b[:-1] == 0) & (b[1:] == 0)); n01 = np.sum((b[:-1] == 0) & (b[1:] == 1))
    n10 = np.sum((b[:-1] == 1) & (b[1:] == 0)); n11 = np.sum((b[:-1] == 1) & (b[1:] == 1))
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11); p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p11 == 0.0:
        ll1 = n00 * math.log(1 - p01) + n01 * math.log(p01)
    else:
        ll1 = n00 * math.log(1 - p01) + n01 * math.log(p01) + n10 * math.log(1 - p11) + n11 * math.log(p11)
    ll0 = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    return max(0.0, -2.0 * (ll0 - ll1))


def dq_test(r, v, a, lags=DQ_LAGS):
    """Engle & Manganelli (2004) dynamic quantile test: regress Hit_t on a constant, 1..lags
    lags of Hit, and VaR_t."""
    hit = (r < v).astype(float) - a
    n = len(hit)
    X = np.column_stack([np.ones(n - lags)] + [hit[lags - j:n - j] for j in range(1, lags + 1)] + [v[lags:]])
    y = hit[lags:]
    XtX = X.T @ X
    if np.linalg.matrix_rank(XtX) < X.shape[1]:
        return np.nan, np.nan
    bhat = np.linalg.solve(XtX, X.T @ y)
    stat = float(bhat @ XtX @ bhat / (a * (1 - a)))
    return stat, float(stats.chi2.sf(stat, X.shape[1]))


def az(r, v, e, a):
    """Acerbi–Szekely Z1, Z2 (return-form VaR/ES). r may be n×B."""
    r = np.asarray(r, float)
    if r.ndim == 1:
        r = r[:, None]
    v = v[:, None]; e = e[:, None]
    br = r < v
    ratio = np.where(br, r / (-e), 0.0)
    nb = br.sum(0)
    z1 = np.where(nb > 0, ratio.sum(0) / np.maximum(nb, 1) + 1.0, np.nan)
    z2 = ratio.sum(0) / (r.shape[0] * a) + 1.0
    return z1, z2


def mcneil_frey(r, v, e, sd, rng, B=B_MF):
    br = r < v
    if br.sum() < 2:
        return np.nan, np.nan
    d = (r[br] - e[br]) / sd[br]
    m = d.mean(); dc = d - m
    bs = dc[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return float(m), float((bs <= m).mean())


def pinball(r, v, a):
    return (a - (r < v).astype(float)) * (r - v)


def fz0(r, v, e, a):
    """FZ0 loss (Patton, Ziegel & Chen 2019); with positive-loss convention
    L = −r, V = −VaR, E = −ES."""
    L, V, E = -r, -v, -e
    return (1.0 / (a * E)) * (L > V) * (L - V) + V / E + np.log(E) - 1.0


def nw_var(u, lag=HAC_LAG):
    u = u - u.mean(); n = len(u)
    s = u @ u / n
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / n
    return s


def dm_test(d, lag=HAC_LAG):
    """Unconditional forecast ability (the GW test with instrument h_t = 1, i.e. the
    HAC version of Diebold–Mariano). Returns the t statistic and two-sided p."""
    d = d[np.isfinite(d)]
    t = d.mean() / math.sqrt(nw_var(d, lag) / len(d))
    return float(t), float(2 * stats.norm.sf(abs(t)))


def gw_test(d, lag=HAC_LAG):
    """Giacomini–White conditional forecast ability: regress d_t on (1, d_{t−1}), Wald
    test, χ²(2)."""
    # pair (d_t, d_{t-1}) first, then drop pairs with a missing value; dropping NaNs first would
    # treat two non-adjacent days as a lag
    y, x1 = d[1:], d[:-1]
    keep = np.isfinite(y) & np.isfinite(x1)
    y, x1 = y[keep], x1[keep]
    X = np.column_stack([np.ones(len(y)), x1])
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    u = y - X @ b
    XtXi = np.linalg.pinv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lag + 1)) * (A + A.T)
    V = XtXi @ S @ XtXi
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, float(stats.chi2.sf(stat, 2))


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


# ====================================================== Evaluate one design ====
def evaluate(P, FC, rngs, detail=True):
    """Backtest each portfolio × confidence level × model; run DM/GW for the 15 model
    pairs; run the MCS."""
    out = {}
    for nm, r in P.items():
        ro = r[N_START:]
        F = FC[nm]
        for a in ALPHAS:
            res = {}
            loss_fz, loss_pb = {}, {}
            for m in MODELS:
                f = F[m]; v = f["VaR"][a]; e = f["ES"][a]
                br = ro < v; x = int(br.sum()); n = len(ro)
                uc = lr_uc(x, n, a); ind = lr_ind(br)
                dqs, dqp = dq_test(ro, v, a)
                z1, z2 = (float(z[0]) for z in az(ro, v, e, a))
                d = dict(x=x, n=n, rate=x / n, uc_p=float(stats.chi2.sf(uc, 1)), ind_p=float(stats.chi2.sf(ind, 1)),
                         cc_p=float(stats.chi2.sf(uc + ind, 2)), dq=dqs, dq_p=dqp, z1=z1, z2=z2,
                         fz0=float(np.mean(fz0(ro, v, e, a))), pinball=float(np.mean(pinball(ro, v, a))),
                         mean_es=float(np.mean(-e)), mean_var=float(np.mean(-v)))
                if detail:
                    sims = simulate(f, B_AS, rngs["as"])
                    z1s, z2s = az(sims, v, e, a)
                    z1s = z1s[np.isfinite(z1s)]
                    d["z1_p"] = float((z1s <= z1).mean()) if np.isfinite(z1) else np.nan
                    d["z2_p"] = float((z2s <= z2).mean())
                    d["mf"], d["mf_p"] = mcneil_frey(ro, v, e, f["sd"], rngs["mf"])
                    # Monte Carlo p-value for DQ: the asymptotic χ² at the 1% level clearly
                    # over-rejects with 357 observations (see self-test),
                    # so we simulate the null distribution of the DQ statistic under the
                    # model's own predictive distribution.
                    dq_sim = np.array([dq_test(sims[:, b], v, a)[0] for b in range(sims.shape[1])])
                    dq_sim = dq_sim[np.isfinite(dq_sim)]
                    d["dq_p_sim"] = float((dq_sim >= dqs).mean()) if np.isfinite(dqs) else np.nan
                res[m] = d
                loss_fz[m] = fz0(ro, v, e, a); loss_pb[m] = pinball(ro, v, a)
            pairs = {}
            for i, m1 in enumerate(MODELS):
                for m2 in MODELS[i + 1:]:
                    q = {}
                    for lname, L in (("fz0", loss_fz), ("pinball", loss_pb)):
                        dd = L[m1] - L[m2]
                        t, p = dm_test(dd); g, gp = gw_test(dd)
                        q[lname] = dict(mean=float(dd.mean()), dm_t=t, dm_p=p, gw=g, gw_p=gp)
                    pairs[f"{m1}|{m2}"] = q
            mcs = {}
            if detail:
                for lname, L in (("fz0", loss_fz), ("pinball", loss_pb)):
                    df_ = pd.DataFrame({m: L[m] for m in MODELS})
                    mc = MCS(df_, size=MCS_SIZE, reps=B_MCS, block_size=MCS_BLOCK, method="R",
                             bootstrap="stationary", seed=int(rngs["mcs"].integers(1, 2 ** 31 - 1)))
                    mc.compute()
                    pv = mc.pvalues.iloc[:, 0].to_dict()
                    mcs[lname] = dict(included=list(mc.included), pvalues={k: float(v) for k, v in pv.items()})
            out[(nm, a)] = dict(res=res, pairs=pairs, mcs=mcs)
    return out


def oos_ratio(FC, P, rng, num="IBIT portfolio", den="GLD portfolio", a=0.05):
    """Out-of-sample ES ratio: the ratio of average ES forecasts; stationary block bootstrap
    interval (jointly resampling the 357 forecast days)."""
    n = len(P[num]) - N_START
    idx = stationary_boot(n, B_RATIO, L_RATIO, rng)
    out = {}
    for m in MODELS:
        ei = -FC[num][m]["ES"][a]; eg = -FC[den][m]["ES"][a]
        pt = float(ei.mean() / eg.mean())
        bs = ei[idx].mean(0) / eg[idx].mean(0)
        out[m] = dict(ratio=pt, ci=[float(x) for x in np.percentile(bs, [2.5, 97.5])],
                      median_daily=float(np.median(ei / eg)))
    ri = P[num][N_START:]; rg = P[den][N_START:]

    def hs_es(x):
        q = np.quantile(x, a); return -x[x <= q].mean()
    bsr = np.array([hs_es(ri[i]) / hs_es(rg[i]) for i in idx.T])
    out["realized_HS"] = dict(ratio=float(hs_es(ri) / hs_es(rg)), ci=[float(x) for x in np.percentile(bsr, [2.5, 97.5])])
    return out


# ====================================================== Full-sample GARCH ====
def ljung_box(z, lags=20):
    z = z - z.mean(); n = len(z)
    ac = np.array([z[k:] @ z[:-k] for k in range(1, lags + 1)]) / (z @ z)
    q = n * (n + 2) * np.sum(ac ** 2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))


def full_sample_params(series):
    rows = []
    out = {}
    for nm, r in series.items():
        for dist in ("t", "normal"):
            res = fit_arch(r * 100, dist)
            p, se = res.params, res.std_err
            al, be = p["alpha[1]"], p["beta[1]"]
            pers = al + be
            z = np.asarray(res.std_resid, float); z = z[np.isfinite(z)]
            q1, p1 = ljung_box(z); q2, p2 = ljung_box(z ** 2)
            o = dict(mu=float(p["mu"]), mu_se=float(se["mu"]), omega=float(p["omega"]), omega_se=float(se["omega"]),
                     alpha=float(al), alpha_se=float(se["alpha[1]"]), beta=float(be), beta_se=float(se["beta[1]"]),
                     nu=float(p["nu"]) if dist == "t" else None, nu_se=float(se["nu"]) if dist == "t" else None,
                     persistence=float(pers), half_life=float(math.log(0.5) / math.log(pers)) if 0 < pers < 1 else None,
                     uncond_vol=float(math.sqrt(p["omega"] / (1 - pers) * 252)) if pers < 1 else None,
                     llf=float(res.loglikelihood), aic=float(res.aic), bic=float(res.bic),
                     lb_z=[q1, p1], lb_z2=[q2, p2], n=int(len(r)))
            out[f"{nm}|{dist}"] = o
            nu_txt = f"{o['nu']:.2f} ({o['nu_se']:.2f})" if dist == "t" else "—"
            rows.append([nm, "t" if dist == "t" else "Normal", f"{o['mu']:.4f} ({o['mu_se']:.4f})",
                         f"{o['omega']:.4f} ({o['omega_se']:.4f})", f"{o['alpha']:.3f} ({o['alpha_se']:.3f})",
                         f"{o['beta']:.3f} ({o['beta_se']:.3f})", nu_txt, f"{pers:.3f}",
                         f"{o['half_life']:.1f}" if o["half_life"] else "—",
                         f"{o['uncond_vol']:.1f}" if o["uncond_vol"] else "—", f"{o['llf']:.1f}",
                         f"{pf(p1)} / {pf(p2)}"])
    return out, rows


# ============================================================== Self-test ====
def selftest():
    ok = True
    worst = 0.0
    for nu in (4.0, 8.0, 30.0):
        for a in ALPHAS:
            num = integrate.quad(lambda p: var_z_t(p, nu), 0, a)[0] / a
            worst = max(worst, abs(num - es_z_t(a, nu)))
            num2 = integrate.quad(lambda p: stats.t.ppf(p, nu), 0, a)[0] / a
            worst = max(worst, abs(num2 - es_raw_t(a, nu)))
    ok &= check("t-distribution ES closed forms (standardized and unstandardized) vs numerical integration", worst < 1e-7, f"{worst:.1e}")
    rng = np.random.default_rng(0)
    rej = 0
    for _ in range(400):
        rr = rng.standard_normal(357)
        _, p = dq_test(rr, np.full(357, Z_N[0.05]) + rng.normal(0, 1e-3, 357), 0.05)
        rej += p < 0.05
    ok &= check("DQ test rejection rate under the correct model is about 5% (5% level)", 0.01 <= rej / 400 <= 0.10, f"{rej / 400:.3f}")
    rej = nn_ = 0
    for _ in range(1000):
        rr = rng.standard_normal(357)
        _, p = dq_test(rr, np.full(357, Z_N[0.01]) + rng.normal(0, 1e-3, 357), 0.01)
        if np.isfinite(p):
            rej += p < 0.05; nn_ += 1
    w(f"   INFO  Actual rejection rate of the asymptotic DQ test at the 1% level with 357 "
      f"observations: {rej / nn_:.3f} (nominal 0.05)"
      "— over-rejects, so the table below also reports simulated p-values")
    KEY["dq_size_1pct"] = rej / nn_
    rej = 0
    for _ in range(300):
        _, p = dm_test(rng.standard_normal(357)); rej += p < 0.05
    ok &= check("DM (HAC) test nominal level about 5%", 0.01 <= rej / 300 <= 0.10, f"{rej / 300:.3f}")
    L = pd.DataFrame({"a": rng.standard_normal(357) + 1, "b": rng.standard_normal(357) + 1,
                      "c": rng.standard_normal(357) + 3})
    mc = MCS(L, size=0.1, reps=500, block_size=5, seed=1); mc.compute()
    ok &= check("MCS: a clearly worse model is excluded, equivalent models are retained", "c" not in mc.included and {"a", "b"} <= set(mc.included),
                f"included={list(mc.included)}")
    return ok


# ============================================================== Main program ====
def find_data():
    c = []
    for base in (HERE, os.getcwd()):
        for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv")):
            if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p)):
                c.append(os.path.abspath(p))
    c = sorted(set(c), key=os.path.basename)
    return c[-1] if c else None


def table_backtest(ev, P, a, detail=True):
    rows = []
    for nm in P:
        for m in MODELS:
            d = ev[(nm, a)]["res"][m]
            row = [nm, MODEL_EN[m], f"{d['x']} ({d['rate'] * 100:.2f}%)", pf(d["uc_p"]), pf(d["cc_p"]),
                   pf(d["dq_p"]) + (f" / {pf(d['dq_p_sim'])}" if detail and np.isfinite(d.get("dq_p_sim", np.nan)) else ""),
                   f"{d['z1']:+.3f}" if np.isfinite(d["z1"]) else "—"]
            if detail:
                row += [pf(d["z1_p"]) if np.isfinite(d.get("z1_p", np.nan)) else "—", f"{d['z2']:+.3f}", pf(d["z2_p"]),
                        pf(d["mf_p"]) if np.isfinite(d.get("mf_p", np.nan)) else "—"]
            row += [f"{d['fz0']:.4f}"]
            rows.append(row)
    head = ["Portfolio", "Model", "Breaches", "UC p", "CC p", "DQ p (asymptotic / simulated)" if detail else "DQ p", "Z1"]
    if detail:
        head += ["Z1 p", "Z2", "Z2 p", "MF p"]
    head += ["Mean FZ0"]
    return md(rows, head)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default="out/21")
    ap.add_argument("--quick", action="store_true", help="skip the GARCH daily-refit robustness")
    a = ap.parse_args()
    if a.data is None:
        a.data = find_data()
        if a.data is None:
            sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; please specify with --data.")
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(HERE, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    ss = np.random.SeedSequence(SEED)
    kids = ss.spawn(6)
    rngs = {"as": np.random.default_rng(kids[0]), "mf": np.random.default_rng(kids[1]),
            "mcs": np.random.default_rng(kids[2]), "ratio": np.random.default_rng(kids[3])}
    rngs_rob = {"as": np.random.default_rng(kids[4]), "mf": np.random.default_rng(kids[4]),
                "mcs": np.random.default_rng(kids[5]), "ratio": np.random.default_rng(kids[5])}

    w("# Forecast design for VaR/ES and forecast-accuracy tests (script 21)\n")
    ok_self = selftest()

    d = pd.read_csv(a.data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(d)
    # The vol-matched weight uses only the initial estimation window (first N_START days); the full-sample
    # weight would carry information from the forecast period. The full-sample value is reported for comparison.
    vm = (1 / 3) * d.GLD.iloc[:N_START].std() / d.IBIT.iloc[:N_START].std()
    vm_full = (1 / 3) * d.GLD.std() / d.IBIT.std()
    KEY["vol_matched_weight"] = dict(first_window=float(vm), full_sample=float(vm_full), n_days=N_START)
    spec = {"IBIT portfolio": {"IBIT": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3}, "GLD portfolio": {"GLD": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3},
            "IBIT vol-matched": {"IBIT": vm, "SPY": (1 - vm) / 2, "TLT": (1 - vm) / 2}, "60/40 benchmark": {"SPY": 0.6, "TLT": 0.4}}
    P = {nm: sum(d[k].values * v for k, v in wt.items()) for nm, wt in spec.items()}
    dates = d.Date.values
    oos0, oos1 = pd.Timestamp(dates[N_START]).date(), pd.Timestamp(dates[-1]).date()

    # ------------------------------------------------------------------ 0
    w("\n## 0. Setup overview (can be cited directly in the paper)\n")
    rows = [
        ["Data", f"{os.path.basename(a.data)}; {n} trading days, {pd.Timestamp(dates[0]).date()} → {pd.Timestamp(dates[-1]).date()}; simple daily returns"],
        ["In-sample (full-sample) estimation", "Table 2 and §3.3 of the paper: full-sample estimate over 607 days, descriptive; the GARCH-t column is the conditional one-step forecast at the end of the sample"],
        ["Out-of-sample forecast period", f"{oos0} → {oos1}, {n - N_START} one-step forecasts (days 251–{n})"],
        ["Main-design window", f"Moving window, length {N_START} trading days"],
        ["Robustness window", f"Expanding window (initial {N_START} days, growing daily to {n - 1} days)"],
        ["Refit frequency", f"GARCH re-estimated every {REFIT} trading days ({math.ceil((n - N_START) / REFIT)} times per portfolio), with the variance recursion advanced daily at fixed parameters in between; "
                   "robustness: daily refit. Static normal, static t, and historical simulation are re-estimated daily using the latest window; EWMA is a daily recursion (λ = 0.94, not estimated)"],
        ["Information set", "The forecast for day t uses only returns up to day t−1 and earlier"],
        ["Vol-matched portfolio", f"IBIT weight = (1/3)·sd(GLD)/sd(IBIT) from the first {N_START} days: {vm:.4f}, SPY and "
                                  f"TLT {(1 - vm) / 2:.4f} each; full sample (includes the forecast period): {vm_full:.4f}, not used"],
        ["Mean equation", "GARCH: constant mean; static normal: window sample mean; static t: MLE location parameter; EWMA: zero mean"],
        ["Variance equation", "GARCH(1,1): σ²_t = ω + α ε²_{t−1} + β σ²_{t−1}, no leverage term, no variance targeting"],
        ["Innovation distribution", "GARCH-t: unit-variance standardized Student-t, degrees of freedom ν estimated jointly with the other parameters by MLE at each refit (arch lower bound 2.05); "
                   "GARCH-n: standard normal; static t: location-scale t, ν estimated by MLE (restricted to 2.05–500)"],
        ["Estimation method", "MLE from arch 8.0 (Sheppard), on the percentage scale; standard errors are Bollerslev–Wooldridge robust standard errors"],
        ["ES formula", "Closed-form for normal and t (t uses the degrees of freedom day by day); historical simulation uses the mean of returns in the window not exceeding VaR"],
        ["Number of simulations", f"Acerbi–Szekely p-value {B_AS} draws (under each model's predictive distribution); McNeil–Frey {B_MF} bootstrap draws; "
                   f"MCS {B_MCS} stationary bootstrap draws (expected block length {MCS_BLOCK}); out-of-sample ES ratio {B_RATIO} draws (block length {L_RATIO})"],
        ["Random seed", f"{SEED} (numpy SeedSequence derives an independent substream for each task)"],
        ["HAC", f"DM and GW tests use Newey–West with lag {HAC_LAG}"],
    ]
    w(md(rows, ["Item", "Setting"]))
    KEY["design"] = {r[0]: r[1] for r in rows}

    # ------------------------------------------------------------------ A
    w("## A. Full-sample GARCH(1,1) parameters (robust standard errors in parentheses; returns in %; ω in %²)\n")
    series = dict(P)
    for k in ("IBIT", "GLD", "SPY", "TLT"):
        series[k] = d[k].values
    FP, rows = full_sample_params(series)
    w(md(rows, ["Series", "Innovation", "μ", "ω", "α", "β", "ν", "α+β", "Half-life (days)", "Uncond. ann. vol %", "Log-likelihood",
                "LB(20) z / z² p"]))
    KEY["full_params"] = FP
    # Compare against script 09 (the self-implemented MLE used in Table 8 and §3.4 of the paper)
    ref09 = {"IBIT portfolio": dict(alpha=0.069, beta=0.853, nu=9.65), "GLD portfolio": dict(alpha=0.097, beta=0.777, nu=5.32)}
    for nm, rf in ref09.items():
        o = FP[f"{nm}|t"]
        check(f"A. {nm} arch estimates are close to script 09's", abs(o["alpha"] - rf["alpha"]) < 0.02 and abs(o["beta"] - rf["beta"]) < 0.03,
              f"α {o['alpha']:.3f} vs {rf['alpha']}; β {o['beta']:.3f} vs {rf['beta']}; ν {o['nu']:.2f} vs {rf['nu']}")
    pd.DataFrame([dict(series=k.split("|")[0], dist=k.split("|")[1], **v) for k, v in FP.items()]).to_csv(
        os.path.join(a.outdir, "garch_params_full_sample.csv"), index=False)

    # ------------------------------------------------------------------ C Main design
    w("\n## B/C. Main design (moving window 250 days, GARCH refit every 20 days): out-of-sample forecasts\n")
    LOG = {}
    FC = {}
    for nm, r in P.items():
        LOG[nm] = {}
        FC[nm] = build_forecasts(r, "moving", REFIT, LOG[nm])
        w(f"  {nm}: done ({time.time() - t0:.0f}s)")
    n_fit = sum(len(LOG[nm][m]) for nm in P for m in ("GARCH-t", "GARCH-n"))
    n_fail = sum(LOG[nm]["fails"][m] for nm in P for m in ("GARCH-t", "GARCH-n"))
    n_exp = len(P) * 2 * math.ceil((n - N_START) / REFIT)
    KEY["fits"] = dict(fits=n_fit, fails=n_fail, expected=n_exp)
    w(f"\n  Rolling GARCH fits: {n_fit} successful, {n_fail} failed or not converged (previous parameters kept), "
      f"{n_exp} expected")
    check("C. Rolling fits counted automatically and all successful", n_fit == n_exp and n_fail == 0,
          f"{n_fit}/{n_exp} successful, {n_fail} failed")
    mx = max(RECURSION_DEV) if RECURSION_DEV else float("inf")
    check(f"C. Script recursion == arch conditional volatility (all {len(RECURSION_DEV)} fits)", mx < 1e-8, f"maxdev={mx:.2e}")
    # Summary of rolling re-estimation parameters
    w("\n### B. Rolling re-estimation parameters (full parameters for each refit in garch_params_rolling.csv)\n")
    rows, recs = [], []
    for nm in P:
        for m in ("GARCH-t", "GARCH-n"):
            lg = pd.DataFrame(LOG[nm][m])
            lg["persistence"] = lg.alpha + lg.beta
            for _, rr in lg.iterrows():
                recs.append(dict(portfolio=nm, model=m, **rr.to_dict()))

            def s(col, dd=3):
                x = lg[col]; return f"{x.median():.{dd}f} [{x.min():.{dd}f}, {x.max():.{dd}f}]"
            rows.append([nm, MODEL_EN[m], len(lg), s("omega", 4), s("alpha"), s("beta"), s("persistence"),
                         s("nu", 2) if m == "GARCH-t" else "—"])
    w(md(rows, ["Portfolio", "Model", "Refits", "ω median [min, max]", "α", "β", "α+β", "ν"]))
    rp = pd.DataFrame(recs); rp["pers"] = rp.alpha + rp.beta
    deg = rp.groupby(["portfolio", "model"]).apply(lambda x: pd.Series(dict(
        n=len(x), alpha0=int((x.alpha < 0.01).sum()), integ=int((x.pers > 0.99).sum()), low=int((x.pers < 0.5).sum()))))
    w("Degenerate-estimate counts (α < 0.01: ARCH effects not identified within the window; "
      "α+β > 0.99: near unit root; α+β < 0.5: almost no persistence):\n")
    w(md([[i[0], MODEL_EN[i[1]], r_.n, r_.alpha0, r_.integ, r_.low] for i, r_ in deg.iterrows()],
         ["Portfolio", "Model", "Refits", "α < 0.01", "α+β > 0.99", "α+β < 0.5"]))
    KEY["rolling_degenerate"] = {f"{i[0]}|{i[1]}": [int(v) for v in r_.values] for i, r_ in deg.iterrows()}
    pd.DataFrame(recs).to_csv(os.path.join(a.outdir, "garch_params_rolling.csv"), index=False)
    KEY["rolling_params"] = {f"{r_[0]}|{r_[1]}": r_[2:] for r_ in rows}
    # Static-t degrees of freedom
    for nm in ("IBIT portfolio", "GLD portfolio"):
        dfv = FC[nm]["Student-t"]["df"]
        w(f"{nm} static-t window MLE degrees of freedom: median {np.median(dfv):.2f}, range {dfv.min():.2f}–{dfv.max():.2f}")

    # ------------------------------------------------------------------ D/E
    ev = evaluate(P, FC, rngs, detail=True)
    for a_ in ALPHAS:
        w(f"\n## D. Backtests (main design, {int(a_ * 100)}% level, {n - N_START} forecasts; nominal breaches {a_ * (n - N_START):.1f})\n")
        w(table_backtest(ev, P, a_))
    w("UC: Kupiec; CC: Christoffersen conditional coverage; DQ: Engle–Manganelli dynamic quantile (4 lags + VaR); "
      "Z1, Z2: Acerbi–Szekely (one-sided left-tail p-value, simulated under each model's own predictive distribution); MF: McNeil–Frey; "
      "mean FZ0: smaller is better.\n")

    for a_ in ALPHAS:
        w(f"\n## E. Forecast-accuracy comparison ({int(a_ * 100)}% level)\n")
        for nm in ("IBIT portfolio", "GLD portfolio", "IBIT vol-matched", "60/40 benchmark"):
            E = ev[(nm, a_)]
            rows = []
            for key, q in E["pairs"].items():
                m1, m2 = key.split("|")
                fz, pb = q["fz0"], q["pinball"]
                better = m1 if fz["mean"] < 0 else m2
                rows.append([f"{MODEL_EN[m1]} vs {MODEL_EN[m2]}", f"{fz['mean']:+.4f}", MODEL_EN[better],
                             f"{fz['dm_t']:+.2f} ({pf(fz['dm_p'])})", f"{fz['gw']:.2f} ({pf(fz['gw_p'])})",
                             f"{pb['dm_t']:+.2f} ({pf(pb['dm_p'])})", f"{pb['gw']:.2f} ({pf(pb['gw_p'])})"])
            w(f"**{nm}**\n")
            w(md(rows, ["Comparison", "FZ0 mean difference", "FZ0 better", "DM t (p): FZ0", "GW χ²(2) (p): FZ0",
                        "DM t (p): pinball", "GW (p): pinball"]))
            for lname in ("fz0", "pinball"):
                mc = E["mcs"][lname]
                w(f"  90% model confidence set ({lname}): {', '.join(MODEL_EN[m] for m in mc['included'])}; "
                  f"MCS p-values " + ", ".join(f"{MODEL_EN[m]} {mc['pvalues'][m]:.3f}" for m in MODELS))
            w("")
    # Summary: models excluded by the MCS per portfolio × level, count of significant losses
    w("\n### E summary: number of times each model enters the 90% model confidence set (4 portfolios × 2 levels = 8 cases)\n")
    rows = []
    for m in MODELS:
        inc_fz = sum(m in ev[(nm, a_)]["mcs"]["fz0"]["included"] for nm in P for a_ in ALPHAS)
        inc_pb = sum(m in ev[(nm, a_)]["mcs"]["pinball"]["included"] for nm in P for a_ in ALPHAS)
        rank = np.mean([sorted(MODELS, key=lambda mm: ev[(nm, a_)]["res"][mm]["fz0"]).index(m) + 1
                        for nm in P for a_ in ALPHAS])
        lose = sum(1 for nm in P for a_ in ALPHAS for key, q in ev[(nm, a_)]["pairs"].items()
                   if m in key.split("|") and q["fz0"]["dm_p"] < 0.05
                   and ((key.split("|")[0] == m and q["fz0"]["mean"] > 0) or (key.split("|")[1] == m and q["fz0"]["mean"] < 0)))
        rows.append([MODEL_EN[m], f"{inc_fz}/8", f"{inc_pb}/8", f"{rank:.2f}", lose])
    w(md(rows, ["Model", "MCS(FZ0)", "MCS(pinball)", "FZ0 mean rank (1 = best)", "Times significantly beaten by DM(FZ0) at 5%"]))
    KEY["mcs_summary"] = rows

    # ------------------------------------------------------------------ F
    w("\n## F. Out-of-sample ES95 ratio (IBIT portfolio / GLD portfolio)\n")
    R5 = oos_ratio(FC, P, rngs["ratio"])
    rows = [[MODEL_EN[m], f"{R5[m]['ratio']:.2f} [{R5[m]['ci'][0]:.2f}, {R5[m]['ci'][1]:.2f}]", f"{R5[m]['median_daily']:.2f}"]
            for m in MODELS]
    rows.append(["Out-of-sample realized (357-day historical simulation)", f"{R5['realized_HS']['ratio']:.2f} [{R5['realized_HS']['ci'][0]:.2f}, "
                 f"{R5['realized_HS']['ci'][1]:.2f}]", "—"])
    w(md(rows, ["Model", "Ratio of average ES forecasts [95% CI]", "Median daily ratio"]))
    w("For comparison: the paper's main result is the in-sample full-sample historical-simulation ES95 ratio of 1.63 [1.36, 1.99].\n")
    KEY["oos_ratio"] = R5
    check("F. All models' out-of-sample ES ratios > 1", all(R5[m]["ratio"] > 1 for m in MODELS),
          f"{min(R5[m]['ratio'] for m in MODELS):.2f}–{max(R5[m]['ratio'] for m in MODELS):.2f}")

    # ------------------------------------------------------------------ H Reconciliation
    w("\n## H. Reconciliation with script 04 (Table 3 of the paper)\n")
    t3 = {("IBIT portfolio", 0.05, "GARCH-t"): 20, ("GLD portfolio", 0.05, "GARCH-t"): 22, ("IBIT portfolio", 0.01, "GARCH-t"): 6,
          ("GLD portfolio", 0.01, "GARCH-t"): 9, ("IBIT portfolio", 0.05, "HS"): 24, ("GLD portfolio", 0.05, "HS"): 25,
          ("GLD portfolio", 0.01, "GARCH-n"): 10, ("60/40 benchmark", 0.05, "EWMA"): 18,
          ("IBIT vol-matched", 0.01, "GARCH-t"): 8, ("IBIT vol-matched", 0.01, "GARCH-n"): 8,
          ("IBIT vol-matched", 0.05, "GARCH-t"): 22, ("IBIT vol-matched", 0.05, "EWMA"): 20}
    mism = [(k, ev[(k[0], k[1])]["res"][k[2]]["x"], v) for k, v in t3.items() if ev[(k[0], k[1])]["res"][k[2]]["x"] != v]
    check(f"H. Breach counts match those in script 04 ({len(t3)} check points)", not mism, str(mism) if mism else "")
    g = ev[("IBIT portfolio", 0.05)]["pairs"]["GARCH-n|GARCH-t"]["fz0"]["gw"]
    check("H. GW(FZ0) IBIT portfolio 95% GARCH-t vs GARCH-n matches script 04 (14.57)", abs(g - 14.57) < 0.02, f"{g:.2f}")
    g = ev[("IBIT vol-matched", 0.05)]["pairs"]["EWMA|GARCH-t"]["fz0"]
    check("H. GW(FZ0) vol-matched 95% GARCH-t vs EWMA matches script 04 (9.72, EWMA better)",
          abs(g["gw"] - 9.72) < 0.02 and g["mean"] < 0, f"{g['gw']:.2f}, p = {g['gw_p']:.3f}")

    # ------------------------------------------------------------------ G Robustness
    ROB = {}
    designs = [("Expanding window, GARCH refit every 20 days", "expanding", REFIT)]
    if not a.quick:
        designs.append(("Moving window, GARCH refit daily", "moving", 1))
    for label, win, rf in designs:
        w(f"\n## G. Robustness: {label}\n")
        FCr = {}
        nfail_r = 0
        for nm, r in P.items():
            lg_r = {}
            FCr[nm] = build_forecasts(r, win, rf, lg_r)
            nfail_r += sum(lg_r["fails"].values())
        check(f"G. {label}: no failed GARCH fits", nfail_r == 0, f"{nfail_r} failed")
        evr = evaluate(P, FCr, rngs_rob, detail=True)
        rows = []
        for nm in ("IBIT portfolio", "GLD portfolio"):
            for a_ in ALPHAS:
                for m in MODELS:
                    dm_ = evr[(nm, a_)]["res"][m]; d0 = ev[(nm, a_)]["res"][m]
                    rows.append([nm, f"{int(a_ * 100)}%", MODEL_EN[m], f"{dm_['x']} (main design {d0['x']})", pf(dm_["uc_p"]),
                                 pf(dm_["cc_p"]), pf(dm_["dq_p"]), f"{dm_['fz0']:.4f} ({d0['fz0']:.4f})",
                                 "Yes" if m in evr[(nm, a_)]["mcs"]["fz0"]["included"] else "No"])
        w(md(rows, ["Portfolio", "Level", "Model", "Breaches", "UC p", "CC p", "DQ p", "Mean FZ0 (main design)", "In MCS(FZ0)"]))
        Rr = oos_ratio(FCr, P, rngs_rob["ratio"])
        w("Out-of-sample ES95 ratio: " + "; ".join(f"{MODEL_EN[m]} {Rr[m]['ratio']:.2f} [{Rr[m]['ci'][0]:.2f}, {Rr[m]['ci'][1]:.2f}]"
                                        for m in MODELS))
        ROB[label] = dict(res={f"{k[0]}|{k[1]}": {m: v["res"][m] for m in MODELS} for k, v in evr.items()},
                          mcs={f"{k[0]}|{k[1]}": v["mcs"] for k, v in evr.items()}, oos_ratio=Rr)
    KEY["robustness"] = ROB

    # ------------------------------------------------------------------ Outputs
    KEY["main"] = {f"{k[0]}|{k[1]}": dict(res=v["res"], pairs=v["pairs"], mcs=v["mcs"]) for k, v in ev.items()}
    w("\n## Self-test summary\n")
    for c in CHECKS:
        w("  " + c)
    nf = sum(c.startswith("[FAIL]") for c in CHECKS)
    w(f"\n{len(CHECKS)} checks, {nf} failures. Elapsed {time.time() - t0:.0f} seconds.")
    with open(os.path.join(a.outdir, "forecast_design.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    with open(os.path.join(a.outdir, "forecast_design.json"), "w", encoding="utf8") as fh:
        json.dump(clean(KEY), fh, indent=1, ensure_ascii=False)
    return 0 if (nf == 0 and ok_self) else 1


if __name__ == "__main__":
    sys.exit(main())