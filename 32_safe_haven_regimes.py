#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
32_safe_haven_regimes.py — Hedge and safe-haven regressions on the long samples, by market regime,
with a VIX threshold model and an episode study of equity drawdowns since 2000
(second-round revision: Reviewer 2, comments 6, 7 and 8; Reviewer 1, comment 4)

Definitions (Baur and Lucey 2010; Baur and McDermott 2010). An asset is a hedge if it is
uncorrelated (weak) or negatively correlated (strong) with equities on average; a safe haven if it
is uncorrelated (weak) or negatively correlated (strong) with equities in periods of extreme equity
losses; a diversifier if it is positively but not perfectly correlated on average.

A  Baur–McDermott regressions:  r_X = a + b0 r_S + b1 r_S D10 + b2 r_S D5 + e (Equation (2) of the paper),
   D_q = 1{r_S <= q-th quantile of r_S}; total coefficients at the 10% and 5% quantiles, and at the 1%
   quantile from the same regression with an added indicator b3 r_S D1;
   Newey–West standard errors (5 lags). r_S = SPY. One-sided p-values for H0: total <= 0 (a small
   p-value rejects the weak safe-haven property).
B  Regime-switching safe-haven regressions with the three SPY regimes of script 31
   (tranquil / turbulent / crisis): r_X = sum_s 1{S_t = s} (a_s + b_s r_S) + e, and the same with the
   smoothed regime probabilities in place of the indicators (no hard classification).
C  Markov-switching regression of r_X on r_S with two regimes in the intercept, slope and variance
   estimated from the asset's own relation with equities; slope by regime and the share of
   SPY-crisis days falling in the high-variance regime.
D  Hansen (2000) threshold regression on the previous day's VIX:
   r_X = (a1 + b1 r_S) 1{VIX_{t-1} <= g} + (a2 + b2 r_S) 1{VIX_{t-1} > g} + e,
   g on a grid between the 15th and 85th percentiles; heteroskedasticity-robust sup-F test of
   linearity with Hansen's (1996) fixed-regressor bootstrap; 95% confidence set for g from the
   likelihood-ratio statistic (critical value 7.35).
E  Episodes: every SPY decline of at least 10% from peak to trough (total-return prices; a trough is
   confirmed by a 20% rise, the conventional end of a bear market) in the gold long sample (from August
   2000) and in the Bitcoin long sample; cumulative return, correlation and beta of gold, Treasuries
   and Bitcoin from peak to trough.
F  The primary dependence measures of script 22 (quantile-regression slope at the 0.05 quantile and rank-based
   lambda_L(5%), with block-bootstrap intervals) for gold and Treasuries on the gold long sample.

Samples: Bitcoin long sample (BTC-USD, GLD, TLT, SPY, VIX; Sep 2014 - Jun 2026) and gold long
sample (gold futures then GLD, TLT, SPY, VIX; Aug 2000 - Jun 2026). The ETF sample is included in
A and B for reference. Requires out/31/regimes.csv (run script 31 first; it is run automatically
if missing).

Usage
    python 32_safe_haven_regimes.py               # writes out/32/
    python 32_safe_haven_regimes.py --boot 300    # quicker trial
"""

from __future__ import annotations

import os

# single-threaded linear algebra: the Markov-switching likelihood is then evaluated in the same order on every
# machine, so that the regime classification is reproducible to the last day
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261023
HAC = 5
QS = (0.10, 0.05, 0.01)
REG = ["tranquil", "turbulent", "crisis"]
# Names are attached only to well-documented episodes (window of peak and trough months); any other
# drawdown is labelled by its dates.
NAMES = [("2000-09", "2002-10", "Dot-com bust and 11 September"),
         ("2007-10", "2009-03", "Global financial crisis"),
         ("2010-04", "2010-07", "Flash crash and euro-area debt crisis"),
         ("2011-04", "2011-10", "Euro-area debt crisis and US downgrade"),
         ("2015-05", "2016-02", "China devaluation and oil-price collapse"),
         ("2018-01", "2018-02", "February 2018 volatility shock"),
         ("2018-09", "2018-12", "Q4 2018 monetary tightening"),
         ("2020-02", "2020-03", "COVID-19 crash"),
         ("2022-01", "2022-10", "Inflation, rate hikes and energy crisis"),
         ("2025-02", "2025-04", "2025 tariff shock")]


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def nw_ols(y, X, lag=HAC):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = y - X @ b
    XtXi = np.linalg.inv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += (1 - l / (lag + 1)) * (A + A.T)
    return b, XtXi @ S @ XtXi, u


def lin_comb(b, V, c):
    c = np.asarray(c, float)
    est = float(c @ b); se = float(math.sqrt(c @ V @ c))
    return est, se


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ------------------------------------------------------------------ A
def baur_mcdermott(x, s):
    """Equation (2) with indicators for the 10% and 5% quantiles gives the average slope b0 and the totals at
    the 10% and 5% quantiles; adding an indicator for the 1% quantile gives the total on the 1% worst days (in
    the three-indicator model the 5% total would refer to days between the 1% and 5% quantiles only)."""
    qv = {q: np.quantile(s, q) for q in QS}
    out = {}
    X2 = np.column_stack([np.ones_like(s), s, s * (s <= qv[0.10]), s * (s <= qv[0.05])])
    b, V, _ = nw_ols(x, X2)
    out["b0"] = lin_comb(b, V, [0, 1, 0, 0])
    out["q10"] = lin_comb(b, V, [0, 1, 1, 0])
    out["q5"] = lin_comb(b, V, [0, 1, 1, 1])
    X3 = np.column_stack([X2, s * (s <= qv[0.01])])
    b3, V3, _ = nw_ols(x, X3)
    out["q1"] = lin_comb(b3, V3, [0, 1, 1, 1, 1])
    res = {}
    for k, (est, se) in out.items():
        t = est / se
        res[k] = dict(est=est, se=se, t=t, p_pos=float(stats.norm.sf(t)), p_neg=float(stats.norm.cdf(t)),
                      p_two=float(2 * stats.norm.sf(abs(t))))
    return res


def bm_label(r):
    def cls(d, kind):
        if d["p_neg"] < 0.05:
            return f"strong {kind}"
        if d["p_pos"] >= 0.05:
            return f"weak {kind}"
        return "diversifier" if kind == "hedge" else "no safe haven"
    return cls(r["b0"], "hedge"), cls(r["q1"], "safe haven")


# ------------------------------------------------------------------ B
def regime_reg(x, s, lab, P=None):
    """Regime-specific intercepts and slopes; lab = regime index (hard) or P = n x 3 probabilities."""
    W = P if P is not None else np.column_stack([(lab == i).astype(float) for i in range(3)])
    X = np.column_stack([W, W * s[:, None]])
    keep = W.sum(0) > 0
    cols = np.r_[keep, keep]
    b, V, _ = nw_ols(x, X[:, cols])
    names = [f"a_{REG[i]}" for i in range(3) if keep[i]] + [f"b_{REG[i]}" for i in range(3) if keep[i]]
    out = {}
    for j, nm in enumerate(names):
        se = math.sqrt(V[j, j]); t = b[j] / se
        out[nm] = dict(est=float(b[j]), se=se, p_pos=float(stats.norm.sf(t)), p_neg=float(stats.norm.cdf(t)))
    # equality of slopes crisis vs tranquil
    if "b_crisis" in names and "b_tranquil" in names:
        c = np.zeros(len(names)); c[names.index("b_crisis")] = 1; c[names.index("b_tranquil")] = -1
        est, se = lin_comb(b, V, c)
        out["crisis_minus_tranquil"] = dict(est=est, se=se, p_two=float(2 * stats.norm.sf(abs(est / se))))
    return out


# ------------------------------------------------------------------ C
def ms_regression(x, s, seed=SEED):
    np.random.seed(seed)
    mod = MarkovRegression(x * 100, k_regimes=2, trend="c", exog=s * 100, switching_variance=True)
    best = None
    for em in (50, 200):
        try:
            r = mod.fit(search_reps=20, em_iter=em, maxiter=1000, disp=False)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.llf > best.llf:
            best = r
    par = dict(zip(best.model.param_names, np.asarray(best.params)))
    bse = dict(zip(best.model.param_names, np.asarray(best.bse)))
    sig = np.array([par["sigma2[0]"], par["sigma2[1]"]])
    order = np.argsort(sig)
    out = {}
    for lab, i in zip(("low-variance", "high-variance"), order):
        out[lab] = dict(beta=float(par[f"x1[{i}]"]), se=float(bse[f"x1[{i}]"]),
                        vol=float(math.sqrt(sig[i]) * math.sqrt(252)),
                        share=float((np.asarray(best.smoothed_marginal_probabilities).argmax(1) == i).mean()))
    hv = np.asarray(best.smoothed_marginal_probabilities)[:, order[1]]
    return out, hv, float(best.llf)


# ------------------------------------------------------------------ D
def threshold(x, s, v, B, rng, trim=0.15):
    ok = np.isfinite(x) & np.isfinite(s) & np.isfinite(v)
    x, s, v = x[ok], s[ok], v[ok]
    n = len(x)
    grid = np.unique(np.quantile(v, np.linspace(trim, 1 - trim, 141)))
    X0 = np.column_stack([np.ones(n), s])
    b0 = np.linalg.lstsq(X0, x, rcond=None)[0]
    e0 = x - X0 @ b0
    ssr0 = e0 @ e0

    def ssr_alt(Y):
        """SSR of the threshold model at each grid value for every column of Y (n x m)."""
        out = np.empty((len(grid), Y.shape[1]))
        for k, g in enumerate(grid):
            h = (v > g).astype(float)
            X = np.column_stack([1 - h, s * (1 - h), h, s * h])
            Q, _ = np.linalg.qr(X)
            R = Y - Q @ (Q.T @ Y)
            out[k] = (R * R).sum(0)
        return out

    S1 = ssr_alt(x[:, None])[:, 0]
    k_hat = int(np.argmin(S1))
    g_hat = float(grid[k_hat])
    F = n * (ssr0 - S1) / S1
    supF = float(F.max())
    # Hansen (1996) fixed-regressor bootstrap, heteroskedasticity-robust: y* = e0 * N(0,1)
    sup_b = []
    for c0 in range(0, B, 100):
        m = min(100, B - c0)
        Y = e0[:, None] * rng.standard_normal((n, m))
        Q0, _ = np.linalg.qr(X0)
        R0 = Y - Q0 @ (Q0.T @ Y)
        s0 = (R0 * R0).sum(0)
        s1 = ssr_alt(Y)
        sup_b.append((n * (s0[None, :] - s1) / s1).max(0))
    sup_b = np.concatenate(sup_b)
    LR = n * (S1 - S1[k_hat]) / S1[k_hat]
    cs = grid[LR <= 7.35]
    h = (v > g_hat).astype(float)
    X = np.column_stack([1 - h, s * (1 - h), h, s * h])
    b, V, _ = nw_ols(x, X)
    se = np.sqrt(np.diag(V))
    dlt = lin_comb(b, V, [0, -1, 0, 1])
    return dict(n=n, gamma=g_hat, gamma_cs=[float(cs.min()), float(cs.max())], supF=supF,
                p_boot=float((sup_b >= supF).mean()), n_high=int(h.sum()), n_low=int(n - h.sum()),
                b_low=float(b[1]), se_low=float(se[1]), b_high=float(b[3]), se_high=float(se[3]),
                diff=dlt[0], diff_se=dlt[1], diff_p=float(2 * stats.norm.sf(abs(dlt[0] / dlt[1]))),
                share_grid_quantile=float((v <= g_hat).mean()))


# ------------------------------------------------------------------ E
def drawdown_episodes(r, dates, depth=0.10, rebound=0.20):
    """Peak-to-trough declines of at least `depth` on the total-return price of SPY. A peak is confirmed by a
    subsequent fall of `depth`; a trough by a subsequent rise of `rebound` (20%, the conventional end of a bear
    market) or by a new high above the episode's peak, so that declines separated by such a rally (for example 2010 and 2011) are separate episodes,
    while bear-market rallies within an episode do not split it."""
    p = np.cumprod(1 + np.asarray(r, float))
    eps = []
    peak_i, trough_i, down = 0, 0, False
    for t in range(1, len(p)):
        if not down:
            if p[t] > p[peak_i]:
                peak_i = t
            elif p[t] <= p[peak_i] * (1 - depth):
                down, trough_i = True, t
        else:
            if p[t] < p[trough_i]:
                trough_i = t
            elif p[t] >= p[trough_i] * (1 + rebound) or p[t] > p[peak_i]:
                eps.append((peak_i, trough_i))      # a 20% rally or a new high ends the episode
                down, peak_i = False, t
    if down:
        eps.append((peak_i, trough_i))
    return [dict(peak=str(dates[pk].date()), trough=str(dates[tr].date()), depth=float(p[tr] / p[pk] - 1),
                 i0=pk + 1, i1=tr) for pk, tr in eps]


def name_of(peak, trough):
    """Name of a well-documented episode whose window contains both the peak and the trough month."""
    for a, b, nm in NAMES:
        if a <= peak[:7] <= b and a <= trough[:7] <= b:
            return nm
    return f"Drawdown {peak[:7]} to {trough[:7]}"


def episode_table(D, assets, label):
    eps = drawdown_episodes(D.SPY.values, D.index)
    rows, out = [], []
    for e in eps:
        sl = slice(e["i0"], e["i1"] + 1)
        sub = D.iloc[sl]
        o = dict(e, name=name_of(e["peak"], e["trough"]), days=len(sub))
        cells = []
        for a in assets:
            x = sub[a].dropna()
            if len(x) < 5:
                o[a] = None; cells.append("—"); continue
            s = sub.SPY.reindex(x.index)
            cum = float(np.prod(1 + x.values) - 1)
            rho = float(np.corrcoef(x.values, s.values)[0, 1])
            beta = float(np.polyfit(s.values, x.values, 1)[0])
            o[a] = dict(cum=cum, rho=rho, beta=beta, n=len(x))
            cells.append(f"{cum * 100:+.1f} / {rho:+.2f}")
        out.append(o)
        rows.append([o["name"], f"{e['peak']} → {e['trough']}", o["days"], f"{e['depth'] * 100:.1f}"] + cells)
    return rows, out


# ------------------------------------------------------------------ F
def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def qr_slope(y, x, tau=0.05):
    import statsmodels.api as sm
    return float(sm.QuantReg(y, sm.add_constant(x)).fit(q=tau, max_iter=5000, p_tol=1e-8).params[1])


def lam_lower(x, y, u=0.05):
    U, V = stats.rankdata(x) / (len(x) + 1.0), stats.rankdata(y) / (len(y) + 1.0)
    return float(np.mean((U <= u) & (V <= u)) / u)


def primary_measures(x, s, rng, B_qr=500, B_lam=2000, L=5):
    """The primary measures of script 22 (quantile-regression slope at 0.05, rank-based lambda_L(5%)) with
    stationary block-bootstrap percentile intervals (expected block length 5)."""
    n = len(x)
    idx = stationary_boot(n, max(B_qr, B_lam), L, rng)
    q = qr_slope(x, s)
    qb = [qr_slope(x[i], s[i]) for i in idx[:, :B_qr].T]
    lam = lam_lower(x, s)
    lb = [lam_lower(x[i], s[i]) for i in idx[:, :B_lam].T]
    return dict(qr=q, qr_ci=list(np.percentile(qb, [2.5, 97.5])), lam=lam, lam_ci=list(np.percentile(lb, [2.5, 97.5])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/32")
    ap.add_argument("--regimes", default="out/31/regimes.csv")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if not os.path.exists(a.regimes):
        subprocess.run([sys.executable, os.path.join(HERE, "31_regimes_contagion.py")], check=True)
    reg = pd.read_csv(a.regimes, parse_dates=["Date"]).set_index("Date")
    rng = np.random.default_rng(SEED)
    E = LD.load_etf(); B = LD.load_btc_long(); G = LD.load_gold_long()
    samples = [("etf", "ETF sample", E, ["IBIT", "GLD", "TLT"]),
               ("btc_long", "Bitcoin long sample", B, ["BTC", "GLD", "TLT"])]
    if G is not None:
        samples.append(("gold_long", "Gold long sample", G, ["GOLD", "TLT"]))
    OUT = {}
    L = ["# Hedge and safe-haven regressions by regime, VIX threshold model and drawdown episodes (script 32)\n"]

    # ---------------- A
    rows = []
    for key, lab, D, assets in samples:
        for x in assets:
            d = D[[x, "SPY"]].dropna()
            r = baur_mcdermott(d[x].values, d.SPY.values)
            hl, sl = bm_label(r)
            OUT.setdefault(key, {}).setdefault("baur_mcdermott", {})[x] = dict(r, hedge=hl, safe_haven=sl, n=len(d))
            rows.append([lab, x, len(d)] + [f"{r[k]['est']:+.3f}{stars(r[k]['p_two'])} ({r[k]['se']:.3f})"
                                            for k in ("b0", "q10", "q5", "q1")] + [hl, sl])
    L.append("## A. Baur–McDermott regressions on SPY (total coefficients; Newey–West s.e.)\n")
    L.append(md(rows, ["Sample", "Asset", "Days", "Average (b0)", "10% quantile", "5% quantile", "1% quantile",
                       "Hedge", "Safe haven (1%)"]))
    L.append("Stars: two-sided significance at 10/5/1%. 'Weak' = not significantly different from zero; "
             "'strong' = significantly negative; 'no safe haven' = significantly positive at the 1% quantile "
             "(one-sided 5% test).\n")

    # ---------------- B
    rows = []
    for key, lab, D, assets in samples:
        R = reg.reindex(D.index)
        ok = R.regime.notna().values
        for x in assets:
            d = D.loc[ok, [x, "SPY"]].dropna()
            Rr = R.loc[d.index]
            lab_ = Rr.regime.values.astype(int)
            P = Rr[["p_tranquil", "p_turbulent", "p_crisis"]].values
            hard = regime_reg(d[x].values, d.SPY.values, lab_)
            soft = regime_reg(d[x].values, d.SPY.values, None, P)
            OUT[key].setdefault("regime_reg", {})[x] = dict(hard=hard, soft=soft,
                                                            days={REG[i]: int((lab_ == i).sum()) for i in range(3)})
            cells = []
            for g in REG:
                k = f"b_{g}"
                cells.append(f"{hard[k]['est']:+.3f}{stars(2 * min(hard[k]['p_pos'], hard[k]['p_neg']))} "
                             f"({hard[k]['se']:.3f}) [{int((lab_ == REG.index(g)).sum())}]" if k in hard else "—")
            dif = hard.get("crisis_minus_tranquil")
            cells.append(f"{dif['est']:+.3f} (p = {dif['p_two']:.3f})" if dif else "—")
            cells.append(f"{soft['b_crisis']['est']:+.3f} ({soft['b_crisis']['se']:.3f})" if "b_crisis" in soft else "—")
            rows.append([lab, x] + cells)
    L.append("## B. Regime-switching safe-haven regressions: slope on SPY by regime (script 31 regimes)\n")
    L.append(md(rows, ["Sample", "Asset", "Tranquil β [days]", "Turbulent β [days]", "Crisis β [days]",
                       "Crisis − tranquil", "Crisis β, probability-weighted"]))

    # ---------------- C
    rows = []
    for key, lab, D, assets in samples[1:]:
        R = reg.reindex(D.index)
        for x in assets:
            d = D[[x, "SPY"]].dropna()
            res, hv, llf = ms_regression(d[x].values, d.SPY.values)
            cr = (R.regime.reindex(d.index) == 2).values
            share = float((hv[cr] > 0.5).mean()) if cr.any() else float("nan")
            OUT[key].setdefault("ms_regression", {})[x] = dict(res, crisis_days_in_high_var=share, llf=llf)
            rows.append([lab, x] + [f"{res[k]['beta']:+.3f} ({res[k]['se']:.3f}); vol {res[k]['vol']:.1f}%; "
                                    f"{res[k]['share'] * 100:.0f}% of days" for k in ("low-variance", "high-variance")]
                        + [f"{share * 100:.0f}%"])
    L.append("## C. Two-regime Markov-switching regression of the asset on SPY (asset-specific regimes)\n")
    L.append(md(rows, ["Sample", "Asset", "Low-variance regime: β (s.e.); vol; share",
                       "High-variance regime: β (s.e.); vol; share", "SPY-crisis days in high-variance regime"]))

    # ---------------- D
    rows = []
    for key, lab, D, assets in samples[1:]:
        vlag = D.VIX.shift(1).values
        for x in assets:
            th = threshold(D[x].values, D.SPY.values, vlag, a.boot, rng)
            OUT[key].setdefault("threshold", {})[x] = th
            rows.append([lab, x, f"{th['gamma']:.1f} [{th['gamma_cs'][0]:.1f}, {th['gamma_cs'][1]:.1f}]",
                         f"{th['share_grid_quantile'] * 100:.0f}%", f"{th['n_low']} / {th['n_high']}",
                         f"{th['b_low']:+.3f} ({th['se_low']:.3f})", f"{th['b_high']:+.3f} ({th['se_high']:.3f})",
                         f"{th['diff']:+.3f} (p = {th['diff_p']:.3f})", f"{th['supF']:.1f} (p = {th['p_boot']:.3f})"])
    # D2: does the threshold reflect stress or a change over time? Slopes below and above the full-sample
    # threshold within sub-periods (the high-VIX days of the Bitcoin sample are concentrated in 2020-2022).
    rows2 = []
    for key, lab, D, assets in samples[1:]:
        cuts = (("2014–2019", "2014-01-01", "2019-12-31"), ("2020–2026", "2020-01-01", "2026-12-31")) \
            if key == "btc_long" else (("2000–2012", "2000-01-01", "2012-12-31"), ("2013–2026", "2013-01-01", "2026-12-31"))
        for x in assets:
            g = OUT[key]["threshold"][x]["gamma"]
            for nm, d0, d1 in cuts:
                d = D.loc[d0:d1]
                v = d.VIX.shift(1)
                ok = v.notna() & d[x].notna()
                h = (v[ok] > g).astype(float).values
                sx, y = d.SPY[ok].values, d[x][ok].values
                if h.sum() < 30 or (1 - h).sum() < 30:
                    continue
                Xm = np.column_stack([1 - h, sx * (1 - h), h, sx * h])
                b_, V_, _ = nw_ols(y, Xm)
                dd = lin_comb(b_, V_, [0, -1, 0, 1])
                OUT[key]["threshold"][x].setdefault("subperiods", {})[nm] = dict(
                    n_low=int((1 - h).sum()), n_high=int(h.sum()), b_low=float(b_[1]), se_low=float(np.sqrt(V_[1, 1])),
                    b_high=float(b_[3]), se_high=float(np.sqrt(V_[3, 3])), diff=dd[0], diff_p=float(2 * stats.norm.sf(abs(dd[0] / dd[1]))))
                q = OUT[key]["threshold"][x]["subperiods"][nm]
                rows2.append([lab, x, nm, f"{g:.1f}", f"{q['n_low']} / {q['n_high']}", f"{q['b_low']:+.3f} ({q['se_low']:.3f})",
                              f"{q['b_high']:+.3f} ({q['se_high']:.3f})", f"{q['diff']:+.3f} (p = {q['diff_p']:.3f})"])
    L.append(f"## D. Threshold regression on the previous day's VIX (Hansen 2000; bootstrap B = {a.boot})\n")
    L.append(md(rows, ["Sample", "Asset", "Threshold VIX [95% set]", "VIX percentile", "Days low / high",
                       "β, low VIX", "β, high VIX", "High − low", "sup-F (bootstrap p)"]))
    L.append("Slopes below and above the full-sample threshold within sub-periods (Newey–West s.e.):\n")
    L.append(md(rows2, ["Sample", "Asset", "Period", "Threshold", "Days low / high", "β, low VIX", "β, high VIX",
                        "High − low"]))

    # ---------------- F: primary dependence measures on the gold long sample (script 22 covers the other two)
    if G is not None:
        rows = []
        for x in ("GOLD", "TLT"):
            d = G[[x, "SPY"]].dropna()
            pm = primary_measures(d[x].values, d.SPY.values, rng)
            OUT["gold_long"].setdefault("primary", {})[x] = pm
            cr = OUT["gold_long"]["regime_reg"][x]["hard"].get("b_crisis")
            rows.append([x, len(d), f"{pm['qr']:+.3f} [{pm['qr_ci'][0]:+.3f}, {pm['qr_ci'][1]:+.3f}]",
                         f"{pm['lam']:.3f} [{pm['lam_ci'][0]:.3f}, {pm['lam_ci'][1]:.3f}]",
                         f"{cr['est']:+.3f} ({cr['se']:.3f})" if cr else "—"])
        L.append("## F. Gold long sample: primary dependence measures (stationary block bootstrap, block length 5)\n")
        L.append(md(rows, ["Asset", "Days", "Quantile-regression slope at 0.05 [95% CI]", "λ_L(5%) [95% CI] (independence 0.05)",
                           "Crisis-regime slope (s.e.)"]))

    # ---------------- E
    for key, lab, D, assets in samples[1:]:
        cols = [c for c in (["GOLD", "TLT", "BTC"] if key == "gold_long" else ["BTC", "GLD", "TLT"]) if c in D or c == "BTC"]
        DD = D.copy()
        if key == "gold_long":
            DD["BTC"] = B.BTC.reindex(DD.index)
        rows, out = episode_table(DD, cols, lab)
        OUT[key]["episodes"] = out
        L.append(f"## E. {lab}: SPY drawdowns of 10% or more (peak to trough)\n")
        L.append("Cells: cumulative return from peak to trough, % / daily correlation with SPY.\n")
        L.append(md(rows, ["Episode", "Peak → trough", "Days", "SPY %"] + cols))
    if "etf" in OUT:
        rowsE, outE = episode_table(E, ["IBIT", "GLD", "TLT"], "ETF sample")
        OUT["etf"]["episodes"] = outE
        L.append("## E. ETF sample: SPY drawdowns of 10% or more\n")
        L.append(md(rowsE, ["Episode", "Peak → trough", "Days", "SPY %", "IBIT", "GLD", "TLT"]))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.outdir, "safe_haven_regimes.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "safe_haven_regimes.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
