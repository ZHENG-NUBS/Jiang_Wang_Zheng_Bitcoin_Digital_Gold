#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_tail_diagnostics.py —

  (a) Skewness and excess kurtosis of standardized residuals
  (b) QQ plots
  (c) Ljung–Box test on squared standardized residuals
  (d) Hill tail-index estimation
Plus a paragraph explaining: the IBIT portfolio's larger tail losses come mainly from
the volatility level rather than the distributional shape.

This script treats that last sentence as a **testable proposition**, not an assertion:

    ES ≈ μ + σ × ES_z(α)
     log(ES_I / ES_G) = log(σ_I / σ_G) + log(ES_z,I / ES_z,G)

The two terms are computed separately, then a block bootstrap test is run on the Hill
tail indices of the standardized residuals to see whether their difference is significant.
If the shape contribution is near 0 and the two tail indices do not differ significantly,
the proposition holds; otherwise it must be rewritten.

Dependencies: numpy, pandas, matplotlib (plotting is skipped if missing). scipy and
              rugarch are not needed. The t distribution, chi-square distribution, and
              the GARCH(1,1)-t MLE are all self-implemented; --selftest verifies them.

Usage
    python 10_tail_diagnostics.py --selftest
    python 10_tail_diagnostics.py
    python 10_tail_diagnostics.py --data data/processed/returns__20260921.csv
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import pandas as pd

# =========================================================================
# Numerical kernel (self-implemented)
# =========================================================================

def _betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / max(abs(1.0 + aa * d), 1e-300) * np.sign(1.0 + aa * d)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / max(abs(1.0 + aa * d), 1e-300) * np.sign(1.0 + aa * d)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, nu):
    x = nu / (nu + t * t)
    p = 0.5 * betainc(nu / 2.0, 0.5, x)
    return p if t <= 0 else 1.0 - p


def t_ppf(p, nu, lo=-1e3, hi=1e3):
    if not (0.0 < p < 1.0):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def _gser(a, x, itmax=500, eps=3e-14):
    ap, s, de = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1
        de *= x / ap
        s += de
        if abs(de) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x, itmax=500, eps=3e-14):
    b, c = x + 1.0 - a, 1e300
    d = 1.0 / b
    h = d
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def chi2_sf(x, k):
    """Survival function of the chi-square distribution P(X > x)."""
    if x <= 0:
        return 1.0
    a = k / 2.0
    return 1.0 - _gser(a, x / 2.0) if x / 2.0 < a + 1.0 else _gcf(a, x / 2.0)


def nelder_mead(f, x0, step=0.1, itmax=4000, tol=1e-10):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step if p[i] == 0 else step * abs(p[i])
        pts.append(p)
    pts = np.array(pts)
    val = np.array([f(p) for p in pts])
    for _ in range(itmax):
        o = np.argsort(val)
        pts, val = pts[o], val[o]
        if abs(val[-1] - val[0]) < tol:
            break
        c = pts[:-1].mean(0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < val[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            pts[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < val[-2]:
            pts[-1], val[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < val[-1]:
                pts[-1], val[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                val[1:] = [f(p) for p in pts[1:]]
    o = np.argsort(val)
    return pts[o][0], val[o][0]


def fit_garch11_t(r, ar=0):
    """MLE for (AR(p)-)GARCH(1,1) + standardized t.

    ar=0 is a constant mean (the paper's original setting); ar=1 adds an AR(1) term.
    Motivation for adding the AR term: diagnostics find the IBIT portfolio's Q(z) is
    significant at lag 20 (p≈0.002), i.e. the mean equation has unmodeled serial
    correlation, while Q(z²) is fine (no problem with the volatility equation).
    The parameterization guarantees α+β<1; the AR coefficient is restricted to (-1,1)
    via tanh to ensure stationarity.
    """
    r = np.asarray(r, float)
    npar = 5 + ar

    def unpack(th):
        sig = lambda z: 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))
        p = sig(th[1]) * 0.999
        s = sig(th[2])
        phi = math.tanh(th[5]) if ar else 0.0
        return math.exp(th[0]), p * s, p * (1 - s), 2.05 + math.exp(th[3]), th[4], phi

    def resid(th):
        om, al, be, nu, mu, phi = unpack(th)
        if ar:
            e = np.empty(len(r))
            e[0] = r[0] - mu
            e[1:] = r[1:] - mu - phi * (r[:-1] - mu)
        else:
            e = r - mu
        return e, om, al, be, nu, mu, phi

    def nll(th):
        e, om, al, be, nu, mu, phi = resid(th)
        s2 = np.empty(len(r))
        s2[0] = e.var()
        for i in range(1, len(r)):
            s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
        if np.any(~np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        sc = np.sqrt(s2 * (nu - 2) / nu)
        z = e / sc
        return -np.sum(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                       - 0.5 * math.log(nu * math.pi) - np.log(sc)
                       - (nu + 1) / 2 * np.log1p(z * z / nu))

    v = r.var()
    best, bf = None, np.inf
    for p0, s0, nu0 in ((2.2, -2.4, 4.0), (3.0, -1.5, 6.0), (1.5, -2.0, 10.0)):
        x0 = [math.log(v * 0.05), p0, s0, math.log(nu0 - 2.05), r.mean()]
        if ar:
            x0.append(0.0)
        th, f = nelder_mead(nll, x0, step=0.4, itmax=8000 + 2000 * ar)
        if f < bf:
            best, bf = th, f
    e, om, al, be, nu, mu, phi = resid(best)
    s2 = np.empty(len(r))
    s2[0] = e.var()
    for i in range(1, len(r)):
        s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
    return dict(omega=om, alpha=al, beta=be, nu=nu, mu=mu, phi=phi,
                sigma=np.sqrt(s2), z=e / np.sqrt(s2), persistence=al + be,
                nll=bf, npar=npar, ar=ar,
                aic=2 * bf + 2 * npar, bic=2 * bf + npar * math.log(len(r)))


def ljung_box(x, lags):
    """Ljung–Box Q statistic and p-value."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x) / n
    q = 0.0
    for k in range(1, lags + 1):
        rk = np.dot(x[k:], x[:-k]) / n / c0
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    return q, chi2_sf(q, lags)


def hill(x, k):
    """Hill tail-index estimate (for the positive upper tail). Returns (tail index alpha, standard error)."""
    y = np.sort(np.asarray(x, float)[np.asarray(x, float) > 0])[::-1]
    if k >= len(y) or k < 2:
        return np.nan, np.nan
    h = np.mean(np.log(y[:k]) - math.log(y[k]))
    return 1.0 / h, (1.0 / h) / math.sqrt(k)


def stationary_boot_idx(n, R, L, rng):
    out = np.empty((n, R), dtype=int)
    for r in range(R):
        idx = []
        while len(idx) < n:
            s = rng.integers(0, n)
            ln = rng.geometric(1.0 / L)
            idx.extend([(s + k) % n for k in range(ln)])
        out[:, r] = idx[:n]
    return out


# =========================================================================
def selftest():
    ok = True

    def chk(lab, a, b, tol):
        nonlocal ok
        g = abs(a - b) < tol
        ok &= g
        print(f"  [{'PASS' if g else 'FAIL'}] {lab:<44} {a:.6f} vs {b:.6f}")

    print("=" * 78); print("Numerical kernel self-test"); print("=" * 78)
    print("\n-- Chi-square survival function (vs textbook critical values: sf(crit, df) should = 0.05) --")
    for k, crit in [(1, 3.8415), (2, 5.9915), (5, 11.0705), (10, 18.3070), (20, 31.4104)]:
        chk(f"chi2_sf(crit_5%, df={k})", chi2_sf(crit, k), 0.05, 1e-3)
    chk("chi2_sf(0, 5)", chi2_sf(0.0, 5), 1.0, 1e-12)

    print("\n-- t quantiles --")
    for nu, p, ref in [(5, 0.975, 2.5706), (10, 0.975, 2.2281), (30, 0.95, 1.6973)]:
        chk(f"t_ppf({p}, {nu})", t_ppf(p, nu), ref, 1e-3)

    print("\n-- Ljung–Box (empirical level under white noise should be ≈ 5%) --")
    rng = np.random.default_rng(11)
    rej = np.mean([ljung_box(rng.standard_normal(600), 10)[1] < 0.05 for _ in range(1500)])
    chk("LB rejection rate under white noise", rej, 0.05, 0.025)
    ar = np.zeros(2000)
    for i in range(1, 2000):
        ar[i] = 0.5 * ar[i - 1] + rng.standard_normal()
    chk("LB p-value for AR(1) (should be ≈0)", ljung_box(ar, 10)[1], 0.0, 1e-6)

    print("\n-- Hill estimation (a Pareto(alpha) sample should recover alpha) --")
    for a_true in (2.0, 3.5):
        x = (1 - rng.random(200000)) ** (-1.0 / a_true)
        est, _ = hill(x, 4000)
        chk(f"Hill recovers alpha={a_true}", est, a_true, a_true * 0.08)

    print("\n-- GARCH(1,1)-t MLE --")
    om, al, be, nu = 2e-6, 0.08, 0.90, 6.0
    n = 5000
    e = np.empty(n); s2 = np.empty(n); s2[0] = om / (1 - al - be)
    z = rng.standard_t(nu, n) / math.sqrt(nu / (nu - 2))
    e[0] = math.sqrt(s2[0]) * z[0]
    for i in range(1, n):
        s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
        e[i] = math.sqrt(s2[i]) * z[i]
    g = fit_garch11_t(e)
    chk("GARCH alpha", g["alpha"], al, 0.04)
    chk("GARCH beta", g["beta"], be, 0.06)
    chk("GARCH nu", g["nu"], nu, 2.0)

    try:
        from scipy import stats as st
        print("\n-- scipy cross-validation (optional) --")
        chk("chi2_sf vs scipy", chi2_sf(12.0, 5), st.chi2.sf(12.0, 5), 1e-9)
        chk("t_ppf vs scipy", t_ppf(0.01, 8), st.t.ppf(0.01, 8), 1e-6)
    except ImportError:
        print("\n  (scipy not installed, skipping cross-validation)")

    print("\n" + "=" * 78)
    print("Self-test " + ("all passed." if ok else "has failures, do not use the results."))
    return ok


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--outdir", default="out/10_tail")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.data is None:
        c = sorted(glob.glob(os.path.join("data", "processed", "returns__*.csv")))
        if not c:
            print("Cannot find data/processed/returns__*.csv, please specify with --data")
            sys.exit(1)
        a.data = c[-1]
    os.makedirs(a.outdir, exist_ok=True)

    d = pd.read_csv(a.data, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    P = {"IBIT portfolio": (d.IBIT + d.SPY + d.TLT) / 3,
         "GLD portfolio": (d.GLD + d.SPY + d.TLT) / 3}
    M = []

    def w(s=""):
        M.append(s); print(s)

    w("# Tail diagnostics (reviewer comment 8)\n")
    w(f"Data: `{a.data}`, {len(d)} trading days\n")

    # ---------------------------------------------------------- Fitting ----
    G = {k: fit_garch11_t(r.values) for k, r in P.items()}
    w("## A. GARCH(1,1)-t fit and moments of standardized residuals\n")
    rows = []
    for k, g in G.items():
        z = g["z"]
        nu = g["nu"]
        # Theoretical excess kurtosis of the standardized t = 6/(nu-4), exists when nu>4
        th_k = 6.0 / (nu - 4) if nu > 4 else float("inf")
        rows.append([k, f"{g['alpha']:.3f}", f"{g['beta']:.3f}", f"{g['persistence']:.3f}",
                     f"{g['nu']:.2f}", f"{pd.Series(z).skew():+.3f}",
                     f"{pd.Series(z).kurt():.3f}",
                     "∞" if math.isinf(th_k) else f"{th_k:.3f}"])
    w(md(rows, ["Portfolio", "α", "β", "α+β", "ν", "Residual skew", "Residual excess kurtosis", "t(ν) theoretical kurtosis"]))
    w("Residual kurtosis close to the theoretical value = the model has absorbed the fat tails; "
      "clearly above the theoretical value = there is still unmodeled tail risk.\n")

    # ------------------------------------------------------ Ljung-Box ----
    w("## B. Ljung–Box test\n")
    rows = []
    bad_mean = []
    for k, g in G.items():
        z = g["z"]
        for lag in (5, 10, 20):
            q1, p1 = ljung_box(z, lag)
            q2, p2 = ljung_box(z ** 2, lag)
            if p1 < 0.05:
                bad_mean.append(k)
            rows.append([k if lag == 5 else "", lag, f"{q1:.2f}",
                         f"{p1:.4f}" + ("**" if p1 < .05 else ""),
                         f"{q2:.2f}", f"{p2:.4f}" + ("**" if p2 < .05 else "")])
    w(md(rows, ["Portfolio", "Lag", "Q(z)", "p", "Q(z²)", "p"]))
    w("Q(z²) not significant = conditional heteroskedasticity is adequately captured by GARCH (no residual ARCH effects).")
    w("Q(z) significant = the **mean equation** has unmodeled serial correlation, unrelated to the volatility equation.\n")

    # If constant mean is insufficient, fit AR(1)-GARCH and compare
    bad_mean = sorted(set(bad_mean))
    if bad_mean:
        w(f"### B2. Constant mean insufficient → switch to an AR(1) mean equation\n")
        w(f"The following portfolios have significant Q(z) at some lags: **{', '.join(bad_mean)}**. "
          "The constant mean equation misses predictable serial correlation, so an "
          "AR(1)-GARCH(1,1)-t is fitted for comparison.\n")
        G2 = {k: fit_garch11_t(P[k].values, ar=1) for k in G}
        rows = []
        for k in G:
            g0, g1 = G[k], G2[k]
            r0 = [f"{ljung_box(g0['z'], L)[1]:.4f}" for L in (5, 10, 20)]
            r1 = [f"{ljung_box(g1['z'], L)[1]:.4f}" for L in (5, 10, 20)]
            rows.append([k, "Constant mean", "—", *r0, f"{g0['aic']:.1f}", f"{g0['bic']:.1f}"])
            rows.append(["", "AR(1)", f"{g1['phi']:+.3f}", *r1,
                         f"{g1['aic']:.1f}", f"{g1['bic']:.1f}"])
        w(md(rows, ["Portfolio", "Mean equation", "φ", "Q(z) p@5", "p@10", "p@20", "AIC", "BIC"]))
        fixed = [k for k in bad_mean
                 if all(ljung_box(G2[k]["z"], L)[1] > .05 for L in (5, 10, 20))]
        better = [k for k in G if G2[k]["bic"] < G[k]["bic"]]
        if fixed:
            w(f"**AR(1) fixed the serial-correlation problem for {', '.join(fixed)}** "
              f"(all lags p > 0.05).")
        else:
            w("AR(1) did not fully eliminate the serial correlation; consider a higher-order "
              "ARMA or checking for outliers.")
        w(f"Settings where BIC is better: {', '.join(better) if better else 'none (constant mean has lower BIC)'}\n")
        w("**Handling recommendation**: if AR(1) both fixes the diagnostics and has better BIC, "
          "switch the main text to AR(1)-GARCH(1,1)-t; otherwise keep the constant mean and "
          "honestly report this diagnostic result in the limitations.\n")

    # ----------------------------------------------------------- Hill ----
    w("## C. Hill tail index\n")
    w("Estimated on the upper tail of **losses** (−returns). A smaller tail index = fatter tail.\n")
    rng = np.random.default_rng(20260922)
    HILL = {}
    rows = []
    for k, g in G.items():
        raw = -P[k].values
        std = -g["z"]
        n_eff = int((raw > 0).sum())
        ks = [int(x) for x in (0.05 * n_eff, 0.10 * n_eff, 0.15 * n_eff, 0.20 * n_eff)]
        HILL[k] = {}
        for nm, x in (("Raw returns", raw), ("Standardized residuals", std)):
            est = [hill(x, kk)[0] for kk in ks]
            HILL[k][nm] = (x, ks, est)
            rows.append([k if nm == "Raw returns" else "", nm,
                         *[f"{e:.2f}" for e in est]])
    w(md(rows, ["Portfolio", "Series", *[f"k={int(p*100)}%" for p in (.05, .10, .15, .20)]]))
    w("**The Hill estimate is highly sensitive to k, and a single point estimate is not "
      "trustworthy** — hence four levels are reported together, and a full Hill plot is "
      "given in the figure.\n")

    # ------------------------------------------------ Core test: decomposition ----
    w("## D. Core test: do tail differences come from the volatility level or the distributional shape?\n")
    w("Decomposition  log(ES_I / ES_G) = log(σ_I / σ_G) + log(ES_z,I / ES_z,G)\n")
    alpha = 0.05
    res = {}
    for k, g in G.items():
        r = P[k].values
        z = g["z"]
        v = np.percentile(r, alpha * 100)
        es = -r[r <= v].mean()
        sd = r.std(ddof=1)
        vz = np.percentile(z, alpha * 100)
        esz = -z[z <= vz].mean()
        res[k] = dict(es=es, sd=sd, esz=esz)
    kI, kG = "IBIT portfolio", "GLD portfolio"
    lr_es = math.log(res[kI]["es"] / res[kG]["es"])
    lr_sd = math.log(res[kI]["sd"] / res[kG]["sd"])
    lr_sh = math.log(res[kI]["esz"] / res[kG]["esz"])
    w(md([["ES95 ratio", f"{res[kI]['es']/res[kG]['es']:.3f}x", f"{lr_es:+.4f}", "100.0%"],
          ["└ Volatility contribution σ_I/σ_G", f"{res[kI]['sd']/res[kG]['sd']:.3f}x",
           f"{lr_sd:+.4f}", f"{lr_sd/lr_es*100:.1f}%"],
          ["└ Shape contribution ES_z,I/ES_z,G", f"{res[kI]['esz']/res[kG]['esz']:.3f}x",
           f"{lr_sh:+.4f}", f"{lr_sh/lr_es*100:.1f}%"]],
         ["Component", "Ratio", "Log ratio", "Share"]))

    # Block bootstrap test of the difference in tail indices of standardized residuals
    zI, zG = G[kI]["z"], G[kG]["z"]
    n = min(len(zI), len(zG))
    kk = max(20, int(0.10 * n))
    dobs = hill(-zI, kk)[0] - hill(-zG, kk)[0]
    IDX = stationary_boot_idx(n, a.boot, 10, rng)
    bs = []
    for j in range(a.boot):
        i = IDX[:, j]
        h1 = hill(-zI[i], kk)[0]
        h2 = hill(-zG[i], kk)[0]
        if np.isfinite(h1) and np.isfinite(h2):
            bs.append(h1 - h2)
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    w(f"Hill tail-index difference of the standardized residuals (IBIT − GLD, k={kk}):")
    w(f"**{dobs:+.3f}**, block bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}], two-sided p = {pv:.3f}\n")

    # The verdict must consider the SIGN: a negative shape contribution means it pulls
    # the difference smaller, in which case "the difference comes from volatility" is
    # even stronger than the original proposition, not false.
    # (The first version only looked at |shape share| < 0.25 and misjudged -25.3% as
    # "proposition does not hold" — now fixed.)
    share_sd = lr_sd / lr_es
    share_sh = lr_sh / lr_es
    tail_same = pv > 0.10
    if share_sh <= 0:
        v = "**The proposition holds, and is stronger than the original statement**"
        expl = (f"Volatility contribution {share_sd*100:.0f}%, shape contribution {share_sh*100:.0f}% (**negative**).\n\n"
                "The shape difference does not amplify the tail gap; instead it pulls it smaller — "
                "that is, the left tail of the IBIT portfolio's standardized residuals is **thinner** "
                "than that of the GLD portfolio.\n\n"
                "You can write: all of the difference in tail losses comes from the volatility level; "
                "as for distributional shape, the IBIT portfolio does not have fatter tails than "
                "the GLD portfolio.")
    elif share_sh < 0.25 and tail_same:
        v = "**The proposition holds**"
        expl = (f"Volatility contribution {share_sd*100:.0f}%, shape contribution {share_sh*100:.0f}%, "
                "and the tail indices of the two standardized-residual series do not differ significantly.\n\n"
                "You can write: the tail-loss difference comes mainly from the volatility level.")
    elif share_sh < 0.25:
        v = "**The proposition partially holds**"
        expl = (f"Shape contribution is only {share_sh*100:.0f}%, but the two tail indices differ "
                f"significantly (p={pv:.3f}).\n\n"
                "Suggested wording: the difference is dominated by the volatility level, with a "
                "measurable contribution from distributional shape as well.")
    else:
        v = "**The proposition does not hold**"
        expl = (f"Shape contribution reaches {share_sh*100:.0f}%, so one cannot say it comes "
                "\"mainly from the volatility level\". That passage must be rewritten.")
    w(f"### Verdict: {v}\n")
    w(expl + "\n")
    if tail_same:
        w(f"The tail-index difference is not significant (p = {pv:.3f}), further supporting "
          "\"the two distributions have comparable shapes\".\n")

    # ------------------------------------------------------------ Figures ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure labels are all English: the default font lacks CJK, so Chinese would
        # turn into boxes and spam warnings
        EN = {"IBIT portfolio": "IBIT portfolio", "GLD portfolio": "GLD portfolio"}
        fig, ax = plt.subplots(2, 2, figsize=(11, 9))
        for j, (k, g) in enumerate(G.items()):
            z = np.sort(g["z"])
            m = len(z)
            pp = (np.arange(1, m + 1) - 0.5) / m
            qt = np.array([t_ppf(p, g["nu"]) / math.sqrt(g["nu"] / (g["nu"] - 2))
                           for p in pp])
            qn = np.array([t_ppf(p, 1e6) for p in pp])
            ax[0, j].plot(qt, z, ".", ms=3, color="#1f4e79", label="vs fitted t(ν)")
            ax[0, j].plot(qn, z, ".", ms=3, color="#c0504d", alpha=.5, label="vs Normal")
            lim = [min(qt.min(), z.min()), max(qt.max(), z.max())]
            ax[0, j].plot(lim, lim, "k-", lw=.8)
            ax[0, j].set_title(f"QQ plot - {EN.get(k,k)} (nu={g['nu']:.1f})")
            ax[0, j].set_xlabel("Theoretical quantile")
            ax[0, j].set_ylabel("Standardized residual")
            ax[0, j].legend(fontsize=8, loc="upper left")

            for nm, col in (("Raw returns", "#c0504d"), ("Standardized residuals", "#1f4e79")):
                x, _, _ = HILL[k][nm]
                y = np.sort(x[x > 0])[::-1]
                kr = np.arange(10, min(len(y) - 1, int(.35 * len(y))))
                hv = [hill(x, int(t))[0] for t in kr]
                ax[1, j].plot(kr, hv, lw=1.3, color=col,
                              label="Raw returns" if nm == "Raw returns" else "Std. residuals")
            ax[1, j].set_title(f"Hill plot - {EN.get(k,k)}")
            ax[1, j].set_xlabel("k (order statistics used)")
            ax[1, j].set_ylabel("Tail index")
            ax[1, j].legend(fontsize=8)
            ax[1, j].set_ylim(0, 8)
        plt.tight_layout()
        p1 = os.path.join(a.outdir, "fig_tail_diagnostics.pdf")
        plt.savefig(p1); plt.close()
        w(f"Figure written: `{p1}` (top row QQ plots, bottom row Hill plots)\n")
    except Exception as e:
        w(f"(plotting skipped: {e})\n")

    # ------------------------------------------------------- Ready-to-use paragraph ----
    w("## E. Paragraph ready to be written into the paper\n")
    w("> We assess the adequacy of the GARCH(1,1)-t specification through standardized "
      "residual diagnostics. Residual skewness and excess kurtosis are reported in "
      "Table X; Ljung–Box tests on squared standardized residuals show "
      f"{'no' if all(ljung_box(g['z']**2, 10)[1] > .05 for g in G.values()) else 'some'} "
      "remaining conditional heteroskedasticity at the 5% level, indicating the "
      "volatility dynamics are adequately captured. QQ plots against the fitted "
      "Student-t are shown in Figure Y.\n")
    w(f"> Hill tail-index estimates are reported over a range of order statistics "
      f"(k = 5%–20% of observations) rather than at a single k, since the estimator is "
      f"known to be sensitive to this choice; the corresponding Hill plots appear in "
      f"Figure Y. Decomposing the ES95 ratio between the two portfolios gives "
      f"{lr_sd/lr_es*100:.0f}% attributable to the difference in volatility level and "
      f"{lr_sh/lr_es*100:.0f}% to the difference in distributional shape. The tail "
      f"indices of the standardized residuals differ by {dobs:+.2f} "
      f"(block-bootstrap 95% CI [{lo:+.2f}, {hi:+.2f}], p = {pv:.2f}).\n")

    p = os.path.join(a.outdir, "tail_diagnostics.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(M))
    print(f"\nWritten {p}")


def md(rows, header):
    o = ["| " + " | ".join(str(h) for h in header) + " |",
         "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        o.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    main()