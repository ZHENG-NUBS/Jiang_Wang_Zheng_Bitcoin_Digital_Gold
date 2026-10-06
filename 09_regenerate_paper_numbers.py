#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_regenerate_paper_numbers.py — Recompute all numbers in the paper from the new data and output paste-ready tables

Input   data/processed/returns__<ASOF>.csv (produced by script 02; the SPY column is already the ETF total return)
Output  out/paper_numbers.md  -- Markdown tables, one section per section of an earlier draft of the paper
        (section labels such as §3.1 refer to that draft; see README.md for the mapping to the tables of
        the published paper)

Coverage
    §3.1  Performance of the two portfolios (Sharpe ratios with rf = 0 and with rf = 3-month T-bill)
    §3.2  Degrees of freedom of the t distribution (MLE)
    §3.3  VaR/ES under four models (historical simulation / normal / t / GARCH(1,1)-t Monte Carlo)
    §3.5  VaR/ES by regime, correlations by regime, worst 20 days
    §3.6  Stress scenarios (rule P1: SPY peak to trough, drawdown > 5%)
    §3.8  Table 3: three sets of optimized weights (with an explicit mathematical definition of the aggressive portfolio)
    Extra Volatility-matched portfolio (robustness; responds to reviewer comments 11/12)

Dependencies
    numpy + pandas only. The t-distribution CDF/quantile function, the t MLE and the
    GARCH(1,1)-t MLE are all implemented here and checked against known values in
    --selftest; scipy is not required.
    If scipy is installed, it is used automatically for cross-validation (results are unaffected).

Usage
    python 09_regenerate_paper_numbers.py --selftest      # verify the numerical kernels first
    python 09_regenerate_paper_numbers.py                 # generate all tables
    python 09_regenerate_paper_numbers.py --rf 0.043      # specify the risk-free rate
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
# Numerical kernels (self-implemented; verifiable with --selftest)
# =========================================================================

def _betacf(a, b, x, itmax=300, eps=3e-14):
    """Continued-fraction expansion of the incomplete beta function (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a,b)."""
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
    """Student-t cumulative distribution function."""
    x = nu / (nu + t * t)
    p = 0.5 * betainc(nu / 2.0, 0.5, x)
    return p if t <= 0 else 1.0 - p


def t_ppf(p, nu, lo=-1e3, hi=1e3, tol=1e-12):
    """Student-t quantile function (bisection on the CDF)."""
    if not (0.0 < p < 1.0):
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def t_pdf(t, nu):
    return math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                    - 0.5 * math.log(nu * math.pi)
                    - (nu + 1) / 2 * math.log1p(t * t / nu))


def es_std_t(alpha, nu):
    """Alpha-tail expected shortfall of the t distribution STANDARDIZED TO UNIT VARIANCE (closed form)."""
    q = t_ppf(alpha, nu)
    es = -t_pdf(q, nu) * (nu + q * q) / ((nu - 1) * alpha)
    return es / math.sqrt(nu / (nu - 2))


def nelder_mead(f, x0, step=0.1, itmax=4000, tol=1e-10):
    """Compact Nelder–Mead, used for the few low-dimensional MLEs in this script."""
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


def fit_t(x):
    """Fit a Student-t (location-scale family) to the data by MLE; returns (nu, loc, scale)."""
    x = np.asarray(x, float)

    def nll(th):
        nu = 2.05 + math.exp(th[0])      # nu > 2
        loc = th[1]
        sc = math.exp(th[2])
        z = (x - loc) / sc
        return -np.sum(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                       - 0.5 * math.log(nu * math.pi) - math.log(sc)
                       - (nu + 1) / 2 * np.log1p(z * z / nu))

    th0 = [math.log(max(x.var(ddof=1) and 4.0, 1.0)), float(np.median(x)),
           math.log(x.std(ddof=1) * 0.8)]
    th, _ = nelder_mead(nll, th0, step=0.3, itmax=6000)
    return 2.05 + math.exp(th[0]), th[1], math.exp(th[2])


def fit_garch11_t(r):
    """MLE of GARCH(1,1) with Student-t innovations. Returns a dict."""
    r = np.asarray(r, float)
    mu0 = r.mean()

    # Parameterization: the persistence p = α+β and the share s = α/p each pass
    # through a sigmoid, so α+β = p < 1 always holds and stationarity is automatic;
    # no penalty function is needed.
    # (The first version parameterized α and β with independent sigmoids; its starting
    #  point α+β = 1.03 violated the constraint, the whole simplex was penalized to a
    #  constant and the optimizer did not move -- this was caught by the self-test.)
    def unpack(th):
        sig = lambda z: 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))
        p = sig(th[1]) * 0.999
        s = sig(th[2])
        return math.exp(th[0]), p * s, p * (1 - s), 2.05 + math.exp(th[3]), th[4]

    def nll(th):
        om, al, be, nu, mu = unpack(th)
        e = r - mu
        s2 = np.empty(len(r))
        s2[0] = e.var()
        for i in range(1, len(r)):
            s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
        if np.any(~np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        sc = np.sqrt(s2 * (nu - 2) / nu)          # scale of the standardized t
        z = e / sc
        ll = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
              - 0.5 * math.log(nu * math.pi) - np.log(sc)
              - (nu + 1) / 2 * np.log1p(z * z / nu))
        return -np.sum(ll)

    v = r.var()
    # Multiple starting points: the simplex method is sensitive to the start, so keep the best
    best, bf = None, np.inf
    for p0, s0, nu0 in ((2.2, -2.4, 4.0), (3.0, -1.5, 6.0), (1.5, -2.0, 10.0)):
        th0 = [math.log(v * 0.05), p0, s0, math.log(nu0 - 2.05), mu0]
        th, f = nelder_mead(nll, th0, step=0.4, itmax=8000)
        if f < bf:
            best, bf = th, f
    th, f = best, bf
    om, al, be, nu, mu = unpack(th)
    e = r - mu
    s2 = np.empty(len(r))
    s2[0] = e.var()
    for i in range(1, len(r)):
        s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
    s2_next = om + al * e[-1] ** 2 + be * s2[-1]
    return dict(omega=om, alpha=al, beta=be, nu=nu, mu=mu,
                sigma2=s2, sigma2_next=s2_next, nll=f,
                persistence=al + be,
                uncond_vol=math.sqrt(om / max(1 - al - be, 1e-9) * 252) * 100)


# =========================================================================
def selftest():
    ok = True

    def chk(lab, a, b, tol):
        nonlocal ok
        good = abs(a - b) < tol
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {lab:<46} {a:.6f} vs {b:.6f}")

    print("=" * 78)
    print("Numerical kernel self-test")
    print("=" * 78)
    print("\n-- t distribution CDF / quantiles (vs. textbook critical values) --")
    for nu, p, ref in [(1, 0.975, 12.7062), (5, 0.975, 2.5706), (10, 0.975, 2.2281),
                       (30, 0.975, 2.0423), (100, 0.95, 1.6602), (8, 0.99, 2.8965)]:
        chk(f"t_ppf(p={p}, nu={nu})", t_ppf(p, nu), ref, 1e-3)
    chk("t_cdf(t_ppf(0.05,7),7) == 0.05", t_cdf(t_ppf(0.05, 7), 7), 0.05, 1e-9)
    chk("t_cdf(0, 5) == 0.5", t_cdf(0.0, 5), 0.5, 1e-12)

    print("\n-- ES of the standardized t (vs. independently derived values) --")
    for a, nu, ref in [(0.05, 3.5, -2.268060), (0.05, 5, -2.238684),
                       (0.05, 8, -2.177060), (0.01, 5, -3.448837),
                       (0.01, 30, -2.768459)]:
        chk(f"ES_z(alpha={a}, nu={nu})", es_std_t(a, nu), ref, 1e-4)

    print("\n-- t distribution MLE (synthetic samples with known parameters) --")
    rng = np.random.default_rng(7)
    for nu_true in (4.0, 8.0):
        x = rng.standard_t(nu_true, 40000) * 0.01 + 0.0005
        nu, loc, sc = fit_t(x)
        chk(f"fit_t recovers nu={nu_true}", nu, nu_true, max(0.6, nu_true * 0.12))

    print("\n-- GARCH(1,1)-t MLE (synthetic sample with known parameters) --")
    om_t, al_t, be_t, nu_t = 2e-6, 0.08, 0.90, 6.0
    n = 6000
    e = np.empty(n); s2 = np.empty(n); s2[0] = om_t / (1 - al_t - be_t)
    z = rng.standard_t(nu_t, n) / math.sqrt(nu_t / (nu_t - 2))
    e[0] = math.sqrt(s2[0]) * z[0]
    for i in range(1, n):
        s2[i] = om_t + al_t * e[i - 1] ** 2 + be_t * s2[i - 1]
        e[i] = math.sqrt(s2[i]) * z[i]
    g = fit_garch11_t(e)
    chk("GARCH recovers alpha=0.08", g["alpha"], al_t, 0.04)
    chk("GARCH recovers beta=0.90", g["beta"], be_t, 0.05)
    chk("GARCH recovers nu=6.0", g["nu"], nu_t, 2.0)
    chk("GARCH persistence alpha+beta", g["persistence"], al_t + be_t, 0.03)

    try:
        from scipy import stats as st
        print("\n-- Cross-validation against scipy (optional) --")
        for nu, p in [(5, 0.975), (8, 0.01), (30, 0.05)]:
            chk(f"t_ppf vs scipy (nu={nu},p={p})", t_ppf(p, nu), st.t.ppf(p, nu), 1e-6)
    except ImportError:
        print("\n  (scipy not installed; cross-validation skipped. The self-tests above are sufficient.)")

    print("\n" + "=" * 78)
    print("Self-test: " + ("all checks passed." if ok else "some checks FAILED; do not use the results."))
    return ok


# =========================================================================
# Paper numbers
# =========================================================================
A4 = ["GLD", "SPY", "IBIT", "TLT"]


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def ann(r):
    yrs = len(r) / 252
    return ((1 + r).prod() ** (1 / yrs) - 1) * 100


def mdd(r):
    c = (1 + r).cumprod()
    return (c / c.cummax() - 1).min() * 100


def hs_var_es(x, a):
    v = np.percentile(x, a * 100)
    return -v * 100, -x[x <= v].mean() * 100


def project_simplex(w):
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1
    rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
    return np.maximum(w - css[rho] / (rho + 1), 0)


def solve_pg(grad, n, lr, iters=60000, tol=1e-13):
    w = np.repeat(1 / n, n)
    for _ in range(iters):
        wn = project_simplex(w - lr * grad(w))
        if np.abs(wn - w).max() < tol:
            return wn
        w = wn
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--rf", type=float, default=0.043,
                    help="Annualized risk-free rate used for the Sharpe ratio (default 4.3%%, roughly the 3-month T-bill yield)")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if selftest() else 1)

    if a.data is None:
        c = sorted(glob.glob(os.path.join("data", "processed", "returns__*.csv")))
        if not c:
            print("data/processed/returns__*.csv not found; specify the file with --data")
            sys.exit(1)
        a.data = c[-1]

    d = pd.read_csv(a.data, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    d = d[[c for c in A4 if c in d.columns] + (["VIX_Close"] if "VIX_Close" in d else [])]
    n = len(d)
    os.makedirs(a.outdir, exist_ok=True)
    M = []

    def w(s=""):
        M.append(s)
        print(s)

    w(f"# Paper numbers recomputed (new data)\n")
    w(f"Data: `{a.data}`, {n} trading days, {d.index[0].date()} → {d.index[-1].date()}")
    w(f"The SPY column is the **total return of the SPY ETF (AdjClose)** (the original manuscript mistakenly used the ^GSPC index).")
    w(f"Risk-free rate: {a.rf*100:.2f}% per year.\n")

    P = {"IBIT portfolio": (d.IBIT + d.SPY + d.TLT) / 3,
         "GLD portfolio": (d.GLD + d.SPY + d.TLT) / 3}

    # ---------------------------------------------------------- §3.1 ----
    w("## §3.1 Portfolio performance\n")
    rows = []
    for k, r in P.items():
        A, V, D = ann(r), r.std() * np.sqrt(252) * 100, mdd(r)
        rows.append([k, f"{A:.2f}%", f"{V:.2f}%", f"{D:.2f}%",
                     f"{A/V:.2f}", f"{(A-a.rf*100)/V:.2f}"])
    w(md_table(rows, ["Portfolio", "Ann. return", "Ann. volatility", "Max drawdown",
                      "Sharpe (rf=0)", f"Sharpe (rf={a.rf*100:.1f}%)"]))
    w("**The original manuscript used rf=0; the text must state the source of the risk-free rate.**\n")

    # ---------------------------------------------------------- §3.2 ----
    w("## §3.2 Fat tails of the return distribution (t-distribution MLE)\n")
    rows = []
    TDF = {}
    for k, r in P.items():
        nu, loc, sc = fit_t(r.values)
        TDF[k] = nu
        rows.append([k, f"{nu:.2f}", f"{r.skew():.2f}", f"{r.kurt():.2f}"])
    w(md_table(rows, ["Portfolio", "t degrees of freedom (MLE)", "Skewness", "Excess kurtosis"]))

    # ---------------------------------------------------------- §3.3 ----
    w("## §3.3 VaR / ES under four models (full sample, daily, %)\n")
    rng = np.random.default_rng(20260921)
    rows = []
    GARCH = {}
    for k, r in P.items():
        x = r.values
        g = fit_garch11_t(x)
        GARCH[k] = g
        sim = g["mu"] + math.sqrt(g["sigma2_next"]) * (
            rng.standard_t(g["nu"], 400000) / math.sqrt(g["nu"] / (g["nu"] - 2)))
        for al in (0.05, 0.01):
            hv, he = hs_var_es(x, al)
            zn = -(x.mean() + x.std(ddof=1) * t_ppf(al, 1e6)) * 100
            en = -(x.mean() - x.std(ddof=1) * math.exp(-0.5 * t_ppf(al, 1e6) ** 2)
                   / math.sqrt(2 * math.pi) / al) * 100
            nu = TDF[k]
            zt = -(x.mean() + x.std(ddof=1) * t_ppf(al, nu) / math.sqrt(nu / (nu - 2))) * 100
            et = -(x.mean() + x.std(ddof=1) * es_std_t(al, nu)) * 100
            gv, ge = hs_var_es(sim, al)
            rows.append([k, f"{al:.0%}", f"{hv:.2f}", f"{he:.2f}", f"{zn:.2f}", f"{en:.2f}",
                         f"{zt:.2f}", f"{et:.2f}", f"{gv:.2f}", f"{ge:.2f}"])
    w(md_table(rows, ["Portfolio", "α", "HS VaR", "HS ES", "Normal VaR", "Normal ES",
                      "t VaR", "t ES", "GARCH-t VaR", "GARCH-t ES"]))
    w("GARCH(1,1)-t parameters:\n")
    rows = [[k, f"{g['omega']:.2e}", f"{g['alpha']:.3f}", f"{g['beta']:.3f}",
             f"{g['persistence']:.3f}", f"{g['nu']:.2f}", f"{g['uncond_vol']:.2f}%"]
            for k, g in GARCH.items()]
    w(md_table(rows, ["Portfolio", "ω", "α", "β", "α+β", "ν", "Unconditional ann. volatility"]))

    # ---------------------------------------------------------- §3.5 ----
    w("## §3.5 Risk by regime\n")
    if "VIX_Close" in d.columns:
        v = d.VIX_Close
        regs = {"Low volatility (VIX≤15)": v <= 15,
                "Normal (15<VIX≤25)": (v > 15) & (v <= 25),
                "High volatility (VIX>25)": v > 25}
        w("**Note**: the original manuscript text says \"VIX < 15\" but the counts correspond to \"VIX ≤ 15\". "
          "**≤15** is used throughout here; the text must be revised accordingly.\n")
        rows = []
        for rg, m in regs.items():
            m = m.values
            for k, r in P.items():
                hv, he = hs_var_es(r.values[m], 0.05)
                rows.append([rg if k.startswith("IBIT") else "", f"{int(m.sum())}",
                             k, f"{hv:.2f}", f"{he:.2f}"])
        w(md_table(rows, ["Regime", "Days", "Portfolio", "VaR95", "ES95"]))

        rows = []
        for rg, m in regs.items():
            m = m.values
            rows.append([rg, f"{int(m.sum())}",
                         f"{np.corrcoef(d.IBIT[m], d.SPY[m])[0,1]:+.3f}",
                         f"{np.corrcoef(d.GLD[m], d.SPY[m])[0,1]:+.3f}",
                         f"{np.corrcoef(d.TLT[m], d.SPY[m])[0,1]:+.3f}"])
        w(md_table(rows, ["Regime", "Days", "IBIT–SPY", "GLD–SPY", "TLT–SPY"]))

    w("### Mean return on the worst k days (robustness: the conclusion does not depend on k)\n")
    rows = []
    for k_ in (10, 20, 30, 61):
        ww = d.nsmallest(k_, "SPY")
        rows.append([k_, f"{ww.SPY.mean()*100:+.2f}%", f"{ww.IBIT.mean()*100:+.2f}%",
                     f"{ww.GLD.mean()*100:+.2f}%", f"{ww.TLT.mean()*100:+.2f}%",
                     f"{(ww.IBIT.mean()-ww.SPY.mean())*100:+.2f}"])
    w(md_table(rows, ["Worst k days", "SPY", "IBIT", "GLD", "TLT", "IBIT−SPY (pp)"]))

    w("### Beta-adjusted performance on tail days (volatility scale removed)\n")
    thr = np.percentile(d.SPY, 5)
    m = (d.SPY <= thr).values
    rows = []
    for c in ["IBIT", "GLD", "TLT"]:
        b = np.cov(d[c], d.SPY)[0, 1] / np.var(d.SPY, ddof=1)
        act = d[c][m].mean() * 100
        exp = b * d.SPY[m].mean() * 100
        rows.append([c, f"{b:.2f}", f"{act:+.2f}%", f"{exp:+.2f}%", f"{act-exp:+.2f}"])
    w(md_table(rows, ["Asset", "Full-sample β", f"Mean on SPY's worst 5% days (n={int(m.sum())})",
                      "Expected from β", "Excess (pp)"]))
    w("Positive excess = a smaller fall than β implies (a cushion). **This measure is not contaminated by differences in volatility scale.**\n")

    # ---------------------------------------------------------- §3.6 ----
    w("## §3.6 Stress scenarios (rule P1: SPY from previous peak to trough, drawdown > 5%)\n")
    cp = (1 + d.SPY).cumprod()
    dd = cp / cp.cummax() - 1
    eps, i = [], 0
    while i < n:
        if dd.iloc[i] < 0:
            j = i
            while j < n - 1 and dd.iloc[j + 1] < 0:
                j += 1
            tr = i + int(np.argmin(dd.iloc[i:j + 1].values))
            if dd.iloc[tr] <= -0.05:
                eps.append((max(0, i - 1), tr))
            i = j + 1
        else:
            i += 1
    rows = []
    for s, t in eps:
        sl = slice(s, t + 1)
        cu = lambda c: ((1 + d[c].iloc[sl]).prod() - 1) * 100
        pi = ((1 + P["IBIT portfolio"].iloc[sl]).prod() - 1) * 100
        pg = ((1 + P["GLD portfolio"].iloc[sl]).prod() - 1) * 100
        rows.append([f"{d.index[s].date()}..{d.index[t].date()}", t - s + 1,
                     f"{dd.iloc[t]*100:.1f}%", f"{cu('SPY'):+.2f}%", f"{cu('IBIT'):+.2f}%",
                     f"{cu('GLD'):+.2f}%", f"{pi:+.2f}%", f"{pg:+.2f}%"])
    w(md_table(rows, ["Window", "Days", "SPY drawdown", "SPY", "IBIT", "GLD",
                      "IBIT portfolio", "GLD portfolio"]))
    ib = [float(r[4].rstrip('%')) for r in rows]
    sp = [float(r[3].rstrip('%')) for r in rows]
    pi_ = [float(r[6].rstrip('%')) for r in rows]
    pg_ = [float(r[7].rstrip('%')) for r in rows]
    w(f"IBIT fell more than SPY: **{sum(1 for x,y in zip(ib,sp) if x<y)}/{len(rows)}**; "
      f"IBIT portfolio lost more than GLD portfolio: **{sum(1 for x,y in zip(pi_,pg_) if x<y)}/{len(rows)}**\n")
    w("**Never use universal claims such as \"in every scenario\"** — see the counterexamples in the table above.\n")

    # ---------------------------------------------------------- §3.8 ----
    w("## §3.8 / Table 3 Optimized weights (%)\n")
    R = d[A4].values
    mu = R.mean(0) * 252
    S = np.cov(R.T) * 252
    lr = 1.0 / (2 * np.linalg.eigvalsh(S).max())
    mv = solve_pg(lambda x: 2 * S @ x, 4, lr)

    def ns_grad(x):
        v = math.sqrt(x @ S @ x)
        return -(mu / v - (mu @ x) * (S @ x) / v ** 3)
    ms = solve_pg(ns_grad, 4, 1e-3)

    def ns_grad_rf(x):
        v = math.sqrt(x @ S @ x)
        return -((mu) / v - (mu @ x - a.rf) * (S @ x) / v ** 3)
    ms_rf = solve_pg(ns_grad_rf, 4, 1e-3)

    rows = [["Minimum variance min wʹΣw", *[f"{x*100:.2f}" for x in mv],
             f"{mu@mv*100:.2f}", f"{math.sqrt(mv@S@mv)*100:.2f}"],
            ["Maximum Sharpe (rf=0)", *[f"{x*100:.2f}" for x in ms],
             f"{mu@ms*100:.2f}", f"{math.sqrt(ms@S@ms)*100:.2f}"],
            [f"Maximum Sharpe (rf={a.rf*100:.1f}%)", *[f"{x*100:.2f}" for x in ms_rf],
             f"{mu@ms_rf*100:.2f}", f"{math.sqrt(ms_rf@S@ms_rf)*100:.2f}"]]
    w("### Two portfolios with explicit objectives\n")
    w(md_table(rows, ["Objective", *A4, "Ann. return", "Ann. volatility"]))

    w("### Aggressive portfolio: an explicit mathematical definition\n")
    w("The original manuscript gives only a verbal description (reviewer comment 11). Proposed definition:\n")
    w("> max wʹμ  s.t.  √(wʹΣw) ≤ σ\\*,  Σw = 1,  w ≥ 0\n")
    rows = []
    for tv in (0.12, 0.15, 0.17, 0.18, 0.20):
        best, bw = -1e9, None
        for lam in np.logspace(-3, 3, 220):
            wv = solve_pg(lambda x: -mu + lam * 2 * (S @ x), 4, lr, iters=20000)
            v = math.sqrt(wv @ S @ wv)
            if v <= tv + 1e-4 and mu @ wv > best:
                best, bw = mu @ wv, wv
        if bw is not None:
            rows.append([f"σ* = {tv:.0%}", *[f"{x*100:.2f}" for x in bw],
                         f"{best*100:.2f}", f"{math.sqrt(bw@S@bw)*100:.2f}"])
    rows.append(["**Original Table 3**", "73.04", "24.92", "2.03", "0.00", "—", "—"])
    w(md_table(rows, ["Volatility cap", *A4, "Ann. return", "Ann. volatility"]))
    w("Choose one σ\\* for the text and use the corresponding row in Table 3.\n")

    # ---------------------------------------------------- Robustness ----
    w("## Appendix: volatility-matched portfolio (robustness; reviewer comments 11/12)\n")
    sI, sG = d.IBIT.std() * np.sqrt(252), d.GLD.std() * np.sqrt(252)
    wI = (1 / 3) * sG / sI
    rest = (1 - wI) / 2
    VM = wI * d.IBIT + rest * d.SPY + rest * d.TLT
    hi = (d.VIX_Close > 25).values if "VIX_Close" in d else np.zeros(n, bool)
    rows = []
    for k, r in [("IBIT portfolio (equal weight 1/3)", P["IBIT portfolio"]),
                 (f"IBIT portfolio (volatility-matched {wI*100:.1f}%)", VM),
                 ("GLD portfolio (equal weight 1/3)", P["GLD portfolio"])]:
        hv, he = hs_var_es(r.values, 0.05)
        row = [k, f"{r.std()*np.sqrt(252)*100:.2f}%", f"{hv:.2f}", f"{he:.2f}"]
        if hi.any():
            hv2, he2 = hs_var_es(r.values[hi], 0.05)
            row += [f"{hv2:.2f}", f"{he2:.2f}"]
        rows.append(row)
    hdr = ["Portfolio", "Ann. volatility", "VaR95", "ES95"] + (["High-vol VaR95", "High-vol ES95"] if hi.any() else [])
    w(md_table(rows, hdr))
    if hi.any():
        e1 = hs_var_es(P["IBIT portfolio"].values[hi], .05)[1]
        e2 = hs_var_es(P["GLD portfolio"].values[hi], .05)[1]
        e3 = hs_var_es(VM.values[hi], .05)[1]
        w(f"IBIT's annualized volatility is **{sI/sG:.2f} times** that of GLD. High-volatility ES95 ratio: "
          f"equal weight **{e1/e2:.2f}x** → volatility-matched **{e3/e2:.2f}x**.\n")
        w("**Report both versions to pre-empt reviewer objections.**\n")

    p = os.path.join(a.outdir, "paper_numbers.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(M))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
