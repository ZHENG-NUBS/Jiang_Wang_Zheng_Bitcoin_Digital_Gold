#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
35_full_bootstrap_dependence.py — Copula and DCC intervals that include the estimation error of the
marginal GARCH models (second-round revision: Reviewer 1, comment 4)

In script 22 the copula intervals resample the probability integral transforms (PITs) and the DCC
intervals are Newey–West intervals for a regression on the estimated correlation path: both are
conditional on the first-stage GARCH estimates. This script re-estimates every stage in every
bootstrap replicate (a two-stage filtered bootstrap; Pascual, Romo and Ruiz 2006):

  1. Fit GARCH(1,1)-t to the asset and to SPY; keep the standardized residual pairs z_t.
  2. Resample the pairs with a stationary block bootstrap (mean block 10 days, preserving their
     contemporaneous and short-run joint dependence) and rebuild two return series with the
     estimated GARCH recursions.
  3. Re-estimate both GARCH models, recompute the PITs, refit the copula family selected by AIC in
     the original data and the Student-t copula, and refit the DCC(1,1) model.
  4. Record lambda_L(5%) = C(0.05, 0.05) / 0.05 of each copula (0.05 under independence) and the DCC
     statistics of script 22: the mean conditional correlation on SPY 5%-tail days (c0 + c1 + c2)
     and its difference from other days (c1 + c2).

Percentile intervals from this bootstrap are reported next to the conditional intervals.

Usage
    python 35_full_bootstrap_dependence.py               # B = 500 per asset, writes out/35/
    python 35_full_bootstrap_dependence.py --boot 100    # quicker trial
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402
from arch.univariate import ConstantMean, GARCH, StudentsT  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261025
L_BOOT = 10
HAC = 20


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m22 = load("m22", "22_safe_haven_dependence.py")
warnings.filterwarnings("ignore")


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def garch_fit(r):
    am = ConstantMean(r * 100)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = StudentsT()
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 2000})
    p = res.params
    z = np.asarray(res.std_resid, float)
    nu = float(p["nu"])
    u = stats.t.cdf(z * math.sqrt(nu / (nu - 2)), nu)
    return dict(mu=float(p["mu"]), omega=float(p["omega"]), alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
                nu=nu, z=z, u=u, s0=float(np.asarray(res.conditional_volatility)[0]))


def rebuild(par, z):
    """Return series (decimal) from standardized residuals z with the GARCH(1,1) recursion."""
    n = len(z)
    x = np.empty(n); s2 = par["s0"] ** 2
    om, al, be, mu = par["omega"], par["alpha"], par["beta"], par["mu"]
    for t in range(n):
        if t > 0:
            s2 = om + al * e * e + be * s2
        e = math.sqrt(s2) * z[t]
        x[t] = mu + e
    return x / 100


def dcc_path_fast(a, b, Z):
    Qbar = np.corrcoef(Z.T)
    x11, x22, x12 = Z[:, 0] ** 2, Z[:, 1] ** 2, Z[:, 0] * Z[:, 1]
    q = []
    for xx, qb in ((x11, Qbar[0, 0]), (x22, Qbar[1, 1]), (x12, Qbar[0, 1])):
        q.append(qb + signal.lfilter([0.0, a], [1.0, -b], xx - qb))
    return q[2] / np.sqrt(q[0] * q[1])


def dcc_nll_fast(p, Z):
    a, b = p
    if a < 0 or b < 0 or a + b >= 0.999:
        return 1e10
    rho = dcc_path_fast(a, b, Z)
    if not np.all(np.abs(rho) < 1):
        return 1e10
    x, y = Z[:, 0], Z[:, 1]
    return 0.5 * np.sum(np.log(1 - rho ** 2) + (x * x + y * y - 2 * rho * x * y) / (1 - rho ** 2) - (x * x + y * y))


def dcc_fit_fast(Z, x0=None):
    best = None
    starts = ([x0] if x0 is not None else []) + [(0.03, 0.95), (0.05, 0.90), (0.10, 0.80), (0.02, 0.97)]
    for s in starts:
        r = optimize.minimize(dcc_nll_fast, list(s), args=(Z,), method="Nelder-Mead",
                              options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    a, b = best.x
    return float(a), float(b), dcc_path_fast(a, b, Z)


def dcc_tail_stats(rho, spy):
    q10, q05, q01 = (np.quantile(spy, q) for q in (0.10, 0.05, 0.01))
    X = np.column_stack([np.ones(len(spy)), spy <= q10, spy <= q05, spy <= q01]).astype(float)
    b, V = m22.nw_ols(rho, X, HAC)
    c5 = np.array([1, 1, 1, 0.0]); d5 = np.array([0, 1, 1, 0.0])
    return dict(tail5=float(c5 @ b), tail5_se=float(math.sqrt(c5 @ V @ c5)),
                diff5=float(d5 @ b), diff5_se=float(math.sqrt(d5 @ V @ d5)), mean=float(rho.mean()))


def pct(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) > 10 else [np.nan, np.nan]


def analyse(x, s, B, rng):
    gx, gs = garch_fit(x), garch_fit(s)
    fits = {nm: m22.fit_copula(nm, gx["u"], gs["u"]) for nm in m22.COPULAS}
    best = min(fits, key=lambda k: fits[k]["aic"])
    fams = list(dict.fromkeys([best, "Student-t"]))
    pt = {nm: m22.copula_lam_u(nm, fits[nm]["par"], 0.05) for nm in fams}
    Z = np.column_stack([gx["z"], gs["z"]])
    a0, b0, rho = dcc_fit_fast(Z)
    d0 = dcc_tail_stats(rho, s)
    # conditional (script 22 style) intervals: PIT block bootstrap for the copula
    n = len(x)
    idx_c = m22.stationary_boot(n, B, 5, rng)
    cond = {nm: [] for nm in fams}
    for i in idx_c.T:
        for nm in fams:
            fb = m22.fit_copula(nm, gx["u"][i], gs["u"][i], x0=fits[nm]["par"])
            cond[nm].append(m22.copula_lam_u(nm, fb["par"], 0.05))
    # full two-stage bootstrap
    idx = m22.stationary_boot(n, B, L_BOOT, rng)
    full = {nm: [] for nm in fams}
    dd = {"tail5": [], "diff5": [], "a": [], "b": []}
    fails = 0
    for i in idx.T:
        try:
            xs = rebuild(gx, Z[i, 0]); ss_ = rebuild(gs, Z[i, 1])
            hx, hs = garch_fit(xs), garch_fit(ss_)
            for nm in fams:
                fb = m22.fit_copula(nm, hx["u"], hs["u"], x0=fits[nm]["par"])
                full[nm].append(m22.copula_lam_u(nm, fb["par"], 0.05))
            ab, bb, rb = dcc_fit_fast(np.column_stack([hx["z"], hs["z"]]), x0=(a0, b0))
            t = dcc_tail_stats(rb, ss_)
            dd["tail5"].append(t["tail5"]); dd["diff5"].append(t["diff5"]); dd["a"].append(ab); dd["b"].append(bb)
        except Exception:  # noqa: BLE001
            fails += 1
    out = dict(n=n, best=best, fails=fails, B=B, dcc=dict(a=a0, b=b0, **d0,
               tail5_ci_hac=[d0["tail5"] - 1.96 * d0["tail5_se"], d0["tail5"] + 1.96 * d0["tail5_se"]],
               diff5_ci_hac=[d0["diff5"] - 1.96 * d0["diff5_se"], d0["diff5"] + 1.96 * d0["diff5_se"]],
               tail5_ci_full=pct(dd["tail5"]), diff5_ci_full=pct(dd["diff5"]),
               a_ci_full=pct(dd["a"]), b_ci_full=pct(dd["b"])), copula={})
    for nm in fams:
        out["copula"][nm] = dict(lam05=pt[nm], ci_conditional=pct(cond[nm]), ci_full=pct(full[nm]),
                                 width_ratio=(np.diff(pct(full[nm]))[0] / np.diff(pct(cond[nm]))[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/35")
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--items", default="", help="comma-separated subset, e.g. 'btc_long|GLD'; default all")
    a = ap.parse_args()
    parts = os.path.join(a.outdir, "parts")
    os.makedirs(parts, exist_ok=True)
    E = LD.load_etf(); Bl = LD.load_btc_long()
    samples = [("etf", "ETF sample", E, ["IBIT", "GLD", "TLT"]), ("btc_long", "Bitcoin long sample", Bl, ["BTC", "GLD", "TLT"])]
    items = [(k, (key, lab, D, x)) for k, (key, lab, D, x) in
             enumerate((key, lab, D, x) for key, lab, D, assets in samples for x in assets)]
    want = set(filter(None, a.items.split(",")))
    for k, (key, lab, D, x) in items:
        if want and f"{key}|{x}" not in want:
            continue
        # one generator per item, so each item is reproducible on its own and items can run in parallel
        rng = np.random.default_rng([SEED, k])
        t0 = time.time()
        d = D[[x, "SPY"]].dropna()
        o = analyse(d[x].values, d.SPY.values, a.boot, rng)
        json.dump(o, open(os.path.join(parts, f"{key}_{x}.json"), "w"), indent=1, default=float)
        print(f"{lab} {x}: {time.time() - t0:.0f} s, failures {o['fails']}", flush=True)
    # assemble once every item is available
    OUT, rows_c, rows_d = {}, [], []
    for k, (key, lab, D, x) in items:
        f = os.path.join(parts, f"{key}_{x}.json")
        if not os.path.exists(f):
            print(f"waiting for {key}|{x}; tables not written yet")
            return
        o = OUT[f"{key}|{x}"] = json.load(open(f))
        for nm, c in o["copula"].items():
            rows_c.append([lab, x, nm + (" (AIC best)" if nm == o["best"] else ""), f"{c['lam05']:.3f}",
                           f"[{c['ci_conditional'][0]:.3f}, {c['ci_conditional'][1]:.3f}]",
                           f"[{c['ci_full'][0]:.3f}, {c['ci_full'][1]:.3f}]", f"{c['width_ratio']:.2f}"])
        q = o["dcc"]
        rows_d.append([lab, x, f"{q['a']:.3f} / {q['b']:.3f}", f"{q['tail5']:+.3f}",
                       f"[{q['tail5_ci_hac'][0]:+.3f}, {q['tail5_ci_hac'][1]:+.3f}]",
                       f"[{q['tail5_ci_full'][0]:+.3f}, {q['tail5_ci_full'][1]:+.3f}]", f"{q['diff5']:+.3f}",
                       f"[{q['diff5_ci_hac'][0]:+.3f}, {q['diff5_ci_hac'][1]:+.3f}]",
                       f"[{q['diff5_ci_full'][0]:+.3f}, {q['diff5_ci_full'][1]:+.3f}]"])
    json.dump(OUT, open(os.path.join(a.outdir, "full_bootstrap.json"), "w"), indent=1, default=float)
    L = ["# Copula and DCC intervals with re-estimated marginals (script 35)\n",
         f"Two-stage filtered bootstrap, B = {a.boot}, stationary blocks of mean length {L_BOOT} days. "
         "Conditional intervals: copula, block bootstrap of the PITs (as in script 22); DCC, Newey–West "
         f"({HAC} lags) on the estimated correlation path.\n"]
    L.append("## Copula: λ_L(5%) with SPY (independence = 0.05)\n")
    L.append(md(rows_c, ["Sample", "Asset", "Copula", "λ_L(5%)", "95% CI, conditional on marginals",
                         "95% CI, full bootstrap", "Width ratio (full / conditional)"]))
    L.append("## DCC(1,1): correlation with SPY on SPY 5%-tail days\n")
    L.append(md(rows_d, ["Sample", "Asset", "a / b", "ρ on tail days (c0+c1+c2)", "95% CI, HAC (conditional)",
                         "95% CI, full bootstrap", "Tail minus other days (c1+c2)", "95% CI, HAC", "95% CI, full bootstrap"]))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.outdir, "full_bootstrap.md"), "w", encoding="utf-8").write(txt)


if __name__ == "__main__":
    main()
