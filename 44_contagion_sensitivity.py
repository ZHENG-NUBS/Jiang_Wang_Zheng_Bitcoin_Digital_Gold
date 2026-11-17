# -*- coding: utf-8 -*-
"""
44_contagion_sensitivity.py — how robust is "no contagion detected" for Bitcoin? (round-3 comment 2.1)

A. Sensitivity of the Forbes–Rigobon test to delta. Script 31 adjusts the high-regime correlation of BTC-USD with SPY
   by rho* = rho_c / sqrt(1 + delta (1 - rho_c^2)), delta = var_high(SPY) / var_tranquil(SPY) - 1. Here delta is
   scaled by k (k delta in the original sample and in every bootstrap replication), the smallest k at which the 95%
   interval of rho* - rho_tranquil excludes zero is reported, and delta is also measured robustly from the ratio of
   squared median absolute deviations of SPY returns.
B. Slopes by regime. The slope of BTC-USD on SPY in each regime of the three-state Markov-switching model (Equation (7),
   Newey–West standard errors, 5 lags), the differences crisis - tranquil and turbulent - tranquil with Wald tests, on
   the full Bitcoin long sample and within 2014–2019 and 2020–2026; and the same with Dimson (1979) slopes that add
   the coefficients on SPY's previous- and next-day returns, which allows for BTC-USD's 00:00 UTC close.
Same data, regimes (out/31/regimes.csv) and bootstrap design (stationary, expected block length 10, 2000 replications)
as script 31.

Usage: python 44_contagion_sensitivity.py [--outdir out/44] [--boot 2000]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402

SEED = 20261111
L_BOOT = 10
NAMES = {0: "tranquil", 1: "turbulent", 2: "crisis"}


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def fr(rc, d):
    return rc / math.sqrt(1 + d * (1 - rc ** 2))


def mad_var(x):
    return (1.4826 * np.median(np.abs(x - np.median(x)))) ** 2


def fr_scan(x, s, lab, hi, lo, idx, ks):
    mh, ml = lab == hi, lab == lo
    rc, rt = np.corrcoef(x[mh], s[mh])[0, 1], np.corrcoef(x[ml], s[ml])[0, 1]
    d = s[mh].var(ddof=1) / s[ml].var(ddof=1) - 1
    d_mad = mad_var(s[mh]) / mad_var(s[ml]) - 1
    B = []
    for b in range(idx.shape[1]):
        I = idx[:, b]
        h, l_ = lab[I] == hi, lab[I] == lo
        if h.sum() < 10 or l_.sum() < 10:
            continue
        X, S = x[I], s[I]
        B.append((np.corrcoef(X[h], S[h])[0, 1], np.corrcoef(X[l_], S[l_])[0, 1],
                  S[h].var(ddof=1) / S[l_].var(ddof=1) - 1, mad_var(S[h]) / mad_var(S[l_]) - 1))
    B = np.array(B)
    out = dict(n_hi=int(mh.sum()), n_lo=int(ml.sum()), rho_hi=float(rc), rho_lo=float(rt), delta=float(d),
               delta_mad=float(d_mad), scan=[])
    for k in ks:
        bd = np.array([fr(r1, k * dd) - r0 for r1, r0, dd, _ in B])
        out["scan"].append(dict(k=float(k), diff=float(fr(rc, k * d) - rt), ci=[float(v) for v in np.percentile(bd, [2.5, 97.5])]))
    bd = np.array([fr(r1, dm) - r0 for r1, r0, _, dm in B])
    out["mad"] = dict(diff=float(fr(rc, d_mad) - rt), ci=[float(v) for v in np.percentile(bd, [2.5, 97.5])])
    # smallest k at which the lower bound exceeds zero (fine grid)
    grid = np.round(np.arange(0.01, 1.001, 0.01), 2)
    lo_b = [np.percentile([fr(r1, g * dd) - r0 for r1, r0, dd, _ in B], 2.5) for g in grid]
    pos = [g for g, v in zip(grid, lo_b) if v > 0]
    out["k_breakeven"] = float(max(pos)) if pos else None          # largest k (<= 1) with a significant increase
    return out


def hac(y, X, lag=5):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        A = Xe[L:].T @ Xe[:-L]
        S += (1 - L / (lag + 1)) * (A + A.T)
    return b, XtXi @ S @ XtXi * n / (n - k)


def regime_slopes(btc, spy, lab, dimson=False):
    """Equation (7) with regime-specific intercepts and slopes; Dimson: slopes on SPY(t-1), SPY(t), SPY(t+1) summed."""
    n = len(btc)
    cols, names = [], []
    lags = (-1, 0, 1) if dimson else (0,)
    # sl[L][t] = spy[t + L]
    sl = {L: (np.r_[spy[L:], np.full(L, np.nan)] if L > 0 else (np.r_[np.full(-L, np.nan), spy[:L]] if L < 0 else spy))
          for L in lags}
    for s in range(3):
        d = (lab == s).astype(float)
        cols.append(d); names.append(f"a{s}")
        for L in lags:
            cols.append(d * sl[L]); names.append(f"b{s}_{L}")
    X = np.column_stack(cols)
    ok = np.all(np.isfinite(X), axis=1)
    b, V = hac(btc[ok], X[ok])
    res = {}
    for s in range(3):
        w = np.array([1.0 if nm.startswith(f"b{s}_") else 0.0 for nm in names])
        res[NAMES[s]] = dict(slope=float(w @ b), se=float(math.sqrt(w @ V @ w)), n=int((lab[ok] == s).sum()))
    for hi in (1, 2):
        w = np.array([(1.0 if nm.startswith(f"b{hi}_") else (-1.0 if nm.startswith("b0_") else 0.0)) for nm in names])
        est, se = float(w @ b), float(math.sqrt(w @ V @ w))
        res[f"{NAMES[hi]}_minus_tranquil"] = dict(diff=est, se=se, p=float(2 * stats.norm.sf(abs(est / se))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/44")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    B = LD.load_btc_long()
    reg = pd.read_csv(os.path.join(HERE, "out", "31", "regimes.csv"), parse_dates=["Date"]).set_index("Date")
    lab = reg.regime.reindex(B.index)
    ok = lab.notna().values
    B, lab = B[ok], lab[ok].astype(int).values
    x, s = B.BTC.values, B.SPY.values
    rng = np.random.default_rng(SEED)
    idx = stationary_boot(len(B), a.boot, L_BOOT, rng)
    ks = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    out = dict(n=len(B), fr={NAMES[hi]: fr_scan(x, s, lab, hi, 0, idx, ks) for hi in (2, 1)})
    # check against script 31
    d31 = json.load(open(os.path.join(HERE, "out", "31", "regimes_contagion.json")))["btc_long"]["forbes_rigobon"]
    ref = d31["BTC|crisis_vs_tranquil"]
    one = [r for r in out["fr"]["crisis"]["scan"] if r["k"] == 1.0][0]
    out["check_script31"] = dict(diff=abs(one["diff"] - ref["diff_adj"]) < 1e-9,
                                 ci_close=bool(np.allclose(one["ci"], ref["ci"], atol=0.01)))
    yrs = B.index.year
    out["slopes"] = {}
    for nm, m in (("2014–2026", np.ones(len(B), bool)), ("2014–2019", yrs <= 2019), ("2020–2026", yrs >= 2020)):
        out["slopes"][nm] = dict(plain=regime_slopes(x[m], s[m], lab[m]), dimson=regime_slopes(x[m], s[m], lab[m], True))
    json.dump(out, open(os.path.join(a.outdir, "contagion_sensitivity.json"), "w"), indent=1)
    L = ["# Contagion sensitivity (script 44)\n", "## A. Forbes–Rigobon adjustment with scaled delta (Bitcoin long sample)\n"]
    for hi, r in out["fr"].items():
        L.append(f"**{hi} vs tranquil** (days {r['n_hi']} / {r['n_lo']}; rho {r['rho_hi']:.3f} vs {r['rho_lo']:.3f}; "
                 f"delta {r['delta']:.2f}; MAD-based delta {r['delta_mad']:.2f})\n")
        L += ["| k (delta multiplier) | rho* − rho tranquil | 95% CI |", "|---|---|---|"]
        L += [f"| {q['k']:.2f} | {q['diff']:+.3f} | [{q['ci'][0]:+.3f}, {q['ci'][1]:+.3f}] |" for q in r["scan"]]
        L.append(f"| MAD-based delta | {r['mad']['diff']:+.3f} | [{r['mad']['ci'][0]:+.3f}, {r['mad']['ci'][1]:+.3f}] |")
        L.append(f"\nLargest multiplier k ≤ 1 at which the increase is significant: {r['k_breakeven']}\n")
    L += ["## B. Slopes of BTC-USD on SPY by regime (Newey–West, 5 lags)\n",
          "| Period | Version | Tranquil | Turbulent | Crisis | Turbulent − tranquil [p] | Crisis − tranquil [p] |",
          "|---|---|---|---|---|---|---|"]
    for per, v in out["slopes"].items():
        for ver in ("plain", "dimson"):
            q = v[ver]
            L.append(f"| {per} | {ver} | " + " | ".join(f"{q[r]['slope']:.2f} ({q[r]['se']:.2f}; {q[r]['n']})"
                                                        for r in ("tranquil", "turbulent", "crisis")) +
                     f" | {q['turbulent_minus_tranquil']['diff']:+.2f} [{q['turbulent_minus_tranquil']['p']:.3f}] | "
                     f"{q['crisis_minus_tranquil']['diff']:+.2f} [{q['crisis_minus_tranquil']['p']:.3f}] |")
    L.append(f"\nCHECK {'PASS' if out['check_script31']['diff'] else 'FAIL'}: k = 1 reproduces script 31's adjusted "
             f"difference; interval within 0.01: {out['check_script31']['ci_close']}")
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "contagion_sensitivity.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
