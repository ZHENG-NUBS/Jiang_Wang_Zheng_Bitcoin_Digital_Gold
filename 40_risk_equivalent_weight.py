# -*- coding: utf-8 -*-
"""
40_risk_equivalent_weight.py — definition, existence and stability of the risk-equivalent weight w*.

Target. w* is the IBIT (or BTC-USD) weight g in the portfolio (g, (1-g)/2 SPY, (1-g)/2 TLT), rebalanced daily, at
which the portfolio's historical ES95 equals that of the fixed benchmark (1/3 GLD, 1/3 SPY, 1/3 TLT).
Domain and root. g runs over [0, 0.5] in steps of 0.005; ES is computed at each grid point and w* is the crossing on
the increasing branch of the curve (at or above the ES-minimizing weight), located by linear interpolation. Because
ES is convex in the portfolio weights, the curve is convex along this path and has at most one crossing on that branch.
Non-crossing. A resample has no root if the curve lies above the target on the whole domain (the ES-minimizing
portfolio is already riskier than the benchmark: gold lowers ES by more than any Bitcoin weight can match), or below
it (a root above 0.5). Both are counted separately, together with resamples with a crossing on the decreasing branch.

The script reproduces the point estimate and the conditional percentile interval of script 24 with the same resamples,
reports the share of resamples in each category, inverts the bootstrap distribution of the ES difference to obtain
an unconditional confidence set (the weights at which the 95% interval of ES(g) - ES(benchmark) contains zero), and
estimates w* on rolling 250-day windows.

Usage: python 40_risk_equivalent_weight.py [--outdir out/40]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402

spec = importlib.util.spec_from_file_location("m24", os.path.join(HERE, "24_controlled_comparison.py"))
m24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m24)

GRID = np.round(np.arange(0.0, 0.5001, 0.005), 4)


def curve(R3, idx=None):
    if idx is None:
        return np.array([m24.es(R3 @ m24.wv(g)) for g in GRID])
    Rb = R3[idx]
    return np.array([m24.es(Rb @ m24.wv(g)) for g in GRID])          # (G, B)


def root(cv, target):
    """(w*, category): category 'root', 'above' (curve above target everywhere), 'below' (root beyond 0.5)."""
    i0 = int(np.argmin(cv))
    if cv[i0] > target:
        return np.nan, "above"
    for j in range(i0, len(cv) - 1):
        if cv[j] <= target <= cv[j + 1]:
            return GRID[j] + (target - cv[j]) / (cv[j + 1] - cv[j]) * (GRID[j + 1] - GRID[j]), "root"
    return np.nan, "below"


def analyse(X, G, SPY, TLT, idx):
    R3i = np.stack([X, SPY, TLT], -1)
    R3g = np.stack([G, SPY, TLT], -1)
    tgt = m24.es(R3g @ m24.wv(1 / 3))
    cv = curve(R3i)
    w0, cat0 = root(cv, tgt)
    cvb = curve(R3i, idx)
    tb = m24.es(R3g[idx] @ m24.wv(1 / 3))
    rb = [root(cvb[:, b], tb[b]) for b in range(idx.shape[1])]
    ws = np.array([r[0] for r in rb])
    cats = pd.Series([r[1] for r in rb]).value_counts(normalize=True).to_dict()
    decreasing = float(np.mean([cvb[0, b] > tb[b] and cvb[:, b].min() <= tb[b] for b in range(idx.shape[1])]))
    diff = cvb - tb[None, :]                                          # (G, B)
    lo, hi = np.percentile(diff, 2.5, axis=1), np.percentile(diff, 97.5, axis=1)
    inset = GRID[(lo <= 0) & (hi >= 0)]
    return dict(target=float(tgt), es_min=float(cv.min()), w_es_min=float(GRID[int(np.argmin(cv))]),
                w_star=float(w0), category=cat0, shares=cats, share_root=float(np.mean(np.isfinite(ws))),
                ci_conditional=m24.pct(ws), share_decreasing_branch_crossing=decreasing,
                inverted_set=[float(inset.min()), float(inset.max())] if len(inset) else None,
                inverted_set_contiguous=bool(len(inset) == 0 or np.allclose(np.diff(inset), 0.005)),
                es_at_zero=float(cv[0]), es_at_half=float(cv[-1]))


def rolling(X, G, SPY, TLT, dates, win=250, step=21):
    out = []
    for e in range(win, len(X) + 1, step):
        s = e - win
        R3i = np.stack([X[s:e], SPY[s:e], TLT[s:e]], -1)
        R3g = np.stack([G[s:e], SPY[s:e], TLT[s:e]], -1)
        w_, c_ = root(curve(R3i), m24.es(R3g @ m24.wv(1 / 3)))
        out.append(dict(end=str(dates[e - 1].date()), w_star=float(w_), category=c_))
    return out


def summary(rl):
    ws = np.array([r["w_star"] for r in rl])
    f = ws[np.isfinite(ws)]
    return dict(windows=len(rl), share_root=float(np.mean(np.isfinite(ws))),
                median=float(np.median(f)) if len(f) else None,
                p10=float(np.percentile(f, 10)) if len(f) else None, p90=float(np.percentile(f, 90)) if len(f) else None,
                min=float(f.min()) if len(f) else None, max=float(f.max()) if len(f) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/40")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = LD.load_etf()
    n = len(E)
    idx = m24.stationary_boot(n, m24.B_MAIN, m24.L_BOOT, np.random.default_rng(m24.SEED))
    res = dict(etf=analyse(E.IBIT.values, E.GLD.values, E.SPY.values, E.TLT.values, idx))
    res["etf"]["rolling"] = rolling(E.IBIT.values, E.GLD.values, E.SPY.values, E.TLT.values, E.index)
    res["etf"]["rolling_summary"] = summary(res["etf"]["rolling"])
    B = LD.load_btc_long()
    Bx = B.BTC.values - 0.0025 / 252                                   # as script 24: fee-adjusted spot price
    nl = len(B)
    idxl = m24.stationary_boot(nl, m24.B_LONG, m24.L_BOOT, np.random.default_rng(m24.SEED + 7 * nl))
    res["long"] = analyse(Bx, B.GLD.values, B.SPY.values, B.TLT.values, idxl)
    res["long"]["rolling"] = rolling(Bx, B.GLD.values, B.SPY.values, B.TLT.values, B.index)
    res["long"]["rolling_summary"] = summary(res["long"]["rolling"])
    json.dump(res, open(os.path.join(a.outdir, "risk_equivalent_weight.json"), "w"), indent=1, default=float)
    for k in ("etf", "long"):
        r = {kk: v for kk, v in res[k].items() if kk != "rolling"}
        print(k, json.dumps(r, indent=1, default=float))
    yr = pd.DataFrame(res["long"]["rolling"])
    yr["year"] = yr.end.str[:4]
    print(yr.groupby("year").agg(share_root=("w_star", lambda s: np.isfinite(s).mean()),
                                 median=("w_star", "median")).round(3).to_string())


if __name__ == "__main__":
    main()
