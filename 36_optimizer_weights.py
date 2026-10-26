#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
36_optimizer_weights.py — Out-of-sample weight distributions of the fifteen strategies and the effect
of the maximum-Sharpe fallback rule (second-round revision: Reviewer 1, comment 7; Reviewer 2, comment 3)

Uses the strategy code of script 23 (imported) in its main settings: 250-day rolling estimation
window, rebalancing every 21 trading days, long-only, 10 bp per unit traded notional.

A  Allocation: distribution over rebalancing dates of the Bitcoin weight (IBIT in the ETF sample,
   BTC-USD in the long sample) for each strategy (mean, standard deviation, 5th/50th/95th
   percentiles, share of rebalances with a weight below 1%), and the mean weight of each asset.
B  Risk and return, reported separately: annualized volatility, ES95 and maximum drawdown (risk);
   annualized mean, Sharpe ratio and certainty-equivalent return (return), each net of costs, with
   the Sharpe difference from 1/N (HAC, Holm-adjusted over the fourteen comparisons).
C  The fallback rule: when no long-only portfolio in the window has a positive expected excess
   return, the maximum-Sharpe problem has no meaningful solution and script 23 uses the
   minimum-variance portfolio. Counts of such rebalances, and the maximum-Sharpe strategy computed
   (i) with that rule, (ii) with 1/N as the fallback and (iii) without a fallback (the
   raw maximiser of the Sharpe ratio, which then loads on the asset with the least negative ratio).

Usage
    python 36_optimizer_weights.py               # writes out/36/
    python 36_optimizer_weights.py --quick
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
SEED = 20261026


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m23 = load("m23", "23_portfolio_robust.py")
warnings.filterwarnings("ignore")


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def drift_returns(R, start, reb, WL):
    """Same holding logic as script 23's run_setting for an arbitrary list of rebalance weights."""
    T = R.shape[0]; n = T - start
    pr = np.empty(n); trade = np.zeros(n); w_cur = None; k = 0
    for i in range(n):
        t = start + i
        if k < len(reb) and t == reb[k]:
            w_new = WL[k]
            trade[i] = float(np.abs(w_new - (w_cur if w_cur is not None else 0)).sum())
            w_cur = w_new; k += 1
        pr[i] = float(w_cur @ R[t])
        gr = w_cur * (1 + R[t]); w_cur = gr / gr.sum()
    return dict(gross=pr, trade=trade, w_reb=np.array(WL), reb_pos=np.array(reb) - start)


def msr_variants(R, rf, start, win, rebal):
    reb = list(range(start, R.shape[0], rebal))
    V = {"gmv": [], "ew": [], "raw": []}; neg = []
    for t in reb:
        X = R[t - win:t]
        mu = X.mean(0); S = np.cov(X, rowvar=False)
        v, sr = m23.max_sharpe_raw(mu, S, rf[t])
        neg.append(bool(sr <= 0))
        V["raw"].append(v)
        V["gmv"].append(v if sr > 0 else m23.gmv(S))
        V["ew"].append(v if sr > 0 else np.ones(R.shape[1]) / R.shape[1])
    return {k: drift_returns(R, start, reb, w) for k, w in V.items()}, neg, reb


def block(label, R, rf, dates, assets, start, win, rebal, rng, B_res, B_test):
    o, info = m23.run_setting(R, rf, start, win, rebal, rng, B_res, 2, 3)
    rfo = rf[start:]
    bp = m23.MAIN_BP
    out = {"label": label, "oos": [str(dates[start].date()), str(dates[-1].date())], "n_reb": len(info)}
    # A. allocation
    rowsA, A = [], {}
    for c in m23.CODES:
        W = o[c]["w_reb"]; w0 = W[:, 0]
        A[c] = dict(mean=float(w0.mean()), sd=float(w0.std()), p5=float(np.percentile(w0, 5)),
                    p50=float(np.median(w0)), p95=float(np.percentile(w0, 95)), lt1=float((w0 < 0.01).mean()),
                    mean_all={a_: float(W[:, j].mean()) for j, a_ in enumerate(assets)})
        rowsA.append([m23.GRP[c], m23.NAME[c], f"{A[c]['mean'] * 100:.1f}", f"{A[c]['sd'] * 100:.1f}",
                      f"{A[c]['p5'] * 100:.1f} / {A[c]['p50'] * 100:.1f} / {A[c]['p95'] * 100:.1f}",
                      f"{A[c]['lt1'] * 100:.0f}%"] + [f"{A[c]['mean_all'][a_] * 100:.0f}" for a_ in assets])
    out["allocation"] = A
    # B. risk and return
    n = len(rfo)
    idx = m23.stationary_boot(n, B_test, m23.L_BOOT, rng)
    ew = m23.net(o["EW"], bp)
    M = {c: m23.metrics(m23.net(o[c], bp), rfo, o[c]) for c in m23.CODES}
    C = {c: m23.compare(m23.net(o[c], bp), ew, rfo, idx) for c in m23.CODES if c != "EW"}
    cs = list(C); h = m23.holm([C[c]["p_hac"] for c in cs])
    for k, c in enumerate(cs):
        C[c]["p_hac_holm"] = float(h[k])
    rowsB = []
    for c in m23.CODES:
        m = M[c]
        rowsB.append([m23.GRP[c], m23.NAME[c], f"{m['vol'] * 100:.1f}", f"{m['es95'] * 100:.2f}", f"{m['mdd'] * 100:.1f}",
                      f"{m['ret_a'] * 100:.1f}", f"{m['sharpe']:.2f}", f"{m['cer'] * 100:.1f}",
                      "—" if c == "EW" else f"{C[c]['dsr']:+.2f} ({m23.pf(C[c]['p_hac'])}; Holm {m23.pf(C[c]['p_hac_holm'])})"])
    out["metrics"] = M; out["vs_ew"] = C
    # C. fallback
    V, neg, reb = msr_variants(R, rf, start, win, rebal)
    rowsC = []
    out["fallback"] = {"n_neg": int(sum(neg)), "n_reb": len(neg),
                       "dates_neg": [str(dates[t].date()) for t, g in zip(reb, neg) if g]}
    for k, nm in (("gmv", "Fallback to min variance (script 23)"), ("ew", "Fallback to 1/N"), ("raw", "No fallback")):
        r = m23.net(V[k], bp)
        m = m23.metrics(r, rfo, V[k])
        cmp_ = m23.compare(r, ew, rfo, idx)
        out["fallback"][k] = dict(metrics=m, vs_ew=cmp_)
        rowsC.append([nm, f"{m['w0_mean'] * 100:.1f}", f"{m['vol'] * 100:.1f}", f"{m['es95'] * 100:.2f}",
                      f"{m['sharpe']:.2f}", f"{m['cer'] * 100:.1f}", f"{cmp_['dsr']:+.2f} ({m23.pf(cmp_['p_hac'])})"])
    gm = V["gmv"]["w_reb"]; ok = np.allclose(gm, o["MSR-S"]["w_reb"], atol=1e-8)
    out["fallback"]["matches_script23"] = bool(ok)
    L = [f"## {label}: out of sample {out['oos'][0]} → {out['oos'][1]}, {out['n_reb']} rebalances "
         f"(window {win}, every {rebal} days, {bp} bp)\n",
         f"### A. Allocation: {assets[0]} weight across rebalances, and mean weight of each asset (%)\n",
         md(rowsA, ["Group", "Strategy", "Mean", "SD", "P5 / median / P95", "Share < 1%"] + [f"{a_} mean" for a_ in assets]),
         "### B. Risk (net) and return (net), separately\n",
         md(rowsB, ["Group", "Strategy", "Vol %", "ES95 %", "Max DD %", "Mean %", "Sharpe", "CER %",
                    "ΔSharpe vs 1/N (p HAC; Holm)"]),
         f"### C. Maximum-Sharpe fallback: {out['fallback']['n_neg']} of {len(neg)} rebalances had no long-only "
         f"portfolio with a positive expected excess return (identical to script 23: {'yes' if ok else 'NO'})\n",
         md(rowsC, ["Max Sharpe (sample) variant", f"{assets[0]} mean weight %", "Vol %", "ES95 %", "Sharpe", "CER %",
                    "ΔSharpe vs 1/N (p HAC)"])]
    return out, "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/36")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    B_res, B_test = (100, 500) if a.quick else (m23.B_RES_OOS, m23.B_TEST)
    rng = np.random.default_rng(SEED)
    data = m23.find_main(); raw = m23.find_raw()
    d = pd.read_csv(data, parse_dates=["Date"]).set_index("Date").sort_index()
    A_M = ["IBIT", "GLD", "SPY", "TLT"]
    OUT = {}; txt = ["# Optimizer weights, risk/return separation and the maximum-Sharpe fallback (script 36)\n"]
    o, t = block("ETF sample", d[A_M].values, np.full(len(d), m23.RF_MAIN / 252), d.index, A_M, 250, 250, 21,
                 rng, B_res, B_test)
    OUT["etf"] = o; txt.append(t)
    if raw is not None:
        RL, rfl, _ = m23.load_long(raw)
        A_L = ["BTC", "GLD", "SPY", "TLT"]
        o, t = block("Bitcoin long sample", RL[A_L].values, rfl, RL.index, A_L, 500, 250, 21, rng, B_res, B_test)
        OUT["btc_long"] = o; txt.append(t)
    s = "\n".join(txt)
    print(s)
    open(os.path.join(a.outdir, "optimizer_weights.md"), "w", encoding="utf-8").write(s)
    json.dump(OUT, open(os.path.join(a.outdir, "optimizer_weights.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
