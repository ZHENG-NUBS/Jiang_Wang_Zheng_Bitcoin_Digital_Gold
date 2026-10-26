#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
34_es_decomposition_equivalence.py — Exact volatility/shape decomposition of the ES ratio and an
equivalence test for the shape term (second-round revision, Reviewer 1, comment 2)

Decomposition
    For a portfolio return r with sample standard deviation s, the historical-simulation ES of
    the standardized series z = r / s equals ES(r) / s, because ES is positively homogeneous.
    Hence, exactly,
        log(ES_A / ES_B) = log(s_A / s_B) + log(ES_z,A / ES_z,B)
    (volatility term + shape term). The decomposition in the first revision standardized by
    GARCH volatility, for which the identity holds only approximately.

Equivalence test for the shape term (two one-sided tests, TOST)
    H0: |shape term| >= delta against H1: |shape term| < delta at the 5% level, by the bootstrap
    percentile method: equivalence is concluded if the 90% interval of the shape term lies inside
    (-delta, delta). The margin is delta = log(c), where c is the ratio of the standardized ES of a
    Student-t distribution with 4 degrees of freedom to that of the normal distribution at the same
    level (c = 1.10 at 95%): a shape difference counts as small if it is smaller than the
    difference between normal tails and very heavy t(4) tails at equal volatility. The script also
    reports the smallest margin at which equivalence would be concluded.

Common normalizations (supplementary)
    Shape ratio after standardizing each portfolio by its own conditional volatility from
    (i) EWMA (lambda = 0.94) and (ii) GARCH(1,1)-t. These intervals are conditional on the
    first-stage volatility estimates.

Samples
    ETF sample: IBIT or GLD with SPY and TLT, one-third each, daily rebalanced (607 days). The
    bootstrap draws are those of script 11 (same seed and sequence), so the ES-ratio interval equals
    the one in the paper. Long sample: BTC-USD spot instead of IBIT, September 2014 - June 2026
    (snapshot __20260921L), constructed as in script 22.

Usage
    python 34_es_decomposition_equivalence.py               # writes out/34/
    python 34_es_decomposition_equivalence.py --boot 1000   # quicker trial
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHAS = (0.05, 0.01)
CHUNK = 1000


def load(name, fname):
    p = os.path.join(HERE, fname)
    if not os.path.exists(p):
        sys.exit(f"{fname} must be in the same folder as this script.")
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def std_es_ratio_t_normal(alpha, nu=4.0):
    """Standardized ES of Student-t(nu) divided by that of the normal, at tail probability alpha."""
    q = stats.t.ppf(alpha, nu)
    es_t = stats.t.pdf(q, nu) / alpha * (nu + q ** 2) / (nu - 1) * math.sqrt((nu - 2) / nu)
    es_n = stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
    return es_t / es_n


def boot_es(S, alpha):
    """ES of each bootstrap column: mean of the floor(alpha*n) worst returns (as in script 11)."""
    n = S.shape[0]
    m = max(1, int(math.floor(alpha * n)))
    return -np.sort(S, axis=0)[:m, :].mean(axis=0)


def boot_stats(rA, rB, IDX):
    """Bootstrap ES (both levels) and standard deviations, computed in column chunks."""
    B = IDX.shape[1]
    o = {k: np.empty(B) for k in ("sdA", "sdB")}
    for a in ALPHAS:
        o[f"esA{a}"] = np.empty(B)
        o[f"esB{a}"] = np.empty(B)
    for c0 in range(0, B, CHUNK):
        I = IDX[:, c0:c0 + CHUNK]
        SA, SB = rA[I], rB[I]
        o["sdA"][c0:c0 + CHUNK] = SA.std(axis=0, ddof=1)
        o["sdB"][c0:c0 + CHUNK] = SB.std(axis=0, ddof=1)
        for a in ALPHAS:
            o[f"esA{a}"][c0:c0 + CHUNK] = boot_es(SA, a)
            o[f"esB{a}"][c0:c0 + CHUNK] = boot_es(SB, a)
    return o


def ewma_sigma(r, lam=0.94, init=60):
    s2 = np.empty(len(r))
    s2[0] = np.var(r[:init], ddof=1)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(s2)


def garch_std(r):
    from arch import arch_model
    res = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    return (r * 100 - res.params["mu"]) / res.conditional_volatility


def analyse(label, rA, rB, IDX, delta, hs_es):
    n = len(rA)
    out = {"label": label, "n": n}
    bs = boot_stats(rA, rB, IDX)
    bvol = np.log(bs["sdA"] / bs["sdB"])
    for a in ALPHAS:
        esA, esB = hs_es(rA, a), hs_es(rB, a)
        s_A, s_B = rA.std(ddof=1), rB.std(ddof=1)
        tot, vol = math.log(esA / esB), math.log(s_A / s_B)
        shp = tot - vol                                  # exact
        assert abs(shp - math.log(hs_es(rA / s_A, a) / hs_es(rB / s_B, a))) < 1e-12
        btot = np.log(bs[f"esA{a}"] / bs[f"esB{a}"])
        bshp = btot - bvol
        lo90, hi90 = np.percentile(bshp, [5, 95])
        lo95, hi95 = np.percentile(bshp, [2.5, 97.5])
        d = delta[a]
        out[f"{a}"] = {
            "es_A": esA, "es_B": esB, "ratio": esA / esB,
            "ratio_ci": list(np.exp(np.percentile(btot, [2.5, 97.5]))),
            "vol_ratio": s_A / s_B, "vol_ratio_ci": list(np.exp(np.percentile(bvol, [2.5, 97.5]))),
            "shape_ratio": math.exp(shp), "shape_ratio_ci95": [math.exp(lo95), math.exp(hi95)],
            "shape_ratio_ci90": [math.exp(lo90), math.exp(hi90)],
            "vol_share": vol / tot, "shape_share": shp / tot,
            "vol_share_ci": list(np.percentile(bvol / btot, [2.5, 97.5])),
            "shape_share_ci": list(np.percentile(bshp / btot, [2.5, 97.5])),
            "margin": math.exp(d), "equivalent": bool(-d < lo90 and hi90 < d),
            "smallest_margin": math.exp(max(abs(lo90), abs(hi90))),
            "share_boot_shape_above_margin": float((bshp >= d).mean()),
            "share_boot_shape_below_minus_margin": float((bshp <= -d).mean()),
            "n_tail": int(max(1, math.floor(a * n))),
        }
    return out


def shape_common(rA, rB, IDX, hs_es):
    """Shape ratio after standardizing by each portfolio's own conditional volatility."""
    res = {}
    for meth in ("EWMA", "GARCH-t"):
        if meth == "EWMA":
            zA, zB = rA / ewma_sigma(rA), rB / ewma_sigma(rB)
        else:
            zA, zB = garch_std(rA), garch_std(rB)
        bs = boot_stats(zA, zB, IDX)
        res[meth] = {}
        for a in ALPHAS:
            pt = hs_es(zA, a) / hs_es(zB, a)
            b = bs[f"esA{a}"] / bs[f"esB{a}"]
            res[meth][f"{a}"] = {"shape_ratio": pt, "ci95": list(np.percentile(b, [2.5, 97.5]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--outdir", default="out/34")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    m11 = load("m11", "11_es_bootstrap_vix.py")
    m22 = load("m22", "22_safe_haven_dependence.py")
    data = a.data or m22.find_main()
    raw = a.raw or m22.find_raw_L()
    if data is None:
        sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; pass --data.")
    os.makedirs(a.outdir, exist_ok=True)
    delta = {al: math.log(std_es_ratio_t_normal(al)) for al in ALPHAS}

    # ---------------- ETF sample, with the bootstrap draws of script 11 (A2 then A3)
    d = pd.read_csv(data, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    n = len(d)
    P = {"IBIT portfolio": ((d.IBIT + d.SPY + d.TLT) / 3).values,
         "GLD portfolio": ((d.GLD + d.SPY + d.TLT) / 3).values}
    rng = np.random.default_rng(20260922)
    Lopt = {k: m11.opt_block_length(r) for k, r in P.items()}
    for k in P:
        for L in sorted({5, 10, 20, int(round(Lopt[k]))}):
            m11.stationary_boot(n, a.boot, L, rng)          # consume draws exactly as script 11 does
    L = int(round(np.mean(list(Lopt.values()))))
    IDX = m11.stationary_boot(n, a.boot, L, rng)
    OUT = {"delta_ratio": {f"{al}": math.exp(delta[al]) for al in ALPHAS}, "boot": a.boot}
    OUT["etf"] = analyse("ETF sample (IBIT vs GLD)", P["IBIT portfolio"], P["GLD portfolio"], IDX, delta, m11.hs_es)
    OUT["etf"]["block_length"] = L
    OUT["etf"]["common"] = shape_common(P["IBIT portfolio"], P["GLD portfolio"], IDX, m11.hs_es)
    del IDX

    # ---------------- long sample (BTC-USD spot)
    if raw:
        RL = m22.load_long(raw)
        rA = ((RL.BTC + RL.SPY + RL.TLT) / 3).values
        rB = ((RL.GLD + RL.SPY + RL.TLT) / 3).values
        nL = len(rA)
        LL = max(2, int(round(np.mean([m11.opt_block_length(rA), m11.opt_block_length(rB)]))))
        rngL = np.random.default_rng(20261021)
        IDXL = m11.stationary_boot(nL, a.boot, LL, rngL)
        OUT["long"] = analyse("Long sample (BTC-USD vs GLD)", rA, rB, IDXL, delta, m11.hs_es)
        OUT["long"]["block_length"] = LL
        OUT["long"]["dates"] = [str(RL.index[0].date()), str(RL.index[-1].date())]
        OUT["long"]["common"] = shape_common(rA, rB, IDXL, m11.hs_es)

    # ---------------- report
    L_ = ["# Exact ES decomposition and equivalence test of the shape term (script 34)\n",
          f"Margin: standardized ES of t(4) / normal = {OUT['delta_ratio']['0.05']:.3f} at 95% and "
          f"{OUT['delta_ratio']['0.01']:.3f} at 99%. Bootstrap B = {a.boot}.\n"]
    for key in ("etf", "long"):
        if key not in OUT:
            continue
        o = OUT[key]
        L_.append(f"## {o['label']} — {o['n']} days, block length {o['block_length']}\n")
        L_.append("| Level | ES ratio [95% CI] | Volatility ratio [95% CI] | Shape ratio [95% CI] | "
                  "Shape ratio, 90% CI | Volatility / shape share of log ratio | Equivalent at margin? | "
                  "Smallest margin | Tail obs. |")
        L_.append("|---|---|---|---|---|---|---|---|---|")
        for al in ALPHAS:
            x = o[f"{al}"]
            L_.append(f"| {int((1 - al) * 100)}% | {x['ratio']:.3f} [{x['ratio_ci'][0]:.2f}, {x['ratio_ci'][1]:.2f}] | "
                      f"{x['vol_ratio']:.3f} [{x['vol_ratio_ci'][0]:.2f}, {x['vol_ratio_ci'][1]:.2f}] | "
                      f"{x['shape_ratio']:.3f} [{x['shape_ratio_ci95'][0]:.2f}, {x['shape_ratio_ci95'][1]:.2f}] | "
                      f"[{x['shape_ratio_ci90'][0]:.3f}, {x['shape_ratio_ci90'][1]:.3f}] | "
                      f"{x['vol_share'] * 100:.0f}% / {x['shape_share'] * 100:.0f}% | "
                      f"{'yes' if x['equivalent'] else 'no'} (±{(x['margin'] - 1) * 100:.0f}%) | "
                      f"±{(x['smallest_margin'] - 1) * 100:.1f}% | {x['n_tail']} |")
        L_.append("")
        L_.append("Shape ratio after standardizing by each portfolio's own conditional volatility "
                  "(conditional on the first-stage estimates):\n")
        L_.append("| Normalization | Shape ratio ES95 [95% CI] | Shape ratio ES99 [95% CI] |")
        L_.append("|---|---|---|")
        for meth, v in o["common"].items():
            L_.append(f"| {meth} | {v['0.05']['shape_ratio']:.3f} [{v['0.05']['ci95'][0]:.2f}, {v['0.05']['ci95'][1]:.2f}] | "
                      f"{v['0.01']['shape_ratio']:.3f} [{v['0.01']['ci95'][0]:.2f}, {v['0.01']['ci95'][1]:.2f}] |")
        L_.append("")
    e = OUT["etf"]["0.05"]
    chk = [("ETF ES95 ratio equals the paper (1.633)", abs(e["ratio"] - 1.633) < 0.0006),
           ("ETF ES95 ratio interval equals script 11 ([1.36, 1.99])",
            f"{e['ratio_ci'][0]:.2f}" == "1.36" and f"{e['ratio_ci'][1]:.2f}" == "1.99"),
           ("Decomposition exact",
            abs(math.log(e["ratio"]) - math.log(e["vol_ratio"]) - math.log(e["shape_ratio"])) < 1e-12)]
    L_.append("## Checks\n")
    for lab, ok in chk:
        L_.append(f"   CHECK [{'PASS' if ok else 'FAIL'}] {lab}")
    txt = "\n".join(L_) + "\n"
    print(txt)
    open(os.path.join(a.outdir, "es_decomposition.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "es_decomposition.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
