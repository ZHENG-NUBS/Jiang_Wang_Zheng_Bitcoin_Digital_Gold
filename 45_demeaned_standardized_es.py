# -*- coding: utf-8 -*-
"""
45_demeaned_standardized_es.py — does the standardized ES component of Table 6 reflect location as well as shape?

Script 34 divides each portfolio's returns by their sample standard deviation s, so its second component,
ES(r/s)_B / ES(r/s)_G, also reflects the mean relative to the volatility. Removing the mean first gives
ES((r - mean)/s) = (ES(r) + mean)/s, with ES reported as a positive loss. This script reports the demeaned
standardized ES ratio at 95%, 97.5% and 99% with the same stationary-bootstrap draws as script 34 (5000 replications),
its 90% and 95% intervals, the equivalence test against the margins of script 34 and the smallest margin at which
equivalence would be concluded.

Usage: python 45_demeaned_standardized_es.py [--outdir out/45] [--boot 5000]
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

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m34 = load("m34", "34_es_decomposition_equivalence.py")
ALPHAS = m34.ALPHAS


def es(x, a):
    """Point estimate as in scripts 11 and 34: mean loss at or beyond the empirical alpha-quantile."""
    v = np.quantile(x, a)
    return -x[x <= v].mean()


def analyse(rA, rB, IDX, margins):
    out = {}
    B = IDX.shape[1]
    stats = {k: [] for k in ("mA", "mB", "sA", "sB")}
    E = {a: ([], []) for a in ALPHAS}
    for c0 in range(0, B, 1000):
        I = IDX[:, c0:c0 + 1000]
        SA, SB = rA[I], rB[I]
        stats["mA"].append(SA.mean(0)); stats["mB"].append(SB.mean(0))
        stats["sA"].append(SA.std(0, ddof=1)); stats["sB"].append(SB.std(0, ddof=1))
        for a in ALPHAS:
            E[a][0].append(m34.boot_es(SA, a)); E[a][1].append(m34.boot_es(SB, a))
    st = {k: np.concatenate(v) for k, v in stats.items()}
    for a in ALPHAS:
        eA, eB = np.concatenate(E[a][0]), np.concatenate(E[a][1])
        raw = (eA / st["sA"]) / (eB / st["sB"])
        dm = ((eA + st["mA"]) / st["sA"]) / ((eB + st["mB"]) / st["sB"])
        pr = (es(rA, a) / rA.std(ddof=1)) / (es(rB, a) / rB.std(ddof=1))
        pd_ = ((es(rA, a) + rA.mean()) / rA.std(ddof=1)) / ((es(rB, a) + rB.mean()) / rB.std(ddof=1))
        lo90, hi90 = np.percentile(dm, [5, 95])
        c = margins[f"{a}"]
        out[f"{a}"] = dict(shape_ratio=float(pr), demeaned=float(pd_), demeaned_ci95=[float(v) for v in np.percentile(dm, [2.5, 97.5])],
                           demeaned_ci90=[float(lo90), float(hi90)], margin=c,
                           equivalent=bool(lo90 > 1 / c and hi90 < c), smallest_margin=float(max(hi90, 1 / lo90)),
                           p_above_margin=float(np.mean(dm >= c)), raw_ci90_check=[float(v) for v in np.percentile(raw, [5, 95])])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/45")
    ap.add_argument("--boot", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    m11 = load("m11", "11_es_bootstrap_vix.py")
    m22 = load("m22", "22_safe_haven_dependence.py")
    d34 = json.load(open(os.path.join(HERE, "out", "34", "es_decomposition.json")))
    margins = d34["delta_ratio"]
    # ETF sample: the same draws as scripts 11 and 34
    d = pd.read_csv(m22.find_main(), parse_dates=["Date"]).sort_values("Date").set_index("Date")
    n = len(d)
    P = {"IBIT portfolio": ((d.IBIT + d.SPY + d.TLT) / 3).values, "GLD portfolio": ((d.GLD + d.SPY + d.TLT) / 3).values}
    rng = np.random.default_rng(20260922)
    Lopt = {k: m11.opt_block_length(r) for k, r in P.items()}
    for k in P:
        for L in sorted({5, 10, 20, int(round(Lopt[k]))}):
            m11.stationary_boot(n, a.boot, L, rng)
    L = int(round(np.mean(list(Lopt.values()))))
    IDX = m11.stationary_boot(n, a.boot, L, rng)
    out = dict(etf=analyse(P["IBIT portfolio"], P["GLD portfolio"], IDX, margins))
    del IDX
    RL = m22.load_long(m22.find_raw_L())
    rA = ((RL.BTC + RL.SPY + RL.TLT) / 3).values
    rB = ((RL.GLD + RL.SPY + RL.TLT) / 3).values
    LL = max(2, int(round(np.mean([m11.opt_block_length(rA), m11.opt_block_length(rB)]))))
    IDXL = m11.stationary_boot(len(rA), a.boot, LL, np.random.default_rng(20261021))
    out["long"] = analyse(rA, rB, IDXL, margins)
    # checks against script 34: same point estimate and the same 90% interval of the (non-demeaned) shape ratio
    chk = []
    for s in ("etf", "long"):
        for al in ("0.05", "0.025", "0.01"):
            x, y = out[s][al], d34[s][al]
            chk.append(abs(x["shape_ratio"] - y["shape_ratio"]) < 1e-9 and np.allclose(x["raw_ci90_check"], y["shape_ratio_ci90"], atol=1e-9))
    out["check_script34"] = bool(all(chk))
    json.dump(out, open(os.path.join(a.outdir, "demeaned_standardized_es.json"), "w"), indent=1)
    Ls = ["# Demeaned standardized ES ratio (script 45)\n",
          "| Sample | Level | Standardized ES ratio (Table 6) | Demeaned | 90% CI | Margin c | Equivalent | Smallest margin |",
          "|---|---|---|---|---|---|---|---|"]
    for s in ("etf", "long"):
        for al, lab in (("0.05", "95%"), ("0.025", "97.5%"), ("0.01", "99%")):
            x = out[s][al]
            Ls.append(f"| {s} | {lab} | {x['shape_ratio']:.3f} | {x['demeaned']:.3f} | [{x['demeaned_ci90'][0]:.3f}, "
                      f"{x['demeaned_ci90'][1]:.3f}] | {x['margin']:.3f} | {'yes' if x['equivalent'] else 'no'} | {x['smallest_margin']:.3f} |")
    Ls.append(f"\nCHECK {'PASS' if out['check_script34'] else 'FAIL'}: point estimates and 90% intervals of the shape ratio reproduce script 34")
    txt = "\n".join(Ls)
    open(os.path.join(a.outdir, "demeaned_standardized_es.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
