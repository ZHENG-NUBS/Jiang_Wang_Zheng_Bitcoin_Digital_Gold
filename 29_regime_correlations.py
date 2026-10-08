#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
29_regime_correlations.py — Correlation matrix of the four ETFs by VIX regime (Table A2)

The VIX regimes are those of Table 5: low (VIX <= 15), medium (15 < VIX <= 25) and high
(VIX > 25), using the same-day VIX close; results with the previous day's VIX are written to
the Markdown output as a robustness check. For every pair of IBIT, GLD, TLT and SPY the script
reports the Pearson correlation of daily simple returns in each regime and in the full sample,
with 95% stationary-block-bootstrap confidence intervals, and the high-minus-low difference
with its interval and two-sided p-value. The regimes are re-assigned in every bootstrap
replication, as in script 19, and the bootstrap draws are the same as in script 19 (same seed
and block length), so the IBIT–SPY and GLD–SPY intervals equal those of Table 5.

To keep the conventions identical to Table 5, the bootstrap, block-length and regime
functions are imported from 19_sample_size_robustness.py, which must be in the same folder.

Input   data/processed/returns__20260921.csv (script 02)
Output  out/29/regime_correlations.md and out/29/regime_correlations.json

Usage
    python 29_regime_correlations.py
    python 29_regime_correlations.py --data data/processed/returns__20260921.csv --boot 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = ["IBIT", "GLD", "TLT", "SPY"]
REGIMES = ["low", "mid", "high"]
LABELS = {"low": "Low (VIX ≤ 15)", "mid": "Medium (15 < VIX ≤ 25)", "high": "High (VIX > 25)"}


def load_19():
    path = os.path.join(HERE, "19_sample_size_robustness.py")
    if not os.path.exists(path):
        sys.exit("19_sample_size_robustness.py must be in the same folder as this script.")
    spec = importlib.util.spec_from_file_location("s19", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="returns CSV; default: data/processed/returns__YYYYMMDD.csv")
    ap.add_argument("--outdir", default="out/29")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    s19 = load_19()
    data = a.data or s19.find_data()
    if data is None or not os.path.exists(data):
        sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; run 02_build_returns.py or pass --data.")
    os.makedirs(a.outdir, exist_ok=True)

    d = pd.read_csv(data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(d)
    R = {k: d[k].to_numpy(float) for k in ASSETS}
    vix = d.VIX_Close.to_numpy(float)
    vix_lag = np.r_[np.nan, vix[:-1]]

    # block length exactly as in script 19 (mean Politis-White length of the two portfolios)
    rI = (R["IBIT"] + R["SPY"] + R["TLT"]) / 3
    rG = (R["GLD"] + R["SPY"] + R["TLT"]) / 3
    L = max(2, int(round(np.mean([s19.opt_block_length(rI), s19.opt_block_length(rG)]))))
    SEED = s19.SEED                    # same draws as the first bootstrap matrix of script 19
    rng = np.random.default_rng(SEED)
    IDX = s19.stationary_boot(n, a.boot, L, rng)
    split = s19.three_state(15, 25)
    pairs = list(itertools.combinations(ASSETS, 2))

    lines = ["# Correlation matrix of the four ETFs by VIX regime (script 29)\n",
             f"Data: `{os.path.basename(data)}`, {n} trading days, {d.Date.iloc[0].date()} → "
             f"{d.Date.iloc[-1].date()}. Regimes: VIX ≤ 15 / 15–25 / > 25. Stationary block bootstrap, "
             f"expected block length {L}, B = {a.boot}, seed {SEED}; regimes re-assigned in every "
             "replication.\n"]
    out = {"block_length": L, "boot": a.boot, "seed": int(SEED)}

    for timing, vv in (("same-day", vix), ("lagged", vix_lag)):
        ok = ~np.isnan(vv)
        masks = dict(zip(REGIMES, [m & ok for m in split(vv)]))
        masks["full"] = ok
        point = {p: {g: corr(R[p[0]][m], R[p[1]][m]) for g, m in masks.items()} for p in pairs}
        boots = {p: {g: [] for g in masks} for p in pairs}
        for c in range(IDX.shape[1]):
            i = IDX[:, c]
            vb = vv[i]
            okb = ~np.isnan(vb)
            mb = dict(zip(REGIMES, [m & okb for m in split(vb)]))
            mb["full"] = okb
            for p in pairs:
                x, y = R[p[0]][i], R[p[1]][i]
                for g, m in mb.items():
                    boots[p][g].append(corr(x[m], y[m]) if m.sum() >= 10 else np.nan)
        res = {}
        rows = []
        for p in pairs:
            key = f"{p[0]}-{p[1]}"
            res[key] = {}
            line = [f"{p[0]}–{p[1]}"]
            for g in REGIMES + ["full"]:
                lo, hi = s19.ci(boots[p][g])
                res[key][g] = {"rho": point[p][g], "ci": [float(lo), float(hi)]}
                line.append(f"{point[p][g]:.2f} [{lo:.2f}, {hi:.2f}]")
            dd = np.array(boots[p]["high"]) - np.array(boots[p]["low"])
            lo, hi = s19.ci(dd)
            pv = s19.pval(dd)
            dpe = point[p]["high"] - point[p]["low"]
            res[key]["high_minus_low"] = {"d": dpe, "ci": [float(lo), float(hi)], "p": pv}
            line.append(f"{dpe:+.2f} [{lo:+.2f}, {hi:+.2f}]; p = {pv:.3f}")
            rows.append(line)
        days = {g: int(m.sum()) for g, m in masks.items()}
        head = ["Pair"] + [f"{LABELS[g]}, {days[g]} days" for g in REGIMES] + [f"Full sample, {days['full']} days",
                                                                               "High − low"]
        lines.append(f"## {timing} VIX\n")
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        lines.append("")
        out[timing] = {"days": days, "pairs": res}

    # consistency with script 19 / Table 5 (same-day VIX)
    s = out["same-day"]["pairs"]
    checks = [("IBIT–SPY low 0.224", abs(s["IBIT-SPY"]["low"]["rho"] - 0.224) < 6e-4),
              ("IBIT–SPY high 0.561", abs(s["IBIT-SPY"]["high"]["rho"] - 0.561) < 6e-4),
              ("days 161/408/38", out["same-day"]["days"] == {"low": 161, "mid": 408, "high": 38, "full": n})]
    lines.append("## Checks\n")
    for lab, okc in checks:
        lines.append(f"   CHECK [{'PASS' if okc else 'FAIL'}] {lab}")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(a.outdir, "regime_correlations.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(a.outdir, "regime_correlations.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"Written {a.outdir}/regime_correlations.md and regime_correlations.json")


if __name__ == "__main__":
    main()
