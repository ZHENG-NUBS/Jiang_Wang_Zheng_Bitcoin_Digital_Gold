#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
38_specification_families.py — Robustness results grouped by family of specifications, the
exceptions, and the decomposition of rebalancing effects (second-round revision: Reviewer 1,
comments 5 and 6)

Reads the outputs of script 24 (specification_curve.csv, multiverse.csv) and script 28
(tab_R_rebalancing.csv); runs nothing else.

A  Families of one-dimensional specifications (script 24): number of specifications, range and
   median of the ES95 ratio, number below one, and — only for the specifications that carry a
   confidence interval — the number whose interval lies above one. Families are labelled
   descriptive (point estimates on the same data) or inferential (with an interval). The
   specifications share the data and the portfolio components, so they are not independent
   replications.
B  Joint specifications (multiverse): share of ratios above one by risk measure and by weight, and
   the list of exceptions (ratio <= 1).
C  Rebalancing rules (script 28): ES95 and volatility of the IBIT and GLD portfolios under daily,
   weekly, monthly, quarterly and threshold rebalancing, and the split of each portfolio's ES into
   an asset effect (log ES ratio within a rule) and a rule effect (log ES under the rule minus log ES
   under daily rebalancing), with annual costs.

Usage
    python 38_specification_families.py --spec out/24 --costs out/28
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


FAMILY = [("Baseline", "Baseline"), ("A ", "Portfolio weights"), ("B ", "Rebalancing rule"),
          ("C ", "Sample period"), ("D ", "Equity and bond legs"), ("E ", "Risk measure"),
          ("F Pricing", "Pricing time"), ("F Bitcoin", "Bitcoin and gold instruments"),
          ("F Forecast", "Forecast model (out of sample)")]


def family(dim):
    for k, v in FAMILY:
        if dim.startswith(k):
            return v
    return dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="out/24")
    ap.add_argument("--costs", default="out/28")
    ap.add_argument("--outdir", default="out/38")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    OUT = {}
    L = ["# Robustness by family of specifications and rebalancing decomposition (script 38)\n"]
    sp = pd.read_csv(os.path.join(a.spec, "specification_curve.csv"))
    sp["family"] = sp.dim.map(family)
    rows = []
    for fam, g in sp.groupby("family", sort=False):
        has = g.lo.notna() & g.hi.notna()
        ex = g[g.ratio <= 1]
        exi = g[has & (g.lo <= 1)]
        o = dict(n=len(g), min=float(g.ratio.min()), median=float(g.ratio.median()), max=float(g.ratio.max()),
                 n_below1=int((g.ratio <= 1).sum()), n_ci=int(has.sum()), n_ci_above1=int((g.lo[has] > 1).sum()),
                 exceptions=ex.label.tolist(), ci_includes_1=exi.label.tolist(),
                 kind="inferential" if has.all() else ("mixed" if has.any() else "descriptive"))
        OUT.setdefault("families", {})[fam] = o
        rows.append([fam, o["kind"], o["n"], f"{o['min']:.2f}", f"{o['median']:.2f}", f"{o['max']:.2f}", o["n_below1"],
                     f"{o['n_ci_above1']}/{o['n_ci']}" if o["n_ci"] else "—",
                     "; ".join(o["ci_includes_1"][:6]) + (" …" if len(o["ci_includes_1"]) > 6 else "") or "—"])
    L.append("## A. One-dimensional specifications by family (ES95 ratio, Bitcoin portfolio / gold portfolio)\n")
    L.append(md(rows, ["Family", "Type", "Specifications", "Min", "Median", "Max", "Ratio ≤ 1",
                       "CI above 1 / with CI", "Specifications whose CI includes 1"]))
    L.append("The specifications re-use the same returns and portfolio components; their number is not a count of "
             "independent confirmations.\n")

    p = os.path.join(a.spec, "multiverse.csv")
    if os.path.exists(p):
        mv = pd.read_csv(p)
        rows = []
        for m_, g in mv.groupby("measure", sort=False):
            cells = []
            for wt, gg in g.groupby("w"):
                cells.append(f"{(gg.ratio > 1).mean() * 100:.0f}% (min {gg.ratio.min():.2f})")
            rows.append([m_] + cells)
        ws = sorted(mv.w.unique())
        L.append("## B. Joint specifications: share with ratio > 1 by risk measure and weight\n")
        L.append(md(rows, ["Risk measure"] + [f"w = {w * 100:.0f}%" for w in ws]))
        low = mv[mv.ratio <= 1]
        OUT["multiverse"] = dict(n=len(mv), share_gt1=float((mv.ratio > 1).mean()), n_exceptions=len(low),
                                 exceptions_by_measure=low.measure.value_counts().to_dict(),
                                 exceptions_by_weight={str(k): int(v) for k, v in low.w.value_counts().items()},
                                 exceptions_by_rebal=low.rebal.value_counts().to_dict())
        L.append(f"{len(low)} of {len(mv)} joint specifications have a ratio ≤ 1: by measure "
                 + ", ".join(f"{k} {v}" for k, v in OUT["multiverse"]["exceptions_by_measure"].items())
                 + "; by weight " + ", ".join(f"{float(k) * 100:.0f}% {v}" for k, v in OUT["multiverse"]["exceptions_by_weight"].items())
                 + "; by rebalancing " + ", ".join(f"{k} {v}" for k, v in OUT["multiverse"]["exceptions_by_rebal"].items()) + ".\n")

    p = os.path.join(a.costs, "tab_R_rebalancing.csv")
    if os.path.exists(p):
        R = pd.read_csv(p)
        I = R[R.portfolio.str.contains("IBIT/SPY/TLT")].set_index("schedule")
        G = R[R.portfolio.str.contains("GLD/SPY/TLT")].set_index("schedule")
        rows = []
        for s in I.index:
            ai = math.log(I.loc[s, "es95"] / G.loc[s, "es95"])
            ri = math.log(I.loc[s, "es95"] / I.loc["daily", "es95"]); rg = math.log(G.loc[s, "es95"] / G.loc["daily", "es95"])
            OUT.setdefault("rebalancing", {})[s] = dict(es_ibit=float(I.loc[s, "es95"]), es_gld=float(G.loc[s, "es95"]),
                                                        ratio=float(math.exp(ai)), asset_effect_log=ai,
                                                        rule_effect_ibit_log=ri, rule_effect_gld_log=rg,
                                                        cost_bp_ibit=float(I.loc[s, "cost_bp_ann"]),
                                                        cost_bp_gld=float(G.loc[s, "cost_bp_ann"]),
                                                        n_rebal=int(I.loc[s, "n_rebal"]))
            rows.append([s, int(I.loc[s, "n_rebal"]), f"{I.loc[s, 'es95'] * 100:.2f} / {G.loc[s, 'es95'] * 100:.2f}",
                         f"{math.exp(ai):.2f}", f"{ri * 100:+.1f} / {rg * 100:+.1f}",
                         f"{I.loc[s, 'vol'] * 100:.1f} / {G.loc[s, 'vol'] * 100:.1f}",
                         f"{I.loc[s, 'cost_bp_ann']:.1f} / {G.loc[s, 'cost_bp_ann']:.1f}",
                         f"{I.loc[s, 'sharpe_net']:.2f} / {G.loc[s, 'sharpe_net']:.2f}"])
        L.append("## C. Rebalancing rules: asset effect versus rule effect (ETF sample; IBIT portfolio / GLD portfolio)\n")
        L.append(md(rows, ["Rule", "Rebalances", "ES95 %", "ES95 ratio (asset effect)",
                           "Rule effect on log ES vs daily (%)", "Vol %", "Cost bp/yr", "Net Sharpe"]))
        L.append("ES95 is computed on returns net of the cost model of script 28; the asset effect is the ratio within "
                 "a rule; the rule effect is the change in each portfolio's ES relative to daily rebalancing.\n")
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.outdir, "specification_families.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "specification_families.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
