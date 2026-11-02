# -*- coding: utf-8 -*-
"""
41_structural_scenarios.py — severe-state scenarios for the Bitcoin and gold sleeves under stated assumptions.

The scenarios are not forecasts and carry no probabilities: the historical data do not identify the probability of
technological displacement, a permanent collapse in demand, a contested change to the protocol rules or a failure of
custody or market access. They show how the portfolio conclusion depends on what is assumed about such states.

Each scenario specifies the path of the sleeve asset (severity, duration, recovery) and whether the sleeve can be
traded and rebalanced. SPY and TLT are held flat, so the portfolio outcome isolates the sleeve. Portfolios hold
one-third each of the sleeve asset, SPY and TLT, and are either rebalanced daily to one-third weights (when the
sleeve can be traded) or held without rebalancing. Paths are geometric (constant daily return) over the stated
number of trading days. Historical anchors (the deepest peak-to-trough declines of BTC-USD since 2014, of gold since
2000 and of IBIT since its listing, with their durations and recovery times) are computed from the snapshots.

Usage: python 41_structural_scenarios.py [--outdir out/41]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import longdata as LD  # noqa: E402

W = 1 / 3


def path(loss, n_down, n_flat=0, recover_to=None, n_up=0):
    """Daily sleeve returns: geometric fall of `loss` over n_down days, optional flat spell, optional geometric
    recovery to `recover_to` (fraction of the starting value) over n_up days."""
    r = [(1 - loss) ** (1 / n_down) - 1] * n_down + [0.0] * n_flat
    if recover_to is not None and n_up:
        r += [(recover_to / (1 - loss)) ** (1 / n_up) - 1] * n_up
    return np.array(r)


def outcome(r):
    """Portfolio value paths (start 1) with daily rebalancing and without; returns trough and terminal returns."""
    reb = np.cumprod(1 + W * r)
    held = W * np.cumprod(1 + r) + 2 * W
    return dict(reb_trough=float(reb.min() - 1), reb_end=float(reb[-1] - 1),
                held_trough=float(held.min() - 1), held_end=float(held[-1] - 1))


def drawdowns(px: pd.Series, top=2):
    """Deepest peak-to-trough declines with peak, trough, trading days to trough and days from trough to recovery."""
    px = px.dropna()
    peak = px.cummax()
    dd = px / peak - 1
    out, used = [], pd.Series(False, index=px.index)
    for _ in range(top):
        d = dd[~used]
        if d.empty or d.min() >= 0:
            break
        t = d.idxmin()
        p = px.loc[:t].idxmax()
        after = px.loc[t:]
        rec = after[after >= px.loc[p]]
        r_date = rec.index[0] if len(rec) else None
        out.append(dict(peak=str(p.date()), trough=str(t.date()), depth=float(dd.loc[t]),
                        days_to_trough=int(px.index.get_loc(t) - px.index.get_loc(p)),
                        recovery=str(r_date.date()) if r_date is not None else None,
                        days_to_recovery=int(px.index.get_loc(r_date) - px.index.get_loc(t)) if r_date is not None else None))
        end = r_date if r_date is not None else px.index[-1]
        used.loc[p:end] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/41")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    raw = LD.find_raw()
    btc = LD._px(raw, "BTC-USD", "L", "AdjClose")
    gold = LD.load_gold_long()
    gpx = (1 + gold["GOLD"]).cumprod()
    E = LD.load_etf()
    ibit = (1 + E["IBIT"]).cumprod()
    anchors = dict(btc=drawdowns(btc, 2), gold=drawdowns(gpx, 1), ibit=drawdowns(ibit, 1))

    S = [
        ("S0", "Mechanical one-day shock (first version)", "−50% in one day; other assets unchanged", "Tradable",
         dict(btc=path(0.50, 1), gold=path(0.50, 1))),
        ("S1", "Temporary drawdown", "−75% (Bitcoin) / −45% (gold) over one year, full recovery over the next two years",
         "Tradable; rebalancing possible",
         dict(btc=path(0.75, 252, 0, 1.0, 504), gold=path(0.45, 252, 0, 1.0, 504))),
        ("S2", "Permanent impairment", "−75% / −45% over one year, no recovery", "Tradable; rebalancing possible",
         dict(btc=path(0.75, 252, 252), gold=path(0.45, 252, 252))),
        ("S3", "Near-total loss", "−95% over one year, no recovery", "Tradable; rebalancing possible",
         dict(btc=path(0.95, 252, 252), gold=path(0.95, 252, 252))),
        ("S4", "Total loss", "−100% at once (e.g., an irrecoverable failure)", "Not tradable",
         dict(btc=np.array([-1.0]), gold=np.array([-1.0]))),
    ]
    rows = []
    for code, name, assump, trade, P in S:
        o = {k: outcome(v) for k, v in P.items()}
        rows.append(dict(code=code, name=name, assumption=assump, trading=trade, btc=o["btc"], gold=o["gold"]))
    # custody or access failure: the sleeve is frozen (no trading, no rebalancing) and a fraction R is recovered
    custody = []
    for R in (1.0, 0.5, 0.0):
        custody.append(dict(recovered=R, portfolio_loss=float(W * (R - 1))))
    out = dict(anchors=anchors, scenarios=rows, custody=custody)
    json.dump(out, open(os.path.join(a.outdir, "structural_scenarios.json"), "w"), indent=1, default=float)
    L = ["# Structural scenarios (script 41)\n", "## Historical anchors\n"]
    for k, v in anchors.items():
        for d in v:
            L.append(f"- {k}: {d['peak']} → {d['trough']} {d['depth']*100:.1f}% in {d['days_to_trough']} days; "
                     f"recovered {d['recovery']} ({d['days_to_recovery']} days after the trough)")
    L += ["", "## Scenarios (SPY and TLT flat; one-third sleeve)\n",
          "| Code | Scenario | Sleeve path | Trading | Bitcoin port.: rebalanced trough / end | held trough / end | "
          "Gold port.: rebalanced trough / end | held trough / end |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        b, g = r["btc"], r["gold"]
        L.append(f"| {r['code']} | {r['name']} | {r['assumption']} | {r['trading']} | "
                 f"{b['reb_trough']*100:.1f} / {b['reb_end']*100:.1f} | {b['held_trough']*100:.1f} / {b['held_end']*100:.1f} | "
                 f"{g['reb_trough']*100:.1f} / {g['reb_end']*100:.1f} | {g['held_trough']*100:.1f} / {g['held_end']*100:.1f} |")
    L += ["", "Custody or access failure (sleeve frozen, fraction R recovered): " +
          "; ".join(f"R = {c['recovered']:.0%}: {c['portfolio_loss']*100:.1f}%" for c in custody)]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "structural_scenarios.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
