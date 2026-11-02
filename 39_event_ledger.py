# -*- coding: utf-8 -*-
"""
39_event_ledger.py — one auditable ledger of the SPY drawdown episodes used in the stress tables.

The first version of the paper used two conventions for the window return of a peak-to-trough episode:
  * Table 13 (in-sample stress windows, script 09 and Panel A of 05): the window started at the close BEFORE the
    peak day, so the return earned on the peak day itself was included;
  * Table G4 (episodes since 2014, script 15) and Table 12 (episodes since 2000, script 32): the window runs from
    the close OF the peak day to the close of the trough day.
Both select an episode by the same quantity, the SPY drawdown from its running maximum (adjusted closes, i.e. total
return) to the trough. Under the first convention the reported SPY window return is not the drawdown used for
selection (one window selected as a drawdown of more than 5% showed an SPY return of -4.56%), and the SPY return and
day count of the same episode differ between tables.

This script
  1. re-identifies every episode with a single rule per table (thresholds inclusive: drawdown <= -threshold);
  2. computes all window returns from the close of the peak day to the close of the trough day, so that the SPY
     window return equals the selection drawdown exactly, and counts the number of daily returns in the window;
  3. reproduces the first-version numbers (close-before-peak convention) as a reconciliation column;
  4. checks every asset return against the product of its daily gross returns and against the endpoint price ratio
     where prices are available, and computes portfolio returns from compounded weighted daily returns (daily
     rebalancing) and from weighted compounded asset returns (buy-and-hold);
  5. recomputes the in-sample scenario counts and the shifted-window sensitivity of Section 4.6 (rules P1–P3 of
     script 05) under the corrected P1 convention.

Inputs: data/processed/returns__20260921.csv (ETF sample), the long-sample snapshot (via longdata.py) and
out/32/safe_haven_regimes.json (episodes since 2000). Output: out/39/.

Usage:  python 39_event_ledger.py [--outdir out/39]
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

W3 = 1 / 3
DD_MIN, VIX_HI, CRASH = 0.05, 25.0, 0.10      # rules P1, P2, P3 of script 05
G4_THRESH = 0.08                               # script 15


def p1_episodes(spy: np.ndarray, thresh: float):
    """Peak-to-trough episodes of the SPY drawdown from its running maximum. Returns (i_peak, i_trough, dd)
    with indices into the return series; the peak is the last day with zero drawdown before the trough."""
    cum = np.cumprod(1 + spy)
    dd = cum / np.maximum.accumulate(cum) - 1
    out, i, n = [], 0, len(spy)
    while i < n:
        if dd[i] >= -1e-12:
            i += 1
            continue
        j = i
        while j < n and dd[j] < -1e-12:
            j += 1
        t = i + int(np.argmin(dd[i:j]))
        if dd[t] <= -thresh + 1e-12:
            out.append((i - 1 if i > 0 else 0, t, float(dd[t])))
        i = j
    return out


def cum(x):
    return float(np.prod(1 + np.asarray(x)) - 1)


def window(df, a, b):
    """Daily returns in (a, b]: from the close of day a to the close of day b."""
    return df.iloc[a + 1:b + 1]


def window_old(df, a, b):
    """First-version Table 13 convention: [a, b], i.e. from the close before day a."""
    return df.iloc[a:b + 1]


def port(rows, sleeve):
    reb = cum(rows[[sleeve, "SPY", "TLT"]].mean(axis=1))
    bh = float(np.mean([cum(rows[c]) for c in (sleeve, "SPY", "TLT")]))
    return reb, bh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/39")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = LD.load_etf()
    n = len(E)
    led = []
    checks = []

    # ------------------------------------------------------------ Table 13: in-sample P1 windows (ETF sample)
    t13 = []
    for k, (p, t, dd) in enumerate(p1_episodes(E.SPY.values, DD_MIN), 1):
        r, ro = window(E, p, t), window_old(E, p, t)
        spy = cum(r.SPY)
        checks.append(("Table 13 SPY window return equals selection drawdown", abs(spy - dd) < 1e-12, f"{spy:.6f} vs {dd:.6f}"))
        pi, pib = port(r, "IBIT"); pg, pgb = port(r, "GLD")
        pio, _ = port(ro, "IBIT"); pgo, _ = port(ro, "GLD")
        row = dict(table="13", rule="P1: SPY drawdown >= 5%, ETF sample", peak=str(E.index[p].date()),
                   trough=str(E.index[t].date()), n_returns=t - p, selection_drawdown=dd, spy=spy,
                   ibit=cum(r.IBIT), gld=cum(r.GLD), tlt=cum(r.TLT), ibit_port=pi, gld_port=pg,
                   ibit_port_bh=pib, gld_port_bh=pgb,
                   old_days=t - p + 1, old_spy=cum(ro.SPY), old_ibit=cum(ro.IBIT), old_gld=cum(ro.GLD),
                   old_ibit_port=pio, old_gld_port=pgo)
        t13.append(row)
        led.append(row)

    # ------------------------------------------------------------ in-sample scenario counts (rules P1-P3)
    def scen(a_, b_, inclusive_start):
        rows = E.iloc[a_ + (0 if inclusive_start else 1):b_ + 1]
        return port(rows, "IBIT")[0], port(rows, "GLD")[0]
    S = []
    for p, t, _ in p1_episodes(E.SPY.values, DD_MIN):
        S.append(("P1", p, t, False))
    vix = E.VIX.values
    h = np.where(vix > VIX_HI)[0]
    if len(h):
        cur = [h[0]]
        for k in h[1:]:
            if k - cur[-1] <= 3:
                cur.append(k)
            else:
                S.append(("P2", cur[0], cur[-1], True)); cur = [k]
        S.append(("P2", cur[0], cur[-1], True))
    for asset in ("IBIT", "GLD"):
        x = E[asset].values
        r5 = np.array([np.nan if k < 4 else cum(x[k - 4:k + 1]) for k in range(n)])
        hh = np.where(r5 < -CRASH)[0]
        if len(hh):
            cur = [hh[0]]
            for k in hh[1:]:
                if k - cur[-1] <= 5:
                    cur.append(k)
                else:
                    S.append(("P3-" + asset, max(0, cur[0] - 4), cur[-1], True)); cur = [k]
            S.append(("P3-" + asset, max(0, cur[0] - 4), cur[-1], True))
    worse_gold = [(r_, str(E.index[a_].date()), str(E.index[b_].date())) for r_, a_, b_, inc in S
                  if scen(a_, b_, inc)[1] < scen(a_, b_, inc)[0]]
    # first-version convention for P1 (inclusive start), to reproduce the published counts
    worse_gold_old = [r_ for r_, a_, b_, inc in S
                      if (lambda v: v[1] < v[0])(scen(a_, b_, True if r_ == "P1" else inc))]

    # ------------------------------------------------------------ shifted windows (start and end +-5 days)
    shift = []
    for p, t, _ in p1_episodes(E.SPY.values, DD_MIN):
        res = []
        for ds in range(-5, 6):
            for de in range(-5, 6):
                a_, b_ = p + ds, t + de
                if a_ < 0 or b_ >= n or b_ - a_ < 3:
                    continue
                rows = window(E, a_, b_)
                res.append((cum(rows.SPY), cum(rows.IBIT), port(rows, "IBIT")[0], port(rows, "GLD")[0]))
        res = np.array(res)
        shift.append(dict(peak=str(E.index[p].date()), trough=str(E.index[t].date()), n=len(res),
                          share_ibit_not_below_spy=float(np.mean(res[:, 1] >= res[:, 0])),
                          share_ranking_reversed=float(np.mean(res[:, 2] >= res[:, 3]))))

    # ------------------------------------------------------------ Table G4: episodes of 8% or more since 2014
    B = LD.load_btc_long()
    raw = LD.find_raw()
    prices = {c: LD._px(raw, f, "L", "AdjClose") for c, f in (("SPY", "SPY"), ("BTC", "BTC-USD"), ("GLD", "GLD"))}
    for p, t, dd in p1_episodes(B.SPY.values, G4_THRESH):
        r = window(B, p, t)
        row = dict(table="G4", rule="SPY drawdown >= 8%, Bitcoin long sample", peak=str(B.index[p].date()),
                   trough=str(B.index[t].date()), n_returns=t - p, selection_drawdown=dd, spy=cum(r.SPY),
                   btc=cum(r.BTC), gld=cum(r.GLD), tlt=cum(r.TLT))
        for c in ("SPY", "BTC", "GLD"):
            px = prices[c].reindex(B.index)
            ratio = float(px.iloc[t] / px.iloc[p] - 1) if c != "BTC" else None
            if ratio is not None:
                checks.append((f"G4 {c} window return equals endpoint price ratio", abs(ratio - row[c.lower()]) < 1e-9,
                               f"{row['peak']}: {ratio:.6f} vs {row[c.lower()]:.6f}"))
        checks.append(("G4 SPY window return equals selection drawdown", abs(row["spy"] - dd) < 1e-9, row["peak"]))
        led.append(row)

    # ------------------------------------------------------------ Table 12: episodes since 2000 (script 32)
    d32 = json.load(open("out/32/safe_haven_regimes.json"))
    for e in d32["gold_long"]["episodes"]:
        led.append(dict(table="12", rule="SPY drawdown >= 10%, confirmed by 20% rebound or new high, since 2000",
                        peak=e["peak"], trough=e["trough"], n_returns=e["days"], selection_drawdown=e["depth"],
                        spy=e["depth"], name=e.get("name")))

    # ------------------------------------------------------------ cross-table reconciliation of common episodes
    L = pd.DataFrame(led)
    L["event"] = L.groupby(["peak", "trough"]).ngroup() + 1
    common = (L.groupby(["peak", "trough"])["table"].apply(lambda s: ", ".join(sorted(set(s))))
              .reset_index().rename(columns={"table": "tables"}))
    L = L.merge(common, on=["peak", "trough"])
    for (pk, tr), g in L.groupby(["peak", "trough"]):
        if g.table.nunique() > 1:
            ok = np.allclose(g.spy.values, g.spy.values[0], atol=5e-5) and g.n_returns.nunique() == 1
            checks.append(("Common episode has the same SPY return and day count in every table", ok,
                           f"{pk}..{tr}: " + "; ".join(f"T{a}: {s*100:.2f}%/{d}" for a, s, d in
                                                      zip(g.table, g.spy, g.n_returns))))
    L.sort_values(["peak", "table"]).to_csv(os.path.join(a.outdir, "event_ledger.csv"), index=False)

    out = dict(table13=t13, scenarios_total=len(S), gold_worse=worse_gold, gold_worse_old=worse_gold_old,
               scenario_rules={r_: sum(1 for x in S if x[0] == r_) for r_ in sorted(set(x[0] for x in S))},
               shifted=shift, checks=[dict(name=c[0], ok=bool(c[1]), detail=c[2]) for c in checks])
    json.dump(out, open(os.path.join(a.outdir, "event_ledger.json"), "w"), indent=1, default=float)

    lines = ["# Event ledger and reconciliation of the stress windows (script 39)\n",
             "All windows run from the close of the SPY peak day to the close of the trough day; thresholds are "
             "inclusive. 'Old' columns reproduce the first-version Table 13 convention (from the close before the "
             "peak day).\n", "## Table 13 windows\n",
             "| Peak | Trough | Returns | SPY drawdown | SPY | IBIT | GLD | IBIT port. | GLD port. | B&H IBIT port. | "
             "B&H GLD port. | Old days | Old SPY | Old IBIT port. | Old GLD port. |", "|" + "---|" * 15]
    for r in t13:
        lines.append("| " + " | ".join([r["peak"], r["trough"], str(r["n_returns"])] +
                                       [f"{r[k]*100:+.2f}" for k in ("selection_drawdown", "spy", "ibit", "gld",
                                                                      "ibit_port", "gld_port", "ibit_port_bh",
                                                                      "gld_port_bh")] +
                                       [str(r["old_days"])] + [f"{r[k]*100:+.2f}" for k in ("old_spy", "old_ibit_port",
                                                                                            "old_gld_port")]) + " |")
    lines += ["", f"In-sample scenarios (rules P1–P3): {len(S)} ({out['scenario_rules']}); gold portfolio worse "
              f"than the IBIT portfolio in {len(worse_gold)}: {worse_gold}. First-version P1 convention: "
              f"{len(worse_gold_old)}.", "", "## Shifted windows (start and end ±5 trading days)\n",
              "| Peak | Trough | Windows | IBIT not below SPY | Portfolio ranking reversed |", "|---|---|---|---|---|"]
    for s in shift:
        lines.append(f"| {s['peak']} | {s['trough']} | {s['n']} | {s['share_ibit_not_below_spy']:.0%} | "
                     f"{s['share_ranking_reversed']:.0%} |")
    lines += ["", "## Checks\n"] + [f"- [{'PASS' if c[1] else 'FAIL'}] {c[0]} — {c[2]}" for c in checks]
    txt = "\n".join(lines)
    open(os.path.join(a.outdir, "event_ledger.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
