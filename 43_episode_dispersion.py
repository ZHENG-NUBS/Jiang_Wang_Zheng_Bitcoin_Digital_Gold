# -*- coding: utf-8 -*-
"""
43_episode_dispersion.py — volatility scaling and leave-one-out checks for the episode multipliers of Table G4.

Table G4 (script 15) reports, for the nine SPY drawdowns of 8% or more since September 2014, each asset's multiplier
m = asset return / SPY return over the window (prior peak to trough) and compares the dispersion of Bitcoin's and
gold's multipliers. Because an asset's multiplier dispersion grows with its own volatility even if its behavior is
unchanged, this script reports
  (i)  the dispersion ratio after scaling each multiplier by the ratio of SPY's to the asset's daily volatility over
       the same window (m × σ_SPY / σ_asset, a correlation-like quantity),
  (ii) the dispersion ratio expected from each asset's full-sample beta and residual variance: under a stable
       relation r_a = α + β r_SPY + e, the multiplier is β + (Σe)/(Σr_SPY) approximately, whose standard deviation
       across windows is computed from the window lengths and SPY window returns,
  (iii) leave-one-episode-out dispersion ratios, and the range of Bitcoin's multiplier without the 2015–2016 episode.
Same data, calendar and windows as script 15 (BTC-USD at 00:00 UTC on the SPY calendar).

Usage: python 43_episode_dispersion.py [--raw data/raw] [--asof 20260921L] [--outdir out/43]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_raw(raw_dir, asof, ticker):
    fn = ticker.replace("^", "IDX_").replace("-", "_")
    for cand in (f"{fn}__{asof}.csv", f"{ticker}__{asof}.csv"):
        p = os.path.join(raw_dir, cand)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            col = "AdjClose" if "AdjClose" in d.columns else "Close"
            return d.set_index("Date")[col].rename(ticker)
    return None


def drawdown_episodes(r, thresh):
    """As script 15: (prior peak, trough) index pairs of drawdowns of at least `thresh`."""
    cum = np.cumprod(1 + r)
    dd = cum / np.maximum.accumulate(cum) - 1.0
    eps, i, n = [], 0, len(r)
    while i < n:
        if dd[i] >= -1e-12:
            i += 1
            continue
        j = i
        while j < n and dd[j] < -1e-12:
            j += 1
        seg = dd[i:j]
        if seg.min() <= -thresh:
            eps.append((i - 1 if i > 0 else 0, i + int(np.argmin(seg))))
        i = j
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default="20260921L")
    ap.add_argument("--outdir", default="out/43")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    S = {t: load_raw(a.raw, a.asof, t) for t in ("SPY", "BTC-USD", "GLD")}
    px = pd.DataFrame(S).reindex(S["SPY"].index)
    R = px.pct_change().dropna()
    rs = R["SPY"].values
    eps = drawdown_episodes(rs, 0.08)
    rows = []
    for p, t in eps:
        w_ = slice(p + 1, t + 1)
        rec = dict(peak=str(R.index[p].date()), trough=str(R.index[t].date()), days=t - p)
        spy = float(np.prod(1 + rs[w_]) - 1)
        rec["spy"] = spy
        for k, c in (("btc", "BTC-USD"), ("gld", "GLD")):
            x = R[c].values
            rec[k] = float(np.prod(1 + x[w_]) - 1)
            rec[f"m_{k}"] = rec[k] / spy
            rec[f"scaled_{k}"] = rec[f"m_{k}"] * float(np.std(rs[w_], ddof=1) / np.std(x[w_], ddof=1))
        rows.append(rec)
    E = pd.DataFrame(rows)
    assert len(E) == 9, len(E)
    sd = lambda v: float(np.std(v, ddof=1))  # noqa: E731
    out = dict(episodes=rows)
    out["ratio_raw"] = sd(E.m_btc) / sd(E.m_gld)
    out["ratio_scaled"] = sd(E.scaled_btc) / sd(E.scaled_gld)
    # expected dispersion under a stable linear relation with full-sample beta and residual variance
    exp = {}
    for k, c in (("btc", "BTC-USD"), ("gld", "GLD")):
        x = R[c].values
        b = np.polyfit(rs, x, 1)[0]
        s2 = float(np.var(x - b * rs - (x - b * rs).mean(), ddof=1))
        # m ≈ β + (sum of residuals over the window) / (SPY window return); var = n σ_e² / r_SPY²
        exp[k] = dict(beta=float(b), resid_sd=float(np.sqrt(s2)),
                      implied_sd=float(np.sqrt(np.mean([d * s2 / (s ** 2) for d, s in zip(E.days, E.spy)]))))
    out["implied"] = exp
    out["ratio_implied"] = exp["btc"]["implied_sd"] / exp["gld"]["implied_sd"]
    loo = []
    for i in range(len(E)):
        Ei = E.drop(index=i)
        loo.append(dict(dropped=f"{E.peak[i]}..{E.trough[i]}", ratio=sd(Ei.m_btc) / sd(Ei.m_gld),
                        ratio_scaled=sd(Ei.scaled_btc) / sd(Ei.scaled_gld)))
    out["leave_one_out"] = loo
    first = E.peak.str.startswith("2015")
    out["btc_range_without_2015"] = [float(E.m_btc[~first].min()), float(E.m_btc[~first].max())]
    out["btc_fell"] = int((E.btc < 0).sum())
    json.dump(out, open(os.path.join(a.outdir, "episode_dispersion.json"), "w"), indent=1)
    L = ["# Episode multipliers: volatility scaling and leave-one-out (script 43)\n",
         "| Peak → trough | Days | SPY % | BTC multiplier | GLD multiplier | BTC scaled | GLD scaled |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['peak']} → {r['trough']} | {r['days']} | {r['spy'] * 100:.2f} | {r['m_btc']:.2f} | "
                 f"{r['m_gld']:.2f} | {r['scaled_btc']:.2f} | {r['scaled_gld']:.2f} |")
    L += ["", f"Dispersion ratio, raw multipliers: {out['ratio_raw']:.2f}; volatility-scaled: {out['ratio_scaled']:.2f}; "
              f"implied by full-sample beta and residual variance: {out['ratio_implied']:.2f}",
          "", "| Episode left out | Ratio | Ratio, scaled |", "|---|---|---|"]
    L += [f"| {x['dropped']} | {x['ratio']:.2f} | {x['ratio_scaled']:.2f} |" for x in loo]
    L += ["", f"Bitcoin fell in {out['btc_fell']} of 9 episodes; multiplier range without 2015–2016: "
              f"{out['btc_range_without_2015'][0]:.2f} to {out['btc_range_without_2015'][1]:.2f}"]
    ok = abs(out["ratio_raw"] - 2.62) < 0.01
    L.append(f"CHECK {'PASS' if ok else 'FAIL'}: raw dispersion ratio reproduces script 15 (2.62): {out['ratio_raw']:.3f}")
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "episode_dispersion.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
