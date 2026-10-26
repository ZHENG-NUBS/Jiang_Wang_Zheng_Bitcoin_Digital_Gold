#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
37_mean_specification.py — Residual diagnostics of the GARCH models, the source of the residual
autocorrelation of the IBIT portfolio, and sensitivity to an AR(1) conditional mean
(second-round revision: Reviewer 1, comments 8 and 9; Reviewer 2, comments 2 and 10)

A  GARCH(1,1)-t with a constant mean and with an AR(1) mean for the two ETF-sample portfolios, IBIT
   and GLD, and for the long-sample portfolios: AR coefficient, Ljung–Box tests on standardized
   residuals and their squares (lags 5, 10, 20), Engle's ARCH-LM test, BIC on a common sample.
B  Where the autocorrelation comes from. First-order autocorrelation and Ljung–Box Q(10) of daily
   returns over the ETF sample for: IBIT at its market close, IBIT at its net asset value (NAV),
   Bitcoin at 16:00 ET (Coinbase hourly bars; the NAV benchmark time), and BTC-USD at 00:00 UTC;
   AR(1) coefficient and half-life of the IBIT closing premium to NAV. If the market-close series is
   negatively autocorrelated while the NAV and 16:00 series are not, the autocorrelation is a
   property of the traded vehicle and of the measurement (premium reversal), not of Bitcoin.
C  Sensitivity of the tail comparison: the historical-simulation ES95 ratio (no model), the shape
   ratio after standardizing by the sample standard deviation (script 34), by GARCH-t volatility
   with a constant mean and by AR(1)-GARCH-t volatility.

Usage
    python 37_mean_specification.py              # writes out/37/
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
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402
from arch import arch_model  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402

warnings.filterwarnings("ignore")


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def es(x, a=0.05):
    """Historical-simulation ES as in scripts 11 and 34: mean of the returns at or below the a-quantile."""
    x = np.asarray(x, float); v = np.quantile(x, a)
    return -x[x <= v].mean()


def fit(y, ar):
    m = arch_model(y, mean="AR" if ar else "Constant", lags=1 if ar else 0, vol="GARCH", p=1, q=1, dist="t")
    return m.fit(disp="off", show_warning=False, first_obs=None if ar else 1, options={"maxiter": 3000})


def diag(res):
    z = np.asarray(res.std_resid, float); z = z[np.isfinite(z)]
    lb = acorr_ljungbox(z, lags=[5, 10, 20], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[5, 10, 20], return_df=True)
    lm = het_arch(z, nlags=10)
    return dict(lb_p=lb.lb_pvalue.tolist(), lb2_p=lb2.lb_pvalue.tolist(), archlm_p=float(lm[1]),
                bic=float(res.bic), nobs=int(res.nobs), nu=float(res.params["nu"]), z=z,
                vol=np.asarray(res.conditional_volatility, float))


def ac1(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    r = float(np.corrcoef(x[1:], x[:-1])[0, 1])
    q = acorr_ljungbox(x, lags=[10], return_df=True)
    se = 1 / math.sqrt(len(x))
    return dict(rho1=r, p_rho1=float(2 * stats.norm.sf(abs(r) / se)), q10_p=float(q.lb_pvalue.iloc[0]), n=len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/37")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    E = LD.load_etf()
    series = {"ETF: IBIT portfolio": ((E.IBIT + E.SPY + E.TLT) / 3).values,
              "ETF: GLD portfolio": ((E.GLD + E.SPY + E.TLT) / 3).values,
              "ETF: IBIT": E.IBIT.values, "ETF: GLD": E.GLD.values}
    try:
        B = LD.load_btc_long()
        series.update({"Long: BTC portfolio": ((B.BTC + B.SPY + B.TLT) / 3).values,
                       "Long: GLD portfolio": ((B.GLD + B.SPY + B.TLT) / 3).values, "Long: BTC": B.BTC.values})
    except FileNotFoundError:
        B = None
    OUT = {"A": {}, "B": {}, "C": {}}
    rows = []
    fits = {}
    for nm, r in series.items():
        y = r * 100
        c0, c1 = fit(y, False), fit(y, True)
        d0, d1 = diag(c0), diag(c1)
        assert d0["nobs"] == d1["nobs"]
        k = [k_ for k_ in c1.params.index if k_.startswith("y[")][0]
        phi, php = float(c1.params[k]), float(c1.pvalues[k])
        fits[nm] = (d0, d1)
        OUT["A"][nm] = dict(phi=phi, phi_p=php, const={kk: v for kk, v in d0.items() if kk not in ("z", "vol")},
                            ar1={kk: v for kk, v in d1.items() if kk not in ("z", "vol")})
        for lab, d in (("constant", d0), ("AR(1)", d1)):
            rows.append([nm, lab, f"{phi:+.3f} ({pf(php)})" if lab == "AR(1)" else "—",
                         " / ".join(pf(p) for p in d["lb_p"]), " / ".join(pf(p) for p in d["lb2_p"]),
                         pf(d["archlm_p"]), f"{d['nu']:.1f}", f"{d['bic']:.1f}"])
    L = ["# GARCH mean specification and residual diagnostics (script 37)\n",
         "## A. GARCH(1,1)-t with constant and AR(1) mean (common estimation sample)\n",
         md(rows, ["Series", "Mean", "φ (p)", "Ljung–Box p, z (5/10/20)", "Ljung–Box p, z² (5/10/20)",
                   "ARCH-LM(10) p", "ν", "BIC"])]

    # ---------------- B
    m20 = load("m20", "20_scope_expansion.py")
    raw = LD.find_raw()
    cal = E.index
    rowsB = []
    ib = E.IBIT
    OUT["B"]["IBIT close"] = ac1(ib.values)
    h = m20.load_hourly(raw, "coinbase") if raw else None
    full_cal = None
    if h is not None:
        spy_raw = pd.read_csv(os.path.join(raw, "SPY__20260921.csv"), parse_dates=["Date"]).set_index("Date")
        full_cal = spy_raw.index[spy_raw.index <= cal[-1]]
        b16, _ = m20.btc_at_close(h, full_cal)
        OUT["B"]["BTC 16:00 ET"] = ac1(b16.pct_change().reindex(cal).values)
    nav = None
    p_nav = os.path.join(raw, "IBIT_nav__20260921S.xls") if raw else None
    if p_nav and os.path.exists(p_nav):
        nav = m20.parse_ishares_nav(p_nav)
    if nav is None:
        OUT["B"]["nav_note"] = "IBIT NAV history not available in the snapshot (the iShares download returned a web page)"
    if nav is not None:
        navr = nav.sort_index().pct_change().reindex(cal)
        OUT["B"]["IBIT NAV"] = ac1(navr.values)
        cl = pd.read_csv(os.path.join(raw, "IBIT__20260921.csv"), parse_dates=["Date"]).set_index("Date")["Close"]
        prem = (cl / nav.reindex(cl.index) - 1).reindex(cal).dropna()
        b = float(np.polyfit(prem.values[:-1], prem.values[1:], 1)[0])
        OUT["B"]["premium"] = dict(mean_bp=float(prem.mean() * 1e4), sd_bp=float(prem.std() * 1e4), ar1=b,
                                   half_life=float(math.log(0.5) / math.log(b)) if 0 < b < 1 else None, n=len(prem))
        # contribution: IBIT close return = NAV return + change in premium (approximately)
        dprem = prem.diff().reindex(cal)
        OUT["B"]["premium_change"] = ac1(dprem.values)
    bt = LD._px(raw, "BTC-USD", "S", "AdjClose") if raw else None
    if bt is not None:
        OUT["B"]["BTC-USD 00:00 UTC"] = ac1(bt.reindex(cal).pct_change().values)
    for k, v in OUT["B"].items():
        if "rho1" in v:
            rowsB.append([k, v["n"], f"{v['rho1']:+.3f} ({pf(v['p_rho1'])})", pf(v["q10_p"])])
    L.append("## B. First-order autocorrelation of daily returns, ETF sample\n")
    L.append(md(rowsB, ["Series", "Days", "ρ1 (p)", "Ljung–Box Q(10) p"]))
    if "nav_note" in OUT["B"]:
        L.append(OUT["B"]["nav_note"] + "; the NAV comparison is replaced by Bitcoin at 16:00 ET, the reference time of "
                 "the benchmark used to strike IBIT's NAV.\n")
    if "premium" in OUT["B"]:
        pr = OUT["B"]["premium"]
        L.append(f"IBIT closing premium to NAV: mean {pr['mean_bp']:+.1f} bp, s.d. {pr['sd_bp']:.1f} bp, AR(1) "
                 f"coefficient {pr['ar1']:.3f}" + (f" (half-life {pr['half_life']:.1f} days)" if pr["half_life"] else "") + ".\n")

    # ---------------- C
    rowsC = []
    for lab, (A, Bn) in (("ETF sample", ("ETF: IBIT portfolio", "ETF: GLD portfolio")),
                         ("Bitcoin long sample", ("Long: BTC portfolio", "Long: GLD portfolio"))):
        if A not in series:
            continue
        rA, rB = series[A], series[Bn]
        hs = es(rA) / es(rB)
        sd = (es(rA / rA.std(ddof=1)) / es(rB / rB.std(ddof=1)))
        g0 = es(fits[A][0]["z"]) / es(fits[Bn][0]["z"])
        g1 = es(fits[A][1]["z"]) / es(fits[Bn][1]["z"])
        OUT["C"][lab] = dict(es_ratio_hs=hs, shape_sd=sd, shape_garch_const=g0, shape_garch_ar1=g1)
        rowsC.append([lab, f"{hs:.3f}", f"{sd:.3f}", f"{g0:.3f}", f"{g1:.3f}"])
    L.append("## C. Tail comparison under different normalizations (ES95 ratio, Bitcoin portfolio / GLD portfolio)\n")
    L.append(md(rowsC, ["Sample", "Raw returns (HS, no model)", "Standardized by sample s.d.",
                        "GARCH-t residuals, constant mean", "GARCH-t residuals, AR(1) mean"]))
    txt = "\n".join(L)
    print(txt)
    open(os.path.join(a.outdir, "mean_specification.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "mean_specification.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
