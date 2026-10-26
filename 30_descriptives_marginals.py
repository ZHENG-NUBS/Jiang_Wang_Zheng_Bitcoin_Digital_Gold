#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30_descriptives_marginals.py — Distribution of the individual assets and their marginal volatility
models in the three samples (second-round revision: Reviewer 2, comments 2, 4 and 10; Reviewer 1,
comment 9)

A  Descriptive statistics of daily simple returns for each asset: annualized mean (252 x mean) and
   volatility (sqrt(252) x sd), skewness, excess kurtosis, minimum, maximum, Jarque-Bera test,
   Ljung-Box Q(10) of returns and of squared returns, Engle's ARCH-LM(10) test.
B  Marginal GARCH(1,1) models with Student-t innovations and a constant mean, estimated by maximum
   likelihood with the arch package on percentage returns: omega, alpha, beta, persistence, half-life
   of a volatility shock, nu; and, for comparison, the BIC of GJR-GARCH(1,1)-t, EGARCH(1,1)-t,
   GARCH(1,1)-t with an AR(1) mean, and GARCH(1,1)-skewed-t. Ljung-Box Q(20) on standardized
   residuals and their squares.
C  Short versus long samples: the same GARCH(1,1)-t parameters for Bitcoin, gold, equities and bonds
   estimated on the ETF sample (2024-2026), the Bitcoin long sample (2014-2026) and the gold long
   sample (2000-2026). The data are raw daily returns; no smoothing or moving average is applied
   anywhere in the analysis.

Samples (see longdata.py): ETF sample (IBIT, GLD, TLT, SPY, 607 days); Bitcoin long sample (BTC-USD,
GLD, TLT, SPY, Sep 2014 - Jun 2026); gold long sample (GOLD = gold futures then GLD, SPY, TLT,
Aug 2000 - Jun 2026), if the snapshot __20261021G is present.

Usage
    python 30_descriptives_marginals.py          # writes out/30/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402
from arch import arch_model  # noqa: E402
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402


def desc(r):
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    jb = stats.jarque_bera(r)
    lb = acorr_ljungbox(r, lags=[10], return_df=True)
    lb2 = acorr_ljungbox(r ** 2, lags=[10], return_df=True)
    lm = het_arch(r, nlags=10)
    return dict(n=len(r), mean_a=252 * r.mean(), vol_a=math.sqrt(252) * r.std(ddof=1),
                skew=float(stats.skew(r)), exkurt=float(stats.kurtosis(r)), min=r.min(), max=r.max(),
                jb=float(jb.statistic), jb_p=float(jb.pvalue),
                q10=float(lb.lb_stat.iloc[0]), q10_p=float(lb.lb_pvalue.iloc[0]),
                q10sq=float(lb2.lb_stat.iloc[0]), q10sq_p=float(lb2.lb_pvalue.iloc[0]),
                arch_lm=float(lm[0]), arch_lm_p=float(lm[1]))


def fit(y, mean="Constant", vol="GARCH", o=0, dist="t", lags=0, first_obs=None):
    kw = dict(mean=mean, vol=vol, p=1, o=o, q=1, dist=dist)
    if mean == "AR":
        kw["lags"] = lags
    return arch_model(y, **kw).fit(disp="off", show_warning=False, first_obs=first_obs,
                                   options={"maxiter": 3000})


def garch_block(r):
    y = np.asarray(r, float) * 100
    y = y[np.isfinite(y)]
    base = fit(y)
    p = base.params
    a, b = float(p["alpha[1]"]), float(p["beta[1]"])
    pers = a + b
    z = np.asarray(base.std_resid)
    z = z[np.isfinite(z)]
    lbz = acorr_ljungbox(z, lags=[20], return_df=True)
    lbz2 = acorr_ljungbox(z ** 2, lags=[20], return_df=True)
    # Model comparison on a common estimation sample: the AR(1) mean uses the first observation as
    # its lag, so every constant-mean model is evaluated from the second observation (first_obs=1)
    # and all likelihoods and BICs refer to the same n - 1 observations.
    alt = {}
    for lab, kw in (("GARCH-t", dict(first_obs=1)), ("GJR-GARCH-t", dict(o=1, first_obs=1)),
                    ("EGARCH-t", dict(vol="EGARCH", first_obs=1)), ("AR(1)-GARCH-t", dict(mean="AR", lags=1)),
                    ("GARCH-skew-t", dict(dist="skewt", first_obs=1))):
        try:
            res = fit(y, **kw)
            alt[lab] = dict(bic=float(res.bic), aic=float(res.aic), llf=float(res.loglikelihood), nobs=int(res.nobs))
            if lab == "GJR-GARCH-t":
                alt[lab]["gamma"] = float(res.params["gamma[1]"])
                alt[lab]["gamma_p"] = float(res.pvalues["gamma[1]"])
            if lab == "AR(1)-GARCH-t":
                k = [c for c in res.params.index if c.startswith("y[")][0]
                alt[lab]["phi"] = float(res.params[k]); alt[lab]["phi_p"] = float(res.pvalues[k])
        except Exception as e:  # noqa: BLE001
            alt[lab] = dict(error=str(e))
    return dict(n=len(y), mu=float(p["mu"]), omega=float(p["omega"]), alpha=a, beta=b, persistence=pers,
                half_life=(math.log(0.5) / math.log(pers)) if 0 < pers < 1 else float("inf"),
                nu=float(p["nu"]), se={k: float(v) for k, v in base.std_err.items()},
                bic=float(base.bic), aic=float(base.aic), llf=float(base.loglikelihood),
                uncond_vol_a=math.sqrt(252 * float(p["omega"]) / max(1 - pers, 1e-9)) if pers < 1 else float("inf"),
                lb_z20=float(lbz.lb_stat.iloc[0]), lb_z20_p=float(lbz.lb_pvalue.iloc[0]),
                lb_z2_20=float(lbz2.lb_stat.iloc[0]), lb_z2_20_p=float(lbz2.lb_pvalue.iloc[0]), alt=alt)


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/30")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    samples = {}
    E = LD.load_etf()
    samples["etf"] = ("ETF sample", E, ["IBIT", "GLD", "TLT", "SPY"])
    try:
        L = LD.load_btc_long()
        samples["btc_long"] = ("Bitcoin long sample", L, ["BTC", "GLD", "TLT", "SPY"])
    except FileNotFoundError as e:
        print("(no Bitcoin long sample:", e, ")")
    G = LD.load_gold_long()
    if G is not None:
        samples["gold_long"] = ("Gold long sample", G, ["GOLD", "SPY", "TLT"])
    else:
        print("(no gold long sample __20261021G; run 01f_fetch_gold_long.py)")

    OUT, lines = {}, ["# Descriptive statistics and marginal volatility models (script 30)\n",
                      "Daily simple returns, no smoothing. Annualized mean = 252 × daily mean; annualized "
                      "volatility = √252 × daily standard deviation. GARCH models on percentage returns.\n"]
    for key, (lab, D, assets) in samples.items():
        OUT[key] = {"label": lab, "dates": [str(D.index[0].date()), str(D.index[-1].date())], "assets": {}}
        lines.append(f"## {lab}: {D.index[0].date()} → {D.index[-1].date()}\n")
        rows, grows, arows = [], [], []
        for s in assets:
            r = D[s].dropna()
            ds = desc(r.values)
            gb = garch_block(r.values)
            OUT[key]["assets"][s] = dict(desc=ds, garch=gb, first=str(r.index[0].date()))
            rows.append([s, ds["n"], f"{ds['mean_a'] * 100:.1f}", f"{ds['vol_a'] * 100:.1f}", f"{ds['skew']:.2f}",
                         f"{ds['exkurt']:.2f}", f"{ds['min'] * 100:.2f}", f"{ds['max'] * 100:.2f}",
                         f"{ds['jb']:.0f} ({ds['jb_p']:.3f})", f"{ds['q10']:.1f} ({ds['q10_p']:.3f})",
                         f"{ds['q10sq']:.1f} ({ds['q10sq_p']:.3f})", f"{ds['arch_lm']:.1f} ({ds['arch_lm_p']:.3f})"])
            grows.append([s, f"{gb['omega']:.4f}", f"{gb['alpha']:.3f} ({gb['se']['alpha[1]']:.3f})",
                          f"{gb['beta']:.3f} ({gb['se']['beta[1]']:.3f})", f"{gb['persistence']:.3f}",
                          f"{gb['half_life']:.1f}", f"{gb['nu']:.2f}", f"{gb['uncond_vol_a']:.1f}",
                          f"{gb['lb_z20']:.1f} ({gb['lb_z20_p']:.3f})", f"{gb['lb_z2_20']:.1f} ({gb['lb_z2_20_p']:.3f})"])
            assert len({v["nobs"] for v in gb["alt"].values() if "nobs" in v}) == 1, "unequal estimation samples"
            best = min([(k, v["bic"]) for k, v in gb["alt"].items() if "bic" in v], key=lambda t: t[1])[0]
            arows.append([s] + [f"{gb['alt'][k]['bic']:.1f}" if "bic" in gb["alt"][k] else "—"
                                for k in ("GARCH-t", "GJR-GARCH-t", "EGARCH-t", "AR(1)-GARCH-t", "GARCH-skew-t")]
                         + [best,
                            f"{gb['alt']['GJR-GARCH-t'].get('gamma', float('nan')):.3f} ({gb['alt']['GJR-GARCH-t'].get('gamma_p', float('nan')):.3f})",
                            f"{gb['alt']['AR(1)-GARCH-t'].get('phi', float('nan')):.3f} ({gb['alt']['AR(1)-GARCH-t'].get('phi_p', float('nan')):.3f})"])
        lines.append("### A. Descriptive statistics (p-values in parentheses)\n")
        lines.append(md(rows, ["Asset", "Days", "Mean %/yr", "Vol %/yr", "Skew", "Ex. kurt", "Min %", "Max %",
                               "Jarque–Bera", "Q(10)", "Q²(10)", "ARCH-LM(10)"]))
        lines.append("### B. GARCH(1,1)-t (robust standard errors in parentheses)\n")
        lines.append(md(grows, ["Asset", "ω", "α", "β", "α+β", "Half-life (days)", "ν", "Uncond. vol %/yr",
                                "Q(20) of z", "Q(20) of z²"]))
        lines.append("### C. Alternative specifications: BIC on a common sample of n − 1 days (lower is better)\n")
        lines.append(md(arows, ["Asset", "GARCH-t", "GJR-GARCH-t", "EGARCH-t", "AR(1)-GARCH-t", "GARCH-skew-t",
                                "Lowest BIC", "GJR γ (p)", "AR(1) φ (p)"]))
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.outdir, "descriptives_marginals.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "descriptives_marginals.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
