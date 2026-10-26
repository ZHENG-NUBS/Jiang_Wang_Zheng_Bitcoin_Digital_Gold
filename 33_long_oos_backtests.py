#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
33_long_oos_backtests.py — Out-of-sample VaR/ES backtests on the long samples, by market regime and
sub-period, with an AR(1)-GARCH-t mean specification
(second-round revision: Reviewer 2, comments 5, 8 and 10; Reviewer 1, comment 9)

Design. Exactly the forecasting and evaluation code of script 21 (imported): one-step-ahead VaR and
ES at 95% and 99% from six models (Normal, Student-t, historical simulation, EWMA, GARCH-normal,
GARCH-t), moving estimation window, GARCH parameters re-estimated every 20 days with the recursion
advanced daily between re-estimations; Kupiec, Christoffersen, dynamic-quantile (simulated
p-value), Acerbi–Szekely Z1/Z2, McNeil–Frey tests, FZ0 and pinball losses, and the 90% model
confidence set. A seventh model, GARCH-t with an AR(1) conditional mean, is added (Reviewer 1,
comment 9: residual autocorrelation in the IBIT portfolio).

Samples and portfolios (one-third each, daily rebalanced):
    ETF sample         IBIT portfolio, GLD portfolio                      window 250 (as script 21)
    Bitcoin long       BTC portfolio (BTC-USD, SPY, TLT), GLD portfolio   windows 250 and 1000
    Gold long          gold portfolio (gold, SPY, TLT), from July 2002    windows 250 and 1000
Additional breakdowns from the stored forecasts: VaR breach rates and Kupiec p-values in each SPY
regime of script 31 (tranquil / turbulent / crisis) and in calendar sub-periods; the out-of-sample
ES ratio (BTC or IBIT portfolio over GLD portfolio) with a stationary block bootstrap interval.

Usage
    python 33_long_oos_backtests.py                 # full run, writes out/33/ (about 30-60 minutes)
    python 33_long_oos_backtests.py --quick         # fewer simulation draws
    python 33_long_oos_backtests.py --only btc_long # one sample
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 20261024
REG = ["tranquil", "turbulent", "crisis"]
AR_NAME = "AR-GARCH-t"


def load(name, fname):
    p = os.path.join(HERE, fname)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m21 = load("m21", "21_forecast_design.py")
warnings.filterwarnings("ignore")


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def pf(p):
    return "—" if p is None or not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def roll_ar_garch(r, window, refit, fails):
    """AR(1)-GARCH(1,1)-t one-step forecasts; information up to t-1 only."""
    from arch import arch_model
    N = m21.N_START
    n = len(r); x = r * 100.0
    n_out = n - N
    mu = np.full(n_out, np.nan); sd = np.full(n_out, np.nan); nu = np.full(n_out, np.nan)
    par = None
    for k in range(n_out):
        t = N + k
        if k % refit == 0:
            lo = t - N if window == "moving" else 0
            y = x[lo:t]
            try:
                res = arch_model(y, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t").fit(
                    disp="off", show_warning=False, options={"maxiter": 2000})
                if res.convergence_flag != 0:
                    raise ArithmeticError
                p = res.params
                c = p["Const"]; phi = [v for k_, v in p.items() if k_.startswith("y[")][0]
                par = dict(c=c, phi=phi, omega=p["omega"], alpha=p["alpha[1]"], beta=p["beta[1]"], nu=p["nu"])
                cv = np.asarray(res.conditional_volatility, float)
                s2p = cv[-1] ** 2
                ep = y[-1] - (c + phi * y[-2])
            except Exception:  # noqa: BLE001
                fails.append(k)
        if par is None:
            continue
        s2 = par["omega"] + par["alpha"] * ep ** 2 + par["beta"] * s2p
        m_t = par["c"] + par["phi"] * x[t - 1]
        mu[k] = m_t / 100; sd[k] = math.sqrt(s2) / 100; nu[k] = par["nu"]
        s2p = s2; ep = x[t] - m_t
    return mu, sd, nu


def add_ar(F, r, window, logs):
    fl = []
    mu, sd, nu = roll_ar_garch(r, window, m21.REFIT, fl)
    logs.setdefault("fails", {})[AR_NAME] = len(fl)
    f = dict(kind="t_std", mu=mu, sd=sd, nu=nu, VaR={}, ES={})
    for a in m21.ALPHAS:
        f["VaR"][a] = mu + sd * m21.var_z_t(a, nu)
        f["ES"][a] = mu + sd * m21.es_z_t(a, nu)
    F[AR_NAME] = f


def kupiec_p(x, n, a):
    return float(stats.chi2.sf(m21.lr_uc(int(x), int(n), a), 1)) if n > 0 else np.nan


def breakdown(ro, F, labels, groups):
    """Breach rate and Kupiec p by group label (regime or sub-period)."""
    out = {}
    for m in m21.MODELS:
        out[m] = {}
        for a in m21.ALPHAS:
            v = F[m]["VaR"][a]
            out[m][a] = {}
            for g in groups:
                sel = labels == g
                n = int(sel.sum())
                x = int((ro[sel] < v[sel]).sum()) if n else 0
                out[m][a][g] = dict(n=n, x=x, rate=x / n if n else np.nan, kupiec_p=kupiec_p(x, n, a))
    return out


def run_design(key, label, P, dates, window, rngs, reg, periods, ratio_pair, detail=True):
    m21.N_START = window
    t0 = time.time()
    FC, LOG = {}, {}
    for nm, r in P.items():
        LOG[nm] = {}
        FC[nm] = m21.build_forecasts(r, "moving", m21.REFIT, LOG[nm])
        add_ar(FC[nm], r, "moving", LOG[nm])
    ev = m21.evaluate(P, FC, rngs, detail=detail)
    od = dates[window:]
    res = {"label": label, "window": window, "oos": [str(od[0].date()), str(od[-1].date())], "n_oos": len(od),
           "fails": {nm: LOG[nm].get("fails", {}) for nm in P}, "tests": {}, "regimes": {}, "periods": {}}
    for (nm, a), v in ev.items():
        res["tests"][f"{nm}|{a}"] = dict(res=v["res"], mcs=v["mcs"])
    lab_r = reg.regime.reindex(od).values
    lab_r = np.where(np.isnan(lab_r.astype(float)), -1, lab_r).astype(int)
    lab_p = np.array([next((p for p, (s, e) in periods.items() if s <= str(d.date()) <= e), "other") for d in od])
    for nm, r in P.items():
        ro = r[window:]
        res["regimes"][nm] = breakdown(ro, FC[nm], lab_r, [0, 1, 2])
        res["periods"][nm] = breakdown(ro, FC[nm], lab_p, list(periods))
    if ratio_pair:
        res["oos_ratio"] = m21.oos_ratio(FC, P, rngs["ratio"], num=ratio_pair[0], den=ratio_pair[1])
    res["seconds"] = time.time() - t0
    return res


def report(res):
    L = [f"## {res['label']}, moving window {res['window']} days — out of sample {res['oos'][0]} → "
         f"{res['oos'][1]} ({res['n_oos']} days)\n"]
    for k, t in res["tests"].items():
        nm, a = k.split("|"); a = float(a)
        rows = []
        mcs = t["mcs"].get("fz0", {}).get("included", [])
        for m in m21.MODELS:
            d = t["res"][m]
            rows.append([m, f"{d['x']} ({d['rate'] * 100:.2f}%)", pf(d["uc_p"]), pf(d["cc_p"]),
                         pf(d.get("dq_p_sim", d["dq_p"])), pf(d.get("z2_p")), pf(d.get("mf_p")),
                         f"{d['fz0']:.4f}", "yes" if m in mcs else "no"])
        L.append(f"### {nm}, {int((1 - a) * 100)}% level\n")
        L.append(md(rows, ["Model", "Breaches (rate)", "Kupiec p", "Christoffersen p", "DQ p (sim.)",
                           "AS Z2 p", "McNeil–Frey p", "FZ0 loss", "In 90% MCS (FZ0)"]))
    for nm, br in res["regimes"].items():
        rows = []
        for m in ("HS", "GARCH-t", AR_NAME, "EWMA"):
            cells = []
            for a in m21.ALPHAS:
                for g in (0, 1, 2):
                    c = br[m][a][g]
                    cells.append(f"{c['rate'] * 100:.1f}% [{c['n']}] ({pf(c['kupiec_p'])})" if c["n"] else "—")
            rows.append([m] + cells)
        L.append(f"### {nm}: breach rate by SPY regime [days] (Kupiec p)\n")
        L.append(md(rows, ["Model"] + [f"{int((1 - a) * 100)}% {REG[g]}" for a in m21.ALPHAS for g in (0, 1, 2)]))
    for nm, br in res["periods"].items():
        rows = []
        groups = list(next(iter(br.values()))[0.05].keys())
        for m in ("HS", "GARCH-t", AR_NAME):
            rows.append([m] + [f"{br[m][0.05][g]['rate'] * 100:.1f}% ({pf(br[m][0.05][g]['kupiec_p'])}) / "
                               f"{br[m][0.01][g]['rate'] * 100:.1f}% ({pf(br[m][0.01][g]['kupiec_p'])})"
                               if br[m][0.05][g]["n"] else "—" for g in groups])
        L.append(f"### {nm}: breach rate by sub-period, 95% (Kupiec p) / 99% (Kupiec p)\n")
        L.append(md(rows, ["Model"] + groups))
    if "oos_ratio" in res:
        rows = [[m, f"{v['ratio']:.3f}", f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}]"] for m, v in res["oos_ratio"].items()]
        L.append("### Out-of-sample ES95 ratio (mean ES forecast; realized HS in the last row)\n")
        L.append(md(rows, ["Model", "Ratio", "95% CI (block bootstrap)"]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/33")
    ap.add_argument("--regimes", default="out/31/regimes.csv")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.quick:
        m21.B_AS, m21.B_MF, m21.B_MCS, m21.B_RATIO = 300, 1000, 1000, 500
    m21.MODELS = list(m21.MODELS) + [AR_NAME]
    reg = pd.read_csv(a.regimes, parse_dates=["Date"]).set_index("Date")
    ss = np.random.SeedSequence(SEED)
    E = LD.load_etf(); B = LD.load_btc_long(); G = LD.load_gold_long()
    designs = []
    designs.append(("etf", "ETF sample", {"IBIT portfolio": ((E.IBIT + E.SPY + E.TLT) / 3).values,
                                          "GLD portfolio": ((E.GLD + E.SPY + E.TLT) / 3).values}, E.index, (250,),
                    {"2025": ("2025-01-01", "2025-12-31"), "2026": ("2026-01-01", "2026-12-31")},
                    ("IBIT portfolio", "GLD portfolio")))
    designs.append(("btc_long", "Bitcoin long sample", {"BTC portfolio": ((B.BTC + B.SPY + B.TLT) / 3).values,
                                                        "GLD portfolio": ((B.GLD + B.SPY + B.TLT) / 3).values},
                    B.index, (250, 1000),
                    {"2015–2019": ("2015-01-01", "2019-12-31"), "2020–2022": ("2020-01-01", "2022-12-31"),
                     "2023–2026": ("2023-01-01", "2026-12-31")}, ("BTC portfolio", "GLD portfolio")))
    if G is not None:
        G2 = G[G.TLT.notna()]
        designs.append(("gold_long", "Gold long sample", {"Gold portfolio": ((G2.GOLD + G2.SPY + G2.TLT) / 3).values},
                        G2.index, (250, 1000),
                        {"2003–2007": ("2003-01-01", "2007-12-31"), "2008–2012": ("2008-01-01", "2012-12-31"),
                         "2013–2019": ("2013-01-01", "2019-12-31"), "2020–2026": ("2020-01-01", "2026-12-31")}, None))
    OUT, txt = {}, ["# Out-of-sample VaR/ES backtests on the long samples (script 33)\n",
                    f"Models: {', '.join(m21.MODELS)}. GARCH re-estimated every {m21.REFIT} days. "
                    f"Simulation draws: AS {m21.B_AS}, McNeil–Frey {m21.B_MF}, MCS {m21.B_MCS}.\n"]
    for key, label, P, dates, wins, periods, pair in designs:
        if a.only and key != a.only:
            continue
        for win in wins:
            kids = ss.spawn(4)
            rngs = {"as": np.random.default_rng(kids[0]), "mf": np.random.default_rng(kids[1]),
                    "mcs": np.random.default_rng(kids[2]), "ratio": np.random.default_rng(kids[3])}
            res = run_design(key, label, P, dates, win, rngs, reg, periods, pair)
            OUT[f"{key}|{win}"] = res
            txt.append(report(res))
            print(f"{label}, window {win}: {res['seconds']:.0f} s", flush=True)
            open(os.path.join(a.outdir, "long_oos_backtests.md"), "w", encoding="utf-8").write("\n".join(txt))
            json.dump(OUT, open(os.path.join(a.outdir, "long_oos_backtests.json"), "w"), indent=1,
                      default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("\n".join(txt))


if __name__ == "__main__":
    main()
