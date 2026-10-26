#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
31_regimes_contagion.py — Markov-switching market regimes and a Forbes–Rigobon contagion test
(second-round revision: Reviewer 2, comments 3, 5, 6 and 7)

A  Market regimes. A Markov-switching model with regime-dependent mean and variance
   (Hamilton 1989) is fitted to daily SPY returns over the longest available sample
   (gold long sample, August 2000 - June 2026; the Bitcoin long sample if the gold snapshot is
   absent). Three regimes, ordered by variance, are labelled tranquil, turbulent and crisis; a
   two-regime model is reported for comparison (BIC). Days are classified by the largest smoothed
   probability; the classification by filtered (real-time) probabilities is a robustness check.
B  Regime-dependent behaviour of Bitcoin, gold and Treasuries: mean, volatility, correlation and
   beta with SPY, ES95 of the asset and of the two one-third portfolios, in each regime and sample.
C  Forbes and Rigobon (2002) test of contagion. The crisis-regime correlation is adjusted for the
   higher variance of the source market (SPY): rho* = rho_c / sqrt(1 + delta (1 - rho_c^2)),
   delta = var_crisis(SPY) / var_tranquil(SPY) - 1, and compared with the tranquil-regime
   correlation (Fisher z and a stationary block bootstrap that keeps the regime labels attached
   to each day). A significant increase indicates contagion; no increase after adjustment
   indicates interdependence. The same comparison is made for the turbulent regime.
D  ETF sample: number of ETF-sample days in each regime (how many stress days the 607-day sample
   actually contains).
E  Figure: smoothed regime probabilities and SPY drawdown.

The regime series is written to out/31/regimes.csv and read by scripts 32 and 33.

Usage
    python 31_regimes_contagion.py               # writes out/31/
    python 31_regimes_contagion.py --boot 500    # quicker trial
"""

from __future__ import annotations

import os

# single-threaded linear algebra: the Markov-switching likelihood is then evaluated in the same order on every
# machine, so that the regime classification is reproducible to the last day
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression  # noqa: E402

warnings.filterwarnings("ignore")   # again after the imports (statsmodels resets some filters)

SEED = 20261022
LABELS = {2: ["calm", "stress"], 3: ["tranquil", "turbulent", "crisis"]}
L_BOOT = 10


def md(rows, head):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def fit_ms(y, k, seed=SEED, reps=20):
    """Markov-switching mean and variance; returns a dict with regimes ordered by variance."""
    np.random.seed(seed + k)
    mod = MarkovRegression(y, k_regimes=k, trend="c", switching_variance=True)
    best = None
    for em in (50, 200):
        try:
            r = mod.fit(search_reps=reps, em_iter=em, maxiter=1000, disp=False)
        except Exception:  # noqa: BLE001
            continue
        if best is None or r.llf > best.llf:
            best = r
    r = best
    names = r.model.param_names
    par = dict(zip(names, np.asarray(r.params)))
    sig2 = np.array([par[f"sigma2[{i}]"] for i in range(k)])
    mu = np.array([par[f"const[{i}]"] for i in range(k)])
    order = np.argsort(sig2)
    P = np.asarray(r.regime_transition)[:, :, 0]            # P[j, i] = Pr(S_t = j | S_{t-1} = i)
    Pt = P[np.ix_(order, order)].T                           # row i -> column j, ordered
    sm = np.asarray(r.smoothed_marginal_probabilities)[:, order]
    fl = np.asarray(r.filtered_marginal_probabilities)[:, order]
    return dict(k=k, llf=float(r.llf), bic=float(r.bic), aic=float(r.aic),
                mu=mu[order], sigma=np.sqrt(sig2[order]), P=Pt,
                dur=1.0 / (1.0 - np.diag(Pt)), smoothed=sm, filtered=fl,
                converged=bool(r.mle_retvals.get("converged", False)) if r.mle_retvals else None)


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def es(x, a=0.05):
    x = np.sort(np.asarray(x, float))
    m = max(1, int(math.floor(a * len(x))))
    return -x[:m].mean()


def fr_adjust(rho_c, delta):
    return rho_c / math.sqrt(1 + delta * (1 - rho_c ** 2))


def corr_cols(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (X * Y).sum(0) / np.sqrt((X * X).sum(0) * (Y * Y).sum(0))


def forbes_rigobon(x, s, lab, hi, lo, idx):
    """x, s: asset and SPY returns; lab: regime labels; compares regime hi with regime lo."""
    mh, ml = lab == hi, lab == lo
    nh, nl = int(mh.sum()), int(ml.sum())
    if nh < 20 or nl < 20:
        return None
    rc = float(np.corrcoef(x[mh], s[mh])[0, 1]); rt = float(np.corrcoef(x[ml], s[ml])[0, 1])
    delta = float(s[mh].var(ddof=1) / s[ml].var(ddof=1) - 1)
    ra = fr_adjust(rc, delta)
    z = (math.atanh(ra) - math.atanh(rt)) / math.sqrt(1 / (nh - 3) + 1 / (nl - 3))
    # bootstrap: resample days in blocks, keep each day's regime label
    bd = []
    for c0 in range(0, idx.shape[1], 500):
        I = idx[:, c0:c0 + 500]
        X, S, Lb = x[I], s[I], lab[I]
        for b in range(I.shape[1]):
            h, l_ = Lb[:, b] == hi, Lb[:, b] == lo
            if h.sum() < 10 or l_.sum() < 10:
                continue
            rcb = np.corrcoef(X[h, b], S[h, b])[0, 1]; rtb = np.corrcoef(X[l_, b], S[l_, b])[0, 1]
            db = S[h, b].var(ddof=1) / S[l_, b].var(ddof=1) - 1
            bd.append(fr_adjust(rcb, db) - rtb)
    bd = np.array(bd)
    return dict(n_hi=nh, n_lo=nl, rho_hi=rc, rho_lo=rt, delta=delta, rho_hi_adj=ra, diff_unadj=rc - rt,
                diff_adj=ra - rt, z=z, p_increase=float(stats.norm.sf(z)), p_two=float(2 * stats.norm.sf(abs(z))),
                ci=list(np.percentile(bd, [2.5, 97.5])) if len(bd) else [np.nan, np.nan],
                p_boot_increase=float((bd <= 0).mean()) if len(bd) else np.nan)


def regime_table(D, lab, assets, names, ports):
    rows, out = [], {}
    for g, nm in enumerate(names):
        m = lab == g
        if m.sum() < 5:
            continue
        s = D.SPY.values[m]
        o = {"n": int(m.sum()), "spy_vol": float(s.std(ddof=1) * math.sqrt(252)),
             "spy_mean": float(s.mean() * 252)}
        for a in assets:
            x = D[a].values[m]
            b = float(np.polyfit(s, x, 1)[0])
            o[a] = dict(mean=float(x.mean() * 252), vol=float(x.std(ddof=1) * math.sqrt(252)),
                        rho=float(np.corrcoef(x, s)[0, 1]), beta=b, es95=float(es(x)))
        for pn, cols in ports.items():
            pr = D[cols].mean(axis=1).values[m]
            o[pn] = float(es(pr))
            o[pn + " | mean"] = float(pr.mean() * 252)
            o[pn + " | vol"] = float(pr.std(ddof=1) * math.sqrt(252))
        out[nm] = o
        rows.append([nm, o["n"], f"{o['spy_vol'] * 100:.1f}"] +
                    [f"{o[a]['mean'] * 100:+.1f} / {o[a]['vol'] * 100:.1f} / {o[a]['rho']:+.2f} / {o[a]['beta']:+.2f}"
                     for a in assets] + [f"{o[pn] * 100:.2f} ({o[pn + ' | mean'] * 100:+.0f} / {o[pn + ' | vol'] * 100:.0f})"
                                         for pn in ports])
    return rows, out


def drawdown(p):
    return p / np.maximum.accumulate(p) - 1


def figure(dates, sm, spy, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    cols = ["#d9e6f2", "#7fa7cf", "#1f4e79"]
    ax[0].stackplot(dates, sm.T, colors=cols, labels=["Tranquil", "Turbulent", "Crisis"], linewidth=0)
    ax[0].set_ylim(0, 1); ax[0].set_ylabel("Smoothed probability")
    ax[0].legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.18))
    dd = drawdown(np.cumprod(1 + spy))
    ax[1].fill_between(dates, dd * 100, 0, color="#7f7f7f", linewidth=0)
    ax[1].set_ylabel("SPY drawdown (%)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"fig_ms_regimes.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/31")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    G = LD.load_gold_long()
    B = LD.load_btc_long()
    E = LD.load_etf()
    base = G if G is not None else B
    base_lab = "gold long sample" if G is not None else "Bitcoin long sample"
    y = base.SPY.values * 100
    dates = base.index
    lines = ["# Markov-switching market regimes and Forbes–Rigobon contagion test (script 31)\n",
             f"Regimes estimated on daily SPY returns of the {base_lab}, {dates[0].date()} → {dates[-1].date()} "
             f"({len(y)} days). Mean and variance switch; regimes ordered by variance.\n"]
    MS = {k: fit_ms(y, k) for k in (2, 3)}
    OUT = {"base": base_lab, "dates": [str(dates[0].date()), str(dates[-1].date())], "models": {}}
    rows = []
    for k, m in MS.items():
        OUT["models"][k] = {kk: (v.tolist() if isinstance(v, np.ndarray) and v.ndim <= 2 and v.size < 50 else v)
                            for kk, v in m.items() if kk not in ("smoothed", "filtered")}
        for i, nm in enumerate(LABELS[k]):
            share = float((m["smoothed"].argmax(1) == i).mean())
            rows.append([f"{k}-regime", nm, f"{m['mu'][i] * 2.52:+.1f}", f"{m['sigma'][i] * math.sqrt(252):.1f}",
                         f"{m['P'][i, i]:.3f}", f"{m['dur'][i]:.0f}", f"{share * 100:.1f}",
                         f"{m['llf']:.1f}", f"{m['bic']:.1f}"])
    lines.append("## A. Markov-switching models for SPY\n")
    lines.append(md(rows, ["Model", "Regime", "Mean %/yr", "Vol %/yr", "Stay prob.", "Exp. duration (days)",
                           "Share of days %", "Log-lik.", "BIC"]))
    m3 = MS[3]
    lines.append("Transition matrix of the three-regime model (row = today, column = tomorrow):\n")
    lines.append(md([[LABELS[3][i]] + [f"{m3['P'][i, j]:.3f}" for j in range(3)] for i in range(3)],
                    ["from \\ to"] + LABELS[3]))
    reg = pd.DataFrame({"p_tranquil": m3["smoothed"][:, 0], "p_turbulent": m3["smoothed"][:, 1],
                        "p_crisis": m3["smoothed"][:, 2], "regime": m3["smoothed"].argmax(1),
                        "regime_filtered": m3["filtered"].argmax(1),
                        "f_crisis": m3["filtered"][:, 2]}, index=dates)
    reg.index.name = "Date"
    reg.to_csv(os.path.join(a.outdir, "regimes.csv"))
    agree = float((reg.regime == reg.regime_filtered).mean())
    lines.append(f"Smoothed and filtered classifications agree on {agree * 100:.1f}% of days.\n")

    # ------------------------------------------------ B/C by sample
    rng = np.random.default_rng(SEED)
    samples = [("btc_long", "Bitcoin long sample", B, ["BTC", "GLD", "TLT"],
                {"BTC portfolio": ["BTC", "SPY", "TLT"], "GLD portfolio": ["GLD", "SPY", "TLT"]})]
    if G is not None:
        G2 = G[G.TLT.notna()]
        samples.append(("gold_long", "Gold long sample (from TLT inception)", G2, ["GOLD", "TLT"],
                        {"Gold portfolio": ["GOLD", "SPY", "TLT"]}))
    for key, lab_, D, assets, ports in samples:
        lab = reg.regime.reindex(D.index).values
        ok = ~np.isnan(lab.astype(float))
        D = D[ok]; lab = lab[ok].astype(int)
        for variant, labv in (("smoothed", lab), ("filtered", reg.regime_filtered.reindex(D.index).values.astype(int))):
            rows, out = regime_table(D, labv, assets, LABELS[3], ports)
            OUT.setdefault(key, {})[f"by_regime_{variant}"] = out
            if variant == "smoothed":
                lines.append(f"## B. {lab_}: behaviour by regime ({D.index[0].date()} → {D.index[-1].date()})\n")
                lines.append("Asset cells: annualized mean % / annualized volatility % / correlation with SPY / beta on SPY. "
                             "Portfolio cells: daily ES95 % (annualized mean % / annualized volatility %).\n")
                lines.append(md(rows, ["Regime", "Days", "SPY vol %"] + assets + list(ports)))
        idx = stationary_boot(len(D), a.boot, L_BOOT, rng)
        fr_rows = []
        for x in assets:
            for hi, lo in ((2, 0), (1, 0)):
                r = forbes_rigobon(D[x].values, D.SPY.values, lab, hi, lo, idx)
                if r is None:
                    continue
                OUT[key].setdefault("forbes_rigobon", {})[f"{x}|{LABELS[3][hi]}_vs_{LABELS[3][lo]}"] = r
                # The Forbes–Rigobon adjustment shrinks any crisis correlation toward zero; for an asset that
                # is negatively correlated with equities a less negative adjusted value is not contagion.
                if r["rho_hi"] < 0 and r["rho_lo"] < 0:
                    verdict = "not applicable (negative correlation in both regimes)"
                elif r["p_boot_increase"] < 0.05 and r["ci"][0] > 0:
                    verdict = "contagion (significant increase)"
                elif r["ci"][1] < 0:
                    verdict = "decoupling (significant decrease)"
                else:
                    verdict = "interdependence (no significant change)"
                fr_rows.append([x, f"{LABELS[3][hi]} vs {LABELS[3][lo]}", f"{r['n_hi']} / {r['n_lo']}",
                                f"{r['rho_lo']:+.3f}", f"{r['rho_hi']:+.3f}", f"{r['delta']:.2f}",
                                f"{r['rho_hi_adj']:+.3f}", f"{r['diff_adj']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]",
                                f"{r['p_increase']:.3f}", f"{r['p_boot_increase']:.3f}", verdict])
        del idx
        lines.append(f"## C. {lab_}: Forbes–Rigobon test (stationary block bootstrap, mean block {L_BOOT} days, "
                     f"B = {a.boot})\n")
        lines.append(md(fr_rows, ["Asset", "Comparison", "Days (high / low)", "ρ low", "ρ high", "δ",
                                  "ρ* high (adjusted)", "ρ* − ρ low [95% CI]", "p (Fisher z, increase)",
                                  "p (bootstrap, increase)", "Reading"]))

    # ------------------------------------------------ D. ETF sample
    labE = reg.regime.reindex(E.index)
    cnt = {LABELS[3][i]: int((labE == i).sum()) for i in range(3)}
    cntf = {LABELS[3][i]: int((reg.regime_filtered.reindex(E.index) == i).sum()) for i in range(3)}
    OUT["etf_days_by_regime"] = {"smoothed": cnt, "filtered": cntf, "unclassified": int(labE.isna().sum())}
    lines.append("## D. ETF sample (2024-01-12 → 2026-06-15): days by regime\n")
    lines.append(md([["smoothed"] + [cnt[k] for k in LABELS[3]], ["filtered"] + [cntf[k] for k in LABELS[3]]],
                    ["Classification"] + LABELS[3]))
    vix = base.VIX.reindex(reg.index)
    lines.append("VIX by regime (whole regime sample): " + "; ".join(
        f"{LABELS[3][i]} median {vix[reg.regime == i].median():.1f}" for i in range(3)) + "\n")
    OUT["vix_median_by_regime"] = {LABELS[3][i]: float(vix[reg.regime == i].median()) for i in range(3)}
    figure(dates, m3["smoothed"], base.SPY.values, a.outdir)
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.outdir, "regimes_contagion.md"), "w", encoding="utf-8").write(txt)
    json.dump(OUT, open(os.path.join(a.outdir, "regimes_contagion.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
