# -*- coding: utf-8 -*-
"""
42_reviewer3_checks.py — spanning tests, an equivalence test of weak safe-haven status, the SPY × VIX interaction
without the April 2025 episode, and bias-corrected intervals for the quantile-regression tail slope.

A. Mean–variance spanning (Huberman & Kandel, 1987; step-down form of Kan & Zhou, 2012; long-only condition of
   De Roon, Nijman & Werker, 2001). For a test asset X and benchmark assets B (the other three of IBIT or BTC-USD,
   GLD, SPY, TLT): r_X = α + β'r_B + ε, δ = 1 − β'1. Spanning: α = 0 and δ = 0. δ = 0 means that the
   global-minimum-variance (GMV) portfolios of B and of B ∪ {X} coincide; it involves covariances only. α = 0
   concerns the tangency portfolio (zero-beta rate zero). Step-down: F1 tests α = 0; F2 tests δ = 0 with α = 0
   imposed. Newey–West HAC Wald tests (automatic bandwidth, as script 11) replace the normal-theory F tests. With a
   long-only position in X and an interior benchmark GMV, X lowers the GMV variance if and only if δ > 0 (its
   covariance with the benchmark GMV is below the GMV variance), so the long-only GMV test is one-sided,
   H0: δ ≤ 0. The long-only GMV of B and of B ∪ {X} is also computed directly, with a stationary block bootstrap
   (2000 replications, expected block length 5) of the volatility reduction and of X's weight.
B. Equivalence test of weak safe-haven status. λL(5%) = P(asset rank ≤ 5% | SPY rank ≤ 5%), 0.05 under independence.
   Same rule for every asset: (i) positive lower-tail dependence if the 95% interval lies above 0.05 (the convention
   of script 22 and Table 11);
   (ii) equivalence with independence (weak safe haven at the margin) if the 90% interval lies inside
   (0.05 − m, 0.05 + m) with m = 0.05, i.e. the upper 95% bound is below 0.10 (two one-sided tests; the lower
   bound is not binding). The smallest margin at which (ii) holds is also reported. Stationary block bootstrap,
   2000 replications, expected block length 5 (as script 22). Because the cross-quantilogram at τ = 0.05 with
   equal hit rates equals (λL − 0.05)/0.95, the rule in terms of the cross-quantilogram is the same test.
C. SPY × VIX interaction (script 11, section C): r_IBIT = a + b r_SPY + c r_SPY·z + d z + ε, z the standardized VIX
   (same day and previous day), HAC errors. Re-estimated without the April 2025 episode, defined as in the
   episode jackknife of script 03 (the cluster of SPY 5%-tail days containing 4 April 2025, extended by three trading
   days on either side), and without the whole of April 2025; leave-one-episode-out over all tail episodes; the
   implied slope b + c z at z = 0, 1, 2.
D. Quantile-regression tail slope β(τ) of the asset on SPY, τ = 0.05 and 0.10: kernel interval (statsmodels,
   as script 22), percentile, bias-corrected (BC) percentile and basic bootstrap intervals from 2000 stationary
   block resamples (expected block length 5; 1000 for the long samples).

Usage: python 42_reviewer3_checks.py [--outdir out/42] [--B 2000]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm  # noqa: E402
from statsmodels.tools.sm_exceptions import IterationLimitWarning  # noqa: E402

warnings.filterwarnings("ignore")                     # after the imports, which reset some filters
warnings.simplefilter("ignore", IterationLimitWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import longdata as LD  # noqa: E402

SEED = 20261029
L_BOOT = 5
FEE = 0.0025 / 252
CHECKS = []


def check(label, ok, detail=""):
    CHECKS.append(f"CHECK {'PASS' if ok else 'FAIL'}: {label} {detail}")
    print(CHECKS[-1])


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newb = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newb[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def pct(a, q=(2.5, 97.5)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, q)]


def ols_hac(y, X, lag=None):
    """OLS with Newey–West HAC covariance (automatic bandwidth, small-sample factor), as script 11."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    if lag is None:
        lag = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lag + 1):
        A = Xe[L:].T @ Xe[:-L]
        S += (1.0 - L / (lag + 1.0)) * (A + A.T)
    V = XtXi @ S @ XtXi * n / (n - k)
    return b, V


def wald(Rm, q, b, V):
    Rm = np.atleast_2d(Rm)
    d = Rm @ b - q
    W = float(d @ np.linalg.solve(Rm @ V @ Rm.T, d))
    return W, float(stats.chi2.sf(W, Rm.shape[0]))


# ============================================================ A. spanning
def lo_gmv(S):
    """Long-only GMV by enumeration of active sets (exact for small N)."""
    N = S.shape[0]
    best, wb = np.inf, None
    for k in range(1, N + 1):
        for sub in itertools.combinations(range(N), k):
            Si = S[np.ix_(sub, sub)]
            try:
                x = np.linalg.solve(Si, np.ones(k))
            except np.linalg.LinAlgError:
                continue
            w = x / x.sum()
            if (w < -1e-12).any():
                continue
            v = float(w @ Si @ w)
            if v < best:
                best, wb = v, np.zeros(N)
                wb[list(sub)] = w
    return best, wb


def spanning(X, Bm, names, rng, B):
    n = len(X)
    Z = np.column_stack([np.ones(n), Bm])
    b, V = ols_hac(X, Z)
    K = Bm.shape[1]
    alpha, beta = b[0], b[1:]
    delta = 1 - beta.sum()
    Rj = np.zeros((2, K + 1)); Rj[0, 0] = 1; Rj[1, 1:] = 1
    Wj, pj = wald(Rj, np.array([0.0, 1.0]), b, V)
    Wa, pa = wald(Rj[:1], np.array([0.0]), b, V)                       # F1: alpha = 0
    b0, V0 = ols_hac(X, Bm)                                            # alpha = 0 imposed
    d0 = 1 - b0.sum()
    se_d0 = math.sqrt(np.ones(K) @ V0 @ np.ones(K))
    p_d_given_a = float(2 * stats.norm.sf(abs(d0 / se_d0)))            # F2: delta = 0 | alpha = 0
    se_d = math.sqrt(Rj[1] @ V @ Rj[1])
    p_d = float(2 * stats.norm.sf(abs(delta / se_d)))
    p_d_pos = float(stats.norm.sf(delta / se_d))                       # long-only GMV: H0 delta <= 0
    se_a = math.sqrt(V[0, 0])
    p_a_pos = float(stats.norm.sf(alpha / se_a))                       # long-only tangency: H0 alpha <= 0
    # direct long-only GMV
    R4 = np.column_stack([X, Bm])

    def gmv_pair(R):
        S = np.cov(R, rowvar=False)
        v3, w3 = lo_gmv(S[1:, 1:])
        v4, w4 = lo_gmv(S)
        return math.sqrt(252 * v3), math.sqrt(252 * v4), w4[0], w3, w4

    s3, s4, wx, w3, w4 = gmv_pair(R4)
    idx = stationary_boot(n, B, L_BOOT, rng)
    red, wxb = np.empty(B), np.empty(B)
    for j in range(B):
        a3, a4, ww, _, _ = gmv_pair(R4[idx[:, j]])
        red[j], wxb[j] = a3 - a4, ww
    return dict(n=n, alpha_ann=float(alpha * 252), alpha_se_ann=float(se_a * 252), beta=dict(zip(names, map(float, beta))),
                delta=float(delta), delta_se=float(se_d), p_joint=pj, wald_joint=Wj, p_alpha=pa, p_delta=p_d,
                p_delta_given_alpha0=p_d_given_a, delta_alpha0=float(d0), p_delta_pos=p_d_pos, p_alpha_pos=p_a_pos,
                gmv_vol_bench=s3, gmv_vol_ext=s4, gmv_vol_reduction=s3 - s4, w_test_in_gmv=float(wx),
                w_gmv_bench=dict(zip(names, map(float, w3))), w_gmv_ext=dict(zip(["X"] + names, map(float, w4))),
                boot_reduction_ci=pct(red), boot_w_ci=pct(wxb), boot_share_w_pos=float(np.mean(wxb > 1e-6)),
                boot_share_red_gt_10bp=float(np.mean(red > 0.001)))


# ============================================================ B. TOST on lambda_L
def rank_u(x):
    return stats.rankdata(x) / (len(x) + 1.0)


def lam_emp(x, y, u=0.05):
    U, V = rank_u(x), rank_u(y)
    return float(np.mean((U <= u) & (V <= u)) / u)


def lam_tost(y, spy, rng, B, u=0.05, m=0.05):
    n = len(y)
    pt = lam_emp(y, spy, u)
    idx = stationary_boot(n, B, L_BOOT, rng)
    bs = np.array([lam_emp(y[i], spy[i], u) for i in idx.T])
    lo5, hi95 = np.percentile(bs, [5, 95])
    lo25, hi975 = np.percentile(bs, [2.5, 97.5])
    return dict(n=n, lam=pt, ci90=[float(lo5), float(hi95)], ci95=[float(lo25), float(hi975)],
                p_positive=float(np.mean(bs <= u)),                    # H0: lam <= u
                p_equiv_upper=float(np.mean(bs >= u + m)),            # H0: lam >= u + m
                positive=bool(lo25 > u), equivalent=bool(hi95 < u + m and lo5 > u - m),
                smallest_margin=float(max(hi95 - u, u - lo5)), n_joint=int(round(pt * u * n)),
                cq=float((pt - u) / (1 - u)))


def lam_diff(y1, y2, spy, rng, B, u=0.05):
    """Paired difference λL(asset 1) − λL(asset 2) on the same resampled days."""
    n = len(spy)
    idx = stationary_boot(n, B, L_BOOT, rng)
    d = np.array([lam_emp(y1[i], spy[i], u) - lam_emp(y2[i], spy[i], u) for i in idx.T])
    return dict(diff=lam_emp(y1, spy, u) - lam_emp(y2, spy, u), ci95=pct(d), p_le0=float(np.mean(d <= 0)))


# ============================================================ C. interaction
def interaction(y, spy, vix, keep):
    v = vix[keep]
    z = (v - v.mean()) / v.std()
    s = spy[keep]
    X = np.column_stack([np.ones(len(s)), s, s * z, z])
    b, V = ols_hac(y[keep], X)
    se = np.sqrt(np.diag(V))
    p = 2 * stats.norm.sf(np.abs(b / se))
    sl = {}
    for k in (0, 1, 2):
        w = np.array([0, 1, k, 0.0])
        e = float(w @ b); s_ = math.sqrt(w @ V @ w)
        sl[str(k)] = dict(slope=e, ci=[e - 1.96 * s_, e + 1.96 * s_])
    return dict(n=int(keep.sum()), b=float(b[1]), p_b=float(p[1]), c=float(b[2]), se_c=float(se[2]), p_c=float(p[2]),
                slope_at_z=sl, vix_max_z=float(z.max()))


def tail_episodes(spy, dates, q=0.05, gap=5, pad=3):
    thr = np.quantile(spy, q)
    ti = np.where(spy <= thr)[0]
    grp = np.cumsum(np.r_[1, np.diff(ti) > gap])
    out = []
    n = len(spy)
    for g in np.unique(grp):
        r = ti[grp == g]
        lo, hi = max(0, r.min() - pad), min(n - 1, r.max() + pad)
        out.append(dict(first=str(dates[r.min()].date()), last=str(dates[r.max()].date()), lo=int(lo), hi=int(hi),
                        n_tail=int(len(r))))
    return out


# ============================================================ D. QR intervals
def qr_beta(y, x, tau):
    res = sm.QuantReg(y, sm.add_constant(x)).fit(q=tau, max_iter=5000, p_tol=1e-8)
    return float(res.params[1]), float(res.bse[1])


def qr_intervals(y, x, tau, rng, B):
    b, se = qr_beta(y, x, tau)
    idx = stationary_boot(len(y), B, L_BOOT, rng)
    bs = np.array([qr_beta(y[i], x[i], tau)[0] for i in idx.T])
    bs = bs[np.isfinite(bs)]
    prop = float(np.mean(bs < b)) + 0.5 * float(np.mean(bs == b))
    z0 = stats.norm.ppf(min(max(prop, 1e-4), 1 - 1e-4))
    a1, a2 = stats.norm.cdf(2 * z0 - 1.96), stats.norm.cdf(2 * z0 + 1.96)
    perc = pct(bs)
    return dict(beta=b, kernel=[b - 1.96 * se, b + 1.96 * se], percentile=perc,
                bc=[float(np.percentile(bs, 100 * a1)), float(np.percentile(bs, 100 * a2))],
                basic=[2 * b - perc[1], 2 * b - perc[0]], boot_median=float(np.median(bs)), share_below=prop, z0=float(z0),
                boot_sd=float(bs.std()), B=int(len(bs)))


def fmt(x, d=2):
    return f"{x:.{d}f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out/42")
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    E = LD.load_etf()
    BL = LD.load_btc_long()
    BL = BL.assign(BTC=BL.BTC - FEE)
    GL = LD.load_gold_long()
    GLt = GL[GL.TLT.notna()]
    out = dict(spanning={}, tost={}, interaction={}, qr={})

    # ---------------- A
    for samp, D, btc in (("etf", E, "IBIT"), ("long", BL, "BTC")):
        assets = [btc, "GLD", "SPY", "TLT"]
        for x in (btc, "GLD"):
            bn = [c for c in assets if c != x]
            out["spanning"][f"{samp}|{x}"] = spanning(D[x].values, D[bn].values, bn,
                                                      np.random.default_rng([SEED, len(out["spanning"])]), a.B)
            r = out["spanning"][f"{samp}|{x}"]
            print(samp, x, json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}, default=float))
    # self-check of the GMV identity: delta > 0  <=> unconstrained 4-asset GMV puts positive weight on X
    for k, r in out["spanning"].items():
        samp, x = k.split("|")
        D = E if samp == "etf" else BL
        btc = "IBIT" if samp == "etf" else "BTC"
        cols = [x] + [c for c in [btc, "GLD", "SPY", "TLT"] if c != x]
        S = np.cov(D[cols].values, rowvar=False)
        wu = np.linalg.solve(S, np.ones(4)); wu /= wu.sum()
        check(f"sign(delta) = sign(unconstrained GMV weight of X) [{k}]", np.sign(wu[0]) == np.sign(r["delta"]),
              f"delta {r['delta']:.4f}, w {wu[0]:.4f}")

    # ---------------- B
    for samp, D, cols in (("etf", E, ["IBIT", "GLD", "TLT"]), ("long", BL, ["BTC", "GLD", "TLT"]),
                          ("gold", GL, ["GOLD"]), ("gold_tlt", GLt, ["TLT"])):
        for c in cols:
            out["tost"][f"{samp}|{c}"] = lam_tost(D[c].values, D.SPY.values, np.random.default_rng([SEED, 100 + len(out["tost"])]), a.B)
            print("TOST", samp, c, out["tost"][f"{samp}|{c}"])

    out["tost_diff"] = {}
    for samp, D, b_ in (("etf", E, "IBIT"), ("long", BL, "BTC")):
        out["tost_diff"][samp] = lam_diff(D[b_].values, D.GLD.values, D.SPY.values,
                                          np.random.default_rng([SEED, 150 + len(out["tost_diff"])]), a.B)
        print("LAMDIFF", samp, out["tost_diff"][samp])

    # ---------------- C
    spy, y, vix = E.SPY.values, E.IBIT.values, E.VIX.values
    vlag = np.r_[np.nan, vix[:-1]]
    dates = E.index
    eps = tail_episodes(spy, dates)
    apr = [e for e in eps if e["first"] <= "2025-04-04" <= e["last"]]
    check("April 2025 tail episode found", len(apr) == 1, str(apr))
    ep = apr[0]
    for tag, vv in (("same_day", vix), ("previous_day", vlag)):
        ok = np.isfinite(vv)
        base = interaction(y, spy, vv, ok)
        k1 = ok.copy(); k1[ep["lo"]:ep["hi"] + 1] = False
        k2 = ok & ~((dates >= "2025-04-01") & (dates <= "2025-04-30"))
        loo = []
        for e in eps:
            kk = ok.copy(); kk[e["lo"]:e["hi"] + 1] = False
            r_ = interaction(y, spy, vv, kk)
            loo.append(dict(episode=f"{e['first']}..{e['last']}", c=r_["c"], p_c=r_["p_c"]))
        out["interaction"][tag] = dict(full=base, no_april_episode=interaction(y, spy, vv, k1),
                                       no_april_month=interaction(y, spy, vv, k2), episode=ep, leave_one_out=loo)
        print("INTERACTION", tag, json.dumps(out["interaction"][tag], default=float)[:1500])
    check("Full-sample interaction reproduces script 11 (same-day c = −0.18, previous-day c = −0.12)",
          abs(out["interaction"]["same_day"]["full"]["c"] + 0.18) < 0.006 and
          abs(out["interaction"]["previous_day"]["full"]["c"] + 0.12) < 0.006,
          f"{out['interaction']['same_day']['full']['c']:.3f}, {out['interaction']['previous_day']['full']['c']:.3f}")

    # ---------------- D
    for samp, D, cols in (("etf", E, ["IBIT", "GLD"]), ("long", BL, ["BTC", "GLD"]), ("gold", GL, ["GOLD"])):
        for c in cols:
            for tau in ((0.05, 0.10) if samp != "gold" else (0.05,)):
                out["qr"][f"{samp}|{c}|{tau}"] = qr_intervals(D[c].values, D.SPY.values, tau,
                                                              np.random.default_rng([SEED, 200 + len(out["qr"])]),
                                                              a.B if samp == "etf" else max(500, a.B // 2))
                print("QR", samp, c, tau, out["qr"][f"{samp}|{c}|{tau}"])

    out["checks"] = CHECKS
    json.dump(out, open(os.path.join(a.outdir, "reviewer3_checks.json"), "w"), indent=1, default=float)

    # ---------------- markdown
    L = ["# Reviewer 3 checks (script 42)\n", "## A. Spanning tests\n",
         "| Sample | Test asset | α (ann., %) | p(α = 0) | δ | p(δ = 0) | p(δ = 0 given α = 0) | p joint (HK) | "
         "p(δ ≤ 0) long-only | Long-only GMV vol, bench / with X (%) | Reduction (pp) [95% CI] | X weight [95% CI] |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["spanning"].items():
        s, x = k.split("|")
        L.append(f"| {s} | {x} | {fmt(r['alpha_ann'] * 100, 1)} | {r['p_alpha']:.3f} | {fmt(r['delta'], 3)} | "
                 f"{r['p_delta']:.3f} | {r['p_delta_given_alpha0']:.3f} | {r['p_joint']:.3f} | {r['p_delta_pos']:.3f} | "
                 f"{r['gmv_vol_bench'] * 100:.2f} / {r['gmv_vol_ext'] * 100:.2f} | {r['gmv_vol_reduction'] * 100:.2f} "
                 f"[{r['boot_reduction_ci'][0] * 100:.2f}, {r['boot_reduction_ci'][1] * 100:.2f}] | "
                 f"{r['w_test_in_gmv'] * 100:.1f} [{r['boot_w_ci'][0] * 100:.1f}, {r['boot_w_ci'][1] * 100:.1f}] |")
    L += ["", "## B. Equivalence test of weak safe-haven status, λL(5%) (independence 0.05; margin ±0.05)\n",
          "| Sample | Asset | n | Joint tail days | λL(5%) | 95% CI | 90% CI | p (λL ≤ 0.05) | p (λL ≥ 0.10) | Positive dependence | "
          "Equivalent to independence | Smallest margin |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["tost"].items():
        s, c = k.split("|")
        L.append(f"| {s} | {c} | {r['n']} | {r['n_joint']} | {r['lam']:.3f} | [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | "
                 f"[{r['ci90'][0]:.3f}, {r['ci90'][1]:.3f}] | "
                 f"{r['p_positive']:.3f} | {r['p_equiv_upper']:.3f} | {'yes' if r['positive'] else 'no'} | "
                 f"{'yes' if r['equivalent'] else 'no'} | ±{r['smallest_margin']:.3f} |")
    L += ["", "Bitcoin minus gold, paired: " + "; ".join(
        f"{k}: {v['diff']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}], p(≤ 0) = {v['p_le0']:.3f}"
        for k, v in out["tost_diff"].items())]
    L += ["", "## C. SPY × VIX interaction for IBIT (ETF sample)\n",
          "| VIX | Sample | n | b | c [p] | slope at z = 0 / 1 / 2 |", "|---|---|---|---|---|---|"]
    for tag, r in out["interaction"].items():
        for nm in ("full", "no_april_episode", "no_april_month"):
            q = r[nm]
            L.append(f"| {tag} | {nm} | {q['n']} | {fmt(q['b'])} | {fmt(q['c'], 3)} [{q['p_c']:.3f}] | " +
                     " / ".join(f"{fmt(q['slope_at_z'][z]['slope'])} [{fmt(q['slope_at_z'][z]['ci'][0])}, "
                                f"{fmt(q['slope_at_z'][z]['ci'][1])}]" for z in ("0", "1", "2")) + " |")
        L.append(f"| {tag} | leave-one-episode-out range of c | | | "
                 f"{fmt(min(e['c'] for e in r['leave_one_out']), 3)} to {fmt(max(e['c'] for e in r['leave_one_out']), 3)}; "
                 f"p ≤ 0.05 in {sum(e['p_c'] <= 0.05 for e in r['leave_one_out'])} of {len(r['leave_one_out'])} | |")
    L += ["", f"April 2025 episode excluded: {ep['first']} .. {ep['last']} ± 3 trading days", "",
          "## D. Quantile-regression slope on SPY\n",
          "| Sample | Asset | τ | β | Kernel | Percentile | Bias-corrected | Basic | Share of draws below β |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, r in out["qr"].items():
        s, c, tau = k.split("|")
        L.append(f"| {s} | {c} | {tau} | {fmt(r['beta'])} | [{fmt(r['kernel'][0])}, {fmt(r['kernel'][1])}] | "
                 f"[{fmt(r['percentile'][0])}, {fmt(r['percentile'][1])}] | [{fmt(r['bc'][0])}, {fmt(r['bc'][1])}] | "
                 f"[{fmt(r['basic'][0])}, {fmt(r['basic'][1])}] | {r['share_below']:.3f} |")
    L += ["", *CHECKS]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "reviewer3_checks.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
