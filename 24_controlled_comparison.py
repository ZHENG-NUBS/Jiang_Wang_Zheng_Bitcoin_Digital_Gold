"""
24_controlled_comparison.py — Specification sensitivity of the controlled portfolio comparison

Two equally weighted portfolios cannot identify the causal "net effect" of replacing gold
with Bitcoin. The difference may also come from the arbitrary 1/3 weights, the particular
sample period, the rebalancing assumption, the chosen equity and bond ETFs, different
market microstructures, and the choice of risk measure. Please rename it a "controlled
portfolio comparison".

The paper's wording was already changed to controlled portfolio comparison in update 8
(§2.4 states explicitly that it does not identify a causal net effect). This script fills
in the parts of the reviewer's six dimensions that had not been systematically done before,
and assembles all dimensions into a "specification curve" (Simonsohn, Simmons & Nelson
2020), answering: how does the ES95 ratio between the IBIT portfolio and the GLD portfolio
change with each specification, by how much, and does the sign ever reverse?

  A. Weights: Bitcoin/gold sleeve weight w = 5%…50% (the rest split equally between SPY
     and TLT), the ES95, ES99, and volatility ratios for the two portfolios at the same
     weight; with GLD fixed at 1/3, the "risk-equivalent weight" w* that makes the IBIT
     portfolio's ES95 (or volatility) equal to it, with a block bootstrap interval.
  B. Rebalancing: daily, weekly, monthly, quarterly, semiannual, annual, buy-and-hold;
     for non-daily rules, the range over all starting phases; report the sleeve's average
     realized weight (weights drift with prices).
  C. Sample period: the main sample by year, first/second half, rolling 250-day windows;
     the long sample (BTC-USD spot minus a 0.25% annual fee, from 2014-09) full sample,
     pre/post ETF launch, by year, rolling 250-day windows.
  D. Equity and bond legs: 5 equity × 5 bond benchmarks (same convention as script 20's
     Section E2).
  E. Risk measures: annualized volatility, downside semi-deviation, VaR95, ES95, VaR99,
     ES99, maximum drawdown, CDaR95, 10-day ES95, normal and Student-t parametric ES95.
  F. Instruments, trading hours, forecast models: read the outputs of scripts 20 and 21
     (if present) and fold them into the specification curve.
  G. Multiverse: weights 4 × rebalancing 4 × benchmarks 25 × risk measures 5 = 2000 joint
     specifications, with the distribution.
  H. Specification-curve figure and weight-scan figure.

ES is historical simulation (the average loss of daily returns below the 5% quantile; same
convention as scripts 11 and 20; the baseline specification reproduces 1.633). Intervals
are stationary block bootstrap (expected block length 2, consistent with scripts 11, 19,
20), with the portfolio reconstructed on each bootstrap resample of asset returns.

Dependencies: numpy, pandas, scipy, matplotlib. Can be run standalone.
Usage:
    python 24_controlled_comparison.py
    python 24_controlled_comparison.py --data data/processed/returns__20260921.csv --raw data/raw --outdir out/24
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260929
B_MAIN, B_LONG = 2000, 1000
L_BOOT = 2
BTC_FEE = 0.0025
EQUITY = ["SPY", "VTI", "VT", "QQQ", "EFA"]
BOND = ["TLT", "IEF", "AGG", "SHY", "TIP"]
REBAL = [("Daily", 1), ("Weekly (5 days)", 5), ("Monthly (21 days)", 21), ("Quarterly (63 days)", 63), ("Semiannual (126 days)", 126),
         ("Annual (252 days)", 252), ("Buy-and-hold", None)]
LINES: list[str] = []
CHECKS: list[str] = []
SPEC: list[dict] = []


def w(s=""):
    LINES.append(s)
    print(s, flush=True)


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    CHECKS.append(f"[{tag}] {label} {detail}")
    w(f"   CHECK [{tag}] {label} {detail}")
    return bool(ok)


def ci(pt, lo, hi, d=2):
    f = lambda x: "—" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    return f"{f(pt)} [{f(lo)}, {f(hi)}]"


def pct(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(x) for x in np.percentile(a, [2.5, 97.5])] if len(a) else [np.nan, np.nan]


def stationary_boot(n, R, L, rng):
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    nb = rng.random((n, R)) < p
    jp = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(nb[t], jp[t], (idx[t - 1] + 1) % n)
    return idx


# ============================================================ Risk measures (x may be n or n×B)
def es(x, a=0.05):
    x = np.asarray(x, float)
    q = np.quantile(x, a, axis=0); m = x <= q
    return -(np.where(m, x, 0.0).sum(0) / m.sum(0))


def var(x, a=0.05):
    return -np.quantile(np.asarray(x, float), a, axis=0)


def vol(x):
    return np.asarray(x, float).std(0, ddof=1) * math.sqrt(252)


def semidev(x):
    x = np.asarray(x, float)
    return np.sqrt((np.minimum(x, 0) ** 2).mean(0)) * math.sqrt(252)


def drawdowns(x):
    v = np.cumprod(1 + np.asarray(x, float), axis=0)
    return 1 - v / np.maximum.accumulate(v, axis=0)


def mdd(x):
    return drawdowns(x).max(0)


def cdar(x, a=0.05):
    d = np.sort(drawdowns(x), axis=0)
    k = max(1, int(math.ceil(a * d.shape[0])))
    return d[-k:].mean(0)


def es10(x, a=0.05):
    lx = np.log1p(np.asarray(x, float))
    c = np.cumsum(lx, axis=0)
    r10 = np.expm1(c[9:] - np.concatenate([np.zeros((1,) + c.shape[1:]), c[:-10]], axis=0))
    return es(r10, a)


def es_norm(x, a=0.05):
    x = np.asarray(x, float)
    m, s = x.mean(0), x.std(0, ddof=1)
    return -(m - s * stats.norm.pdf(stats.norm.ppf(a)) / a)


def es_t(x, a=0.05):
    """ES of a unit-variance Student-t (ν estimated by maximum likelihood; only for a
    one-dimensional series)."""
    nu, loc, sc = stats.t.fit(np.asarray(x, float))
    q = stats.t.ppf(a, nu)
    e_std = (nu + q * q) / (nu - 1) * stats.t.pdf(q, nu) / a
    return -(loc - sc * e_std)


MEASURES = [("Annualized volatility", vol), ("Downside semi-deviation", semidev), ("VaR95", lambda x: var(x, .05)), ("ES95", lambda x: es(x, .05)),
            ("VaR99", lambda x: var(x, .01)), ("ES99", lambda x: es(x, .01)), ("Max drawdown", mdd), ("CDaR95", cdar),
            ("10-day ES95", es10)]


# ============================================================ Portfolio construction
def port(R3, w0, k=1, offset=0):
    """R3: (n, …, 3) = [sleeve asset, equity, bond]; target weights
    w0 = (w, (1−w)/2, (1−w)/2).
    k = 1 daily rebalancing; k = integer rebalance every k days (when t ≡ offset mod k);
    k = None buy-and-hold.
    Returns (portfolio returns, sleeve asset's realized weight)."""
    w0 = np.asarray(w0, float)
    if k == 1:
        return R3 @ w0, np.full(R3.shape[:-1], w0[0])
    n = R3.shape[0]
    W = np.broadcast_to(w0, R3.shape[1:]).copy()
    out = np.empty(R3.shape[:-1]); ws = np.empty(R3.shape[:-1])
    for t in range(n):
        if k is not None and t > 0 and (t - offset) % k == 0:
            W[...] = w0
        ws[t] = W[..., 0]
        g = W * (1 + R3[t]); s = g.sum(-1)
        out[t] = s - 1; W = g / s[..., None]
    return out, ws


def wv(wt):
    return np.array([wt, (1 - wt) / 2, (1 - wt) / 2])


def add_spec(dim, label, ratio, lo=np.nan, hi=np.nan, source="this script"):
    SPEC.append(dict(dim=dim, label=label, ratio=float(ratio), lo=float(lo), hi=float(hi), source=source))


# ============================================================ Data
def find_main():
    c = []
    for base in (HERE, os.getcwd()):
        c += [os.path.abspath(p) for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv"))
              if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p))]
    return sorted(set(c), key=os.path.basename)[-1] if c else None


def find_raw():
    for base in (HERE, os.getcwd()):
        d = os.path.join(base, "data", "raw")
        if glob.glob(os.path.join(d, "*__*S.csv")) or glob.glob(os.path.join(d, "*__*L.csv")):
            return d
    return None


def load_px(raw, tick, tag):
    fs = sorted(glob.glob(os.path.join(raw, f"{tick.replace('^', 'IDX_')}__*{tag}.csv")))
    if not fs:
        return None
    d = pd.read_csv(fs[-1], parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    return d["AdjClose"]


def load_bench(raw, dates):
    """S-snapshot equity/bond benchmarks, aligned to the SPY trading calendar, converted to
    daily returns, then truncated to the main-sample dates."""
    spy = load_px(raw, "SPY", "S")
    if spy is None:
        return None
    cal = spy.index
    out = {}
    for t in EQUITY + BOND:
        p = load_px(raw, t, "S")
        if p is not None:
            out[t] = p.reindex(cal).pct_change().reindex(dates)
    return pd.DataFrame(out)


def load_long(raw):
    S = {t: load_px(raw, f, "L") for t, f in (("BTC", "BTC-USD"), ("GLD", "GLD"), ("SPY", "SPY"), ("TLT", "TLT"))}
    if any(v is None for v in S.values()):
        return None
    cal = S["SPY"].index
    P = pd.DataFrame({k: v.reindex(cal) for k, v in S.items()})
    P = P[P.index >= S["BTC"].index.min()]
    R = P.pct_change().dropna()
    R = R[R.index <= pd.Timestamp("2026-06-15")]
    R["BTC"] = R["BTC"] - BTC_FEE / 252
    return R


def find_json(name):
    for base in (HERE, os.getcwd(), os.path.join(HERE, "out", "20"), os.path.join(HERE, "out", "21")):
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    fs = glob.glob(os.path.join(HERE, "**", name), recursive=True)
    return fs[0] if fs else None


# ============================================================ Sections
def ratio_boot(R_i, R_g, idx, fn=lambda x: es(x, .05)):
    """R_i, R_g: n-dimensional portfolio returns; idx: n×B bootstrap indices."""
    return float(fn(R_i) / fn(R_g)), pct(fn(R_i[idx]) / fn(R_g[idx]))


def sec_weight(label, X_i, X_g, EQ, BD, idx, grid_fine):
    """A. Weight scan and risk-equivalent weight. X_i/X_g are sleeve-asset returns (n);
    EQ/BD are equity/bond."""
    R3i = np.stack([X_i, EQ, BD], -1); R3g = np.stack([X_g, EQ, BD], -1)
    rows = []; res = {}
    for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1 / 3, 0.40, 0.45, 0.50):
        pi, pg = R3i @ wv(wt), R3g @ wv(wt)
        r95, c95 = ratio_boot(pi, pg, idx)
        r99, c99 = ratio_boot(pi, pg, idx, lambda x: es(x, .01))
        rv = float(vol(pi) / vol(pg))
        shape = r95 / rv
        res[f"{wt:.3f}"] = dict(es95=r95, es95_ci=c95, es99=r99, es99_ci=c99, vol=rv, shape=shape,
                               es95_i=float(es(pi) * 100), es95_g=float(es(pg) * 100))
        rows.append([f"{wt * 100:.1f}", f"{es(pi) * 100:.2f}", f"{es(pg) * 100:.2f}", ci(r95, *c95), ci(r99, *c99),
                     f"{rv:.2f}", f"{shape:.2f}"])
        if label == "main sample":
            add_spec("A Weight (both sleeves)", f"w = {wt * 100:.0f}%", r95, *c95)
    w(f"### A1. {label}: same sleeve weight w for both portfolios (the rest split equally between SPY and TLT, daily rebalancing)\n")
    w(md(rows, ["w %", "IBIT/BTC portfolio ES95 %", "GLD portfolio ES95 %", "ES95 ratio [CI]", "ES99 ratio [CI]", "Volatility ratio",
                "Shape term = ES ratio / volatility ratio"]))
    # Risk-equivalent weight: GLD fixed at 1/3
    target_es = es(R3g @ wv(1 / 3)); target_vol = vol(R3g @ wv(1 / 3))
    Ri_b = R3i[idx]; Rg_b = R3g[idx]                      # (n, B, 3)
    tes_b = es(Rg_b @ wv(1 / 3)); tvol_b = vol(Rg_b @ wv(1 / 3))
    es_grid = np.array([es(R3i @ wv(g)) for g in grid_fine])
    vol_grid = np.array([vol(R3i @ wv(g)) for g in grid_fine])
    es_grid_b = np.array([es(Ri_b @ wv(g)) for g in grid_fine])      # (G, B)
    vol_grid_b = np.array([vol(Ri_b @ wv(g)) for g in grid_fine])

    def cross(curve, target):
        i0 = int(np.argmin(curve))
        for j in range(i0, len(curve) - 1):
            if curve[j] <= target <= curve[j + 1]:
                return grid_fine[j] + (target - curve[j]) / (curve[j + 1] - curve[j]) * (grid_fine[j + 1] - grid_fine[j])
        return np.nan
    w_es = cross(es_grid, target_es); w_vol = cross(vol_grid, target_vol)
    w_es_b = [cross(es_grid_b[:, b], tes_b[b]) for b in range(idx.shape[1])]
    w_vol_b = [cross(vol_grid_b[:, b], tvol_b[b]) for b in range(idx.shape[1])]
    sd_i, sd_g = X_i.std(ddof=1), X_g.std(ddof=1)
    sh_es = float(np.mean(np.isfinite(w_es_b))); sh_vol = float(np.mean(np.isfinite(w_vol_b)))
    res["risk_equiv"] = dict(w_es=float(w_es), w_es_ci=pct(w_es_b) if sh_es > 0.5 else [np.nan, np.nan], w_es_exist=sh_es,
                             w_vol=float(w_vol), w_vol_ci=pct(w_vol_b) if sh_vol > 0.5 else [np.nan, np.nan], w_vol_exist=sh_vol,
                             w_volmatch=float(sd_g / sd_i / 3), min_es_w=float(grid_fine[int(np.argmin(es_grid))]),
                             es_at_min=float(es_grid.min()), target_es=float(target_es),
                             vol_at_min=float(vol_grid.min()), target_vol=float(target_vol))
    q = res["risk_equiv"]
    w(f"### A2. {label}: risk-equivalent weight (GLD portfolio fixed at 1/3)\n")

    def say(nm, wv_, civ, sh, at_min, tgt, unit):
        if np.isfinite(wv_):
            return (f"- The sleeve weight that makes the portfolio's {nm} equal to the GLD portfolio's ({tgt * 100:.2f}%) = **{wv_ * 100:.1f}%** "
                    f"[{civ[0] * 100:.1f}, {civ[1] * 100:.1f}] (exists in {sh * 100:.0f}% of resamples);")
        return (f"- There is **no** sleeve weight that makes the portfolio's {nm} equal to the GLD portfolio's: at zero sleeve weight (only SPY and TLT, half each) the {nm} is already "
                f"{at_min * 100:.2f}%, above the GLD portfolio's {tgt * 100:.2f}%; that is, the gold sleeve itself lowers the portfolio's {nm} "
                f"(a crossing exists in {sh * 100:.0f}% of resamples);")
    w(say("ES95 ", q["w_es"], q["w_es_ci"], sh_es, q["es_at_min"], target_es, "%"))
    w(say("volatility", q["w_vol"], q["w_vol_ci"], sh_vol, q["vol_at_min"], target_vol, "%"))
    w(f"- The paper's volatility-matched weight (1/3 × σ_GLD/σ_sleeve, matching only the sleeve's own volatility) = {q['w_volmatch'] * 100:.1f}%; "
      f"the sleeve weight that minimizes ES95 = {q['min_es_w'] * 100:.1f}%.\n")
    return res, dict(grid=list(map(float, grid_fine)), es=list(map(float, es_grid)),
                     es_lo=list(map(float, np.percentile(es_grid_b, 2.5, axis=1))),
                     es_hi=list(map(float, np.percentile(es_grid_b, 97.5, axis=1))),
                     target=float(target_es), target_lo=float(np.percentile(tes_b, 2.5)),
                     target_hi=float(np.percentile(tes_b, 97.5)))


def sec_rebal(label, X_i, X_g, EQ, BD, idx, dim=True, boot=True):
    R3i = np.stack([X_i, EQ, BD], -1); R3g = np.stack([X_g, EQ, BD], -1)
    Ri_b, Rg_b = R3i[idx], R3g[idx]
    rows = []; res = {}
    for nm, k in REBAL:
        pi, wi = port(R3i, wv(1 / 3), k); pg, wg = port(R3g, wv(1 / 3), k)
        r = float(es(pi) / es(pg))
        if boot:
            bi, _ = port(Ri_b, wv(1 / 3), k); bg, _ = port(Rg_b, wv(1 / 3), k)
            c = pct(es(bi) / es(bg))
        else:
            c = [np.nan, np.nan]
        rng_ = [r, r]
        if k not in (1, None):
            rs = []
            for off in range(k):
                a_, _ = port(R3i, wv(1 / 3), k, off); b_, _ = port(R3g, wv(1 / 3), k, off)
                rs.append(float(es(a_) / es(b_)))
            rng_ = [min(rs), max(rs)]
        res[nm] = dict(ratio=r, ci=c, phase_range=rng_, w_i=float(wi.mean()), w_g=float(wg.mean()),
                       w_i_end=float(wi[-1]), w_g_end=float(wg[-1]), vol_ratio=float(vol(pi) / vol(pg)))
        rows.append([nm, ci(r, *c), f"{rng_[0]:.2f}–{rng_[1]:.2f}" if k not in (1, None) else "—",
                     f"{wi.mean() * 100:.1f} / {wg.mean() * 100:.1f}", f"{wi[-1] * 100:.1f} / {wg[-1] * 100:.1f}",
                     f"{vol(pi) / vol(pg):.2f}"])
        if dim:
            add_spec("B Rebalancing", nm, r, *c)
    w(f"### B. {label}: rebalancing rule (target weight 1/3, starting phase 0; for non-daily rules the range over all starting phases is also reported)\n")
    w(md(rows, ["Rule", "ES95 ratio [CI]", "Starting-phase range", "Average realized sleeve weight % (Bitcoin / gold)", "End weight %", "Volatility ratio"]))
    return res


def rolling_ratio(pi, pg, win=250):
    return np.array([es(pi[s:s + win]) / es(pg[s:s + win]) for s in range(0, len(pi) - win + 1)])


def sec_period(label, dates, pi, pg, idx_fn, cuts, dim_name):
    rows = []; res = {}
    for nm, m in cuts:
        m = np.asarray(m)
        if m.sum() < 100:
            continue
        a_, b_ = pi[m], pg[m]
        r = float(es(a_) / es(b_))
        ii = idx_fn(int(m.sum()))
        c = pct(es(a_[ii]) / es(b_[ii]))
        res[nm] = dict(n=int(m.sum()), ratio=r, ci=c, vol_ratio=float(vol(a_) / vol(b_)))
        rows.append([nm, int(m.sum()), ci(r, *c), f"{vol(a_) / vol(b_):.2f}", f"{es(a_) * 100:.2f} / {es(b_) * 100:.2f}"])
        add_spec(dim_name, nm, r, *c)
    rr = rolling_ratio(pi, pg)
    res["rolling"] = dict(n=len(rr), min=float(rr.min()), p5=float(np.percentile(rr, 5)), median=float(np.median(rr)),
                          p95=float(np.percentile(rr, 95)), max=float(rr.max()), share_gt1=float((rr > 1).mean()),
                          argmin=str(dates[int(np.argmin(rr)) + 249].date()))
    w(f"### C. {label}: sample period\n")
    w(md(rows, ["Sub-period", "Days", "ES95 ratio [CI]", "Volatility ratio", "ES95 % (Bitcoin / gold portfolio)"]))
    q = res["rolling"]
    w(f"Rolling 250-day windows ({q['n']}): the ES95 ratio has minimum {q['min']:.2f} (window ending {q['argmin']}), 5th percentile {q['p5']:.2f}, "
      f"median {q['median']:.2f}, 95th percentile {q['p95']:.2f}, maximum {q['max']:.2f}; the share of windows with ratio > 1 is {q['share_gt1'] * 100:.0f}%.\n")
    return res, rr


# ============================================================ Self-test
def selftest():
    ok = True
    g = np.random.default_rng(0)
    x = g.standard_normal(400000)
    ok &= check("ES95 matches the normal theoretical value (2.063σ)", abs(es(x) - 2.0627) < 0.02, f"{es(x):.4f}")
    ok &= check("Normal parametric ES95 formula", abs(es_norm(x) - 2.0627) < 0.01, f"{es_norm(x):.4f}")
    xt = stats.t.rvs(5, size=200000, random_state=1) * 0.01
    th = 0.01 * (5 + stats.t.ppf(.05, 5) ** 2) / 4 * stats.t.pdf(stats.t.ppf(.05, 5), 5) / .05
    ok &= check("Student-t parametric ES95 (ν = 5)", abs(es_t(xt) / th - 1) < 0.03, f"{es_t(xt):.5f} vs {th:.5f}")
    R3 = g.normal(0.0003, 0.01, (300, 3))
    a1, _ = port(R3, wv(1 / 3), 1); a2, _ = port(R3, wv(1 / 3), 1000, 0)
    ok &= check("Rebalancing: interval longer than the sample = buy-and-hold", np.allclose(a2, port(R3, wv(1 / 3), None)[0]))
    V = np.cumprod(1 + port(R3, wv(1 / 3), None)[0]); V2 = (np.cumprod(1 + R3, 0) @ wv(1 / 3))
    ok &= check("Buy-and-hold = sum of asset cumulative net values weighted by initial weights", np.allclose(V, V2), f"max diff {np.abs(V - V2).max():.1e}")
    a3, _ = port(R3, wv(1 / 3), 5, 0)
    ok &= check("Vectorized daily rebalancing = day-by-day loop version", np.allclose(a1, port(R3[:, None, :], wv(1 / 3), 1)[0][:, 0]))
    ok &= check("10-day ES: with constant daily returns = 10-day compounded loss", abs(es10(np.full(50, -0.01)) - (1 - 0.99 ** 10)) < 1e-12)
    d = drawdowns(np.array([0.1, -0.5, 0.2]))
    ok &= check("Drawdown series", np.allclose(d, [0, 0.5, 0.4]) and abs(mdd(np.array([0.1, -0.5, 0.2])) - 0.5) < 1e-12)
    return ok


# ============================================================ Main program
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--outdir", default="out/24")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    a.data = a.data or find_main(); a.raw = a.raw or find_raw()
    if a.data is None:
        sys.exit("Cannot find data/processed/returns__YYYYMMDD.csv; please specify with --data.")
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(HERE, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    global B_MAIN, B_LONG
    if a.quick:
        B_MAIN, B_LONG = 300, 200
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT = {}

    w("# Specification sensitivity of the controlled portfolio comparison (script 24)\n")
    w(f"Seed {SEED}; stationary block bootstrap B = {B_MAIN} (main sample) / {B_LONG} (long sample), expected block length {L_BOOT}; ES is historical simulation.\n")
    w("Baseline specification: IBIT portfolio = 1/3 IBIT + 1/3 SPY + 1/3 TLT, GLD portfolio = 1/3 GLD + 1/3 SPY + 1/3 TLT, daily rebalancing, "
      "main sample 607 days, risk measure ES95. Each section below changes one dimension at a time (Section G changes them jointly).\n")
    ok_self = selftest()

    d = pd.read_csv(a.data, parse_dates=["Date"]).set_index("Date").sort_index()
    dates = d.index; n = len(d)
    I, G, SPY, TLT = (d[c].values for c in ("IBIT", "GLD", "SPY", "TLT"))
    idx = stationary_boot(n, B_MAIN, L_BOOT, rng)
    pi0, pg0 = (I + SPY + TLT) / 3, (G + SPY + TLT) / 3
    base = float(es(pi0) / es(pg0))
    check("Baseline specification's ES95 ratio reproduces the paper's 1.63 (script 20's 1.633)", abs(base - 1.6335) < 0.001, f"{base:.4f}")
    bc = pct(es(pi0[idx]) / es(pg0[idx]))
    add_spec("Baseline", "Baseline specification", base, *bc)
    w(f"\nBaseline specification: ES95 ratio {ci(base, *bc, 3)}; IBIT portfolio ES95 {es(pi0) * 100:.2f}%, GLD portfolio ES95 {es(pg0) * 100:.2f}%.\n")

    # ---------------- A Weights
    w("\n## A. Weights\n")
    grid = np.round(np.arange(0.0, 0.5001, 0.005), 4)
    OUT["A_main"], CURVE = sec_weight("main sample", I, G, SPY, TLT, idx, grid)
    check("Volatility-matched weight reproduces the paper's 14.5%", abs(OUT["A_main"]["risk_equiv"]["w_volmatch"] - 0.145) < 0.0015,
          f"{OUT['A_main']['risk_equiv']['w_volmatch'] * 100:.2f}%")

    # ---------------- B Rebalancing
    w("\n## B. Rebalancing\n")
    OUT["B_main"] = sec_rebal("main sample", I, G, SPY, TLT, idx)

    # ---------------- C Sample period (main sample)
    w("\n## C. Sample period\n")
    yr = dates.year
    half = np.arange(n) < n // 2

    def idx_fn_main(m):
        return stationary_boot(m, B_MAIN, L_BOOT, np.random.default_rng(SEED + m))
    cuts = [("2024", yr == 2024), ("2025", yr == 2025), ("2026 H1", yr == 2026),
            (f"first half ({dates[0].date()} → {dates[n // 2 - 1].date()})", half),
            (f"second half ({dates[n // 2].date()} → {dates[-1].date()})", ~half)]
    OUT["C_main"], RR_main = sec_period("main sample", dates, pi0, pg0, idx_fn_main, cuts, "C Sample period (IBIT, main sample)")

    # ---------------- Long sample
    RL = load_long(a.raw) if a.raw else None
    if RL is not None:
        dl = RL.index; nl = len(RL)
        B_, GL, SL, TL = (RL[c].values for c in ("BTC", "GLD", "SPY", "TLT"))
        pil, pgl = (B_ + SL + TL) / 3, (GL + SL + TL) / 3

        def idx_fn_long(m):
            return stationary_boot(m, B_LONG, L_BOOT, np.random.default_rng(SEED + 7 * m))
        yl = dl.year
        cutsl = [(f"full sample {dl[0].date()} → {dl[-1].date()}", np.ones(nl, bool)),
                 ("pre-spot-ETF (→ 2023-12)", dl < pd.Timestamp("2024-01-01")),
                 ("post-spot-ETF (2024-01 →)", dl >= pd.Timestamp("2024-01-01"))]
        cutsl += [(str(y), yl == y) for y in range(2015, 2026)] + [("2026 H1", yl == 2026)]
        w("Long sample: BTC-USD (00:00 UTC close, re-indexed to the SPY trading calendar, daily returns minus 0.25%/252 to mimic a spot ETF's annual fee). Before 2024 no spot ETF "
          "existed, so this is a hypothetical portfolio; the closing time differs from the US close, which pushes down Bitcoin's same-day correlation with stocks.\n")
        OUT["C_long"], RR_long = sec_period("long sample (BTC-USD)", dl, pil, pgl, idx_fn_long, cutsl, "C Sample period (BTC-USD, long sample)")
        idxl = idx_fn_long(nl)
        w("\n### Long-sample weights and rebalancing\n")
        OUT["A_long"], CURVE_L = sec_weight("long sample", B_, GL, SL, TL, idxl, grid)
        OUT["B_long"] = sec_rebal("long sample", B_, GL, SL, TL, idxl, dim=False, boot=False)
        w("For the long sample we do not report bootstrap intervals for the rebalancing rules: over 12 years Bitcoin's cumulative gain makes the realized weight under non-daily rules highly dependent on the specific price path, "
          "so resampling that shuffles daily return order no longer represents the same holding. The starting-phase range is used instead.\n")
    else:
        w("(No long-sample snapshot data/raw/*__*L.csv found; skipping the long-sample part)")

    # ---------------- D Equity and bond legs
    w("\n## D. Equity and bond legs (5 × 5)\n")
    BM = load_bench(a.raw, dates) if a.raw else None
    E2 = {}
    if BM is not None and BM.notna().all().all() and len(BM.columns) == 10:
        rows = []
        for eq in EQUITY:
            row = [eq]
            for bd in BOND:
                x1 = (I + BM[eq].values + BM[bd].values) / 3; x2 = (G + BM[eq].values + BM[bd].values) / 3
                r = float(es(x1) / es(x2)); c = pct(es(x1[idx]) / es(x2[idx]))
                E2[f"{eq}|{bd}"] = dict(ratio=r, ci=c)
                row.append(ci(r, *c))
                if (eq, bd) != ("SPY", "TLT"):
                    add_spec("D Equity/bond legs", f"{eq} + {bd}", r, *c)
            rows.append(row)
        w(md(rows, ["Equity \\ bond"] + BOND))
        rr = [v["ratio"] for v in E2.values()]
        w(f"Across the 25 combinations the ES95 ratio is {min(rr):.2f}–{max(rr):.2f}, with the smallest CI lower bound {min(v['ci'][0] for v in E2.values()):.2f}.\n")
        check("The 5 × 5 benchmark grid reproduces script 20 (1.44–1.71)", abs(min(rr) - 1.44) < 0.006 and abs(max(rr) - 1.71) < 0.006,
              f"{min(rr):.3f}–{max(rr):.3f}")
        check("The SPY + TLT cell = the baseline specification (S-snapshot vs main-snapshot return difference ≤ 1e-6)", abs(E2["SPY|TLT"]["ratio"] - base) < 1e-3, f"{E2['SPY|TLT']['ratio']:.4f}")
    else:
        w("(No complete S-snapshot equity/bond benchmarks found; skipping)")
    OUT["D"] = E2

    # ---------------- E Risk measures
    w("\n## E. Risk measures (baseline specification)\n")
    rows = []; EM = {}
    for nm, fn in MEASURES:
        r = float(fn(pi0) / fn(pg0)); c = pct(fn(pi0[idx]) / fn(pg0[idx]))
        EM[nm] = dict(ratio=r, ci=c, ibit=float(fn(pi0)), gld=float(fn(pg0)))
        rows.append([nm, f"{fn(pi0) * 100:.2f} / {fn(pg0) * 100:.2f}", ci(r, *c)])
        if nm != "ES95":
            add_spec("E Risk measure", nm, r, *c)
    for nm, fn in (("Normal parametric ES95", es_norm), ("Student-t parametric ES95", es_t)):
        r = float(fn(pi0) / fn(pg0))
        bs = [fn(pi0[idx[:, b]]) / fn(pg0[idx[:, b]]) for b in range(min(idx.shape[1], 500 if fn is es_t else idx.shape[1]))]
        c = pct(bs)
        EM[nm] = dict(ratio=r, ci=c, ibit=float(fn(pi0)), gld=float(fn(pg0)))
        rows.append([nm, f"{fn(pi0) * 100:.2f} / {fn(pg0) * 100:.2f}", ci(r, *c)])
        add_spec("E Risk measure", nm, r, *c)
    w(md(rows, ["Measure", "IBIT portfolio / GLD portfolio (%)", "Ratio [CI]"]))
    w("Maximum drawdown and CDaR95 are path-dependent; over 607 days there are only a few drawdowns, so the intervals should be read by direction rather than magnitude; the t-parametric ES interval uses the first 500 resamples.\n")
    OUT["E"] = EM

    # ---------------- F Instruments, trading hours, and models (read scripts 20, 21)
    w("\n## F. Instruments, trading hours, and forecast models (reading scripts 20, 21 outputs)\n")
    p20, p21 = find_json("scope_expansion.json"), find_json("forecast_design.json")
    F = {}
    if p20:
        J = json.load(open(p20, encoding="utf8"))
        vals = J["B_btc"]["ratio_ES95"]
        for nm, v, c in zip(("BTC-USD (00:00 UTC)", "Bitcoin 16:00 ET (Coinbase)"), vals["values"][:2], vals["ci"][:2]):
            add_spec("F Pricing time", nm, v, *c, source="script 20")
        nmax = max(v["n"] for v in J["E1"].values())
        pairs = {k: v for k, v in J["E1"].items() if v["n"] == nmax and k != "IBIT|GLD"}
        for k, v in pairs.items():
            add_spec("F Bitcoin/gold instruments", k.replace("|", " × "), v["ratio"], *v["ci"], source="script 20")
        F["timing"] = vals; F["n_pairs"] = len(pairs)
        rr = [v["ratio"] for v in pairs.values()]
        w(f"- Trading hours: BTC-USD 00:00 UTC {vals['values'][0]:.2f}, Bitcoin 16:00 ET {vals['values'][1]:.2f}, IBIT {vals['values'][2]:.2f}"
          f" (neither the time effect nor the wrapper effect is significant).")
        w(f"- Instruments: {len(pairs)} full-sample Bitcoin × gold instrument pairs (excluding IBIT × GLD), ES95 ratio {min(rr):.2f}–{max(rr):.2f}.")
    else:
        w("- scope_expansion.json (script 20) not found; skipping instruments and trading hours.")
    if p21:
        J = json.load(open(p21, encoding="utf8"))
        for k, v in J["oos_ratio"].items():
            if k == "realized_HS":
                continue
            add_spec("F Forecast model (out-of-sample)", k, v["ratio"], *v["ci"], source="script 21")
        F["oos"] = J["oos_ratio"]
        w("- Out-of-sample forecasts (357 days, six models): the ratio of average ES95 forecasts " +
          ", ".join(f"{k} {v['ratio']:.2f}" for k, v in J["oos_ratio"].items() if k != "realized_HS") + ".\n")
    else:
        w("- forecast_design.json (script 21) not found; skipping forecast models.\n")
    OUT["F"] = F

    # ---------------- G Multiverse
    w("\n## G. Multiverse: jointly varying weights × rebalancing × benchmarks × risk measures\n")
    MV = []
    if E2:
        meas5 = [("VaR95", lambda x: var(x, .05)), ("ES95", lambda x: es(x, .05)), ("ES99", lambda x: es(x, .01)),
                 ("Annualized volatility", vol), ("Max drawdown", mdd)]
        for wt in (0.10, 0.20, 1 / 3, 0.50):
            for rn, k in (("Daily", 1), ("Monthly", 21), ("Quarterly", 63), ("Buy-and-hold", None)):
                for eq in EQUITY:
                    for bd in BOND:
                        pi_, _ = port(np.stack([I, BM[eq].values, BM[bd].values], -1), wv(wt), k)
                        pg_, _ = port(np.stack([G, BM[eq].values, BM[bd].values], -1), wv(wt), k)
                        for mn, fn in meas5:
                            MV.append(dict(w=wt, rebal=rn, eq=eq, bd=bd, measure=mn, ratio=float(fn(pi_) / fn(pg_))))
        mv = pd.DataFrame(MV)
        rows = []
        for key in ("w", "rebal", "measure", "eq", "bd"):
            for v, gdf in mv.groupby(key, sort=False):
                rows.append([key, f"{v * 100:.0f}%" if key == "w" else v, len(gdf), f"{gdf.ratio.min():.2f}",
                             f"{gdf.ratio.median():.2f}", f"{gdf.ratio.max():.2f}", f"{(gdf.ratio > 1).mean() * 100:.0f}%"])
        w(f"{len(mv)} joint specifications (point estimates): {mv.ratio.gt(1).mean() * 100:.1f}% have ratio > 1; minimum {mv.ratio.min():.2f}, "
          f"median {mv.ratio.median():.2f}, maximum {mv.ratio.max():.2f}.\n")
        w(md(rows, ["Dimension", "Value", "Specifications", "Min", "Median", "Max", "Ratio > 1"]))
        low = mv[mv.ratio <= 1]
        if len(low):
            w("Joint specifications with ratio ≤ 1: " + "; ".join(f"w {r.w * 100:.0f}%, {r.rebal}, {r.eq}+{r.bd}, {r.measure} {r.ratio:.2f}"
                                          for r in low.itertuples()) + "\n")
        # Variance decomposition: main effects of each dimension on log ratio
        y = np.log(mv.ratio.values); tot = ((y - y.mean()) ** 2).sum()
        share = {}
        for key in ("w", "rebal", "measure", "eq", "bd"):
            gm = mv.assign(y=y).groupby(key)["y"].transform("mean").values
            share[key] = float(((gm - y.mean()) ** 2).sum() / tot)
        nmK = {"w": "Weight", "rebal": "Rebalancing", "measure": "Risk measure", "eq": "Equity leg", "bd": "Bond leg"}
        w("Share of the variance of log(ratio) explained by each dimension's main effect: " + "; ".join(f"{nmK[k]} {v * 100:.0f}%" for k, v in share.items()) +
          f"; interactions and residual {max(0.0, 1 - sum(share.values())) * 100:.0f}%.\n")
        OUT["G"] = dict(n=len(mv), share_gt1=float((mv.ratio > 1).mean()), min=float(mv.ratio.min()),
                        median=float(mv.ratio.median()), max=float(mv.ratio.max()), var_share=share)
        mv.to_csv(os.path.join(a.outdir, "multiverse.csv"), index=False)

    # ---------------- H Specification-curve summary
    w("\n## H. Specification-curve summary (one dimension varied at a time; baseline 1.63)\n")
    sp = pd.DataFrame(SPEC)
    rows = []
    for dim, gdf in sp.groupby("dim", sort=False):
        lo_ok = gdf.lo.notna()
        rows.append([dim, len(gdf), f"{gdf.ratio.min():.2f}", f"{gdf.ratio.median():.2f}", f"{gdf.ratio.max():.2f}",
                     f"{(gdf.lo[lo_ok] > 1).sum()}/{lo_ok.sum()}", f"{(gdf.hi[lo_ok] < 1).sum()}/{lo_ok.sum()}",
                     "; ".join(sorted(set(gdf.source)))])
    w(md(rows, ["Dimension", "Specifications", "Min", "Median", "Max", "CI entirely above 1", "CI entirely below 1", "Source"]))
    w(f"{len(sp)} specifications in total; specifications with ratio < 1: {', '.join(sp[sp.ratio < 1].label) if (sp.ratio < 1).any() else 'none'}.\n")
    sp.to_csv(os.path.join(a.outdir, "specification_curve.csv"), index=False)
    OUT["spec_summary"] = rows

    # ---------------- Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e6e5e1", "#d9d8d4"
        C_IBIT, C_GLD = "#2a78d6", "#eb6834"
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                             "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                             "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                             "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 7})
        EN = {"Baseline": "Baseline", "A Weight (both sleeves)": "Weight (both sleeves)", "B Rebalancing": "Rebalancing",
              "C Sample period (IBIT, main sample)": "Sub-period (IBIT)", "C Sample period (BTC-USD, long sample)": "Period (BTC-USD, 2014–26)",
              "D Equity/bond legs": "Equity/bond legs", "E Risk measure": "Risk measure", "F Pricing time": "Pricing time",
              "F Bitcoin/gold instruments": "Bitcoin/gold instruments", "F Forecast model (out-of-sample)": "Forecast model (OOS)"}
        dims = list(dict.fromkeys(sp.dim))
        cols = ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e"]
        cmap = {dm: cols[i % len(cols)] for i, dm in enumerate(dims)}
        s2 = sp.sort_values("ratio").reset_index(drop=True)
        fig, axs = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
        ax = axs[0]
        x = np.arange(len(s2))
        for dm in dims:
            m = (s2.dim == dm).values
            ax.vlines(x[m], s2.lo[m], s2.hi[m], color=cmap[dm], lw=0.6, alpha=0.5)
            ax.plot(x[m], s2.ratio[m], "o", ms=2.6, color=cmap[dm], label=EN.get(dm, dm))
        ax.axhline(1, color=INK2, lw=0.8); ax.axhline(base, color=INK2, lw=0.6, ls=(0, (3, 2)))
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.8, 1, 1.25, 1.5, 2, 3, 4, 6]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel("ES95 ratio, Bitcoin / gold portfolio (log scale)")
        ax.set_title("a. Specification curve (one dimension varied at a time; 95% bootstrap intervals)", loc="left", fontsize=8)
        ax.legend(loc="upper left", ncol=2, fontsize=6.5)
        ax = axs[1]
        for j, dm in enumerate(dims):
            m = (s2.dim == dm).values
            ax.plot(x[m], np.full(m.sum(), j), "|", ms=6, mew=1.0, color=cmap[dm])
        ax.set_yticks(range(len(dims))); ax.set_yticklabels([EN.get(dm, dm) for dm in dims], fontsize=6.5)
        ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Specifications, sorted by ratio")
        ax.set_title("b. Dimension varied", loc="left", fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.outdir, f"fig_specification_curve.{ext}"), dpi=300)
        plt.close(fig)
        # Weight-scan figure
        panels = [("Main sample: IBIT vs GLD", CURVE, OUT["A_main"]["risk_equiv"])]
        if RL is not None:
            panels.append(("Long sample: BTC-USD vs GLD", CURVE_L, OUT["A_long"]["risk_equiv"]))
        fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.8), squeeze=False)
        for ax, (tt, cv, re_) in zip(axs[0], panels):
            gx = np.array(cv["grid"]) * 100
            ax.fill_between(gx, np.array(cv["es_lo"]) * 100, np.array(cv["es_hi"]) * 100, color=C_IBIT, alpha=0.15, lw=0)
            ax.plot(gx, np.array(cv["es"]) * 100, color=C_IBIT, lw=1.3, label="Bitcoin portfolio ES95")
            ax.axhspan(cv["target_lo"] * 100, cv["target_hi"] * 100, color=C_GLD, alpha=0.12, lw=0)
            ax.axhline(cv["target"] * 100, color=C_GLD, lw=1.1, label="Gold portfolio ES95 (w = 1/3)")
            ax.axvline(100 / 3, color=INK2, lw=0.7, ls=(0, (3, 2)))
            if np.isfinite(re_["w_es"]):
                ax.plot([re_["w_es"] * 100], [cv["target"] * 100], "o", color=INK, ms=4)
                ax.annotate(f"w* = {re_['w_es'] * 100:.1f}%", (re_["w_es"] * 100, cv["target"] * 100),
                            textcoords="offset points", xytext=(4, -12), fontsize=7)
            ax.set_xlabel("Weight of Bitcoin sleeve (%)"); ax.set_ylabel("Daily ES95 (%)")
            ax.set_title(tt, loc="left", fontsize=8)
        axs[0][0].legend(loc="upper left", fontsize=6.5)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(a.outdir, f"fig_weight_scan.{ext}"), dpi=300)
        plt.close(fig)
        w("Figures saved: fig_specification_curve.png/.pdf, fig_weight_scan.png/.pdf")
    except Exception as e:                                     # noqa: BLE001
        w(f"(plotting failed: {e})")

    # ---------------- Outputs
    w("\n## Self-test summary\n")
    for c in CHECKS:
        w("  " + c)
    nf = sum(c.startswith("[FAIL]") for c in CHECKS)
    w(f"\n{len(CHECKS)} checks, {nf} failures. Elapsed {time.time() - t0:.0f} seconds.")
    with open(os.path.join(a.outdir, "controlled_comparison.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.ndarray):
            return clean(x.tolist())
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        return x
    OUT["baseline"] = dict(ratio=base, ci=bc)
    with open(os.path.join(a.outdir, "controlled_comparison.json"), "w", encoding="utf8") as fh:
        json.dump(clean(OUT), fh, indent=1, ensure_ascii=False)
    return 0 if (nf == 0 and ok_self) else 1


if __name__ == "__main__":
    sys.exit(main())