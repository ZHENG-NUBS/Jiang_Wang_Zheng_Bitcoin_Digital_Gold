"""
19_sample_size_robustness.py — Interval estimation and regime sensitivity under a short sample

Weaken the conclusions; report confidence intervals for 99% VaR, ES, tail dependence, and
crisis-period results; the regime split must not rely only on the two fixed cut points
15/25, but should vary the thresholds, split by quantiles, and use a continuous VIX
specification.

This script fills in the parts previously missing (the rest have already been done in
11 / 03 / 04):

  A. Regime-definition sensitivity: two-state cut points 17/20/22/25/30, three-state
     15/25 and 20/30, tertiles/quartiles/quintiles, top 10% VIX; each with both
     contemporaneous and lagged VIX. For each definition, report the block-bootstrap
     interval of the high- and low-regime ES95 multiples, and the test of the
     "high − low" difference. On each resample the regimes are [re-drawn] (quantile
     thresholds are also recomputed).
  B. Continuous specification: 5% quantile regression of each portfolio's daily return on
     lagged VIX, giving the conditional VaR95 as a continuous function of VIX; report the
     ratio of the two portfolios' conditional VaR95 at VIX = 15/20/25/30 and R(30) − R(15).
  C. Block-bootstrap intervals of full-sample VaR95 / VaR99 (including the ratio of the
     two portfolios).
  D. Intervals for each regime's ρ and β of IBIT/GLD with SPY, and the high − low
     difference test.
  E. Intervals for the "beta-adjusted excess return" on the worst 5% of SPY trading days.
  F. Cross-quantilogram (0.05, 0.05) and the interval of the IBIT − GLD difference.
  G. Parameter uncertainty in the 10-day Monte Carlo: parameter bootstrap (Christoffersen
     & Gonçalves 2005) — simulate a series of the same length from the fitted model,
     re-estimate with arch, filter the real data with the re-estimated parameters to get
     the starting variance, then run the 10-day simulation. Report intervals for
     VaR99 / ES99 / VaR99.9 / P(>10%) / P(>15%).

Dependencies: numpy, pandas, scipy, statsmodels, arch. Can be run standalone (the
numerical kernels from 09 / 11 are embedded).
Usage (in a project folder containing data/processed/, just run it directly):
    python 19_sample_size_robustness.py
    python 19_sample_size_robustness.py --data data/processed/returns__20260921.csv --outdir out/19
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

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260924


# ------------------------------------------------------------ Dependency check ----
_MISSING = []
for _pkg in ("scipy", "statsmodels", "arch"):
    if importlib.util.find_spec(_pkg) is None:
        _MISSING.append(_pkg)
if _MISSING:
    sys.exit("Missing Python packages: " + ", ".join(_MISSING) + "\nPlease first run:\n    "
             + os.path.basename(sys.executable) + " -m pip install " + " ".join(_MISSING))


# ============================================================ Numerical kernel ====
# The following six functions are copied verbatim from 09_regenerate_paper_numbers.py
# (nelder_mead, fit_garch11_t)
# and 11_es_bootstrap_vix.py (opt_block_length, stationary_boot, hs_es, hs_var),
# so that this script can run standalone without depending on 09 / 11 in the same
# directory. Results are bit-for-bit identical to the originals.

def nelder_mead(f, x0, step=0.1, itmax=4000, tol=1e-10):
    """Compact Nelder–Mead, used for the several low-dimensional MLEs in this script."""
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] += step if p[i] == 0 else step * abs(p[i])
        pts.append(p)
    pts = np.array(pts)
    val = np.array([f(p) for p in pts])
    for _ in range(itmax):
        o = np.argsort(val)
        pts, val = pts[o], val[o]
        if abs(val[-1] - val[0]) < tol:
            break
        c = pts[:-1].mean(0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < val[0]:
            xe = c + 2.0 * (c - pts[-1])
            fe = f(xe)
            pts[-1], val[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < val[-2]:
            pts[-1], val[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c)
            fc = f(xc)
            if fc < val[-1]:
                pts[-1], val[-1] = xc, fc
            else:
                pts[1:] = pts[0] + 0.5 * (pts[1:] - pts[0])
                val[1:] = [f(p) for p in pts[1:]]
    o = np.argsort(val)
    return pts[o][0], val[o][0]


def fit_garch11_t(r):
    """MLE for GARCH(1,1) + Student-t innovations. Returns a dict."""
    r = np.asarray(r, float)
    mu0 = r.mean()

    # Parameterization: persistence p = α+β and the share s = α/p each go through a
    # sigmoid,
    # so α+β = p < 1 always holds, stationarity is automatic, and no penalty function is
    # needed.
    # (The first version parameterized α and β with independent sigmoids; the initial
    #  point had α+β=1.03 which violates the constraint,
    #  the whole simplex got penalized to a constant, and the optimizer did not move —
    #  this was caught by the self-test.)
    def unpack(th):
        sig = lambda z: 1 / (1 + math.exp(-max(-60.0, min(60.0, z))))
        p = sig(th[1]) * 0.999
        s = sig(th[2])
        return math.exp(th[0]), p * s, p * (1 - s), 2.05 + math.exp(th[3]), th[4]

    def nll(th):
        om, al, be, nu, mu = unpack(th)
        e = r - mu
        s2 = np.empty(len(r))
        s2[0] = e.var()
        for i in range(1, len(r)):
            s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
        if np.any(~np.isfinite(s2)) or np.any(s2 <= 0):
            return 1e10
        sc = np.sqrt(s2 * (nu - 2) / nu)          # scale of the standardized t
        z = e / sc
        ll = (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
              - 0.5 * math.log(nu * math.pi) - np.log(sc)
              - (nu + 1) / 2 * np.log1p(z * z / nu))
        return -np.sum(ll)

    v = r.var()
    # Multiple starting points: the simplex method is sensitive to the start, take the best
    best, bf = None, np.inf
    for p0, s0, nu0 in ((2.2, -2.4, 4.0), (3.0, -1.5, 6.0), (1.5, -2.0, 10.0)):
        th0 = [math.log(v * 0.05), p0, s0, math.log(nu0 - 2.05), mu0]
        th, f = nelder_mead(nll, th0, step=0.4, itmax=8000)
        if f < bf:
            best, bf = th, f
    th, f = best, bf
    om, al, be, nu, mu = unpack(th)
    e = r - mu
    s2 = np.empty(len(r))
    s2[0] = e.var()
    for i in range(1, len(r)):
        s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
    s2_next = om + al * e[-1] ** 2 + be * s2[-1]
    return dict(omega=om, alpha=al, beta=be, nu=nu, mu=mu,
                sigma2=s2, sigma2_next=s2_next, nll=f,
                persistence=al + be,
                uncond_vol=math.sqrt(om / max(1 - al - be, 1e-9) * 252) * 100)


def opt_block_length(x):
    """Politis & White (2004) optimal block length for the stationary bootstrap
    (simplified implementation).

    Steps: find the autocorrelation truncation point m using a bandwidth of
    2*sqrt(log10(n)), then compute l_opt = (2 G^2 / D_SB)^(1/3) n^(1/3) from G and D_SB.
    """
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    c0 = np.dot(x, x) / n
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    mmax = int(math.ceil(math.sqrt(n))) + kn
    rho = np.array([np.dot(x[k:], x[:-k]) / n / c0 for k in range(1, mmax + 1)])
    # Find the first position where kn consecutive |rho| are all below 2*sqrt(log10(n)/n)
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    m = mmax
    for i in range(len(rho) - kn):
        if np.all(np.abs(rho[i:i + kn]) < thr):
            m = i + 1
            break
    M = min(2 * m, mmax)
    lam = lambda t: 1.0 if abs(t) <= 0.5 else (2 * (1 - abs(t)) if abs(t) <= 1 else 0.0)
    G = sum(lam(k / M) * k * abs(rho[k - 1]) * c0 for k in range(1, M + 1)) * 2
    Ghat = c0 + 2 * sum(lam(k / M) * rho[k - 1] * c0 for k in range(1, M + 1))
    D = 2.0 * Ghat ** 2
    if D <= 0 or G <= 0:
        return 10.0
    return float(min(max((2 * G ** 2 / D) ** (1 / 3) * n ** (1 / 3), 2.0), n / 4))


def stationary_boot(n, R, L, rng):
    """Row-index matrix for the stationary block bootstrap (Politis–Romano), geometric
    block lengths with circular wraparound."""
    out = np.empty((n, R), dtype=np.int64)
    p = 1.0 / L
    for r in range(R):
        idx = np.empty(n, dtype=np.int64)
        f = 0
        while f < n:
            s = rng.integers(0, n)
            ln = min(rng.geometric(p), n - f)
            idx[f:f + ln] = (s + np.arange(ln)) % n
            f += ln
        out[:, r] = idx
    return out


def hs_es(x, a):
    v = np.quantile(x, a)
    return -x[x <= v].mean()


def hs_var(x, a):
    return -np.quantile(x, a)


def find_data():
    """Automatically locate the main-sample data/processed/returns__YYYYMMDD.csv
    (excluding extended snapshots like L / X).
    Searches in the [script directory] and the [current working directory], and takes the
    most recent date."""
    import glob
    import re
    cands = []
    for base in (HERE, os.getcwd()):
        for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv")):
            if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p)):
                cands.append(os.path.abspath(p))
    cands = sorted(set(cands), key=os.path.basename)
    return cands[-1] if cands else None


LINES: list[str] = []
KEY: dict = {}
CHECKS: list[str] = []


def w(s=""):
    LINES.append(s)
    print(s)


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def check(label, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    CHECKS.append(f"[{tag}] {label} {detail}")
    w(f"   CHECK [{tag}] {label} {detail}")


def ci(bs, lo=2.5, hi=97.5):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return np.percentile(bs, [lo, hi])


def pval(bs, null=0.0):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return float(2 * min((bs <= null).mean(), (bs >= null).mean()))


# ============================================================== Regime definitions ====
def two_state(c):
    return lambda v: [v <= c, v > c]


def three_state(a, b):
    return lambda v: [v <= a, (v > a) & (v <= b), v > b]


def quantile_groups(k):
    def f(v):
        qs = [np.nanpercentile(v, 100 * j / k) for j in range(1, k)]
        edges = [-np.inf] + qs + [np.inf]
        return [(v > edges[j]) & (v <= edges[j + 1]) for j in range(k)]
    return f


def top_decile(v):
    q = np.nanpercentile(v, 90)
    return [v <= q, v > q]


SCHEMES = [
    ("Two states, cut at 17", two_state(17)),
    ("Two states, cut at 20", two_state(20)),
    ("Two states, cut at 22", two_state(22)),
    ("Two states, cut at 25", two_state(25)),
    ("Two states, cut at 30", two_state(30)),
    ("Three states, 15/25 (baseline)", three_state(15, 25)),
    ("Three states, 20/30", three_state(20, 30)),
    ("Terciles", quantile_groups(3)),
    ("Quartiles", quantile_groups(4)),
    ("Quintiles", quantile_groups(5)),
    ("Top decile vs rest", top_decile),
]


def es_ratio(rI, rG, m, a=0.05):
    e2 = hs_es(rG[m], a)
    return hs_es(rI[m], a) / e2 if e2 > 1e-10 else np.nan


# ============================================================== Quantile regression ====
def qreg(y, x, tau):
    """Exact solution of linear quantile regression: linear program (HiGHS).
    min Σ τ u⁺ + (1−τ) u⁻  s.t.  a + b x_i + u⁺_i − u⁻_i = y_i,  u± ≥ 0.
    (statsmodels' IRLS fails to converge on some bootstrap samples, so we use the LP.)"""
    from scipy.optimize import linprog
    from scipy.sparse import hstack, identity, csr_matrix
    y = np.asarray(y, float); x = np.asarray(x, float)
    n = len(y)
    c = np.r_[0.0, 0.0, np.full(n, tau), np.full(n, 1 - tau)]
    A = hstack([csr_matrix(np.c_[np.ones(n), x]), identity(n), -identity(n)], format="csr")
    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:2]


# ============================================================== GARCH simulation ====
def garch_filter_next(r, mu, om, al, be):
    e = r - mu
    s2 = np.empty(len(r))
    s2[0] = e.var()
    for i in range(1, len(r)):
        s2[i] = om + al * e[i - 1] ** 2 + be * s2[i - 1]
    return om + al * e[-1] ** 2 + be * s2[-1]


def simulate_series(n, mu, om, al, be, nu, rng, burn=500):
    sc = math.sqrt((nu - 2) / nu)
    s2 = om / max(1 - al - be, 1e-6)
    out = np.empty(n + burn)
    e_prev = 0.0
    for t in range(n + burn):
        s2 = om + al * e_prev ** 2 + be * s2 if t > 0 else s2
        z = rng.standard_t(nu) * sc
        e_prev = math.sqrt(s2) * z
        out[t] = mu + e_prev
    return out[burn:]


def mc10(mu, om, al, be, nu, s2_0, mult, rng, H=10, NP=20000):
    sc = math.sqrt((nu - 2) / nu)
    s2 = np.full(NP, s2_0 * mult ** 2)
    cum = np.ones(NP)
    for _ in range(H):
        z = rng.standard_t(nu, NP) * sc
        e = np.sqrt(s2) * z
        cum *= (1 + mu + e)
        s2 = om + al * e ** 2 + be * s2
    L = -(cum - 1) * 100
    q99 = np.quantile(L, 0.99)
    return dict(VaR99=q99, ES99=L[L >= q99].mean(), VaR999=np.quantile(L, 0.999),
                P10=(L > 10).mean() * 100, P15=(L > 15).mean() * 100)


def arch_fit(r):
    from arch.univariate import ConstantMean, GARCH, StudentsT
    am = ConstantMean(r * 100)
    am.volatility = GARCH(p=1, o=0, q=1)
    am.distribution = StudentsT()
    res = am.fit(disp="off", show_warning=False, options={"maxiter": 2000})
    p = res.params
    assert set(p.index) == {"mu", "omega", "alpha[1]", "beta[1]", "nu"}, list(p.index)
    return dict(mu=p["mu"] / 100, omega=p["omega"] / 1e4, alpha=p["alpha[1]"], beta=p["beta[1]"],
                nu=p["nu"])


# ============================================================== Main program ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="main-sample returns CSV; if omitted, auto-locate data/processed/returns__YYYYMMDD.csv")
    ap.add_argument("--outdir", default="out/19")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--mc-boot", type=int, default=500)
    args = ap.parse_args()
    if args.data is None:
        args.data = find_data()
        if args.data is None:
            sys.exit("Cannot find the main-sample data data/processed/returns__YYYYMMDD.csv.\n"
                     "Please place the data folder next to this script, or specify a path with --data, e.g.:\n"
                     "    python 19_sample_size_robustness.py --data data/processed/returns__20260921.csv")
        print(f"Auto-using data: {args.data}")
    if not os.path.isabs(args.outdir):
        args.outdir = os.path.join(HERE, args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    d = pd.read_csv(args.data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(d)
    rI = ((d.IBIT + d.SPY + d.TLT) / 3).values
    rG = ((d.GLD + d.SPY + d.TLT) / 3).values
    spy, ibit, gld, tlt = d.SPY.values, d.IBIT.values, d.GLD.values, d.TLT.values
    vix = d.VIX_Close.values
    vix_lag = np.r_[np.nan, vix[:-1]]
    w("# Interval estimation and regime sensitivity under a short sample (script 19)\n")
    w(f"Data: `{args.data}`, {n} trading days, {d.Date.iloc[0].date()} → {d.Date.iloc[-1].date()}")
    Lopt = {"IBIT": opt_block_length(rI), "GLD": opt_block_length(rG)}
    Lb = max(2, int(round(np.mean(list(Lopt.values())))))
    w(f"Politis–White block length: IBIT portfolio {Lopt['IBIT']:.1f}, GLD portfolio {Lopt['GLD']:.1f}; "
      f"ratios and regime statistics use L = {Lb} (same as script 11); tail-event statistics use L = 10 "
      "(same as script 03).")
    w(f"Bootstrap draws B = {args.boot}; seed {SEED}.\n")
    IDX_R = stationary_boot(n, args.boot, Lb, rng)
    IDX_T = stationary_boot(n, args.boot, 10, rng)

    # ------------------------------------------------------------------ A0
    w("## A0. Baseline split (15/25): ES95 ratio intervals for the three regimes (for Table 4)\n")
    KEY["A0"] = {}
    f0 = three_state(15, 25)
    rows0 = []
    for timing, vv in (("same-day", vix), ("lagged", vix_lag)):
        ok = ~np.isnan(vv)
        ms = [m & ok for m in f0(vv)]
        pts = [es_ratio(rI, rG, m) for m in ms]
        bs = [[], [], []]
        for c in range(IDX_R.shape[1]):
            i = IDX_R[:, c]
            vb = vv[i]
            okb = ~np.isnan(vb)
            mb = [m & okb for m in f0(vb)]
            for j in range(3):
                bs[j].append(es_ratio(rI[i], rG[i], mb[j]) if mb[j].sum() >= 10 else np.nan)
        for j, lab in enumerate(["low", "mid", "high"]):
            lo, hi = ci(bs[j])
            rows0.append([timing, lab, int(ms[j].sum()), f"{pts[j]:.2f} [{lo:.2f}, {hi:.2f}]"])
            KEY["A0"][f"{timing}_{lab}"] = [pts[j], lo, hi]
    w(md(rows0, ["VIX", "Regime", "Days", "ES95 ratio [95% CI]"]))

    # ------------------------------------------------------------------ A
    w("## A. Regime-definition sensitivity: ES95 ratio (IBIT portfolio / GLD portfolio)\n")
    w("Each definition compares the [lowest] and [highest] regime. On each bootstrap "
      "resample the regimes are re-drawn. Within a regime, ES95 is based on the worst 5% of "
      "days in that regime — when a high regime has only 15–61 days, ES95 is based on only "
      "1–3 observations.\n")
    rowsA, sigA = [], 0
    KEY["A"] = []
    for timing, vv in (("same-day", vix), ("lagged", vix_lag)):
        ok = ~np.isnan(vv)
        for name, f in SCHEMES:
            ms = f(vv)
            mlo, mhi = ms[0] & ok, ms[-1] & ok
            plo, phi = es_ratio(rI, rG, mlo), es_ratio(rI, rG, mhi)
            blo, bhi = [], []
            for c in range(IDX_R.shape[1]):
                i = IDX_R[:, c]
                vb = vv[i]
                okb = ~np.isnan(vb)
                mb = f(vb)
                l_, h_ = mb[0] & okb, mb[-1] & okb
                if l_.sum() < 10 or h_.sum() < 10:
                    blo.append(np.nan); bhi.append(np.nan)
                    continue
                blo.append(es_ratio(rI[i], rG[i], l_))
                bhi.append(es_ratio(rI[i], rG[i], h_))
            blo, bhi = np.array(blo), np.array(bhi)
            diff = bhi - blo
            clo, chi, cd = ci(blo), ci(bhi), ci(diff)
            pv = pval(diff)
            sig = pv < 0.05
            sigA += int(sig)
            rowsA.append([name, timing, int(mlo.sum()), int(mhi.sum()),
                          f"{plo:.2f} [{clo[0]:.2f}, {clo[1]:.2f}]",
                          f"{phi:.2f} [{chi[0]:.2f}, {chi[1]:.2f}]",
                          f"{phi - plo:+.2f} [{cd[0]:+.2f}, {cd[1]:+.2f}]", f"{pv:.3f}"])
            KEY["A"].append(dict(scheme=name, timing=timing, n_low=int(mlo.sum()), n_high=int(mhi.sum()),
                                 low=plo, low_ci=list(clo), high=phi, high_ci=list(chi),
                                 diff=phi - plo, diff_ci=list(cd), p=pv))
    w(md(rowsA, ["Definition", "VIX", "Days (low)", "Days (high)", "Low-regime ratio [95% CI]",
                 "High-regime ratio [95% CI]", "High − low [95% CI]", "p"]))
    w(f"**Under {sigA}/{len(rowsA)} definitions, the high − low difference is significant at the 5% level.**\n")
    KEY["A_sig"] = sigA
    KEY["A_n"] = len(rowsA)
    base = [k for k in KEY["A"] if k["scheme"].startswith("Three states, 15/25") and k["timing"] == "same-day"][0]
    check("A. Baseline definition's high-regime point estimate matches script 11 (2.04)", abs(base["high"] - 2.04) < 0.006,
          f"{base['high']:.3f}")
    check("A. Baseline regime day counts 161/38", base["n_low"] == 161 and base["n_high"] == 38,
          f"{base['n_low']}/{base['n_high']}")

    # ------------------------------------------------------------------ B
    w("## B. Continuous specification: 5% quantile regression r_p,t = a + b·VIX + u\n")
    w("Conditional VaR95(v) = −(a + b·v). R(v) = VaR95_IBIT(v) / VaR95_GLD(v). "
      "Bootstrap resamples (r, VIX) jointly, L = " + str(Lb) + ".\n")
    grid = [15, 20, 25, 30]
    KEY["B"] = {}
    for timing, vv in (("lagged", vix_lag), ("same-day", vix)):
        ok = ~np.isnan(vv)
        x, yI, yG = vv[ok], rI[ok], rG[ok]
        aI = qreg(yI, x, 0.05); aG = qreg(yG, x, 0.05)
        import statsmodels.api as sm
        smI = sm.QuantReg(yI, sm.add_constant(x)).fit(q=0.05, max_iter=20000).params
        check(f"B. LP quantile regression agrees with statsmodels ({timing})", np.allclose(aI, smI, atol=1e-4),
              f"LP {np.round(aI, 6)} vs sm {np.round(np.asarray(smI), 6)}")
        R = lambda pI, pG, v: (-(pI[0] + pI[1] * v)) / (-(pG[0] + pG[1] * v))
        pt = {v: R(aI, aG, v) for v in grid}
        bs = {v: [] for v in grid}
        bslope = {"I": [], "G": []}
        idx_ok = np.where(ok)[0]
        m = len(idx_ok)
        IDXq = stationary_boot(m, args.boot, Lb, rng)
        for c in range(IDXq.shape[1]):
            i = IDXq[:, c]
            try:
                bI = qreg(yI[i], x[i], 0.05); bG = qreg(yG[i], x[i], 0.05)
            except Exception:
                continue
            for v in grid:
                bs[v].append(R(bI, bG, v))
            bslope["I"].append(bI[1]); bslope["G"].append(bG[1])
        d3015 = np.array(bs[30]) - np.array(bs[15])
        rows = []
        for v in grid:
            lo, hi = ci(bs[v])
            varI = -(aI[0] + aI[1] * v) * 100
            varG = -(aG[0] + aG[1] * v) * 100
            rows.append([v, f"{varI:.2f}", f"{varG:.2f}", f"{pt[v]:.2f} [{lo:.2f}, {hi:.2f}]"])
        w(f"### {timing} VIX\n")
        w(f"Slope b (per 1 point of VIX, in percentage points): IBIT portfolio {aI[1]*100:+.4f} "
          f"[{ci(bslope['I'])[0]*100:+.4f}, {ci(bslope['I'])[1]*100:+.4f}]; "
          f"GLD portfolio {aG[1]*100:+.4f} [{ci(bslope['G'])[0]*100:+.4f}, {ci(bslope['G'])[1]*100:+.4f}]\n")
        w(md(rows, ["VIX", "IBIT cond. VaR95 (%)", "GLD cond. VaR95 (%)", "Ratio [95% CI]"]))
        lo, hi = ci(d3015)
        pv = pval(d3015)
        w(f"R(30) − R(15) = {pt[30]-pt[15]:+.2f}, 95% CI [{lo:+.2f}, {hi:+.2f}], p = {pv:.3f}\n")
        KEY["B"][timing] = dict(slopeI=aI[1] * 100, slopeI_ci=list(ci(bslope["I"]) * 100),
                                slopeG=aG[1] * 100, slopeG_ci=list(ci(bslope["G"]) * 100),
                                ratio={v: [pt[v]] + list(ci(bs[v])) for v in grid},
                                varI={v: -(aI[0] + aI[1] * v) * 100 for v in grid},
                                varG={v: -(aG[0] + aG[1] * v) * 100 for v in grid},
                                d3015=pt[30] - pt[15], d3015_ci=[lo, hi], p=pv)
    check("B. Quantile-regression slope is negative (higher VIX, lower 5% quantile)",
          KEY["B"]["lagged"]["slopeI"] < 0 and KEY["B"]["lagged"]["slopeG"] < 0,
          f"{KEY['B']['lagged']['slopeI']:+.4f}, {KEY['B']['lagged']['slopeG']:+.4f}")

    # ------------------------------------------------------------------ C
    w("## C. Block-bootstrap intervals of full-sample VaR (historical simulation)\n")
    IDX_I = stationary_boot(n, 5000, max(2, int(round(Lopt["IBIT"]))), rng)
    IDX_G = stationary_boot(n, 5000, max(2, int(round(Lopt["GLD"]))), rng)
    IDX_2 = stationary_boot(n, 5000, Lb, rng)
    rows = []
    KEY["C"] = {}
    for a in (0.05, 0.01):
        vI, vG = hs_var(rI, a) * 100, hs_var(rG, a) * 100
        bI = np.array([hs_var(rI[IDX_I[:, c]], a) for c in range(IDX_I.shape[1])]) * 100
        bG = np.array([hs_var(rG[IDX_G[:, c]], a) for c in range(IDX_G.shape[1])]) * 100
        bR = np.array([hs_var(rI[IDX_2[:, c]], a) / hs_var(rG[IDX_2[:, c]], a) for c in range(IDX_2.shape[1])])
        cI, cG, cR = ci(bI), ci(bG), ci(bR)
        nt = int(round(a * n))
        rows.append([f"{int((1-a)*100)}%", nt, f"{vI:.2f} [{cI[0]:.2f}, {cI[1]:.2f}]",
                     f"{vG:.2f} [{cG[0]:.2f}, {cG[1]:.2f}]", f"{vI/vG:.2f} [{cR[0]:.2f}, {cR[1]:.2f}]",
                     f"{(bR <= 1).mean():.4f}"])
        KEY["C"][str(a)] = dict(I=vI, I_ci=list(cI), G=vG, G_ci=list(cG), R=vI / vG, R_ci=list(cR),
                                p_ratio_le1=float((bR <= 1).mean()))
    w(md(rows, ["Level", "Tail obs.", "IBIT VaR [95% CI]", "GLD VaR [95% CI]", "Ratio [95% CI]",
                "Share of replications with ratio ≤ 1"]))
    check("C. VaR point estimates match Table 2 (IBIT 1.95/3.35, GLD 1.09/2.00)",
          abs(KEY["C"]["0.05"]["I"] - 1.95) < 0.006 and abs(KEY["C"]["0.01"]["I"] - 3.35) < 0.006
          and abs(KEY["C"]["0.05"]["G"] - 1.09) < 0.006 and abs(KEY["C"]["0.01"]["G"] - 2.00) < 0.006, "")

    # ------------------------------------------------------------------ D
    w("## D. Intervals for ρ and β by regime (contemporaneous and lagged VIX, 15/25 split)\n")
    f = three_state(15, 25)

    def rb(a, b):
        va = np.var(b, ddof=1)
        beta = np.cov(a, b, ddof=1)[0, 1] / va if va > 0 else np.nan
        return np.corrcoef(a, b)[0, 1], beta

    KEY["D"] = {}
    for timing, vv in (("same-day", vix), ("lagged", vix_lag)):
        ok = ~np.isnan(vv)
        ms = [m & ok for m in f(vv)]
        stats = {}
        for j, lab in enumerate(["low", "mid", "high"]):
            m = ms[j]
            rI_, bI_ = rb(ibit[m], spy[m])
            rG_, bG_ = rb(gld[m], spy[m])
            stats[lab] = dict(rhoI=rI_, betaI=bI_, rhoG=rG_, betaG=bG_, n=int(m.sum()))
        boots = {lab: {k: [] for k in ("rhoI", "betaI", "rhoG", "betaG")} for lab in ("low", "mid", "high")}
        for c in range(IDX_R.shape[1]):
            i = IDX_R[:, c]
            vb = vv[i]
            okb = ~np.isnan(vb)
            mb = [m & okb for m in f(vb)]
            for j, lab in enumerate(["low", "mid", "high"]):
                m = mb[j]
                if m.sum() < 10:
                    for k in boots[lab]:
                        boots[lab][k].append(np.nan)
                    continue
                a1, b1 = rb(ibit[i][m], spy[i][m])
                a2, b2 = rb(gld[i][m], spy[i][m])
                boots[lab]["rhoI"].append(a1); boots[lab]["betaI"].append(b1)
                boots[lab]["rhoG"].append(a2); boots[lab]["betaG"].append(b2)
        rows = []
        for lab in ("low", "mid", "high"):
            s = stats[lab]
            line = [lab, s["n"]]
            for k in ("rhoI", "betaI", "rhoG", "betaG"):
                lo, hi = ci(boots[lab][k])
                line.append(f"{s[k]:.3f} [{lo:.2f}, {hi:.2f}]")
                s[k + "_ci"] = [lo, hi]
            rows.append(line)
        w(f"### {timing} VIX\n")
        w(md(rows, ["Regime", "Days", "ρ IBIT–SPY", "β IBIT", "ρ GLD–SPY", "β GLD"]))
        drows = []
        diffs = {}
        for k in ("rhoI", "betaI", "rhoG", "betaG"):
            dd = np.array(boots["high"][k]) - np.array(boots["low"][k])
            lo, hi = ci(dd)
            pv = pval(dd)
            pe = stats["high"][k] - stats["low"][k]
            drows.append([k, f"{pe:+.3f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{pv:.3f}"])
            diffs[k] = dict(d=pe, ci=[lo, hi], p=pv)
        w(md(drows, ["Statistic", "High − low", "95% CI", "p"]))
        KEY["D"][timing] = dict(stats=stats, diffs=diffs)
    s0 = KEY["D"]["same-day"]["stats"]
    check("D. Point estimates match script 11 Section D (ρ 0.224/0.561, β 1.373/0.922)",
          abs(s0["low"]["rhoI"] - 0.224) < 6e-4 and abs(s0["high"]["rhoI"] - 0.561) < 6e-4
          and abs(s0["low"]["betaI"] - 1.373) < 6e-4 and abs(s0["high"]["betaI"] - 0.922) < 6e-4, "")

    # ------------------------------------------------------------------ E
    w("## E. Beta-adjusted excess return on the worst 5% of SPY trading days (percentage points)\n")
    w("Excess = the asset's average return on those days − full-sample β × SPY's average "
      "return on those days. On each bootstrap resample the threshold and β are both "
      "recomputed; L = 10.\n")

    def excess(a, s):
        thr = np.percentile(s, 5)
        m = s <= thr
        beta = np.cov(a, s, ddof=1)[0, 1] / np.var(s, ddof=1)
        return (a[m].mean() - beta * s[m].mean()) * 100, int(m.sum())

    rows = []
    KEY["E"] = {}
    for lab, a in (("IBIT", ibit), ("GLD", gld), ("TLT", tlt)):
        pe, nm = excess(a, spy)
        bs = np.array([excess(a[IDX_T[:, c]], spy[IDX_T[:, c]])[0] for c in range(IDX_T.shape[1])])
        lo, hi = ci(bs)
        pv = pval(bs)
        rows.append([lab, nm, f"{pe:+.2f}", f"[{lo:+.2f}, {hi:+.2f}]", f"{pv:.3f}"])
        KEY["E"][lab] = dict(x=pe, ci=[lo, hi], p=pv, n=nm)
    w(md(rows, ["Asset", "Days", "Excess (pp)", "95% CI", "p"]))
    check("E. Point estimates match existing results (IBIT −0.02, GLD +0.29, TLT +0.21)",
          abs(KEY["E"]["IBIT"]["x"] + 0.02) < 0.006 and abs(KEY["E"]["GLD"]["x"] - 0.29) < 0.006
          and abs(KEY["E"]["TLT"]["x"] - 0.21) < 0.006,
          f"{KEY['E']['IBIT']['x']:+.3f} {KEY['E']['GLD']['x']:+.3f} {KEY['E']['TLT']['x']:+.3f}")

    # ------------------------------------------------------------------ F
    w("## F. Cross-quantilogram ρ(τ1 = 0.05, τ2 = 0.05), contemporaneous with SPY\n")

    def cq(y, x, t1=0.05, t2=0.05):
        px = (x <= np.quantile(x, t1)) - t1
        py = (y <= np.quantile(y, t2)) - t2
        return float(np.sum(px * py) / math.sqrt(np.sum(px ** 2) * np.sum(py ** 2)))

    pts = {lab: cq(a, spy) for lab, a in (("IBIT", ibit), ("GLD", gld), ("TLT", tlt))}
    bs = {lab: [] for lab in pts}
    for c in range(IDX_T.shape[1]):
        i = IDX_T[:, c]
        for lab, a in (("IBIT", ibit), ("GLD", gld), ("TLT", tlt)):
            bs[lab].append(cq(a[i], spy[i]))
    rows = []
    KEY["F"] = {}
    for lab in pts:
        lo, hi = ci(bs[lab])
        rows.append([lab, f"{pts[lab]:.3f}", f"[{lo:.3f}, {hi:.3f}]"])
        KEY["F"][lab] = dict(x=pts[lab], ci=[lo, hi])
    dd = np.array(bs["IBIT"]) - np.array(bs["GLD"])
    lo, hi = ci(dd)
    pv = pval(dd)
    rows.append(["IBIT − GLD", f"{pts['IBIT'] - pts['GLD']:+.3f}", f"[{lo:+.3f}, {hi:+.3f}] (p = {pv:.3f})"])
    KEY["F"]["diff"] = dict(x=pts["IBIT"] - pts["GLD"], ci=[lo, hi], p=pv)
    w(md(rows, ["Asset", "ρ(0.05, 0.05)", "95% CI"]))
    check("F. Point estimates match script 03 (R) (0.218 / 0.116 / 0.116)",
          abs(pts["IBIT"] - 0.218) < 6e-4 and abs(pts["GLD"] - 0.116) < 6e-4 and abs(pts["TLT"] - 0.116) < 6e-4,
          f"{pts['IBIT']:.3f} {pts['GLD']:.3f} {pts['TLT']:.3f}")

    # ------------------------------------------------------------------ G
    w("## G. Parameter uncertainty in the 10-day Monte Carlo (parameter bootstrap)\n")
    w(f"B = {args.mc_boot}: simulate {n} days from the point-estimate model → re-estimate "
      "GARCH(1,1)-t with arch → filter the real data with the re-estimated parameters to get "
      "the starting variance → 20,000 10-day paths. stressed = starting volatility × 2.\n")
    KEY["G"] = {}
    rowsG = []
    for lab, r in (("IBIT", rI), ("GLD", rG)):
        f0 = fit_garch11_t(r)
        fa = arch_fit(r)
        w(f"{lab} portfolio point estimates (09 self-implemented): α={f0['alpha']:.3f} β={f0['beta']:.3f} ν={f0['nu']:.2f}; "
          f"arch on the same data: α={fa['alpha']:.3f} β={fa['beta']:.3f} ν={fa['nu']:.2f}")
        check(f"G. {lab} arch and 09 estimates are close", abs(f0["alpha"] - fa["alpha"]) < 0.02
              and abs(f0["beta"] - fa["beta"]) < 0.03, "")
        pt = {}
        for state, mult in (("current", 1.0), ("stressed", 2.0)):
            pt[state] = mc10(f0["mu"], f0["omega"], f0["alpha"], f0["beta"], f0["nu"], f0["sigma2_next"],
                             mult, np.random.default_rng(20260921))
        boots = {s: {k: [] for k in pt["current"]} for s in pt}
        vols = {s: [] for s in pt}
        nfail = 0
        for b in range(args.mc_boot):
            sim = simulate_series(n, f0["mu"], f0["omega"], f0["alpha"], f0["beta"], f0["nu"], rng)
            try:
                fb = arch_fit(sim)
            except Exception:
                nfail += 1
                continue
            s2n = garch_filter_next(r, fb["mu"], fb["omega"], fb["alpha"], fb["beta"])
            for state, mult in (("current", 1.0), ("stressed", 2.0)):
                o = mc10(fb["mu"], fb["omega"], fb["alpha"], fb["beta"], fb["nu"], s2n, mult, rng)
                for k in o:
                    boots[state][k].append(o[k])
                vols[state].append(math.sqrt(s2n * 252) * 100 * mult)
        for state in ("current", "stressed"):
            line = [f"{lab} ({state})", f"{math.sqrt(f0['sigma2_next']*252)*100*(2 if state=='stressed' else 1):.1f} "
                    f"[{ci(vols[state])[0]:.1f}, {ci(vols[state])[1]:.1f}]"]
            KEY["G"][f"{lab}_{state}"] = {"vol": [math.sqrt(f0['sigma2_next'] * 252) * 100 * (2 if state == 'stressed' else 1)]
                                          + list(ci(vols[state]))}
            for k in ("VaR99", "ES99", "VaR999", "P10", "P15"):
                lo, hi = ci(boots[state][k])
                fmt = "{:.2f}" if k.startswith("P") else "{:.1f}"
                line.append(f"{fmt.format(pt[state][k])} [{fmt.format(lo)}, {fmt.format(hi)}]")
                KEY["G"][f"{lab}_{state}"][k] = [pt[state][k], lo, hi]
            rowsG.append(line)
        w(f"   {lab}: re-estimation failures {nfail}/{args.mc_boot}")
    w("")
    w(md(rowsG, ["Portfolio (state)", "Initial vol. (%)", "VaR99", "ES99", "VaR99.9", "P(>10%)", "P(>15%)"]))
    check("G. Point estimates reproduce Table 8 (IBIT current VaR99 9.2, GLD stressed VaR99 11.5)",
          abs(KEY["G"]["IBIT_current"]["VaR99"][0] - 9.2) < 0.051
          and abs(KEY["G"]["GLD_stressed"]["VaR99"][0] - 11.5) < 0.051,
          f"{KEY['G']['IBIT_current']['VaR99'][0]:.2f} {KEY['G']['GLD_stressed']['VaR99'][0]:.2f}")

    w("\n## Self-test summary\n")
    for c in CHECKS:
        w("  " + c)
    w(f"\nElapsed {time.time() - t0:.0f} seconds")
    with open(os.path.join(args.outdir, "sample_size_robustness.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(args.outdir, "sample_size_robustness.json"), "w", encoding="utf8") as fh:
        json.dump(KEY, fh, indent=1, default=float)


if __name__ == "__main__":
    main()