"""
18b_backtest_supplement.py — Supplementary backtests for (incomplete backtest)

Relation to 04
--------------
04_var_es_backtest.py already did rolling out-of-sample forecasts, Kupiec /
Christoffersen / conditional coverage, Acerbi-Szekely (two GARCH models), McNeil-Frey
and GW tests.
This script [directly reuses 04's forecast functions] (same n_start / refit / window /
seed conventions), and only fills in the parts 04 did not do or did not do thoroughly:

  A. Small-sample Monte Carlo p-values (Dufour 2006)
     There are only 357 out-of-sample days, and the 1% level has a nominal 3.6 breaches,
     so the chi2 asymptotic distribution of the LR statistics is very unreliable. For
     UC / IND / CC / DQ we simulate hit sequences under H0 to give exact p-values, and
     report the asymptotic p-values side by side.
       UC and CC H0: hits iid Bernoulli(alpha)
       IND H0: hits iid Bernoulli(x/n) (independent but with coverage not restricted)

  B. Engle-Manganelli (2004) dynamic quantile DQ test
     Christoffersen only looks at a first-order Markov chain (yesterday's breach →
     today's breach), which has low power when there are only 6–25 breaches. DQ regresses
     hits on a constant + 4 lagged hits + the same-day VaR, which can capture more general
     clustering and also the misspecification "VaR level correlated with breach
     probability".

  C. Unified ES backtest across the four models
     In 04 the AS tests for historical simulation and EWMA had no p-values. In fact both
     have well-defined predictive distributions:
       EWMA-normal : N(0, sigma_t^2)
       Historical simulation : the empirical distribution of the past 250 days' returns
     So both can be simulated under H0. This script gives, uniformly for all four models:
       - Acerbi-Szekely Z1 / Z2 and simulated p-values
       - Du-Escanciano (2017) unconditional U_ES and conditional C_ES(m) tests
         (based on the PIT cumulative breach H_t; the only input needed is u_t = F_t(r_t))

  D. Description of breach clustering: number of consecutive-breach pairs, longest run,
     shortest gap, breaches in the April 2025 tariff-shock month and on days with
     VIX > 25, and export of all breach dates.

  E. Summary table + figure of VaR bounds and breach points.

Dependencies: numpy, pandas, scipy, arch, matplotlib
Usage:
    python 18b_backtest_supplement.py
    python 18b_backtest_supplement.py --selftest
    python 18b_backtest_supplement.py --bvar 9999 --bes 4999   # more simulation draws
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_04 = [
    "04 var es backtest.py",
    "04_var_es_backtest.py",
    os.path.join("Claude outputs", "04_var_es_backtest.py"),
]

ALPHAS = (0.01, 0.05)
MODELS = ("GARCH-t", "GARCH-normal", "Historical simulation", "EWMA-normal")
MODEL_EN = {"GARCH-t": "GARCH-t", "GARCH-normal": "GARCH-N",
            "Historical simulation": "HS", "EWMA-normal": "EWMA-N"}
PORT_EN = {"IBIT portfolio": "IBIT portfolio", "GLD portfolio": "GLD portfolio",
           "IBIT vol-matched": "IBIT vol-matched", "60/40 benchmark": "60/40 benchmark"}
DQ_LAGS = 4
CES_LAGS = 5
CRISIS = ("2025-04-01", "2025-04-30")
VIX_HI = 25.0

LINES: list[str] = []
CHECKS: list[str] = []


def w(s: str = "") -> None:
    print(s)
    LINES.append(s)


def rule(ch: str = "=") -> None:
    w(ch * 78)


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    CHECKS.append(f"[{tag}] {label} {detail}")
    w(f"   CHECK [{tag}] {label} {detail}")
    return ok


def md(rows, head) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def load_04():
    for c in CANDIDATES_04:
        p = os.path.join(HERE, c)
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("bt04", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod, p
    sys.exit("Cannot find the 04 script (04 var es backtest.py); please place it in the same directory.")


# =========================================================== MC p-values ====
def p_right(obs: float, sims: np.ndarray) -> float:
    """The larger the statistic, the more it rejects. (1 + #{S* >= S}) / (B + 1), slightly
    conservative for discrete statistics."""
    sims = sims[np.isfinite(sims)]
    if not np.isfinite(obs) or len(sims) == 0:
        return np.nan
    return float((1 + np.sum(sims >= obs - 1e-12)) / (len(sims) + 1))


def p_left(obs: float, sims: np.ndarray) -> float:
    sims = sims[np.isfinite(sims)]
    if not np.isfinite(obs) or len(sims) == 0:
        return np.nan
    return float((1 + np.sum(sims <= obs + 1e-12)) / (len(sims) + 1))


# ================================================== VaR tests (vectorized) ====
def lr_uc_vec(x, n: int, a: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pi = x / n
    ll0 = (n - x) * math.log(1 - a) + x * math.log(a)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    return np.maximum(-2.0 * (ll0 - ll1), 0.0)


def lr_ind_vec(H: np.ndarray):
    """Christoffersen (1998) first-order Markov independence. H: (B, n) 0/1.
    Returns (LR, transition counts)."""
    H = np.atleast_2d(H).astype(np.int8)
    a0, a1 = H[:, :-1], H[:, 1:]
    n00 = ((a0 == 0) & (a1 == 0)).sum(1)
    n01 = ((a0 == 0) & (a1 == 1)).sum(1)
    n10 = ((a0 == 1) & (a1 == 0)).sum(1)
    n11 = ((a0 == 1) & (a1 == 1)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p01 = np.where(n00 + n01 > 0, n01 / np.maximum(n00 + n01, 1), 0.0)
        p11 = np.where(n10 + n11 > 0, n11 / np.maximum(n10 + n11, 1), 0.0)
        p = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll1 = xlogy(n00, 1 - p01) + xlogy(n01, p01) + xlogy(n10, 1 - p11) + xlogy(n11, p11)
    ll0 = xlogy(n00 + n10, 1 - p) + xlogy(n01 + n11, p)
    return np.maximum(-2.0 * (ll0 - ll1), 0.0), np.stack([n00, n01, n10, n11], 1)


def dq_stat(hit: np.ndarray, var: np.ndarray, a: float, K: int = DQ_LAGS) -> float:
    """
    Engle-Manganelli DQ: regress Hit_t = I_t - a on [1, Hit_{t-1..t-K}, VaR_t],
    DQ = Hit' X (X'X)^- X' Hit / (a(1-a)) ~ chi2(K+2).
    Multiplying VaR by 100 is only for numerical conditioning and does not affect the
    statistic (projection invariance).
    """
    I = hit.astype(float) - a
    n = len(I)
    y = I[K:]
    cols = [np.ones(n - K)] + [I[K - j:n - j] for j in range(1, K + 1)] + [var[K:] * 100.0]
    X = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    f = X @ b
    return float(f @ f / (a * (1 - a)))


# ======================================================= ES tests ====
def as_z_mat(R: np.ndarray, v: np.ndarray, e: np.ndarray, a: float):
    """Acerbi-Szekely Z1/Z2; R may be (B, n). VaR/ES are return-form negative values,
    same convention as 04."""
    R = np.atleast_2d(R)
    n = R.shape[1]
    br = R < v
    ratio = np.where(br, R / (-e), 0.0)
    cnt = br.sum(1)
    z1 = np.where(cnt > 0, ratio.sum(1) / np.maximum(cnt, 1) + 1.0, np.nan)
    z2 = ratio.sum(1) / (n * a) + 1.0
    return z1, z2


def de_tests(u: np.ndarray, a: float, m: int = CES_LAGS):
    """
    Du & Escanciano (2017): cumulative breach H_t = (a - u_t) 1{u_t <= a} / a.
    Under H0, H_t is iid with mean a/2 and variance a(1/3 - a/4).
      U_ES = sqrt(n) (mean H - a/2) / sqrt(a(1/3 - a/4))  ~ N(0,1), positive = underestimates risk
      C_ES = n * sum_{j=1..m} rho_j^2                     ~ chi2(m)
    """
    u = np.atleast_2d(u)
    H = (a - u) * (u <= a) / a
    n = H.shape[1]
    U = math.sqrt(n) * (H.mean(1) - a / 2) / math.sqrt(a * (1 / 3 - a / 4))
    Hc = H - a / 2
    g0 = (Hc ** 2).mean(1)
    C = np.zeros(H.shape[0])
    for j in range(1, m + 1):
        gj = (Hc[:, j:] * Hc[:, :-j]).sum(1) / (n - j)
        C += (gj / g0) ** 2
    return U, n * C


# ============================================ Predictive distributions: sampling and PIT ====
class Forecast:
    """Out-of-sample forecasts for a given portfolio, model, and alpha, together with an
    H0 sampler and the PIT."""

    def __init__(self, kind, r, v, e, mu=None, sd=None, nu=None, W=None):
        self.kind, self.r, self.v, self.e = kind, r, v, e
        self.mu, self.sd, self.nu, self.W = mu, sd, nu, W

    def sample(self, rng, B: int) -> np.ndarray:
        n = len(self.r)
        if self.kind == "t":
            z = rng.standard_t(self.nu, size=(B, n)) / np.sqrt(self.nu / (self.nu - 2.0))
            return self.mu + self.sd * z
        if self.kind in ("normal", "ewma"):
            return self.mu + self.sd * rng.standard_normal((B, n))
        if self.kind == "hs":
            idx = rng.integers(0, self.W.shape[1], size=(B, n))
            return self.W[np.arange(n)[None, :], idx]
        raise ValueError(self.kind)

    def pit(self) -> np.ndarray:
        if self.kind == "t":
            z = (self.r - self.mu) / self.sd
            return stats.t.cdf(z * np.sqrt(self.nu / (self.nu - 2.0)), self.nu)
        if self.kind in ("normal", "ewma"):
            return stats.norm.cdf((self.r - self.mu) / self.sd)
        if self.kind == "hs":
            rr = self.r[:, None]
            return ((self.W < rr).sum(1) + 0.5 * (self.W == rr).sum(1)) / self.W.shape[1]
        raise ValueError(self.kind)


# ======================================================== All tests ====
def run_tests(fc: Forecast, a: float, rng, B_var: int, B_es: int) -> dict:
    r, v, e = fc.r, fc.v, fc.e
    n = len(r)
    hit = (r < v).astype(np.int8)
    x = int(hit.sum())
    out = dict(n=n, x=x, rate=x / n)

    # --- Observed statistics
    uc = float(lr_uc_vec(x, n, a))
    ind_arr, cnt = lr_ind_vec(hit)
    ind = float(ind_arr[0])
    cc = uc + ind
    dq = dq_stat(hit, v, a)
    out.update(uc=uc, ind=ind, cc=cc, dq=dq,
               n00=int(cnt[0, 0]), n01=int(cnt[0, 1]), n10=int(cnt[0, 2]), n11=int(cnt[0, 3]),
               p_uc_asy=float(stats.chi2.sf(uc, 1)), p_ind_asy=float(stats.chi2.sf(ind, 1)),
               p_cc_asy=float(stats.chi2.sf(cc, 2)), p_dq_asy=float(stats.chi2.sf(dq, DQ_LAGS + 2)))

    # --- MC: UC / CC under Bernoulli(a); IND under Bernoulli(x/n)
    Hs = (rng.random((B_var, n)) < a).astype(np.int8)
    uc_s = lr_uc_vec(Hs.sum(1), n, a)
    ind_s, _ = lr_ind_vec(Hs)
    out["p_uc_mc"] = p_right(uc, uc_s)
    out["p_cc_mc"] = p_right(cc, uc_s + ind_s)
    ph = max(x / n, 1.0 / n)
    Hs2 = (rng.random((B_var, n)) < ph).astype(np.int8)
    ind_s2, _ = lr_ind_vec(Hs2)
    out["p_ind_mc"] = p_right(ind, ind_s2)
    # DQ: hits iid Bernoulli(a), VaR series fixed
    B_dq = min(B_var, 4999)
    dq_s = np.array([dq_stat(Hs[b], v, a) for b in range(B_dq)])
    out["p_dq_mc"] = p_right(dq, dq_s)

    # --- ES: Acerbi-Szekely (all four models simulated under their H0 predictive distribution)
    z1, z2 = as_z_mat(r, v, e, a)
    out["z1"], out["z2"] = float(z1[0]), float(z2[0])
    Rs = fc.sample(rng, B_es)
    z1s, z2s = as_z_mat(Rs, v, e, a)
    out["p_z1"] = p_left(out["z1"], z1s)
    out["p_z2"] = p_left(out["z2"], z2s)

    # --- ES: Du-Escanciano
    u = fc.pit()
    U, C = de_tests(u, a)
    out["ues"], out["ces"] = float(U[0]), float(C[0])
    out["p_ues_asy"] = float(stats.norm.sf(out["ues"]))            # one-sided: underestimates risk
    out["p_ces_asy"] = float(stats.chi2.sf(out["ces"], CES_LAGS))
    Us, Cs = de_tests(rng.random((B_es, n)), a)
    out["p_ues_mc"] = p_right(out["ues"], Us)
    out["p_ces_mc"] = p_right(out["ces"], Cs)
    out["pit_agree"] = float(np.mean((u < a) == (r < v)))
    out["u"], out["hit"] = u, hit
    return out


def cluster_stats(hit: np.ndarray, dates: np.ndarray, vix: np.ndarray) -> dict:
    idx = np.where(hit == 1)[0]
    runs, cur = [], 0
    for h in hit:
        cur = cur + 1 if h else 0
        if h:
            runs.append(cur)
    gaps = np.diff(idx)
    crisis = (dates >= np.datetime64(CRISIS[0])) & (dates <= np.datetime64(CRISIS[1]))
    hi = vix > VIX_HI
    return dict(max_run=max(runs) if runs else 0,
                min_gap=int(gaps.min()) if len(gaps) else np.nan,
                med_gap=float(np.median(gaps)) if len(gaps) else np.nan,
                crisis_br=int(hit[crisis].sum()), crisis_days=int(crisis.sum()),
                hivix_br=int(hit[hi].sum()), hivix_days=int(hi.sum()),
                dates=[str(pd.Timestamp(dates[i]).date()) for i in idx])


# ============================================================ Self-test ====
def selftest(m04) -> int:
    rng = np.random.default_rng(7)
    ok = True
    print("=" * 78); print("Self-test"); print("=" * 78)

    # 1. Vectorized LR agrees with 04's scalar version
    worst = 0.0
    for _ in range(300):
        a = rng.choice([0.01, 0.05]); n = 357
        h = (rng.random(n) < rng.choice([a, 2 * a, 4 * a])).astype(int)
        if rng.random() < 0.3:                      # artificially create clustering
            k = rng.integers(0, n - 5); h[k:k + 3] = 1
        worst = max(worst, abs(float(lr_uc_vec(h.sum(), n, a)) - m04.lr_uc(int(h.sum()), n, a)),
                    abs(float(lr_ind_vec(h)[0][0]) - m04.lr_ind(h.astype(bool))))
    ok &= check("lr_uc_vec / lr_ind_vec == 04's scalar functions", worst < 1e-9, f"maxdiff={worst:.2e}")

    # 2. Vectorized AS agrees with 04
    r = rng.standard_normal(357) * 0.01
    v = np.full(357, 0.01 * m04.var_z_n(0.05)); e = np.full(357, 0.01 * m04.es_z_n(0.05))
    z1, z2 = as_z_mat(r, v, e, 0.05)
    d = max(abs(z1[0] - m04.as_z1(r, v, e)), abs(z2[0] - m04.as_z2(r, v, e, 0.05)))
    ok &= check("as_z_mat == 04's as_z1/as_z2", d < 1e-12, f"maxdiff={d:.2e}")

    # 3. Du-Escanciano: mean and variance of H under H0
    a = 0.05; u = rng.random(2_000_000)
    H = (a - u) * (u <= a) / a
    ok &= check("DE: E[H]=a/2, Var[H]=a(1/3-a/4)",
                abs(H.mean() - a / 2) < 2e-4 and abs(H.var() - a * (1 / 3 - a / 4)) < 2e-4,
                f"mean={H.mean():.5f}/{a/2:.5f} var={H.var():.5f}/{a*(1/3-a/4):.5f}")

    # 4. Level: rejection rates of each test under the correct model ~5% (small B, sanity only)
    reps, n, a = 200, 357, 0.05
    rej = {k: 0 for k in ("cc", "dq", "ues", "z2")}
    for _ in range(reps):
        sd = np.full(n, 0.01); mu = np.zeros(n)
        rr = mu + sd * rng.standard_normal(n)
        fc = Forecast("normal", rr, mu + sd * m04.var_z_n(a), mu + sd * m04.es_z_n(a), mu, sd)
        o = run_tests(fc, a, rng, 199, 199)
        rej["cc"] += o["p_cc_mc"] < 0.05; rej["dq"] += o["p_dq_mc"] < 0.05
        rej["ues"] += o["p_ues_mc"] < 0.05; rej["z2"] += o["p_z2_mc"] < 0.05
    rates = {k: v / reps for k, v in rej.items()}
    ok &= check("Rejection rate of MC tests under the correct model ∈ [0.5%, 11%]",
                all(0.005 <= x <= 0.11 for x in rates.values()), str(rates))

    # 5. Power: true GARCH volatility + constant VaR (a typical clustered-breach case)
    rej_ind = rej_dq = 0
    for _ in range(100):
        s2, xs = 1.0, np.empty(n)
        for i in range(n):
            xs[i] = math.sqrt(s2) * rng.standard_normal(); s2 = 0.05 + 0.15 * xs[i] ** 2 + 0.80 * s2
        sd = np.full(n, xs.std()); mu = np.zeros(n)
        fc = Forecast("normal", xs, sd * m04.var_z_n(a), sd * m04.es_z_n(a), mu, sd)
        o = run_tests(fc, a, rng, 199, 99)
        rej_ind += o["p_ind_mc"] < 0.05; rej_dq += o["p_dq_mc"] < 0.05
    ok &= check("Under clustered breaches, DQ power > IND and > 30%",
                rej_dq / 100 > 0.30 and rej_dq >= rej_ind,
                f"IND={rej_ind/100:.2f} DQ={rej_dq/100:.2f}")

    print("\nSelf-test all passed." if ok else "\nSelf-test has failures.")
    return 0 if ok else 1


# ============================================================ Plotting ====
def make_figure(store, dates_out, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for i, nm in enumerate(("IBIT portfolio", "GLD portfolio")):
        for j, a in enumerate((0.05, 0.01)):
            ax = axes[i, j]
            g = store[(a, nm, "GARCH-t")]; h = store[(a, nm, "Historical simulation")]
            dt = g["dates"]
            ax.axvspan(np.datetime64(CRISIS[0]), np.datetime64(CRISIS[1]), color="0.85", lw=0)
            ax.plot(dt, g["r"] * 100, color="0.55", lw=0.6, label="Portfolio return")
            ax.plot(dt, g["v"] * 100, color="#1f5aa6", lw=1.1, label="GARCH-t VaR")
            ax.plot(h["dates"], h["v"] * 100, color="#c0392b", lw=1.0, ls="--", label="HS VaR")
            br = g["r"] < g["v"]
            ax.scatter(dt[br], g["r"][br] * 100, color="#1f5aa6", s=18, zorder=5,
                       label=f"GARCH-t breaches ({br.sum()})")
            ax.set_title(f"{PORT_EN[nm]} — {int((1-a)*100)}% VaR", fontsize=10)
            ax.set_ylabel("Daily return (%)")
            ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=True, framealpha=0.85)
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m"))
    fig.suptitle("Out-of-sample one-day VaR forecasts and breaches "
                 "(rolling 250-day window, refit every 20 days; shaded: April 2025)", fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_var_breaches.{ext}"), dpi=200)
    plt.close(fig)


# ============================================================ Main program ====
def fmt_p(p):
    return "—" if p is None or not np.isfinite(p) else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data/processed/returns__20260921.csv"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "out/18b_backtest"))
    ap.add_argument("--bvar", type=int, default=9999, help="MC draws for VaR tests")
    ap.add_argument("--bes", type=int, default=4999, help="MC draws for ES tests")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    m04, p04 = load_04()
    if args.selftest:
        sys.exit(selftest(m04))

    t0 = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    SEED = int(m04.SEED) + 18
    rng = np.random.default_rng(SEED)
    NS, RF, WIN, HSW, LAM = m04.N_START, m04.REFIT, m04.WINDOW, m04.HS_WIN, m04.EWMA_LAMBDA

    rule(); w("Section 0  Setup (identical conventions to 04)"); rule()
    d = pd.read_csv(args.data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(d)
    vm = (1 / 3) * d.GLD.std() / d.IBIT.std()
    spec = {"IBIT portfolio": {"IBIT": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3},
            "GLD portfolio": {"GLD": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3},
            "IBIT vol-matched": {"IBIT": vm, "SPY": (1 - vm) / 2, "TLT": (1 - vm) / 2},
            "60/40 benchmark": {"SPY": 0.60, "TLT": 0.40}}
    P = {nm: sum(d[k].values * wt for k, wt in s.items()) for nm, s in spec.items()}
    dates_out = d.Date.values[NS:]
    vix_out = d.VIX_Close.values[NS:]
    w(f"Forecast functions from: {os.path.basename(p04)}")
    w(f"Data: {os.path.relpath(args.data, HERE)}, {n} days; out-of-sample {n-NS} days "
      f"({pd.Timestamp(dates_out[0]).date()} → {pd.Timestamp(dates_out[-1]).date()})")
    w(f"n_start={NS}, refit={RF}, window={WIN}, HS window={HSW}, EWMA lambda={LAM}")
    w(f"MC draws: VaR tests B={args.bvar} (DQ uses min(B,4999)), ES tests B={args.bes}; seed {SEED}")

    # ---------------------------------------------------- Generate forecasts ----
    store, results, fc_rows = {}, [], []
    for nm, r in P.items():
        fcs = {dist: m04.roll_garch(r, dist, NS, RF, WIN) for dist in ("t", "normal")}
        W = np.stack([r[t - HSW:t] for t in range(NS, n)])
        ro = r[NS:]
        for a in ALPHAS:
            for mn in MODELS:
                if mn.startswith("GARCH"):
                    dist = "t" if mn == "GARCH-t" else "normal"
                    fc = fcs[dist]
                    v, e = m04.forecast_var_es(fc, dist, a)
                    mu, sd, nu, Wm, kind = fc["mu"], fc["sigma"], fc["nu"], None, dist
                elif mn == "Historical simulation":
                    v, e = m04.roll_hs(r, NS, a, HSW)
                    mu = sd = nu = None; Wm, kind = W, "hs"
                else:
                    v, e = m04.roll_ewma(r, NS, a, LAM)
                    sd = v / m04.var_z_n(a); mu = np.zeros_like(v); nu = None; Wm = None; kind = "ewma"
                ok = np.isfinite(v) & np.isfinite(e)
                sel = lambda z: None if z is None else z[ok]
                F = Forecast(kind, ro[ok], v[ok], e[ok], sel(mu), sel(sd), sel(nu),
                             None if Wm is None else Wm[ok])
                o = run_tests(F, a, rng, args.bvar, args.bes)
                cs = cluster_stats(o["hit"], dates_out[ok], vix_out[ok])
                o.update(cs); o.update(alpha=a, portfolio=nm, model=mn)
                results.append(o)
                store[(a, nm, mn)] = dict(r=ro[ok], v=v[ok], e=e[ok], dates=dates_out[ok])
                for i in range(ok.sum()):
                    fc_rows.append((str(pd.Timestamp(dates_out[ok][i]).date()), nm, mn, a,
                                    ro[ok][i], v[ok][i], e[ok][i], int(o["hit"][i]), o["u"][i]))
        w(f"  {nm:12s} done ({time.time()-t0:.0f}s)")

    R = pd.DataFrame([{k: v for k, v in o.items() if k not in ("u", "hit", "dates")} for o in results])

    # ------------------------------------------- Consistency with 04 ----
    w("")
    rule("-"); w("Consistency check against 04"); rule("-")
    ref04 = {(0.05, "IBIT portfolio", "GARCH-t"): 20, (0.05, "GLD portfolio", "GARCH-t"): 22,
             (0.01, "IBIT portfolio", "GARCH-t"): 6, (0.01, "GLD portfolio", "GARCH-t"): 9}
    got = {k: int(R[(R.alpha == k[0]) & (R.portfolio == k[1]) & (R.model == k[2])].x.iloc[0])
           for k in ref04}
    check("Breach counts match those reported in 04", got == ref04, str(got))
    par = R[R.model.isin(["GARCH-t", "GARCH-normal", "EWMA-normal"])].pit_agree.min()
    check("Parametric models: PIT<alpha agrees with r<VaR", par > 0.999, f"minimum agreement={par:.4f}")
    hs = R[R.model == "Historical simulation"].pit_agree.min()
    w(f"   Note: for historical simulation, the minimum agreement between PIT and the quantile-")
    w(f"         interpolation convention is {hs:.4f} (the empirical distribution is discrete; this is normal)")

    # ---------------------------------------------- Section 1 VaR ----
    w("")
    rule(); w("Section 1  VaR backtest: statistics + asymptotic p + Monte Carlo p"); rule()
    w("In parentheses: asymptotic p / MC p. At the 1% level the expected number of breaches "
      "is only 3.6, so take the MC p as authoritative.")
    w("The transition counts n00,n01,n10,n11 are the complete input to Christoffersen's "
      "independence test; the reviewer asked for \"all statistics to be reported\".")
    for a in ALPHAS:
        rows = []
        for _, o in R[R.alpha == a].iterrows():
            rows.append([o.portfolio, o.model, f"{o.x}/{o.n}", f"{o.rate:.2%}",
                         f"{o.n00},{o.n01},{o.n10},{o.n11}",
                         f"{o.uc:.2f} ({fmt_p(o.p_uc_asy)}/{fmt_p(o.p_uc_mc)})",
                         f"{o.ind:.2f} ({fmt_p(o.p_ind_asy)}/{fmt_p(o.p_ind_mc)})",
                         f"{o.cc:.2f} ({fmt_p(o.p_cc_asy)}/{fmt_p(o.p_cc_mc)})",
                         f"{o.dq:.2f} ({fmt_p(o.p_dq_asy)}/{fmt_p(o.p_dq_mc)})"])
        w(f"\n### alpha = {a:.0%} (nominal breaches {a*(n-NS):.1f})")
        w(md(rows, ["Portfolio", "Model", "Breaches", "Rate", "n00,n01,n10,n11",
                    "LR_uc", "LR_ind", "LR_cc", "DQ"]))

    # -------------------------------------------------- Output files ----
    R.to_csv(os.path.join(args.outdir, "backtest_results.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(fc_rows, columns=["Date", "portfolio", "model", "alpha", "ret", "VaR", "ES",
                                   "hit", "PIT"]).to_csv(
        os.path.join(args.outdir, "oos_forecasts.csv"), index=False, encoding="utf-8-sig")
    br_rows = [(o["alpha"], o["portfolio"], o["model"], dt) for o in results for dt in o["dates"]]
    pd.DataFrame(br_rows, columns=["alpha", "portfolio", "model", "breach_date"]).to_csv(
        os.path.join(args.outdir, "breach_dates.csv"), index=False, encoding="utf-8-sig")

    # English table for the paper
    en = ["# Table X. Out-of-sample VaR and ES backtests", "",
          f"Rolling window {NS} days, parameters re-estimated every {RF} days; "
          f"{n-NS} out-of-sample days ({pd.Timestamp(dates_out[0]).date()} to "
          f"{pd.Timestamp(dates_out[-1]).date()}). Entries are statistics with Monte Carlo p-values "
          f"in brackets (B = {args.bvar} for VaR tests, {args.bes} for ES tests). "
          "LR_uc: Kupiec (1995); LR_ind and LR_cc: Christoffersen (1998); "
          f"DQ: Engle and Manganelli (2004) with {DQ_LAGS} lagged hits and the VaR forecast; "
          "Z2: Acerbi and Szekely (2014), negative values indicate underestimated tail risk; "
          f"U_ES and C_ES({CES_LAGS}): Du and Escanciano (2017).", ""]
    for a in ALPHAS:
        en.append(f"## Panel: {int((1-a)*100)}% confidence level (expected breaches = {a*(n-NS):.1f})")
        en.append("")
        rows = []
        for _, o in R[R.alpha == a].iterrows():
            rows.append([PORT_EN[o.portfolio], MODEL_EN[o.model], o.x,
                         f"{o.uc:.2f} [{o.p_uc_mc:.3f}]", f"{o.ind:.2f} [{o.p_ind_mc:.3f}]",
                         f"{o.cc:.2f} [{o.p_cc_mc:.3f}]", f"{o.dq:.2f} [{o.p_dq_mc:.3f}]",
                         f"{o.z2:+.3f} [{o.p_z2:.3f}]", f"{o.ues:+.2f} [{o.p_ues_mc:.3f}]",
                         f"{o.ces:.2f} [{o.p_ces_mc:.3f}]"])
        en.append(md(rows, ["Portfolio", "Model", "Breaches", "LR_uc", "LR_ind", "LR_cc", "DQ",
                            "Z2", "U_ES", f"C_ES({CES_LAGS})"]))
    with open(os.path.join(args.outdir, "table_backtest_en.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(en))

    try:
        make_figure(store, dates_out, args.outdir)
        w("\nFigure written: fig_var_breaches.pdf / .png")
    except Exception as ex:                          # plotting failure does not affect results
        w(f"\nPlotting failed (does not affect test results): {ex}")

    w("")
    rule(); w("CHECK summary"); rule()
    for c in CHECKS:
        w("  " + c)
    w(f"\nElapsed {time.time()-t0:.0f}s. Output directory: {os.path.relpath(args.outdir, HERE)}")
    with open(os.path.join(args.outdir, "backtest_supplement.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))


if __name__ == "__main__":
    main()