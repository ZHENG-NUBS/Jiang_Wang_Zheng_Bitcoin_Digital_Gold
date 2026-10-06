"""
12_btc_proxy_validity.py —

Problem
    IBIT has only 607 days and 17 independent tail events, so the tail beta's MDE = 4.91
    and it is not identifiable. Can we use the long BTC spot sample (from 2014-09) to
    raise the number of tail events?

    The precondition is that BTC spot must be a valid proxy for IBIT, and that there is
    no structural break before and after. This script tests exactly those two things, and
    **empirically measures** how far the extended sample can lower the MDE — replacing a
    back-of-the-envelope estimate like "it should improve by about 2x" with a verifiable
    number.

Four tests
    A. Overlap-period consistency   IBIT vs BTC (from 2024-01-12), overall + tail days only
    B. Non-synchronous alignment    The BTC daily bar is stamped 00:00 UTC, about 3–4
                                    hours after the SPY close, so same-day alignment makes
                                    BTC contain extra post-close information. Compare
                                    same-day / lagged / leading.
    C. Structural break             Before vs after ETF launch, are the tail regression
                                    coefficients the same? (HAC-Wald / Chow)
    D. Empirical returns            Extended-sample tail event count, tail x spread,
                                    block-bootstrap MDE

Verdict (the script gives it automatically, not by eyeballing)
    A passes + C not significant  → the proxy holds up, do item 8
    C significant                 → do not do it; "ETF launch changed the tail behavior"
                                    is itself a publishable finding

Data preparation (you need to fetch a long-sample snapshot once on a machine with network
access)
    Copy 01a_fetch_raw.py to 01b_fetch_long.py and change three things:
        TICKERS    = {"BTC-USD": "...", "IBIT": "...", "SPY": "...",
                      "GLD": "...", "TLT": "...", "^VIX": "..."}
        START_DATE = "2014-09-01"
        AS_OF      = "20260921L"

Usage
    python 12_btc_proxy_validity.py --asof 20260921L
    python 12_btc_proxy_validity.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ETF_LAUNCH = pd.Timestamp("2024-01-11")   # IBIT launch date
QTAIL = 0.05
HAC_LAG = 5
BOOT_R = 4000
BOOT_L = 10
SEED = 20260921

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


# ============================================================ Statistical tools ====
def ols_hac(y, X, lag=HAC_LAG):
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        wt = 1.0 - l / (lag + 1.0)
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += wt * (A + A.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 0))
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    p = 2 * stats.norm.sf(np.abs(t))
    return b, se, t, p, u, V


def wald(b, V, R, q=None):
    """Test R b = q. Returns (chi-square statistic, degrees of freedom, p)."""
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.atleast_1d(q)
    d = R @ b - q
    M = R @ V @ R.T
    stat = float(d @ np.linalg.pinv(M) @ d)
    df = int(np.linalg.matrix_rank(M))
    return stat, df, float(stats.chi2.sf(stat, df))


def stationary_boot(n, R, L, rng):
    """Stationary block bootstrap (Politis–Romano, geometric block lengths).
    Returns an (n, R) index matrix."""
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    start = rng.integers(0, n, R)
    idx[0] = start
    newblk = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newblk[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def tail_beta_free(y, x, q=QTAIL):
    """Tail-subsample regression slope with a free intercept (honest specification)."""
    thr = np.quantile(x, q)
    m = x <= thr
    if m.sum() < 5:
        return np.nan
    xx, yy = x[m], y[m]
    return float(np.cov(yy, xx, ddof=1)[0, 1] / np.var(xx, ddof=1))


def n_episodes(mask, gap=5):
    """Collapse adjacent observations into independent events (a gap of more than
    `gap` trading days starts a new event)."""
    i = np.where(mask)[0]
    if len(i) == 0:
        return 0
    return 1 + int((np.diff(i) > gap).sum())


def mde_of(y, x, rng, q=QTAIL, R=BOOT_R, L=BOOT_L):
    """Block-bootstrap standard error of the tail beta and the MDE (80% power, 5% level)."""
    n = len(x)
    idx = stationary_boot(n, R, L, rng)
    bs = []
    for c in range(R):
        i = idx[:, c]
        b = tail_beta_free(y[i], x[i], q)      # recompute the quantile threshold on each resample
        if np.isfinite(b):
            bs.append(b)
    bs = np.array(bs)
    se = float(bs.std(ddof=1))
    return se, 2.80 * se, bs


# ============================================================== Loading ====
def load_raw(raw_dir, asof, ticker):
    fn = ticker.replace("^", "IDX_").replace("-", "_")
    for cand in (f"{fn}__{asof}.csv", f"{ticker}__{asof}.csv"):
        p = os.path.join(raw_dir, cand)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            col = "AdjClose" if "AdjClose" in d.columns else "Close"
            return d.set_index("Date")[col].rename(ticker)
    return None


# ============================================================ Self-test ====
def selftest() -> int:
    print("=" * 78)
    print("Self-test (synthetic data, validates the statistics, no real snapshot needed)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    ok = True

    # 1. Nominal level of the Wald test
    rej = 0
    for _ in range(400):
        n = 800
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        y = X @ np.array([0.0, 1.0]) + rng.standard_normal(n)
        b, se, t, p, u, V = ols_hac(y, X)
        _, _, pv = wald(b, V, np.array([0.0, 1.0]), np.array([1.0]))
        rej += int(pv < 0.05)
    ok &= check("HAC-Wald nominal level ~5%", 0.01 <= rej / 400 <= 0.12, f"rejection rate={rej/400:.3f}")

    # 2. Wald has power against a real break
    rej = 0
    for _ in range(200):
        n = 800
        d = (np.arange(n) >= 400).astype(float)
        x = rng.standard_normal(n)
        y = 1.0 * x + 0.8 * d * x + rng.standard_normal(n)   # slope +0.8 after the break
        X = np.column_stack([np.ones(n), x, d, d * x])
        b, se, t, p, u, V = ols_hac(y, X)
        _, _, pv = wald(b, V, np.array([0, 0, 0, 1.0]))
        rej += int(pv < 0.05)
    ok &= check("HAC-Wald has power against a real break", rej / 200 > 0.8, f"power={rej/200:.3f}")

    # 3. Event collapsing
    m = np.zeros(100, bool); m[[1, 2, 3, 40, 41, 80]] = True
    ok &= check("Independent-event collapsing (gap>5)", n_episodes(m) == 3, f"got {n_episodes(m)}")

    # 4. Tail beta recovers the true value
    n = 6000
    x = rng.standard_normal(n) * 0.01
    y = 1.5 * x + rng.standard_normal(n) * 0.005
    b = tail_beta_free(y, x)
    ok &= check("Tail beta recovers true value 1.5", abs(b - 1.5) < 0.25, f"got {b:.3f}")

    # 5. MDE shrinks as 1/sqrt(n) with sample size
    se1, m1, _ = mde_of(y[:1000], x[:1000], np.random.default_rng(1), R=600)
    se2, m2, _ = mde_of(y, x, np.random.default_rng(1), R=600)
    ratio = m1 / m2
    ok &= check("MDE shrinks as ~1/sqrt(n) (6x sample should be about 2.4x)",
                1.5 <= ratio <= 4.0, f"measured {ratio:.2f}x")

    print()
    print("Self-test all passed." if ok else "Self-test has failures.")
    return 0 if ok else 1


# ============================================================== Main program ====
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default="20260921L")
    ap.add_argument("--outdir", default="out/12_btc_proxy")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    rng = np.random.default_rng(SEED)
    os.makedirs(a.outdir, exist_ok=True)

    # -------------------------------------------------------- Loading ----
    rule(); w("Section 0  Data"); rule()
    need = ["BTC-USD", "IBIT", "SPY"]
    opt = ["GLD", "TLT", "^VIX"]
    S = {}
    miss = []
    for t in need + opt:
        s = load_raw(a.raw, a.asof, t)
        if s is None:
            (miss if t in need else []).append(t)
        else:
            S[t] = s
    if miss:
        w(f"Missing required tickers: {miss}")
        w(f"Please first run 01a_fetch_raw.py with the long-sample snapshot parameters "
          f"(AS_OF={a.asof}, START_DATE=2014-09-01, TICKERS including BTC-USD). "
          "See the note at the top of the script.")
        sys.exit(1)

    spy_cal = S["SPY"].index                       # Use SPY trading days as the base calendar
    px = pd.DataFrame({k: v for k, v in S.items()}).reindex(spy_cal)
    w(f"SPY trading days: {len(spy_cal)}, {spy_cal.min().date()} → {spy_cal.max().date()}")
    for k in px.columns:
        v = px[k].dropna()
        w(f"  {k:8s} available {len(v):5d} days  {v.index.min().date()} → {v.index.max().date()}")

    R = px.pct_change()
    w("")
    w("Note: BTC-USD is sampled on SPY trading days (weekend and US-holiday BTC moves are")
    w("      folded into the next trading day's return interval). This is standard in the")
    w("      literature, but it introduces session mismatch; see Section B.")

    # ============================================ A. Overlap-period consistency ====
    w("")
    rule(); w("Section A  Overlap-period consistency: IBIT vs BTC spot"); rule()
    ov = R[["IBIT", "BTC-USD", "SPY"]].dropna()
    w(f"Overlap period: {len(ov)} days, {ov.index.min().date()} → {ov.index.max().date()}")
    yi, xb, xs = ov["IBIT"].values, ov["BTC-USD"].values, ov["SPY"].values

    X = np.column_stack([np.ones(len(xb)), xb])
    b, se, t, p, u, V = ols_hac(yi, X)
    r2 = 1 - u.var() / yi.var()
    te = u.std(ddof=2) * math.sqrt(252)
    _, _, p_b1 = wald(b, V, np.array([0.0, 1.0]), np.array([1.0]))

    rows = [["Correlation", f"{np.corrcoef(yi, xb)[0,1]:.4f}", "Closer to 1 is better"],
            ["Regression slope b", f"{b[1]:.4f}", f"p-value for testing b=1: {p_b1:.4f}"],
            ["R²", f"{r2:.4f}", "—"],
            ["Annualized tracking error", f"{te*100:.2f}%", "Annualized std of residuals"],
            ["Annualized return difference (IBIT−BTC)",
             f"{(np.mean(yi)-np.mean(xb))*252*100:+.2f}%", "Should be about −0.25% management fee"]]
    w(md(rows, ["Metric", "Value", "Note"]))

    # Look at tail days separately — this is the key point, since the proxy is meant to
    # estimate tail behavior, and overall consistency does not imply tail consistency
    thr = np.quantile(xs, QTAIL)
    m = xs <= thr
    w(f"**Looking only at SPY's 5% tail days (n={m.sum()})** — the proxy will be used in "
      "the tail, and overall consistency does not imply tail consistency:\n")
    Xt = np.column_stack([np.ones(m.sum()), xb[m]])
    bt, set_, tt, pt, ut, Vt = ols_hac(yi[m], Xt, lag=2)
    _, _, p_bt1 = wald(bt, Vt, np.array([0.0, 1.0]), np.array([1.0]))
    rows = [["Tail correlation", f"{np.corrcoef(yi[m], xb[m])[0,1]:.4f}"],
            ["Tail regression slope", f"{bt[1]:.4f} (p-value for testing =1: {p_bt1:.4f})"],
            ["Tail mean difference (IBIT−BTC)", f"{np.mean(yi[m]-xb[m])*100:+.3f} pp/day"]]
    w(md(rows, ["Metric", "Value"]))

    okA = (np.corrcoef(yi, xb)[0, 1] >= 0.95) and (p_b1 > 0.05) and \
          (np.corrcoef(yi[m], xb[m])[0, 1] >= 0.90)
    check("A. Proxy consistency (overall ρ≥0.95 and b not significantly different from 1 and tail ρ≥0.90)", okA)

    # ------------------------------------------- A2. Multi-day horizon diagnostic ----
    # If A does not pass, the first thing to determine is whether this is [session
    # mismatch] or [genuine divergence].
    # BTC's Yahoo daily bar closes at 00:00 UTC, about 3–4 hours after IBIT's 16:00 ET
    # close; this mismatch is [intraday], so a whole-day shift cannot remove it (see
    # Section B), but if the holding period is stretched to k days, the mismatch share
    # falls to about (3/24)/k, and the correlation should rise monotonically with k.
    # If ρ clearly recovers at k=5 → the mismatch explanation works, BTC is still a valid
    # proxy, only noisy at daily frequency; if ρ does not recover → it is genuine
    # divergence, and the proxy does not hold.
    w("")
    w("**A2. Multi-day holding-period diagnostic — distinguishing [session mismatch] from [genuine divergence]**\n")
    w("BTC's daily bar closes at 00:00 UTC and IBIT's at 16:00 ET, a mismatch of about 3–4 "
      "hours, which is an [intraday] mismatch and cannot be removed by a whole-day shift. "
      "But if the holding period is stretched to k days, the mismatch share falls to about "
      "(3/24)/k; if ρ rises monotonically with k, the low daily ρ is due to the mismatch "
      "rather than genuine divergence.\n")
    pi = (1 + pd.Series(yi, index=ov.index))
    pb = (1 + pd.Series(xb, index=ov.index))
    rows = []
    for k in (1, 2, 3, 5, 10, 21):
        ri = pi.rolling(k).apply(np.prod, raw=True) - 1
        rb = pb.rolling(k).apply(np.prod, raw=True) - 1
        z = pd.concat([ri, rb], axis=1).dropna()
        zi, zb = z.iloc[:, 0].values[::k], z.iloc[:, 1].values[::k]   # non-overlapping
        if len(zi) < 10:
            continue
        bb = np.cov(zi, zb, ddof=1)[0, 1] / np.var(zb, ddof=1)
        rows.append([f"{k} days", len(zi), f"{np.corrcoef(zi, zb)[0,1]:.4f}", f"{bb:.4f}"])
    w(md(rows, ["Holding period", "Non-overlapping obs", "ρ(IBIT, BTC)", "Slope"]))
    rho5 = float([r[2] for r in rows if r[0] == "5 days"][0])
    okA2 = rho5 >= 0.95
    check("A2. 5-day holding-period ρ≥0.95 (session mismatch can explain the low daily value)", okA2, f"ρ_5d={rho5:.4f}")

    # ============================================ B. Non-synchronous alignment ====
    w("")
    rule(); w("Section B  Non-synchronous trading-day alignment"); rule()
    w("BTC-USD's Yahoo daily bar is stamped 00:00 UTC, corresponding to about 19:00–20:00 "
      "US Eastern, 3–4 hours after SPY's 16:00 close. Same-day alignment lets BTC contain "
      "extra post-close information, thereby **overstating** BTC's contemporaneous beta to "
      "SPY. The table below compares three alignments:\n")
    rows = []
    for lab, sh in (("BTC lagged 1 day", 1), ("Same day (default)", 0), ("BTC leading 1 day", -1)):
        bb = pd.Series(xb, index=ov.index).shift(sh).values
        ok2 = ~np.isnan(bb)
        rho_i = np.corrcoef(yi[ok2], bb[ok2])[0, 1]
        Xs = np.column_stack([np.ones(ok2.sum()), xs[ok2]])
        bs_, _, _, _, _, _ = ols_hac(bb[ok2], Xs)
        rows.append([lab, f"{rho_i:.4f}", f"{bs_[1]:+.4f}"])
    w(md(rows, ["Alignment", "Correlation with IBIT", "BTC beta to SPY"]))
    w("If the \"same day\" correlation is the highest, the session mismatch is acceptable "
      "(BTC and IBIT are still the same day's moves); if BTC's beta to SPY is clearly higher "
      "under the same-day convention than under the lagged convention, the difference is the "
      "contribution of post-close information, which should be honestly reported in the main "
      "text, with the lagged convention as a robustness check.")

    # ============================================ C. Structural break ====
    w("")
    rule(); w("Section C  Structural break: is tail behavior the same before and after ETF launch"); rule()
    bt_all = R[["BTC-USD", "SPY"]].dropna()
    xs2, yb = bt_all["SPY"].values, bt_all["BTC-USD"].values
    post = np.asarray(bt_all.index >= ETF_LAUNCH, dtype=float)
    w(f"Pre-ETF period: {int((1-post).sum())} days, ETF period: {int(post.sum())} days, "
      f"break at {ETF_LAUNCH.date()} (known break → use Chow / HAC-Wald, not sup-Wald)")

    # C1 Break in the unconditional beta
    X1 = np.column_stack([np.ones(len(xs2)), xs2, post, post * xs2])
    b1, se1, t1, p1, u1, V1 = ols_hac(yb, X1)
    s_unc, df_unc, p_unc = wald(b1, V1, np.array([0, 0, 0, 1.0]))
    # C2 Break in the cumulative tail coefficients (Baur–McDermott specification + post-launch interaction)
    q10, q05 = np.quantile(xs2, 0.10), np.quantile(xs2, 0.05)
    d10, d05 = (xs2 <= q10).astype(float), (xs2 <= q05).astype(float)
    X2 = np.column_stack([np.ones(len(xs2)), xs2, xs2 * d10, xs2 * d05,
                          post, post * xs2, post * xs2 * d10, post * xs2 * d05])
    b2, se2, t2, p2, u2, V2 = ols_hac(yb, X2)
    Rm = np.zeros((3, X2.shape[1])); Rm[0, 5] = Rm[1, 6] = Rm[2, 7] = 1.0
    s_tail, df_tail, p_tail = wald(b2, V2, Rm)
    # C3 Break in volatility
    v_pre = yb[post == 0].std(ddof=1) * math.sqrt(252)
    v_post = yb[post == 1].std(ddof=1) * math.sqrt(252)

    rows = [["Unconditional beta change post-listing", f"{b1[3]:+.4f}", f"{s_unc:.2f}", str(df_unc), f"{p_unc:.4f}",
             "Significant" if p_unc < .05 else "Not significant"],
            ["Joint invariance of the three tail coefficients", "—", f"{s_tail:.2f}", str(df_tail), f"{p_tail:.4f}",
             "Significant" if p_tail < .05 else "Not significant"],
            ["Annualized volatility", f"{v_pre*100:.1f}% → {v_post*100:.1f}%", "—", "—", "—", "—"]]
    w(md(rows, ["Test", "Change", "Chi-square", "df", "p", "Verdict"]))
    okC = (p_unc > 0.05) and (p_tail > 0.05)
    check("C. No significant structural break (both p > 0.05)", okC)

    # ============================================ D. Empirical payoff ====
    w("")
    rule(); w("Section D  How much the extended sample actually buys: measured MDE"); rule()
    rows = []
    for lab, sub in (("ETF period only (current situation, BTC as IBIT proxy)", bt_all.index >= ETF_LAUNCH),
                     ("Full sample (from 2014)", np.ones(len(bt_all), bool))):
        x_, y_ = xs2[sub], yb[sub]
        thr_ = np.quantile(x_, QTAIL)
        mm = x_ <= thr_
        se_, mde_, _ = mde_of(y_, x_, np.random.default_rng(SEED))
        rows.append([lab, len(x_), int(mm.sum()), n_episodes(mm),
                     f"{x_[mm].std(ddof=1)*100:.3f}",
                     f"{(x_[mm].max()-x_[mm].min())*100:.2f}",
                     f"{se_:.3f}", f"{mde_:.2f}"])
    w(md(rows, ["Sample", "Trading days", "Tail obs", "Independent events", "Tail SD(x) pp",
                "Tail spread pp", "Bootstrap SE", "MDE"]))
    mde_now = float(rows[0][7]); mde_ext = float(rows[1][7])
    w(f"Measured improvement **{mde_now/mde_ext:.2f}x** (my earlier estimate was about 2.7x; "
      "this table is authoritative).\n")
    w("Criterion: MDE < 1 is the minimum to distinguish \"safe haven β<0\" from \"amplifier "
      "β>1\"; MDE < 0.4 is the minimum to give a credible point-estimate magnitude.")
    okD = mde_ext < 1.0
    check("D. MDE < 1 after extending the sample (enough to distinguish safe haven from amplifier)", okD, f"MDE={mde_ext:.2f}")

    # ============================================ Overall verdict ====
    w("")
    rule(); w("Overall verdict"); rule()
    # Whether the proxy is usable falls into three tiers. The key econometric fact: in
    # route A, BTC is the [dependent variable] (regress BTC on SPY to get the tail beta),
    # not the explanatory variable. Classical measurement error in a dependent variable
    # [does not bias the slope], it only inflates the residual variance and hence the MDE.
    # So a proxy that is "noisy daily but consistent at multi-day horizons" is usable, at
    # the cost of a worse MDE; only genuine divergence makes it unusable.
    proxy_strong = okA
    proxy_usable = okA or (okA2 and p_bt1 > 0.05)
    if not okA:
        w("**On the interpretation of the Section A FAIL**\n")
        if proxy_usable:
            w(f"> The daily ρ = {np.corrcoef(yi, xb)[0,1]:.3f} does not reach 0.95, but the "
              f"5-day horizon recovers to {rho5:.3f}, and the tail slope is not significantly "
              f"different from 1 (p = {p_bt1:.3f}).")
            w("> This is consistent with [session mismatch] rather than [genuine divergence]: "
              "BTC's daily bar closes at 00:00 UTC and IBIT's at 16:00 ET, a 3–4 hour gap.\n")
            w("> **This is not fatal for route A.** In route A, BTC is the dependent variable "
              "(regressing BTC on SPY), and classical measurement error in the dependent "
              "variable does not bias the slope, it only inflates the residual variance.")
            w("> The cost shows up in the MDE, and the MDE is exactly what Section D measures "
              "— so just read Section D; no separate discount is needed.")
            w("> **But BTC cannot be used as an explanatory variable** (for example, treating "
              "BTC as a factor to explain other assets); in that setting the same measurement "
              "error would cause attenuation bias toward zero.\n")
        else:
            w(f"> Daily ρ = {np.corrcoef(yi, xb)[0,1]:.3f}, 5-day horizon {rho5:.3f}, and "
              "lengthening the holding period shows no recovery → this is not session "
              "mismatch, it is genuine divergence.\n")

    if proxy_usable and okC:
        w("> **Proxy is usable and there is no structural break → recommend doing item 8.**")
        if not proxy_strong:
            w("> (The proxy is in the tier of \"noisy daily but no systematic deviation\"; "
              "see above. The main text must state this honestly.)")
        w("")
        if okD:
            w(f"> After extension, MDE = {mde_ext:.2f} < 1, so the directional conclusion on "
              "the tail beta is identifiable.")
        else:
            w(f"> But after extension, MDE = {mde_ext:.2f} is still ≥ 1. The direction of the "
              "tail beta still cannot be reliably determined,")
            w("> and the payoff is mainly in [having more stress episodes] (2018Q4, 2020-03, "
              "2022, 2023-03),")
            w("> rather than in identifying the tail beta. Item 8 is still worth doing, but "
              "the goal should be reframed as an event study,")
            w("> and you should no longer claim that the tail beta can be estimated.")
    elif not okC:
        w("> **A structural break was detected → do not merge the samples.**\n")
        w("> The tail behavior differs between the pre-ETF and ETF periods, so pooling the "
          "two segments to estimate a common parameter is wrong.")
        w("> But this is itself a publishable finding: the ETF launch (institutional inflows) "
          "changed Bitcoin's tail behavior.")
        w("> And it directly answers the reviewer's question \"why not use a longer BTC "
          "history\" — there is a test result to cite.")
    else:
        w("> **Proxy consistency does not meet the bar (and it is not explainable by session "
          "mismatch) → do not use BTC spot in place of IBIT.**\n")
        w("> The difference between the two during the overlap period is already too large, "
          "and parameters estimated from the long sample cannot be extrapolated to IBIT.")

    w("")
    for c in CHECKS:
        w("  " + c)

    p = os.path.join(a.outdir, "btc_proxy_validity.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\nWritten {p}")


if __name__ == "__main__":
    main()