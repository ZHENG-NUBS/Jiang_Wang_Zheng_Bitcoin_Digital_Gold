"""
13_break_robustness.py — Robustness of the structural break

Background
    12_btc_proxy_validity.py found: before vs after ETF launch, the joint test of BTC's
    three tail coefficients gives chi2=17.96, df=3, p=0.0004 (significant); while the
    unconditional beta is not significant (p=0.109). Annualized volatility over the same
    period fell from 70.0% to 49.0%.

    Problem: a halving of volatility can itself **mechanically** cause tail coefficients
    to change. The tail slope b = rho * sigma_y/sigma_x; if sigma_y halves and rho is
    unchanged, b also halves. So we must distinguish whether what broke is the **tail
    dependence structure rho** or merely the **volatility level**.

Three tests
    1. Redo the break test after volatility standardization
       Standardize each side by its own EWMA conditional volatility (z = r / sigma_ewma);
       the standardized slope is the dependence structure itself (b_std = rho), independent
       of scale.
         · Still significant after standardization → the tail dependence really changed,
           and Finding 1 holds
         · Disappears after standardization → the "break" is only a change in volatility
           level, and Finding 1 must be substantially weakened

    2. Event-by-event jackknife
       The ETF-period tail has only 17 independent events, and the 2025-04 single event is
       known to drive the global conclusion. Drop each ETF-period stress event in turn and
       redo the break test, to confirm the break is not a disguise for
       "2025-04 vs everything else".

    3. Placebo breaks
       Set a series of fake breaks **inside the pre-ETF period** (both sides are pre-ETF,
       so there should be no break), and look at the rejection rate. If the placebo
       rejection rate is far above 5%, the test itself over-rejects, and the real break's
       p=0.0004 is not worth as much.

Dependencies: numpy, pandas, scipy
Usage:
    python 13_break_robustness.py --asof 20260921L
    python 13_break_robustness.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ETF_LAUNCH = pd.Timestamp("2024-01-11")
QS = (0.10, 0.05)
HAC_LAG = 5
EWMA_LAM = 0.94
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
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        wt = 1.0 - l / (lag + 1.0)
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += wt * (A + A.T)
    V = XtXi @ S @ XtXi
    return b, V


def wald(b, V, R, q=None):
    R = np.atleast_2d(R)
    q = np.zeros(R.shape[0]) if q is None else np.atleast_1d(q)
    d = R @ b - q
    M = R @ V @ R.T
    stat = float(d @ np.linalg.pinv(M) @ d)
    df = int(np.linalg.matrix_rank(M))
    return stat, df, float(stats.chi2.sf(stat, df))


def ewma_vol(r, lam=EWMA_LAM, warm=100):
    """EWMA conditional volatility. Uses only information up to t-1, so it is a filter,
    not an estimate, and therefore is not contaminated by in-sample structural breaks."""
    n = len(r)
    s2 = np.empty(n)
    v = float(np.var(r[:min(warm, n)], ddof=1))
    for t in range(n):
        s2[t] = v
        v = lam * v + (1 - lam) * r[t] ** 2
    return np.sqrt(s2)


def break_test(y, x, post, qs=QS):
    """
    Baur–McDermott tail specification + post-listing interactions; tests whether the
    three tail interactions are jointly zero.
    Returns (chi2, df, p, coefficient vector).
    """
    dums = [(x <= np.quantile(x, q)).astype(float) for q in qs]
    cols = [np.ones(len(x)), x] + [x * d for d in dums]
    cols += [post] + [post * c for c in cols[1:]]
    X = np.column_stack(cols)
    b, V = ols_hac(y, X)
    k = 2 + len(qs)                 # number of base terms: constant, x, x*d10, x*d05
    R = np.zeros((len(qs) + 1, X.shape[1]))
    for j in range(len(qs) + 1):    # post*x, post*x*d10, post*x*d05
        R[j, k + 1 + j] = 1.0
    s, df, p = wald(b, V, R)
    return s, df, p, b


def n_episodes_idx(mask, gap=5):
    """Returns a list of index arrays, one per independent event."""
    i = np.where(mask)[0]
    if len(i) == 0:
        return []
    out, cur = [], [i[0]]
    for a, b_ in zip(i[:-1], i[1:]):
        if b_ - a > gap:
            out.append(np.array(cur)); cur = [b_]
        else:
            cur.append(b_)
    out.append(np.array(cur))
    return out


# ============================================================== Self-test ====
def selftest() -> int:
    print("=" * 78)
    print("Self-test (synthetic data)")
    print("=" * 78)
    rng = np.random.default_rng(0)
    ok = True
    n = 3000
    post = (np.arange(n) >= 2000).astype(float)

    # 1. Pure scale break: the raw test should reject, but not after standardization
    rej_raw = rej_std = 0
    for _ in range(120):
        x = rng.standard_normal(n) * 0.01
        scale = np.where(post == 1, 0.5, 1.0)          # y's volatility halves after the break
        y = (1.2 * x + rng.standard_normal(n) * 0.02) * scale
        _, _, p_raw, _ = break_test(y, x, post)
        zy, zx = y / ewma_vol(y), x / ewma_vol(x)
        _, _, p_std, _ = break_test(zy, zx, post)
        rej_raw += int(p_raw < .05); rej_std += int(p_std < .05)
    ok &= check("Pure scale break: raw test rejects", rej_raw / 120 > 0.5, f"rejection rate={rej_raw/120:.2f}")
    ok &= check("Pure scale break: no longer rejects after standardization", rej_std / 120 < 0.20, f"rejection rate={rej_std/120:.2f}")

    # 2. Real dependence break: should still reject after standardization
    rej_std2 = 0
    for _ in range(120):
        x = rng.standard_normal(n) * 0.01
        beta = np.where(post == 1, 2.4, 1.2)           # the dependence structure really changed
        y = beta * x + rng.standard_normal(n) * 0.02
        zy, zx = y / ewma_vol(y), x / ewma_vol(x)
        _, _, p_std, _ = break_test(zy, zx, post)
        rej_std2 += int(p_std < .05)
    ok &= check("Real dependence break: still rejects after standardization", rej_std2 / 120 > 0.5, f"power={rej_std2/120:.2f}")

    # 3. Nominal level when there is no break
    rej0 = 0
    for _ in range(200):
        x = rng.standard_normal(n) * 0.01
        y = 1.2 * x + rng.standard_normal(n) * 0.02
        _, _, p0, _ = break_test(y, x, post)
        rej0 += int(p0 < .05)
    ok &= check("Nominal level ~5% when there is no break", 0.01 <= rej0 / 200 <= 0.15, f"rejection rate={rej0/200:.3f}")

    # 4. Event grouping
    m = np.zeros(100, bool); m[[1, 2, 3, 40, 41, 80]] = True
    ok &= check("Event group count = 3", len(n_episodes_idx(m)) == 3, f"got {len(n_episodes_idx(m))}")

    print()
    print("Self-test all passed." if ok else "Self-test has failures.")
    return 0 if ok else 1


# ============================================================== Main program ====
def load_raw(raw_dir, asof, ticker):
    fn = ticker.replace("^", "IDX_").replace("-", "_")
    for cand in (f"{fn}__{asof}.csv", f"{ticker}__{asof}.csv"):
        p = os.path.join(raw_dir, cand)
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date")
            col = "AdjClose" if "AdjClose" in d.columns else "Close"
            return d.set_index("Date")[col].rename(ticker)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default="20260921L")
    ap.add_argument("--outdir", default="out/13_break")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    os.makedirs(a.outdir, exist_ok=True)

    rule(); w("Section 0  Data"); rule()
    S = {t: load_raw(a.raw, a.asof, t) for t in ("BTC-USD", "SPY")}
    if any(v is None for v in S.values()):
        w(f"Missing long-sample snapshot; please first run 01b_fetch_long.py (AS_OF={a.asof}).")
        sys.exit(1)
    px = pd.DataFrame(S).reindex(S["SPY"].index)
    R = px.pct_change().dropna()
    dates = R.index
    y, x = R["BTC-USD"].values, R["SPY"].values
    post = np.asarray(dates >= ETF_LAUNCH, dtype=float)
    w(f"{len(R)} trading days, {dates.min().date()} → {dates.max().date()}")
    w(f"Pre-ETF period: {int((1-post).sum())} days, ETF period: {int(post.sum())} days")

    # =================================== 1. Volatility standardization ====
    w("")
    rule(); w("Section 1  After volatility standardization, is the break still there?"); rule()
    w("b = rho * sigma_y/sigma_x. If sigma_y halves while rho is unchanged, b also halves.")
    w("After standardizing each side by its EWMA(0.94) conditional volatility, the slope is "
      "rho itself, independent of scale.\n")
    v_pre = y[post == 0].std(ddof=1) * math.sqrt(252)
    v_post = y[post == 1].std(ddof=1) * math.sqrt(252)
    vx_pre = x[post == 0].std(ddof=1) * math.sqrt(252)
    vx_post = x[post == 1].std(ddof=1) * math.sqrt(252)
    w(md([["BTC annualized volatility", f"{v_pre*100:.1f}%", f"{v_post*100:.1f}%", f"{v_post/v_pre:.3f}"],
          ["SPY annualized volatility", f"{vx_pre*100:.1f}%", f"{vx_post*100:.1f}%", f"{vx_post/vx_pre:.3f}"],
          ["sigma_BTC/sigma_SPY", f"{v_pre/vx_pre:.2f}", f"{v_post/vx_post:.2f}",
           f"{(v_post/vx_post)/(v_pre/vx_pre):.3f}"]],
         ["Quantity", "Pre-ETF period", "ETF period", "Ratio"]))

    zy, zx = y / ewma_vol(y), x / ewma_vol(x)
    rows = []
    for lab, yy, xx in (("Raw returns (the version in 12)", y, x),
                        ("EWMA conditional-volatility standardization", zy, zx)):
        s, df, p, _ = break_test(yy, xx, post)
        rows.append([lab, f"{s:.2f}", str(df), f"{p:.4f}",
                     "**Significant**" if p < .05 else "Not significant"])
    # One more version: divide only by each period's unconditional volatility (cruder, but
    # does not depend on the EWMA setting)
    uy = y / np.where(post == 1, y[post == 1].std(ddof=1), y[post == 0].std(ddof=1))
    ux = x / np.where(post == 1, x[post == 1].std(ddof=1), x[post == 0].std(ddof=1))
    s, df, p, _ = break_test(uy, ux, post)
    rows.append(["Unconditional volatility standardization per period", f"{s:.2f}", str(df), f"{p:.4f}",
                 "**Significant**" if p < .05 else "Not significant"])
    w(md(rows, ["Version", "chi2", "df", "p", "Verdict"]))
    p_std = float(rows[1][3])
    p_unc = float(rows[2][3])
    ok_std = (p_std < 0.05) and (p_unc < 0.05)
    check("1. Break still significant after standardization (both standardization versions p<0.05)", ok_std,
          f"EWMA p={p_std:.4f}, unconditional p={p_unc:.4f}")

    # =================================== 2. Event-by-event jackknife ====
    w("")
    rule(); w("Section 2  Event-by-event jackknife: is the break held up by a single event?"); rule()
    thr = np.quantile(x, 0.05)
    tail_post = (x <= thr) & (post == 1)
    eps = n_episodes_idx(tail_post)
    w(f"The ETF period's 5% tail has {len(eps)} independent events in total. Dropping each "
      "in turn and redoing the break test")
    w("(using the EWMA-standardized version — if Section 1 already rejects it, this section "
      "is for reference only):\n")
    s0, _, p0, _ = break_test(zy, zx, post)
    rows = [["(full sample)", "—", "—", f"{s0:.2f}", f"{p0:.4f}", "—"]]
    worst_p = p0
    for e in eps:
        keep = np.ones(len(x), bool); keep[e] = False
        s, df, p, _ = break_test(zy[keep], zx[keep], post[keep])
        worst_p = max(worst_p, p)
        rows.append([f"{dates[e[0]].date()}", f"{len(e)}",
                     f"{x[e].min()*100:.2f}%", f"{s:.2f}", f"{p:.4f}",
                     "Still significant" if p < .05 else "**Becomes not significant**"])
    w(md(rows, ["Event dropped (start date)", "Days", "Deepest SPY single day", "chi2", "p", "Verdict"]))
    ok_jk = worst_p < 0.05
    check("2. Break still significant after dropping any single event", ok_jk, f"worst p={worst_p:.4f}")

    # =================================== 3. Placebo breaks ====
    w("")
    rule(); w("Section 3  Placebo breaks: does this test itself reject too often?"); rule()
    w("Set fake breaks **inside the pre-ETF period** (both sides are pre-ETF, so there should "
      "be no break).")
    w("If the rejection rate is far above 5%, the test over-rejects and the real break's "
      "p-value must be discounted.\n")
    pre_idx = np.where(post == 0)[0]
    lo, hi = pre_idx[250], pre_idx[-250]
    fake_dates = pd.date_range(dates[lo], dates[hi], freq="91D")
    prs = []
    for fd in fake_dates:
        pf = np.asarray(dates >= fd, dtype=float)
        sub = post == 0                       # use only the pre-ETF period to avoid mixing in the real break
        if pf[sub].sum() < 250 or (1 - pf[sub]).sum() < 250:
            continue
        s, df, p, _ = break_test(zy[sub], zx[sub], pf[sub])
        prs.append((fd.date(), s, p))
    rej = sum(1 for _, _, p in prs if p < .05)
    w(md([[str(d), f"{s:.2f}", f"{p:.4f}", "Reject" if p < .05 else ""]
          for d, s, p in prs],
         ["Fake break date", "chi2", "p", ""]))
    w(f"**Out of {len(prs)} placebo breaks, {rej} rejected, rejection rate {rej/max(len(prs),1):.1%}**"
      f" (nominal 5%).\n")
    ok_pb = (rej / max(len(prs), 1)) <= 0.20
    check("3. Placebo rejection rate ≤20% (test does not severely over-reject)", ok_pb,
          f"{rej}/{len(prs)}")

    # =================================== Overall verdict ====
    w("")
    rule(); w("Overall verdict: can Finding 1 be written?"); rule()
    if ok_std and ok_jk and ok_pb:
        w("> **All three pass → Finding 1 holds and can be written into the paper.**\n")
        w("> The statement \"the ETF launch changed Bitcoin's tail dependence structure\" "
          "stands:")
        w("> the break is still significant after volatility standardization (so it is not "
          "an artifact of a scale change),")
        w("> it does not depend on any single event, and the test itself does not over-reject "
          "under placebos.")
    elif not ok_std:
        w("> **The break disappears after standardization → Finding 1 must be substantially weakened.**\n")
        w("> The \"tail coefficient break\" measured by script 12 is mainly the mechanical "
          "consequence of a **volatility-level change**:")
        w(f"> BTC annualized volatility {v_pre*100:.1f}% → {v_post*100:.1f}%, "
          f"sigma_BTC/sigma_SPY becomes {(v_post/vx_post)/(v_pre/vx_pre):.2f} times its "
          "former value.")
        w("> The only writable statement left is: \"Bitcoin's volatility fell significantly "
          "in the ETF period, while the tail dependence structure itself showed no "
          "significant change\".")
        w("> **Note this does not affect the rejection of route A** — whether the break comes "
          "from scale or structure,")
        w("> pooling the samples to estimate the tail beta is inappropriate either way; and "
          "Section D's measured MDE=1.65 was already insufficient.")
    elif not ok_jk:
        w("> **The break depends on a single event → cannot be written.**\n")
        w("> After dropping one stress event the break becomes not significant, indicating "
          "this is a property of that event,")
        w("> not a property of the institutional change that is \"the ETF launch\".")
    else:
        w("> **The test over-rejects → the conclusion is unreliable.**\n")
        w("> The placebo break rejection rate is too high, indicating the Wald test under "
          "this specification is itself prone to reject,")
        w("> and the real break's p-value cannot be taken at face value.")

    w("")
    for c in CHECKS:
        w("  " + c)

    p = os.path.join(a.outdir, "break_robustness.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\nWritten {p}")


if __name__ == "__main__":
    main()