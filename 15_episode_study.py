"""
15_episode_study.py — Item 8 (part one): a stress-event study of the stock market spanning 11 years

Why do this
    This project has already established that the tail beta is not identifiable in any
    available sample (this sample's MDE is 4.91; pooling all data since 2014 still gives
    1.65), and that the very premise "is the tail relationship stable?" cannot be tested
    either.

    So stop trying to estimate that parameter. Switch to a [non-parametric event study]:
    lay out each stock-market stress event on its own and let the dispersion speak for
    itself. This approach does not depend on any regression specification, so it is not
    affected by the identification problems above.

Core proposition (falsifiable)
    The definition of a safe-haven asset is "reliably provides protection when the stock
    market is under stress". "Reliably" means cross-event [consistency]. Therefore:
        if Bitcoin's event-level reactions are far more dispersed than gold's,
        then even if it does well in some events, that does not constitute a safe-haven
        property.
    This script turns that consistency into a testable quantity.

Event identification: algorithmic, not hand-picked
    Compute the drawdown on SPY's cumulative net value, and take all independent intervals
    where the drawdown reaches -THRESH; the window = prior peak day → trough day. Do a
    three-level sensitivity on the threshold at 5%/8%/10%.
    Hand-picking events is a practice this project already warned against in item 9, and
    is not repeated here.

Convention discipline
    Before 2024-01-11 the data used is [Bitcoin spot], not IBIT.
    Mark the segments in the table, and do not phrase it as "what the IBIT portfolio would
    have done at the time" — script 14's conclusion is that the ETF-launch break is neither
    confirmed nor ruled out, and that extrapolation has no basis.

Usage
    python 15_episode_study.py --raw data/raw --asof 20260921L
    python 15_episode_study.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

ETF_LAUNCH = pd.Timestamp("2024-01-11")
THRESHOLDS = (0.05, 0.08, 0.10)
MAIN_THRESH = 0.08
BOOT_R = 10000
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


# ================================================== Event identification (algorithmic) ====
def drawdown_episodes(r: np.ndarray, thresh: float):
    """
    Find all independent intervals with drawdown >= thresh from the return series.
    Returns [(i_peak, i_trough)], with indices into the return series.

    Rule: compute the drawdown on cumulative net value; find the local trough of the
    drawdown with depth >= thresh;
    prior peak = the last position before the trough where the drawdown was 0;
    the interval ends at the first return to 0 after the trough (or at the end of the
    sample). Overlapping intervals are merged.
    """
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    eps, i, n = [], 0, len(r)
    while i < n:
        if dd[i] >= -1e-12:            # at a prior peak
            i += 1
            continue
        j = i                          # find the end of this drawdown segment
        while j < n and dd[j] < -1e-12:
            j += 1
        seg = dd[i:j]
        if seg.min() <= -thresh:
            t = i + int(np.argmin(seg))
            p = i - 1 if i > 0 else 0  # prior peak day
            eps.append((p, t))
        i = j
    return eps


def ep_return(px_like: np.ndarray, i0: int, i1: int) -> float:
    """Simple return over the interval (i0, i1], compounded from daily returns."""
    return float(np.prod(1 + px_like[i0 + 1:i1 + 1]) - 1)


# ============================================================== Self-test ====
def selftest() -> int:
    print("=" * 78); print("Self-test"); print("=" * 78)
    ok = True

    # Construct a known drawdown: rise 10%, then fall 20%, then rise back
    r = np.array([0.05] * 2 + [-0.10] * 2 + [0.05] * 4)
    eps = drawdown_episodes(r, 0.08)
    ok &= check("Can find a single >8% drawdown", len(eps) == 1, f"found {len(eps)}")
    if eps:
        p, t = eps[0]
        ok &= check("Prior peak and trough located correctly", (p, t) == (1, 3), f"(peak={p}, trough={t})")
        ok &= check("Interval return = -19%", abs(ep_return(r, p, t) - (-0.19)) < 1e-9,
                    f"{ep_return(r,p,t):.4f}")

    # A threshold above the actual drawdown should find nothing
    ok &= check("Threshold 25% finds nothing", len(drawdown_episodes(r, 0.25)) == 0)

    # Two independent drawdowns. Note the first day must be an up day: the start of the
    # series is itself the initial prior peak, so a series that falls from day one does
    # not, by definition, constitute a "drawdown from a prior peak".
    r2 = np.array([0.05, -0.10, 0.12, 0.02, -0.10, 0.12])
    ok &= check("Can distinguish two independent drawdowns", len(drawdown_episodes(r2, 0.08)) == 2,
                f"found {len(drawdown_episodes(r2,0.08))}")

    # Nominal level of the dispersion test
    rng = np.random.default_rng(0)
    rej = 0
    for _ in range(300):
        a = rng.standard_normal(8); b = rng.standard_normal(8)
        d = np.log(a.std(ddof=1) / b.std(ddof=1))
        bs = []
        for _ in range(500):
            ia = rng.integers(0, 8, 8); ib = rng.integers(0, 8, 8)
            sa, sb = a[ia].std(ddof=1), b[ib].std(ddof=1)
            if sa > 0 and sb > 0:
                bs.append(math.log(sa / sb))
        bs = np.array(bs)
        pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        rej += int(pv < .05)
    ok &= check("Nominal level of the dispersion bootstrap test (n=8 small sample, loose band)",
                rej / 300 <= 0.20, f"rejection rate={rej/300:.3f}")

    print()
    print("Self-test all passed." if ok else "Self-test has failures.")
    return 0 if ok else 1


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default="20260921L")
    ap.add_argument("--outdir", default="out/15_episodes")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    rng = np.random.default_rng(SEED)
    os.makedirs(a.outdir, exist_ok=True)

    rule(); w("Section 0  Data and event-identification rules"); rule()
    S = {t: load_raw(a.raw, a.asof, t) for t in ("SPY", "BTC-USD", "GLD", "TLT", "^VIX")}
    if S["SPY"] is None or S["BTC-USD"] is None:
        w(f"Missing long-sample snapshot; please first run 01b_fetch_long.py (AS_OF={a.asof})."); sys.exit(1)
    px = pd.DataFrame({k: v for k, v in S.items() if v is not None}).reindex(S["SPY"].index)
    R = px[["SPY", "BTC-USD", "GLD", "TLT"]].pct_change()
    R["VIX"] = px["^VIX"]
    R = R.dropna(subset=["SPY", "BTC-USD", "GLD", "TLT"])
    dates = R.index
    rs = R["SPY"].values
    w(f"{len(R)} trading days, {dates.min().date()} → {dates.max().date()}")
    w("")
    w("**Event-identification rules (declared in advance, not hand-picked)**")
    w(f"Compute the drawdown on SPY's cumulative net value, and take all independent "
      f"intervals with drawdown depth ≥ {MAIN_THRESH:.0%};")
    w("the window = prior peak day → trough day. Do a three-level sensitivity on the "
      "threshold at 5%/8%/10%.")
    w("")
    w("**Convention discipline**: before 2024-01-11 the data used is [Bitcoin spot], not "
      "IBIT.")
    w("Mark the segments in the table. Do not phrase it as \"what the IBIT portfolio would "
      "have done at the time\" —")
    w("script 14's conclusion is that the ETF-launch break is neither confirmed nor ruled "
      "out, and that extrapolation has no basis.")

    # ============================================ Main table ====
    w("")
    rule(); w(f"Section 1  Full table of stock-market stress events (threshold {MAIN_THRESH:.0%})"); rule()
    eps = drawdown_episodes(rs, MAIN_THRESH)
    rows, rec = [], []
    for p, t in eps:
        d0, d1 = dates[p], dates[t]
        spy = ep_return(rs, p, t)
        btc = ep_return(R["BTC-USD"].values, p, t)
        gld = ep_return(R["GLD"].values, p, t)
        tlt = ep_return(R["TLT"].values, p, t)
        vix = np.nanmax(R["VIX"].values[p:t + 1]) if "VIX" in R else np.nan
        era = "ETF period" if d1 >= ETF_LAUNCH else "Pre-ETF period"
        rows.append([f"{d0.date()} → {d1.date()}", t - p, era,
                     f"{spy*100:+.2f}%", f"{vix:.1f}",
                     f"{btc*100:+.2f}%", f"{gld*100:+.2f}%", f"{tlt*100:+.2f}%",
                     f"{btc/spy:+.2f}", f"{gld/spy:+.2f}"])
        rec.append(dict(d0=d0, d1=d1, era=era, spy=spy, btc=btc, gld=gld, tlt=tlt))
    w(md(rows, ["Window (prior peak → trough)", "Trading days", "Period", "SPY", "Peak VIX",
                "Bitcoin", "GLD", "TLT", "BTC/SPY", "GLD/SPY"]))
    w("The last two columns are the [event-level realized multiple] = asset interval return / "
      "SPY interval return.")
    w("When SPY falls: a positive value = fell in the same direction (a larger multiple means "
      "a harder fall); negative = rose against the trend.")
    E = pd.DataFrame(rec)
    n_ep = len(E)
    check(f"Number of identified stress events ≥ 5 (threshold {MAIN_THRESH:.0%})", n_ep >= 5, f"{n_ep} events")

    # ============================================ Consistency ====
    w("")
    rule(); w("Section 2  Core proposition: consistency"); rule()
    w("The definition of a safe-haven asset is \"reliably provides protection when the stock "
      "market is under stress\".")
    w("\"Reliably\" means cross-event consistency. Below we turn that into a comparable "
      "quantity.\n")
    rb = (E.btc / E.spy).values
    rg = (E.gld / E.spy).values
    rt = (E.tlt / E.spy).values
    rows = []
    for nm, v in (("Bitcoin", rb), ("GLD", rg), ("TLT", rt)):
        rows.append([nm, f"{v.mean():+.2f}", f"{np.median(v):+.2f}",
                     f"{v.std(ddof=1):.2f}", f"{v.min():+.2f}", f"{v.max():+.2f}",
                     f"{v.max()-v.min():.2f}",
                     f"{int((v<0).sum())}/{n_ep}"])
    w(md(rows, ["Asset", "Mean", "Median", "**SD**", "Min", "Max", "Range",
                "Times rising against the trend"]))
    w("\"Times rising against the trend\" = the number of events in which SPY fell but the "
      "asset rose.\n")

    # Is the dispersion significantly different (bootstrap over events)?
    w("**Is Bitcoin's event-level reaction significantly more dispersed than gold's?** "
      "(resampling events, B=10000)\n")
    obs = math.log(rb.std(ddof=1) / rg.std(ddof=1))
    bs = []
    for _ in range(BOOT_R):
        i = rng.integers(0, n_ep, n_ep)
        sb, sg = rb[i].std(ddof=1), rg[i].std(ddof=1)
        if sb > 0 and sg > 0:
            bs.append(math.log(sb / sg))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pv = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    w(md([["SD(Bitcoin multiple) / SD(GLD multiple)",
           f"{rb.std(ddof=1)/rg.std(ddof=1):.2f}x",
           f"[{math.exp(lo):.2f}, {math.exp(hi):.2f}]", f"{pv:.4f}",
           "**Significant**" if pv < .05 else "Not significant"]],
         ["Quantity", "Point estimate", "95% CI", "p", "Verdict"]))
    ok_disp = pv < 0.05
    check("2. Bitcoin's event-level reaction is significantly more dispersed than gold's", ok_disp, f"p={pv:.4f}")

    # ============================================ Threshold sensitivity ====
    w("")
    rule(); w("Section 3  Event-threshold sensitivity"); rule()
    rows = []
    for th in THRESHOLDS:
        ee = drawdown_episodes(rs, th)
        if len(ee) < 3:
            continue
        b = np.array([ep_return(R["BTC-USD"].values, p, t) / ep_return(rs, p, t)
                      for p, t in ee])
        g = np.array([ep_return(R["GLD"].values, p, t) / ep_return(rs, p, t)
                      for p, t in ee])
        rows.append([f"{th:.0%}", len(ee), f"{b.std(ddof=1):.2f}", f"{g.std(ddof=1):.2f}",
                     f"{b.std(ddof=1)/g.std(ddof=1):.2f}x",
                     f"{int((b<0).sum())}/{len(ee)}", f"{int((g<0).sum())}/{len(ee)}"])
    w(md(rows, ["Drawdown threshold", "Events", "SD(BTC multiple)", "SD(GLD multiple)",
                "Dispersion ratio", "BTC against-trend", "GLD against-trend"]))
    ratios = [float(r[4].rstrip('x')) for r in rows]
    check("3. Dispersion ratio points the same way across the three thresholds (all > 1)",
          all(x > 1 for x in ratios),
          f"{[f'{x:.2f}' for x in ratios]}")

    # ============================================ Period comparison ====
    w("")
    rule(); w("Section 4  Pre-ETF period vs ETF period (descriptive only, no inference)"); rule()
    rows = []
    for era in ("Pre-ETF period", "ETF period"):
        sub = E[E.era == era]
        if len(sub) == 0:
            continue
        v = (sub.btc / sub.spy).values
        g = (sub.gld / sub.spy).values
        rows.append([era, len(sub), f"{v.mean():+.2f}",
                     f"{v.std(ddof=1):.2f}" if len(sub) > 1 else "—",
                     f"{int((v<0).sum())}/{len(sub)}",
                     f"{g.mean():+.2f}", f"{int((g<0).sum())}/{len(sub)}"])
    w(md(rows, ["Period", "Events", "Mean BTC multiple", "SD BTC multiple",
                "BTC against-trend", "Mean GLD multiple", "GLD against-trend"]))
    w("**Descriptive only.** The ETF period has too few events, and inference on the "
      "difference between the two periods was already shown to be infeasible in script 14.")

    # ============================================ Conclusion ====
    w("")
    rule(); w("Wording that can go into the paper"); rule()
    w(f"> We algorithmically identified all {n_ep} stock-market stress events in 2014–2026 "
      f"in which the SPY drawdown was ≥ {MAIN_THRESH:.0%} (prior peak to trough), and "
      "examined each asset's event-level reaction.\n")
    w(f"> Bitcoin's event-level realized multiple ranges over [{rb.min():+.2f}, {rb.max():+.2f}], "
      f"with a standard deviation of {rb.std(ddof=1):.2f}; gold's is "
      f"[{rg.min():+.2f}, {rg.max():+.2f}], with a standard deviation of {rg.std(ddof=1):.2f}.")
    w(f"> Bitcoin's dispersion is {rb.std(ddof=1)/rg.std(ddof=1):.2f} times that of gold "
      f"(event-level bootstrap 95% CI [{math.exp(lo):.2f}, {math.exp(hi):.2f}], p = {pv:.4f}).\n")
    w(f"> Out of {n_ep} events, Bitcoin rose against the trend {int((rb<0).sum())} times, "
      f"gold {int((rg<0).sum())} times.\n")
    w("> **This argument does not rely on any regression specification**, and is therefore "
      "not affected by the tail-beta non-identifiability described in Section X of this "
      "paper: it compares the dispersion of event-level outcomes, not a point estimate of "
      "some conditional moment.")
    w("")
    w("**Key point**: the safe-haven property requires cross-event consistency. Even if "
      "Bitcoin performs well in individual events,")
    w("the dispersion of its reactions by itself rules out the property of \"reliably "
      "providing protection\".")

    w("")
    for c in CHECKS:
        w("  " + c)

    pth = os.path.join(a.outdir, "episode_study.md")
    with open(pth, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    E.to_csv(os.path.join(a.outdir, "episodes.csv"), index=False)
    print(f"\nWritten {pth}")


if __name__ == "__main__":
    main()