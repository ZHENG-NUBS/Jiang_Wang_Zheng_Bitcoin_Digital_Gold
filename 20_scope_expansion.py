#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
20_scope_expansion.py — Expanding the comparison set: Bitcoin itself vs the ETF wrapper vs broader benchmarks (reviewer comment: empirical scope)

The reviewer asks that IBIT be compared not only with GLD but also with spot Bitcoin, other spot
Bitcoin ETFs, other gold instruments, and broader equity and bond benchmarks; that results be
reported at both the single-asset and the portfolio level; and that the economic properties of
Bitcoin be separated from ETF-specific factors such as fees, tracking error, liquidity, and
trading hours.

Data: data/raw/*__20260921S.* fetched by 01d_fetch_scope.py (Yahoo daily bars, hourly Bitcoin
bars, GLD archive from the SPDR website).
The sample is the same as the main sample: 2024-01-12 → 2026-06-15 (SPY trading calendar; prices
are aligned by date before computing simple returns, with no forward filling; when the LBMA gold
price has no quote because of a UK holiday, the previous quote is carried forward, so the change
over both days is booked on the next trading day).

Sections:
  0. Data and alignment: column-by-column check against the main snapshot; construction of the
     Bitcoin 16:00 ET price and cross-check between two exchanges; timing-alignment test.
  A. Single-asset risk table: Bitcoin (spot at two pricing times, spot ETFs, futures ETF), gold
     (LBMA, futures, ETFs), equity and bond benchmarks. Annualized return/volatility, VaR/ES,
     maximum drawdown, ρ and β with SPY, tail slope (free intercept).
  B. Decomposition: BTC-USD (00:00 UTC) → BTC (16:00 ET) → IBIT.
       Trading-hours effect = m(BTC16) − m(BTC00); ETF wrapper effect = m(IBIT) − m(BTC16).
     Both single-asset and portfolio levels, with paired block-bootstrap intervals. The
     corresponding gold decomposition: LBMA (London 15:00) → futures → GLD.
  C. Fund cross-section: fees, tracking difference (annualized), tracking error, liquidity
     (dollar volume, Amihud), and their relation to risk measures and the portfolio ES95 ratio.
  D. Trading hours: accumulation over non-trading days (Monday/post-holiday variance ratio);
     overnight vs intraday variance shares of ETFs; share of the realized variance of hourly
     Bitcoin returns that falls in hours when the US equity market is closed.
  E. Portfolio level: grid of ES95 ratios for Bitcoin instruments × gold instruments; grid of
     equity × bond benchmarks (5 × 5).
  F. Single-asset correlations and β with broader benchmarks.

Usage:
    python 20_scope_expansion.py                     # finds the __20260921S files under data/raw automatically
    python 20_scope_expansion.py --raw data/raw --main data/processed/returns__20260921.csv
    python 20_scope_expansion.py --selftest          # self-test on synthetic data
Dependencies: numpy, pandas.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ASOF = "20260921S"
SEED = 20260925
START, END = pd.Timestamp("2024-01-12"), pd.Timestamp("2026-06-15")
B_BOOT = 2000
L_BOOT = 2          # ratio, volatility and correlation statistics (consistent with scripts 11 and 19)
L_TAIL = 10         # tail-event statistics (consistent with scripts 03 and 19)

SPOT = ["IBIT", "FBTC", "ARKB", "BITB", "GBTC", "BTC", "HODL", "BTCO", "BRRR", "EZBC", "BTCW", "DEFI"]
FUTB = ["BITO"]
GOLD_ETF = ["GLD", "IAU", "GLDM", "SGOL", "BAR", "AAAU", "IAUM"]
GOLD_FUT = ["GC=F"]
EQUITY = ["SPY", "VTI", "VT", "QQQ", "EFA"]
BOND = ["TLT", "IEF", "AGG", "SHY", "TIP"]
FIRST_VALID = {"BTC": "2024-07-31", "DEFI": "2024-03-27"}   # Mini Trust listing date; date DEFI converted to spot

# Sponsor fees (annual %). launch = standard fee announced at the January 2024 listing; 2026 = standard fee in H1 2026.
# Most funds had temporary waivers after launch (see WAIVER), so the *actual* cost is inferred from the data
# (tracking difference); this table is for reference only.
# Sources: sponsors' prospectuses and announcements, cross-checked against compilations by Nasdaq (2024-01-11),
# Swan Bitcoin (2024-07), Amppfy (2026-03), and others.
FEE = {
    "IBIT": (0.25, 0.25), "FBTC": (0.25, 0.25), "ARKB": (0.21, 0.21), "BITB": (0.20, 0.20),
    "GBTC": (1.50, 1.50), "BTC": (0.15, 0.15), "HODL": (0.25, 0.20), "BTCO": (0.39, 0.25),
    "BRRR": (0.49, 0.25), "EZBC": (0.19, 0.19), "BTCW": (0.30, 0.25), "DEFI": (0.90, 0.25),
    "BITO": (0.95, 0.95),
    "GLD": (0.40, 0.40), "IAU": (0.25, 0.25), "GLDM": (0.10, 0.10), "SGOL": (0.17, 0.17),
    "BAR": (0.1749, 0.1749), "AAAU": (0.18, 0.18), "IAUM": (0.09, 0.09),
}
WAIVER = {
    "IBIT": "0.12% for 12 months or first $5bn", "FBTC": "0% until 31 Jul 2024",
    "ARKB": "0% for 6 months or first $1bn", "BITB": "0% for 6 months or first $1bn",
    "HODL": "0% on first $1.5bn to 31 Mar 2025 (later extended)", "BTCO": "0% for 6 months or first $5bn",
    "BRRR": "0% for 3 months", "EZBC": "0% until 2 Aug 2024 or first $10bn",
    "BTCW": "0% for 6 months or first $1bn",
}
EARLY_CLOSE = {"2024-07-03", "2024-11-29", "2024-12-24", "2025-07-03", "2025-11-28", "2025-12-24"}

LINES: list[str] = []
CHECKS: list[str] = []
KEY: dict = {}


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
    return ok


def f2(x, d=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(pt, lo, hi, d=2, sign=False):
    fmt = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"{fmt.format(pt)} [{fmt.format(lo)}, {fmt.format(hi)}]"


# ============================================================ Statistical kernels ====
def stationary_boot(n, R, L, rng):
    """Row-index matrix (n×R) for the stationary block bootstrap (Politis–Romano): geometric block lengths, circular wrap-around."""
    p = 1.0 / L
    idx = np.empty((n, R), dtype=np.int64)
    idx[0] = rng.integers(0, n, R)
    newblk = rng.random((n, R)) < p
    jump = rng.integers(0, n, (n, R))
    for t in range(1, n):
        idx[t] = np.where(newblk[t], jump[t], (idx[t - 1] + 1) % n)
    return idx


def es(x, a=0.05):
    """Historical-simulation ES (positive = loss; same convention as script 11). x may be n or n×B."""
    x = np.asarray(x, float)
    q = np.quantile(x, a, axis=0)
    m = x <= q
    return -(np.where(m, x, 0.0).sum(0) / m.sum(0))


def var(x, a=0.05):
    return -np.quantile(np.asarray(x, float), a, axis=0)


def vol(x):
    return np.asarray(x, float).std(0, ddof=1) * math.sqrt(252)


def corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean(0); ym = y - y.mean(0)
    return (xm * ym).sum(0) / np.sqrt((xm ** 2).sum(0) * (ym ** 2).sum(0))


def beta(y, x):
    x = np.asarray(x, float); y = np.asarray(y, float)
    xm = x - x.mean(0); ym = y - y.mean(0)
    return (xm * ym).sum(0) / (xm ** 2).sum(0)


def tail_slope(y, x, q=0.05):
    """Tail slope (free intercept): OLS of y on x over the days on which x is at or below its q-quantile."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = x <= np.quantile(x, q)
    return float(beta(y[m], x[m]))


def mdd(r):
    v = np.cumprod(1 + np.asarray(r, float))
    return float((v / np.maximum.accumulate(v) - 1).min())


def ann_geo(r):
    r = np.asarray(r, float)
    return float(np.prod(1 + r) ** (252 / len(r)) - 1)


METRICS = {
    "vol": lambda r, s: vol(r) * 100,
    "VaR95": lambda r, s: var(r, 0.05) * 100,
    "ES95": lambda r, s: es(r, 0.05) * 100,
    "ES99": lambda r, s: es(r, 0.01) * 100,
    "rho": lambda r, s: corr(r, s),
    "beta": lambda r, s: beta(r, s),
}


def boot_metric(fn, cols, idx):
    """Compute a statistic for several series on the same set of bootstrap indices; returns a B-vector. cols is a tuple of length-n series."""
    return fn(*[c[idx] for c in cols])


def pvalue(bs, null=0.0):
    bs = np.asarray(bs, float)
    bs = bs[np.isfinite(bs)]
    return float(min(1.0, 2 * min((bs <= null).mean(), (bs >= null).mean())))


# ============================================================ Loading ====
def find_raw_dir(arg):
    cands = [arg] if arg else []
    cands += [os.path.join(HERE, "data", "raw"), os.path.join(os.getcwd(), "data", "raw")]
    for c in cands:
        if c and glob.glob(os.path.join(c, f"*__{ASOF}.csv")):
            return c
    return None


def safe_name(t):
    return t.replace("^", "IDX_").replace("=", "_")


def load_daily(raw, t):
    p = os.path.join(raw, f"{safe_name(t)}__{ASOF}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").drop_duplicates("Date")
    return d.set_index("Date")


def load_hourly(raw, name):
    p = os.path.join(raw, f"BTCUSD_1h_{name}__{ASOF}.csv")
    if not os.path.exists(p):
        return None
    h = pd.read_csv(p)
    h["time_utc"] = pd.to_datetime(h["time_utc"], utc=True)
    return h.set_index("time_utc").sort_index()


def btc_at_close(h, dates):
    """Bitcoin price at the US Eastern close of each trading day (16:00; 13:00 on early-close days):
    the close of the hourly bar that *ends* at that time; if that bar is missing, the most recent bar within the preceding 3 hours."""
    out, fallback = [], 0
    starts = h.index
    close = h["close"].values
    for d in dates:
        hh = 13 if d.strftime("%Y-%m-%d") in EARLY_CLOSE else 16
        t_end = pd.Timestamp(d.year, d.month, d.day, hh, tz="America/New_York").tz_convert("UTC")
        k = starts.searchsorted(t_end - pd.Timedelta(hours=1), side="right") - 1
        if k >= 0 and starts[k] == t_end - pd.Timedelta(hours=1):
            out.append(close[k])
        elif k >= 0 and t_end - starts[k] <= pd.Timedelta(hours=4):
            out.append(close[k]); fallback += 1
        else:
            out.append(np.nan)
    return pd.Series(out, index=dates), fallback


def parse_gld_archive(path):
    """GLD archive from the SPDR website: returns DataFrame[LBMA, NAV] indexed by date. Columns are identified by keyword; explanatory lines before the header are tolerated."""
    raw = open(path, "rb").read().decode("utf-8", errors="ignore")
    lines = raw.splitlines()
    hdr = next((i for i, l in enumerate(lines) if l.lower().startswith("date")), None)
    if hdr is None:
        return None, "no header line starting with Date found"
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    low = {c: c.lower() for c in df.columns}
    c_lbma = next((c for c, l in low.items() if "lbma" in l), None)
    c_nav = next((c for c, l in low.items() if "nav" in l
                  and not any(k in l for k in ("ounce", "total", "value", "premium", "discount", "tonne"))), None)
    if c_lbma is None:
        return None, f"LBMA column not found; columns {list(df.columns)}"
    dt = pd.to_datetime(df[df.columns[0]].astype(str).str.strip(), errors="coerce", format="mixed", dayfirst=True)
    num = lambda c: pd.to_numeric(df[c].astype(str).str.replace(",", "").str.replace("$", "").str.strip(),
                                  errors="coerce").values
    out = pd.DataFrame({"LBMA": num(c_lbma)}, index=pd.DatetimeIndex(dt))
    if c_nav:
        out["NAV"] = num(c_nav)
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated()]
    return out, f"LBMA column = '{c_lbma}'; NAV column = '{c_nav}'"


def parse_ishares_nav(path):
    """iShares historical data (SpreadsheetML): locate the As Of / NAV per Share columns in the Historical sheet. Returns None if they cannot be found."""
    try:
        txt = open(path, "rb").read().decode("utf-8", errors="ignore")
        sheet = re.search(r'<ss:Worksheet ss:Name="Historical">(.*?)</ss:Worksheet>', txt, re.S)
        if not sheet:
            return None
        rows = re.findall(r"<ss:Row>(.*?)</ss:Row>", sheet.group(1), re.S)
        cells = [re.findall(r"<ss:Data[^>]*>(.*?)</ss:Data>", r, re.S) for r in rows]
        head = cells[0]
        ia = next(i for i, c in enumerate(head) if "as of" in c.lower())
        inav = next(i for i, c in enumerate(head) if "nav" in c.lower())
        recs = [(pd.to_datetime(c[ia], errors="coerce"), pd.to_numeric(c[inav], errors="coerce"))
                for c in cells[1:] if len(c) > max(ia, inav)]
        s = pd.Series({d: v for d, v in recs if pd.notna(d)}).sort_index()
        return s if len(s) > 100 else None
    except Exception:                                   # noqa: BLE001
        return None


# ============================================================ Liquidity ====
def amihud(r, dollar_vol):
    """Amihud (2002): |r| / dollar volume, expressed in *basis points per $1 million traded*."""
    m = (dollar_vol > 0) & r.notna()
    return float((r[m].abs() * 1e4 / (dollar_vol[m] / 1e6)).mean())


# ============================================================ Self-test ====
def selftest():
    print("=" * 78); print("Self-test (synthetic data)"); print("=" * 78)
    rng = np.random.default_rng(0)
    ok = True
    x = rng.standard_normal(400000)
    ok &= check("es() matches the normal theoretical value (ES95 = 2.063σ)", abs(es(x) - 2.0627) < 0.02, f"{es(x):.4f}")
    M = rng.standard_normal((500, 3))
    ok &= check("es() column-wise equals per-column computation", np.allclose(es(M), [es(M[:, j]) for j in range(3)]))
    y = 0.5 * x[:1000] + rng.standard_normal(1000)
    ok &= check("beta() recovers the true value 0.5", abs(beta(y, x[:1000]) - 0.5) < 0.1, f"{beta(y, x[:1000]):.3f}")
    # Weekend accumulation: a 24/7 random walk sampled on weekdays should have Monday variance about 3 times that of other days
    days = pd.date_range("2020-01-01", periods=7 * 400, freq="D")
    p = pd.Series(np.exp(np.cumsum(rng.standard_normal(len(days)) * 0.02)), index=days)
    bd = days[days.dayofweek < 5]
    r = p.reindex(bd).pct_change().dropna()
    mon = r.index.dayofweek == 0
    vr = r[mon].var() / r[~mon].var()
    ok &= check("24/7 random walk: Monday variance ratio ≈ 3", 2.4 < vr < 3.6, f"{vr:.2f}")
    # Fee regression: the slope of tracking difference on fee should be about −1
    fees = np.array([0.15, 0.2, 0.25, 0.25, 0.5, 1.5])
    td = -fees + rng.normal(0, 0.02, len(fees))
    sl = np.polyfit(fees, td, 1)[0]
    ok &= check("slope of tracking difference on fee ≈ −1", abs(sl + 1) < 0.1, f"{sl:.3f}")
    print("\nAll self-tests passed." if ok else "\nSome self-tests failed.")
    return 0 if ok else 1


# ============================================================ Main ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None)
    ap.add_argument("--main", default=None, help="main-sample returns__20260921.csv, used for the column-by-column check")
    ap.add_argument("--outdir", default="out/20")
    ap.add_argument("--boot", type=int, default=B_BOOT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    raw = find_raw_dir(a.raw)
    if raw is None:
        sys.exit(f"Cannot find data/raw/*__{ASOF}.csv. Run 01d_fetch_scope.py first, or specify the directory with --raw.")
    if not os.path.isabs(a.outdir):
        a.outdir = os.path.join(HERE, a.outdir)
    os.makedirs(a.outdir, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Report the data directory as given on the command line (if relative), otherwise only its basename,
    # so that no absolute path is written to the output.
    raw_disp = raw if (a.raw and not os.path.isabs(raw)) else os.path.basename(os.path.normpath(raw))
    w("# Expanding the comparison set: Bitcoin itself, the ETF wrapper, and broader benchmarks (script 20)\n")
    w(f"Data directory: `{raw_disp}`; snapshot `{ASOF}`; sample {START.date()} → {END.date()}; block bootstrap B = {a.boot}, "
      f"ratio/volatility/correlation L = {L_BOOT}, tail slope L = {L_TAIL}; seed {SEED}.\n")

    # ------------------------------------------------------------------ 0. Data
    w("## 0. Data and alignment\n")
    tick = SPOT + FUTB + GOLD_ETF + GOLD_FUT + EQUITY + BOND + ["BTC-USD", "^VIX", "^IRX"]
    D = {t: load_daily(raw, t) for t in tick}
    missing = [t for t, d in D.items() if d is None]
    w(f"Daily bars: loaded {len(tick) - len(missing)} / {len(tick)} tickers" + (f"; missing {missing}" if missing else "") + "\n")
    if D["SPY"] is None or D["IBIT"] is None or D["GLD"] is None or D["TLT"] is None:
        sys.exit("Core tickers (SPY/IBIT/GLD/TLT) are missing; cannot continue.")
    cal_all = D["SPY"].index
    cal = cal_all[(cal_all >= START - pd.Timedelta(days=10)) & (cal_all <= END)]
    # Prices (AdjClose) aligned to SPY trading days
    P = pd.DataFrame({t: D[t]["AdjClose"].reindex(cal) for t in tick if D[t] is not None and t not in ("^VIX", "^IRX")})
    for t, d0 in FIRST_VALID.items():
        if t in P:
            P.loc[P.index < pd.Timestamp(d0), t] = np.nan
    P = P.rename(columns={"BTC-USD": "BTC00"})

    # Bitcoin at 16:00 ET
    H = {k: load_hourly(raw, k) for k in ("coinbase", "bitstamp")}
    hsrc = "coinbase" if H["coinbase"] is not None else ("bitstamp" if H["bitstamp"] is not None else None)
    if hsrc is None:
        sys.exit("Hourly bars from both exchanges are missing; cannot construct the 16:00 ET Bitcoin price.")
    P["BTC16"], nfb = btc_at_close(H[hsrc], cal)
    w(f"Bitcoin 16:00 ET price taken from {hsrc} hourly bars (13:00 on early-close days); on {nfb} days the most recent bar "
      f"within the preceding 4 hours was used; {int(P['BTC16'].isna().sum())} days missing.")
    other = "bitstamp" if hsrc == "coinbase" else "coinbase"
    if H[other] is not None:
        b2, _ = btc_at_close(H[other], cal)
        rr = pd.concat([P["BTC16"].pct_change(), b2.pct_change()], axis=1).dropna()
        rr = rr[(rr.index >= START) & (rr.index <= END)]
        cc = float(np.corrcoef(rr.iloc[:, 0], rr.iloc[:, 1])[0, 1])
        te = float((rr.iloc[:, 0] - rr.iloc[:, 1]).std() * math.sqrt(252) * 100)
        w(f"16:00 ET daily returns from the two exchanges: correlation {cc:.5f}; annualized s.d. of the difference {te:.2f}%.")
        check("0. Coinbase vs Bitstamp 16:00 ET daily-return correlation > 0.999", cc > 0.999, f"{cc:.5f}")
        KEY["venue_check"] = dict(corr=cc, te=te)

    # Gold benchmarks: LBMA (London 15:00 = 10:00/11:00 US Eastern) and GLD NAV
    gpath = os.path.join(raw, f"GLD_archive__{ASOF}.csv")
    G = None
    if os.path.exists(gpath):
        G, note = parse_gld_archive(gpath)
        w(f"GLD archive (SPDR website): {note}")
    if G is not None and G["LBMA"].notna().sum() > 300:
        lb = G["LBMA"].dropna()
        P["LBMA"] = lb.reindex(cal.union(lb.index)).ffill(limit=3).reindex(cal)
        n_ff = int(cal.isin(cal.difference(lb.index)).sum())
        w(f"LBMA gold price: {n_ff} US trading days have no London quote (UK holidays); the previous quote is carried forward.")
        if "NAV" in G and G["NAV"].notna().sum() > 300:
            P["GLDNAV"] = G["NAV"].reindex(cal)
    else:
        w("GLD archive missing or unparseable: the gold benchmark uses COMEX futures only.")
    R = P.pct_change()
    R = R[(R.index >= START) & (R.index <= END)]
    n = len(R)
    w(f"\nMain sample window: {n} trading days (main snapshot: 607).")
    check("0. Number of trading days matches the main sample (607)", n == 607, str(n))

    # Column-by-column check against the main snapshot
    mp = a.main
    if mp is None:
        for c in (os.path.join(HERE, "data", "processed", "returns__20260921.csv"),
                  os.path.join(os.getcwd(), "data", "processed", "returns__20260921.csv")):
            if os.path.exists(c):
                mp = c; break
    if mp and os.path.exists(mp):
        M0 = pd.read_csv(mp, parse_dates=["Date"]).set_index("Date")
        rows = []
        for t in ("IBIT", "GLD", "TLT", "SPY"):
            d = (R[t] - M0[t].reindex(R.index)).abs()
            rows.append([t, f"{d.max():.1e}", int((d > 1e-6).sum())])
        w("\nColumn-by-column check against the main snapshot (absolute difference in returns):\n")
        w(md(rows, ["Ticker", "Max difference", "Days with difference > 1e-6"]))
        check("0. IBIT/GLD/SPY/TLT match the main snapshot (difference ≤ 1e-6)", all(r[2] == 0 for r in rows))
    else:
        w("(Main snapshot not found; column-by-column check skipped)")

    # Timing-alignment test: the same-day correlation should be the highest
    w("\nTiming-alignment test (correlation with the reference series at lags −1/0/+1 days; the same day should be highest):\n")
    rows = []
    for t, ref in (("BTC00", "IBIT"), ("BTC16", "IBIT"), ("GC=F", "GLD"), ("LBMA", "GLD")):
        if t not in R or R[t].notna().sum() < 300:
            continue
        cs = [R[t].corr(R[ref].shift(k)) for k in (-1, 0, 1)]
        rows.append([t, ref] + [f"{c:.3f}" for c in cs])
        check(f"0. {t} has the highest same-day correlation with {ref}", np.argmax(cs) == 1, f"{cs[1]:.3f}")
    w(md(rows, ["Series", "Reference", "Reference leads 1 day", "Same day", "Reference lags 1 day"]))

    irx = D["^IRX"]
    if irx is not None:
        m_irx = float(irx["Close"].reindex(R.index).mean())
        w(f"\n^IRX (13-week Treasury bill yield) sample mean {m_irx:.2f}% (to replace the 4.3% placeholder in the Sharpe ratio).")
        KEY["irx_mean"] = m_irx

    # Common bootstrap indices
    IDX = stationary_boot(n, a.boot, L_BOOT, rng)
    IDXT = stationary_boot(n, a.boot, L_TAIL, rng)
    spy = R["SPY"].values

    def avail(t, min_n=250):
        return t in R and R[t].notna().sum() >= min_n

    # ------------------------------------------------------------------ A. Single assets
    w("\n## A. Single-asset risk (each on its own available sample; % figures are daily unless marked annualized)\n")
    groups = [("Bitcoin", ["BTC00", "BTC16"] + SPOT + FUTB), ("Gold", ["LBMA", "GC=F"] + GOLD_ETF),
              ("Equity", EQUITY), ("Bonds", BOND)]
    A = {}
    rows = []
    for g, lst in groups:
        for t in lst:
            if not avail(t, 200):
                continue
            s = R[[t]].join(R[["SPY"]].rename(columns={"SPY": "_SPY"})).dropna()
            r, x = s[t].values, s["_SPY"].values
            A[t] = dict(group=g, n=len(r), start=str(s.index[0].date()), ret=ann_geo(r) * 100, vol=float(vol(r) * 100),
                        VaR95=float(var(r, .05) * 100), ES95=float(es(r, .05) * 100), VaR99=float(var(r, .01) * 100),
                        ES99=float(es(r, .01) * 100), mdd=mdd(r) * 100, rho=float(corr(r, x)), beta=float(beta(r, x)),
                        tail=tail_slope(r, x))
            v = A[t]
            rows.append([g, t, v["n"], f2(v["ret"], 1), f2(v["vol"], 1), f2(v["VaR95"]), f2(v["ES95"]),
                         f2(v["VaR99"]), f2(v["ES99"]), f2(v["mdd"], 1), f2(v["rho"]), f2(v["beta"]), f2(v["tail"])])
    w(md(rows, ["Group", "Ticker", "Days", "Ann. return %", "Ann. vol. %", "VaR95", "ES95", "VaR99", "ES99", "Max drawdown %",
                "ρ(SPY)", "β(SPY)", "Tail slope"]))
    KEY["A"] = A
    spot_full = [t for t in SPOT if t in A and A[t]["n"] == n]
    if spot_full:
        for k in ("vol", "ES95", "ES99", "rho", "beta"):
            vals = [A[t][k] for t in spot_full]
            w(f"Full-sample spot ETFs ({len(spot_full)} funds), {k}: {min(vals):.3f} – {max(vals):.3f}")
        KEY["A_spot_range"] = {k: [min(A[t][k] for t in spot_full), max(A[t][k] for t in spot_full)]
                               for k in ("vol", "ES95", "ES99", "rho", "beta", "tail")}
        KEY["A_spot_full"] = spot_full
    gold_full = [t for t in GOLD_ETF if t in A and A[t]["n"] == n]
    if gold_full:
        KEY["A_gold_range"] = {k: [min(A[t][k] for t in gold_full), max(A[t][k] for t in gold_full)]
                               for k in ("vol", "ES95", "ES99", "rho", "beta", "tail")}
        KEY["A_gold_full"] = gold_full

    # ------------------------------------------------------------------ B. Decomposition
    w("\n## B. Decomposition: trading-hours effect and ETF wrapper effect\n")
    w("Bitcoin: BTC00 (Yahoo, 00:00 UTC) → BTC16 (16:00 US Eastern) → IBIT. Trading-hours effect = BTC16 − BTC00; "
      "wrapper effect = IBIT − BTC16. Intervals are from a paired stationary block bootstrap (the same dates are resampled jointly).\n")

    def port(x, eq="SPY", bd="TLT"):
        return (x + R[eq].values + R[bd].values) / 3

    def tail_cols(y, x):
        """Tail slope (free intercept), computed column by column on n×B bootstrap samples; each column redefines the tail by its own SPY 5% quantile."""
        y = np.asarray(y, float); x = np.asarray(x, float)
        if y.ndim == 1:
            return tail_slope(y, x)
        return np.array([tail_slope(y[:, j], x[:, j]) for j in range(y.shape[1])])

    def decomp(chain, label, eff_names, ref_port=None):
        """Compare metrics step by step along chain: effect k = m(chain[k+1]) − m(chain[k]). Paired bootstrap (same set of dates)."""
        ok = R[list(chain) + ["SPY"]].notna().all(1).values
        m_ok = int(ok.sum())
        idx2 = IDX if ok.all() else stationary_boot(m_ok, a.boot, L_BOOT, np.random.default_rng(SEED + 3))
        idx10 = IDXT if ok.all() else stationary_boot(m_ok, a.boot, L_TAIL, np.random.default_rng(SEED + 4))
        x = [R[c].values for c in chain]
        sp = spy[ok]
        out, rows = {}, []
        mets = dict(METRICS)
        mets["tail"] = tail_cols
        for lvl in ("asset", "portfolio"):
            for mname, fn in mets.items():
                if lvl == "portfolio" and mname in ("rho", "beta", "tail"):
                    continue
                ser = [(port(v) if lvl == "portfolio" else v)[ok] for v in x]
                idx = idx10 if mname == "tail" else idx2
                pts = [float(fn(s_, sp)) for s_ in ser]
                bss = [fn(s_[idx], sp[idx]) for s_ in ser]
                o = dict(values=pts, effects=[])
                d = 3 if mname in ("rho", "beta", "tail") else 2
                row = [lvl, mname] + [f2(v, d) for v in pts]
                for k in range(len(chain) - 1):
                    db = bss[k + 1] - bss[k]
                    lo, hi = np.percentile(db[np.isfinite(db)], [2.5, 97.5])
                    e = dict(name=eff_names[k], est=pts[k + 1] - pts[k], ci=[float(lo), float(hi)], p=pvalue(db))
                    o["effects"].append(e)
                    row.append(ci(e["est"], lo, hi, d, True) + f" p={e['p']:.2f}")
                out[f"{lvl}_{mname}"] = o
                rows.append(row)
        if ref_port is not None:
            gp = ref_port[ok]
            ratios = [float(es(port(v)[ok]) / es(gp)) for v in x]
            bsr = [es(port(v)[ok][idx2]) / es(gp[idx2]) for v in x]
            o = dict(values=ratios, ci=[list(np.percentile(b, [2.5, 97.5])) for b in bsr], effects=[])
            row = ["portfolio", "ES95 ratio vs GLD portfolio"] + [f2(v) for v in ratios]
            for k in range(len(chain) - 1):
                db = bsr[k + 1] - bsr[k]
                lo, hi = np.percentile(db, [2.5, 97.5])
                e = dict(name=eff_names[k], est=ratios[k + 1] - ratios[k], ci=[float(lo), float(hi)], p=pvalue(db))
                o["effects"].append(e)
                row.append(ci(e["est"], lo, hi, 2, True) + f" p={e['p']:.2f}")
            out["ratio_ES95"] = o
            rows.append(row)
        w(f"**{label}** ({m_ok} days)\n")
        heads = [f"{eff_names[k]} {chain[k + 1]}−{chain[k]}" for k in range(len(chain) - 1)]
        w(md(rows, ["Level", "Metric"] + list(chain) + heads))
        return out

    gld_port = port(R["GLD"].values)
    KEY["B_btc"] = decomp(("BTC00", "BTC16", "IBIT"), "Bitcoin: BTC00 → BTC16 → IBIT", ["Time effect", "Wrapper effect"], gld_port)
    bv = KEY["B_btc"]["asset_vol"]
    check("B. Decomposition identity: time effect + wrapper effect = IBIT − BTC00",
          abs(bv["effects"][0]["est"] + bv["effects"][1]["est"] - (bv["values"][2] - bv["values"][0])) < 1e-9)
    check("B. IBIT/GLD portfolio ES95 ratio reproduces 1.63", abs(KEY["B_btc"]["ratio_ES95"]["values"][2] - 1.633) < 0.006,
          f"{KEY['B_btc']['ratio_ES95']['values'][2]:.3f}")

    # Daily tracking
    trk = R[["BTC00", "BTC16", "IBIT"]].dropna()
    KEY["B_track"] = dict(
        corr_ibit_btc00=float(trk.IBIT.corr(trk.BTC00)), corr_ibit_btc16=float(trk.IBIT.corr(trk.BTC16)),
        corr_btc16_btc00=float(trk.BTC16.corr(trk.BTC00)),
        te_ibit_btc00=float((trk.IBIT - trk.BTC00).std() * math.sqrt(252) * 100),
        te_ibit_btc16=float((trk.IBIT - trk.BTC16).std() * math.sqrt(252) * 100),
        te_btc16_btc00=float((trk.BTC16 - trk.BTC00).std() * math.sqrt(252) * 100),
        td_ibit_btc16=float((trk.IBIT - trk.BTC16).mean() * 252 * 100),
        td_ibit_btc00=float((trk.IBIT - trk.BTC00).mean() * 252 * 100))
    kt = KEY["B_track"]
    w(f"Daily return correlations: IBIT–BTC00 {kt['corr_ibit_btc00']:.4f}, IBIT–BTC16 {kt['corr_ibit_btc16']:.4f}, "
      f"BTC16–BTC00 {kt['corr_btc16_btc00']:.4f}.")
    w(f"Annualized tracking error: IBIT−BTC00 {kt['te_ibit_btc00']:.2f}%, of which timing mismatch BTC16−BTC00 {kt['te_btc16_btc00']:.2f}% "
      f"and wrapper IBIT−BTC16 {kt['te_ibit_btc16']:.2f}%. Annualized tracking difference IBIT−BTC16 {kt['td_ibit_btc16']:+.2f}%.\n")
    # Variance decomposition: Var(IBIT−BTC00) = Var(timing) + Var(wrapper) + 2Cov
    dt_ = trk.BTC16 - trk.BTC00; dw_ = trk.IBIT - trk.BTC16
    tot = (trk.IBIT - trk.BTC00).var()
    KEY["B_track"].update(share_time=float(dt_.var() / tot), share_wrap=float(dw_.var() / tot),
                          share_cov=float(2 * np.cov(dt_, dw_)[0, 1] / tot))
    w(f"Variance of IBIT's deviation from BTC00: timing mismatch {KEY['B_track']['share_time'] * 100:.1f}%, "
      f"wrapper {KEY['B_track']['share_wrap'] * 100:.1f}%, covariance term {KEY['B_track']['share_cov'] * 100:+.1f}%.\n")

    # Corresponding gold decomposition: asynchronous gold price → GLD. With LBMA: LBMA (London 15:00) → futures → GLD;
    # otherwise futures → GLD.
    gchain = ("LBMA", "GC=F", "GLD") if avail("LBMA", 500) else ("GC=F", "GLD")
    gnames = ["Timing effect", "Timing + wrapper effect"] if len(gchain) == 3 else ["Timing + wrapper effect"]
    KEY["B_gold"] = decomp(gchain, "Gold: " + " → ".join(gchain), gnames, None)
    KEY["B_gold"]["chain"] = list(gchain)
    tg = R[list(gchain)].dropna()
    dgf = tg["GLD"] - tg[gchain[-2]]
    KEY["B_gold_track"] = dict(bench=gchain[-2], corr=float(tg["GLD"].corr(tg[gchain[-2]])),
                               te=float(dgf.std() * math.sqrt(252) * 100), ac1=float(dgf.autocorr(1)))
    for k in (2, 5):
        agg = (1 + tg).rolling(k).apply(np.prod, raw=True).iloc[k - 1::k] - 1
        KEY["B_gold_track"][f"corr_{k}d"] = float(agg["GLD"].corr(agg[gchain[-2]]))
    bt = KEY["B_gold_track"]
    w(f"GLD vs {bt['bench']} daily return correlation {bt['corr']:.3f} (2-day {bt['corr_2d']:.3f}, 5-day {bt['corr_5d']:.3f}); "
      f"first-order autocorrelation of the difference {bt['ac1']:+.2f} (negative = asynchronous pricing); annualized tracking error {bt['te']:.2f}%.\n")
    # Same diagnostic for Bitcoin: multi-day holding-period correlations
    for k in (2, 5):
        agg = (1 + trk).rolling(k).apply(np.prod, raw=True).iloc[k - 1::k] - 1
        KEY["B_track"][f"corr_ibit_btc00_{k}d"] = float(agg["IBIT"].corr(agg["BTC00"]))
    KEY["B_track"]["ac1_btc16_btc00"] = float(dt_.autocorr(1))
    w(f"Same diagnostic for Bitcoin: IBIT vs BTC00 correlation at 2 days {KEY['B_track']['corr_ibit_btc00_2d']:.3f}, at 5 days "
      f"{KEY['B_track']['corr_ibit_btc00_5d']:.3f}; first-order autocorrelation of BTC16−BTC00 {KEY['B_track']['ac1_btc16_btc00']:+.2f}.\n")
    if "GLDNAV" in R and R["GLDNAV"].notna().sum() > 300:
        prem = (P["GLD"] / P["GLDNAV"] - 1).reindex(R.index)
        # Note: P["GLD"] is AdjClose; GLD pays no distributions, so AdjClose = Close
        KEY["gld_premium"] = dict(mean=float(prem.mean() * 1e4), sd=float(prem.std() * 1e4),
                                  absmax=float(prem.abs().max() * 1e4))
        w(f"GLD close relative to NAV (NAV is priced off LBMA, at a different time): mean {KEY['gld_premium']['mean']:+.1f} bp, "
          f"s.d. {KEY['gld_premium']['sd']:.1f} bp.")
    navp = os.path.join(raw, f"IBIT_nav__{ASOF}.xls")
    if os.path.exists(navp):
        nav = parse_ishares_nav(navp)
        if nav is not None:
            pr = (D["IBIT"]["Close"] / nav.reindex(D["IBIT"].index) - 1).reindex(R.index).dropna()
            KEY["ibit_premium"] = dict(n=int(len(pr)), mean=float(pr.mean() * 1e4), sd=float(pr.std() * 1e4),
                                       absmax=float(pr.abs().max() * 1e4))
            w(f"IBIT closing premium to NAV ({len(pr)} days): mean {KEY['ibit_premium']['mean']:+.1f} bp, "
              f"s.d. {KEY['ibit_premium']['sd']:.1f} bp, maximum absolute value {KEY['ibit_premium']['absmax']:.1f} bp.")
        else:
            w("IBIT NAV file could not be parsed; premium analysis skipped.")

    # ------------------------------------------------------------------ C. Fund cross-section
    w("\n## C. Fund cross-section: fees, tracking, liquidity and risk\n")
    w("TD = annualized mean return difference, TE = annualized tracking error. Two benchmarks: (i) the underlying asset price: "
      "BTC16 for Bitcoin funds, LBMA for gold funds (COMEX futures if LBMA is unavailable); (ii) the flagship peer fund: IBIT for "
      "Bitcoin funds, GLD for gold funds. In (ii) both sides share the same closing time and hold the same asset, so benchmark "
      "noise cancels out; it is used to estimate the role of fees. \"Late period\" = from 2025-01-02 onward (all launch fee "
      "waivers except HODL's had ended). Dollar volume = median of close × volume ($ million); Amihud = |r| in basis points "
      "per $1 million traded.\n")
    gbench = "LBMA" if avail("LBMA", 500) else "GC=F"
    C = {}
    rows = []
    late0 = pd.Timestamp("2025-01-02")
    for t in SPOT + FUTB + GOLD_ETF:
        if not avail(t, 200):
            continue
        isb = t in SPOT + FUTB
        bench, peer = ("BTC16", "IBIT") if isb else (gbench, "GLD")
        s_ = R[[t, bench]].dropna()
        d = D[t].reindex(R.index)
        dv = (d["Close"] * d["Volume"]).dropna()
        c = dict(bench=bench, peer=peer, n=len(s_), start=str(s_.index[0].date()),
                 fee_launch=FEE.get(t, (np.nan, np.nan))[0], fee_2026=FEE.get(t, (np.nan, np.nan))[1],
                 td=float((s_[t] - s_[bench]).mean() * 252 * 100),
                 te=float((s_[t] - s_[bench]).std() * math.sqrt(252) * 100),
                 corr=float(s_[t].corr(s_[bench])),
                 dvol=float(dv.median() / 1e6), amihud=amihud(R[t], (d["Close"] * d["Volume"])))
        if t != peer:
            q = R[[t]].join(R[[peer]].rename(columns={peer: "_peer"})).dropna()
            dd = q[t] - q["_peer"]
            ql = dd[dd.index >= late0]
            c.update(td_rel=float(dd.mean() * 252 * 100), te_rel=float(dd.std() * math.sqrt(252) * 100),
                     td_rel_late=float(ql.mean() * 252 * 100), te_rel_late=float(ql.std() * math.sqrt(252) * 100),
                     se_rel_late=float(ql.std() * math.sqrt(252) * 100 / math.sqrt(len(ql) / 252)),
                     corr_rel=float(q[t].corr(q["_peer"])))
        # Portfolio ES95 ratio: Bitcoin-fund portfolio / GLD portfolio; for gold funds: IBIT portfolio / that gold fund's portfolio
        ok = R[t].notna().values
        pt = port(R[t].values)[ok]
        ref = gld_port[ok] if isb else port(R["IBIT"].values)[ok]
        m = int(ok.sum())
        idx = IDX if m == n else stationary_boot(m, a.boot, L_BOOT, np.random.default_rng(SEED + len(t)))
        if isb:
            rt = float(es(pt) / es(ref)); bs = es(pt[idx]) / es(ref[idx])
        else:
            rt = float(es(ref) / es(pt)); bs = es(ref[idx]) / es(pt[idx])
        c["port_ratio"] = rt
        c["port_ratio_ci"] = list(np.percentile(bs, [2.5, 97.5]))
        C[t] = c
        rows.append([t, c["n"], f2(c["fee_2026"]), f"{c['td']:+.2f}", f2(c["te"]), f"{c['corr']:.4f}",
                     f"{c.get('td_rel_late', 0):+.2f} (±{1.96 * c.get('se_rel_late', 0):.2f})" if t != peer else "—",
                     f2(c.get("te_rel")) if t != peer else "—",
                     f"{c['dvol']:.0f}", f2(c["amihud"], 2), ci(rt, *c["port_ratio_ci"])])
    w(md(rows, ["Fund", "Days", "Fee %", "TD% vs underlying", "TE% vs underlying", "Corr. vs underlying",
                "TD% vs flagship (late, ±95%)", "TE% vs flagship", "Dollar volume $m", "Amihud", "Portfolio ES95 ratio [CI]"]))
    KEY["C"] = C
    # Cross-section: late-period tracking difference relative to the flagship regressed on the fee difference
    spot_c = [t for t in SPOT if t in C and t != "IBIT"]
    if len(spot_c) >= 5:
        def ols(xv, yv):
            X = np.c_[np.ones(len(xv)), xv]
            bcoef, *_ = np.linalg.lstsq(X, yv, rcond=None)
            res = yv - X @ bcoef
            s2 = res @ res / (len(xv) - 2)
            se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
            return float(bcoef[1]), float(se[1]), float(bcoef[0])
        fee_rel = np.array([C[t]["fee_2026"] - FEE["IBIT"][1] for t in spot_c])
        tdr = np.array([C[t]["td_rel_late"] for t in spot_c])
        sl, sl_se, ic = ols(fee_rel, tdr)
        ex = [t for t in spot_c if t != "GBTC"]
        fe2 = np.array([C[t]["fee_2026"] - FEE["IBIT"][1] for t in ex]); td2 = np.array([C[t]["td_rel_late"] for t in ex])
        sl2, sl2_se, _ = ols(fe2, td2) if np.ptp(fe2) > 0 else (np.nan, np.nan, np.nan)
        w(f"\nRegression of spot Bitcoin ETFs' late-period tracking difference vs IBIT on the fee difference ({len(spot_c)} funds): "
          f"slope {sl:.2f} (SE {sl_se:.2f}), intercept {ic:+.2f}%. Excluding GBTC: slope {sl2:.2f} (SE {sl2_se:.2f}).")
        gbtc = C.get("GBTC")
        if gbtc:
            w(f"GBTC: fee {gbtc['fee_2026'] - FEE['IBIT'][1]:.2f} percentage points above IBIT; in the late period it trails IBIT by "
              f"{-gbtc['td_rel_late']:.2f}% per year (±{1.96 * gbtc['se_rel_late']:.2f}).")
        ldv = np.log([C[t]["dvol"] for t in spot_c + ["IBIT"]])
        te_rel = np.array([C[t]["te_rel"] for t in spot_c])
        rr_te = float(np.corrcoef(np.log([C[t]["dvol"] for t in spot_c]), te_rel)[0, 1])
        ratios = np.array([C[t]["port_ratio"] for t in spot_c + ["IBIT"]])
        rr_ratio_liq = float(np.corrcoef(ldv, ratios)[0, 1])
        full = [t for t in spot_c + ["IBIT"] if C[t]["n"] == n]
        rng_full = [min(C[t]["port_ratio"] for t in full), max(C[t]["port_ratio"] for t in full)]
        dv_rng = [min(C[t]["dvol"] for t in spot_c + ["IBIT"]), max(C[t]["dvol"] for t in spot_c + ["IBIT"])]
        te_rng = [min(te_rel), max(te_rel)]
        w(f"Tracking error vs IBIT {te_rng[0]:.2f}–{te_rng[1]:.2f}%, correlation with log dollar volume {rr_te:+.2f}; "
          f"dollar volume {dv_rng[0]:.0f}–{dv_rng[1]:.0f} $ million (a {dv_rng[1] / dv_rng[0]:.0f}-fold range); "
          f"correlation of the portfolio ES95 ratio with log dollar volume {rr_ratio_liq:+.2f}; portfolio ES95 ratio of "
          f"full-sample spot ETFs {rng_full[0]:.3f}–{rng_full[1]:.3f}.")
        KEY["C_reg"] = dict(slope=sl, slope_se=sl_se, intercept=ic, slope_ex_gbtc=sl2, slope_ex_gbtc_se=sl2_se,
                            corr_te_liq=rr_te, corr_ratio_liq=rr_ratio_liq, ratio_range_full=rng_full,
                            dvol_range=dv_rng, te_rel_range=te_rng, n_spot=len(spot_c) + 1, n_full=len(full))
        check("C. Slope of tracking difference vs IBIT on the fee difference is negative", sl < 0, f"{sl:.2f}")
    gold_c = [t for t in GOLD_ETF if t in C and t != "GLD"]
    if gold_c:
        KEY["C_gold"] = dict(te_rel_range=[min(C[t]["te_rel"] for t in gold_c), max(C[t]["te_rel"] for t in gold_c)],
                             td_rel_late={t: C[t]["td_rel_late"] for t in gold_c},
                             fee_rel={t: C[t]["fee_2026"] - FEE["GLD"][1] for t in gold_c})
        fg = np.array([C[t]["fee_2026"] - FEE["GLD"][1] for t in gold_c]); tg_ = np.array([C[t]["td_rel"] for t in gold_c])
        Xg = np.c_[np.ones(len(fg)), fg]
        bg, *_ = np.linalg.lstsq(Xg, tg_, rcond=None)
        rg = tg_ - Xg @ bg
        seg = float(np.sqrt(np.diag((rg @ rg / (len(fg) - 2)) * np.linalg.inv(Xg.T @ Xg)))[1])
        tl_ = np.array([C[t]["td_rel_late"] for t in gold_c])
        bl, *_ = np.linalg.lstsq(Xg, tl_, rcond=None)
        rl = tl_ - Xg @ bl
        sel = float(np.sqrt(np.diag((rl @ rl / (len(fg) - 2)) * np.linalg.inv(Xg.T @ Xg)))[1])
        KEY["C_gold"].update(slope=float(bg[1]), slope_se=seg, intercept=float(bg[0]),
                             slope_late=float(bl[1]), slope_late_se=sel,
                             td_rel_full={t: C[t]["td_rel"] for t in gold_c})
        w(f"Regression of gold ETFs' full-sample tracking difference vs GLD on the fee difference ({len(gold_c)} funds): "
          f"slope {bg[1]:.2f} (SE {seg:.2f}), intercept {bg[0]:+.2f}%.")
        w(f"Same regression using only data from 2025-01-02 onward: slope {bl[1]:.2f} (SE {sel:.2f}).")
        w("Gold ETFs vs GLD: tracking error {:.2f}–{:.2f}%; late-period tracking difference {}.".format(
            *KEY["C_gold"]["te_rel_range"],
            ", ".join(f"{t} {C[t]['td_rel_late']:+.2f}% (fee difference {C[t]['fee_2026'] - FEE['GLD'][1]:+.2f})" for t in gold_c)))

    # ------------------------------------------------------------------ D. Trading hours
    w("\n## D. Trading hours\n")
    prev = pd.Series(cal, index=cal).shift(1)
    gap = ((pd.Series(cal, index=cal) - prev).dt.days > 1).reindex(R.index).values
    w(f"\"First day after a market closure\" (the previous calendar day is a non-trading day: Mondays and post-holiday days): "
      f"{int(gap.sum())} days; other days: {int((~gap).sum())}.\n")
    rows = []
    KEY["D_gap"] = {}
    rs = np.random.default_rng(SEED + 1)
    ig, inn = np.where(gap)[0], np.where(~gap)[0]
    for t in ("BTC16", "IBIT", "BTC00", "GLD", "LBMA", "SPY", "TLT"):
        if not avail(t, 500):
            continue
        x = R[t].values
        g_, o_ = x[ig], x[inn]
        okg, oko = np.isfinite(g_), np.isfinite(o_)
        vr = float(np.var(g_[okg], ddof=1) / np.var(o_[oko], ddof=1))
        bs = []
        for _ in range(a.boot):
            a1 = rs.choice(g_[okg], okg.sum()); a2 = rs.choice(o_[oko], oko.sum())
            bs.append(np.var(a1, ddof=1) / np.var(a2, ddof=1))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        KEY["D_gap"][t] = dict(vr=vr, ci=[lo, hi])
        rows.append([t, f"{np.std(g_[okg], ddof=1) * 100:.2f}", f"{np.std(o_[oko], ddof=1) * 100:.2f}", ci(vr, lo, hi)])
    w(md(rows, ["Series", "First day after closure: daily s.d. %", "Other days: daily s.d. %", "Variance ratio [95% CI]"]))

    # Overnight vs intraday (AdjClose basis: the open is scaled by the same day's adjustment factor)
    w("Variance shares of overnight (previous close → today's open) and intraday (today's open → today's close) returns:\n")
    rows = []
    KEY["D_on"] = {}
    for t in ("IBIT", "FBTC", "GBTC", "BITO", "GLD", "GC=F", "SPY", "QQQ", "TLT"):
        if D.get(t) is None:
            continue
        d = D[t].reindex(cal)
        f = d["AdjClose"] / d["Close"]
        on = (d["Open"] * f / d["AdjClose"].shift(1) - 1).reindex(R.index)
        idr = (d["Close"] / d["Open"] - 1).reindex(R.index)
        ok = on.notna() & idr.notna()
        if ok.sum() < 200:
            continue
        v_on, v_id = on[ok].var(), idr[ok].var()
        share = float(v_on / (v_on + v_id))
        share_gap = float(on[ok & gap].var() / (on[ok & gap].var() + idr[ok & gap].var()))
        KEY["D_on"][t] = dict(share=share, share_gapdays=share_gap, n=int(ok.sum()))
        rows.append([t, int(ok.sum()), f"{share * 100:.1f}", f"{share_gap * 100:.1f}"])
    w(md(rows, ["Ticker", "Days", "Overnight variance share %", "Same, first days after closure %"]))

    # Hourly Bitcoin: share of realized variance during hours when the US equity market is closed
    h = H[hsrc]["close"]
    lr = np.log(h).diff().dropna()
    lr = lr[(lr.index >= (START - pd.Timedelta(days=1)).tz_localize("UTC")) &
            (lr.index < (END + pd.Timedelta(days=1)).tz_localize("UTC"))]
    et = lr.index.tz_convert("America/New_York")
    # Hourly bars whose *end* time is 10:00–16:00 ET (i.e. starting 09:00–15:00) on a trading day -> approximate US trading hours
    end_et = et + pd.Timedelta(hours=1)
    tday = pd.DatetimeIndex(end_et.tz_localize(None).normalize()).isin(cal)
    hr = end_et.hour + end_et.minute / 60
    open_ = tday & (hr > 9.5) & (hr <= 16)
    sq = lr.values ** 2
    share_closed = float(sq[~open_].sum() / sq.sum())
    hours_closed = float((~open_).mean())
    wkend = et.dayofweek >= 5
    wk_ratio = float((sq[wkend]).mean() / (sq[~wkend]).mean())
    KEY["D_hourly"] = dict(share_var_closed=share_closed, share_hours_closed=hours_closed, weekend_ratio=wk_ratio)
    w(f"The mean squared hourly return at weekends (US Eastern Saturday and Sunday) is {wk_ratio:.2f} times that on weekdays; "
      f"if weekends were as volatile as weekdays, the Monday variance ratio would be about 3; it is in fact about "
      f"1 + 2 × {wk_ratio:.2f} ≈ {1 + 2 * wk_ratio:.2f}.")
    w(f"\nHourly Bitcoin: hours when the US equity market is closed make up {hours_closed * 100:.1f}% of all hours and "
      f"contribute {share_closed * 100:.1f}% of realized variance"
      " (trading hours approximated at hourly resolution as 09:00–16:00 ET).")

    # ------------------------------------------------------------------ E. Portfolio grids
    w("\n## E. Portfolio level\n")
    w("### E1. Bitcoin instruments × gold instruments (equity SPY, bonds TLT), ES95 ratio\n")
    btc_list = [t for t in ["BTC00", "BTC16"] + SPOT + FUTB if avail(t)]
    gold_list = [t for t in ["LBMA", "GC=F"] + GOLD_ETF if avail(t)]
    E1 = {}
    for b in btc_list:
        for g in gold_list:
            ok = R[[b, g]].notna().all(1).values
            x1, x2 = port(R[b].values)[ok], port(R[g].values)[ok]
            m = int(ok.sum())
            if m == n:
                bs = es(x1[IDX]) / es(x2[IDX])
            else:
                ii = stationary_boot(m, a.boot, L_BOOT, np.random.default_rng(SEED + m))
                bs = es(x1[ii]) / es(x2[ii])
            E1[f"{b}|{g}"] = dict(n=m, ratio=float(es(x1) / es(x2)), ci=list(np.percentile(bs, [2.5, 97.5])))
    KEY["E1"] = E1
    head = ["Bitcoin \\ Gold"] + gold_list
    rows = [[b] + [f"{E1[f'{b}|{g}']['ratio']:.3f}" for g in gold_list] for b in btc_list]
    w(md(rows, head))
    full_pairs = [v["ratio"] for k, v in E1.items() if v["n"] == n]
    KEY["E1_range_full"] = [min(full_pairs), max(full_pairs), len(full_pairs)]
    w(f"Full-sample portfolios ({len(full_pairs)} pairs), ES95 ratio: {min(full_pairs):.3f} – {max(full_pairs):.3f}.")
    spot_pairs = [v["ratio"] for k, v in E1.items() if v["n"] == n and k.split("|")[0] in SPOT
                  and k.split("|")[1] in GOLD_ETF]
    if spot_pairs:
        KEY["E1_range_spot_etf"] = [min(spot_pairs), max(spot_pairs), len(spot_pairs)]
        w(f"Of which spot Bitcoin ETFs × gold ETFs ({len(spot_pairs)} pairs): {min(spot_pairs):.3f} – {max(spot_pairs):.3f}.")
    all_lo = min(v["ci"][0] for v in E1.values())
    check("E1. Lower CI bound of the ES95 ratio > 1 for all pairs", all_lo > 1, f"smallest lower bound {all_lo:.2f}")

    w("\n### E2. Equity × bond benchmark grid (IBIT portfolio / GLD portfolio)\n")
    E2 = {}
    rows = []
    for eq in EQUITY:
        if not avail(eq):
            continue
        row = [eq]
        for bd in BOND:
            if not avail(bd):
                row.append("—"); continue
            x1 = (R["IBIT"].values + R[eq].values + R[bd].values) / 3
            x2 = (R["GLD"].values + R[eq].values + R[bd].values) / 3
            ok = np.isfinite(x1) & np.isfinite(x2)
            x1, x2 = x1[ok], x2[ok]
            ii = IDX if ok.all() else stationary_boot(int(ok.sum()), a.boot, L_BOOT, np.random.default_rng(SEED + 7))
            bs = es(x1[ii]) / es(x2[ii])
            rt = float(es(x1) / es(x2)); vr = float(vol(x1) / vol(x2))
            E2[f"{eq}|{bd}"] = dict(ratio=rt, ci=list(np.percentile(bs, [2.5, 97.5])), vol_ratio=vr,
                                    es_ibit=float(es(x1) * 100), es_gld=float(es(x2) * 100),
                                    es99_ratio=float(es(x1, .01) / es(x2, .01)))
            row.append(ci(rt, *E2[f"{eq}|{bd}"]["ci"]))
        rows.append(row)
    w(md(rows, ["Equity \\ Bond"] + BOND))
    KEY["E2"] = E2
    rr = [v["ratio"] for v in E2.values()]
    KEY["E2_range"] = [min(rr), max(rr)]
    w(f"Across the 25 benchmark portfolios, ES95 ratio: {min(rr):.2f} – {max(rr):.2f}; smallest lower CI bound "
      f"{min(v['ci'][0] for v in E2.values()):.2f}.")
    for bd in BOND:
        vs = [E2[f"{eq}|{bd}"]["ratio"] for eq in EQUITY if f"{eq}|{bd}" in E2]
        w(f"  Bond leg {bd}: {min(vs):.2f} – {max(vs):.2f}")
    check("E2. Under all 25 benchmark combinations the IBIT portfolio ES95 exceeds that of the GLD portfolio", min(rr) > 1, f"{min(rr):.2f}")

    # ------------------------------------------------------------------ F. Correlation with broader benchmarks
    w("\n## F. Single-asset correlation ρ with broader benchmarks (β in parentheses)\n")
    F = {}
    bm = [t for t in EQUITY + BOND if avail(t)]
    rows = []
    for t in ("BTC00", "BTC16", "IBIT", "BITO", "LBMA", "GC=F", "GLD"):
        if not avail(t):
            continue
        row = [t]
        F[t] = {}
        for b in bm:
            s = R[[t, b]].dropna()
            F[t][b] = dict(rho=float(corr(s[t].values, s[b].values)), beta=float(beta(s[t].values, s[b].values)))
            row.append(f"{F[t][b]['rho']:.2f} ({F[t][b]['beta']:.2f})")
        rows.append(row)
    w(md(rows, ["Ticker"] + bm))
    KEY["F"] = F

    # ------------------------------------------------------------------ Summary
    w("\n## Self-check summary\n")
    for c in CHECKS:
        w("  " + c)
    w(f"\nElapsed time {time.time() - t0:.0f} s")
    with open(os.path.join(a.outdir, "scope_expansion.md"), "w", encoding="utf8") as fh:
        fh.write("\n".join(LINES) + "\n")
    with open(os.path.join(a.outdir, "scope_expansion.json"), "w", encoding="utf8") as fh:
        json.dump(KEY, fh, indent=1, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
