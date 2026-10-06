"""
08_test_gspc.py — Determine: is the SPY column in the original file the ETF or the index?

Background
    The SPY column in the original file has two types of differences from the SPY ETF
    in the new snapshot:
      (a) a systematic difference of +1.51pp/yr over the whole sample — exactly equal to
          SPY's dividend yield
      (b) a single-day difference of 0.987pp on 2025-04-09 — the original file shows
          +9.515%, which matches the S&P 500 index's +9.52% that day (recorded by CNBC/AP
          and others as the third-largest single-day gain since WWII)

    If the SPY column in the original file is actually the ^GSPC index, then both (a) and
    (b) are explained at once: the index has no dividends (so the annualized figure equals
    the price return), and 4/9 naturally matches the index.

    This script compares the original file's SPY column against both ^GSPC and the SPY ETF
    across four conventions, and gives a verdict.

Usage
    python 08_test_gspc.py
    python 08_test_gspc.py --old "/path/to/ETF_Returns.csv" --raw data/raw --asof 20260921
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

NEED = {"Date", "IBIT", "GLD", "TLT", "SPY"}


def find_old(path):
    if os.path.exists(path):
        return path
    print(f"  [!] {path} does not exist, searching by column names...")
    for roots in ([os.getcwd()],
                  [os.path.dirname(os.path.abspath(__file__))]):
        out, seen = [], set()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dp, dn, fn in os.walk(root):
                dn[:] = [x for x in dn if not x.startswith(".")
                         and x not in ("venv", ".venv", "__pycache__", "site-packages")]
                if dp.count(os.sep) - root.count(os.sep) > 3:
                    dn[:] = []
                    continue
                for f in fn:
                    if not f.lower().endswith(".csv"):
                        continue
                    p = os.path.realpath(os.path.join(dp, f))
                    if p in seen or os.path.basename(os.path.dirname(p)) == "processed":
                        continue
                    seen.add(p)
                    try:
                        if NEED.issubset(set(pd.read_csv(p, nrows=0).columns)):
                            out.append(p)
                    except Exception:
                        pass
        if out:
            break
    if len(out) == 1:
        print(f"  [OK] Auto-selected: {out[0]}")
        return out[0]
    if len(out) > 1:
        print("  Multiple candidates, please specify with --old:")
        for p in out[:10]:
            print("       ", p)
    else:
        print("  Not found, please specify with --old")
    sys.exit(1)


def conventions(px):
    """Generate return series under four conventions from a price table."""
    out = {}
    for col in ("AdjClose", "Close"):
        if col not in px.columns:
            continue
        p = px[col]
        out[(col, "simple")] = p.pct_change().dropna()
        out[(col, "log")] = np.log(p / p.shift(1)).dropna()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="ETF_Returns.csv")
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default="20260921", help="snapshot suffix of the main sample")
    a = ap.parse_args()

    if a.asof is None:
        m = sorted(glob.glob(os.path.join(a.raw, "manifest__*.json")))
        if m:
            a.asof = os.path.basename(m[-1]).split("__")[1].split(".")[0]
        else:
            c = sorted(glob.glob(os.path.join(a.raw, "SPY__*.csv")))
            if c:
                a.asof = os.path.basename(c[-1]).split("__")[1].split(".")[0]
    a.old = find_old(a.old)

    old = pd.read_csv(a.old, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    print(f"  Original file: {a.old}   ({len(old)} rows)")
    print(f"  Snapshot date: {a.asof}\n")

    src = {}
    for key, fn in (("SPY ETF", f"SPY__{a.asof}.csv"),
                    ("^GSPC index", f"IDX_GSPC__{a.asof}.csv")):
        p = os.path.join(a.raw, fn)
        if not os.path.exists(p):
            print(f"  [!] Missing {p}")
            if key == "^GSPC index":
                print("      Please add \"^GSPC\": \"S&P 500 Index\" to TICKERS in 01a_fetch_raw.py")
                print("      and rerun")
            sys.exit(1)
        src[key] = pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").set_index("Date")

    print("=" * 84)
    print("Verdict: comparison of the original file's SPY column against the two candidate sources (max absolute deviation)")
    print("=" * 84)
    print(f"  {'Source':<14}{'AdjClose+simple':>16}{'AdjClose+log':>16}{'Close+simple':>16}{'Close+log':>16}")
    best = (None, np.inf)
    for key, px in src.items():
        line = []
        for k in [("AdjClose", "simple"), ("AdjClose", "log"),
                  ("Close", "simple"), ("Close", "log")]:
            cv = conventions(px)
            if k not in cv:
                line.append(np.nan)
                continue
            x, y = old["SPY"].align(cv[k], join="inner")
            v = (x - y).abs().max() if len(x) else np.nan
            line.append(v)
            if np.isfinite(v) and v < best[1]:
                best = ((key, k), v)
        print(f"  {key:<14}" + "".join(f"{v:>16.2e}" for v in line))

    print()
    if best[1] < 1e-6:
        (srcname, conv) = best[0]
        print(f"  [Verdict] The original file's SPY column = [{srcname}] {conv[0]} + "
              f"{'simple returns' if conv[1]=='simple' else 'log returns'}   (deviation {best[1]:.2e})")
        if "GSPC" in srcname:
            print()
            print("  ** The original file uses the S&P 500 [index], not the SPY ETF. **")
            print("     Section 2.1 of the paper says \"the S&P 500 equity ETF (SPY, SPDR S&P 500")
            print("     ETF Trust)\", which does not match the actual data and must be corrected.")
            print("     Two possible fixes:")
            print("       (1) Change the data: switch to the SPY ETF's AdjClose total return and")
            print("           rerun all results (recommended, because the paper's subject is an ETF")
            print("           portfolio, so using ETF data is self-consistent);")
            print("       (2) Change the description: state plainly that the equity leg uses the")
            print("           index rather than the ETF, and explain why.")
            print("     Note: the index is not directly investable and has no fees and no dividends;")
            print("     using it for portfolio performance evaluation would overstate achievable")
            print("     returns, so (1) is safer.")
    else:
        print(f"  [Verdict] Neither source matches under any of the four conventions (minimum deviation {best[1]:.2e}).")
        print("        This means the original file's SPY column comes from an earlier snapshot")
        print("        or another data source.")
        print("        It is recommended to switch directly to the new snapshot's SPY ETF AdjClose")
        print("        and rerun.")

    # ---- Day-by-day comparison around the disputed date ----
    print("\n" + "=" * 84)
    print("Four days around 2025-04-09: the three series side by side")
    print("=" * 84)
    rows = pd.date_range("2025-04-07", "2025-04-11", freq="B")
    print(f"  {'Date':<12}{'Original':>10}{'SPY ETF':>11}{'^GSPC':>10}{'ETF-orig':>10}{'Index-orig':>10}")
    for dt in rows:
        if dt not in old.index:
            continue
        o = old["SPY"][dt] * 100
        vals = {}
        for key, px in src.items():
            r = px["AdjClose"].pct_change()
            vals[key] = r[dt] * 100 if dt in r.index else np.nan
        e, g = vals.get("SPY ETF", np.nan), vals.get("^GSPC index", np.nan)
        print(f"  {str(dt.date()):<12}{o:>9.3f}%{e:>10.3f}%{g:>9.3f}%{e-o:>+10.3f}{g-o:>+10.3f}")
    print("\n  External fact: the S&P 500 index rose 9.52% on 2025-04-09 (recorded by CNBC/AP and others)")

    # ---- Full-sample convention impact ----
    print("\n" + "=" * 84)
    print("Annualized returns of the three series (same calendar)")
    print("=" * 84)
    cal = old.index
    print(f"  {'Series':<26}{'Ann.%':>10}{'Ann.vol%':>12}{'Skew':>9}{'Kurt':>9}")
    series = {"Original SPY column": old["SPY"]}
    for key, px in src.items():
        for col in ("AdjClose", "Close"):
            if col in px.columns:
                s = px[col].pct_change().reindex(cal).dropna()
                if (px["Close"] - px["AdjClose"]).abs().max() < 1e-8 and col == "Close":
                    continue   # No-dividend instrument, both columns identical, report once only
                series[f"{key} / {col}"] = s
    for k, s in series.items():
        yrs = len(s) / 252
        print(f"  {k:<26}{((1+s).prod()**(1/yrs)-1)*100:>10.2f}"
              f"{s.std()*np.sqrt(252)*100:>12.2f}{s.skew():>9.2f}{s.kurt():>9.2f}")
    print("\n  The index and ETF-Close should differ by about 0.09%/yr (ETF expense ratio);")
    print("  ETF-AdjClose and ETF-Close should differ by about 1.2%/yr (dividends).")


if __name__ == "__main__":
    main()