"""
06_audit_spy.py — Old vs new data audit: locate differences, determine convention, quantify impact on the paper

Background
    After rerunning 01/02, the four moments of IBIT / GLD / TLT match the
    original ETF_Returns.csv exactly, but SPY does not, and the magnitude of the
    difference (kurtosis 16.73 -> 21.69) far exceeds what dividends can explain.
    This script answers three questions:

      1. Which days differ? Is it a difference in values or in the date set?
      2. Which convention does each column of the original file come from?
         (AdjClose/Close x simple/log, one of four)
      3. If the new data is used instead, which numbers in the paper change, and
         by how much?

Usage
    python 06_audit_spy.py
    python 06_audit_spy.py --old ETF_Returns.csv --new data/processed/returns__20260921.csv \\
                           --raw data/raw --asof 20260921

Dependencies: pandas, numpy, scipy (the optimized-weights section is skipped if
scipy is missing)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

TICKERS = ["IBIT", "GLD", "TLT", "SPY"]
TOL = 1e-9


# ---------------------------------------------------------------- Utilities ----
def hr(ch="=", n=84):
    print(ch * n)


def head(title):
    print()
    hr()
    print(title)
    hr()


NEED_COLS = {"Date", "IBIT", "GLD", "TLT", "SPY"}


def find_old_candidates(limit=400):
    """Search common locations for CSVs containing the Date/IBIT/GLD/TLT/SPY columns.

    Filenames may contain spaces, differ in case, or live in other directories,
    so identification is by [column names] rather than by filename.
    """
    # Tiered search: look only in the working directory first; if there is a hit,
    # do not search further, to avoid picking up identically named data from
    # sibling projects
    tiers = [[os.getcwd()],
             [os.path.dirname(os.path.abspath(__file__))]]
    for roots in tiers:
        res = _scan(roots, limit)
        if res:
            return res
    return []


def _scan(roots, limit):
    seen, out, checked = set(), [], 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames
                           if not x.startswith(".") and x not in
                           ("node_modules", "__pycache__", "venv", ".venv", "site-packages")]
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                dirnames[:] = []
                continue
            for fn in filenames:
                if not fn.lower().endswith(".csv"):
                    continue
                p = os.path.realpath(os.path.join(dirpath, fn))
                if p in seen:
                    continue
                seen.add(p)
                checked += 1
                if checked > limit:
                    return out
                # Files under processed/ are new files, exclude them
                if os.path.basename(os.path.dirname(p)) == "processed":
                    continue
                try:
                    cols = set(pd.read_csv(p, nrows=0).columns)
                except Exception:
                    continue
                if NEED_COLS.issubset(cols):
                    out.append(p)
    return out


def autodetect(args):
    """Try to find the three inputs automatically; report clearly if not found."""
    print(f"  Working directory: {os.getcwd()}")
    if args.new is None:
        cands = sorted(glob.glob(os.path.join("data", "processed", "returns__*.csv")))
        if cands:
            args.new = cands[-1]
    if args.asof is None and args.new:
        base = os.path.basename(args.new)
        if "__" in base:
            args.asof = base.split("__")[1].split(".")[0]

    if not os.path.exists(args.old):
        print(f"  [!] {args.old} does not exist, searching for candidates by column name...")
        c = find_old_candidates()
        if len(c) == 1:
            args.old = c[0]
            print(f"  [OK] Auto-selected: {args.old}")
        elif len(c) > 1:
            print("  Found multiple candidates, please specify one with --old:")
            for p in c[:10]:
                try:
                    n = len(pd.read_csv(p))
                except Exception:
                    n = "?"
                print(f"        {p}   ({n} rows)")
            sys.exit(1)
        else:
            print("  Not found. Please specify explicitly with --old, e.g.:")
            print('        python 06_audit_spy.py --old "/path/to/ETF_Returns.csv"')
            sys.exit(1)
    return args


def load_returns(path, label):
    if not path or not os.path.exists(path):
        print(f"  [!] Cannot find {label}: {path}")
        return None
    d = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    d = d.set_index("Date")
    print(f"  {label}: {path}")
    print(f"      {len(d)} rows  {d.index[0].date()} -> {d.index[-1].date()}  columns: {list(d.columns)[:8]}"
          + (" ..." if len(d.columns) > 8 else ""))
    return d


def moments(r):
    r = pd.Series(r).dropna()
    yrs = len(r) / 252
    cum = (1 + r).prod()
    c = (1 + r).cumprod()
    return dict(n=len(r),
                ann=(cum ** (1 / yrs) - 1) * 100,
                vol=r.std() * np.sqrt(252) * 100,
                skew=r.skew(), kurt=r.kurt(),
                mdd=(c / c.cummax() - 1).min() * 100,
                mn=r.min() * 100, mx=r.max() * 100)


# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="ETF_Returns.csv")
    ap.add_argument("--new", default=None)
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default=None)
    args = autodetect(ap.parse_args())

    head("Section 0  Input files")
    old = load_returns(args.old, "Original file (old)")
    new = load_returns(args.new, "New file (new)")
    if old is None or new is None:
        print("\nPlease specify the paths explicitly with --old / --new and retry.")
        sys.exit(1)

    # The new file contains _log columns; take the simple return columns
    new_simple = new[[c for c in TICKERS if c in new.columns]].copy()

    # ---------------------------------------------------------------- 1 --
    head("Section 1  Date set comparison")
    io, iN = set(old.index), set(new_simple.index)
    print(f"  old {len(io)} days, new {len(iN)} days, intersection {len(io & iN)} days")
    only_o, only_n = sorted(io - iN), sorted(iN - io)
    if not only_o and not only_n:
        print("  [OK] Date sets are identical -> the difference comes from [values], not [dates]")
    else:
        print(f"  [!!] Only in old ({len(only_o)}): {[str(x.date()) for x in only_o[:12]]}")
        print(f"  [!!] Only in new ({len(only_n)}): {[str(x.date()) for x in only_n[:12]]}")
    common = sorted(io & iN)

    # ---------------------------------------------------------------- 2 --
    head("Section 2  Day-by-day difference localization")
    bad_dates = {}
    for c in TICKERS:
        if c not in old.columns or c not in new_simple.columns:
            print(f"  {c}: one side is missing this column, skipping")
            continue
        a = old.loc[common, c]
        b = new_simple.loc[common, c]
        diff = (a - b).abs()
        bad = diff[diff > TOL]
        bad_dates[c] = bad
        status = "OK " if len(bad) == 0 else "!!!"
        print(f"  [{status}] {c:5s} {len(bad):3d}/{len(a)} days inconsistent   max deviation {diff.max():.3e}")
        if len(bad):
            print(f"        {'Date':<12}{'old':>10}{'new':>10}{'diff(pp)':>10}")
            for dt in bad.sort_values(ascending=False).index[:15]:
                print(f"        {str(dt.date()):<12}{a[dt]*100:>9.3f}%{b[dt]*100:>9.3f}%"
                      f"{(a[dt]-b[dt])*100:>10.3f}")
            if len(bad) > 15:
                print(f"        ... {len(bad)-15} more days, full list in audit_diff.csv")

    if "VIX_Close" in old.columns and "VIX_Close" in new.columns:
        a = old.loc[common, "VIX_Close"]; b = new.loc[common, "VIX_Close"]
        nan_new = b.isna().sum()
        d2 = (a - b).abs()
        print(f"\n  VIX_Close: new is missing {nan_new} days; max deviation where present {d2.max():.4f}")
        if nan_new:
            print(f"        Dates missing in new: {[str(x.date()) for x in b[b.isna()].index[:12]]}")
            print("        -> old has values on these dates, indicating filling was done during")
            print("           manual merging; the filling method needs to be documented")

    # ---------------------------------------------------------------- 3 --
    head("Section 3  Convention fingerprint: which convention does each column of the original file come from?")
    print("  Recompute returns under the four conventions from the raw snapshots and")
    print("  compare with old. A deviation of ~1e-16 identifies the true source.\n")
    raw_ok = True
    RAW = {}
    for t in TICKERS:
        f = os.path.join(args.raw, f"{t}__{args.asof}.csv")
        if not os.path.exists(f):
            print(f"  [!] Missing raw snapshot {f}")
            raw_ok = False
            break
        RAW[t] = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date").set_index("Date")

    if raw_ok:
        cal = None
        for t in TICKERS:
            cal = RAW[t].index if cal is None else cal.intersection(RAW[t].index)
        print(f"  Raw snapshot intersection calendar {len(cal)} days -> {len(cal)-1} returns\n")
        print(f"  {'Ticker':<6}{'AdjClose+simple':>16}{'AdjClose+log':>16}{'Close+simple':>16}{'Close+log':>16}  Verdict")
        verdict = {}
        for t in TICKERS:
            px = RAW[t].loc[cal]
            res = {}
            for col in ("AdjClose", "Close"):
                p = px[col]
                res[(col, "simple")] = p.pct_change().dropna()
                res[(col, "log")] = np.log(p / p.shift(1)).dropna()
            line, best, bv = [], None, np.inf
            for key in [("AdjClose", "simple"), ("AdjClose", "log"),
                        ("Close", "simple"), ("Close", "log")]:
                s = res[key]
                a, b = old[t].align(s, join="inner")
                v = (a - b).abs().max() if len(a) else np.nan
                line.append(v)
                if np.isfinite(v) and v < bv:
                    bv, best = v, key
            verdict[t] = (best, bv)
            if bv < 1e-6:
                tag = f"{best[0]}+{'simple' if best[1]=='simple' else 'log'}"
                # For tickers with no distributions, Close == AdjClose, so both match
                # at once; note this
                ties = sum(1 for v in line if np.isfinite(v) and v < 1e-6)
                if ties > 1:
                    tag += " (this ticker has no distributions, Close == AdjClose, equivalent)"
            else:
                tag = "**no match**"
            print(f"  {t:<6}" + "".join(f"{v:>16.2e}" for v in line) + f"  {tag}")
        print()
        matched = [t for t, (b, v) in verdict.items() if v < 1e-6]
        unmatched = [t for t, (b, v) in verdict.items() if v >= 1e-6]
        if matched:
            print(f"  Matched: {', '.join(matched)} -> convention = "
                  f"{verdict[matched[0]][0][0]} + "
                  f"{'simple returns' if verdict[matched[0]][0][1]=='simple' else 'log returns'}")
        if unmatched:
            print(f"  [!!] None of the four conventions match: {', '.join(unmatched)}")
            print("       -> This column was not generated from this snapshot. Most likely causes:")
            print("         (a) The original file was fetched earlier, and Yahoo has since")
            print("             revised prices on these trading days;")
            print("         (b) This column was taken from a different download during manual merging.")
            print("       Either way, reproducibility requires using the new snapshot and")
            print("       rerunning all results.")
    else:
        print("  Skipping this section (specify --raw / --asof to enable)")

    # ---------------------------------------------------------------- 4 --
    head("Section 4  Descriptive statistics comparison")
    print(f"  {'Ticker':<6}{'':<4}{'Ann%':>9}{'Vol%':>9}{'Skew':>8}{'Kurt':>8}{'MaxDD%':>11}{'Min%':>8}{'Max%':>8}")
    for t in TICKERS:
        for lab, src in (("old", old[t]), ("new", new_simple[t])):
            m = moments(src)
            print(f"  {t if lab=='old' else '':<6}{lab:<4}{m['ann']:>9.2f}{m['vol']:>9.2f}"
                  f"{m['skew']:>8.2f}{m['kurt']:>8.2f}{m['mdd']:>11.2f}{m['mn']:>8.2f}{m['mx']:>8.2f}")
        print()

    # ---------------------------------------------------------------- 5 --
    head("Section 5  Impact on the paper's numbers (old -> new)")

    def paper_numbers(df, vix):
        P = {"IBIT portfolio": (df.IBIT + df.SPY + df.TLT) / 3,
             "GLD portfolio": (df.GLD + df.SPY + df.TLT) / 3}
        out = {"port": {}, "reg": {}, "cor": {}, "worst20": {}, "tdf": {}}
        for k, r in P.items():
            m = moments(r)
            out["port"][k] = (m["ann"], m["vol"], m["mdd"], m["ann"] / m["vol"])
            try:
                from scipy import stats as st
                out["tdf"][k] = st.t.fit(r)[0]
            except Exception:
                out["tdf"][k] = np.nan
        regs = {"low vol": vix < 15, "normal": (vix >= 15) & (vix <= 25), "high vol": vix > 25}
        for rg, m_ in regs.items():
            m_ = m_.reindex(df.index).fillna(False).values
            out["reg"][rg] = {}
            for k, r in P.items():
                x = r[m_]
                if len(x) < 5:
                    continue
                v = -np.percentile(x, 5) * 100
                e = -x[x <= np.percentile(x, 5)].mean() * 100
                out["reg"][rg][k] = (v, e, int(m_.sum()))
            out["cor"][rg] = (df.IBIT[m_].corr(df.SPY[m_]), df.GLD[m_].corr(df.SPY[m_]))
        w = df.nsmallest(20, "SPY")
        out["worst20"] = (w.SPY.mean() * 100, w.IBIT.mean() * 100, w.GLD.mean() * 100)
        return out

    vix_old = old["VIX_Close"] if "VIX_Close" in old else None
    vix_new = new["VIX_Close"] if "VIX_Close" in new else vix_old
    if vix_new is not None and vix_new.isna().any():
        print(f"  Note: new VIX has {int(vix_new.isna().sum())} missing values; this section "
              f"uses old VIX for comparability\n")
        vix_new = vix_old
    O = paper_numbers(old.loc[common], vix_old.loc[common])
    N = paper_numbers(new_simple.loc[common], vix_new.loc[common])

    print("  Section 3.1  Portfolio performance")
    print(f"    {'Portfolio':<10}{'':<5}{'Ann%':>9}{'Vol%':>9}{'DD%':>9}{'Sharpe(rf=0)':>12}")
    for k in O["port"]:
        for lab, S in (("old", O), ("new", N)):
            a, v, d_, s = S["port"][k]
            print(f"    {k if lab=='old' else '':<10}{lab:<5}{a:>9.2f}{v:>9.2f}{d_:>9.2f}{s:>12.2f}")
    print(f"\n  Section 3.2  t degrees of freedom   IBIT portfolio {O['tdf']['IBIT portfolio']:.2f} -> {N['tdf']['IBIT portfolio']:.2f}"
          f"    GLD portfolio {O['tdf']['GLD portfolio']:.2f} -> {N['tdf']['GLD portfolio']:.2f}")

    print("\n  Section 3.5  Regime-conditional VaR95 / ES95")
    print(f"    {'Regime':<16}{'Portfolio':<10}{'old VaR':>9}{'new VaR':>9}{'old ES':>9}{'new ES':>9}")
    for rg in O["reg"]:
        first = True
        for k in O["reg"][rg]:
            ov, oe, nn = O["reg"][rg][k]
            nv, ne, _ = N["reg"][rg][k]
            lbl = f"{rg} (n={nn})" if first else ""
            first = False
            print(f"    {lbl:<16}{k:<10}{ov:>9.2f}{nv:>9.2f}{oe:>9.2f}{ne:>9.2f}")

    print("\n  Section 3.5  Regime-conditional correlations (IBIT-SPY / GLD-SPY)")
    for rg in O["cor"]:
        print(f"    {rg:<8} old {O['cor'][rg][0]:+.3f} / {O['cor'][rg][1]:+.3f}"
              f"    new {N['cor'][rg][0]:+.3f} / {N['cor'][rg][1]:+.3f}")

    print("\n  Section 3.5  Worst 20 days mean (SPY / IBIT / GLD)")
    print(f"    old {O['worst20'][0]:+.2f}% / {O['worst20'][1]:+.2f}% / {O['worst20'][2]:+.2f}%")
    print(f"    new {N['worst20'][0]:+.2f}% / {N['worst20'][1]:+.2f}% / {N['worst20'][2]:+.2f}%")

    # Table 3 optimized weights
    try:
        from scipy import optimize
        print("\n  Table 3  Optimized weights (%)")
        A = ["GLD", "SPY", "IBIT", "TLT"]
        cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
        bnds = [(0, 1)] * 4
        w0 = np.repeat(.25, 4)

        def solve(mu, S, f):
            return optimize.minimize(f, w0, bounds=bnds, constraints=cons,
                                     method="SLSQP", options={"ftol": 1e-12, "maxiter": 1000}).x
        print(f"    {'Objective':<14}{'':<5}" + "".join(f"{a:>9}" for a in A))
        for lab, df in (("old", old.loc[common]), ("new", new_simple.loc[common])):
            R = df[A].values
            mu = R.mean(0) * 252
            S = np.cov(R.T) * 252
            mv = solve(mu, S, lambda w: w @ S @ w)
            ms = solve(mu, S, lambda w: -(mu @ w) / np.sqrt(w @ S @ w))
            if lab == "old":
                print(f"    {'Max Sharpe(rf=0)':<14}{lab:<5}" + "".join(f"{x*100:>9.2f}" for x in ms))
            else:
                print(f"    {'':<14}{lab:<5}" + "".join(f"{x*100:>9.2f}" for x in ms))
            if lab == "new":
                print()
        for lab, df in (("old", old.loc[common]), ("new", new_simple.loc[common])):
            R = df[A].values
            S = np.cov(R.T) * 252
            mv = solve(R.mean(0) * 252, S, lambda w: w @ S @ w)
            tag = "Min variance" if lab == "old" else ""
            print(f"    {tag:<14}{lab:<5}" + "".join(f"{x*100:>9.2f}" for x in mv))
    except ImportError:
        print("\n  (scipy not installed, skipping optimized weights)")

    # ---------------------------------------------------------------- 6 --
    head("Section 6  Conclusions and recommendations")
    tot_bad = {c: len(b) for c, b in bad_dates.items()}
    clean = [c for c, k in tot_bad.items() if k == 0]
    dirty = [c for c, k in tot_bad.items() if k > 0]
    print(f"  Fully identical: {', '.join(clean) if clean else 'none'}")
    print(f"  Differences present: {', '.join(f'{c}({tot_bad[c]} days)' for c in dirty) if dirty else 'none'}")
    print()
    if dirty:
        print("  Recommendations:")
        print("   1. Use the [new snapshot]. It has a SHA-256 manifest and a reproducible")
        print("      script; the original file has neither.")
        print("   2. Rerun all of Sections 3.1 / 3.2 / 3.5 / Table 3 with the new data.")
        print("      SPY has identical weights in both portfolios, so directional conclusions")
        print("      are expected to be unchanged, but all numbers must be updated.")
        print("   3. Document in the data appendix: data from Yahoo Finance, snapshot date "
              f"{args.asof or '<ASOF>'}, raw files submitted with the paper.")
        print("   4. If the differences are concentrated on a few days, manually verify the")
        print("      actual market data for those days before deciding how to handle them.")
    else:
        print("  The two datasets are fully identical; proceed to the next step.")

    # Export
    rows = []
    for c, b in bad_dates.items():
        for dt in b.index:
            rows.append({"ticker": c, "date": dt.date(),
                         "old": old.loc[dt, c], "new": new_simple.loc[dt, c],
                         "diff_pp": (old.loc[dt, c] - new_simple.loc[dt, c]) * 100})
    if rows:
        pd.DataFrame(rows).sort_values(["ticker", "date"]).to_csv("audit_diff.csv", index=False)
        print(f"\n  Full difference list written to audit_diff.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()