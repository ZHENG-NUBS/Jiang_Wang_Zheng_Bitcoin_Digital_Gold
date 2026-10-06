"""
07_diagnose_spy_vix.py — Diagnose the three discrepancies found by script 06

Results from 06:
    IBIT / GLD   fully identical
    TLT          604/607 days "inconsistent", but max deviation 6.4e-07  → likely
                 floating-point / rounding noise
    SPY          607/607 days inconsistent, max deviation 9.9e-03 (0.99pp)
    VIX          max deviation 2.61 points

This script answers separately:

  1. Is the TLT discrepancy pure rounding? (If so, the 1e-9 threshold in 06 was too
     strict and should be ignored.)

  2. What makes up the SPY discrepancy? Split into two parts:
     (a) the equal-and-opposite difference on 2025-04-09/10 → a single closing price
         differs, so the implied price on each side can be back-solved
     (b) the systematic small difference on the remaining days → test whether it equals
         the dividend drift (i.e. old = price return, new = total return)
     And cross-check with an external fact: on 2025-04-09 the S&P 500 index rose 9.52%
     (recorded by multiple media outlets).
     If old matches the index while new is about 1pp higher, then old's SPY column may
     actually be the index (^GSPC) rather than the ETF.

  3. Where does the VIX difference come from? Test three possibilities: date misalignment,
     taking Open/High/Low, or coming from a different fetch.

Bonus: when scipy is not available locally, use moment estimation and projected gradient
       descent to fill in the t degrees of freedom and optimal weights, so that the two
       items skipped in 06 can also get numerical values.

Usage
    python 07_diagnose_spy_vix.py --old "/path/to/ETF_Returns.csv"
    (--new / --raw / --asof same as 06, can be auto-detected)
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

T = ["IBIT", "GLD", "TLT", "SPY"]


def hr(c="=", n=84):
    print(c * n)


def head(t):
    print()
    hr()
    print(t)
    hr()


def pick(path, pattern, label):
    if path and os.path.exists(path):
        return path
    c = sorted(glob.glob(pattern))
    if c:
        print(f"  Auto-selected {label}: {c[-1]}")
        return c[-1]
    return None


NEED_COLS = {"Date", "IBIT", "GLD", "TLT", "SPY"}


def _scan(roots, limit=400):
    seen, out, checked = set(), [], 0
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [x for x in dirnames if not x.startswith(".") and x not in
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
                if os.path.basename(os.path.dirname(p)) == "processed":
                    continue
                try:
                    cols = set(pd.read_csv(p, nrows=0).columns)
                except Exception:
                    continue
                if NEED_COLS.issubset(cols):
                    out.append(p)
    return out


def find_old(path):
    """Locate the original file by column names: the filename may contain spaces or be
    elsewhere, so we do not rely on the filename."""
    if os.path.exists(path):
        return path
    print(f"  [!] {path} does not exist, searching for candidates by column names...")
    for roots in ([os.getcwd()],
                  [os.path.dirname(os.path.abspath(__file__))]):
        c = _scan(roots)
        if c:
            break
    if len(c) == 1:
        print(f"  [OK] Auto-selected: {c[0]}")
        return c[0]
    if len(c) > 1:
        print("  Found multiple candidates, please specify one with --old:")
        for p in c[:10]:
            try:
                n = len(pd.read_csv(p))
            except Exception:
                n = "?"
            print(f"        {p}   ({n} rows)")
        sys.exit(1)
    print("  Not found. Please specify explicitly with --old, for example:")
    print('        python 07_diagnose_spy_vix.py --old "/path/to/ETF_Returns.csv"')
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="ETF_Returns.csv")
    ap.add_argument("--new", default=None)
    ap.add_argument("--raw", default=os.path.join("data", "raw"))
    ap.add_argument("--asof", default=None)
    a = ap.parse_args()

    a.new = pick(a.new, os.path.join("data", "processed", "returns__*.csv"), "new file")
    if a.asof is None and a.new and "__" in os.path.basename(a.new):
        a.asof = os.path.basename(a.new).split("__")[1].split(".")[0]
    a.old = find_old(a.old)

    old = pd.read_csv(a.old, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    print(f"  Original file: {a.old}")
    new = pd.read_csv(a.new, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    idx = old.index.intersection(new.index)
    old, new = old.loc[idx], new.loc[idx]
    print(f"Aligned {len(idx)} trading days: {idx[0].date()} → {idx[-1].date()}")

    # =================================================================== 1 =
    head("Section 1  Is the TLT discrepancy pure rounding?")
    d = (old.TLT - new.TLT).abs()
    print(f"  |diff| quantiles: 50% {d.quantile(.5):.2e}   90% {d.quantile(.9):.2e}   "
          f"99% {d.quantile(.99):.2e}   max {d.max():.2e}")
    print(f"  Days above 1e-6: {(d > 1e-6).sum()}    Days above 1e-4: {(d > 1e-4).sum()}")
    if d.max() < 1e-5:
        print("  [Conclusion] All below 1e-5 = rounding noise caused by prices keeping only")
        print("               two decimals. The 1e-9 threshold in script 06 was too strict;")
        print("               TLT should be judged [consistent]. The only substantive")
        print("               difference is SPY.")
    else:
        print("  [Conclusion] Differences beyond the rounding range exist; further")
        print("               investigation is needed.")

    # =================================================================== 2 =
    head("Section 2  Composition of the SPY discrepancy")
    dif = (old.SPY - new.SPY) * 100          # pp
    print(f"  Diff (pp) quantiles: 1% {dif.quantile(.01):+.3f}   25% {dif.quantile(.25):+.3f}   "
          f"50% {dif.quantile(.5):+.3f}   75% {dif.quantile(.75):+.3f}   99% {dif.quantile(.99):+.3f}")
    print(f"  Mean {dif.mean():+.4f}pp    Std {dif.std():.4f}pp    Days with |diff|>0.05pp "
          f"{(dif.abs() > 0.05).sum()}/{len(dif)}")

    print("\n  (a) Equal-and-opposite difference on 2025-04-09 / 04-10 → a single closing price differs")
    for dt in ["2025-04-08", "2025-04-09", "2025-04-10", "2025-04-11"]:
        ts = pd.Timestamp(dt)
        if ts in old.index:
            print(f"      {dt}  old {old.SPY[ts]*100:+8.3f}%   new {new.SPY[ts]*100:+8.3f}%"
                  f"   diff {(old.SPY[ts]-new.SPY[ts])*100:+7.3f}pp")
    try:
        i9, i10 = pd.Timestamp("2025-04-09"), pd.Timestamp("2025-04-10")
        # Using the 4/8 close as 1, back-solve the implied ratio of the 4/9 close
        ro, rn = old.SPY[i9], new.SPY[i9]
        print(f"      Implied ratio of 4/9 closing prices (new/old) = {(1+rn)/(1+ro):.6f}"
              f"  → new's 4/9 close is {((1+rn)/(1+ro)-1)*100:.3f}% higher than old's")
        ro2, rn2 = old.SPY[i10], new.SPY[i10]
        print(f"      4/10 reverse difference {((1+rn2)/(1+ro2)-1)*100:+.3f}% → the two re-converge at the 4/10 close")
        print("      [Conclusion] The discrepancy is concentrated on this single 4/9 close,")
        print("                   a single-point data disagreement, not a methodology issue.")
    except Exception as e:
        print(f"      (skipped: {e})")

    print("\n  (b) Excluding 4/9 and 4/10, the systematic difference on the remaining days")
    mask = ~old.index.isin([pd.Timestamp("2025-04-09"), pd.Timestamp("2025-04-10")])
    o2, n2 = old.SPY[mask], new.SPY[mask]
    yrs = mask.sum() / 252
    ao = ((1 + o2).prod() ** (1 / yrs) - 1) * 100
    an = ((1 + n2).prod() ** (1 / yrs) - 1) * 100
    print(f"      old annualized {ao:.2f}%   new annualized {an:.2f}%   diff {an-ao:+.2f}pp/yr")
    print(f"      SPY dividend yield is about 1.2%/yr; if old = price return and new = total")
    print(f"      return, the expected difference is about +1.2pp")
    print(f"      Measured {an-ao:+.2f}pp → {'consistent with the dividend explanation' if 0.8 < an-ao < 1.7 else 'inconsistent with the dividend explanation, there is another cause'}")
    d2 = (o2 - n2) * 100
    print(f"      On the remaining days, median |diff| {d2.abs().median():.4f}pp, mean {d2.mean():+.4f}pp")
    print(f"      If it were index vs ETF tracking difference, the daily median difference")
    print(f"      is usually < 0.03pp")
    print(f"      → Measured median {d2.abs().median():.4f}pp, "
          f"{'within the tracking-difference range' if d2.abs().median() < 0.03 else 'larger than a typical tracking difference'}")

    print("\n  (c) External cross-check: on 2025-04-09 the S&P 500 index rose 9.52% (recorded by CNBC/AP and others)")
    i9 = pd.Timestamp("2025-04-09")
    print(f"      old {old.SPY[i9]*100:.3f}%  ← {'matches the index' if abs(old.SPY[i9]*100-9.52)<0.1 else 'does not match the index'}")
    print(f"      new {new.SPY[i9]*100:.3f}%  ← {'matches the index' if abs(new.SPY[i9]*100-9.52)<0.1 else 'does not match the index'}")
    print("      Note: the SPY ETF and the index can differ slightly on the day (premium/discount,")
    print("            closing mechanism), but a 1pp gap is on the large side; it is advisable to")
    print("            verify the actual SPY closing prices on 4/8 and 4/9.")

    # =================================================================== 3 =
    head("Section 3  Where does the VIX difference come from?")
    if "VIX_Close" not in old.columns or "VIX_Close" not in new.columns:
        print("  One side is missing the VIX_Close column, skipping")
    else:
        vo, vn = old.VIX_Close, new.VIX_Close
        dv = (vo - vn)
        print(f"  |diff| quantiles: 50% {dv.abs().quantile(.5):.4f}   90% {dv.abs().quantile(.9):.4f}   "
              f"max {dv.abs().max():.4f}")
        print(f"  Fully identical days: {(dv.abs() < 1e-9).sum()}/{len(dv)}")
        big = dv.abs().sort_values(ascending=False).head(8)
        print(f"\n  The 8 days with the largest differences:")
        print(f"      {'Date':<12}{'old':>9}{'new':>9}{'diff':>9}")
        for dt in big.index:
            print(f"      {str(dt.date()):<12}{vo[dt]:>9.2f}{vn[dt]:>9.2f}{dv[dt]:>+9.2f}")

        print("\n  Test 1: Date misalignment? (the closer the correlation is to 1, the more likely that shift)")
        for lag in (-2, -1, 0, 1, 2):
            c = vo.corr(vn.shift(lag))
            print(f"      new shifted {lag:+d} day(s) → corr = {c:.6f}")

        print("\n  Test 2: Did old take Open / High / Low?")
        f = os.path.join(a.raw, f"IDX_VIX__{a.asof}.csv")
        if os.path.exists(f):
            raw = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
            for col in ("Open", "High", "Low", "Close"):
                if col in raw.columns:
                    s = raw[col].reindex(vo.index)
                    m = (vo - s).abs()
                    print(f"      vs {col:6s} max deviation {m.max():.4f}   "
                          f"fully identical {(m < 1e-9).sum()}/{m.notna().sum()}")
        else:
            print(f"      Cannot find {f}, skipping (use --raw/--asof to specify)")

    # =================================================================== 4 =
    head("Section 4  Fill in the two items skipped in 06 (no scipy needed)")

    def t_df_from_kurt(r):
        """Moment estimation: excess kurtosis k = 6/(v-4) → v = 6/k + 4 (valid when v>4)"""
        k = pd.Series(r).kurt()
        return 6.0 / k + 4.0 if k > 0 else np.nan

    print("  §3.2 t degrees of freedom of the portfolios (moment estimation, not MLE, for reference only)")
    for lab, df in (("old", old), ("new", new)):
        ib = (df.IBIT + df.SPY + df.TLT) / 3
        gl = (df.GLD + df.SPY + df.TLT) / 3
        print(f"      {lab}:  IBIT portfolio {t_df_from_kurt(ib):.2f}    GLD portfolio {t_df_from_kurt(gl):.2f}")
    print("      (The paper uses MLE and gets 7.9 / 4.2; moment estimation runs low; here we")
    print("       only look at the direction of change from old to new)")

    print("\n  Table 3 optimal weights (projected gradient descent, equivalent to a quadratic")
    print("  program with non-negativity and sum-to-one constraints)")

    def solve(mu, S, obj_grad, iters=200000, lr=1e-3):
        w = np.repeat(1 / len(mu), len(mu))
        for _ in range(iters):
            g = obj_grad(w)
            w = w - lr * g
            # Project onto the simplex
            u = np.sort(w)[::-1]
            css = np.cumsum(u) - 1
            rho = np.nonzero(u - css / (np.arange(len(u)) + 1) > 0)[0][-1]
            w = np.maximum(w - css[rho] / (rho + 1), 0)
        return w

    A = ["GLD", "SPY", "IBIT", "TLT"]
    print(f"      {'Target':<16}{'':<5}" + "".join(f"{x:>9}" for x in A))
    for lab, df in (("old", old), ("new", new)):
        R = df[A].values
        mu = R.mean(0) * 252
        S = np.cov(R.T) * 252
        mv = solve(mu, S, lambda w: 2 * S @ w)

        def neg_sharpe_grad(w):
            v = np.sqrt(w @ S @ w)
            return -(mu / v - (mu @ w) * (S @ w) / v ** 3)
        ms = solve(mu, S, neg_sharpe_grad, lr=1e-4)
        print(f"      {'Max Sharpe (rf=0)' if lab=='old' else '':<16}{lab:<5}"
              + "".join(f"{x*100:>9.2f}" for x in ms))
        if lab == "new":
            print()
    for lab, df in (("old", old), ("new", new)):
        R = df[A].values
        S = np.cov(R.T) * 252
        mv = solve(R.mean(0) * 252, S, lambda w: 2 * S @ w)
        print(f"      {'Min variance' if lab=='old' else '':<16}{lab:<5}"
              + "".join(f"{x*100:>9.2f}" for x in mv))

    # =================================================================== 5 =
    head("Section 5  Questions for you to judge")
    print("  1. Which SPY closing price on 2025-04-09 is correct?")
    print("     The index rose 9.52% that day; old is +9.515%, new is +10.502%.")
    print("     Please verify the actual SPY closing prices on 4/8 and 4/9 (iShares/SSGA")
    print("     official site or a broker terminal).")
    print("     If old is correct → the new snapshot has a bad value that day; fix it and rerun;")
    print("     If new is correct → old's SPY column may have been taken from the index (^GSPC)")
    print("                       rather than the ETF, whereas §2.1 of the paper explicitly says")
    print("                       SPDR S&P 500 ETF Trust, so it must be corrected.")
    print()
    print("  2. The two VIX versions differ; which one is authoritative? See the three test")
    print("     results in Section 3.")
    print()
    print("  3. Whichever is authoritative, reproducibility requires that the final data be")
    print("     generated by scripts from a frozen snapshot, not merged by hand. It is")
    print("     recommended to take the new pipeline as authoritative, correct the confirmed")
    print("     bad single points, and rerun all results.")


if __name__ == "__main__":
    main()