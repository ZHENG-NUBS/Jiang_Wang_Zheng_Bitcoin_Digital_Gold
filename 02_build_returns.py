from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AS_OF = "20260921"
RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
RF_PATH = None  # e.g. Path("data/raw/DGS3MO__20260921.csv"), two columns date,value(%)

PRICE_COL = "AdjClose"  # <- convention switch. Change to "Close" for pure price returns (not recommended)
BASE = "SPY"  # trading-day base
ASSETS = ["IBIT", "GLD", "TLT", "SPY"]
EXTRAS = {"^VIX": "VIX_Close"}


def load_raw(ticker: str) -> pd.DataFrame:
    safe = ticker.replace("^", "IDX_")
    p = RAW_DIR / f"{safe}__{AS_OF}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}; please run 01a_fetch_raw.py first")
    return pd.read_csv(p, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", type=Path, default=None,
                    help="Take an existing returns CSV and infer which convention it used")
    args = ap.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        log.append(s)

    # ---------------------------------------------------------------- Load
    raw = {t: load_raw(t) for t in ASSETS}
    say(f"Convention: PRICE_COL = {PRICE_COL}")
    say("")
    say("Raw snapshots for each ticker:")
    for t, df in raw.items():
        gap = (df.Close - df.AdjClose).abs().max()
        kind = "no distributions (Close == AdjClose)" if gap < 1e-8 else f"has distributions (max diff {gap:.4f})"
        say(f"  {t:5s} {len(df):4d} rows  {df.Date.min().date()} -> {df.Date.max().date()}  {kind}")

    # ------------------------------------------------------- Trading-day alignment
    cal = set(raw[BASE].Date)
    for t, df in raw.items():
        cal &= set(df.Date)
    cal = pd.DatetimeIndex(sorted(cal))
    say("")
    say(f"Trading-day alignment: inner join on {BASE} as base, {len(cal)} trading days total")
    for t, df in raw.items():
        dropped = sorted(set(df.Date) - set(cal))
        if dropped:
            say(f"  {t}: dropped {len(dropped)} days -- {[str(d.date()) for d in dropped[:8]]}"
                f"{' ...' if len(dropped) > 8 else ''}")

    px = pd.DataFrame({t: raw[t].set_index("Date")[PRICE_COL].reindex(cal) for t in ASSETS})
    assert not px.isna().any().any(), "Still missing values after alignment; check raw data"

    # -------------------------------------------------------------- Build returns
    simple = px.pct_change().dropna()
    logret = np.log(px / px.shift(1)).dropna()

    out = pd.DataFrame(index=simple.index)
    for t in ASSETS:
        out[t] = simple[t]  # main series = simple returns
        out[f"{t}_log"] = logret[t]

    for tk, col in EXTRAS.items():
        ex = load_raw(tk).set_index("Date")["Close"].reindex(out.index)
        if ex.isna().any():
            say(f"  Note: {tk} is missing on {int(ex.isna().sum())} trading days, kept as NaN (not filled)")
        out[col] = ex

    if RF_PATH and Path(RF_PATH).exists():
        rf = pd.read_csv(RF_PATH, parse_dates=[0]).set_index(pd.read_csv(RF_PATH).columns[0])
        ann = pd.to_numeric(rf.iloc[:, 0], errors="coerce") / 100.0
        out["rf_daily"] = ((1 + ann.reindex(out.index).ffill()) ** (1 / 252) - 1)
        say("\nRisk-free rate: merged into rf_daily (3M T-bill, converted using 252)")
    else:
        say("\nRisk-free rate: not provided. Please supply RF_PATH (FRED DGS3MO) before computing Sharpe ratios.")

    out.index.name = "Date"
    out_path = PROC_DIR / f"returns__{AS_OF}.csv"
    out.to_csv(out_path)

    # ---------------------------------------------------------------- Summary
    say("")
    say("=" * 66)
    say("Summary (simple returns, annualized)")
    say("=" * 66)
    yrs = len(out) / 252
    say(f"{'':6s}{'AnnRet':>10s}{'AnnVol':>10s}{'Skew':>9s}{'ExKurt':>10s}")
    for t in ASSETS:
        r = out[t]
        ann = (1 + r).prod() ** (1 / yrs) - 1
        say(f"{t:6s}{ann * 100:9.2f}%{r.std() * np.sqrt(252) * 100:9.2f}%"
            f"{r.skew():9.2f}{r.kurt():10.2f}")

    # Quantify convention impact: compute both price columns
    say("")
    say("Convention impact (annualized return difference between AdjClose and Close = distribution contribution):")
    for t in ASSETS:
        pa = raw[t].set_index("Date")["AdjClose"].reindex(cal)
        pc = raw[t].set_index("Date")["Close"].reindex(cal)
        aa = (pa.iloc[-1] / pa.iloc[0]) ** (1 / yrs) - 1
        ac = (pc.iloc[-1] / pc.iloc[0]) ** (1 / yrs) - 1
        vol = out[t].std() * np.sqrt(252)
        say(f"  {t:5s} AdjClose {aa * 100:+7.2f}%/yr   Close {ac * 100:+7.2f}%/yr"
            f"   diff {(aa - ac) * 100:+6.2f}pp   ~ Sharpe diff {(aa - ac) / vol:+.3f}")

    # --------------------------------------------- Infer old file's convention
    if args.compare and args.compare.exists():
        say("")
        say("=" * 66)
        say(f"Convention inference: {args.compare.name}")
        say("=" * 66)
        old = pd.read_csv(args.compare, parse_dates=["Date"]).set_index("Date")
        best = None
        for pcol in ("AdjClose", "Close"):
            p2 = pd.DataFrame({t: raw[t].set_index("Date")[pcol].reindex(cal) for t in ASSETS})
            for rkind, series in (("simple", p2.pct_change()), ("log", np.log(p2 / p2.shift(1)))):
                errs = {}
                for t in ASSETS:
                    if t not in old.columns:
                        continue
                    a, b = series[t].dropna().align(old[t].dropna(), join="inner")
                    if len(a) == 0:
                        continue
                    errs[t] = float((a - b).abs().max())
                if not errs:
                    continue
                worst = max(errs.values())
                say(f"  {pcol:8s} + {rkind:6s}  max abs deviation = {worst:.3e}   "
                    + "  ".join(f"{k}:{v:.1e}" for k, v in errs.items()))
                if best is None or worst < best[0]:
                    best = (worst, pcol, rkind)
        if best:
            say("")
            verdict = "exact match" if best[0] < 1e-9 else ("close match" if best[0] < 1e-6 else "no match")
            say(f"  Verdict: {verdict} -> old file convention = {best[1]} + {best[2]}"
                if best[0] < 1e-6 else
                f"  Verdict: {verdict} (closest is {best[1]} + {best[2]}, deviation {best[0]:.2e})")
            if best[0] >= 1e-6:
                say("  Deviation too large; possible reasons: different sample period, "
                    "the data is an earlier snapshot, or extra processing was applied "
                    "in between (e.g. outlier removal).")

    rep = PROC_DIR / f"build_report__{AS_OF}.txt"
    rep.write_text("\n".join(log), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Wrote {rep}")


if __name__ == "__main__":
    main()
