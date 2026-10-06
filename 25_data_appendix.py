"""
25_data_appendix.py — Data construction documentation / reproducible data appendix

    (1) Data vendor                       → read the manifest; list the source, download
                                            time, and SHA-256
    (2) Adjusted close vs total-return    → build a total-return series from Close plus
        series                              the dividend detail, and compare it day by day
                                            against Yahoo AdjClose
    (3) Non-synchronous trading days      → check against the exchange calendar; the
                                            misalignment and lead–lag correlation between
                                            BTC-USD (UTC 00:00) and the ETF (16:00 ET)
    (4) Holidays                          → check day by day against the NYSE closure
                                            calendar; early-close days; days when the bond
                                            market was closed but the stock market was open
                                            (TLT)
    (5) Missing values                    → per ticker: NaN / duplicate dates / zero
                                            volume / zero returns / OHLC inconsistency /
                                            extreme values
    (6) Dividends, splits                 → reconcile the event table against dividends
                                            back-solved from AdjClose, line by line; test
                                            split days
    (7) ETF management fees               → fee table + note that prices are already
                                            net of fees + IBIT's empirical tracking
                                            difference vs BTC spot
    (8) Risk-free rate for Sharpe ratios  → download FRED DGS3MO / DTB3 (fall back to Yahoo
                                            ^IRX on failure); compare Sharpe under different
                                            conventions
    (9) Simple vs log returns             → descriptive statistics under both conventions,
                                            and state the use of each

Outputs (all in out/25_data_appendix/)
    report__<ASOF>.txt                 full run log (can be cited directly in the reviewer
                                       reply)
    data_appendix_draft.md             English draft appendix, numbers filled in
                                       automatically
    tab_*.csv                          one table per section
Additionally written
    data/raw/DGS3MO__<ASOF>.csv etc.   raw risk-free-rate snapshots (downloaded on first
                                       run, read locally thereafter)
    data/raw/manifest_rf__<ASOF>.json  SHA-256 of the risk-free-rate snapshots
    data/processed/returns_rf__<ASOF>.csv  returns + daily risk-free rate + excess returns

Usage (in PyCharm just click the green triangle to run; or on the command line)
    python 25_data_appendix.py
    python 25_data_appendix.py --offline            # no network, use only local snapshots
    python 25_data_appendix.py --rf-file somefile.csv  # manually downloaded FRED file (two
                                                       # columns: date, rate %)

Dependencies: pandas numpy requests (all present in your .venv)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration — consistent with 01a_fetch_raw.py / build returns.py
# ============================================================================
AS_OF = "20260921"                 # snapshot used for the main analysis
SNAPSHOT_SUFFIXES = ["", "L", "X"]  # three downloads under the same AS_OF (for consistency checks)
ASSETS = ["IBIT", "GLD", "TLT", "SPY"]
BASE = "SPY"                       # trading-day basis
BTC = "BTC-USD"                    # BTC spot (Yahoo, UTC daily), read from the L snapshot
PRICE_COL = "AdjClose"
TD = 252                           # annualization days

PROXY = None   # no address is stored here; set HTTPS_PROXY in the environment if needed
TIMEOUT = 30

RAW = Path("data/raw")
PROC = Path("data/processed")
OUT = Path("out/25_data_appendix")

# Constant risk-free rate used temporarily in 17 portfolio.py, for sensitivity comparison
RF_CONST_ANNUAL = 0.043

# ---------------------------------------------------------------------------
# NYSE full-day closures (2024-01-01 to 2026-06-30)
# Source: NYSE Holidays & Trading Hours (https://www.nyse.com/markets/hours-calendars)
# 2025-01-09 was an unscheduled closure for President Carter's state funeral
# ---------------------------------------------------------------------------
NYSE_HOLIDAYS = {
    "2024-01-01": "New Year's Day", "2024-01-15": "MLK Day", "2024-02-19": "Washington's Birthday",
    "2024-03-29": "Good Friday", "2024-05-27": "Memorial Day", "2024-06-19": "Juneteenth",
    "2024-07-04": "Independence Day", "2024-09-02": "Labor Day", "2024-11-28": "Thanksgiving",
    "2024-12-25": "Christmas",
    "2025-01-01": "New Year's Day", "2025-01-09": "National Day of Mourning (Carter)",
    "2025-01-20": "MLK Day", "2025-02-17": "Washington's Birthday", "2025-04-18": "Good Friday",
    "2025-05-26": "Memorial Day", "2025-06-19": "Juneteenth", "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day", "2025-11-27": "Thanksgiving", "2025-12-25": "Christmas",
    "2026-01-01": "New Year's Day", "2026-01-19": "MLK Day", "2026-02-16": "Washington's Birthday",
    "2026-04-03": "Good Friday", "2026-05-25": "Memorial Day", "2026-06-19": "Juneteenth",
}
# NYSE 13:00 early-close days
NYSE_EARLY_CLOSE = ["2024-07-03", "2024-11-29", "2024-12-24",
                    "2025-07-03", "2025-11-28", "2025-12-24"]
# U.S. bond market closed (SIFMA recommendation) but stocks open: TLT still trades, but
# its constituent bonds have no new quotes
BOND_ONLY_HOLIDAYS = ["2024-10-14", "2024-11-11", "2025-10-13", "2025-11-11"]

# ---------------------------------------------------------------------------
# ETF expense ratios (annualized, as a share of NAV) — in the paper please re-verify
# against the latest issuer filings
# ---------------------------------------------------------------------------
EXPENSE = {
    "IBIT": dict(er=0.0025, note="Sponsor fee 0.25%; waived to 0.12% on first USD 5bn of assets "
                                 "for 12 months from 2024-01-11",
                 src="iShares Bitcoin Trust prospectus; BlackRock press release 2024-01-11"),
    "GLD":  dict(er=0.0040, note="Sponsor fee 0.40%", src="SPDR Gold Trust prospectus (spdrgoldshares.com)"),
    "TLT":  dict(er=0.0015, note="Management fee 0.15%", src="iShares TLT fact sheet (ishares.com)"),
    "SPY":  dict(er=0.000945, note="Gross expense ratio 0.0945%", src="SPY fact sheet (ssga.com)"),
}

# ============================================================================
# Utilities
# ============================================================================
LOG: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LOG.append(s)


def rule(title: str = "", ch: str = "=") -> None:
    say("")
    say(ch * 78)
    if title:
        say(title)
        say(ch * 78)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(t: str) -> str:
    return t.replace("^", "IDX_")


def load_raw(ticker: str, suffix: str = "") -> pd.DataFrame | None:
    p = RAW / f"{safe(ticker)}__{AS_OF}{suffix}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def ann_geo(r: pd.Series) -> float:
    r = r.dropna()
    return float((1 + r).prod() ** (TD / len(r)) - 1)


def fmt_pct(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def session():
    import requests
    s = requests.Session()
    if PROXY:
        s.proxies.update({"http": PROXY, "https": PROXY})
    s.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
    return s


# ============================================================================
# For Section 8: risk-free-rate download and reading
# ============================================================================
def fetch_fred(series: str, start: str, end: str) -> pd.DataFrame:
    """FRED public CSV (no API key needed). Returns Date, value(%)."""
    r = session().get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                      params={"id": series, "cosd": start, "coed": end}, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["Date", "value"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # FRED marks missing as "." or blank
    return df


def fetch_irx(start: str, end: str) -> pd.DataFrame:
    """Yahoo ^IRX (13-week T-bill discount rate, %). Same endpoint as 01a_fetch_raw.py, used as
    a fallback when FRED is unreachable."""
    params = {"period1": int(pd.Timestamp(start, tz="Etc/GMT-8").timestamp()),
              "period2": int(pd.Timestamp(end, tz="Etc/GMT-8").timestamp()),
              "interval": "1d", "events": "history"}
    r = session().get("https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX",
                      params=params, timeout=TIMEOUT)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    tz = res["meta"]["exchangeTimezoneName"]
    d = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).normalize().tz_localize(None)
    return pd.DataFrame({"Date": d, "value": res["indicators"]["quote"][0]["close"]})


def local_irx(start: str, end: str) -> pd.DataFrame | None:
    """Local Yahoo ^IRX snapshot (the L file from 01e or the S file from 01d; columns Date, Open, ..., Close)."""
    for suf in ("L", "S"):
        p = RAW / f"IDX_IRX__{AS_OF}{suf}.csv"
        if p.exists():
            d = pd.read_csv(p, parse_dates=["Date"])
            d = d[(d.Date >= pd.Timestamp(start)) & (d.Date < pd.Timestamp(end))]
            say(f"  IRX: using the local Yahoo snapshot {p.name}")
            return pd.DataFrame({"Date": d.Date, "value": d.Close})
    return None


def get_rf_snapshots(offline: bool, rf_file: Path | None, start: str, end: str) -> dict[str, pd.DataFrame]:
    """Returns {series name: DataFrame(Date, value %)}. If a local snapshot exists, no
    network call is made (to preserve reproducibility)."""
    out: dict[str, pd.DataFrame] = {}
    manifest_p = RAW / f"manifest_rf__{AS_OF}.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {"files": {}}

    if rf_file is not None:
        df = pd.read_csv(rf_file)
        df = df.iloc[:, :2]
        df.columns = ["Date", "value"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        name = rf_file.stem.split("__")[0].upper()
        if name not in ("DGS3MO", "DTB3", "IRX"):
            name = "DGS3MO"
        out[name] = df
        say(f"  Using manually provided risk-free-rate file {rf_file} (treated as the {name} convention)")

    for name, fetcher in (("DGS3MO", lambda: fetch_fred("DGS3MO", start, end)),
                          ("DTB3", lambda: fetch_fred("DTB3", start, end)),
                          ("IRX", lambda: fetch_irx(start, end))):
        if name in out:
            continue
        p = RAW / f"{name}__{AS_OF}.csv"
        if p.exists():
            out[name] = pd.read_csv(p, parse_dates=["Date"])
            say(f"  {name}: reading local snapshot {p.name} ({len(out[name])} rows)")
            continue
        df = None
        if name == "IRX":
            df = local_irx(start, end)
        if df is None and offline:
            say(f"  {name}: no local snapshot, skipped in --offline mode")
            continue
        try:
            if df is None:
                df = fetcher()
            df.to_csv(p, index=False)
            manifest["files"][p.name] = sha256(p)
            manifest.setdefault("downloaded_at_utc", {})[p.name] = \
                datetime.now(timezone.utc).isoformat(timespec="seconds")
            out[name] = df
            say(f"  {name}: downloaded and saved as snapshot {p.name} ({len(df)} rows)")
        except Exception as e:  # noqa: BLE001
            say(f"  {name}: download failed ({type(e).__name__}: {str(e)[:80]})")

    if manifest["files"]:
        manifest["source"] = {"DGS3MO": "FRED, 3-Month Treasury Constant Maturity Rate (bond-equivalent, %)",
                              "DTB3": "FRED, 3-Month Treasury Bill Secondary Market Rate (discount basis, %)",
                              "IRX": "Yahoo Finance ^IRX, 13-week T-bill (discount basis, %)"}
        manifest_p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def rf_daily_on_calendar(df: pd.DataFrame, kind: str, cal: pd.DatetimeIndex) -> pd.Series:
    """Annualized % → daily risk-free return on trading day t.
    Convention: the risk-free rate for day t's return is the yield known as of the previous
    trading day t-1 (forward-filled over bond-market holidays); discount-basis rates
    (DTB3/IRX) are first converted to a bond-equivalent yield BEY = 365d / (360 - 91d),
    then converted to daily frequency as (1 + y)^(1/252) - 1 (consistent with build returns.py)."""
    s = df.dropna().set_index("Date")["value"].astype(float) / 100.0
    s = s[~s.index.duplicated()].sort_index()
    if kind in ("DTB3", "IRX"):
        s = 365 * s / (360 - 91 * s)
    full = s.reindex(s.index.union(cal)).sort_index().ffill()
    y_lag = full.reindex(cal).shift(1)
    return (1 + y_lag) ** (1 / TD) - 1


# ============================================================================
# Main flow
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="no network, use only local snapshots")
    ap.add_argument("--rf-file", type=Path, default=None, help="manually downloaded risk-free-rate CSV (date, rate %%)")
    a = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent)     # works in PyCharm regardless of the working directory
    OUT.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    say(f"25_data_appendix.py   run time {datetime.now().isoformat(timespec='seconds')}   snapshot AS_OF={AS_OF}")

    # ------------------------------------------------------------------ Load
    raw = {t: load_raw(t) for t in ASSETS}
    for t, df in raw.items():
        if df is None:
            raise FileNotFoundError(f"Missing data/raw/{safe(t)}__{AS_OF}.csv; please first run 01a_fetch_raw.py")
    vix = load_raw("^VIX")
    btc = load_raw(BTC, "L")
    if btc is None:
        btc = load_raw(BTC, "X")
    events = pd.read_csv(RAW / f"events__{AS_OF}.csv", parse_dates=["date"]) \
        if (RAW / f"events__{AS_OF}.csv").exists() else pd.DataFrame(columns=["ticker", "date", "type", "amount", "ratio"])
    events_x = pd.read_csv(RAW / f"events__{AS_OF}X.csv", parse_dates=["date"]) \
        if (RAW / f"events__{AS_OF}X.csv").exists() else pd.DataFrame(columns=events.columns)

    # Common trading calendar (same rule as build returns.py: inner join, no forward filling)
    cal = set(raw[BASE].Date)
    for df in raw.values():
        cal &= set(df.Date)
    cal = pd.DatetimeIndex(sorted(cal))
    px = pd.DataFrame({t: raw[t].set_index("Date")[PRICE_COL].reindex(cal) for t in ASSETS})
    R = px.pct_change().iloc[1:]
    L = np.log(px / px.shift(1)).iloc[1:]
    T = len(R)
    s0, s1 = cal[0].date(), cal[-1].date()
    facts: dict[str, object] = dict(sample_start=str(s0), first_ret=str(R.index[0].date()),
                                     sample_end=str(s1), n_days=len(cal), n_ret=T)

    # ====================================================================== 1
    rule("1. Data vendor and snapshot")
    man_p = RAW / f"manifest__{AS_OF}.json"
    if man_p.exists():
        man = json.loads(man_p.read_text(encoding="utf-8"))
        say(f"  Source        : {man.get('source')}")
        say(f"  Download time UTC : {man.get('downloaded_at_utc')}")
        say(f"  Requested range    : {man.get('start_date')} → {man.get('end_date_exclusive')} (right-open)")
        facts["vendor"] = man.get("source")
        facts["downloaded"] = man.get("downloaded_at_utc")
        bad = []
        for fn, h in man.get("files", {}).items():
            p = RAW / fn
            if p.exists() and sha256(p) != h:
                bad.append(fn)
        say(f"  SHA-256 check: {len(man.get('files', {}))} files, "
            f"{'all match' if not bad else 'mismatch: ' + ', '.join(bad)}")
        facts["sha_ok"] = not bad
    else:
        say("  Manifest not found; please confirm 01a_fetch_raw.py has been run")
    say(f"  Main sample      : {s0} → {s1}, {len(cal)} common trading days, {T} daily returns"
        f"(start determined by the IBIT listing date 2024-01-11)")

    # Consistency of the multiple downloads under the same AS_OF
    rows = []
    for t in ASSETS + ["^VIX"]:
        base = load_raw(t, "")
        for suf in SNAPSHOT_SUFFIXES[1:]:
            other = load_raw(t, suf)
            if base is None or other is None:
                continue
            m = base.merge(other, on="Date", suffixes=("", "_o"))
            rows.append(dict(ticker=t, snapshot=f"{AS_OF}{suf}", common_days=len(m),
                             max_abs_diff_close=float((m.Close - m.Close_o).abs().max()),
                             max_abs_diff_adj=float((m.AdjClose - m.AdjClose_o).abs().max())))
    snap = pd.DataFrame(rows)
    if not snap.empty:
        snap.to_csv(OUT / "tab_1_snapshot_consistency.csv", index=False)
        say("  Maximum absolute difference across the three download snapshots on overlapping "
            "dates (should be 0 or tiny):")
        for _, r in snap.iterrows():
            say(f"    {r.ticker:5s} vs {r.snapshot:10s} overlap {r.common_days:5d} days  "
                f"Close {r.max_abs_diff_close:.2e}  AdjClose {r.max_abs_diff_adj:.2e}")
        facts["snap_max_diff"] = float(snap[["max_abs_diff_close", "max_abs_diff_adj"]].max().max())

    # ====================================================================== 3/4
    rule("3–4. Trading calendar, holidays, and non-synchronous trading")
    spy_d = pd.DatetimeIndex(raw[BASE].Date)
    wk = pd.bdate_range(spy_d.min(), spy_d.max())
    missing_wk = wk.difference(spy_d)
    hol = pd.DatetimeIndex(pd.to_datetime(list(NYSE_HOLIDAYS)))
    hol_in = hol[(hol >= spy_d.min()) & (hol <= spy_d.max())]
    unexpected_missing = missing_wk.difference(hol_in)
    hol_present = hol_in.intersection(spy_d)
    say(f"  Business days in the SPY range {len(wk)}, missing {len(missing_wk)}; "
        f"{len(hol_in)} days in the NYSE closure table")
    say(f"  Missing but not in the closure table : {[str(d.date()) for d in unexpected_missing] or 'none'}")
    say(f"  Present despite being in the closure table : {[str(d.date()) for d in hol_present] or 'none'}")
    facts["calendar_ok"] = len(unexpected_missing) == 0 and len(hol_present) == 0
    facts["n_holidays"] = len(hol_in)
    pd.DataFrame({"date": [d.date() for d in hol_in],
                  "holiday": [NYSE_HOLIDAYS[str(d.date())] for d in hol_in],
                  "in_data": [d in spy_d for d in hol_in]}).to_csv(OUT / "tab_3_nyse_holidays.csv", index=False)

    ec = [d for d in pd.to_datetime(NYSE_EARLY_CLOSE) if d in cal]
    bo = [d for d in pd.to_datetime(BOND_ONLY_HOLIDAYS) if d in cal]
    say(f"  {len(ec)} 13:00 early-close days in the sample: {[str(d.date()) for d in ec]} "
        "(retained, priced at that day's close)")
    if bo:
        say(f"  {len(bo)} days when the bond market was closed but stocks were open: "
            f"{[str(d.date()) for d in bo]}")
        say("    TLT return that day: " + "  ".join(f"{d.date()} {R.loc[d, 'TLT'] * 100:+.2f}%" for d in bo if d in R.index))
    facts["n_early"] = len(ec)
    facts["n_bond_only"] = len(bo)

    for t, df in raw.items():
        d = pd.DatetimeIndex(df.Date)
        extra = d.difference(spy_d)
        lack = spy_d[spy_d >= d.min()].difference(d)
        say(f"  {t:5s} first day {d.min().date()}  relative to SPY: {len(extra)} extra days, "
            f"{len(lack)} missing days")
    if vix is not None:
        vd = pd.DatetimeIndex(vix.Date)
        say(f"  ^VIX  relative to SPY: {len(vd.difference(spy_d))} extra days, "
            f"{len(spy_d.difference(vd))} missing days"
            "(VIX closes at 16:15 ET and is used only as a state variable, not in return computation)")

    # BTC-USD: trades 24/7; Yahoo's daily bar closes at UTC 00:00 (= 19:00/20:00 ET), later
    # than the ETF's 16:00 close
    if btc is not None:
        say("")
        say("  Alignment of BTC-USD (Yahoo, 24/7, UTC daily) with the U.S. equity calendar:")
        b = btc.set_index("Date")["Close"]
        b_on_cal = b.reindex(b.index.union(cal)).sort_index().ffill().reindex(cal)   # take the most recent BTC close on or before t
        rb = b_on_cal.pct_change().iloc[1:]
        n_wkend = int(((R.index.to_series().diff().dt.days) > 1).sum())
        say(f"    Method: take the most recent BTC close on or before day t, so weekend and "
            f"U.S.-holiday BTC returns accrue to the next trading day"
            f"({n_wkend} trading days in the sample span a non-trading day)")
        z = pd.concat([R["IBIT"], rb.rename("BTC")], axis=1).dropna()
        cc = {k: z["IBIT"].corr(z["BTC"].shift(k)) for k in (-1, 0, 1)}
        say(f"    corr(IBIT_t, BTC_t-1) = {cc[1]:.3f}   corr(IBIT_t, BTC_t) = {cc[0]:.3f}   "
            f"corr(IBIT_t, BTC_t+1) = {cc[-1]:.3f}")
        say("    Interpretation: BTC's day-t return window is 20:00 ET on t-1 → 20:00 ET on t, "
            "offset by about 4 hours from IBIT's 16:00 → 16:00.")
        say("          The same-day correlation is clearly below 1 while the weekly correlation "
            "is close to 1, indicating that the daily difference is mainly due to the closing-time "
            "mismatch;")
        say("          the lead/lag correlations " + ("are close to 0, so the mismatch does not "
            "create exploitable cross-day dependence." if max(abs(cc[1]), abs(cc[-1])) < 0.1
            else "are not negligible, so using daily BTC-USD results requires synchronizing "
                 "(e.g. 2-day or weekly returns)."))
        facts.update(corr_btc_lag=cc[1], corr_btc_0=cc[0], corr_btc_lead=cc[-1])
        # Use weekly (Friday-to-Friday) returns to reduce the non-synchrony effect
        wkly_i = (1 + z["IBIT"]).resample("W-FRI").prod() - 1
        wkly_b = (1 + z["BTC"]).resample("W-FRI").prod() - 1
        facts["corr_btc_week"] = float(wkly_i.corr(wkly_b))
        say(f"    Weekly correlation corr = {facts['corr_btc_week']:.3f} (at weekly frequency "
            "the 4-hour mismatch is diluted)")
        R_btc = rb
    else:
        R_btc = None

    # The four main assets all close at 16:00 ET and are synchronous; give daily vs weekly
    # correlation matrices as a robustness check
    wR = (1 + R).resample("W-FRI").prod() - 1
    cm_d, cm_w = R.corr(), wR.corr()
    cm = pd.concat({"daily": cm_d, "weekly": cm_w})
    cm.to_csv(OUT / "tab_3_corr_daily_vs_weekly.csv")
    say("")
    say("  Main-asset correlations: daily / weekly (W-FRI) — a large difference would indicate "
        "the daily results are affected by closing time")
    for i, a1 in enumerate(ASSETS):
        for a2 in ASSETS[i + 1:]:
            say(f"    {a1:4s}-{a2:4s}  daily {cm_d.loc[a1, a2]:+.3f}   weekly {cm_w.loc[a1, a2]:+.3f}")

    # ====================================================================== 5
    rule("5. Missing values and data quality")
    qrows = []
    for t in ASSETS:
        df = raw[t]
        d = df[(df.Date >= cal[0]) & (df.Date <= cal[-1])]
        r = d.set_index("Date")[PRICE_COL].pct_change()
        z = (r - r.mean()) / r.std()
        o = d[["Open", "High", "Low", "Close"]]
        bad_ohlc = int(((o.Low > o[["Open", "Close"]].min(axis=1) + 1e-6) |
                        (o.High < o[["Open", "Close"]].max(axis=1) - 1e-6)).sum())
        out_d = [f"{i.date()}({v * 100:+.1f}%)" for i, v in r[z.abs() > 5].items()]
        qrows.append(dict(ticker=t, rows=len(d),
                          nan_any=int(d[["Open", "High", "Low", "Close", "AdjClose"]].isna().any(axis=1).sum()),
                          dup_dates=int(d.Date.duplicated().sum()),
                          zero_volume=int((d.Volume.fillna(0) == 0).sum()),
                          zero_return=int((r == 0).sum()),
                          ohlc_inconsistent=bad_ohlc,
                          abs_z_gt5=len(out_d), outliers=";".join(out_d)))
    q = pd.DataFrame(qrows)
    q.to_csv(OUT / "tab_5_data_quality.csv", index=False)
    say(f"  {'':5s}{'Rows':>6s}{'NaN':>6s}{'Dup':>6s}{'ZeroVol':>8s}{'ZeroRet':>8s}{'OHLCbad':>8s}{'|z|>5':>7s}")
    for _, r in q.iterrows():
        say(f"  {r.ticker:5s}{r.rows:6d}{r.nan_any:6d}{r.dup_dates:6d}{r.zero_volume:8d}"
            f"{r.zero_return:8d}{r.ohlc_inconsistent:8d}{r.abs_z_gt5:7d}  {r.outliers}")
    say("  Handling rule: no interpolation, no forward filling, no trimming of extreme values; "
        "01a_fetch_raw.py only drops rows with a missing Close/AdjClose,")
    say("            and after alignment it asserts that there are no missing values (assert in "
        "build returns.py). All extreme values are retained and listed in the table above.")
    facts["n_nan"] = int(q.nan_any.sum())
    facts["n_dup"] = int(q.dup_dates.sum())
    facts["n_out"] = int(q.abs_z_gt5.sum())

    # ====================================================================== 2/6
    rule("2 & 6. Adjusted close vs total-return series; dividends and splits")
    say("  Yahoo's Close is split-adjusted but not dividend-adjusted; AdjClose is both "
        "split- and dividend-adjusted (multiplicative method).")
    say("  Yahoo's multiplicative method is equivalent to: ex-date return = Close_t / (Close_t-1 − D_t) − 1")
    say("  CRSP-style total return   : ex-date return = (Close_t + D_t) / Close_t-1 − 1 "
        "(dividend reinvested at the ex-date close)")
    ev = events[(events.date >= cal[0]) & (events.date <= cal[-1])]
    trows, drows = [], []
    for t in ASSETS:
        df = raw[t].set_index("Date").reindex(cal)
        D = ev[(ev.ticker == t) & (ev.type == "dividend")].groupby("date")["amount"].sum().reindex(cal).fillna(0.0)
        r_p = df.Close.pct_change().iloc[1:]
        r_a = df.AdjClose.pct_change().iloc[1:]
        r_tr = ((df.Close + D) / df.Close.shift(1) - 1).iloc[1:]
        # Back-solve each dividend from the AdjClose/Close ratio, and reconcile it line by
        # line against the event table
        f = df.AdjClose / df.Close
        imp = (df.Close.shift(1) * (1 - f.shift(1) / f)).iloc[1:]
        for dt, amt in D[D > 0].items():
            drows.append(dict(ticker=t, ex_date=dt.date(), amount_event=amt,
                              amount_implied_by_adjclose=float(imp.get(dt, np.nan)),
                              diff=float(imp.get(dt, np.nan) - amt)))
        nd = int((D > 0).sum())
        trows.append(dict(ticker=t, n_dividends=nd, dividend_sum=float(D.sum()),
                          ann_price_return=ann_geo(r_p), ann_adjclose_return=ann_geo(r_a),
                          ann_total_return_crsp=ann_geo(r_tr),
                          adj_minus_crsp_pp=(ann_geo(r_a) - ann_geo(r_tr)) * 100,
                          max_abs_daily_diff_adj_vs_crsp=float((r_a - r_tr).abs().max())))
    tr = pd.DataFrame(trows)
    tr.to_csv(OUT / "tab_2_price_vs_total_return.csv", index=False)
    dv = pd.DataFrame(drows)
    dv.to_csv(OUT / "tab_6_dividend_check.csv", index=False)
    say("")
    say(f"  {'':5s}{'#Div':>8s}{'Price ret':>10s}{'AdjClose':>10s}{'CRSP TR':>12s}{'Diff(pp)':>9s}{'MaxDailyDiff':>11s}")
    for _, r in tr.iterrows():
        say(f"  {r.ticker:5s}{r.n_dividends:8d}{fmt_pct(r.ann_price_return):>10s}{fmt_pct(r.ann_adjclose_return):>10s}"
            f"{fmt_pct(r.ann_total_return_crsp):>12s}{r.adj_minus_crsp_pp:+9.3f}{r.max_abs_daily_diff_adj_vs_crsp:11.2e}")
    if not dv.empty:
        say(f"  Dividend reconciliation: {len(dv)} entries; maximum absolute difference between "
            f"the event-table amount and the amount back-solved from AdjClose: "
            f"{dv['diff'].abs().max():.4f} USD")
        facts["div_check_maxdiff"] = float(dv["diff"].abs().max())
    facts["n_div"] = {r.ticker: int(r.n_dividends) for _, r in tr.iterrows()}
    facts["tr_gap"] = {r.ticker: float(r.adj_minus_crsp_pp) for _, r in tr.iterrows()}
    facts["price_vs_adj"] = {r.ticker: (r.ann_adjclose_return - r.ann_price_return) * 100 for _, r in tr.iterrows()}

    # Splits
    say("")
    sp = pd.concat([events[events.type == "split"], events_x[events_x.type == "split"]]).drop_duplicates()
    sp_main = sp[sp.ticker.isin(ASSETS) & (sp.date >= cal[0]) & (sp.date <= cal[-1])]
    say(f"  The four main assets had {len(sp_main)} share splits during the sample.")
    facts["n_split_main"] = len(sp_main)
    for _, e in sp.iterrows():
        df = load_raw(e.ticker, "X")
        if df is None:
            continue
        s = df.set_index("Date")
        if e.date not in s.index:
            continue
        i = s.index.get_loc(e.date)
        rc = s.Close.iloc[i] / s.Close.iloc[i - 1] - 1
        ra = s.AdjClose.iloc[i] / s.AdjClose.iloc[i - 1] - 1
        ok = abs(rc) < 0.25
        say(f"  Split check {e.ticker} {e.date.date()} ratio {e.ratio:g}:1 → that day's Close "
            f"return {rc * 100:+.2f}%, AdjClose return {ra * 100:+.2f}%  → "
            f"{'Close is already split-adjusted, no gap' if ok else 'Gap present, manual adjustment needed!'}")
        facts.setdefault("split_checks", []).append(f"{e.ticker} {e.date.date()} {e.ratio:g}-for-1: "
                                                    f"return on split date {rc * 100:+.2f}%")

    # ====================================================================== 7
    rule("7. ETF management fees")
    say("  ETF management fees accrue daily against fund assets and are already reflected in "
        "NAV and secondary-market prices;")
    say("  therefore all returns in this paper are [net of fees], and no further deduction is "
        "made, which would double-count the fee.")
    erows = []
    for t in ASSETS:
        e = EXPENSE[t]
        erows.append(dict(ticker=t, expense_ratio=e["er"], daily_drag=e["er"] / TD, note=e["note"], source=e["src"]))
        say(f"  {t:5s} {e['er'] * 100:.4f}%/yr (≈ {e['er'] / TD * 1e4:.3f} bp/day)  {e['note']}")
    pd.DataFrame(erows).to_csv(OUT / "tab_7_expense_ratios.csv", index=False)
    if R_btc is not None:
        z = pd.concat([R["IBIT"], R_btc.rename("BTC")], axis=1).dropna()
        yrs = len(z) / TD
        td = (np.log1p(z["IBIT"]).sum() - np.log1p(z["BTC"]).sum()) / yrs
        # The endpoint mismatch (4 hours) introduces noise: shift the first and last day for
        # a sensitivity check
        td_alt = (np.log1p(z["IBIT"].iloc[1:-1]).sum() - np.log1p(z["BTC"].iloc[1:-1]).sum()) / ((len(z) - 2) / TD)
        te = (z["IBIT"] - z["BTC"]).std() * math.sqrt(TD)
        say(f"  Empirical: IBIT's annualized log-return difference relative to BTC-USD is "
            f"{td * 100:+.2f}% (dropping the first and last day: {td_alt * 100:+.2f}%);")
        say(f"        annualized standard deviation of the daily return difference "
            f"{te * 100:.2f}% (mainly due to the 16:00 ET vs 00:00 UTC closing-time mismatch, "
            "not tracking error)")
        facts.update(ibit_btc_gap=td, ibit_btc_gap_alt=td_alt, ibit_btc_te=te)

    # ====================================================================== 8
    rule("8. Risk-free rate and Sharpe ratios")
    rfs = get_rf_snapshots(a.offline, a.rf_file, "2023-12-01", str((cal[-1] + pd.Timedelta(days=1)).date()))
    rf_cols: dict[str, pd.Series] = {}
    for name, df in rfs.items():
        rf_cols[name] = rf_daily_on_calendar(df, name, cal).iloc[1:]
    main_rf = next((n for n in ("DGS3MO", "DTB3", "IRX") if n in rf_cols), None)
    facts["rf_main"] = main_rf
    if main_rf is None:
        say("  !!! No risk-free-rate data available. Please download manually:")
        say("      https://fred.stlouisfed.org/series/DGS3MO  → Download → CSV")
        say("      save as data/raw/DGS3MO__%s.csv and rerun (or use --rf-file to specify a path)" % AS_OF)
    else:
        for n, s in rf_cols.items():
            say(f"  {n:6s} in-sample annualized mean {((1 + s).prod() ** (TD / len(s)) - 1) * 100:.3f}%   "
                f"missing {int(s.isna().sum())} days")
        facts["rf_mean"] = float((1 + rf_cols[main_rf]).prod() ** (TD / len(rf_cols[main_rf])) - 1)

    variants = {"rf = 0": pd.Series(0.0, index=R.index),
                f"constant {RF_CONST_ANNUAL:.1%}": pd.Series((1 + RF_CONST_ANNUAL) ** (1 / TD) - 1, index=R.index)}
    for n, s in rf_cols.items():
        variants[n] = s.reindex(R.index)
    srows = []
    for vn, rf in variants.items():
        for t in ASSETS:
            ex = (R[t] - rf).dropna()
            sr_d = ex.mean() / ex.std()
            se = math.sqrt((1 + 0.5 * sr_d ** 2) / len(ex)) * math.sqrt(TD)   # Lo (2002) iid standard error
            srows.append(dict(rf=vn, ticker=t, sharpe_ann=sr_d * math.sqrt(TD), se_lo2002=se,
                              mean_excess_ann=ex.mean() * TD, vol_ann=ex.std() * math.sqrt(TD)))
    st = pd.DataFrame(srows)
    st.to_csv(OUT / "tab_8_sharpe.csv", index=False)
    piv = st.pivot(index="ticker", columns="rf", values="sharpe_ann").loc[ASSETS]
    say("")
    say("  Annualized Sharpe ratio = mean(r − rf) / sd(r − rf) × √252 (simple returns), "
        "point estimates:")
    say("  " + f"{'':6s}" + "".join(f"{c:>16s}" for c in piv.columns))
    for t in ASSETS:
        say("  " + f"{t:6s}" + "".join(f"{piv.loc[t, c]:16.3f}" for c in piv.columns))
    if main_rf:
        m = st[st.rf == main_rf].set_index("ticker")
        say(f"  Main convention ({main_rf}) Lo (2002) standard errors: " +
            "  ".join(f"{t} {m.loc[t, 'sharpe_ann']:.2f}({m.loc[t, 'se_lo2002']:.2f})" for t in ASSETS))
        facts["sharpe"] = {t: (float(m.loc[t, "sharpe_ann"]), float(m.loc[t, "se_lo2002"])) for t in ASSETS}
        facts["sharpe_rf0"] = {t: float(piv.loc[t, "rf = 0"]) for t in ASSETS}
        out_r = R.copy()
        out_r.columns = [c for c in R.columns]
        for t in ASSETS:
            out_r[f"{t}_log"] = L[t]
        out_r[f"rf_daily_{main_rf}"] = rf_cols[main_rf]
        for t in ASSETS:
            out_r[f"{t}_excess"] = R[t] - rf_cols[main_rf]
        out_r.index.name = "Date"
        out_r.to_csv(PROC / f"returns_rf__{AS_OF}.csv")
        say(f"  Written data/processed/returns_rf__{AS_OF}.csv (with daily risk-free rate and "
            "excess returns)")

    # ====================================================================== 9
    rule("9. Simple vs log returns")
    srows = []
    for t in ASSETS:
        for kind, s in (("simple", R[t]), ("log", L[t])):
            srows.append(dict(ticker=t, kind=kind, mean_daily=s.mean(), sd_daily=s.std(),
                              skew=s.skew(), ex_kurt=s.kurt(), min=s.min(), max=s.max()))
    sl = pd.DataFrame(srows)
    sl.to_csv(OUT / "tab_9_simple_vs_log.csv", index=False)
    say(f"  {'':5s}{'':7s}{'Mean(bp)':>10s}{'SD':>9s}{'Skew':>8s}{'ExKurt':>10s}{'Min':>9s}{'Max':>9s}")
    for _, r in sl.iterrows():
        say(f"  {r['ticker']:5s}{r['kind']:7s}{r['mean_daily'] * 1e4:10.2f}{r['sd_daily'] * 100:8.2f}%"
            f"{r['skew']:8.2f}{r['ex_kurt']:10.2f}{r['min'] * 100:8.2f}%{r['max'] * 100:8.2f}%")
    say("  Uses: simple returns → portfolio returns, VaR/ES, Sharpe; log returns → GARCH-type "
        "models, aggregation over time.")

    # ================================================================ Appendix draft
    write_appendix(facts)
    (OUT / f"report__{AS_OF}.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nDone. Results in {OUT}/  (report__{AS_OF}.txt, data_appendix_draft.md, tab_*.csv)")


# ============================================================================
# English appendix draft
# ============================================================================
def write_appendix(f: dict) -> None:
    def g(k, default="[TBD]"):
        return f.get(k, default)

    sharpe_txt = "[risk-free series missing — see report]"
    if "sharpe" in f:
        sharpe_txt = "; ".join(f"{t} {v[0]:.2f} (s.e. {v[1]:.2f})" for t, v in f["sharpe"].items())
    div_txt = ", ".join(f"{t}: {n}" for t, n in f.get("n_div", {}).items())
    gap_txt = ", ".join(f"{t} {v:+.2f} pp" for t, v in f.get("price_vs_adj", {}).items())
    tr_txt = ", ".join(f"{t} {v:+.3f} pp" for t, v in f.get("tr_gap", {}).items())
    rf_name = {"DGS3MO": "the 3-month Treasury constant-maturity rate (FRED series DGS3MO)",
               "DTB3": "the 3-month Treasury bill secondary-market rate (FRED series DTB3), converted from "
                       "discount basis to bond-equivalent yield",
               "IRX": "the 13-week Treasury bill rate (Yahoo Finance ^IRX), converted from discount basis to "
                      "bond-equivalent yield"}.get(f.get("rf_main"), "[risk-free series]")
    split_txt = "; ".join(f.get("split_checks", [])) or "none observed"

    md = f"""# Appendix D. Data Construction

*Generated by `25_data_appendix.py` from snapshot `{AS_OF}`. All numbers below are reproduced by running
the public code on the archived raw files; see the repository README.*

## D.1 Source and snapshot
Daily open, high, low, close, adjusted close and volume for IBIT, GLD, TLT and SPY, the CBOE VIX index and
BTC-USD are downloaded from Yahoo Finance (chart API v8) on {g('downloaded')} UTC. The raw JSON
responses and the parsed CSV files are archived in `data/raw/` together with a manifest of SHA-256 hashes;
every subsequent step reads only these files and never re-queries the vendor, because Yahoo re-computes the
entire adjusted-close history after each new distribution. The estimation sample runs from {g('sample_start')}
(the first trading day of IBIT) to {g('sample_end')}, giving {g('n_days')} common trading days and
{g('n_ret')} daily returns.

## D.2 Price series: adjusted close versus total return
We use Yahoo's split- and dividend-adjusted close (`AdjClose`). IBIT and GLD are grantor trusts that paid no
distributions during the sample, so their adjusted and unadjusted closes are identical. SPY and TLT pay
distributions (number of ex-dates in the sample: {div_txt}); using unadjusted closes would understate annual
returns by {gap_txt}. Yahoo applies a multiplicative adjustment that is equivalent to reinvesting the
distribution at the pre-ex-date close; a CRSP-style total-return series that reinvests at the ex-date close
differs by {tr_txt} per year, which is immaterial for all results. Each distribution in the event file was
matched to the jump in the AdjClose/Close ratio (maximum discrepancy {f.get('div_check_maxdiff', float('nan')):.4f} USD).

## D.3 Splits
There were {g('n_split_main')} share splits in the four main assets during the sample. Yahoo's `Close` is
already split-adjusted; this was verified on the splits that occur among the alternative ETFs used in the
robustness appendix ({split_txt}).

## D.4 Trading calendar, holidays and non-synchronous trading
All four ETFs trade on U.S. exchanges and close at 16:00 ET, so their daily returns are synchronous. Series
are aligned by an inner join on the SPY trading calendar with no forward filling or interpolation. The
calendar was checked day by day against the NYSE holiday schedule ({g('n_holidays')} full-day closures in
the download window, including the 9 January 2025 National Day of Mourning); no trading day is missing and
no holiday contains data. The {g('n_early')} early-close (13:00 ET) sessions are retained. On the
{g('n_bond_only')} days when the U.S. bond market was closed but equities traded (Columbus Day, Veterans Day),
TLT still has an exchange price and is kept. The VIX (close 16:15 ET) is used only as a state variable.

Bitcoin trades continuously and Yahoo's BTC-USD daily bar closes at 00:00 UTC (19:00/20:00 ET). When BTC-USD
is used, we take the last BTC close on or before each U.S. trading day, so weekend and holiday returns
accrue to the next trading day. The four-hour offset produces a same-day correlation between IBIT and
BTC-USD of only {f.get('corr_btc_0', float('nan')):.3f}, whereas the weekly correlation is
{f.get('corr_btc_week', float('nan')):.3f}; the correlations with the previous and next day's BTC return are
{f.get('corr_btc_lag', float('nan')):.3f} and {f.get('corr_btc_lead', float('nan')):.3f}. The daily gap is therefore a
closing-time artefact rather than tracking error. Main results use ETF prices only, which are synchronous;
weekly correlations among the four ETFs are reported as a robustness check against closing-time effects.

## D.5 Missing values and outliers
Rows with a missing close or adjusted close are dropped at download; after alignment there are
{g('n_nan')} missing values and {g('n_dup')} duplicate dates. Returns are not winsorised or trimmed;
the {g('n_out')} daily returns more than five standard deviations from the mean are retained and listed
in `tab_5_data_quality.csv`.

## D.6 Expense ratios
ETF management fees accrue daily against net assets and are therefore already reflected in NAV and market
prices; all returns are net of fees and no further deduction is made. Headline expense ratios are
IBIT 0.25% (reduced to 0.12% on the first USD 5bn of assets for the 12 months from 11 January 2024),
GLD 0.40%, TLT 0.15% and SPY 0.0945%. Empirically, IBIT's annualised log return differs from BTC-USD by
{f.get('ibit_btc_gap', float('nan')) * 100:+.2f}% over the sample, consistent with the fee plus closing-time noise.

## D.7 Risk-free rate and Sharpe ratios
The risk-free rate is {rf_name}. The annualised yield observed on trading day t−1 is converted to a
daily rate as (1 + y)^(1/252) − 1 and applied to the return of day t; bond-market holidays are forward-filled.
Its average over the sample is {f.get('rf_mean', float('nan')) * 100:.2f}% per year. Sharpe ratios are computed from
simple daily excess returns and annualised by √252: {sharpe_txt}. Standard errors follow Lo (2002) under
i.i.d. returns. Results with rf = 0 and with a constant rate are reported in `tab_8_sharpe.csv`.

## D.8 Simple versus log returns
We compute both simple returns r_t = P_t/P_(t−1) − 1 and log returns ℓ_t = ln(P_t/P_(t−1)). Simple returns are
used wherever returns are aggregated across assets (portfolio returns, VaR/ES, Sharpe ratios), because
portfolio returns are linear in simple returns. Log returns are used for volatility models (GARCH) and for
aggregation over time. Summary statistics under both definitions are reported in `tab_9_simple_vs_log.csv`.

## D.9 Code and data availability
All code (Python 3.11 or later; pandas, numpy, scipy, statsmodels, arch) is publicly available at
https://github.com/ZHENG-NUBS/Jiang_Wang_Zheng_Bitcoin_Digital_Gold. The repository contains the
scripts in execution order, the SHA-256 manifests of the snapshots and a README describing how to reproduce
every table and figure. The price data are not redistributed; the snapshot is available from the
corresponding author on request.
"""
    (OUT / "data_appendix_draft.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()