"""
01e_fetch_rf_long.py — Risk-free rate for the long sample (from 2014-09)

Script No. 23 performs out-of-sample portfolio comparisons on the long sample
(BTC-USD, GLD, SPY, TLT, 2014-09 -> 2026-06), and requires a daily short-term
Treasury bill rate: rates were near 0 from 2014-2021, and around 5% after 2023.
Using a constant would distort the Sharpe ratio and the maximum-Sharpe weights.

Two sources are fetched and written to data/raw/, with filenames carrying the
__20260921L suffix (coexisting with the long-sample snapshot):
  1. Yahoo ^IRX (13-week Treasury bill yield, annualized %)      -> IDX_IRX__20260921L.csv
  2. FRED DTB3 (3-month Treasury bill secondary market rate, annualized %)
                                                                  -> DTB3__20260921L.csv (backup and cross-check)

"""

from __future__ import annotations

import io
import os
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

HERE    = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"

AS_OF      = "20260921L"
START_DATE = "2014-09-01"
END_DATE   = "2026-06-17"            # exclusive
TIMEOUT    = 30
RETRIES    = 4

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5EIRX"   # %5E = ^
FRED_URL  = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# ----------------------------------------------------------------------------


def _find_proxy() -> str | None:
    """The proxy, if any, is read only from the environment (HTTPS_PROXY etc.)."""
    for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def _get(s: requests.Session, url, params, label: str):
    """GET with retries. 429 uses a long backoff; other errors use a short backoff."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = s.get(url, params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"

            if r.status_code == 429 and attempt < RETRIES:
                wait = 20 * attempt + random.uniform(0, 5)
                print(f"    {label} 429 rate-limited, waiting {wait:.0f}s before retrying")
                time.sleep(wait)
                continue
        except requests.RequestException as e:
            last = repr(e)

        if attempt < RETRIES:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"    {label} attempt {attempt} failed ({last}), retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"{label}: still failed after {RETRIES} attempts: {last}")


def _epoch(date_str: str) -> int:
    """Midnight in UTC+8 (the time zone in which the snapshots were downloaded), fixed
    explicitly so that the request does not depend on the time zone of the machine."""
    return int(pd.Timestamp(date_str, tz="Etc/GMT-8").timestamp())


def fetch_yahoo(s: requests.Session) -> pd.DataFrame:
    """Yahoo ^IRX. An index has no events and no adjclose, so params are minimal."""
    params = {
        "period1": _epoch(START_DATE),
        "period2": _epoch(END_DATE),
        "interval": "1d",
        # Key point: do not pass events. ^IRX is an index; passing it easily
        # triggers a 400 or rate limiting.
    }
    js = _get(s, YAHOO_URL, params, "Yahoo ^IRX").json()
    err = js.get("chart", {}).get("error")
    if err:
        raise RuntimeError(f"Yahoo returned error {err}")

    res = js["chart"]["result"][0]
    if not res.get("timestamp"):
        raise RuntimeError("Yahoo returned an empty series")

    tz = res["meta"]["exchangeTimezoneName"]
    dates = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
               .tz_convert(tz).normalize().tz_localize(None))
    q = res["indicators"]["quote"][0]

    df = pd.DataFrame({
        "Date": dates,
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": q["close"], "AdjClose": q["close"], "Volume": q["volume"],
    })
    return df.dropna(subset=["Close"]).drop_duplicates("Date").sort_values("Date")


def fetch_fred(s: requests.Session) -> pd.DataFrame:
    """FRED DTB3 CSV."""
    params = {"id": "DTB3", "cosd": START_DATE, "coed": END_DATE}
    r = _get(s, FRED_URL, params, "FRED DTB3")

    if "DTB3" not in r.text[:200]:
        raise RuntimeError(f"FRED did not return CSV data (first 200 chars: {r.text[:200]!r})")

    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["Date", "DTB3"]
    df["Date"] = pd.to_datetime(df["Date"])
    df["DTB3"] = pd.to_numeric(df["DTB3"], errors="coerce")
    return df.dropna()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    s = requests.Session()
    p = _find_proxy()
    if p:
        s.proxies.update({"http": p, "https": p})
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36"),
    })
    print(f"Output directory: {RAW_DIR}\nProxy: {p or 'none'}\n")

    ok = 0
    for name, fn, fname in (
        ("Yahoo ^IRX", fetch_yahoo, f"IDX_IRX__{AS_OF}.csv"),
        ("FRED DTB3",  fetch_fred,  f"DTB3__{AS_OF}.csv"),
    ):
        try:
            df = fn(s)
            out = RAW_DIR / fname
            df.to_csv(out, index=False)
            print(f"[OK] {name}: {len(df)} rows "
                  f"{df.Date.min().date()} -> {df.Date.max().date()} -> {out.name}")
            ok += 1
        except Exception as e:                       # noqa: BLE001
            print(f"[FAIL] {name} failed: {e}")
        time.sleep(2)   # Brief pause between the two sources

    if ok == 0:
        sys.exit("Both sources failed; please check the network or proxy.")
    if ok == 1:
        print("\nNote: only one source succeeded. One daily risk-free rate series "
              "is enough; the other is only for cross-checking.")
    print("\nDone. Running python 23_portfolio_robust.py afterwards will "
          "automatically use the daily risk-free rate.")


if __name__ == "__main__":
    main()