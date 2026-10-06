"""
01d_fetch_scope.py — Empirical scope expansion: all raw data

IBIT is not only compared with GLD, but also with Bitcoin spot, other spot Bitcoin ETFs,
other gold instruments, and broader stock and bond benchmarks; and Bitcoin's own
properties are separated from ETF-specific factors (expense ratio, tracking error,
liquidity, trading hours).

This script fetches four categories of data and writes them to data/raw/, with filenames
carrying the __20260921S suffix (S = scope), coexisting with the main snapshot
__20260921, the long sample __20260921L, and the ETF appendix __20260921X.

  1. Yahoo daily bars (OHLCV + AdjClose + dividends/splits)
  2. Bitcoin [hourly] candles (Coinbase primary, Bitstamp backup) — take the 16:00
     US Eastern price
  3. SPDR official GLD historical archive (including LBMA gold price and GLD NAV)
  4. iShares official IBIT historical NAV (optional)

Network access (following the approach already verified to work for the main snapshot)
    - Yahoo: query1 + no crumb; split price and events into two requests to
      avoid events=div,split triggering a 429
    - Other endpoints: 429 uses a separate long backoff (20/40/60s + jitter)
    - Failure of an individual ticker does not interrupt the whole run

Usage:
    python 01d_fetch_scope.py

Takes about 5–10 minutes to run (including hourly candle pagination). Prints a
success/failure list at the end.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

YAHOO = {
    # —— Bitcoin spot and spot ETFs ——
    "BTC-USD": ("bitcoin", "Bitcoin spot, Yahoo composite (daily bar closes 00:00 UTC)"),
    "IBIT": ("bitcoin", "iShares Bitcoin Trust ETF"),
    "FBTC": ("bitcoin", "Fidelity Wise Origin Bitcoin Fund"),
    "ARKB": ("bitcoin", "ARK 21Shares Bitcoin ETF"),
    "BITB": ("bitcoin", "Bitwise Bitcoin ETF"),
    "GBTC": ("bitcoin", "Grayscale Bitcoin Trust ETF"),
    "BTC":  ("bitcoin", "Grayscale Bitcoin Mini Trust (listed 2024-07-31)"),
    "HODL": ("bitcoin", "VanEck Bitcoin ETF"),
    "BTCO": ("bitcoin", "Invesco Galaxy Bitcoin ETF"),
    "BRRR": ("bitcoin", "CoinShares Bitcoin ETF (formerly Valkyrie)"),
    "EZBC": ("bitcoin", "Franklin Bitcoin ETF"),
    "BTCW": ("bitcoin", "WisdomTree Bitcoin Fund"),
    "DEFI": ("bitcoin", "Hashdex Bitcoin ETF (spot since 2024-03-27)"),
    "BITO": ("bitcoin", "ProShares Bitcoin Strategy ETF (futures-based)"),
    # —— Gold ——
    "GLD":  ("gold", "SPDR Gold Shares"),
    "IAU":  ("gold", "iShares Gold Trust"),
    "GLDM": ("gold", "SPDR Gold MiniShares Trust"),
    "SGOL": ("gold", "abrdn Physical Gold Shares ETF"),
    "BAR":  ("gold", "GraniteShares Gold Trust"),
    "AAAU": ("gold", "Goldman Sachs Physical Gold ETF"),
    "IAUM": ("gold", "iShares Gold Trust Micro"),
    "GC=F": ("gold", "COMEX gold futures, continuous front month"),
    # —— Equity benchmarks ——
    "SPY":  ("equity", "SPDR S&P 500 ETF Trust"),
    "VTI":  ("equity", "Vanguard Total Stock Market ETF"),
    "VT":   ("equity", "Vanguard Total World Stock ETF"),
    "QQQ":  ("equity", "Invesco QQQ Trust (Nasdaq-100)"),
    "EFA":  ("equity", "iShares MSCI EAFE ETF"),
    # —— Bond benchmarks ——
    "TLT":  ("bond", "iShares 20+ Year Treasury Bond ETF"),
    "IEF":  ("bond", "iShares 7-10 Year Treasury Bond ETF"),
    "AGG":  ("bond", "iShares Core U.S. Aggregate Bond ETF"),
    "SHY":  ("bond", "iShares 1-3 Year Treasury Bond ETF"),
    "TIP":  ("bond", "iShares TIPS Bond ETF"),
    # —— Other ——
    "^VIX": ("other", "CBOE Volatility Index"),
    "^IRX": ("other", "13-week U.S. Treasury bill yield (annualized, %)"),
}

START_DATE = "2024-01-01"
END_DATE   = "2026-06-16"
AS_OF      = "20260921S"

HERE    = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
TIMEOUT = 30
RETRIES = 4

YAHOO_URL    = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
COINBASE_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
GLD_ARCHIVE  = "https://www.spdrgoldshares.com/assets/dynamic/GLD/GLD_US_archive_EN.csv"
IBIT_NAV     = ("https://www.ishares.com/us/products/333011/ishares-bitcoin-trust-etf/"
                "1467271812596.ajax?fileType=xls&fileName=iShares-Bitcoin-Trust-ETF_fund&dataType=fund")

HOURLY_START = "2023-12-29T00:00:00Z"
HOURLY_END   = "2026-06-16T00:00:00Z"

SLEEP_BETWEEN_CALLS = (5.0, 10.0)   # Between the two Yahoo requests

# ----------------------------------------------------------------------------


def _find_proxy() -> str | None:
    """The proxy, if any, is read only from the environment (HTTPS_PROXY etc.)."""
    for k in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        if os.environ.get(k):
            return os.environ[k]
    return None


def _session(proxy: str | None) -> requests.Session:
    s = requests.Session()
    if proxy:
        s.proxies.update({"http": proxy, "https": proxy})
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36"),
    })
    return s


def _get(session, url, params=None, accept="application/json", label=""):
    """GET with retries. 429 uses a long backoff, other errors use a short backoff."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT, headers={"Accept": accept})
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"

            # 429 gets its own long backoff
            if r.status_code == 429 and attempt < RETRIES:
                wait = 20 * attempt + random.uniform(0, 5)
                print(f"    {label} 429 rate limited, waiting {wait:.0f}s before retry "
                      f"(attempt {attempt}/{RETRIES-1})")
                time.sleep(wait)
                continue
        except requests.RequestException as e:
            last = repr(e)

        if attempt < RETRIES:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"    {label} attempt {attempt} failed ({last}), retrying in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Still failing after {RETRIES} attempts: {last}")


def _epoch(date_str: str) -> int:
    """Midnight in UTC+8 (the time zone in which the snapshots were downloaded), fixed
    explicitly so that the request does not depend on the time zone of the machine."""
    return int(pd.Timestamp(date_str, tz="Etc/GMT-8").timestamp())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(ticker: str) -> str:
    return ticker.replace("^", "IDX_").replace("=", "_")


# ----------------------------------------------------------------------------
# 1. Yahoo daily bars (price and events split into two requests)
# ----------------------------------------------------------------------------

def fetch_yahoo_prices(session, ticker):
    """First request: prices only. events='history' is the lightest."""
    params = {
        "period1": _epoch(START_DATE),
        "period2": _epoch(END_DATE),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    payload = _get(session, YAHOO_URL.format(ticker=ticker), params,
                   label=f"[{ticker}] price").json()
    err = payload.get("chart", {}).get("error")
    if err:
        raise RuntimeError(f"Yahoo returned error {err}")
    res = payload["chart"]["result"][0]
    if not res.get("timestamp"):
        raise RuntimeError("Yahoo returned an empty series (possibly delisted or wrong symbol)")

    meta = res["meta"]
    tz_name = meta["exchangeTimezoneName"]
    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    dates = ts.tz_convert(tz_name).normalize().tz_localize(None)
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:
        adj = q["close"]
    px = pd.DataFrame({"Date": dates, "Open": q["open"], "High": q["high"], "Low": q["low"],
                       "Close": q["close"], "AdjClose": adj, "Volume": q["volume"]})
    px = px.dropna(subset=["Close", "AdjClose"]).drop_duplicates("Date").sort_values("Date")

    first_ts = pd.to_datetime(res["timestamp"][0], unit="s", utc=True)
    info = {"exchange_tz": tz_name, "currency": meta.get("currency"),
            "first_bar_utc": first_ts.isoformat(), "instrument_type": meta.get("instrumentType")}
    return px, payload, info


def fetch_yahoo_events(session, ticker):
    """Second request: events only. Returns (None, None) on failure."""
    params = {
        "period1": _epoch(START_DATE),
        "period2": _epoch(END_DATE),
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "false",
    }
    try:
        payload = _get(session, YAHOO_URL.format(ticker=ticker), params,
                       label=f"[{ticker}] events").json()
    except RuntimeError as e:
        print(f"    [{ticker}] events request failed: {e}")
        return None, None

    res = payload["chart"]["result"][0]
    tz_name = res["meta"]["exchangeTimezoneName"]
    ev_rows = []
    for kind in ("dividends", "splits"):
        for _, e in (res.get("events", {}).get(kind) or {}).items():
            d = (pd.to_datetime(e["date"], unit="s", utc=True)
                   .tz_convert(tz_name).normalize().tz_localize(None))
            ev_rows.append({"ticker": ticker, "date": d, "type": kind[:-1],
                            "amount": e.get("amount"),
                            "ratio": (e.get("numerator", 1) / e.get("denominator", 1)
                                      if kind == "splits" else None)})
    return pd.DataFrame(ev_rows), payload


# ----------------------------------------------------------------------------
# 2. Bitcoin hourly candles
# ----------------------------------------------------------------------------

def fetch_coinbase_hourly(session):
    """Coinbase Exchange public endpoint: max 300 candles per call,
    [time, low, high, open, close, volume], newest first."""
    start = pd.Timestamp(HOURLY_START)
    end = pd.Timestamp(HOURLY_END)
    step = pd.Timedelta(hours=300)
    rows, t, n_req = [], start, 0
    while t < end:
        t2 = min(t + step, end)
        params = {"granularity": 3600,
                  "start": t.isoformat().replace("+00:00", "Z"),
                  "end": t2.isoformat().replace("+00:00", "Z")}
        data = _get(session, COINBASE_URL, params, label="[Coinbase]").json()
        if isinstance(data, dict):
            raise RuntimeError(f"Coinbase returned {data}")
        rows.extend(data)
        n_req += 1
        if n_req % 10 == 0:
            print(f"    Coinbase fetched up to {t2.date()} ({len(rows)} candles)")
        t = t2
        time.sleep(0.35)          # Public endpoint rate limit is ~10/s; leave ample margin
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = (df.drop_duplicates("time").sort_values("time")
            [["time_utc", "open", "high", "low", "close", "volume"]])
    return df


def fetch_bitstamp_hourly(session):
    """Bitstamp public endpoint: max 1000 candles per call, paginate forward by start."""
    start = int(pd.Timestamp(HOURLY_START).timestamp())
    end = int(pd.Timestamp(HOURLY_END).timestamp())
    rows, t = [], start
    while t < end:
        data = _get(session, BITSTAMP_URL, {"step": 3600, "limit": 1000, "start": t},
                    label="[Bitstamp]").json()
        chunk = data.get("data", {}).get("ohlc", [])
        if not chunk:
            break
        rows.extend(chunk)
        last = int(chunk[-1]["timestamp"])
        if last <= t:
            break
        t = last + 3600
        time.sleep(0.5)
    df = pd.DataFrame(rows)
    for c in ("timestamp", "open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c])
    df["time_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df[df.timestamp < end].drop_duplicates("timestamp").sort_values("timestamp")
    return df[["time_utc", "open", "high", "low", "close", "volume"]]


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    proxy = _find_proxy()
    session = _session(proxy)
    print(f"Output directory: {RAW_DIR}\nProxy: {'set (environment)' if proxy else 'none'}\n")

    manifest = {
        "as_of": AS_OF,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_date": START_DATE, "end_date_exclusive": END_DATE,
        "sources": {"daily": "Yahoo Finance chart API v8 (query1, no crumb)",
                    "hourly_primary": "Coinbase Exchange public candles (granularity 3600)",
                    "hourly_backup": "Bitstamp public OHLC (step 3600)",
                    "gold_benchmark": GLD_ARCHIVE, "ibit_nav": IBIT_NAV},
        "files": {}, "tickers": {}, "failed": {},
    }
    ok_list, fail_list = [], []

    # ---- 1. Yahoo daily bars
    all_events = []
    for i, (ticker, (group, desc)) in enumerate(YAHOO.items(), 1):
        print(f"[{i:2d}/{len(YAHOO)}] {ticker:8s} {desc}")
        try:
            px, payload_price, info = fetch_yahoo_prices(session, ticker)
        except Exception as e:                       # noqa: BLE001
            print(f"    ✗ Price request failed: {e}")
            manifest["failed"][ticker] = str(e)
            fail_list.append(ticker)
            time.sleep(1)
            continue

        # Sleep between the two requests
        time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))

        ev, payload_events = fetch_yahoo_events(session, ticker)
        events_missing = ev is None

        base = safe_name(ticker)
        csv_path = RAW_DIR / f"{base}__{AS_OF}.csv"
        json_path = RAW_DIR / f"{base}__{AS_OF}.json"
        px.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(payload_price, ensure_ascii=False), encoding="utf-8")

        ev_json_path = None
        if not events_missing:
            ev_json_path = RAW_DIR / f"{base}__events__{AS_OF}.json"
            ev_json_path.write_text(json.dumps(payload_events, ensure_ascii=False),
                                    encoding="utf-8")
            if not ev.empty:
                all_events.append(ev)

        manifest["files"][csv_path.name] = sha256(csv_path)
        manifest["files"][json_path.name] = sha256(json_path)
        if ev_json_path is not None:
            manifest["files"][ev_json_path.name] = sha256(ev_json_path)

        n_div = int((ev.type == "dividend").sum()) if (ev is not None and not ev.empty) else 0
        n_spl = int((ev.type == "split").sum())    if (ev is not None and not ev.empty) else 0

        manifest["tickers"][ticker] = dict(
            group=group, description=desc, rows=int(len(px)),
            first_date=str(px.Date.min().date()),
            last_date=str(px.Date.max().date()),
            n_dividends=n_div, n_splits=n_spl,
            events_missing=events_missing,
            close_equals_adjclose=bool((px.Close - px.AdjClose).abs().max() < 1e-8),
            **info,
        )
        flag = " (events missing)" if events_missing else ""
        print(f"    ✓ {len(px)} rows  {px.Date.min().date()} → {px.Date.max().date()}{flag}")
        ok_list.append(ticker)
        time.sleep(3)   # Between tickers

    if all_events:
        ev_path = RAW_DIR / f"events__{AS_OF}.csv"
        pd.concat(all_events).sort_values(["ticker", "date"]).to_csv(ev_path, index=False)
        manifest["files"][ev_path.name] = sha256(ev_path)
    else:
        print("\nNote: no event data at all; events__*.csv was not generated.")

    # ---- 2. Hourly candles
    for name, fn in (("coinbase", fetch_coinbase_hourly), ("bitstamp", fetch_bitstamp_hourly)):
        print(f"\n[Hourly candles] {name}")
        try:
            df = fn(session)
            p = RAW_DIR / f"BTCUSD_1h_{name}__{AS_OF}.csv"
            df.to_csv(p, index=False)
            manifest["files"][p.name] = sha256(p)
            manifest["tickers"][f"BTCUSD_1h_{name}"] = dict(
                group="bitcoin_hourly", rows=int(len(df)),
                first=str(df.time_utc.min()), last=str(df.time_utc.max()))
            print(f"    ✓ {len(df)} candles  {df.time_utc.min()} → {df.time_utc.max()}")
            ok_list.append(f"BTC 1h {name}")
        except Exception as e:                       # noqa: BLE001
            print(f"    ✗ Failed: {e}")
            manifest["failed"][f"BTCUSD_1h_{name}"] = str(e)
            fail_list.append(f"BTC 1h {name}")

    # ---- 3/4. Official archives (saved as-is; parsing is done in the analysis script)
    for name, url, ext in (("GLD_archive", GLD_ARCHIVE, "csv"), ("IBIT_nav", IBIT_NAV, "xls")):
        print(f"\n[Official archive] {name}")
        try:
            r = _get(session, url, accept="*/*", label=f"[{name}]")
            if len(r.content) < 500:
                raise RuntimeError(f"Response content too short ({len(r.content)} bytes)")
            p = RAW_DIR / f"{name}__{AS_OF}.{ext}"
            p.write_bytes(r.content)
            manifest["files"][p.name] = sha256(p)
            print(f"    ✓ {len(r.content) / 1024:.0f} KB")
            ok_list.append(name)
        except Exception as e:                       # noqa: BLE001
            print(f"    ✗ Failed: {e}")
            manifest["failed"][name] = str(e)
            fail_list.append(name)

    man_path = RAW_DIR / f"manifest__{AS_OF}.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Succeeded: {len(ok_list)}; Failed: {len(fail_list)}")
    if fail_list:
        print("Failed: " + ", ".join(fail_list))
        must = {"BTC-USD", "IBIT", "GLD", "SPY", "TLT", "^VIX"}
        if must & set(fail_list) or {"BTC 1h coinbase", "BTC 1h bitstamp"} <= set(fail_list):
            print("⚠️ Some required items failed (core tickers or both hourly candle sources), "
                  "please check the network/proxy and rerun.")
        else:
            print("Core data is complete; failed items will simply be skipped in the analysis.")
    print(f"manifest: {man_path}")


if __name__ == "__main__":
    sys.exit(main())