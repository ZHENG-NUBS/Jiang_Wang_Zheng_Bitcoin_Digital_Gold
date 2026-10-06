"""
01c_fetch_extra.py — Asset extension snapshot, for the ETF appendix of item 8

Expanded ticker list

outputs
    data/raw/<TICKER>__20260921X.csv           OHLCV + AdjClose for each ticker
    data/raw/<TICKER>__20260921X.json          Raw Yahoo price response (for archival)
    data/raw/<TICKER>__events__20260921X.json  Raw Yahoo events response (may be missing)
    data/raw/events__20260921X.csv             Dividend and split details (may be empty)
    data/raw/manifest__20260921X.json          Download time, request params, SHA-256 of each file

methods
    python 01c_fetch_extra.py
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

TICKERS = {
    # -- Bitcoin spot ETFs (all listed 2024-01-11; GBTC is a trust conversion) --
    "IBIT": "iShares Bitcoin Trust ETF",
    "FBTC": "Fidelity Wise Origin Bitcoin Fund",
    "ARKB": "ARK 21Shares Bitcoin ETF",
    "BITB": "Bitwise Bitcoin ETF",
    "GBTC": "Grayscale Bitcoin Trust ETF",
    "HODL": "VanEck Bitcoin ETF",
    "BTCO": "Invesco Galaxy Bitcoin ETF",
    # -- Futures-based Bitcoin ETFs, as a control
    "BITO": "ProShares Bitcoin Strategy ETF (futures-based)",
    # -- Gold ETFs
    "GLD":  "SPDR Gold Shares",
    "IAU":  "iShares Gold Trust",
    "GLDM": "SPDR Gold MiniShares",
    # -- Anchor tickers --
    "SPY":  "SPDR S&P 500 ETF Trust",
    "TLT":  "iShares 20+ Year Treasury Bond ETF",
    "BTC-USD": "Bitcoin spot (Yahoo composite)",
    "^VIX": "CBOE Volatility Index",
}

START_DATE = "2024-01-01"
END_DATE   = "2026-06-16"   # The day after the sample cutoff (period2 is open-ended)
AS_OF      = "20260921X"    # X = extra, coexists with 20260921 / 20260921L

RAW_DIR = Path("data/raw")
PROXY   = None   # no address is stored here; set HTTPS_PROXY in the environment if needed
TIMEOUT = 30
RETRIES = 4

BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

# Wait between the two requests (seconds)
SLEEP_BETWEEN_CALLS = (5.0, 10.0)

# ----------------------------------------------------------------------------


def _session() -> requests.Session:
    s = requests.Session()
    if PROXY:
        s.proxies.update({"http": PROXY, "https": PROXY})
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36"),
    })
    return s


def _epoch(date_str: str) -> int:
    """Midnight in UTC+8 (the time zone in which the snapshots were downloaded), fixed
    explicitly so that the request does not depend on the time zone of the machine."""
    return int(pd.Timestamp(date_str, tz="Etc/GMT-8").timestamp())


def _get(session: requests.Session, ticker: str, params: dict,
         label: str, retries: int = RETRIES) -> requests.Response | None:
    """GET with retries. Returns None if all attempts fail (does not raise;
    leaves the decision to the caller)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = session.get(BASE_URL.format(ticker=ticker), params=params, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = repr(e)

        if attempt < retries:
            wait = 2 ** attempt + random.uniform(0, 1)
            print(f"    [{ticker}] {label} attempt {attempt} failed ({last_err}), "
                  f"retrying in {wait:.1f}s")
            time.sleep(wait)

    print(f"    [{ticker}] {label} still failed after {retries} attempts, "
          f"last error {last_err}")
    return None


def fetch_prices(session: requests.Session, ticker: str):
    """First request: prices only. events='history' is the lightest form
    (verified to work)."""
    params = {
        "period1": _epoch(START_DATE),
        "period2": _epoch(END_DATE),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    r = _get(session, ticker, params, "prices")
    if r is None:
        raise RuntimeError(f"{ticker}: price request failed")

    payload = r.json()
    err = payload.get("chart", {}).get("error")
    if err:
        raise RuntimeError(f"{ticker}: Yahoo returned error {err}")

    res = payload["chart"]["result"][0]
    meta = res["meta"]
    tz_name = meta["exchangeTimezoneName"]

    ts = pd.to_datetime(res["timestamp"], unit="s", utc=True)
    dates = ts.tz_convert(tz_name).normalize().tz_localize(None)

    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
    if adj is None:                     # Indices like ^VIX have no adjclose
        adj = q["close"]

    px = pd.DataFrame({
        "Date": dates,
        "Open": q["open"], "High": q["high"], "Low": q["low"],
        "Close": q["close"], "AdjClose": adj, "Volume": q["volume"],
    })

    before = len(px)
    px = px.dropna(subset=["Close", "AdjClose"]).drop_duplicates("Date").sort_values("Date")
    if before != len(px):
        print(f"    [{ticker}] dropped {before - len(px)} rows (missing Close/AdjClose)")

    return px, payload


def fetch_events(session: requests.Session, ticker: str):
    """Second request: events only. Returns (None, None) on failure.

    includeAdjustedClose=false, making the request lighter.
    """
    params = {
        "period1": _epoch(START_DATE),
        "period2": _epoch(END_DATE),
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "false",
    }
    r = _get(session, ticker, params, "events")
    if r is None:
        return None, None

    payload = r.json()
    res = payload["chart"]["result"][0]
    tz_name = res["meta"]["exchangeTimezoneName"]

    ev_rows = []
    events = res.get("events", {})
    for kind in ("dividends", "splits"):
        for _, e in (events.get(kind) or {}).items():
            d = (pd.to_datetime(e["date"], unit="s", utc=True)
                   .tz_convert(tz_name).normalize().tz_localize(None))
            ev_rows.append({
                "ticker": ticker, "date": d, "type": kind[:-1],
                "amount": e.get("amount"),
                "ratio": (e.get("numerator", 1) / e.get("denominator", 1)
                          if kind == "splits" else None),
            })
    return pd.DataFrame(ev_rows), payload


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = _session()

    manifest = {
        "as_of": AS_OF,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "Yahoo Finance chart API v8 (query1, no crumb)",
        "start_date": START_DATE,
        "end_date_exclusive": END_DATE,
        "note": ("Asset extension snapshot, for the ETF appendix of item 8. "
                 "Coexists with the main snapshot 20260921 / long sample 20260921L, "
                 "without overwriting each other. "
                 "Prices and events are fetched in two separate requests; Yahoo's "
                 "AdjClose is retroactively recomputed with new dividends, "
                 "so this directory is a point-in-time snapshot and should be "
                 "submitted together with the paper."),
        "files": {},
        "tickers": {},
    }

    all_events = []
    failed = []
    for ticker, desc in TICKERS.items():
        print(f"[{ticker}] {desc}")
        try:
            # --- First: prices ---
            px, payload_price = fetch_prices(session, ticker)

            # Give Yahoo a breather before the second request
            time.sleep(random.uniform(*SLEEP_BETWEEN_CALLS))

            # --- Second: events (failure is not fatal) ---
            ev, payload_events = fetch_events(session, ticker)
        except Exception as exc:
            # Individual tickers may have been renamed/delisted; must not abort the whole run
            print(f"    [!] Fetch failed, skipping: {type(exc).__name__}: {exc}")
            failed.append(ticker)
            continue

        events_missing = ev is None

        safe = ticker.replace("^", "IDX_")
        csv_path  = RAW_DIR / f"{safe}__{AS_OF}.csv"
        json_path = RAW_DIR / f"{safe}__{AS_OF}.json"

        px.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(payload_price, ensure_ascii=False), encoding="utf-8")

        ev_json_path = None
        if not events_missing:
            ev_json_path = RAW_DIR / f"{safe}__events__{AS_OF}.json"
            ev_json_path.write_text(json.dumps(payload_events, ensure_ascii=False),
                                    encoding="utf-8")

        if not events_missing and not ev.empty:
            all_events.append(ev)

        manifest["files"][csv_path.name]  = sha256(csv_path)
        manifest["files"][json_path.name] = sha256(json_path)
        if ev_json_path is not None:
            manifest["files"][ev_json_path.name] = sha256(ev_json_path)

        n_div = int((ev.type == "dividend").sum()) if (ev is not None and not ev.empty) else 0
        n_spl = int((ev.type == "split").sum())    if (ev is not None and not ev.empty) else 0

        meta = payload_price["chart"]["result"][0]["meta"]
        manifest["tickers"][ticker] = {
            "description": desc,
            "rows": int(len(px)),
            "first_date": str(px.Date.min().date()),
            "last_date": str(px.Date.max().date()),
            "exchange_tz": meta["exchangeTimezoneName"],
            "currency": meta.get("currency"),
            "n_dividends": n_div,
            "n_splits": n_spl,
            "events_missing": events_missing,
            "close_equals_adjclose": bool((px.Close - px.AdjClose).abs().max() < 1e-8),
        }

        flag = " (events missing)" if events_missing else ""
        print(f"    {len(px)} rows  {px.Date.min().date()} -> {px.Date.max().date()}"
              f"  {n_div} dividends{flag}")

        # Pause a bit more between each ticker
        time.sleep(3)

    if all_events:
        ev_path = RAW_DIR / f"events__{AS_OF}.csv"
        pd.concat(all_events).sort_values(["ticker", "date"]).to_csv(ev_path, index=False)
        manifest["files"][ev_path.name] = sha256(ev_path)
    else:
        print("\nNote: no event data at all, events__*.csv not generated.")
        print("     This does not affect price/adjusted analysis: Yahoo's AdjClose "
              "already incorporates dividend and split adjustments.")

    man_path = RAW_DIR / f"manifest__{AS_OF}.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed:
        print(f"\n[!] The following tickers failed to fetch and were skipped: {failed}")
        print("    The appendix script only uses successfully fetched tickers; "
              "it is not necessary for all of them to succeed.")
    print(f"\nDone. Snapshot written to {RAW_DIR}/, manifest: {man_path.name}")

if __name__ == "__main__":
    main()
