# -*- coding: utf-8 -*-
"""
longdata.py — shared data loaders for the second-round scripts (30-38)

    load_etf()          ETF sample, 12 Jan 2024 - 15 Jun 2026 (data/processed/returns__20260921.csv):
                        IBIT, GLD, TLT, SPY simple returns and the VIX close.
    load_btc_long()     Bitcoin long sample, Sep 2014 - Jun 2026 (snapshot __20260921L), exactly as in
                        scripts 22 and 23: BTC-USD (00:00 UTC close) re-indexed to the SPY trading
                        calendar, GLD, TLT, SPY adjusted closes; VIX close; daily risk-free rate from ^IRX.
    load_gold_long()    Gold and equity long sample, Aug 2000 - Jun 2026 (snapshot __20261021G):
                        GOLD = COMEX front-month gold futures (GC=F) until GLD's first full trading
                        month, GLD afterwards; GC (futures throughout), GLD, SPY, TLT (from Jul 2002),
                        ^GSPC; VIX close; daily risk-free rate from ^IRX.

All returns are simple daily returns on the SPY trading calendar (inner alignment, no forward
filling of prices; a missing price gives a missing return). The sample ends on 15 June 2026.
The continuous futures series contains small artificial returns on roll dates; this is why GLD
replaces it once GLD has traded for a full month (from 1 Dec 2004).
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
END = pd.Timestamp("2026-06-15")
GLD_FROM = pd.Timestamp("2004-12-01")


def find_raw():
    for base in (HERE, os.getcwd()):
        d = os.path.join(base, "data", "raw")
        if os.path.isdir(d):
            return d
    return None


def find_processed():
    c = []
    for base in (HERE, os.getcwd()):
        for p in glob.glob(os.path.join(base, "data", "processed", "returns__*.csv")):
            if re.fullmatch(r"returns__\d{8}\.csv", os.path.basename(p)):
                c.append(os.path.abspath(p))
    return sorted(set(c), key=os.path.basename)[-1] if c else None


def _px(raw, name, suffix, col="AdjClose"):
    fs = sorted(glob.glob(os.path.join(raw, f"{name}__*{suffix}.csv")))
    fs = [f for f in fs if re.fullmatch(rf"{re.escape(name)}__\d{{8}}{suffix}\.csv", os.path.basename(f))]
    if not fs:
        return None
    d = pd.read_csv(fs[-1], parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    return d[col]


def load_etf(path=None):
    path = path or find_processed()
    if path is None:
        raise FileNotFoundError("data/processed/returns__YYYYMMDD.csv not found")
    d = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    out = d[["IBIT", "GLD", "TLT", "SPY"]].copy()
    out["VIX"] = d["VIX_Close"]
    return out


def load_btc_long(raw=None):
    raw = raw or find_raw()
    S = {t: _px(raw, f, "L", "Close" if t == "VIX" else "AdjClose")
         for t, f in (("BTC", "BTC-USD"), ("GLD", "GLD"), ("TLT", "TLT"), ("SPY", "SPY"), ("VIX", "IDX_VIX"))}
    if any(v is None for v in S.values()):
        raise FileNotFoundError("long-sample snapshot (__20260921L) incomplete")
    cal = S["SPY"].index
    P = pd.DataFrame({k: v.reindex(cal) for k, v in S.items() if k != "VIX"})
    P = P[P.index >= S["BTC"].index.min()]
    R = P.pct_change().dropna()
    R = R[R.index <= END]
    R["VIX"] = S["VIX"].reindex(R.index).values
    irx = _px(raw, "IDX_IRX", "L", "Close")
    R["RF"] = (irx.reindex(R.index).ffill() / 100 / 252).values if irx is not None else np.nan
    return R


def load_gold_long(raw=None, suffix="G"):
    raw = raw or find_raw()
    names = {"GC": ("GC_F", "Close"), "GLD": ("GLD", "AdjClose"), "SPY": ("SPY", "AdjClose"),
             "TLT": ("TLT", "AdjClose"), "GSPC": ("IDX_GSPC", "Close"), "VIX": ("IDX_VIX", "Close"),
             "IRX": ("IDX_IRX", "Close")}
    S = {k: _px(raw, f, suffix, c) for k, (f, c) in names.items()}
    if S["GC"] is None or S["SPY"] is None:
        return None
    cal = S["SPY"].index
    P = pd.DataFrame({k: S[k].reindex(cal) for k in ("GC", "GLD", "SPY", "TLT", "GSPC")})
    R = P.pct_change(fill_method=None)
    R = R[(R.index >= S["GC"].dropna().index.min()) & (R.index <= END)].iloc[1:]
    R["GOLD"] = np.where(R.index >= GLD_FROM, R["GLD"], R["GC"])
    R["VIX"] = S["VIX"].reindex(R.index).values
    R["RF"] = (S["IRX"].reindex(R.index).ffill() / 100 / 252).values if S["IRX"] is not None else np.nan
    R = R[R["SPY"].notna() & R["GOLD"].notna()]
    return R
