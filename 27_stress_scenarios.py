"""
26_stress_scenarios.py — Stress-testing methodology transparency + scenario expansion

    (a) Use tables to list the exact event windows, return aggregation method, shock
        construction method, and rebalancing assumptions;
    (b) For historical replays before IBIT's launch that use Bitcoin spot as an IBIT proxy,
        they must be labeled a "counterfactual proxy experiment" and must not be called
        "out-of-sample evidence" about this ETF itself;
    (c) Add inflation shocks, monetary-policy shocks, banking crises, geopolitical shocks,
        and concurrent liquidity crises.

This script produces
    Table M  Methodology specification table (window anchoring, return aggregation, shock
             construction, rebalancing, proxy rule, non-synchronous handling)
    Table W  Exact windows of all scenarios (event date, start/end trading day, number of
             trading days, basis for selection, evidence type)
    Table H  Historical scenario replay results (two portfolios × two rebalancing modes ×
             window maximum drawdown × worst single day)
             Each row is labeled with evidence type: ETF sample (IBIT actual data) /
             counterfactual proxy experiment (BTC-USD spot)
             In the ETF period, the "if BTC were used as proxy" result is also given →
             directly quantifying the proxy error
    Table S  Hypothetical scenarios (including "macro shock + concurrent liquidity
             crisis"); the shock construction method is in Table M
    Table C  Summary by scenario category
    Figure   fig_stress_scenarios.pdf/png


Data: the __20260921L long-sample snapshot (from 2014-09) and the __20260921 main snapshot
      (IBIT) in data/raw. No network access.

Usage (just run it in PyCharm)
    python 27_stress_scenarios.py
    python 27_stress_scenarios.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
AS_OF_LONG = "20260921L"
AS_OF_MAIN = "20260921"
RAW = Path("data/raw")
OUTDIR = Path("out/27_stress")
ETF_LAUNCH = pd.Timestamp("2024-01-11")
W = 1 / 3                                    # Portfolio weights: risky asset / SPY / TLT at 1/3 each (consistent with §2.3 of the paper)
POST_DAYS = 4                                # Announcement-type event window = [t0, t0+4], 5 trading days in total
TC_BP = 5.0                                  # Rebalancing transaction-cost sensitivity: 5bp per unit of turnover
ADVERSE_Q = 0.05                             # Liquidity overlay: residual at the 5% adverse quantile
SEED = 20260925

EVID_ETF = "ETF sample (IBIT actual)"
EVID_CF = "Counterfactual proxy experiment (BTC-USD spot in place of IBIT)"

# ----------------------------------------------------------------------------
# Scenario catalog. Dates are event dates (non-trading days roll forward to the next
# trading day).
# kind = "announce": announcement type, window [t0, t0+POST_DAYS]
# kind = "episode" : narrative type, window [start, end] are fixed dates
# ----------------------------------------------------------------------------
CATALOG = [
    # —— Inflation shocks: CPI release days above expectations ——
    dict(id="INF-1", cat="Inflation shock", kind="announce", date="2021-11-10",
         label="CPI Oct-2021 6.2% y/y, 30-year high", src="BLS CPI release"),
    dict(id="INF-2", cat="Inflation shock", kind="announce", date="2022-06-10",
         label="CPI May-2022 8.6% y/y, triggered 75bp hike", src="BLS CPI release"),
    dict(id="INF-3", cat="Inflation shock", kind="announce", date="2022-09-13",
         label="CPI Aug-2022 core surprise", src="BLS CPI release"),
    dict(id="INF-4", cat="Inflation shock", kind="announce", date="2024-04-10",
         label="CPI Mar-2024 above consensus", src="BLS CPI release"),
    dict(id="INF-5", cat="Inflation shock", kind="announce", date="2025-02-12",
         label="CPI Jan-2025 above consensus", src="BLS CPI release"),
    # —— Monetary-policy shocks: hawkish FOMC / central-bank speeches ——
    dict(id="MON-1", cat="Monetary policy shock", kind="announce", date="2018-12-19",
         label="FOMC hike; balance-sheet runoff 'on autopilot'", src="FOMC statement / press conference"),
    dict(id="MON-2", cat="Monetary policy shock", kind="announce", date="2022-05-04",
         label="FOMC +50bp; next-day repricing", src="FOMC statement"),
    dict(id="MON-3", cat="Monetary policy shock", kind="announce", date="2022-08-26",
         label="Powell Jackson Hole speech", src="Federal Reserve speech"),
    dict(id="MON-4", cat="Monetary policy shock", kind="announce", date="2022-11-02",
         label="FOMC +75bp; pause 'very premature'", src="FOMC press conference"),
    dict(id="MON-5", cat="Monetary policy shock", kind="announce", date="2024-12-18",
         label="FOMC 'hawkish cut', fewer 2025 cuts in SEP", src="FOMC statement / SEP"),
    # —— Banking crises ——
    dict(id="BNK-1", cat="Banking crisis", kind="episode", start="2023-03-08", end="2023-03-17",
         label="SVB / Signature failures, Credit Suisse", src="FDIC / Federal Reserve announcements"),
    dict(id="BNK-2", cat="Banking crisis", kind="episode", start="2023-04-28", end="2023-05-04",
         label="First Republic failure, regional-bank sell-off", src="FDIC announcement 2023-05-01"),
    # —— Geopolitical shocks ——
    dict(id="GEO-1", cat="Geopolitical shock", kind="episode", start="2020-01-03", end="2020-01-08",
         label="US strike on Soleimani, Iranian retaliation", src="news chronology"),
    dict(id="GEO-2", cat="Geopolitical shock", kind="episode", start="2022-02-24", end="2022-03-08",
         label="Russia's invasion of Ukraine, commodity spike", src="news chronology"),
    dict(id="GEO-3", cat="Geopolitical shock", kind="episode", start="2023-10-09", end="2023-10-13",
         label="Hamas attack on Israel (7 Oct, weekend)", src="news chronology"),
    dict(id="GEO-4", cat="Geopolitical shock", kind="episode", start="2024-04-12", end="2024-04-19",
         label="Iran–Israel strikes (13 & 19 Apr)", src="news chronology"),
    dict(id="GEO-5", cat="Geopolitical shock", kind="episode", start="2025-06-13", end="2025-06-24",
         label="Israel–Iran war, US strikes, ceasefire", src="news chronology"),
    dict(id="GEO-6", cat="Geopolitical shock", kind="episode", start="2025-04-03", end="2025-04-08",
         label="'Liberation Day' tariff shock", src="White House announcement 2025-04-02"),
    # —— Liquidity crises (concurrent with other shocks) ——
    dict(id="LIQ-1", cat="Liquidity crisis (concurrent)", kind="episode", start="2020-03-09", end="2020-03-23",
         label="COVID 'dash for cash': equities and Treasuries sold together", src="Fed facilities 2020-03-15/23"),
    dict(id="LIQ-2", cat="Liquidity crisis (concurrent)", kind="episode", start="2022-11-08", end="2022-11-21",
         label="FTX collapse: crypto-native liquidity crisis", src="FTX bankruptcy filing 2022-11-11"),
    dict(id="LIQ-3", cat="Liquidity crisis (concurrent)", kind="episode", start="2024-08-01", end="2024-08-05",
         label="Yen carry-trade unwind", src="BoJ hike 2024-07-31; market chronology"),
    dict(id="LIQ-4", cat="Liquidity crisis (concurrent)", kind="episode", start="2025-04-07", end="2025-04-11",
         label="Treasury basis-trade unwind during tariff shock", src="market chronology"),
]

LINES: list[str] = []


def w(s: str = "") -> None:
    print(s)
    LINES.append(s)


def md(rows, header) -> str:
    o = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    o += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(o) + "\n"


def pct(x, d=2) -> str:
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"


# ============================================================================
# Return aggregation (matching the definitions in Table M word for word)
# ============================================================================
def cum(r: np.ndarray) -> float:
    """Window cumulative return = Π(1 + r_t) − 1 (compounded simple returns)."""
    return float(np.prod(1 + r) - 1)


def port_rebal(R: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Daily rebalancing to target weights: portfolio daily return = Σ w_i r_{i,t}."""
    return R @ wts


def port_bh(R: np.ndarray, wts: np.ndarray) -> np.ndarray:
    """Buy-and-hold: enter at target weights at the start of the window, no further
    rebalancing. Returns the portfolio daily return series."""
    V = np.cumprod(1 + R, axis=0) * wts           # value of each asset
    tot = np.concatenate([[1.0], V.sum(1)])
    return tot[1:] / tot[:-1] - 1


def turnover_rebal(R: np.ndarray, wts: np.ndarray) -> float:
    """Cumulative one-way turnover from daily rebalancing (for the transaction-cost
    sensitivity)."""
    to = 0.0
    for r in R:
        drift = wts * (1 + r)
        drift = drift / drift.sum()
        to += 0.5 * np.abs(drift - wts).sum()
    return float(to)


def max_dd(r: np.ndarray) -> float:
    v = np.concatenate([[1.0], np.cumprod(1 + r)])
    return float((v / np.maximum.accumulate(v) - 1).min())


# ============================================================================
# Data
# ============================================================================
def load(ticker: str, asof: str) -> pd.Series:
    p = RAW / f"{ticker.replace('^', 'IDX_')}__{asof}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    d = pd.read_csv(p, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()
    return d["AdjClose"] if "AdjClose" in d else d["Close"]


def build_panel() -> tuple[pd.DataFrame, pd.Series]:
    spy = load("SPY", AS_OF_LONG)
    cal = spy.index
    px = pd.DataFrame({"SPY": spy,
                       "GLD": load("GLD", AS_OF_LONG).reindex(cal),
                       "TLT": load("TLT", AS_OF_LONG).reindex(cal)})
    btc = load("BTC-USD", AS_OF_LONG)
    # BTC trades 24/7: take the most recent close on or before day t (weekend/holiday moves
    # accrue to the next trading day)
    px["BTC"] = btc.reindex(btc.index.union(cal)).sort_index().ffill().reindex(cal)
    px["IBIT"] = load("IBIT", AS_OF_MAIN).reindex(cal)
    vix = load("^VIX", AS_OF_LONG).reindex(cal)
    R = px.pct_change()
    R.loc[R.index <= ETF_LAUNCH, "IBIT"] = np.nan    # no prior close on the listing day
    return R.iloc[1:], vix.iloc[1:]


def window_idx(idx: pd.DatetimeIndex, ev: dict) -> tuple[int, int, pd.Timestamp]:
    """Returns the first and last position (i0, i1) in the return series; the window
    return = r[i0..i1], i.e. from the close of the day before i0 to the close of i1."""
    if ev["kind"] == "announce":
        t0 = pd.Timestamp(ev["date"])
        i0 = int(idx.searchsorted(t0))
        i1 = min(i0 + POST_DAYS, len(idx) - 1)
    else:
        t0 = pd.Timestamp(ev["start"])
        i0 = int(idx.searchsorted(t0))
        i1 = int(idx.searchsorted(pd.Timestamp(ev["end"]), side="right")) - 1
    return i0, i1, t0


# ============================================================================
# Self-test
# ============================================================================
def selftest() -> bool:
    ok = True

    def chk(lab, a, b, tol=1e-12):
        nonlocal ok
        g = abs(a - b) < tol
        ok &= g
        print(f"  [{'PASS' if g else 'FAIL'}] {lab:<50} {a:.10f} vs {b:.10f}")

    rng = np.random.default_rng(1)
    R = rng.normal(0, 0.02, (10, 3))
    wts = np.array([W, W, W])
    # Buy-and-hold = sum of asset cumulative returns weighted by initial weights
    chk("Buy-and-hold = Σ w_i (Π(1+r_i) − 1)", cum(port_bh(R, wts)), float((wts * (np.prod(1 + R, 0) - 1)).sum()))
    # For a single period the two rebalancing modes are identical
    chk("Single-day window: rebalancing = buy-and-hold", cum(port_rebal(R[:1], wts)), cum(port_bh(R[:1], wts)))
    # Max drawdown: a monotonically declining series = cumulative return
    r = np.full(5, -0.01)
    chk("MDD = cumulative return for a monotone decline", max_dd(r), cum(r))
    # Turnover: zero return, zero turnover
    chk("Turnover is 0 with zero returns", turnover_rebal(np.zeros((5, 3)), wts), 0.0)
    # Window location: an announce window has length POST_DAYS + 1
    idx = pd.bdate_range("2024-01-01", periods=30)
    i0, i1, _ = window_idx(idx, dict(kind="announce", date="2024-01-06"))  # Saturday → Monday
    chk("Announcement date on a weekend rolls to the next trading day", float(idx[i0] == pd.Timestamp("2024-01-08")), 1.0)
    chk("Announcement window has POST_DAYS+1 trading days", float(i1 - i0 + 1), float(POST_DAYS + 1))
    print("\nSelf-test " + ("all passed." if ok else "has failures!"))
    return ok


# ============================================================================
# Main program
# ============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    here = Path(__file__).resolve().parent
    cands = [here, Path.cwd()]
    root = next((c for c in cands if (c / RAW / f"SPY__{AS_OF_LONG}.csv").exists()), None)
    if root is None:
        sys.exit(f"Cannot find data/raw/SPY__{AS_OF_LONG}.csv. Run this script from the repository root. Tried {cands}")
    os.chdir(root)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Project directory: {Path.cwd()}")

    R, vix = build_panel()
    idx = R.index
    wts = np.array([W, W, W])

    w("# Stress testing: methodology specification and scenario expansion (reviewer comment 10)\n")
    w(f"Sample {idx[0].date()} → {idx[-1].date()}, {len(idx)} SPY trading days. "
      f"IBIT listing date {ETF_LAUNCH.date()}.\n")

    # ================================================================ Table M
    w("## Table M. Stress-testing methodology specification\n")
    M = [
        ["Trading calendar", "SPY (NYSE) trading days; all assets aligned to this calendar, no interpolation"],
        ["Prices", "Yahoo Finance adjusted close (AdjClose, with dividends reinvested); VIX uses the close"],
        ["Daily return", "Simple return r_t = P_t / P_{t−1} − 1"],
        ["BTC non-synchronous handling", "BTC-USD trades 24/7 and closes at 00:00 UTC; take the most recent close on or before day t, folding weekend and US-holiday moves into the next trading day"],
        ["Event date t0", "The event date; if it is a non-trading day (weekend/holiday), roll forward to the next trading day"],
        ["Window (announcement type)", f"[t0, t0+{POST_DAYS}]: from the close of the day before t0 to the close of t0+{POST_DAYS}, {POST_DAYS + 1} trading days in total; the t0 single-day return is also reported"],
        ["Window (narrative type)", "[start, end] fixed dates, see the basis column of Table W; from the close of the day before start to the close of end"],
        ["Window (rule type)", "SPY drawdown ≥ 8%: prior-peak-day close → trough-day close (script 15, with a 5%/8%/10% threshold sensitivity)"],
        ["Return aggregation", "Window cumulative return = Π(1 + r_t) − 1 (compounded); the window maximum drawdown (peak-to-trough) and worst single day are also reported"],
        ["Portfolio", "Equal-weighted 1/3: {bitcoin sleeve, SPY, TLT} vs {GLD, SPY, TLT}; only one asset is swapped"],
        ["Rebalancing (main convention)", "Daily rebalancing back to 1/3 at the close, no transaction cost"],
        ["Rebalancing (robustness)", f"Buy-and-hold: enter at 1/3 at the start of the window, no adjustment within the window; also report daily rebalancing under {TC_BP:.0f}bp one-way cost"],
        ["Bitcoin sleeve (ETF period)", f"The first trading day at or after {ETF_LAUNCH.date()}: IBIT actual returns → evidence type '{EVID_ETF}'"],
        ["Bitcoin sleeve (pre-listing)", f"BTC-USD spot returns in place of IBIT → evidence type '{EVID_CF}'; not treated as out-of-sample evidence about IBIT"],
        ["Proxy error", "In the ETF period, every scenario is recomputed using BTC as proxy, and the difference is reported"],
        ["Shock construction (historical replay)", "For each window in Table W, the actual daily returns are replayed directly, with no scaling"],
        ["Shock construction (hypothetical scenarios)", "See Table S: the SPY/TLT shocks take the actual magnitude of the worst SPY episode in that category (×1 and ×1.5); "
                           "the conditional responses of bitcoin/gold = the fitted value from the weekly regression r_i = a + b_SPY·SPY + b_TLT·TLT + e, scaled by the window length"],
        ["Liquidity overlay", f"The SPY shock is kept at the macro-scenario level (no double-counting of equity declines); two additional liquidity channels are added: "
                   f"(i) bond-hedge failure: TLT shock = min(macro TLT, 0) + the worst 5-day TLT return in-sample when VIX≥30 and stocks and bonds fell together; "
                   f"(ii) diversification failure: the bitcoin and gold regression residuals are both taken at the {int(ADVERSE_Q * 100)}% adverse quantile"],
    ]
    w(md(M, ["Item", "Rule"]))
    pd.DataFrame(M, columns=["item", "rule"]).to_csv(OUTDIR / "tab_M_methodology.csv", index=False)

    # ================================================================ Table W + H
    rows_w, rows_h, rec = [], [], []
    for ev in CATALOG:
        i0, i1, t0 = window_idx(idx, ev)
        if i0 >= len(idx) or i1 < i0:
            continue
        d0, d1 = idx[i0], idx[i1]
        etf = d0 > ETF_LAUNCH
        sub = R.iloc[i0:i1 + 1]
        sleeve = "IBIT" if etf else "BTC"
        evid = EVID_ETF if etf else EVID_CF
        out = dict(id=ev["id"], category=ev["cat"], label=ev["label"], source=ev["src"],
                   event_date=str(t0.date()), window_start=str(d0.date()), window_end=str(d1.date()),
                   anchor_close=str(idx[i0 - 1].date()) if i0 > 0 else "",
                   n_days=i1 - i0 + 1, evidence=evid, vix_peak=float(vix.iloc[i0:i1 + 1].max()))
        for c in ("SPY", "TLT", "GLD", "BTC", "IBIT"):
            out[c] = cum(sub[c].values) if sub[c].notna().all() else np.nan
        out["t0_SPY"] = float(sub["SPY"].iloc[0])
        for name, legs in (("P_btc", [sleeve, "SPY", "TLT"]), ("P_gld", ["GLD", "SPY", "TLT"]),
                           ("P_btcproxy", ["BTC", "SPY", "TLT"])):
            X = sub[legs].values
            if np.isnan(X).any():
                continue
            rr = port_rebal(X, wts)
            out[f"{name}_rebal"] = cum(rr)
            out[f"{name}_bh"] = cum(port_bh(X, wts))
            out[f"{name}_mdd"] = max_dd(rr)
            out[f"{name}_worst"] = float(rr.min())
            out[f"{name}_rebal_tc"] = cum(rr) - turnover_rebal(X, wts) * TC_BP / 1e4
        out["proxy_error"] = out.get("P_btc_rebal", np.nan) - out.get("P_btcproxy_rebal", np.nan) if etf else np.nan
        rec.append(out)
        rows_w.append([ev["id"], ev["cat"], ev["label"], out["event_date"],
                       f"{out['anchor_close']} close", f"{out['window_end']} close", out["n_days"],
                       "Announcement [t0,t0+4]" if ev["kind"] == "announce" else "Narrative (fixed dates)",
                       ev["src"], "ETF sample" if etf else "**Counterfactual proxy experiment**"])
    H = pd.DataFrame(rec)
    H.to_csv(OUTDIR / "tab_H_historical.csv", index=False)

    w("## Table W. Scenario windows (exact dates)\n")
    w("Window return = start close → end close. 'Start close' is the last trading day before the event.\n")
    w(md(rows_w, ["ID", "Category", "Event", "Event date", "Start", "End", "Trading days", "Window rule", "Basis", "Evidence type"]))
    ov = []
    for i in range(len(H)):
        for j in range(i + 1, len(H)):
            if H.window_start[i] <= H.window_end[j] and H.window_start[j] <= H.window_end[i]:
                ov.append(f"{H.id[i]} and {H.id[j]}")
    if ov:
        w("Window overlaps (results are not independent; note this when aggregating): " + "; ".join(ov) + ".\n")

    w("## Table H. Historical scenario replays\n")
    w(f"Portfolios are daily rebalanced (main convention); parentheses show buy-and-hold. MDD = window maximum drawdown (rebalancing convention). "
      f"In **counterfactual proxy experiment** rows the bitcoin sleeve is BTC-USD spot, not IBIT.\n")
    rows = []
    for _, h in H.iterrows():
        tag = "ETF" if h["evidence"] == EVID_ETF else "CF-proxy"
        rows.append([h["id"], tag, f"{h['window_start']}→{h['window_end']}", f"{h['vix_peak']:.1f}",
                     pct(h["SPY"]), pct(h["TLT"]), pct(h["GLD"]),
                     pct(h["IBIT"] if tag == "ETF" else h["BTC"]),
                     f"{pct(h['P_btc_rebal'])} ({pct(h['P_btc_bh'])})",
                     f"{pct(h['P_gld_rebal'])} ({pct(h['P_gld_bh'])})",
                     pct(h["P_btc_rebal"] - h["P_gld_rebal"]),
                     f"{pct(h['P_btc_mdd'])} / {pct(h['P_gld_mdd'])}",
                     pct(h["proxy_error"]) if tag == "ETF" else "n/a"])
    w(md(rows, ["ID", "Evidence", "Window", "Peak VIX", "SPY", "TLT", "GLD", "Bitcoin sleeve",
                "Bitcoin portfolio rebalanced (buy-and-hold)", "Gold portfolio rebalanced (buy-and-hold)", "Difference", "MDD bitcoin/gold", "Proxy error"]))
    w("Proxy error = in the ETF period, within the same window, the difference between the bitcoin portfolio using IBIT and using BTC-USD (measuring how reliable the counterfactual proxy is).\n")
    if H["proxy_error"].notna().any():
        pe = H["proxy_error"].dropna()
        w(f"Across the {len(pe)} scenarios in the ETF period, the proxy error is on average {pe.mean() * 100:+.2f}pp, "
          f"with a median absolute value of {pe.abs().median() * 100:.2f}pp and a maximum of {pe.abs().max() * 100:.2f}pp.\n")
    tc_gap = (H["P_btc_rebal"] - H["P_btc_rebal_tc"]).max()
    w(f"Transaction-cost sensitivity: with daily rebalancing at a {TC_BP:.0f}bp one-way cost, the bitcoin portfolio loses at most {tc_gap * 100:.3f}pp across all scenarios.\n")

    # ================================================================ Table C
    w("## Table C. Summary by scenario category (rebalancing convention)\n")
    rows, C = [], []
    for cat, g in H.groupby("category", sort=False):
        for ev_kind, gg in (("ETF", g[g.evidence == EVID_ETF]), ("CF-proxy", g[g.evidence == EVID_CF])):
            if gg.empty:
                continue
            d = gg["P_btc_rebal"] - gg["P_gld_rebal"]
            rows.append([cat, ev_kind, len(gg), pct(gg["SPY"].mean()), pct(gg["P_btc_rebal"].mean()),
                         pct(gg["P_gld_rebal"].mean()), pct(d.mean()), f"{int((d < 0).sum())}/{len(gg)}"])
            C.append(dict(category=cat, evidence=ev_kind, n=len(gg), spy=gg["SPY"].mean(),
                          p_btc=gg["P_btc_rebal"].mean(), p_gld=gg["P_gld_rebal"].mean(), diff=d.mean(),
                          n_btc_worse=int((d < 0).sum())))
    w(md(rows, ["Category", "Evidence", "Scenarios", "Mean SPY", "Mean bitcoin portfolio", "Mean gold portfolio", "Mean difference", "Times bitcoin portfolio was worse"]))
    pd.DataFrame(C).to_csv(OUTDIR / "tab_C_by_category.csv", index=False)
    w("The ETF sample and the counterfactual proxy experiment are summarized **on separate rows and not pooled**. The number of scenarios is small, so this is descriptive only, with no significance inference.\n")

    # ================================================================ Table S
    w("## Table S. Hypothetical scenarios (including concurrent liquidity crises)\n")
    wk = (1 + R).resample("W-FRI").prod() - 1
    wk = wk[(R.notna()).resample("W-FRI").sum()["SPY"] >= 3]      # weeks with at least 3 trading days
    vix_wk = vix.resample("W-FRI").max().reindex(wk.index)

    def fit_beta(y: pd.Series, X: pd.DataFrame):
        d = pd.concat([y, X], axis=1).dropna()
        A = np.column_stack([np.ones(len(d)), d.iloc[:, 1:].values])
        b, *_ = np.linalg.lstsq(A, d.iloc[:, 0].values, rcond=None)
        e = d.iloc[:, 0].values - A @ b
        return b, e, len(d)

    BETA = {}
    specs = [("IBIT", "ETF period (IBIT actual)", wk.index > ETF_LAUNCH, "IBIT"),
             ("BTC-full", "Full sample 2014–2026 (BTC proxy, counterfactual)", wk.index > wk.index[0], "BTC"),
             ("BTC-stress", "Full sample high-VIX weeks (weekly VIX peak > 25, BTC proxy, counterfactual)", vix_wk > 25, "BTC")]
    brow = []
    for key, desc, mask, col in specs:
        bb, eb, nb = fit_beta(wk.loc[mask, col], wk.loc[mask, ["SPY", "TLT"]])
        bg, eg, ng = fit_beta(wk.loc[mask, "GLD"], wk.loc[mask, ["SPY", "TLT"]])
        BETA[key] = dict(btc=(bb, eb), gld=(bg, eg), n=nb, desc=desc)
        brow.append([desc, nb, f"{bb[1]:+.2f}", f"{bb[2]:+.2f}", f"{eb.std(ddof=3) * 100:.2f}%",
                     f"{bg[1]:+.2f}", f"{bg[2]:+.2f}", f"{eg.std(ddof=3) * 100:.2f}%"])
    w("Conditional response model (weekly, W-FRI): r_i = a + b_SPY·r_SPY + b_TLT·r_TLT + e\n")
    w(md(brow, ["Estimation sample", "Weeks", "Bitcoin b_SPY", "Bitcoin b_TLT", "Bitcoin residual sd",
                "Gold b_SPY", "Gold b_TLT", "Gold residual sd"]))

    # Shock magnitude anchoring: the worst SPY episode in each category
    anchors = {}
    for cat, g in H.groupby("category", sort=False):
        r0 = g.loc[g["SPY"].idxmin()]
        anchors[cat] = dict(spy=r0["SPY"], tlt=r0["TLT"], days=int(r0["n_days"]), ref=r0["id"])
    # Liquidity-overlay bond component: among windows with VIX ≥ 30 and stocks and bonds
    # falling together over 5 days, the worst 5-day TLT return (data-driven)
    tlt5 = (1 + R["TLT"]).rolling(5).apply(np.prod, raw=True) - 1
    spy5 = (1 + R["SPY"]).rolling(5).apply(np.prod, raw=True) - 1
    joint = tlt5[(vix.rolling(5).max() >= 30) & (tlt5 < 0) & (spy5 < 0)].dropna()
    liq = None
    if len(joint):
        d_liq = joint.idxmin()
        liq = dict(tlt=float(joint.min()), end=d_liq, start=idx[idx.get_loc(d_liq) - 4], spy=float(spy5.loc[d_liq]))
        w(f"Liquidity-overlay bond component: {liq['start'].date()} → {liq['end'].date()}, TLT {pct(liq['tlt'])}, "
          f"SPY over the same period {pct(liq['spy'])} (the worst 5-day TLT when VIX≥30 and stocks and bonds fell together).\n")

    def scenario(spy, tlt, days, key, adverse=False):
        h = days / 5.0                                            # holding period in weeks
        out = {}
        for leg in ("btc", "gld"):
            b, e = BETA[key][leg]
            mu = b[0] * h + b[1] * spy + b[2] * tlt
            if adverse:
                mu += np.quantile(e, ADVERSE_Q) * math.sqrt(h)
            out[leg] = mu
        out["P_btc"] = W * (out["btc"] + spy + tlt)
        out["P_gld"] = W * (out["gld"] + spy + tlt)
        return out

    srows, S = [], []
    for cat, an in anchors.items():
        if cat.startswith("Liquidity"):
            continue
        for sev in (1.0, 1.5):
            base = dict(spy=an["spy"] * sev, tlt=an["tlt"] * sev, days=an["days"])
            variants = [("Macro shock", base, False)]
            if liq is not None:
                variants.append(("Macro shock + liquidity crisis",
                                 dict(spy=base["spy"], tlt=min(base["tlt"], 0) + liq["tlt"], days=an["days"]), True))
            for vname, sh, adv in variants:
                for key in ("IBIT", "BTC-stress"):
                    o = scenario(sh["spy"], sh["tlt"], sh["days"], key, adverse=adv)
                    tag = "ETF β" if key == "IBIT" else "CF-proxy stressed β"
                    srows.append([cat, f"×{sev:g}", vname, an["ref"] + (" + liquidity overlay" if adv else ""), sh["days"],
                                  pct(sh["spy"]), pct(sh["tlt"]), tag, pct(o["btc"]), pct(o["gld"]),
                                  pct(o["P_btc"]), pct(o["P_gld"]), pct(o["P_btc"] - o["P_gld"])])
                    S.append(dict(category=cat, severity=sev, variant=vname, anchor=an["ref"], days=sh["days"],
                                  spy_shock=sh["spy"], tlt_shock=sh["tlt"], beta_source=tag,
                                  btc=o["btc"], gld=o["gld"], p_btc=o["P_btc"], p_gld=o["P_gld"]))
    w("Shock anchoring: for each category, the actual cumulative SPY and TLT returns of the worst-SPY historical window (×1 and ×1.5). "
      "Liquidity overlay: SPY unchanged; TLT = min(macro TLT, 0) + the bond component above (stocks and bonds falling together, bond hedge failing); "
      f"the bitcoin and gold residuals are both taken at the {int(ADVERSE_Q * 100)}% adverse quantile (diversification failing). "
      "One-period shock, no rebalancing within the shock period (the two rebalancing conventions coincide for a single period). "
      "The liquidity-crisis category itself is not used as a macro anchor, because it is an overlay layer.\n")
    w(md(srows, ["Category", "Severity", "Scenario", "Anchor", "Days", "SPY shock", "TLT shock", "β source",
                 "Bitcoin sleeve", "Gold sleeve", "Bitcoin portfolio", "Gold portfolio", "Difference"]))
    pd.DataFrame(S).to_csv(OUTDIR / "tab_S_hypothetical.csv", index=False)
    w("Note: the ETF β is based on only about 2.4 years of weekly data, so the estimation error is large; the 'CF-proxy stressed β' uses BTC data from high-VIX weeks since 2014 "
      "and is a counterfactual proxy. Both are reported side by side to show how sensitive the conclusion is to the source of β.\n")

    # ================================================================ Figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        Hs = H.sort_values(["category", "window_start"])
        fig, ax = plt.subplots(figsize=(12, 0.34 * len(Hs) + 1.8))
        y = np.arange(len(Hs))
        cf = (Hs["evidence"] == EVID_CF).values
        ax.barh(y + 0.2, Hs["P_btc_rebal"] * 100, 0.4, color="#d97706",
                hatch=["//" if c else "" for c in cf], edgecolor="white")
        ax.barh(y - 0.2, Hs["P_gld_rebal"] * 100, 0.4, color="#1f4e79")
        ax.set_yticks(y)
        ax.set_yticklabels([f"{i}  {s}" for i, s in zip(Hs["id"], Hs["window_start"])], fontsize=8)
        ax.axvline(0, color="k", lw=.6)
        ax.invert_yaxis()
        ax.set_xlabel("window return, % (daily rebalanced, equal-weight 1/3)")
        ax.legend(handles=[Patch(color="#d97706", label="Bitcoin-sleeve portfolio (IBIT, ETF sample)"),
                           Patch(facecolor="#d97706", hatch="//", edgecolor="white",
                                 label="Bitcoin-sleeve portfolio (BTC-USD, counterfactual proxy)"),
                           Patch(color="#1f4e79", label="Gold portfolio (GLD)")], fontsize=8, loc="lower left")
        ax.set_title("Historical stress scenarios: inflation, monetary, banking, geopolitical, liquidity")
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(OUTDIR / f"fig_stress_scenarios.{ext}", dpi=150)
        plt.close()
        w("Figure: `fig_stress_scenarios` (hatched fill = counterfactual proxy experiment).\n")
    except Exception as e:  # noqa: BLE001
        w(f"(plotting skipped: {e})\n")

    # ================================================================ Text
    etf = H[H.evidence == EVID_ETF]
    cfp = H[H.evidence == EVID_CF]
    d_etf = etf["P_btc_rebal"] - etf["P_gld_rebal"]
    d_cf = cfp["P_btc_rebal"] - cfp["P_gld_rebal"]
    liq_rows = [r for r in S if r["variant"] != "Macro shock" and r["beta_source"] == "ETF β"]
    worst = min(liq_rows, key=lambda r: r["p_btc"]) if liq_rows else None
    pe = H["proxy_error"].dropna()
    w("## Paragraph ready to go into the paper (English)\n")
    w(f"""> **Stress-test design.** Table M specifies the stress-testing protocol: the trading calendar, event-date
> anchoring, window conventions, return aggregation, portfolio construction, rebalancing assumptions and shock
> construction; Table W lists the exact window of every scenario. We replay {len(H)} historical scenarios in five
> categories—inflation shocks (CPI surprises), monetary-policy shocks (hawkish FOMC decisions and speeches), banking
> crises, geopolitical shocks, and liquidity crises that coincided with other shocks—in addition to the rule-based
> equity-drawdown episodes of Section X. Window returns compound daily simple returns from the close before the
> event to the close at the end of the window; portfolios are rebalanced daily to equal weights, with buy-and-hold
> and a {TC_BP:.0f}bp transaction-cost variant as robustness checks.
>
> **Evidence type.** Only the {len(etf)} scenarios after IBIT's launch on {ETF_LAUNCH.date()} use IBIT itself. For the
> {len(cfp)} earlier scenarios the bitcoin sleeve is proxied by BTC-USD spot returns; we label these a
> *counterfactual proxy experiment*: they describe how a portfolio with bitcoin exposure would have behaved,
> not out-of-sample evidence on the ETF. In the ETF sample, replacing IBIT by BTC-USD changes the portfolio's
> window return by {pe.abs().median() * 100:.2f} percentage points in the median scenario (maximum {pe.abs().max() * 100:.2f}pp).
>
> **Results.** In the ETF sample the bitcoin-sleeve portfolio underperforms the gold portfolio in
> {int((d_etf < 0).sum())} of {len(etf)} scenarios (mean difference {d_etf.mean() * 100:+.2f}pp); in the counterfactual proxy
> experiment it does so in {int((d_cf < 0).sum())} of {len(cfp)} (mean {d_cf.mean() * 100:+.2f}pp). Table S reports hypothetical
> scenarios whose equity and bond shocks are anchored on the worst historical episode of each category and scaled
> by 1 and 1.5; the conditional responses of bitcoin and gold come from weekly regressions on SPY and TLT. Adding a
> concurrent liquidity crisis—Treasuries failing as a hedge (an additional {pct(liq['tlt']) if liq else 'n/a'}, the worst five-day
> Treasury decline with VIX ≥ 30 and equities also falling, {liq['start'].date() if liq else ''} to {liq['end'].date() if liq else ''}) and
> bitcoin and gold residuals at their 5% adverse quantile—produces a worst-case loss of
> {pct(worst['p_btc']) if worst else 'n/a'} for the bitcoin-sleeve portfolio versus {pct(worst['p_gld']) if worst else 'n/a'} for the gold portfolio
> ({worst['category'] if worst else ''}, severity ×{worst['severity'] if worst else '':g}).
""")
    w("(Numbers update automatically with the run; please check the dates and bases in Table W, and add or remove scenarios as needed.)\n")

    w("1. New Table M (methodology specification) and Table W (exact start/end dates, anchor close date, number of trading days, basis for selection for each scenario).")
    w(f"2. Return aggregation: simple returns compounded over the window; the maximum drawdown and worst single day are also reported. Rebalancing: daily rebalancing for the main convention, "
      f"with buy-and-hold and a {TC_BP:.0f}bp transaction cost for robustness.")
    w(f"3. All replays before IBIT's launch are labeled a \"counterfactual proxy experiment (BTC-USD spot in place of IBIT)\" and are listed and summarized separately from the ETF sample in both tables and figures; "
      f"in the ETF period the proxy error is also computed (median {pe.abs().median() * 100:.2f}pp).")
    w("4. Five new scenario categories: inflation shocks (5 CPI surprises), monetary-policy shocks (5 hawkish FOMC/speeches), banking crises (SVB, First Republic), "
      "geopolitical shocks (6), and concurrent liquidity crises (the 2020 dash for cash, FTX, the yen carry-trade unwind, the 2025 basis-trade unwind).")
    w("5. New hypothetical scenarios: anchored on each category's worst historical episode (×1, ×1.5), with a liquidity crisis overlaid (bond-hedge failure + residual at the 5% adverse quantile); "
      "the conditional responses of bitcoin and gold are estimated using both the ETF-period β and the proxy β from high-VIX weeks since 2014, reported side by side.")

    (OUTDIR / "stress_scenarios.md").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nDone. Results: {OUTDIR}/stress_scenarios.md")


if __name__ == "__main__":
    main()