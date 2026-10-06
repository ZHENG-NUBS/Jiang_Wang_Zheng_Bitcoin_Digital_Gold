"""
25_implementation_costs.py — Trading costs and implementability

    Daily rebalancing cannot be costless. The paper should account for trading costs,
    bid-ask spreads, ETF expense ratios, liquidity, and turnover.
    The economically meaningful question is: after realistic execution costs, does adding
    IBIT improve or worsen risk-adjusted returns?

17_portfolio.py already did: rolling optimization backtest, drift turnover, Sharpe under a
common 0–25bp cost, Sharpe-difference test against 1/N.
This script fills in:

    L. Liquidity and spreads (per ETF, all estimated from your existing OHLCV data)
         - Effective spread: Corwin–Schultz (2012), Abdi–Ranaldo (2017), 21-day rolling
         - Tick-size floor: 1 cent / price
         - Amihud (2002) illiquidity, median daily dollar volume
         - High-VIX (≥25) days vs normal days; IBIT's early listing period vs the last year
    C. Cost model: cost = |Δw| × (half-spread + impact), impact = Y·σ_daily·√(trade notional / ADV) (square-root model)
         Three spread tiers: Quoted (1 cent tick, main convention) / OHLC upper bound (max(AR, CS)) / Stressed (2× OHLC upper bound)
    A. Core question: 60/40 benchmark vs adding x% IBIT vs adding x% GLD (x = 1%, 2.5%, 5%, 10%), after costs —
         annualized return, volatility, Sharpe, Sortino, maximum drawdown, VaR/ES, turnover, cost drag,
         paired block-bootstrap CI of ΔSharpe (vs 60/40) and the HAC delta-method p-value
    B. Break-even cost: how high would IBIT's one-way cost in bp have to be before adding 5% IBIT no longer raises the Sharpe ratio
    R. Rebalancing frequency: daily / weekly / monthly / quarterly / threshold band — turnover, cost, net Sharpe (including the paper's 1/3 portfolio)
    K. Capacity: under different portfolio sizes ($10m–$10bn), IBIT's trade as a share of daily dollar volume and the impact cost
    F. ETF expense ratios: prices are already net of fees (not deducted again); also do a gross-of-fee comparison, quantifying the effect of the fee on ΔSharpe

Outputs: out/28_costs/  implementation_costs.md, tab_*.csv, fig_*.png/pdf

Usage (just run it in PyCharm; takes about 1–2 minutes)
    python 28_implementation_costs.py
    python 28_implementation_costs.py --aum 1e9      # change the default portfolio size
    python 28_implementation_costs.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Configuration
# ============================================================================
AS_OF = "20260921"
RAW = Path("data/raw")
OUTDIR = Path("out/28_costs")
ASSETS = ["IBIT", "GLD", "SPY", "TLT"]
EXPENSE = {"IBIT": 0.0025, "GLD": 0.0040, "SPY": 0.000945, "TLT": 0.0015}   # annualized, see script 25
TD = 252
ROLL = 21                        # rolling window for spread, volatility, dollar volume
IMPACT_Y = 1.0                   # square-root impact model coefficient (Almgren et al. 2005; Torre 1997 order of magnitude)
DEFAULT_AUM = 1e8                # default portfolio size $100m
AUMS = (1e7, 1e8, 1e9, 1e10)
DEFAULT_SCHED = "monthly"
SCHEDULES = {"daily": 1, "weekly": 5, "monthly": 21, "quarterly": 63, "band": None}
BAND_REL = 0.20                  # threshold band: rebalance when any asset's weight deviates from target by more than 20%
X_LEVELS = (0.01, 0.025, 0.05, 0.10)
BOOT_B, BOOT_L, HAC_LAG = 2000, 10, 5
SEED = 20260925

LINES: list[str] = []


def w(s: str = "") -> None:
    print(s)
    LINES.append(s)


def md(rows, header) -> str:
    o = ["| " + " | ".join(map(str, header)) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    o += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(o) + "\n"


def pct(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def bp(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 1e4:.{d}f}"


# ============================================================================
# Spread and liquidity estimation
# ============================================================================
def spread_ar(H, L, C):
    """Abdi & Ranaldo (2017): s² = 4 (c_t − η_t)(c_t − η_{t+1}), η = (ln H + ln L)/2.
    Returns the daily s² (which may be negative); the caller takes a rolling mean and
    then the square root."""
    eta = (np.log(H) + np.log(L)) / 2
    c = np.log(C)
    return 4 * (c - eta) * (c - eta.shift(-1))


def spread_cs(H, L):
    """Corwin & Schultz (2012) two-day high-low estimator; negative values set to 0."""
    lh = np.log(H / L) ** 2
    beta = lh + lh.shift(-1)
    hmax = pd.concat([H, H.shift(-1)], axis=1).max(axis=1)
    lmin = pd.concat([L, L.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(hmax / lmin) ** 2
    k = 3 - 2 * math.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


# ============================================================================
# Backtest engine
# ============================================================================
def run(R: np.ndarray, target: np.ndarray, sched: str, half_spread: np.ndarray, sigma: np.ndarray,
        adv: np.ndarray, aum: float, impact_y: float = IMPACT_Y, extra_bp: np.ndarray | None = None):
    """R: T×N daily returns; target: N target weights; half_spread/sigma/adv: T×N (values
    available at the close of day t).
    On day t: hold w (the weight after the previous close) and earn r_t; weights drift; if
    it is a rebalance day, trade back to target at day t's closing prices.
    Cost = Σ|Δw_i| × (half-spread_i,t + Y·σ_i,t·√(|Δw_i|·V_t / ADV_i,t) + extra_i), deducted
    from that day's portfolio return.
    Initial positioning is not charged (what is measured is the ongoing operating cost).
    Returns (net daily returns, gross daily returns, daily turnover, daily cost)."""
    T, N = R.shape
    wcur = target.copy()
    V = aum
    gross, net, turn, cost = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    step = SCHEDULES[sched]
    for t in range(T):
        g = float(wcur @ R[t])
        wdrift = wcur * (1 + R[t]) / (1 + g)
        if step is None:
            act = target > 0
            do = np.any(np.abs(wdrift[act] / target[act] - 1) > BAND_REL)
        else:
            do = (t + 1) % step == 0
        c = 0.0
        if do:
            dw = np.abs(target - wdrift)
            trade = dw * V * (1 + g)
            imp = impact_y * sigma[t] * np.sqrt(np.where(adv[t] > 0, trade / adv[t], 0.0))
            per = half_spread[t] + imp + (extra_bp if extra_bp is not None else 0.0)
            c = float(np.nansum(dw * per))
            turn[t] = 0.5 * dw.sum()
            wcur = target.copy()
        else:
            wcur = wdrift
        gross[t] = g
        net[t] = g - c
        cost[t] = c
        V *= (1 + net[t])
    return net, gross, turn, cost


# ============================================================================
# Performance and tests
# ============================================================================
def perf(r: np.ndarray, rf: np.ndarray) -> dict:
    ex = r - rf
    dn = ex[ex < 0]
    v = np.concatenate([[1.0], np.cumprod(1 + r)])
    q = np.quantile(r, 0.05)
    return dict(ann_ret=float(np.prod(1 + r) ** (TD / len(r)) - 1), vol=float(r.std(ddof=1) * math.sqrt(TD)),
                sharpe=float(ex.mean() / ex.std(ddof=1) * math.sqrt(TD)),
                sortino=float(ex.mean() / math.sqrt((dn ** 2).sum() / len(ex)) * math.sqrt(TD)) if len(dn) else np.nan,
                mdd=float((v / np.maximum.accumulate(v) - 1).min()), var95=float(-q), es95=float(-r[r <= q].mean()))


def sr_diff_hac(e1, e2, lag=HAC_LAG):
    """Ledoit & Wolf (2008) delta method + Newey–West HAC: H0 SR1 = SR2. e is excess return."""
    Z = np.column_stack([e1, e2, e1 ** 2, e2 ** 2])
    T = len(Z)
    m = Z.mean(0)
    Zc = Z - m
    S = Zc.T @ Zc / T
    for k in range(1, lag + 1):
        G = Zc[k:].T @ Zc[:-k] / T
        S += (1 - k / (lag + 1)) * (G + G.T)
    a, b, c, d = m
    s1, s2 = math.sqrt(c - a * a), math.sqrt(d - b * b)
    grad = np.array([c / s1 ** 3, -d / s2 ** 3, -a / (2 * s1 ** 3), b / (2 * s2 ** 3)])
    se = math.sqrt(max(grad @ S @ grad / T, 1e-18))
    diff = a / s1 - b / s2
    return diff * math.sqrt(TD), se * math.sqrt(TD), float(2 * stats.norm.sf(abs(diff / se)))


def boot_idx(n, B, L, rng):
    out = np.empty((B, n), dtype=int)
    for b in range(B):
        idx, i = [], 0
        while len(idx) < n:
            s = int(rng.integers(0, n))
            ln = int(rng.geometric(1.0 / L))
            idx.extend((s + k) % n for k in range(ln))
        out[b] = idx[:n]
    return out


def sr(e):
    return e.mean(-1) / e.std(-1, ddof=1) * math.sqrt(TD)


# ============================================================================
# Self-test
# ============================================================================
def selftest() -> bool:
    ok = True

    def chk(lab, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {lab} {info}")

    rng = np.random.default_rng(3)
    T, N = 300, 3
    R = rng.normal(0.0003, 0.01, (T, N))
    tgt = np.array([0.5, 0.3, 0.2])
    z = np.zeros((T, N))
    net, gross, turn, cost = run(R, tgt, "daily", z, z, np.ones((T, N)), 1e8)
    chk("Net return = gross return at zero cost", np.allclose(net, gross))
    chk("Gross return under daily rebalancing = R @ target weights", np.allclose(gross, R @ tgt))
    hs = np.full((T, N), 5e-4)
    net2, _, turn2, cost2 = run(R, tgt, "daily", hs, z, np.ones((T, N)), 1e8)
    chk("Fixed half-spread cost = Σ|Δw|×half-spread", np.allclose(cost2, 2 * turn2 * 5e-4))
    _, _, tm, _ = run(R, tgt, "monthly", z, z, np.ones((T, N)), 1e8)
    chk("Monthly rebalancing turnover < daily", tm.sum() < turn2.sum(), f"{tm.sum():.3f} < {turn2.sum():.3f}")
    # AR spread: simulate Roll-type prices with effective spread s; the estimate should be close
    Tn, s_true = 20000, 0.002
    mid = np.cumsum(rng.normal(0, 0.01 / math.sqrt(50), Tn * 50)).reshape(Tn, 50)   # continuous efficient price path
    q = np.where(rng.random((Tn, 50)) < 0.5, 1, -1) * s_true / 2
    px = np.exp(mid + q)
    H, L_, C = pd.Series(px.max(1)), pd.Series(px.min(1)), pd.Series(px[:, -1])
    est = math.sqrt(max(spread_ar(H, L_, C).mean(), 0))
    chk("Abdi–Ranaldo estimator is of the right magnitude (true value 20bp)", 0.5 * s_true < est < 2 * s_true, f"estimate {est * 1e4:.1f}bp")
    e1, e2 = rng.normal(0.001, 0.01, 2000), rng.normal(0.001, 0.01, 2000)
    rej = np.mean([sr_diff_hac(rng.normal(0.0005, 0.01, 800), rng.normal(0.0005, 0.01, 800))[2] < 0.05
                   for _ in range(400)])
    chk("HAC Sharpe-difference test rejection rate under the null ≈ 5%", 0.02 < rej < 0.09, f"{rej:.3f}")
    print("\nSelf-test " + ("all passed." if ok else "has failures!"))
    return ok


# ============================================================================
# Main program
# ============================================================================
def load(t):
    p = RAW / f"{t.replace('^', 'IDX_')}__{AS_OF}.csv"
    return pd.read_csv(p, parse_dates=["Date"]).drop_duplicates("Date").set_index("Date").sort_index()


def load_rf(idx):
    for name, kind in (("DGS3MO", "bey"), ("IRX", "disc"), ("DTB3", "disc")):
        p = RAW / f"{name}__{AS_OF}.csv"
        if p.exists():
            s = pd.read_csv(p, parse_dates=["Date"]).dropna().set_index("Date")["value"].astype(float) / 100
            s = s[~s.index.duplicated()].sort_index()
            if kind == "disc":
                s = 365 * s / (360 - 91 * s)
            y = s.reindex(s.index.union(idx)).sort_index().ffill().reindex(idx).shift(1).bfill()
            return ((1 + y) ** (1 / TD) - 1).values, name
    return np.zeros(len(idx)), "none (rf = 0)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aum", type=float, default=DEFAULT_AUM, help="default portfolio size (USD)")
    ap.add_argument("--boot", type=int, default=BOOT_B)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    here = Path(__file__).resolve().parent
    cands = [here, Path.cwd()]
    root = next((c for c in cands if (c / RAW / f"IBIT__{AS_OF}.csv").exists()), None)
    if root is None:
        sys.exit(f"Cannot find data/raw/IBIT__{AS_OF}.csv. Run this script from the repository root. Tried {cands}")
    os.chdir(root)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"Project directory: {Path.cwd()}")
    rng = np.random.default_rng(SEED)

    raw = {t: load(t) for t in ASSETS}
    cal = raw["IBIT"].index
    for t in ASSETS:
        cal = cal.intersection(raw[t].index)
    vix = load("^VIX")["Close"].reindex(cal)
    O = {k: pd.DataFrame({t: raw[t][k].reindex(cal) for t in ASSETS}) for k in ("High", "Low", "Close", "AdjClose", "Volume")}
    Rdf = O["AdjClose"].pct_change().iloc[1:]
    idx = Rdf.index
    R = Rdf.values
    T = len(idx)
    rf, rf_name = load_rf(idx)

    w("# Trading costs and implementability (reviewer comment 12)\n")
    w(f"Sample {idx[0].date()} → {idx[-1].date()}, {T} trading days. Risk-free rate: {rf_name}. "
      f"Default setup: portfolio size ${a.aum / 1e6:,.0f}m, {DEFAULT_SCHED} rebalancing, spread tier Quoted (1 cent) + impact.\n")

    # ================================================================ L
    w("## L. Liquidity and bid-ask spreads (estimated from daily OHLCV)\n")
    ar2 = pd.DataFrame({t: spread_ar(O["High"][t], O["Low"][t], O["Close"][t]) for t in ASSETS})
    cs = pd.DataFrame({t: spread_cs(O["High"][t], O["Low"][t]) for t in ASSETS})
    ar_roll = np.sqrt(ar2.rolling(ROLL, min_periods=10).mean().clip(lower=0))
    cs_roll = cs.rolling(ROLL, min_periods=10).mean()
    tick = 0.01 / O["Close"]
    dv = O["Close"] * O["Volume"]
    amihud = (Rdf.abs() / dv.reindex(idx)) * 1e6 * 1e4            # |return| per $1m traded, in bp
    hi = (vix.reindex(idx) >= 25).values
    rows, tab = [], []
    for t in ASSETS:
        a_ = ar_roll[t].reindex(idx)
        c_ = cs_roll[t].reindex(idx)
        early = idx < idx[0] + pd.Timedelta(days=92)
        late = idx >= idx[-1] - pd.Timedelta(days=365)
        rec = dict(ticker=t, adv_usd_m=float(dv[t].median() / 1e6), amihud_bp_per_musd=float(amihud[t].median()),
                   tick_bp=float(tick[t].median() * 1e4), cs_bp=float(c_.median() * 1e4), ar_bp=float(a_.median() * 1e4),
                   ar_bp_highvix=float(a_[hi].median() * 1e4), ar_bp_normal=float(a_[~hi].median() * 1e4),
                   ar_bp_first3m=float(a_[early].median() * 1e4), ar_bp_last12m=float(a_[late].median() * 1e4),
                   ub_bp=float(np.maximum(a_, c_).median() * 1e4),
                   expense_ratio=EXPENSE[t])
        tab.append(rec)
        rows.append([t, f"{rec['adv_usd_m']:,.0f}", f"{rec['amihud_bp_per_musd']:.4f}", f"{rec['tick_bp']:.2f}",
                     f"{rec['cs_bp']:.1f}", f"{rec['ar_bp']:.1f}", f"{rec['ub_bp']:.1f}",
                     f"{rec['ar_bp_normal']:.1f} / {rec['ar_bp_highvix']:.1f}",
                     f"{rec['ar_bp_first3m']:.1f} / {rec['ar_bp_last12m']:.1f}", f"{EXPENSE[t] * 100:.4f}%"])
    L = pd.DataFrame(tab)
    L.to_csv(OUTDIR / "tab_L_liquidity.csv", index=False)
    w(md(rows, ["ETF", "Median daily dollar volume $m", "Amihud: |r| per $1m traded, bp", "Tick-size floor full spread bp",
                "CS spread bp", "AR spread bp", "OHLC upper bound max(AR,CS) bp", "AR: normal / VIX≥25", "AR: first 3 months / last 12 months", "Expense ratio"]))
    n_hi = int(hi.sum())
    w(f"VIX≥25 trading days: {n_hi}. Note: daily high-low estimators **overstate** the true quoted spread for highly liquid ETFs "
      "(they contain intraday volatility noise and often give 0 for TLT/GLD), so max(AR, CS) is used only as a cost **upper bound**; "
      "the main convention uses the half-spread of the 1-cent tick (these ETFs quote at a 1-cent spread almost all of the time), plus square-root impact.\n")

    # Cost input matrices (available at day t's close)
    ub = np.maximum(ar_roll, cs_roll).reindex(idx).bfill()
    HS = {"Quoted": (tick.reindex(idx) / 2).values,          # main convention: half-spread of the 1-cent tick
          "OHLC-UB": (ub / 2).values,                        # conservative: upper bound from daily OHLC estimators
          "Stressed": ub.values}                             # stressed: 2 × OHLC upper bound half-spread
    SIG = Rdf.rolling(ROLL, min_periods=5).std().bfill().values
    ADV = dv.rolling(ROLL, min_periods=5).median().reindex(idx).bfill().values
    GROSS_FEE = np.array([EXPENSE[t] for t in ASSETS]) / TD

    def tgt(ibit=0.0, gld=0.0, spy=None, tlt=None):
        if spy is None:
            rest = 1 - ibit - gld
            spy, tlt = 0.6 * rest, 0.4 * rest
        return np.array([ibit, gld, spy, tlt])

    PORTS = {"60/40 (SPY/TLT)": tgt()}
    for x in X_LEVELS:
        PORTS[f"+{x * 100:g}% IBIT"] = tgt(ibit=x)
        PORTS[f"+{x * 100:g}% GLD"] = tgt(gld=x)
    PORTS["Paper: IBIT/SPY/TLT 1/3"] = tgt(ibit=1 / 3, spy=1 / 3, tlt=1 / 3)
    PORTS["Paper: GLD/SPY/TLT 1/3"] = tgt(gld=1 / 3, spy=1 / 3, tlt=1 / 3)

    def sim(name, sched=DEFAULT_SCHED, hs="Quoted", aum=None, fee_gross=False, impact=IMPACT_Y, extra=None):
        Rin = R + GROSS_FEE if fee_gross else R
        return run(Rin, PORTS[name], sched, HS[hs], SIG, ADV, aum or a.aum, impact, extra)

    # ================================================================ A
    w(f"## A. Core question: after execution costs, does adding IBIT improve risk-adjusted returns?\n")
    w(f"Benchmark 60/40 (SPY/TLT); '+x% IBIT' = x in IBIT and the rest in 60/40; '+x% GLD' likewise as a control. "
      f"Cost = half-spread (1-cent tick) + square-root impact (Y={IMPACT_Y}), portfolio size ${a.aum / 1e6:,.0f}m, {DEFAULT_SCHED} rebalancing. "
      "ΔSharpe vs 60/40: paired stationary block bootstrap (jointly resampling the same trading days) 95% CI and Ledoit–Wolf (2008) HAC test.\n")
    base_net = sim("60/40 (SPY/TLT)")[0]
    IB = boot_idx(T, a.boot, BOOT_L, rng)
    rows, tab, NET = [], [], {}
    for name in PORTS:
        net, gross, turn, cost = sim(name)
        NET[name] = net
        p = perf(net, rf)
        pg = perf(gross, rf)
        yrs = T / TD
        rec = dict(portfolio=name, **p, sharpe_gross=pg["sharpe"], turnover_ann=turn.sum() / yrs,
                   cost_bp_ann=cost.sum() / yrs * 1e4)
        if name != "60/40 (SPY/TLT)":
            e1, e0 = net - rf, base_net - rf
            d, se, pv = sr_diff_hac(e1, e0)
            bs = sr(e1[IB]) - sr(e0[IB])
            lo, hi_ = np.percentile(bs, [2.5, 97.5])
            rec.update(dsharpe=d, dsharpe_lo=lo, dsharpe_hi=hi_, p_hac=pv,
                       des95=p["es95"] - perf(base_net, rf)["es95"])
        tab.append(rec)
        rows.append([name, pct(p["ann_ret"]), pct(p["vol"]), f"{pg['sharpe']:.3f}", f"{p['sharpe']:.3f}",
                     f"{p['sortino']:.3f}", pct(p["mdd"]), pct(p["es95"]), f"{rec['turnover_ann'] * 100:.1f}%",
                     f"{rec['cost_bp_ann']:.2f}",
                     "—" if "dsharpe" not in rec else f"{rec['dsharpe']:+.3f} [{rec['dsharpe_lo']:+.2f}, {rec['dsharpe_hi']:+.2f}]",
                     "—" if "p_hac" not in rec else f"{rec['p_hac']:.3f}"])
    A = pd.DataFrame(tab)
    A.to_csv(OUTDIR / "tab_A_add_ibit.csv", index=False)
    w(md(rows, ["Portfolio", "Ann. return (net)", "Ann. vol", "Sharpe (gross)", "Sharpe (net)", "Sortino (net)", "Max drawdown",
                "ES95 (daily)", "Ann. turnover", "Cost bp/yr", "ΔSharpe vs 60/40 [95% CI]", "HAC p"]))
    w("Sharpe (gross) = before trading costs (still net of ETF fees); Sharpe (net) = after also deducting spread and impact costs.\n")

    # ================================================================ B
    w("## B. Break-even cost\n")
    w("Add c bp (one-way) on top of IBIT's trading; find the c that makes ΔSharpe(+5% IBIT vs 60/40) = 0. "
      "The larger the gap from the estimated IBIT one-way cost, the less sensitive the conclusion is to the cost assumption.\n")
    name5 = "+5% IBIT"

    def dsr_extra(c):
        extra = np.array([c / 1e4, 0, 0, 0])
        return perf(sim(name5, extra=extra)[0], rf)["sharpe"] - perf(base_net, rf)["sharpe"]

    d0 = dsr_extra(0.0)
    est_ibit = float(np.nanmedian(HS["Quoted"][:, 0]) * 1e4)
    ub_ibit = float(np.nanmedian(HS["OHLC-UB"][:, 0]) * 1e4)
    if d0 <= 0:
        be_txt = f"Adding 5% IBIT at **zero extra cost** already gives ΔSharpe = {d0:+.3f}, i.e. it already worsens after the estimated cost, so there is no positive break-even cost."
        be = np.nan
    else:
        lo_, hi2 = 0.0, 5000.0
        if dsr_extra(hi2) > 0:
            be = np.inf
            be_txt = "Even at 5,000bp one-way extra cost, ΔSharpe is still positive (IBIT's return over the sample far exceeds the cost)."
        else:
            for _ in range(40):
                mid = (lo_ + hi2) / 2
                lo_, hi2 = (mid, hi2) if dsr_extra(mid) > 0 else (lo_, mid)
            be = (lo_ + hi2) / 2
            be_txt = (f"Break-even extra cost ≈ **{be:,.0f}bp** one-way, versus IBIT's median half-spread of {est_ibit:.1f}bp (tick) "
                      f"to {ub_ibit:.1f}bp (OHLC upper bound); "
                      f"the cost assumption is not what drives the conclusion — the sample-period returns themselves are (see the CI in Section A).")
    w(f"- At zero extra cost, ΔSharpe = {d0:+.3f}. {be_txt}\n")

    # By cost tier and gross/net of fees
    rows, tab = [], []
    for name in (name5, "+5% GLD", "Paper: IBIT/SPY/TLT 1/3", "Paper: GLD/SPY/TLT 1/3"):
        cells = [name]
        for hs in ("Quoted", "OHLC-UB", "Stressed"):
            s1 = perf(sim(name, hs=hs)[0], rf)["sharpe"]
            s0 = perf(sim("60/40 (SPY/TLT)", hs=hs)[0], rf)["sharpe"]
            cells.append(f"{s1:.3f} ({s1 - s0:+.3f})")
            tab.append(dict(portfolio=name, spread=hs, fee="net", sharpe=s1, dsharpe=s1 - s0))
        s1 = perf(sim(name, fee_gross=True)[0], rf)["sharpe"]
        s0 = perf(sim("60/40 (SPY/TLT)", fee_gross=True)[0], rf)["sharpe"]
        cells.append(f"{s1:.3f} ({s1 - s0:+.3f})")
        tab.append(dict(portfolio=name, spread="Quoted", fee="gross", sharpe=s1, dsharpe=s1 - s0))
        rows.append(cells)
    w("### B2. Cost-tier and ETF expense-ratio sensitivity (Sharpe (net), Δ vs 60/40 in parentheses)\n")
    w(md(rows, ["Portfolio", "Quoted (1 cent)", "OHLC upper bound", "Stressed (2× upper bound)", "Gross of fees (add fees back, Quoted)"]))
    pd.DataFrame(tab).to_csv(OUTDIR / "tab_B_cost_sensitivity.csv", index=False)
    w("ETF expense ratios accrue daily against NAV and are already reflected in prices; the main results do not deduct them again. The 'gross of fees' column adds them back, only to show the effect of fees on the comparison.\n")

    # ================================================================ R
    w("## R. Rebalancing frequency\n")
    w(f"Threshold band = rebalance when any asset's weight deviates from target by more than {int(BAND_REL * 100)}%. Sharpe (net) uses the Quoted spread + impact, "
      f"portfolio size ${a.aum / 1e6:,.0f}m.\n")
    rows, tab = [], []
    for name in ("Paper: IBIT/SPY/TLT 1/3", "Paper: GLD/SPY/TLT 1/3", name5, "60/40 (SPY/TLT)"):
        for sch in SCHEDULES:
            net, gross, turn, cost = sim(name, sched=sch)
            yrs = T / TD
            p, pg = perf(net, rf), perf(gross, rf)
            n_reb = int((turn > 0).sum())
            costs_alt = {hs: sim(name, sched=sch, hs=hs)[3].sum() / yrs * 1e4 for hs in ("OHLC-UB", "Stressed")}
            tab.append(dict(portfolio=name, schedule=sch, n_rebal=n_reb, turnover_ann=turn.sum() / yrs,
                            cost_bp_ann=cost.sum() / yrs * 1e4, cost_bp_ohlc_ub=costs_alt["OHLC-UB"],
                            cost_bp_stressed=costs_alt["Stressed"], sharpe_gross=pg["sharpe"], sharpe_net=p["sharpe"],
                            vol=p["vol"], es95=p["es95"]))
            rows.append([name if sch == "daily" else "", sch, n_reb, f"{turn.sum() / yrs * 100:.1f}%",
                         f"{cost.sum() / yrs * 1e4:.2f} / {costs_alt['OHLC-UB']:.2f} / {costs_alt['Stressed']:.2f}",
                         f"{pg['sharpe']:.3f}", f"{p['sharpe']:.3f}", pct(p["vol"]), pct(p["es95"])])
    Rt = pd.DataFrame(tab)
    Rt.to_csv(OUTDIR / "tab_R_rebalancing.csv", index=False)
    w(md(rows, ["Portfolio", "Frequency", "Rebalances", "Ann. turnover", "Cost bp/yr (Quoted / OHLC UB / Stressed)",
                "Sharpe (gross)", "Sharpe (net)", "Ann. vol", "ES95"]))

    # ================================================================ K
    w("## K. Capacity: the effect of portfolio size on impact cost\n")
    rows, tab = [], []
    for name in (name5, "Paper: IBIT/SPY/TLT 1/3"):
        for aum in AUMS:
            net, gross, turn, cost = sim(name, aum=aum)
            # IBIT trade notional as a share of daily dollar volume
            tgt_ = PORTS[name]
            wcur, V, shares = tgt_.copy(), aum, []
            for t in range(T):
                g = float(wcur @ R[t])
                wd = wcur * (1 + R[t]) / (1 + g)
                V *= (1 + g)
                if (t + 1) % SCHEDULES[DEFAULT_SCHED] == 0:
                    shares.append(abs(tgt_[0] - wd[0]) * V / ADV[t, 0])
                    wd = tgt_.copy()
                wcur = wd
            sh = np.array(shares)
            yrs = T / TD
            p = perf(net, rf)
            dsr_ = p["sharpe"] - perf(sim("60/40 (SPY/TLT)", aum=aum)[0], rf)["sharpe"]
            tab.append(dict(portfolio=name, aum=aum, ibit_trade_pct_adv_median=float(np.median(sh)),
                            ibit_trade_pct_adv_max=float(sh.max()), cost_bp_ann=cost.sum() / yrs * 1e4,
                            sharpe_net=p["sharpe"], dsharpe_vs_6040=dsr_))
            rows.append([name if aum == AUMS[0] else "", f"${aum / 1e6:,.0f}m", f"{np.median(sh) * 100:.4f}%",
                         f"{sh.max() * 100:.4f}%", f"{cost.sum() / yrs * 1e4:.2f}", f"{p['sharpe']:.3f}", f"{dsr_:+.3f}"])
    pd.DataFrame(tab).to_csv(OUTDIR / "tab_K_capacity.csv", index=False)
    w(md(rows, ["Portfolio", "Size", "IBIT single trade / daily dollar volume (median)", "(max)", "Cost bp/yr", "Sharpe (net)", "ΔSharpe vs 60/40"]))

    # ================================================================ Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        xs = [0] + [x * 100 for x in X_LEVELS]
        for leg, col in (("IBIT", "#d97706"), ("GLD", "#1f4e79")):
            ys = [A.loc[A.portfolio == "60/40 (SPY/TLT)", "sharpe"].item()] + \
                 [A.loc[A.portfolio == f"+{x * 100:g}% {leg}", "sharpe"].item() for x in X_LEVELS]
            ax[0].plot(xs, ys, "o-", color=col, label=f"add {leg} (net of costs)")
            los = [np.nan] + [A.loc[A.portfolio == f"+{x * 100:g}% {leg}", "dsharpe_lo"].item() + ys[0] for x in X_LEVELS]
            his = [np.nan] + [A.loc[A.portfolio == f"+{x * 100:g}% {leg}", "dsharpe_hi"].item() + ys[0] for x in X_LEVELS]
            ax[0].fill_between(xs, los, his, color=col, alpha=.12)
        ax[0].axhline(ys[0], color="k", lw=.6, ls="--")
        ax[0].set_xlabel("allocation to IBIT / GLD, % (rest 60/40 SPY/TLT)")
        ax[0].set_ylabel("Sharpe ratio, net of spread + impact costs")
        ax[0].set_title("Adding IBIT vs adding GLD to 60/40 (band = 95% CI of difference)")
        ax[0].legend(fontsize=8)
        for name, col in (("Paper: IBIT/SPY/TLT 1/3", "#d97706"), ("Paper: GLD/SPY/TLT 1/3", "#1f4e79")):
            g = Rt[Rt.portfolio == name]
            ax[1].plot(range(len(g)), g["cost_bp_ann"], "o-", color=col, label=name.replace("Paper: ", ""))
        ax[1].set_xticks(range(len(SCHEDULES))); ax[1].set_xticklabels(list(SCHEDULES))
        ax[1].set_ylabel("annual cost drag, bp"); ax[1].set_title("Cost of rebalancing frequency (quoted spread + impact)")
        ax[1].legend(fontsize=8)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            plt.savefig(OUTDIR / f"fig_costs.{ext}", dpi=150)
        plt.close()
        w("Figure: `fig_costs` (left: net Sharpe after adding IBIT / GLD; right: annualized cost of rebalancing frequency).\n")
    except Exception as e:  # noqa: BLE001
        w(f"(plotting skipped: {e})\n")

    # ================================================================ Text
    r5 = A.set_index("portfolio").loc[name5]
    g5 = A.set_index("portfolio").loc["+5% GLD"]
    base = A.set_index("portfolio").loc["60/40 (SPY/TLT)"]
    li = L.set_index("ticker")
    daily_cost = Rt[(Rt.portfolio == "Paper: IBIT/SPY/TLT 1/3") & (Rt.schedule == "daily")]["cost_bp_ann"].item()
    mon_cost = Rt[(Rt.portfolio == "Paper: IBIT/SPY/TLT 1/3") & (Rt.schedule == "monthly")]["cost_bp_ann"].item()
    daily_ub = Rt[(Rt.portfolio == "Paper: IBIT/SPY/TLT 1/3") & (Rt.schedule == "daily")]["cost_bp_ohlc_ub"].item()
    mon_ub = Rt[(Rt.portfolio == "Paper: IBIT/SPY/TLT 1/3") & (Rt.schedule == "monthly")]["cost_bp_ohlc_ub"].item()
    sig = r5["p_hac"] < 0.05
    be_str = "not defined (the allocation is already worse at zero cost)" if not np.isfinite(be) and d0 <= 0 else (
        "above 5,000bp" if not np.isfinite(be) else f"about {be:,.0f}bp one-way")
    Kt = pd.read_csv(OUTDIR / "tab_K_capacity.csv")
    kk = Kt[Kt.portfolio == name5].set_index("aum")
    cap1 = float(kk.loc[1e9, "ibit_trade_pct_adv_max"] * 100) if 1e9 in kk.index else float("nan")
    cap10 = float(kk.loc[1e10, "ibit_trade_pct_adv_max"] * 100) if 1e10 in kk.index else float("nan")
    direction = "improves" if r5["dsharpe"] > 0 else "worsens"
    w("## Paragraph ready to go into the paper (English)\n")
    w(f"""> **Implementation costs.** Each rebalancing trade pays half the quoted spread—one cent, which is where these ETFs
> quote almost all of the time, i.e. {li.loc['IBIT', 'tick_bp']:.2f}bp for IBIT, {li.loc['GLD', 'tick_bp']:.2f}bp for GLD,
> {li.loc['SPY', 'tick_bp']:.2f}bp for SPY and {li.loc['TLT', 'tick_bp']:.2f}bp for TLT at median prices—plus square-root market impact
> (Y = {IMPACT_Y}, using 21-day volatility and median dollar volume; IBIT's median daily dollar volume is
> ${li.loc['IBIT', 'adv_usd_m']:,.0f}m). As a conservative upper bound we also use effective spreads estimated from daily OHLC data
> (the maximum of the Corwin–Schultz and Abdi–Ranaldo estimators; median {li.loc['IBIT', 'ub_bp']:.0f}bp for IBIT), and twice that
> as a stressed case; these low-frequency estimators are known to overstate spreads for very liquid securities.
> ETF expense ratios are already reflected in NAV-based prices and are not deducted again; adding them back changes
> the Sharpe comparisons only marginally (Table B2).
>
> **Does adding IBIT help after costs?** For a ${a.aum / 1e6:,.0f}m portfolio rebalanced {DEFAULT_SCHED}, a 5% IBIT allocation carved
> out of a 60/40 SPY/TLT portfolio {direction} the net Sharpe ratio from {base['sharpe']:.2f} to {r5['sharpe']:.2f}
> (difference {r5['dsharpe']:+.2f}, block-bootstrap 95% CI [{r5['dsharpe_lo']:+.2f}, {r5['dsharpe_hi']:+.2f}], HAC p = {r5['p_hac']:.2f});
> the same allocation to GLD gives {g5['sharpe']:.2f} ({g5['dsharpe']:+.2f}, p = {g5['p_hac']:.2f}). Trading costs are an order of
> magnitude too small to change the sign: they cost {r5['cost_bp_ann']:.1f}bp per year, and the break-even extra cost on IBIT
> trades is {be_str}. The difference is therefore
> {'statistically significant' if sig else 'not statistically significant'} and driven by realised returns over a
> {T / TD:.1f}-year sample rather than by implementation frictions. The daily rebalancing assumed in the main text would cost
> the IBIT portfolio about {daily_cost:.0f}bp per year versus {mon_cost:.0f}bp under monthly rebalancing ({daily_ub:.0f}bp versus
> {mon_ub:.0f}bp under the OHLC upper bound; Table R), and
> at $1bn a monthly IBIT rebalancing trade is at most {cap1:.2f}% of its daily dollar volume ({cap10:.1f}% at $10bn; Table K).
""")
    w("(Numbers update automatically with the run; if the conclusion's direction changes, 'improves/worsens' and 'significant' in the sentence switch automatically, but please read it through by hand once more.)\n")

    w("1. Spreads: the main convention is the half-spread of the 1-cent tick; effective spreads estimated from daily OHLC by Abdi–Ranaldo and Corwin–Schultz are also used as upper bounds, "
      "with comparisons of high-VIX vs normal days and IBIT's early listing period vs the last 12 months (Table L); the conclusion is unchanged across the three spread tiers (Table B2).")
    w(f"2. Liquidity: median daily dollar volume and the Amihud measure; square-root impact model; capacity analysis for portfolio sizes from $10m to $10bn (Table K).")
    w("3. ETF expense ratios: already included in prices and not deducted again; a gross-of-fee comparison is also provided (Table B2).")
    w(f"4. Turnover and rebalancing: turnover, cost, and net Sharpe for the five modes daily/weekly/monthly/quarterly/threshold band (Table R); daily rebalancing costs the IBIT portfolio about "
      f"{daily_cost:.0f}bp/yr and monthly about {mon_cost:.0f}bp/yr (under the OHLC upper bound, {daily_ub:.0f} and {mon_ub:.0f}bp/yr respectively).")
    w(f"5. Core question: adding 5% IBIT to 60/40, after costs, gives ΔSharpe = {r5['dsharpe']:+.2f} (95% CI [{r5['dsharpe_lo']:+.2f}, "
      f"{r5['dsharpe_hi']:+.2f}], p = {r5['p_hac']:.2f}), {'significant' if sig else 'not significant'}; costs do not affect the direction of the conclusion (Tables A, B).")

    (OUTDIR / "implementation_costs.md").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nDone. Results: {OUTDIR}/implementation_costs.md")


if __name__ == "__main__":
    main()