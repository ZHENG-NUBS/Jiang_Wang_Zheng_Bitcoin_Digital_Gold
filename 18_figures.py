#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
18_figures.py — Redraw Figures 1–3 of the paper (for the revised manuscript, based on the corrected data)

Figure 1  Cumulative value of the two equal-weight portfolios and SPY; VIX>25 shaded
Figure 2  Full-sample GARCH(1,1)-t conditional volatility (annualized); VIX>25 shaded
          The original figure plotted VIX on a secondary right-hand axis. A dual axis
          invites readers to compare two incommensurable scales, so high-VIX periods
          are now marked by shading only.
Figure 3  Strictly out-of-sample one-step-ahead 95% VaR: GARCH(1,1)-t and 250-day
          historical simulation, one panel per portfolio; crosses mark GARCH-t exceptions.
          The original figure used a full-sample fit (with look-ahead bias) and cannot
          be called a backtest.

Dependencies: numpy, pandas, matplotlib, arch (via roll_garch in script 04)
Usage:
    python 18_figures.py --data data/processed/returns__20260921.csv --outdir out/18_fig
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))

# Colours checked for colour-vision-deficiency separation and contrast on a white background
C_IBIT = "#2a78d6"
C_GLD = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
SHADE = "#d9d8d4"
GRID = "#e6e5e1"

W_IN = 5.36          # matches the text width of the .docx, so font sizes equal printed sizes


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "savefig.dpi": 300,
        "figure.dpi": 150,
    })


def shade_high_vix(ax, dates, vix, thr=25.0):
    hi = (vix > thr).values
    start = None
    for i, h in enumerate(hi):
        if h and start is None:
            start = i
        if (not h or i == len(hi) - 1) and start is not None:
            end = i if h else i - 1
            ax.axvspan(dates[start] - pd.Timedelta(hours=12),
                       dates[end] + pd.Timedelta(hours=12),
                       color=SHADE, lw=0, zorder=0)
            start = None


def date_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="out/18_fig")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    style()

    p09 = _load("p09", "09_regenerate_paper_numbers.py")
    p04 = _load("p04", "04_var_es_backtest.py")

    d = pd.read_csv(args.data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    dates = d["Date"]
    P = {
        "IBIT": (d.IBIT + d.SPY + d.TLT) / 3,
        "GLD": (d.GLD + d.SPY + d.TLT) / 3,
    }
    print(f"{len(d)} trading days  {dates.iloc[0].date()} → {dates.iloc[-1].date()}")

    # ------------------------------------------------------------ Figure 1
    fig, ax = plt.subplots(figsize=(W_IN, 2.75))
    shade_high_vix(ax, dates, d.VIX_Close)
    for k, c, lab in (("IBIT", C_IBIT, "IBIT portfolio (1/3 IBIT, SPY, TLT)"),
                      ("GLD", C_GLD, "GLD portfolio (1/3 GLD, SPY, TLT)")):
        nav = (1 + P[k]).cumprod()
        ax.plot(dates, nav, color=c, lw=1.4, label=lab)
        print(f"  Fig1 {k}: final value {nav.iloc[-1]:.3f}, peak {nav.max():.3f} "
              f"({dates[nav.idxmax()].date()})")
    ax.plot(dates, (1 + d.SPY).cumprod(), color=INK2, lw=0.9, ls=(0, (3, 2)), label="SPY")
    ax.fill_between([], [], color=SHADE, label="High volatility (VIX > 25)")
    ax.set_ylabel("Cumulative return (start = 1)")
    date_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=2,
              borderaxespad=0.2)
    ax.set_xlim(dates.iloc[0] - pd.Timedelta(days=5), dates.iloc[-1] + pd.Timedelta(days=5))
    fig.tight_layout(pad=0.3)
    f1 = os.path.join(args.outdir, "fig1_cumulative.png")
    fig.savefig(f1); fig.savefig(f1.replace(".png", ".pdf")); plt.close(fig)

    # ------------------------------------------------------------ Figure 2
    fig, ax = plt.subplots(figsize=(W_IN, 2.2))
    shade_high_vix(ax, dates, d.VIX_Close)
    for k, c, lab in (("IBIT", C_IBIT, "IBIT portfolio"), ("GLD", C_GLD, "GLD portfolio")):
        g = p09.fit_garch11_t(P[k].values)
        vol = np.sqrt(g["sigma2"] * 252) * 100
        i = int(np.argmax(vol))
        print(f"  Fig2 {k}: α={g['alpha']:.3f} β={g['beta']:.3f} ν={g['nu']:.2f}  "
              f"peak {vol[i]:.1f}% ({dates[i].date()})  "
              f"next day after sample end {math.sqrt(g['sigma2_next']*252)*100:.1f}%  "
              f"mean {vol.mean():.1f}%  min {vol.min():.1f}%")
        ax.plot(dates, vol, color=c, lw=1.1, label=lab)
    ax.fill_between([], [], color=SHADE, label="High volatility (VIX > 25)")
    ax.set_ylabel("Annualized conditional\nvolatility (%)")
    date_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=3,
              borderaxespad=0.2)
    ax.set_xlim(dates.iloc[0] - pd.Timedelta(days=5), dates.iloc[-1] + pd.Timedelta(days=5))
    ax.set_ylim(bottom=0)
    fig.tight_layout(pad=0.3)
    f2 = os.path.join(args.outdir, "fig2_condvol.png")
    fig.savefig(f2); fig.savefig(f2.replace(".png", ".pdf")); plt.close(fig)

    # ------------------------------------------------------------ Figure 3
    n_start, refit, a = p04.N_START, p04.REFIT, 0.05
    fig, axes = plt.subplots(2, 1, figsize=(W_IN, 4.0), sharex=True)
    for ax, (k, c, lab) in zip(axes, (("IBIT", C_IBIT, "IBIT portfolio"),
                                      ("GLD", C_GLD, "GLD portfolio"))):
        r = P[k].values
        fc = p04.roll_garch(r, "t", n_start, refit, p04.WINDOW)
        v_g, _ = p04.forecast_var_es(fc, "t", a)
        v_h, _ = p04.roll_hs(r, n_start, a, p04.HS_WIN)
        dd = dates.iloc[n_start:].reset_index(drop=True)
        ro = r[n_start:]
        br = ro < v_g
        brh = ro < v_h
        print(f"  Fig3 {k}: out-of-sample {len(ro)} days {dd.iloc[0].date()} → {dd.iloc[-1].date()}  "
              f"GARCH-t exceptions {br.sum()} ({br.mean()*100:.2f}%)  HS exceptions {brh.sum()} "
              f"({brh.mean()*100:.2f}%)  refitted {fc['refits']} times")
        ax.plot(dd, ro * 100, color="#9d9b95", lw=0.5, label="Realized return")
        ax.plot(dd, v_g * 100, color=c, lw=1.2, label="GARCH(1,1)-t VaR95")
        ax.plot(dd, v_h * 100, color=INK2, lw=1.0, ls=(0, (3, 2)), label="Historical simulation VaR95")
        ax.scatter(dd[br], ro[br] * 100, marker="x", s=14, lw=0.9, color=INK,
                   label=f"GARCH-t exception ({br.sum()}/{len(ro)})", zorder=5)
        ax.set_title(lab, loc="left", fontsize=8, color=INK, pad=14)
        ax.set_ylabel("Daily return (%)")
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=2,
                  fontsize=6.5, borderaxespad=0.1, handlelength=1.8, columnspacing=1.0)
        lo = min(np.nanmin(ro), np.nanmin(v_g)) * 100
        ax.set_ylim(lo * 1.15, max(np.nanmax(ro) * 100 * 1.1, 1))
    date_axis(axes[-1])
    axes[-1].set_xlim(dd.iloc[0] - pd.Timedelta(days=4), dd.iloc[-1] + pd.Timedelta(days=4))
    fig.tight_layout(pad=0.3, h_pad=0.4)
    f3 = os.path.join(args.outdir, "fig3_oos_var.png")
    fig.savefig(f3); fig.savefig(f3.replace(".png", ".pdf")); plt.close(fig)
    print("Wrote", f1, f2, f3)


if __name__ == "__main__":
    main()
