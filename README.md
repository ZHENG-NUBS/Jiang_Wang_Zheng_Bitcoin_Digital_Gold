# Is Bitcoin "Digital Gold"? — replication code

Python and R code for

> Jiang, B., Wang, J. and Zheng, Z. *Is Bitcoin "Digital Gold"? An Empirical Study Based on
> Multi-Model VaR/ES Measures, Stress Testing, and Portfolio Optimization.* Revised manuscript
> submitted to the *Journal of Risk and Financial Management*, October 2026.

The scripts download the data, build the return series and produce every table and figure in the
paper. The main sample is 607 daily returns of IBIT, GLD, TLT and SPY (12 January 2024 – 15 June 2026).
Supplementary samples cover Bitcoin spot prices from September 2014 and further Bitcoin, gold, equity
and bond instruments.

## Contents

| Path | What it is |
|---|---|
| `01a`–`01e_*.py` | Data download (with `25`, the only scripts that go online) |
| `02_build_returns.py` | Builds the analysis file `data/processed/returns__20260921.csv` from the snapshot |
| `03_hedge_safehaven.R`, `05_stress_tables.R` | Analysis scripts in R: safe-haven regressions and stress windows |
| `04`–`28_*.py` | Analysis scripts in Python |
| `data/raw/manifest*.json` | SHA-256 checksums and download metadata of the authors' snapshots |
| `data/raw/checksums__20260921L_X.sha256` | SHA-256 checksums of the long-sample (`L`) and ETF-appendix (`X`) snapshot files |
| `data/checksums_processed.sha256` | SHA-256 checksum of the analysis file used in the paper |
| `requirements.txt` | Package versions the results were produced with |

Every analysis script writes to its own folder under `out/`. Script numbers follow the order in which
the analyses were written, not the order of the paper.

## Requirements

Python 3.11 or later, and R 4.3 or later with the packages `sandwich` and `lmtest` (for scripts 03
and 05).

```bash
python -m pip install -r requirements.txt
Rscript -e 'install.packages(c("sandwich", "lmtest"))'
```

The results in the paper were produced with R 4.3.3, sandwich 3.1.0 and lmtest 0.9.40.

`pyvinecopulib` is optional; script 22 uses it only to cross-check its copula likelihoods.
`scikit-learn` is optional; script 17 uses it only in a self-test.

All scripts are run from the repository root.

## Data

### Sources

| Data | Source | Snapshot suffix | Script |
|---|---|---|---|
| IBIT, GLD, TLT, SPY, VIX, S&P 500 index (daily) | Yahoo Finance | `__20260921` | `01a` |
| BTC-USD, IBIT, GLD, TLT, SPY, VIX from September 2014 | Yahoo Finance | `__20260921L` | `01b` |
| Further Bitcoin and gold ETFs, first version of the ETF appendix | Yahoo Finance | `__20260921X` | `01c` |
| Bitcoin and gold instruments, equity and bond benchmarks; hourly Bitcoin prices | Yahoo Finance; Coinbase and Bitstamp public market-data APIs; SPDR and iShares fund pages | `__20260921S` | `01d` |
| 13-week Treasury bill yield (`^IRX`), FRED `DTB3` | Yahoo Finance; FRED | `__20260921L` | `01e` |
| 13-week Treasury bill yield for the main sample | Yahoo Finance (`^IRX`), FRED as fallback | `IRX__20260921.csv` | `25` |

### The price data are not in this repository

The data vendor's terms restrict redistribution, so `data/raw/` contains only the manifests and
checksums. There are two ways to obtain the data:

1. **Request the authors' snapshot** from the corresponding author. It contains the raw files and the
   analysis file `data/processed/returns__20260921.csv`; with them, every script reproduces the numbers
   in the paper to the last reported digit. Verify the files with

   ```bash
   cd data/raw
   shasum -a 256 -c checksums__20260921L_X.sha256      # macOS; use sha256sum -c on Linux
   ```

   and compare the checksums in `manifest__20260921.json`, `manifest__20260921S.json` and
   `manifest_rf__20260921.json` with the files.

2. **Download the data yourself** with scripts `01a`–`01e` (and `25` for the main-sample risk-free
   rate). The request windows are fixed in the scripts, so a new download covers the same trading
   days. Yahoo Finance recomputes its adjusted closes after every new distribution, however, so a new
   download will not be byte-identical to the authors' snapshot and results can differ in the last
   digits.

If you need a proxy to reach the data sources, set it in the environment before running, for example
`export HTTPS_PROXY=http://host:port`. No proxy address is stored in the scripts or written to the
manifests.

### The analysis file

`02_build_returns.py` turns the main snapshot into `data/processed/returns__20260921.csv`: simple and
log returns of the four ETFs computed from adjusted closes, aligned on the SPY trading calendar by an
inner join without forward filling, plus the VIX close. The file used in the paper has the SHA-256 in
`data/checksums_processed.sha256` and is supplied with the snapshot. Rebuilding it with `02` from the
archived snapshot gives daily returns that differ from it by at most 9 × 10⁻⁷ (SPY and TLT, whose
adjusted closes Yahoo recomputes after each distribution). Most reported numbers are unaffected.
Quantities obtained by simulation, or from nearly flat likelihoods, change more: some bootstrap-interval
endpoints change in the last reported digit (for example, the upper end of the interval for the initial
volatility of the IBIT portfolio in Table 10 becomes 23.3 instead of 23.4), and some Monte Carlo and
forecast-comparison p-values of the backtests change by up to about 0.1, without changing any exception
count, any model confidence set or any conclusion of the paper. To reproduce the paper exactly, use the
analysis file supplied with the snapshot.

## Running the analysis

The commands below reproduce the paper in order. Run times are approximate, for a two-core Linux
machine.

```bash
# 1. Data (online; skip if you have the authors' snapshot)
python 01a_fetch_raw.py          # main sample                         ~1 min
python 01b_fetch_long.py         # long sample, from 2014              ~1 min
python 01c_fetch_extra.py        # ETF appendix                        ~2 min
python 01d_fetch_scope.py        # scope expansion, hourly prices      5–10 min
python 01e_fetch_rf_long.py      # long-sample risk-free rate          <1 min

# 2. Returns (offline from here on)
python 02_build_returns.py       # -> data/processed/returns__20260921.csv

# 3. Analysis
Rscript 03_hedge_safehaven.R                                        # ~1.5 min
Rscript 05_stress_tables.R                                          # ~1 s
python 04_var_es_backtest.py                                        # ~10 s
python 09_regenerate_paper_numbers.py --outdir out/09               # ~2.5 min
python 10_tail_diagnostics.py                                       # ~5 s
python 11_es_bootstrap_vix.py                                       # ~1 min
python 12_btc_proxy_validity.py                                     # ~5 s
python 15_episode_study.py                                          # ~1 s
python 18_figures.py --data data/processed/returns__20260921.csv    # ~10 s
python 19_sample_size_robustness.py                                 # ~4 min
python 20_scope_expansion.py                                        # ~15 s
python 21_forecast_design.py                                        # ~6 min
python 22_safe_haven_dependence.py                                  # ~15 min
python 23_portfolio_robust.py                                       # ~3 min
python 24_controlled_comparison.py                                  # ~2 min
python 25_data_appendix.py --offline                                # ~1 s; without --offline it downloads the risk-free rate
python 26_tail_supplement.py                                        # ~1 min
python 27_stress_scenarios.py                                       # ~1 s
python 28_implementation_costs.py                                   # ~5 s
```

Order constraints: `02` before every analysis script; `10` before `26`; `20` and `21` before `24`;
`25` before `28`, unless `data/raw/IRX__20260921.csv` is already in the snapshot. `18` imports `04` and
`09`, so the three files must stay in the same folder. `23` needs `IDX_IRX__20260921L.csv` from `01e`.
Otherwise the analysis scripts are independent.

Many of the Python scripts have a `--selftest` option that checks their numerical routines against
closed-form results, numerical integration or a reference package, and most print `CHECK [PASS]` lines
during a normal run. Random numbers come from fixed seeds, so repeated runs give identical output.
Scripts `21`–`24` accept `--quick` for a fast trial run with fewer resamples; the paper uses the
default settings.

## Where each result in the paper comes from

Main text

| Paper | Script | Output |
|---|---|---|
| Table 1 (hypotheses) | — | — |
| Table 2 (portfolio strategies) | 23 | `out/23/portfolio_robust.md` |
| Table 3 (in-sample VaR and ES, four models) | 09; bootstrap intervals from 11 and 19 | `out/09/paper_numbers.md`, `out/11_robust/`, `out/19/` |
| Table 4 (out-of-sample backtests, six models) | 21 | `out/21/forecast_design.md` |
| Table 5 (tail risk by VIX regime) | 19 | `out/19/sample_size_robustness.md` |
| Table 6 (safe-haven regressions) | 03 (Sections 1–4; intervals and MDE for block length 10) | `out/03_hedge/results.txt` |
| Table 7 (eight conditional-dependence measures) | 22 | `out/22/safe_haven_dependence.md` |
| Table 8 (in-sample stress windows) | 09 (also Panel A of 05) | `out/09/paper_numbers.md`, `out/05_stress/stress_tables.md` |
| Table 9 (equity drawdowns since 2014) | 15 | `out/15_episodes/episode_study.md` |
| Table 10 (Monte Carlo 10-day loss distribution) | 19 | `out/19/sample_size_robustness.md` |
| Table 11 (optimal weights) | 09 and 23 | `out/09/paper_numbers.md`, `out/23/portfolio_robust.md` |
| Table 12 (out-of-sample performance of 15 strategies) | 23 | `out/23/portfolio_robust.md` |
| Tables 13–16 (asset-level risk, IBIT vs BTC-USD decomposition, fund comparison, alternative legs) | 20 | `out/20/scope_expansion.md` |
| Figures 1–3 (cumulative returns, conditional volatility, out-of-sample VaR) | 18 | `out/18_fig/fig1_cumulative`, `fig2_condvol`, `fig3_oos_var` (`.png`, `.pdf`) |
| Figure 4 (DCC correlation and time-varying beta) | 22 | `out/22/fig_safe_haven_dynamics.png` |

Appendices

| Paper | Script | Output |
|---|---|---|
| Table A1 (VIX-regime definitions) | 19 | `out/19/sample_size_robustness.md` |
| Tables B1–B3 (all instruments) | 20 | `out/20/scope_expansion.md` |
| Tables C1–C5 (GARCH estimates, extra backtests, predictive-ability tests) | 21 | `out/21/forecast_design.md`, `garch_params_*.csv` |
| Tables C6–C9 (distribution and goodness-of-fit tests, extreme-value estimates) | 26 | `out/26_tail/tail_supplement.md`, `tab_*.csv` |
| Table D1 and Appendix D (data construction) | 25 | `out/25_data_appendix/data_appendix_draft.md`, `tab_*.csv` |
| Tables E1–E4, Figures E1–E2 (controlled comparison) | 24 | `out/24/controlled_comparison.md`, `fig_specification_curve`, `fig_weight_scan` |
| Tables F1–F4 (conditional dependence) | 22 | `out/22/safe_haven_dependence.md` |
| Tables G1–G3 (stress scenarios) | 27 | `out/27_stress/stress_scenarios.md`, `tab_*.csv` |
| Tables H1–H4, Figure H1 (portfolio robustness, long sample) | 23 | `out/23/portfolio_robust.md`, `fig_portfolio_oos_long` |
| Tables H5–H8 (trading costs and capacity) | 28 | `out/28_costs/implementation_costs.md`, `tab_*.csv` |

Numbers quoted in the text

| Result | Script |
|---|---|
| Tail index (Hill), decomposition of the ES ratio into volatility and shape | 10 |
| ES95 ratio 1.63 [1.36, 1.99]; robustness to the VIX regime definition | 11 |
| Minimum detectable effect of the long-sample safe-haven test (1.65) | 12 |
| Dispersion of Bitcoin's reactions relative to gold's across stress episodes (2.62) | 15 |
| 28 in-sample scenarios under three window rules, the three counterexamples, sensitivity to the window dates, hypothetical Bitcoin crash (Section 4.7) | 05 |
| Baur–McDermott tail coefficient without the April 2025 episode (1.13 → 1.80, Section 4.6) | 03 (Section 5 of `results.txt`) |
| SPY × VIX interaction regressions (Section 4.5) | 11 |
| Volatility-matched portfolio: weight from the first 250 days, backtest power | 04 |

## Supporting scripts

These scripts document checks made during the revision. Their results are not reported in the paper,
or have been superseded by a later script.

| Script | Purpose |
|---|---|
| `06_audit_spy.py`, `07_diagnose_spy_vix.py`, `08_test_gspc.py` | Audit of the data used in the first submission; they established that its equity column was the S&P 500 index rather than the SPY ETF, which led to the corrected data set. They need that data file, `ETF_Returns.csv` (not distributed), in the repository root |
| `13_break_robustness.py`, `14_break_diagnose.py` | Structural-break test at IBIT's listing; the result was withdrawn because it did not survive placebo tests |
| `16_etf_appendix.py` | First version of the ETF appendix; superseded by `20` |
| `17_portfolio.py` | First version of the portfolio analysis; superseded by `23` |
| `18b_backtest_supplement.py` | Additional backtests; superseded by `21` |

## Verification

The repository was checked by running the commands above in a fresh folder containing only these
scripts and the authors' snapshot (Linux; Python 3.11.15 with the package versions in
`requirements.txt`; R 4.3.3). Every number in the outputs used in the paper was identical to the
outputs from which the manuscript was written, and the PNG files of Figures 1–4, E1–E2 and H1 were
pixel-identical. The Python scripts were also run on macOS during the revision.

## License

MIT; see `LICENSE`.
