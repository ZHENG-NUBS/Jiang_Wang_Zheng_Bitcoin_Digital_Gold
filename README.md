# Is Bitcoin "Digital Gold"? — replication code

Python and R code for

> Jiang, B., Wang, J. and Zheng, Z. *Is Bitcoin "Digital Gold"? An Empirical Study Based on
> Multi-Model VaR/ES Measures, Stress Testing, and Portfolio Optimization.* Second revision, submitted
> to the *Journal of Risk and Financial Management* (manuscript jrfm-4584647), October 2026.

The scripts download the data, build the return series and produce every table and figure in the
paper. The ETF sample is 607 daily returns of IBIT, GLD, TLT and SPY (12 January 2024 – 15 June 2026).
Two long samples are analyzed alongside it: the Bitcoin long sample (BTC-USD with GLD, TLT and SPY from
18 September 2014, 2952 days) and the gold long sample (COMEX gold futures until November 2004 and GLD
thereafter, with SPY, TLT, the VIX and the 13-week Treasury bill rate, from 31 August 2000, 6466 days).
Further data sets cover other Bitcoin, gold, equity and bond instruments and hourly Bitcoin prices.

## Contents

| Path | What it is |
|---|---|
| `01a`–`01f_*.py` | Data download (with `25`, the only scripts that go online) |
| `02_build_returns.py` | Builds the analysis file `data/processed/returns__20260921.csv` from the snapshot |
| `03_hedge_safehaven.R`, `05_stress_tables.R` | Analysis scripts in R: safe-haven regressions and stress windows |
| `04`–`29_*.py` | Analysis scripts in Python (first revision) |
| `30`–`38_*.py`, `longdata.py` | Analysis scripts added in the second revision; `longdata.py` loads the three samples |
| `data/raw/manifest*.json` | SHA-256 checksums and download metadata of the authors' snapshots |
| `data/raw/checksums__20260921L_X.sha256` | SHA-256 checksums of the long-sample (`L`) and ETF-appendix (`X`) snapshot files of September 2026 |
| `data/raw/checksums__20261021_L_G.sha256` | SHA-256 checksums of the long-sample snapshot downloaded again on 21 October 2026 (`L`) and of the gold long-sample snapshot (`G`) used in the second revision |
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
| COMEX gold futures (`GC=F`), GLD, SPY, TLT, S&P 500 index, VIX and `^IRX` from January 2000 | Yahoo Finance | `__20261021G` | `01f` |

For the second revision the long-sample snapshot (`__20260921L`) was downloaded again on 21 October 2026
with `01b` and `01e`; the files keep their suffix, and the new checksums are in
`checksums__20261021_L_G.sha256`. IBIT, GLD, the VIX and the bill rate are byte-for-byte identical to the
September files; SPY and TLT differ only through Yahoo's rescaling of adjusted closes after new
distributions (daily returns change by less than 10⁻⁵).

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
python 01f_fetch_gold_long.py    # gold long sample, from 2000          ~1 min

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
python 29_regime_correlations.py                                    # ~20 s

# 4. Second revision
python 30_descriptives_marginals.py      # descriptive statistics and marginal GARCH models
python 31_regimes_contagion.py           # Markov-switching regimes, Forbes–Rigobon tests; writes out/31/regimes.csv
python 32_safe_haven_regimes.py          # Baur–McDermott and regime regressions, threshold model, episodes
python 33_long_oos_backtests.py          # long-sample out-of-sample backtests (tens of minutes)
python 34_es_decomposition_equivalence.py  # exact ES decomposition, non-inferiority and TOST
python 35_full_bootstrap_dependence.py   # two-stage bootstrap for copula and DCC (tens of minutes)
python 36_optimizer_weights.py           # out-of-sample weight distributions, fallback rule
python 37_mean_specification.py          # constant versus AR(1) mean
python 38_specification_families.py      # specification families, rebalancing decomposition
```

`35` can be run in parallel, one process per asset, with `--items`, for example
`--items "etf|IBIT,etf|TLT,btc_long|TLT"` and `--items "btc_long|BTC,etf|GLD,btc_long|GLD"`; each item has
its own random-number stream, so the results do not depend on how the items are split, and the tables are
written once all six items are available.

Order constraints: `02` before every analysis script; `10` before `26`; `20` and `21` before `24`;
`25` before `28`, unless `data/raw/IRX__20260921.csv` is already in the snapshot. `18` imports `04` and
`09`, and `29` imports `19`, so these files must stay in the same folder. `23` needs
`IDX_IRX__20260921L.csv` from `01e`. `31` before `32` and `33`; `24` and `28` before `38`; `33` imports
`21`, and `36` imports `23`. Otherwise the analysis scripts are independent.

Many of the Python scripts have a `--selftest` option that checks their numerical routines against
closed-form results, numerical integration or a reference package, and most print `CHECK [PASS]` lines
during a normal run. Random numbers come from fixed seeds, so repeated runs give identical output.
Scripts `21`–`24` accept `--quick` for a fast trial run with fewer resamples; the paper uses the
default settings.

## Where each result in the paper comes from

Numbers refer to the second revision.

Main text

| Paper | Script | Output |
|---|---|---|
| Tables 1–2 (descriptive statistics; marginal GARCH models) | 30 | `out/30/` |
| Table 3 (hypotheses) | — | — |
| Table 4 (portfolio strategies) | 23 | `out/23/portfolio_robust.md` |
| Table 5 (in-sample VaR and ES) | 09; intervals from 11 and 19; long-sample ratios from 34 | `out/09/`, `out/11_robust/`, `out/19/`, `out/34/` |
| Table 6 (exact decomposition, equivalence tests) | 34 | `out/34/` |
| Table 7 (out-of-sample backtests, ETF sample) | 21; AR(1) mean from 33 | `out/21/forecast_design.md`, `out/33/` |
| Table 8 (out-of-sample backtests, long samples) | 33 | `out/33/` |
| Table 9, Figure 4 (market regimes, Forbes–Rigobon tests) | 31 | `out/31/`, `fig_ms_regimes.png` |
| Table 10 (hedge and safe-haven regressions) | 32; tail slope and MDE from 03 and 12 | `out/32/`, `out/03_hedge/results.txt` |
| Table 11 (primary and robustness dependence measures) | 22, 32 and 35 | `out/22/`, `out/32/`, `out/35/` |
| Table 12 (equity drawdowns since 2000) | 32 | `out/32/` |
| Table 13 (in-sample stress windows) | 09 (also 05) | `out/09/paper_numbers.md`, `out/05_stress/` |
| Table 14 (Monte Carlo 10-day losses) | 19 | `out/19/sample_size_robustness.md` |
| Table 15 (out-of-sample allocation, risk and return) | 36 (strategies from 23) | `out/36/` |
| Table 16 (IBIT versus BTC-USD decomposition) | 20 | `out/20/scope_expansion.md` |
| Table 17 (specification families, rebalancing rules) | 38 (from 24 and 28) | `out/38/` |
| Figures 1–3 | 18 | `out/18_fig/` |
| Figure 5 (DCC correlation and time-varying beta) | 22 | `out/22/fig_safe_haven_dynamics.png` |

Appendices

| Paper | Script | Output |
|---|---|---|
| Tables A1–A2 (VIX buckets and definitions) | 19 | `out/19/sample_size_robustness.md` |
| Table A3 (correlations by VIX bucket) | 29 | `out/29/regime_correlations.md` |
| Tables B1–B5 (all instruments, fund features, alternative legs) | 20 | `out/20/scope_expansion.md` |
| Tables C1–C5 | 21 | `out/21/forecast_design.md` |
| Tables C6–C9, Figure C1 | 26 | `out/26_tail/` |
| Table C10 (constant versus AR(1) mean) | 37 | `out/37/` |
| Table D1 and Appendix D | 25 | `out/25_data_appendix/` |
| Tables E1–E4, Figures E1–E2 | 24 | `out/24/` |
| Tables F1–F4 | 22 | `out/22/safe_haven_dependence.md` |
| Table F5 (two-stage bootstrap) | 35 | `out/35/full_bootstrap.md` |
| Tables G1–G3 | 27 | `out/27_stress/` |
| Table G4 (drawdowns of 8% or more since 2014) | 15 | `out/15_episodes/episode_study.md` |
| Tables H1, H3, H4, Figure H1 | 23 | `out/23/portfolio_robust.md` |
| Table H2 (long-sample allocation and fallback rule) | 36 | `out/36/` |
| Tables H5–H8 | 28 | `out/28_costs/` |
| Tables I1–I2 (Markov-switching models, Forbes–Rigobon tests) | 31 | `out/31/` |
| Tables I3–I4 (threshold and Markov-switching regressions) | 32 | `out/32/` |
| Table J1 (inference register) | — | — |

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

Second revision: the scripts added in the second revision (`30`–`38`) and scripts `12`, `13`, `15`, `22`,
`23` and `27`, which read the long-sample snapshot, were run on Linux with Python 3.13.16 and the package
versions in `requirements.txt`, on the snapshots listed in `checksums__20261021_L_G.sha256`. The results of
the first revision that use the long-sample snapshot were unchanged at the reported precision.

First revision:

The repository was checked by running the commands above in a fresh folder containing only these
scripts and the authors' snapshot (Linux; Python 3.11.15 with the package versions in
`requirements.txt`; R 4.3.3). Every number in the outputs used in the paper was identical to the
outputs from which the manuscript was written, and the PNG files of Figures 1–4, E1–E2 and H1 were
pixel-identical. Table A2 and Figure C1 were added afterwards and come from scripts 29 and 26 run in the
same environment. The Python scripts were also run on macOS during the revision.

## License

MIT; see `LICENSE`.
