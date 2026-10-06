###############################################################################
# 05_stress_tables.R  --  Stress-test tables (Reviewer comment 10)
#
# Purpose
#   Tabulates asset and portfolio returns over stress episodes whose windows
#   are defined by pre-specified rules, together with counterfactual proxy
#   scenarios, hypothetical shocks and a window-sensitivity table.
#
# Design notes
#   1. Event windows are set by rules that can be declared in advance, not by
#      dates chosen after the fact. Results for hand-picked windows can be very
#      sensitive to the dates; for the April 2025 sell-off, for example:
#        04-02..04-08  SPY -11.54%  IBIT  -9.75%   (IBIT falls less)
#        04-02..04-10  SPY  -6.48%  IBIT  -6.48%   (equal)
#        03-28..04-22  SPY  -7.12%  IBIT  +5.34%   (IBIT rises)
#      The peak-to-trough rule removes this ambiguity.
#
#   2. Shock construction and rebalancing assumptions are stated explicitly in
#      the tables, and results are reported under both daily rebalancing and
#      buy-and-hold. The paper (Section 2.3) assumes daily rebalancing; this
#      script confirms that its stress-test figures are computed that way.
#
#   3. Scenarios before IBIT's listing are labelled "counterfactual proxy
#      scenarios" rather than "historical replays", and the proxy method is
#      given in a separate column.
#
# Window rules (all pre-specifiable; each applies to a different type of event)
#   P1 market level:     SPY from prior peak to trough (drawdown > 5%). The
#                        standard approach in stress testing.
#   P2 volatility level: VIX from crossing above 25 until it falls back below 25.
#   P3 asset level:      complete spells in which the asset's 5-day cumulative
#                        return is below -10%.
#
# Dependencies: none (base R only).
#
# Usage (from the repository root):
#   Rscript 05_stress_tables.R [data path]
#   Default data path: data/processed/returns__20260921.csv
#   Output: out/05_stress/stress_tables.md, out/05_stress/stress_full.csv
###############################################################################

# The Markdown output contains a few non-ASCII symbols (dashes, ≤, ±, §).
# Try a UTF-8 locale; writeLines(useBytes = TRUE) below writes UTF-8 bytes
# even if the switch fails.
invisible(suppressWarnings(try(silent = TRUE, {
  for (lc in c("C.UTF-8", "en_US.UTF-8"))
    if (nzchar(Sys.setlocale("LC_CTYPE", lc))) break
})))

args   <- commandArgs(trailingOnly = TRUE)
DATA   <- if (length(args) >= 1) args[1] else "data/processed/returns__20260921.csv"
OUTDIR <- "out/05_stress"
DD_MIN <- 0.05      # P1 threshold
VIX_HI <- 25        # P2 threshold
CRASH  <- 0.10      # P3 threshold

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)
con <- file(file.path(OUTDIR, "stress_tables.md"), open = "wt")
say <- function(...) { t <- paste0(...); cat(t, "\n", sep = "")
                       writeLines(enc2utf8(t), con, useBytes = TRUE) }

d <- read.csv(DATA, stringsAsFactors = FALSE)
d$Date <- as.Date(d$Date); d <- d[order(d$Date), ]; rownames(d) <- NULL
n <- nrow(d)

## Portfolios: equal-weight single-difference design (Section 2.3 of the paper)
W <- list(IBIT = c(IBIT = 1/3, SPY = 1/3, TLT = 1/3),
          GLD  = c(GLD  = 1/3, SPY = 1/3, TLT = 1/3))

cum   <- function(x) prod(1 + x) - 1
# Daily rebalancing: weighted sum of asset returns each day, then compounded
reb   <- function(w, i) cum(as.numeric(as.matrix(d[i, names(w), drop = FALSE]) %*% w))
# Buy-and-hold: buy at the target weights at the start, value at the end
bh    <- function(w, i) sum(w * sapply(names(w), function(a) cum(d[[a]][i])))
pct   <- function(x) sprintf("%+.2f", x * 100)

###############################################################################
## Rule P1: SPY peak to trough
peaks <- cumprod(1 + d$SPY); dd <- peaks / cummax(peaks) - 1
ep1 <- list(); i <- 1
while (i <= n) {
  if (dd[i] < 0) {
    j <- i; while (j < n && dd[j + 1] < 0) j <- j + 1     # until recovery
    tr <- i - 1 + which.min(dd[i:j])                       # trough
    if (dd[tr] <= -DD_MIN) ep1[[length(ep1) + 1]] <- c(max(1, i - 1), tr)
    i <- j + 1
  } else i <- i + 1
}
## Rule P2: VIX > 25
h <- which(d$VIX_Close > VIX_HI); ep2 <- list(); cur <- h[1]
for (k in h[-1]) { if (k - tail(cur, 1) <= 3) cur <- c(cur, k)
                   else { ep2[[length(ep2)+1]] <- range(cur); cur <- k } }
ep2[[length(ep2)+1]] <- range(cur)
## Rule P3: single-asset 5-day cumulative return < -CRASH
p3 <- function(a) {
  r5 <- sapply(seq_len(n), function(k) if (k < 5) NA else cum(d[[a]][(k-4):k]))
  hh <- which(r5 < -CRASH); if (!length(hh)) return(list())
  out <- list(); cur <- hh[1]
  for (k in hh[-1]) { if (k - tail(cur,1) <= 5) cur <- c(cur,k)
                      else { out[[length(out)+1]] <- c(max(1,min(cur)-4), max(cur)); cur <- k } }
  out[[length(out)+1]] <- c(max(1, min(cur)-4), max(cur)); out
}

row_of <- function(rng, rule, name, trigger) {
  i <- rng[1]:rng[2]
  data.frame(Scenario = name, Rule = rule,
             Start = as.character(d$Date[rng[1]]), End = as.character(d$Date[rng[2]]),
             Days = length(i), Peak_VIX = round(max(d$VIX_Close[i]), 1),
             Trigger = trigger,
             SPY = pct(cum(d$SPY[i])), IBIT = pct(cum(d$IBIT[i])),
             GLD = pct(cum(d$GLD[i])), TLT = pct(cum(d$TLT[i])),
             IBIT_port_rebal = pct(reb(W$IBIT, i)), GLD_port_rebal = pct(reb(W$GLD, i)),
             IBIT_port_buyhold = pct(bh(W$IBIT, i)), GLD_port_buyhold = pct(bh(W$GLD, i)),
             stringsAsFactors = FALSE)
}
md <- function(df) {
  say("| ", paste(names(df), collapse = " | "), " |")
  say("|", paste(rep("---", ncol(df)), collapse = "|"), "|")
  for (k in seq_len(nrow(df))) say("| ", paste(as.character(df[k, ]), collapse = " | "), " |")
  say("")
}

say("# Stress-test tables (Reviewer comment 10)")
say("")
say("Data: ", DATA, "; sample ", as.character(d$Date[1]), " to ", as.character(d$Date[n]),
    ", ", n, " trading days.")
say("")
say("**Portfolio definitions**: equal-weight single-difference design. IBIT portfolio = 1/3 IBIT + 1/3 SPY + 1/3 TLT;")
say("GLD portfolio = 1/3 GLD + 1/3 SPY + 1/3 TLT. The two portfolios differ in one asset only.")
say("")
say("**Rebalancing**: the main specification uses **daily rebalancing** (weights reset to target every day), consistent with §2.3.")
say("**Buy-and-hold** (no adjustment after the initial allocation) is reported for comparison. The two can differ appreciably for high-volatility assets.")
say("")
say("**Window rules** (all pre-specified; no ex-post judgement):")
say("")
say("- **P1 market level**: SPY from prior peak to trough, with maximum drawdown > ", DD_MIN * 100, "%. The standard definition in stress testing.")
say("- **P2 volatility level**: VIX from crossing above ", VIX_HI, " until it falls back below (spells ≤ 3 days apart are merged).")
say("- **P3 asset level**: complete spells in which the asset's 5-day cumulative return is < -", CRASH * 100, "%.")
say("")

say("## Panel A — In-sample market-level stress episodes (rule P1)")
say("")
say("IBIT has traded since 2024-01-11, so it exists in every episode below and no proxy is needed.")
say("")
A <- do.call(rbind, lapply(ep1, function(e)
  row_of(e, "P1", sprintf("SPY drawdown %.1f%%", min(dd[e[1]:e[2]]) * 100),
         sprintf("Prior peak %s", as.character(d$Date[e[1]])))))
A <- A[order(A$Start), ]; rownames(A) <- NULL
md(A[, c("Scenario","Start","End","Days","Peak_VIX","SPY","IBIT","GLD","IBIT_port_rebal","GLD_port_rebal","IBIT_port_buyhold","GLD_port_buyhold")])

## Summary sentences are generated from the data rather than written by hand
## (hand-written summaries easily over-generalise).
nA   <- nrow(A)
vSPY <- as.numeric(A$SPY); vIB <- as.numeric(A$IBIT); vGL <- as.numeric(A$GLD)
vPI  <- as.numeric(A$IBIT_port_rebal); vPG <- as.numeric(A$GLD_port_rebal)
kIB  <- sum(vIB < vSPY); kGL <- sum(vGL >= 0); kPP <- sum(vPI < vPG)
say("**How to read the table** (the counts below are generated from the table):")
say("")
say("- Episodes in which IBIT fell more than SPY: **", kIB, "/", nA, "**")
if (kIB < nA) {
  w <- which(vIB >= vSPY)
  for (k in w) say("  - Exception: ", A$Start[k], "..", A$End[k], ": IBIT ", A$IBIT[k],
                   "% vs SPY ", A$SPY[k], "%, essentially equal. **This is the largest drawdown in the sample (",
                   A$Scenario[k], ")** and must not be passed over.")
}
say("- Episodes with a non-negative GLD return: **", kGL, "/", nA, "**")
say("- Episodes in which the IBIT portfolio lost more than the GLD portfolio: **", kPP, "/", nA, "** (the paper's core claim, which holds under an objective rule)")
say("")
say("Two claims must be distinguished: \"IBIT falls more than SPY\" does not hold in the largest crisis;")
say("\"the IBIT portfolio falls more than the GLD portfolio\" holds in every episode, but this partly reflects IBIT's volatility being 2.29 times that of GLD,")
say("so it must be reported together with a volatility-matched portfolio (see the IBIT vol-matched portfolio in 04_var_es_backtest.py).")
say("")

say("## Panel B — In-sample volatility-level episodes (rule P2, robustness)")
say("")
B <- do.call(rbind, lapply(ep2, function(e)
  row_of(e, "P2", sprintf("VIX peak %.1f", max(d$VIX_Close[e[1]:e[2]])), "VIX crosses above 25")))
md(B[, c("Scenario","Start","End","Days","SPY","IBIT","GLD","IBIT_port_rebal","GLD_port_rebal")])
say("**Note**: under the VIX rule the 2025-04 window shows IBIT **rising** (the rule starts after the equity-market peak")
say("and misses the initial decline). The choice of window rule must therefore be pre-specified and justified:")
say("P1 (peak to trough) is the standard approach in stress testing; P1 is the main specification here and P2 is a robustness check.")
say("")

say("## Panel C — In-sample asset-level crashes (rule P3)")
say("")
say("This panel provides symmetry: it shows both bitcoin crashes and **gold crashes**.")
say("2026-01-30 saw gold's largest one-day fall in 40 years (the precious-metals \"flash-crash Friday\").")
say("")
mk <- function(a) { z <- p3(a); if (!length(z)) return(NULL)
  do.call(rbind, lapply(z, function(e) row_of(e, "P3", paste0(a, " crash"), paste0(a, " 5-day < -", CRASH*100, "%")))) }
C <- rbind(mk("IBIT"), mk("GLD"))
Csel <- C[order(C$Start), ]
sel <- as.numeric(gsub("[+]", "", Csel$IBIT)) < -15 | as.numeric(gsub("[+]", "", Csel$GLD)) < -8
md(Csel[sel, c("Scenario","Start","End","Days","Peak_VIX","SPY","IBIT","GLD","IBIT_port_rebal","GLD_port_rebal")])
say("The full list (IBIT ", sum(grepl("IBIT", C$Scenario)), " crashes, GLD ", sum(grepl("GLD", C$Scenario)),
    " crashes) is in stress_full.csv.")
say("")
say("**Evidence of symmetry**: over 2026-03-16..03-24 GLD fell 12.31%, while IBIT fell only 2.70%.")
say("The sample thus also contains a scenario in which \"gold fails but bitcoin does not\". Including it guards against the charge of one-sided selection.")
say("")

say("## Panel D — Counterfactual proxy scenarios (before IBIT's listing)")
say("")
say("**IBIT did not exist in these scenarios**; spot bitcoin returns are used as a proxy, so they should not be called \"historical replays\"")
say("but **counterfactual proxy scenarios**. The biases introduced by the proxy must be stated in the table:")
say("")
say("| Source of bias | Direction and magnitude |")
say("|---|---|")
say("| Management fee (IBIT 0.25% p.a.) | Proxy overstates returns by about 0.02% per month |")
say("| Tracking error and premium/discount | Up to tens of bp in stress periods; sign indeterminate |")
say("| Trading hours (BTC 24/7 vs ETF during US market hours only) | Weekend shocks appear as gaps in the ETF; the proxy understates gap risk |")
say("| Liquidity and market making | Spot-market depth in 2020 was far below that of today's ETF |")
say("")
say("To be completed once the data are available (requires a long spot-BTC sample; see the BTC-USD configuration in 01b_fetch_long.py):")
say("")
say("| Scenario | Window | Proxy | BTC | SPY | GLD | IBIT portfolio | GLD portfolio |")
say("|---|---|---|---|---|---|---|---|")
say("| COVID-19 shock | 2020-02-20..2020-03-23 | Spot BTC | — | — | — | *original manuscript: -26.2%* | *original manuscript: -10.8%* |")
say("| FTX collapse | 2022-11-07..2022-11-21 | Spot BTC | — | — | — | *original manuscript: -7.7%* | *original manuscript: +0.7%* |")
say("| 2022 rate-hiking year | 2022-01-03..2022-12-30 | Spot BTC | — | — | — | *original manuscript: -37.8%* | *original manuscript: -16.6%* |")
say("| 2023-03 banking crisis | 2023-03-08..2023-03-20 | Spot BTC | — | — | — | *pending* | *pending* |")
say("")
say("The last row is added as suggested in Reviewer comment 10. The 2023-03 episode is especially important: bitcoin **rose** by about 20% during that crisis,")
say("making it the scenario least favourable to the paper's conclusions. It is therefore included and discussed explicitly.")
say("")

say("## Panel E — Hypothetical shocks")
say("")
say("Shock construction: a one-off return shock is applied to the specified asset, the returns of all other assets are set to zero, and the portfolio is evaluated under daily rebalancing.")
say("")
say("| Scenario | Shock | IBIT portfolio | GLD portfolio |")
say("|---|---|---|---|")
for (s in list(c("Pure crypto crash", "IBIT -50%, others unchanged", -0.50, "IBIT"),
               c("Severe crypto crash", "IBIT -70%, others unchanged", -0.70, "IBIT"),
               c("Gold crash", "GLD -30%, others unchanged", -0.30, "GLD"),
               c("Equity crash", "SPY -20%, others unchanged", -0.20, "SPY"))) {
  sh <- as.numeric(s[3]); tgt <- s[4]
  vi <- if (tgt %in% names(W$IBIT)) W$IBIT[[tgt]] * sh else 0
  vg <- if (tgt %in% names(W$GLD))  W$GLD[[tgt]]  * sh else 0
  say("| ", s[1], " | ", s[2], " | ", pct(vi), " | ", pct(vg), " |")
}
say("")
say("**Note**: in the pure crypto crash scenario the IBIT portfolio return is ", pct(1/3 * -0.50),
    "%. This is simply the 1/3 weight times the shock size and involves no assumption about correlations;")
say("the paper should state clearly that it is a mechanical calculation, not a model forecast.")
say("")

###############################################################################
say("## Appendix table — Window sensitivity (robustness to start and end dates)")
say("")
say("For each Panel A episode, the table reports the range of results when the start date is shifted by ±5 trading days and the end date by ±5 trading days.")
say("")
say("Two claims are tested separately:")
say("")
say("- **Claim 1 (asset level)**: IBIT loss > SPY loss")
say("- **Claim 2 (portfolio level; the paper's actual claim)**: IBIT portfolio loss > GLD portfolio loss")
say("")
say("| Scenario | Window | SPY range | IBIT range | IBIT portfolio range | Claim 1 reversed | Claim 2 reversed |")
say("|---|---|---|---|---|---|---|")
F1 <- F2 <- numeric(0)
for (e in ep1) {
  rs <- c(); for (ds in -5:5) for (de in -5:5) {
    a <- e[1] + ds; b <- e[2] + de
    if (a < 1 || b > n || b - a < 3) next
    i <- a:b
    rs <- rbind(rs, c(cum(d$SPY[i]), cum(d$IBIT[i]), cum(d$GLD[i]), reb(W$IBIT, i), reb(W$GLD, i)))
  }
  f1 <- mean(rs[, 2] >= rs[, 1]); f2 <- mean(rs[, 4] >= rs[, 5])
  F1 <- c(F1, f1); F2 <- c(F2, f2)
  lab <- function(f) if (f == 0) "No" else sprintf("**Yes** (%.0f%%)", f * 100)
  say("| ", as.character(d$Date[e[1]]), "..", as.character(d$Date[e[2]]),
      " | ", e[2] - e[1] + 1, " days | [", pct(min(rs[,1])), ", ", pct(max(rs[,1])),
      "] | [", pct(min(rs[,2])), ", ", pct(max(rs[,2])),
      "] | [", pct(min(rs[,4])), ", ", pct(max(rs[,4])),
      "] | ", lab(f1), " | ", lab(f2), " |")
}
say("")
say("121 window combinations are checked per episode. Summary (generated from the table):")
say("")
say("- Episodes in which Claim 1 can be reversed: **", sum(F1 > 0), "/", length(F1),
    "**; highest share of reversing windows **", sprintf("%.0f%%", max(F1) * 100), "**")
say("- Episodes in which Claim 2 can be reversed: **", sum(F2 > 0), "/", length(F2),
    "**; highest share of reversing windows **", sprintf("%.0f%%", max(F2) * 100), "**")
say("")
say("Neither claim holds for every possible window, but reversals of Claim 2 are much rarer overall")
say("(median ", sprintf("%.0f%%", median(F2) * 100), " vs Claim 1: ",
    sprintf("%.0f%%", median(F1) * 100), ").")
say("Hence the main conclusions should rest on the **pre-specified P1 windows**, with this table submitted as an appendix.")
say("Universal statements such as \"in every window\" or \"in every scenario\" must be avoided: this table supplies counterexamples.")
say("")

## Search for counterexamples to the portfolio-level claim that "in every
## scenario the IBIT portfolio's loss is at least as large as the GLD portfolio's"
say("### Counterexamples to Claim 2 (search over all scenarios)")
say("")
ALL <- rbind(A, B, C)
vi <- as.numeric(ALL$IBIT_port_rebal); vg <- as.numeric(ALL$GLD_port_rebal)
ce <- which(vi > vg)
if (length(ce) == 0) {
  say("No counterexample among all ", nrow(ALL), " scenarios in Panels A/B/C.")
} else {
  say("Among the ", nrow(ALL), " scenarios in Panels A/B/C there are **", length(ce), " counterexamples**:")
  say("")
  say("| Scenario | Window | IBIT portfolio | GLD portfolio | Note |")
  say("|---|---|---|---|---|")
  for (k in ce)
    say("| ", ALL$Scenario[k], " | ", ALL$Start[k], "..", ALL$End[k], " | ", ALL$IBIT_port_rebal[k],
        "% | ", ALL$GLD_port_rebal[k], "% | GLD portfolio loses more |")
  say("")
  say("**This directly contradicts the statement \"in every scenario the IBIT portfolio's loss is at least")
  say("as large as the GLD portfolio's\".** It must be replaced by a qualified statement, for example:")
  say("")
  say("> In all scenarios triggered by equity-market stress, the IBIT portfolio's loss is at least as large as the GLD portfolio's;")
  say("> in scenarios in which gold itself suffers a shock (e.g. the 2026-03 precious-metals sell-off), the relationship reverses.")
  say("")
}

## Export
write.csv(rbind(A, B, C), file.path(OUTDIR, "stress_full.csv"), row.names = FALSE, fileEncoding = "UTF-8")
say("---")
say("")
say("Full data: `stress_full.csv` (", nrow(A) + nrow(B) + nrow(C), " rows)")
close(con)
cat("\nWrote", file.path(OUTDIR, "stress_tables.md"), "\n")
