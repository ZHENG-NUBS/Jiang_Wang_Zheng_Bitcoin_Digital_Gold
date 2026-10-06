###############################################################################
# 03_hedge_safehaven.R  --  Hedge, diversifier and safe-haven tests
#                           (Reviewer comments 1, 3 and 9)
#
# Purpose
#   Tests whether IBIT, GLD and TLT act as a hedge, a diversifier or a safe
#   haven for the equity market (SPY): unconditional dependence, Baur-McDermott
#   tail regressions, stationary-block-bootstrap confidence intervals and
#   minimum detectable effects, an episode-level jackknife, cross-quantilograms,
#   a look-ahead robustness check and time-varying correlation figures.
#
# Design notes (each choice is supported by a diagnostic)
#
#   1. The main table uses the original Baur-McDermott specification, without
#      dummy level terms. Adding the level terms raises the maximum VIF from
#      below 6 to several hundred in this sample; the specification is then not
#      identified and the coefficients take economically meaningless values.
#      The script prints the VIFs for the record (Section 1).
#
#   2. A free-intercept subsample regression is reported alongside. This is
#      the key honesty check: the interaction specification forces the tail
#      regression line through the full-sample intercept, whereas all tail
#      values of x lie between about -6% and -1.5%, far from 0. This is a
#      strong extrapolation constraint that manufactures precision. Reporting
#      both side by side shows the true uncertainty.
#
#   3. rq(y ~ x, tau = ...) is not used: its tau is a conditional quantile of
#      y itself, not a state of extreme equity-market decline, so it does not
#      answer the Baur-McDermott question. Tail dependence is examined with
#      the cross-quantilogram (Han et al. 2016) instead.
#
#   4. The cumulative coefficient b0+b1+b2 is tested as a linear combination,
#      not by combining the t-statistics of individual coefficients.
#
#   5. All confidence intervals use the stationary block bootstrap
#      (Politis-Romano), which preserves volatility clustering. Expected block
#      lengths of 5, 10 and 20 are reported instead of a single ad hoc choice.
#
#   6. The minimum detectable effect (MDE) is reported. This answers the
#      sample-size comment directly by quantifying the identification limits
#      of the design.
#
# Dependencies: R packages sandwich and lmtest (the bootstrap is implemented
#               in base R; boot is not required).
#
# Usage (from the repository root):
#   Rscript 03_hedge_safehaven.R [data path]
#   Default data path: data/processed/returns__20260921.csv
#   Output: out/03_hedge/results.txt, out/03_hedge/fig_dynamic_correlation.pdf,
#           out/03_hedge/fig_bootstrap_tailbeta.pdf
###############################################################################

suppressPackageStartupMessages({
  library(sandwich); library(lmtest)
})

# The output contains a few non-ASCII symbols (Greek letters, arrows). Try a
# UTF-8 locale first; writeLines(useBytes = TRUE) below writes the UTF-8 bytes
# even if the switch fails.
invisible(suppressWarnings(try(silent = TRUE, {
  for (lc in c("C.UTF-8", "en_US.UTF-8"))
    if (nzchar(Sys.setlocale("LC_CTYPE", lc))) break
})))

## ------------------------------------------------------- Configuration ----
args     <- commandArgs(trailingOnly = TRUE)
DATA     <- if (length(args) >= 1) args[1] else "data/processed/returns__20260921.csv"
OUTDIR   <- "out/03_hedge"
MKT      <- "SPY"                       # market factor
ASSETS   <- c("IBIT", "GLD", "TLT")     # assets to be tested
QS       <- c(0.10, 0.05)               # tail quantiles (1% has only ~6 observations and is not modelled separately)
HAC_LAG  <- 5
BOOT_R   <- 4000
BOOT_L   <- c(5, 10, 20)                # block-length sensitivity
SEED     <- 20260921

set.seed(SEED)
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

con <- file(file.path(OUTDIR, "results.txt"), open = "wt")
say <- function(...) {
  txt <- paste0(...)
  cat(txt, "\n", sep = "")
  writeLines(enc2utf8(txt), con, useBytes = TRUE)   # bypass locale conversion
}
rule <- function(ch = "=") say(strrep(ch, 78))

## ----------------------------------------------------------- Load data ----
d <- read.csv(DATA, stringsAsFactors = FALSE)
d$Date <- as.Date(d$Date)
d <- d[order(d$Date), ]
rownames(d) <- NULL

stopifnot(MKT %in% names(d))
ASSETS <- ASSETS[ASSETS %in% names(d)]
stopifnot(length(ASSETS) > 0)

n <- nrow(d)
rule(); say("Data: ", DATA)
say(sprintf("  %d trading days, %s → %s", n, d$Date[1], d$Date[n]))
say(sprintf("  Market factor %s; test assets %s", MKT, paste(ASSETS, collapse = ", ")))
if (anyNA(d[, c(MKT, ASSETS)])) {
  say("  Warning: missing values found; incomplete rows removed")
  keep <- complete.cases(d[, c(MKT, ASSETS)]); d <- d[keep, ]; n <- nrow(d)
}

mkt <- d[[MKT]]

## Tail dummies (unconditional quantiles; a rolling-quantile version is in Section 7)
mk_dummies <- function(m, qs) {
  D <- sapply(qs, function(q) as.numeric(m <= quantile(m, q)))
  colnames(D) <- paste0("q", sprintf("%02d", round(qs * 100)))
  as.data.frame(D)
}
DM <- mk_dummies(mkt, QS)
qn <- names(DM)   # e.g. c("q10","q05")

## ------------------------------------- Helper: linear-combination test ----
# Computed directly from the covariance matrix rather than with
# car::linearHypothesis: one dependency fewer and one less version risk.
lc_test <- function(cf, V, w) {
  est <- sum(w * cf)
  se  <- sqrt(as.numeric(t(w) %*% V %*% w))
  tv  <- est / se
  c(est = est, se = se, t = tv, p = 2 * pnorm(-abs(tv)))
}

## -------------------------- Specification A: original Baur-McDermott ----
# y = c0 + c1*m + c2*m*1{m<=Q10} + c3*m*1{m<=Q05}
fit_bm <- function(y, m, DM) {
  X <- data.frame(y = y, m = m)
  for (k in names(DM)) X[[paste0("m_", k)]] <- m * DM[[k]]
  lm(y ~ ., data = X)
}

## ------ Specification B: free-intercept subsample regression (honesty check)
fit_sub <- function(y, m, thr) {
  s <- m <= thr
  lm(y[s] ~ m[s])
}

## ------------------------------ Stationary block bootstrap (Politis-Romano)
# Geometric block lengths with circular wrapping. Returns an (n x R) matrix
# of resampled row indices.
stationary_boot_idx <- function(n, R, L) {
  p <- 1 / L
  out <- matrix(NA_integer_, n, R)
  for (r in seq_len(R)) {
    idx <- integer(0)
    while (length(idx) < n) {
      s <- sample.int(n, 1)
      len <- rgeom(1, p) + 1L
      idx <- c(idx, ((s + seq_len(len) - 2L) %% n) + 1L)
    }
    out[, r] <- idx[seq_len(n)]
  }
  out
}

## For each resample, the quantile thresholds are recomputed (the thresholds
## are functions of the sample and must be resampled with it).
boot_stats <- function(y, m, IDX, qs) {
  R <- ncol(IDX)
  out <- matrix(NA_real_, R, 2,
                dimnames = list(NULL, c("bm_tail", "sub_tail")))
  for (r in seq_len(R)) {
    i  <- IDX[, r]; ys <- y[i]; ms <- m[i]
    Db <- mk_dummies(ms, qs)
    fb <- try(fit_bm(ys, ms, Db), silent = TRUE)
    if (!inherits(fb, "try-error")) out[r, 1] <- sum(coef(fb)[-1])
    thr <- quantile(ms, min(qs))
    sel <- ms <= thr
    if (sum(sel) >= 10 && sd(ms[sel]) > 0) {
      fs <- try(lm(ys[sel] ~ ms[sel]), silent = TRUE)
      if (!inherits(fs, "try-error")) out[r, 2] <- coef(fs)[2]
    }
  }
  out
}

###############################################################################
rule(); say("Section 0  Unconditional dependence: hedge or diversifier (H1)"); rule()
say("Baur & Lucey (2010): negatively correlated or uncorrelated on average = hedge;")
say("                     positively but not perfectly correlated          = diversifier")
say("")
say(sprintf("%-8s %10s %10s %12s %12s %s", "Asset", "Corr.", "beta", "HAC t", "p", "Verdict"))
for (a in ASSETS) {
  m0 <- lm(d[[a]] ~ mkt)
  V0 <- NeweyWest(m0, lag = HAC_LAG, prewhite = FALSE, adjust = TRUE)
  ct <- coeftest(m0, vcov. = V0)
  rho <- cor(d[[a]], mkt)
  verdict <- if (ct[2, 4] > 0.10) "Hedge (not different from 0)" else
             if (ct[2, 1] < 0)   "Hedge (significantly negative)" else "Diversifier"
  say(sprintf("%-8s %10.4f %10.4f %12.2f %12.4f  %s",
              a, rho, ct[2, 1], ct[2, 3], ct[2, 4], verdict))
}

###############################################################################
rule(); say("Section 1  Identifiability of the tail specification (why no level terms)"); rule()

vif_of <- function(X) {
  # X: design matrix without the intercept
  sapply(seq_len(ncol(X)), function(j) {
    r2 <- summary(lm(X[, j] ~ X[, -j, drop = FALSE]))$r.squared
    1 / (1 - r2)
  })
}
Xa <- as.matrix(data.frame(m = mkt, sapply(qn, function(k) mkt * DM[[k]])))
colnames(Xa) <- c("m", paste0("m_", qn))
Xb <- cbind(Xa, as.matrix(DM))

say("Specification A (original form, no level terms) VIF:")
va <- vif_of(Xa); for (j in seq_along(va)) say(sprintf("   %-10s %9.1f", colnames(Xa)[j], va[j]))
say("Specification A+ (with dummy level terms) VIF:")
vb <- vif_of(Xb); for (j in seq_along(vb)) say(sprintf("   %-10s %9.1f", colnames(Xb)[j], vb[j]))
say("")
say(sprintf("Adding the level terms raises the maximum VIF from %.1f to %.1f.", max(va), max(vb)))
say("Within the tail the level and interaction terms are nearly collinear (x spans a very narrow range), so the specification is not identified.")
say("→ The main table uses Specification A; the effect of the intercept constraint is shown by the free-intercept regressions in Section 3.")

###############################################################################
rule(); say("Section 2  Main table: Baur–McDermott tail regressions (HAC Newey–West)"); rule()
say(sprintf("Tail thresholds: 10%% = %.4f%%   5%% = %.4f%%",
            quantile(mkt, .10) * 100, quantile(mkt, .05) * 100))
say(sprintf("Tail observations: 10%% = %d   5%% = %d",
            sum(DM[[qn[1]]]), sum(DM[[qn[2]]])))
say("")

FITS <- list()
for (a in ASSETS) {
  f <- fit_bm(d[[a]], mkt, DM)
  V <- NeweyWest(f, lag = HAC_LAG, prewhite = FALSE, adjust = TRUE)
  FITS[[a]] <- list(fit = f, V = V)
  say(sprintf("### %s      R2 = %.4f", a, summary(f)$r.squared))
  ct <- coeftest(f, vcov. = V)
  for (j in seq_len(nrow(ct)))
    say(sprintf("   %-10s %9.4f  (HAC se %7.4f)  t=%7.2f  p=%.4f",
                rownames(ct)[j], ct[j,1], ct[j,2], ct[j,3], ct[j,4]))
  cf <- coef(f); k <- length(cf)
  say("   — Cumulative slope by regime (linear-combination tests) —")
  specs <- list("Normal days b0"      = c(0, 1, 0, 0),
                "10% tail b0+b1"      = c(0, 1, 1, 0),
                "5% tail b0+b1+b2"    = c(0, 1, 1, 1))
  for (nm in names(specs)) {
    w <- specs[[nm]][seq_len(k)]
    r <- lc_test(cf, V, w)
    tag <- if (r["p"] > 0.10) "not different from 0 → weak safe haven" else
           if (r["est"] < 0)  "significantly negative → strong safe haven" else "significantly positive → not a safe haven"
    say(sprintf("   %-20s %+8.4f  (se %6.4f, t=%+6.2f, p=%.4f)  %s",
                nm, r["est"], r["se"], r["t"], r["p"],
                if (grepl("tail", nm)) tag else ""))
  }
  say("")
}

###############################################################################
rule(); say("Section 3  Honesty check: free-intercept subsample regression"); rule()
say("The interaction specification pins the tail regression line to the full-sample intercept. All tail x values lie far from 0,")
say("so this constraint is a strong extrapolation that manufactures precision. Only with a free intercept do we see the information actually contained in the tail.")
say("")
thr5 <- quantile(mkt, min(QS))
say(sprintf("%-8s %14s %14s %14s %14s", "Asset", "Interacted", "Free intercept", "Interacted se", "Free int. se"))
for (a in ASSETS) {
  f <- FITS[[a]]$fit; V <- FITS[[a]]$V
  cf <- coef(f); k <- length(cf)
  bm <- lc_test(cf, V, c(0, 1, 1, 1)[seq_len(k)])
  fs <- fit_sub(d[[a]], mkt, thr5)
  Vs <- vcovHC(fs, type = "HC3")
  say(sprintf("%-8s %+14.4f %+14.4f %14.4f %14.4f",
              a, bm["est"], coef(fs)[2], bm["se"], sqrt(Vs[2, 2])))
}
say("")
say(sprintf("Within the 5%% tail, %s ranges over [%.3f%%, %.3f%%], a span of only %.2f percentage points.",
            MKT, min(mkt[mkt <= thr5]) * 100, max(mkt[mkt <= thr5]) * 100,
            (max(mkt[mkt <= thr5]) - min(mkt[mkt <= thr5])) * 100))
say("Slope standard error ∝ σ_e /(SD(x)·√n), and all three factors are unfavourable here: this is why tail betas are hard to estimate.")

###############################################################################
rule(); say("Section 4  Stationary block-bootstrap confidence intervals and minimum detectable effect (MDE)"); rule()
say(sprintf("R = %d resamples; the quantile thresholds are recomputed in each resample (they are functions of the sample)", BOOT_R))
say("")
BOOT <- list()
for (L in BOOT_L) {
  IDX <- stationary_boot_idx(n, BOOT_R, L)
  say(sprintf("--- Expected block length L = %d ---", L))
  say(sprintf("%-8s %10s %10s %20s %10s %10s",
              "Asset", "Spec.", "Estimate", "95% CI", "Boot se", "MDE"))
  for (a in ASSETS) {
    B <- boot_stats(d[[a]], mkt, IDX, QS)
    f <- FITS[[a]]$fit; cf <- coef(f); k <- length(cf)
    pe_bm  <- sum(cf[-1])
    pe_sub <- coef(fit_sub(d[[a]], mkt, thr5))[2]
    for (j in 1:2) {
      v  <- B[, j]; v <- v[is.finite(v)]
      ci <- quantile(v, c(.025, .975)); se <- sd(v)
      pe <- if (j == 1) pe_bm else pe_sub
      say(sprintf("%-8s %10s %+10.4f  [%+7.3f, %+7.3f] %10.4f %10.3f",
                  a, c("Interacted", "Free int.")[j], pe, ci[1], ci[2], se, 2.80 * se))
    }
    if (L == 10) BOOT[[a]] <- B
  }
  say("")
}
say("MDE = (z_0.975 + z_0.80) × se ≈ 2.80 × se, i.e. the smallest effect detectable with 80% power.")
say("Interpretation: if MDE > 1, a safe haven (β<0) cannot be distinguished from an amplifier (β>1) in this sample.")

###############################################################################
rule(); say("Section 5  Leverage of single episodes (jackknife over stress episodes)"); rule()
# Identify stress episodes automatically: tail days separated by more than
# 5 trading days start a new episode.
tail_idx <- which(mkt <= thr5)
grp <- cumsum(c(1, diff(tail_idx) > 5))
eps <- split(tail_idx, grp)
say(sprintf("5%% tail: %d observations, grouped into %d independent episodes", length(tail_idx), length(eps)))
say("")
base <- sapply(ASSETS, function(a) sum(coef(FITS[[a]]$fit)[-1]))
say(sprintf("%-26s %s", "Excluded episode", paste(sprintf("%10s", ASSETS), collapse = "")))
say(sprintf("%-26s %s", "(full sample)", paste(sprintf("%+10.4f", base), collapse = "")))
lev <- list()
for (g in seq_along(eps)) {
  rows <- eps[[g]]
  lo <- max(1, min(rows) - 3); hi <- min(n, max(rows) + 3)
  keep <- setdiff(seq_len(n), lo:hi)
  mk2 <- mkt[keep]; D2 <- mk_dummies(mk2, QS)
  vals <- sapply(ASSETS, function(a) sum(coef(fit_bm(d[[a]][keep], mk2, D2))[-1]))
  lab <- sprintf("%s..%s", d$Date[min(rows)], d$Date[max(rows)])
  lev[[lab]] <- vals - base
  say(sprintf("%-26s %s", lab, paste(sprintf("%+10.4f", vals), collapse = "")))
}
say("")
mx <- sapply(lev, function(v) abs(unname(v[1])))
top <- which.max(mx); others <- sort(mx[-top], decreasing = TRUE)[1]
say(sprintf("Most influential episode for %s: %s; excluding it changes the estimate by %+.4f (next largest %.4f, a ratio of %.1f)",
            ASSETS[1], names(lev)[top], unname(lev[[top]][1]), others, mx[top] / others))
say("If one episode's influence far exceeds the combined influence of all others, the result is driven by that episode and the paper must say so.")

###############################################################################
rule(); say("Section 6  Cross-quantilogram (the appropriate tool for tail dependence)"); rule()
say("ρ(τ1,τ2) measures the dependence between the market being at or below its τ1 quantile and the asset being at or below its τ2 quantile.")
say("This is what rq(y~x, tau=) is often used for but cannot deliver: its tau is a conditional quantile of y itself.")
say("")
cqg <- function(y, x, t1, t2, k = 0) {
  qx <- quantile(x, t1); qy <- quantile(y, t2)
  px <- (x <= qx) - t1
  py <- (y <= qy) - t2
  if (k > 0) { py <- py[-seq_len(k)]; px <- px[seq_len(length(px) - k)] }
  sum(px * py) / sqrt(sum(px^2) * sum(py^2))
}
taus <- c(0.05, 0.10, 0.25, 0.50)
for (a in ASSETS) {
  say(sprintf("### %s   (rows = market quantile τ1, columns = asset quantile τ2, lag 0)", a))
  say(sprintf("   %8s %s", "", paste(sprintf("%9s", paste0("τ2=", taus)), collapse = "")))
  for (t1 in taus) {
    v <- sapply(taus, function(t2) cqg(d[[a]], mkt, t1, t2))
    say(sprintf("   τ1=%.2f %s", t1, paste(sprintf("%9.3f", v), collapse = "")))
  }
  # Bootstrap significance (only for the cell of main interest, (0.05, 0.05))
  IDX <- stationary_boot_idx(n, 1000, 10)
  bs <- apply(IDX, 2, function(i) cqg(d[[a]][i], mkt[i], 0.05, 0.05))
  ci <- quantile(bs, c(.025, .975))
  say(sprintf("   (τ1=0.05, τ2=0.05) = %.3f, block-bootstrap 95%% CI [%.3f, %.3f]",
              cqg(d[[a]], mkt, 0.05, 0.05), ci[1], ci[2]))
  say("")
}

###############################################################################
rule(); say("Section 7  Robustness: rolling quantile thresholds (addressing the generated-regressor problem)"); rule()
say("The unconditional quantiles are computed from the full sample, so strictly speaking the tail dummies are generated regressors.")
say("Here expanding-window historical quantiles are used instead (only information before day t), which avoids look-ahead bias.")
W <- 250
roll_q <- function(x, q, w) {
  sapply(seq_along(x), function(i) if (i <= w) NA_real_ else quantile(x[1:(i-1)], q))
}
rq10 <- roll_q(mkt, .10, W); rq05 <- roll_q(mkt, .05, W)
ok <- !is.na(rq10)
D3 <- data.frame(q10 = as.numeric(mkt <= rq10), q05 = as.numeric(mkt <= rq05))
say(sprintf("Usable sample %d (first %d days used for initialisation); tail observations 10%%=%d, 5%%=%d",
            sum(ok), W, sum(D3$q10[ok]), sum(D3$q05[ok])))
say("")
say(sprintf("%-8s %16s %16s", "Asset", "Unconditional q", "Rolling q"))
for (a in ASSETS) {
  f2 <- fit_bm(d[[a]][ok], mkt[ok], D3[ok, ])
  V2 <- NeweyWest(f2, lag = HAC_LAG, prewhite = FALSE, adjust = TRUE)
  r2 <- lc_test(coef(f2), V2, c(0, 1, 1, 1))
  b1 <- sum(coef(FITS[[a]]$fit)[-1])
  say(sprintf("%-8s %+16.4f %+16.4f", a, b1, r2["est"]))
}
say("If the two are close, look-ahead bias is not a material threat and a single sentence in the appendix suffices.")

###############################################################################
rule(); say("Section 8  Time-varying correlation (figures)"); rule()
ewma_cor <- function(x, y, lambda = 0.94) {
  n <- length(x); sxx <- syy <- sxy <- numeric(n)
  sxx[1] <- x[1]^2; syy[1] <- y[1]^2; sxy[1] <- x[1]*y[1]
  for (i in 2:n) {
    sxx[i] <- lambda*sxx[i-1] + (1-lambda)*x[i]^2
    syy[i] <- lambda*syy[i-1] + (1-lambda)*y[i]^2
    sxy[i] <- lambda*sxy[i-1] + (1-lambda)*x[i]*y[i]
  }
  sxy / sqrt(sxx * syy)
}
roll_cor <- function(x, y, w = 60)
  sapply(seq_along(x), function(i) if (i < w) NA else cor(x[(i-w+1):i], y[(i-w+1):i]))

pdf(file.path(OUTDIR, "fig_dynamic_correlation.pdf"), width = 10, height = 4 * length(ASSETS))
par(mfrow = c(length(ASSETS), 1), mar = c(4, 4, 3, 1))
for (a in ASSETS) {
  # Figure labels are plain ASCII: the default PDF device has no CJK or
  # extended-glyph fonts.
  ew <- ewma_cor(d[[a]], mkt); rc <- roll_cor(d[[a]], mkt, 60)
  plot(d$Date, ew, type = "l", lwd = 1.6, col = "#1f4e79", ylim = c(-1, 1),
       xlab = "", ylab = "Correlation",
       main = sprintf("%s vs %s: time-varying correlation", a, MKT))
  lines(d$Date, rc, col = "#c0504d", lwd = 1.1, lty = 2)
  abline(h = 0, col = "grey60"); abline(h = cor(d[[a]], mkt), col = "grey40", lty = 3)
  # Mark stress episodes
  for (g in seq_along(eps)) if (length(eps[[g]]) >= 2)
    abline(v = d$Date[min(eps[[g]])], col = "#f0a020", lwd = 1)
  legend("topleft", c("EWMA(0.94)", "Rolling 60d", "Full-sample mean", "Stress episode"),
         col = c("#1f4e79", "#c0504d", "grey40", "#f0a020"),
         lty = c(1, 2, 3, 1), lwd = c(1.6, 1.1, 1, 1), bty = "n", cex = 0.8)
}
invisible(dev.off())
say(sprintf("Wrote %s/fig_dynamic_correlation.pdf", OUTDIR))
say("If rmgarch is installed, EWMA can be replaced by DCC-GARCH:")
say("  spec <- multispec(replicate(2, ugarchspec(...)));")
say("  dcc  <- dccfit(dccspec(spec, dccOrder=c(1,1), distribution='mvt'), data=...)")
say("For this purpose EWMA and rolling-window correlations lead to the same conclusions and need no extra packages.")

## Bootstrap distribution figure
pdf(file.path(OUTDIR, "fig_bootstrap_tailbeta.pdf"), width = 9, height = 3.4 * length(ASSETS))
par(mfrow = c(length(ASSETS), 2), mar = c(4, 4, 3, 1))
for (a in ASSETS) {
  B <- BOOT[[a]]
  for (j in 1:2) {
    v <- B[, j]; v <- v[is.finite(v)]
    hist(v, breaks = 60, col = "#dce6f1", border = "white",
         main = sprintf("%s - %s", a, c("interacted spec", "free intercept")[j]),
         xlab = "5% tail beta")
    abline(v = quantile(v, c(.025, .975)), col = "#c0504d", lty = 2)
    abline(v = 0, col = "black"); abline(v = 1, col = "grey50", lty = 3)
    legend("topright", c("95% CI", "beta=0", "beta=1"),
           col = c("#c0504d", "black", "grey50"), lty = c(2, 1, 3), bty = "n", cex = 0.75)
  }
}
invisible(dev.off())
say(sprintf("Wrote %s/fig_bootstrap_tailbeta.pdf", OUTDIR))

###############################################################################
rule(); say("Section 9  Suggested wording for the response to reviewers"); rule()
a1 <- ASSETS[1]
B <- BOOT[[a1]]; se_sub <- sd(B[is.finite(B[, 2]), 2])
say(sprintf("\"In this sample (n = %d; 5%% tail: %d observations, grouped into %d independent stress episodes),",
            n, length(tail_idx), length(eps)))
say(sprintf(" the minimum detectable effect for the %s tail beta is %.2f (80%% power, 5%% level).", a1, 2.80 * se_sub))
say(" Effects smaller than this cannot be identified with this research design. We therefore state")
say(" directional conclusions only and report intervals, rather than point estimates, for magnitudes.\"")
say("")
say("This turns 'insufficient sample size' from a passive weakness into an explicit, quantified statement about the research design.")

rule(); say("Done. Results written to ", file.path(OUTDIR, "results.txt"))
close(con)
