"""
04_var_es_backtest.py — Rolling out-of-sample VaR/ES forecasts and full backtest

  1. Portfolios use FIXED weights, not optimized weights

  2. Rolling forecasts are strictly out-of-sample: parameters are re-estimated
     every REFIT days, and on the days in between a fixed parameter set advances
     the GARCH recursion, so sigma^2 on day t uses only information up to t-1.
     This is the same as rugarch::ugarchroll.

  3. ES uses the correct definition for the standardized t:
         ES = Mu + Sigma * ES_z(alpha), where
         ES_z(a) = -[ f_t(q;nu) * (nu + q^2) / ((nu-1) a) ] / sqrt(nu/(nu-2))
     Note that nu changes with each refit, so it must be substituted ROW BY ROW;
     the full-sample mean must not be used.

  4. Independence test LR_ind = LR_cc - LR_uc, 1 degree of freedom. All three
     statistics are reported.

  5. Model comparison uses the Giacomini-White conditional predictive ability test.

  6. ES backtesting includes Acerbi-Szekely Z1/Z2 in addition to McNeil-Frey.

  7. A power table is reported.

Dependencies: numpy, pandas, scipy, arch
Usage:
    python 04_var_es_backtest.py --data data/processed/returns__20260921.csv
    python 04_var_es_backtest.py --selftest
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats, integrate, optimize

warnings.filterwarnings("ignore")

try:
    from arch.univariate import ConstantMean, GARCH, StudentsT, Normal
except ImportError:
    sys.exit("Missing the arch package. Please run: pip install arch")

# ------------------------------------------------------------------ Config ----
ALPHAS = (0.01, 0.05)
N_START = 250          # initial estimation window
REFIT = 20             # re-estimate every 20 days
WINDOW = "moving"      # "moving" or "recursive"
HS_WIN = 250           # historical simulation window
EWMA_LAMBDA = 0.94     # RiskMetrics
SEED = 20260921
HAC_LAG = 5

# The portfolio definitions match the single-difference design in Section 2.3 of
# the paper: equal weights 1/3, replacing only one asset.
# Also add a [volatility-matched] group: IBIT annualized vol 50.3%, GLD 21.9%,
# a ratio of 2.29. Under equal-weight comparison, "the IBIT portfolio's tail risk
# is about 2x" is partly driven by the volatility difference itself.
# Weights are recomputed from the data, not hard-coded.
PORTFOLIOS_SPEC = {
    "IBIT portfolio":     {"IBIT": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3},
    "GLD portfolio":      {"GLD": 1 / 3, "SPY": 1 / 3, "TLT": 1 / 3},
    "IBIT vol-matched":   None,          # computed at runtime as (1/3)*sd(GLD)/sd(IBIT)
    "60/40 benchmark":    {"SPY": 0.60, "TLT": 0.40},
}

class ModelSpecError(RuntimeError):
    """Model specification does not match the recursion. This is a code error and
    must be raised, not swallowed as a fitting failure."""


CHECKS: list[str] = []
RECURSION_DEV: list[float] = []
LINES: list[str] = []


def w(s: str = "") -> None:
    print(s)
    LINES.append(s)


def rule(ch: str = "=") -> None:
    w(ch * 78)


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    CHECKS.append(f"[{tag}] {label} {detail}")
    w(f"   CHECK [{tag}] {label} {detail}")
    return ok


def md(rows, head) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


# ====================================================== Distributions: VaR / ES ====
def var_z_t(a: float, nu: float) -> float:
    """alpha quantile (negative) of the unit-variance standardized t.
    Consistent with arch StudentsT.ppf."""
    return float(stats.t.ppf(a, nu) / math.sqrt(nu / (nu - 2.0)))


def es_z_t(a: float, nu: float) -> float:
    """ES (negative) of the unit-variance standardized t. Closed form."""
    q = stats.t.ppf(a, nu)
    es_raw = -stats.t.pdf(q, nu) * (nu + q * q) / ((nu - 1.0) * a)
    return float(es_raw / math.sqrt(nu / (nu - 2.0)))


def var_z_n(a: float) -> float:
    return float(stats.norm.ppf(a))


def es_z_n(a: float) -> float:
    return float(-stats.norm.pdf(stats.norm.ppf(a)) / a)


# ================================================== Rolling GARCH forecaster ====
def roll_garch(r: np.ndarray, dist: str, n_start: int, refit: int,
               window: str = "moving") -> dict:
    """
    Strictly out-of-sample rolling one-step forecasts.

    r      : return series (decimals, not percent)
    dist   : "normal" or "t"
    returns: {"mu","sigma","nu","refits"}, length n - n_start, corresponding to r[n_start:]

    Key point: sigma_t on day t uses only information up to t-1. On days without
    re-estimation, a fixed parameter set advances the GARCH recursion, the same as
    rugarch::ugarchroll.
    """
    n = len(r)
    x = r * 100.0                      # arch recommends estimating on a percent scale for numerical stability
    n_out = n - n_start
    mu_f = np.full(n_out, np.nan)
    sd_f = np.full(n_out, np.nan)
    nu_f = np.full(n_out, np.nan)
    n_refit = 0
    n_fail = 0                         # number of times an exception was raised or convergence failed and old params were reused

    params = None
    sig2_prev = None                   # sigma^2_{t-1} (percent scale)
    eps_prev = None                    # eps_{t-1}

    for k in range(n_out):
        t = n_start + k                # the day to forecast
        if k % refit == 0:
            lo = (t - n_start) if window == "moving" else 0
            y = x[lo:t]                # uses only up to t-1
            am = ConstantMean(y)
            # All keywords must be given. arch's signature is GARCH(p, o, q), where
            # the second positional parameter is the [asymmetry order o], not q --
            # GARCH(1,1) would be built as GJR-GARCH(1,1,1), adding a gamma[1],
            # while the recursion below has no gamma, so predicted volatility would
            # barely respond to returns. This script previously pinned alpha at 0
            # because of this.
            am.volatility = GARCH(p=1, o=0, q=1)
            am.distribution = StudentsT() if dist == "t" else Normal()
            try:
                res = am.fit(disp="off", show_warning=False,
                             options={"maxiter": 2000})
                p = res.params
                expect = {"mu", "omega", "alpha[1]", "beta[1]"} | (
                    {"nu"} if dist == "t" else set())
                if set(p.index) != expect:
                    raise ModelSpecError(
                        f"Parameter set does not match GARCH(1,1): {sorted(p.index)}. "
                        "The recursion only implements omega/alpha/beta, so the model "
                        "must be a pure GARCH(1,1).")
                # show_warning=False silently ignores non-convergence, so check explicitly
                if res.convergence_flag != 0:
                    raise ArithmeticError(f"Optimization did not converge (flag={res.convergence_flag})")
                params = dict(mu=float(p["mu"]), omega=float(p["omega"]),
                              alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
                              nu=float(p["nu"]) if dist == "t" else np.nan)
                # Recursion self-check: reproduce arch's own conditional volatility
                # with this script's recursion. This is exactly the check that
                # catches the GJR misspecification above.
                cvv = np.asarray(res.conditional_volatility, dtype=float)
                s2c = float(cvv[0] ** 2)
                dev = 0.0
                for j in range(1, len(y)):
                    s2c = (params["omega"] + params["alpha"] * (y[j - 1] - params["mu"]) ** 2
                           + params["beta"] * s2c)
                    dev = max(dev, abs(math.sqrt(s2c) - cvv[j]))
                RECURSION_DEV.append(dev)
                # Use the fitted end-of-window conditional variance and residual as
                # the recursion starting point
                cv = np.asarray(res.conditional_volatility, dtype=float)
                sig2_prev = float(cv[-1] ** 2)
                eps_prev = float(y[-1] - params["mu"])
                n_refit += 1
            except ModelSpecError:
                raise
            except Exception:
                n_fail += 1            # on fit failure or non-convergence, reuse the previous parameters

        if params is None:
            continue

        # One-step forecast: sigma^2_t = omega + alpha*eps_{t-1}^2 + beta*sigma^2_{t-1}
        sig2_t = (params["omega"] + params["alpha"] * eps_prev ** 2
                  + params["beta"] * sig2_prev)
        mu_f[k] = params["mu"] / 100.0
        sd_f[k] = math.sqrt(sig2_t) / 100.0
        nu_f[k] = params["nu"]

        # Advance the recursion, incorporating day t's realized value, for t+1
        sig2_prev = sig2_t
        eps_prev = float(x[t] - params["mu"])

    return {"mu": mu_f, "sigma": sd_f, "nu": nu_f, "refits": n_refit, "fails": n_fail}


def forecast_var_es(fc: dict, dist: str, a: float):
    """Generate VaR/ES row by row from (mu, sigma, nu) (return convention, negative)."""
    mu, sd, nu = fc["mu"], fc["sigma"], fc["nu"]
    n = len(mu)
    v = np.full(n, np.nan)
    e = np.full(n, np.nan)
    if dist == "t":
        for i in range(n):
            if np.isnan(sd[i]) or np.isnan(nu[i]):
                continue
            v[i] = mu[i] + sd[i] * var_z_t(a, nu[i])
            e[i] = mu[i] + sd[i] * es_z_t(a, nu[i])
    else:
        zv, ze = var_z_n(a), es_z_n(a)
        v = mu + sd * zv
        e = mu + sd * ze
    return v, e


def roll_hs(r: np.ndarray, n_start: int, a: float, win: int):
    """Historical simulation: rolling-window empirical quantile and tail mean"""
    n = len(r)
    v = np.full(n - n_start, np.nan)
    e = np.full(n - n_start, np.nan)
    for k in range(n - n_start):
        t = n_start + k
        s = r[max(0, t - win):t]
        q = np.quantile(s, a)
        tail = s[s <= q]
        v[k] = q
        e[k] = tail.mean() if len(tail) else q
    return v, e


def roll_ewma(r: np.ndarray, n_start: int, a: float, lam: float):
    """RiskMetrics EWMA + normal"""
    n = len(r)
    s2 = np.var(r[:n_start], ddof=1)
    v = np.full(n - n_start, np.nan)
    e = np.full(n - n_start, np.nan)
    # First advance the EWMA up to n_start-1
    for t in range(1, n_start):
        s2 = lam * s2 + (1 - lam) * r[t - 1] ** 2
    zv, ze = var_z_n(a), es_z_n(a)
    for k in range(n - n_start):
        t = n_start + k
        s2 = lam * s2 + (1 - lam) * r[t - 1] ** 2      # uses only up to t-1
        sd = math.sqrt(s2)
        v[k] = sd * zv
        e[k] = sd * ze
    return v, e


# ========================================================== Coverage tests ====
def lr_uc(x: int, n: int, a: float) -> float:
    """Kupiec unconditional coverage"""
    if x == 0:
        return -2.0 * (n * math.log(1 - a))
    pi = x / n
    ll0 = (n - x) * math.log(1 - a) + x * math.log(a)
    ll1 = (n - x) * math.log(1 - pi) + x * math.log(pi)
    return -2.0 * (ll0 - ll1)


def lr_ind(br: np.ndarray) -> float:
    """Christoffersen independence (first-order Markov chain)."""
    b = br.astype(int)
    n00 = int(np.sum((b[:-1] == 0) & (b[1:] == 0)))
    n01 = int(np.sum((b[:-1] == 0) & (b[1:] == 1)))
    n10 = int(np.sum((b[:-1] == 1) & (b[1:] == 0)))
    n11 = int(np.sum((b[:-1] == 1) & (b[1:] == 1)))
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return 0.0
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p in (0.0, 1.0) or p01 in (0.0,) or p11 in (0.0,):
        # In degenerate cases independence carries no information
        if p11 == 0.0 and p01 > 0:
            ll1 = n00 * math.log(1 - p01) + n01 * math.log(p01)
        else:
            return 0.0
    else:
        ll1 = (n00 * math.log(1 - p01) + n01 * math.log(p01)
               + n10 * math.log(1 - p11) + n11 * math.log(p11))
    ll0 = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    return max(0.0, -2.0 * (ll0 - ll1))


# ====================================================== ES backtest statistics ====
def as_z1(r: np.ndarray, v: np.ndarray, e: np.ndarray) -> float:
    """
    Acerbi-Szekely Z1. Expected value 0 under a correct model; negative means
    risk is underestimated.
    This script's VaR/ES are in return convention (negative), so use r/(-ES).
    """
    br = r < v
    if br.sum() == 0:
        return np.nan
    return float(np.mean(r[br] / (-e[br])) + 1.0)


def as_z2(r: np.ndarray, v: np.ndarray, e: np.ndarray, a: float) -> float:
    """Acerbi-Szekely Z2."""
    br = r < v
    n = len(r)
    if n == 0:
        return np.nan
    return float(np.sum(r[br] / (-e[br])) / (n * a) + 1.0)


def as_pvalue(z_obs: float, mu: np.ndarray, sd: np.ndarray, nu: np.ndarray,
              dist: str, a: float, which: str, rng, B: int = 2000) -> float:
    """
    p-value for the AS test: simulated under H0 (correct predictive distribution).
    One-sided (left tail): the more negative z, the more risk is underestimated.
    """
    ok = ~np.isnan(sd)
    m, s = mu[ok], sd[ok]
    nn = nu[ok] if dist == "t" else None
    n = len(m)
    if n == 0 or np.isnan(z_obs):
        return np.nan
    # VaR/ES forecasts do not change with resampling, so they must be precomputed
    # outside the loop; otherwise each resample would need 2n scipy calls
    # (with B=1000, n=357 that is 710k calls, too slow to finish).
    if dist == "t":
        vv = m + s * np.array([var_z_t(a, x) for x in nn])
        ee = m + s * np.array([es_z_t(a, x) for x in nn])
    else:
        vv = m + s * var_z_n(a)
        ee = m + s * es_z_n(a)
    zs = np.empty(B)
    for b in range(B):
        if dist == "t":
            u = rng.standard_t(nn) / np.sqrt(nn / (nn - 2.0))
        else:
            u = rng.standard_normal(n)
        rr = m + s * u
        zs[b] = as_z1(rr, vv, ee) if which == "Z1" else as_z2(rr, vv, ee, a)
    zs = zs[~np.isnan(zs)]
    if len(zs) == 0:
        return np.nan
    return float((zs <= z_obs).mean())


def mcneil_frey(r: np.ndarray, v: np.ndarray, e: np.ndarray, sd: np.ndarray,
                rng, B: int = 5000) -> tuple:
    """
    McNeil-Frey: the mean of the standardized excess residual (r - ES)/sigma on
    breach days should be 0. Bootstrap one-sided test (mean significantly negative
    = risk underestimated).
    """
    br = (r < v) & (~np.isnan(e))
    if br.sum() < 2:
        return np.nan, np.nan
    d = (r[br] - e[br]) / sd[br]
    m = float(d.mean())
    dc = d - m
    bs = np.array([dc[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    return m, float((bs <= m).mean())


# ============================================================ Loss functions ====
def pinball(r: np.ndarray, v: np.ndarray, a: float) -> np.ndarray:
    """Quantile loss (VaR)."""
    return (a - (r < v).astype(float)) * (r - v)


def fz0(r: np.ndarray, v: np.ndarray, e: np.ndarray, a: float) -> np.ndarray:
    """
    FZ0 loss (Fissler-Ziegel / Nolde-Ziegel 2017), jointly elicitable for (VaR, ES).
    Uniformly written in [positive loss] convention: L = -r, V = -VaR, E = -ES,
    all three positive magnitudes.
        FZ0 = (1/(a*E)) * 1{L>V} * (L-V) + V/E + log(E) - 1
    """
    L, V, E = -r, -v, -e
    E = np.where(E <= 1e-12, np.nan, E)
    ind = (L > V).astype(float)
    return (1.0 / (a * E)) * ind * (L - V) + V / E + np.log(E) - 1.0


def hac_vcov(X: np.ndarray, u: np.ndarray, lag: int) -> np.ndarray:
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    S = (X * u[:, None]).T @ (X * u[:, None])
    for l in range(1, lag + 1):
        wgt = 1.0 - l / (lag + 1.0)
        A = (X[l:] * u[l:, None]).T @ (X[:-l] * u[:-l, None])
        S += wgt * (A + A.T)
    return XtXi @ S @ XtXi


def gw_test(d: np.ndarray, q: int = 2, lag: int = HAC_LAG) -> tuple:
    """
    Giacomini-White conditional predictive ability test.
    d_t = L1_t - L2_t, regress on h_t = (1, d_{t-1}), Wald test that all
    coefficients are 0. q=2 tests [conditional] predictive ability: power against
    a constant difference is naturally lower than DM, by design.
    """
    # Pair (d_t, d_{t-1}) first and then drop missing values; do not drop NaNs
    # first, which would treat two non-adjacent days as lagged
    y, lag1 = d[1:], d[:-1]
    keep = np.isfinite(y) & np.isfinite(lag1)
    y, lag1 = y[keep], lag1[keep]
    if len(y) < 20:
        return np.nan, np.nan
    X = np.column_stack([np.ones(len(y)), lag1])[:, :q]
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    u = y - X @ b
    V = hac_vcov(X, u, lag)
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, float(stats.chi2.sf(stat, q))


# ============================================================== Self-test ====
def selftest() -> int:
    print("=" * 78)
    print("Self-test")
    print("=" * 78)
    ok_all = True

    # 1. ES_z closed form vs numerical integration
    worst = 0.0
    for nu in (3.5, 5.0, 8.0, 30.0):
        for a in (0.01, 0.05, 0.10):
            num = integrate.quad(lambda p: var_z_t(p, nu), 0, a)[0] / a
            worst = max(worst, abs(num - es_z_t(a, nu)))
    ok_all &= check("ES_z(t) closed form vs numerical integration", worst < 1e-8, f"maxdiff={worst:.2e}")

    # 2. arch's StudentsT agrees with this script's var_z_t (convention consistency)
    d = StudentsT()
    worst = 0.0
    for nu in (4.0, 6.0, 12.0):
        q = d.ppf([0.01, 0.05], np.array([nu]))
        worst = max(worst, max(abs(q[0] - var_z_t(0.01, nu)),
                               abs(q[1] - var_z_t(0.05, nu))))
    ok_all &= check("arch StudentsT.ppf == var_z_t (same convention as rugarch \"std\")",
                    worst < 1e-10, f"maxdiff={worst:.2e}")

    # 3. Known normal ES value
    ok_all &= check("ES_z(normal, 5%) = -2.0627",
                    abs(es_z_n(0.05) + 2.062713) < 1e-5, f"{es_z_n(0.05):.6f}")

    # 4. As nu -> inf, t converges to normal
    ok_all &= check("nu=1e6: ES_z(t) -> ES_z(normal)",
                    abs(es_z_t(0.05, 1e6) - es_z_n(0.05)) < 1e-4,
                    f"{es_z_t(0.05, 1e6):.6f} vs {es_z_n(0.05):.6f}")

    # 5. Kupiec: statistic should be 0 when hit rate equals nominal coverage
    ok_all &= check("Kupiec: LR_uc = 0 when x/n = alpha",
                    abs(lr_uc(25, 500, 0.05)) < 1e-9, f"{lr_uc(25,500,0.05):.2e}")

    # 6. FZ0 is minimized at the true (VaR, ES)
    rng = np.random.default_rng(0)
    a = 0.05
    smp = rng.standard_normal(400000)
    tv, te = var_z_n(a), es_z_n(a)

    def obj(p):
        return np.nanmean(fz0(smp, np.full_like(smp, p[0]), np.full_like(smp, p[1]), a))

    res = optimize.minimize(obj, x0=[tv - 0.4, te - 0.4], method="Nelder-Mead",
                            options={"xatol": 1e-4, "fatol": 1e-8})
    err = max(abs(res.x[0] - tv), abs(res.x[1] - te))
    ok_all &= check("FZ0 is minimized at the true (VaR, ES)",
                    err < 0.05, f"argmin=({res.x[0]:.3f},{res.x[1]:.3f}) truth=({tv:.3f},{te:.3f})")

    # 7. AS Z1/Z2 should be close to 0 under a correct model
    n = 20000
    sd = np.full(n, 0.01)
    mu = np.zeros(n)
    r = mu + sd * rng.standard_normal(n)
    v = mu + sd * var_z_n(a)
    e = mu + sd * es_z_n(a)
    z1, z2 = as_z1(r, v, e), as_z2(r, v, e, a)
    ok_all &= check("AS Z1 ~ 0 under a correct model", abs(z1) < 0.05, f"Z1={z1:+.4f}")
    ok_all &= check("AS Z2 ~ 0 under a correct model", abs(z2) < 0.05, f"Z2={z2:+.4f}")

    # 8. AS should give a significantly negative value for a model that underestimates risk
    e_bad = e * 0.6                     # ES magnitude underestimated by 40%
    z1b = as_z1(r, v, e_bad)
    ok_all &= check("AS Z1 significantly negative for a model underestimating ES", z1b < -0.3, f"Z1={z1b:+.4f}")

    # 9. GW nominal level (should not over-reject under iid no difference)
    rej = 0
    for _ in range(300):
        dd = rng.standard_normal(357) * 0.01
        _, p = gw_test(dd)
        rej += int(p < 0.05)
    ok_all &= check("GW nominal level ~5%", 0.01 <= rej / 300 <= 0.12, f"rejection rate={rej/300:.3f}")

    # 10. Out-of-sample property of the rolling GARCH: feed in known GARCH data,
    #     parameters should be recoverable
    n = 1500
    om, al, be = 0.02, 0.08, 0.90
    s2 = om / (1 - al - be)
    x = np.empty(n)
    for i in range(n):
        z = rng.standard_normal()
        x[i] = math.sqrt(s2) * z
        s2 = om + al * x[i] ** 2 + be * s2
    fc = roll_garch(x / 100.0, "normal", n_start=1000, refit=500)
    ok_all &= check("Rolling GARCH runs without NaNs",
                    np.isfinite(fc["sigma"]).all(), f"refits={fc['refits']}")
    mx = max(RECURSION_DEV) if RECURSION_DEV else float("inf")
    ok_all &= check("This script's recursion == arch's conditional_volatility (catches GJR misspecification)",
                    mx < 1e-8, f"maxdev={mx:.2e}")
    # Nominal coverage should be roughly correct
    v = fc["mu"] + fc["sigma"] * var_z_n(0.05)
    cov = float((x[1000:] / 100.0 < v).mean())
    ok_all &= check("Rolling GARCH 5% coverage close to nominal", 0.02 <= cov <= 0.10, f"{cov:.3f}")

    print()
    print("Self-test all passed." if ok_all else "Self-test has failures, see above.")
    return 0 if ok_all else 1


# ================================================================ Main ====
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/returns__20260921.csv")
    ap.add_argument("--outdir", default="out/04_var_es")
    ap.add_argument("--nstart", type=int, default=N_START)
    ap.add_argument("--refit", type=int, default=REFIT)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    rng = np.random.default_rng(SEED)
    os.makedirs(a.outdir, exist_ok=True)

    rule(); w("Section 0  Data and portfolios"); rule()
    d = pd.read_csv(a.data, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    n = len(d)
    w(f"{n} trading days, {d.Date.iloc[0].date()} -> {d.Date.iloc[-1].date()}")
    w(f"Data file: {a.data}")

    # Determine weights using only the initial estimation window (first nstart days);
    # otherwise weights contain out-of-sample information, contradicting the
    # "strictly out-of-sample" requirement. The full-sample version is only reported
    # as a comparison.
    ins = slice(0, a.nstart)
    vm = (1 / 3) * d.GLD.iloc[ins].std() / d.IBIT.iloc[ins].std()
    vm_full = (1 / 3) * d.GLD.std() / d.IBIT.std()
    PORTFOLIOS_SPEC["IBIT vol-matched"] = {
        "IBIT": vm, "SPY": (1 - vm) / 2, "TLT": (1 - vm) / 2}
    w(f"Vol-matched weight recomputed from the first {a.nstart} days (initial estimation window): (1/3)*sd(GLD)/sd(IBIT) = {vm:.5f}")
    w(f"  Comparison: the full-sample (including out-of-sample) value would be {vm_full:.5f}, not used here to avoid look-ahead.")

    P = {}
    rows = []
    for nm, wt in PORTFOLIOS_SPEC.items():
        r = sum(d[k].values * v for k, v in wt.items())
        P[nm] = r
        rows.append([nm, " + ".join(f"{k} {v:.3f}" for k, v in wt.items()),
                     f"{r.mean()*252*100:.2f}%", f"{r.std()*math.sqrt(252)*100:.2f}%"])
    w("")
    w(md(rows, ["Portfolio", "Weights", "Ann. return (arithmetic)", "Ann. vol"]))

    n_out = n - a.nstart
    w(f"Rolling setup: n_start={a.nstart}, refit.every={a.refit}, window={WINDOW}")
    w(f"Number of out-of-sample forecasts = {n_out}")
    w("sigma_t on day t uses only information up to t-1; on days without re-estimation, "
      "a fixed parameter set advances the recursion.")

    # ---------------------------------------------------- Convention self-check ----
    w("")
    rule("-"); w("Convention self-check (aligned with rugarch \"std\")"); rule("-")
    dd = StudentsT()
    worst = max(abs(dd.ppf([0.01], np.array([nu]))[0] - var_z_t(0.01, nu))
                for nu in (4.0, 6.0, 12.0))
    check("arch StudentsT.ppf == var_z_t", worst < 1e-10, f"maxdiff={worst:.2e}")
    worst = max(abs(integrate.quad(lambda p: var_z_t(p, nu), 0, al)[0] / al - es_z_t(al, nu))
                for nu in (4.0, 8.0) for al in ALPHAS)
    check("ES_z closed form vs numerical integration", worst < 1e-8, f"maxdiff={worst:.2e}")

    # ------------------------------------------------------ Rolling forecasts ----
    w("")
    rule(); w("Section 1  Rolling out-of-sample forecasts"); rule()
    FC = {}
    for nm, r in P.items():
        for dist in ("t", "normal"):
            fc = roll_garch(r, dist, a.nstart, a.refit, WINDOW)
            FC[(nm, f"GARCH-{dist}")] = fc
            nu_txt = ""
            if dist == "t":
                nus = fc["nu"][~np.isnan(fc["nu"])]
                nu_txt = f", Shape nu range [{nus.min():.2f}, {nus.max():.2f}]"
            fail_txt = f" (failed/did not converge {fc['fails']} times, reused old parameters)" if fc["fails"] else ""
            w(f"  {nm:14s} GARCH-{dist:6s} refit {fc['refits']} times{fail_txt}{nu_txt}")
    w("")
    w("Note: Shape nu changes with each refit; ES substitutes the corresponding nu ROW BY ROW, "
      "not the full-sample mean.")
    mx = max(RECURSION_DEV) if RECURSION_DEV else float("inf")
    check(f"This script's recursion == arch's conditional_volatility (all {len(RECURSION_DEV)} fits)",
          mx < 1e-8, f"maxdev={mx:.2e}")
    n_fail = sum(fc["fails"] for fc in FC.values())
    check("All refits converged", n_fail == 0, f"{n_fail} failures")

    # ------------------------------------------- Volatility calibration diagnostics (key) ----
    w("")
    rule("-"); w("Volatility calibration diagnostics: predicted vs realized volatility"); rule("-")
    w("If the ratios below are markedly below 1, GARCH systematically underestimates")
    w("volatility, which would directly cause excess breaches at all VaR levels -- confirm")
    w("this is a model property rather than a code error first.")
    w("")
    dates_out = d.Date.values[a.nstart:]
    crisis = ((dates_out >= np.datetime64("2025-04-01"))
              & (dates_out <= np.datetime64("2025-04-30")))
    rows = []
    for nm, r in P.items():
        ro = r[a.nstart:]
        real = ro.std(ddof=1) * math.sqrt(252) * 100
        for dist in ("t", "normal"):
            fc = FC[(nm, f"GARCH-{dist}")]
            sd = fc["sigma"]
            ok = ~np.isnan(sd)
            # Use RMS rather than mean: the correct aggregation of volatility is the second moment
            pred = math.sqrt(np.mean(sd[ok] ** 2)) * math.sqrt(252) * 100
            pred_c = (math.sqrt(np.mean(sd[ok & crisis] ** 2)) * math.sqrt(252) * 100
                      if (ok & crisis).sum() else np.nan)
            real_c = (ro[crisis].std(ddof=1) * math.sqrt(252) * 100
                      if crisis.sum() > 1 else np.nan)
            rows.append([nm, f"GARCH-{dist}", f"{pred:.2f}%", f"{real:.2f}%",
                         f"{pred/real:.3f}", f"{pred_c:.2f}%", f"{real_c:.2f}%",
                         f"{pred_c/real_c:.3f}"])
    w(md(rows, ["Portfolio", "Model", "Predicted vol (RMS)", "Realized vol", "Ratio",
                "Predicted (2025-04)", "Realized (2025-04)", "Ratio"]))
    w(f"Out-of-sample period {pd.Timestamp(dates_out[0]).date()} -> "
      f"{pd.Timestamp(dates_out[-1]).date()}, of which the 2025-04 tariff shock accounts for {crisis.sum()} days.")

    ratios = [float(x[4]) for x in rows]
    check("GARCH predicted and realized volatility are of the same order (0.7~1.3)",
          all(0.7 <= x <= 1.3 for x in ratios),
          f"ratio range [{min(ratios):.3f}, {max(ratios):.3f}]")

    # ---------------------------------------------------- Backtest main table ----
    w("")
    rule(); w("Section 2  VaR backtest: Kupiec UC / Christoffersen CC / independence"); rule()

    results = {}
    for al in ALPHAS:
        rows = []
        for nm, r in P.items():
            ro = r[a.nstart:]
            models = {}
            for dist in ("t", "normal"):
                fc = FC[(nm, f"GARCH-{dist}")]
                v, e = forecast_var_es(fc, dist, al)
                models[f"GARCH-{dist}"] = (v, e, fc["sigma"], fc["nu"])
            v, e = roll_hs(r, a.nstart, al, HS_WIN)
            models["Historical simulation"] = (v, e, None, None)
            v, e = roll_ewma(r, a.nstart, al, EWMA_LAMBDA)
            models["EWMA-normal"] = (v, e, None, None)

            for mn, (v, e, sd, nu) in models.items():
                ok = ~np.isnan(v)
                rr, vv = ro[ok], v[ok]
                br = rr < vv
                x, nn = int(br.sum()), len(rr)
                uc = lr_uc(x, nn, al)
                ind = lr_ind(br)
                cc = uc + ind
                results[(al, nm, mn)] = dict(
                    r=rr, v=vv, e=e[ok],
                    # Full-length (aligned day by day with r[nstart:]) versions,
                    # used for model comparisons
                    r_full=ro, v_full=v, e_full=e,
                    sd=(sd[ok] if sd is not None else None),
                    nu=(nu[ok] if nu is not None else None),
                    mu=(FC[(nm, mn)]["mu"][ok] if mn.startswith("GARCH") else None),
                    x=x, n=nn)
                rows.append([nm, mn, f"{x}/{nn}", f"{x/nn*100:.2f}%",
                             f"{uc:.2f}", f"{stats.chi2.sf(uc,1):.3f}",
                             f"{ind:.2f}", f"{stats.chi2.sf(ind,1):.3f}",
                             f"{cc:.2f}", f"{stats.chi2.sf(cc,2):.3f}"])
        w(f"### alpha = {al:.0%} (nominal breaches {al*len(P[list(P)[0]][a.nstart:]):.1f})")
        w(md(rows, ["Portfolio", "Model", "Breaches", "Coverage", "LR_uc", "p",
                    "LR_ind", "p", "LR_cc", "p"]))

    # ------------------------------------------------------ ES backtest ----
    rule(); w("Section 3  ES backtest: Acerbi-Szekely Z1/Z2 and McNeil-Frey"); rule()
    w("Sign convention: this script's VaR/ES are in return convention (negative).")
    w("Z1/Z2 have expected value 0 under a correct model; significantly negative =")
    w("underestimating tail risk. p-values come from the simulated distribution under")
    w("H0 (one-sided left tail).")
    w("")
    for al in ALPHAS:
        rows = []
        for nm in P:
            for mn in ("GARCH-t", "GARCH-normal", "Historical simulation", "EWMA-normal"):
                R = results[(al, nm, mn)]
                z1 = as_z1(R["r"], R["v"], R["e"])
                z2 = as_z2(R["r"], R["v"], R["e"], al)
                if mn.startswith("GARCH"):
                    dist = "t" if mn.endswith("t") else "normal"
                    p1 = as_pvalue(z1, R["mu"], R["sd"], R["nu"], dist, al, "Z1", rng, 1000)
                    p2 = as_pvalue(z2, R["mu"], R["sd"], R["nu"], dist, al, "Z2", rng, 1000)
                    mf, pmf = mcneil_frey(R["r"], R["v"], R["e"], R["sd"], rng)
                    rows.append([nm, mn, f"{z1:+.4f}", f"{p1:.3f}",
                                 f"{z2:+.4f}", f"{p2:.3f}", f"{mf:+.4f}", f"{pmf:.3f}"])
                else:
                    rows.append([nm, mn, f"{z1:+.4f}", "—", f"{z2:+.4f}", "—", "—", "—"])
        w(f"### alpha = {al:.0%}")
        w(md(rows, ["Portfolio", "Model", "Z1", "p", "Z2", "p", "McNeil-Frey mean", "p"]))
    w("Historical simulation and EWMA have no parametric predictive distribution, so the")
    w("AS H0 simulation does not apply; p-values are not reported.")

    # -------------------------------------------------- Model comparison GW ----
    w("")
    rule(); w("Section 4  Model comparison: Giacomini-White conditional predictive ability test"); rule()
    w("DM is not appropriate: DM assumes parameters are known, whereas here parameters")
    w("are estimated on a rolling window.")
    w("q=2 (constant + lagged loss difference) tests [conditional] predictive ability;")
    w("power against a constant difference is naturally lower than DM -- this is by design,")
    w("not a bug.")
    w("")
    for al in ALPHAS:
        rows = []
        for nm in P:
            base = results[(al, nm, "GARCH-t")]
            for mn in ("GARCH-normal", "Historical simulation", "EWMA-normal"):
                o = results[(al, nm, mn)]
                # Align day by day: use full-length arrays; if either model lacks a
                # forecast on a given day, that day is NaN and is dropped pairwise by
                # gw_test. Do not drop NaNs separately and then truncate -- that
                # would misalign.
                rf = base["r_full"]
                d_fz = fz0(rf, base["v_full"], base["e_full"], al) - \
                       fz0(rf, o["v_full"], o["e_full"], al)
                d_pb = pinball(rf, base["v_full"], al) - pinball(rf, o["v_full"], al)
                s1, p1 = gw_test(d_fz)
                s2, p2 = gw_test(d_pb)
                mf = np.nanmean(d_fz)
                rows.append([nm, f"GARCH-t vs {mn}",
                             f"{mf:+.4f}", "GARCH-t better" if mf < 0 else f"{mn} better",
                             f"{s1:.2f}", f"{p1:.3f}", f"{s2:.2f}", f"{p2:.3f}"])
        w(f"### alpha = {al:.0%}")
        w(md(rows, ["Portfolio", "Comparison", "FZ0 mean loss diff", "Direction",
                    "GW(FZ0)", "p", "GW(pinball)", "p"]))
    w("FZ0 is jointly elicitable for (VaR, ES) (Fissler-Ziegel) and is the correct loss")
    w("function for comparing ES forecasts; pinball is elicitable only for VaR. Reporting")
    w("both shows whether conclusions depend on the choice of loss function.")

    # ---------------------------------------------------------- Power ----
    w("")
    rule(); w("Section 5  Power table: which level is usable at this sample size"); rule()
    rows = []
    B = 20000
    MULTS = (1.4, 1.6, 2.0, 3.0)
    power = {}
    for al in ALPHAS:
        nn = n_out
        for mult in MULTS:
            true_cov = al * mult
            x = rng.binomial(nn, true_cov, B)
            u, cnt = np.unique(x, return_counts=True)     # compute only for counts that occurred
            rej_u = np.array([stats.chi2.sf(lr_uc(int(xx), nn, al), 1) < 0.05 for xx in u])
            rej = float((cnt[rej_u].sum()) / B)
            power[(al, mult)] = rej
            rows.append([f"{al:.0%}", f"{al*nn:.1f}", f"{true_cov:.2%}",
                         f"{true_cov*nn:.1f}", f"{rej:.0%}"])
    w(md(rows, ["Nominal alpha", "Expected breaches", "True coverage", "True breaches", "Kupiec power"]))
    # Interpretation generated from the table numbers above, to avoid text/table mismatch
    w(f"Interpretation (n = {n_out}, using power >= 80% as the usability criterion):")
    for al in ALPHAS:
        ok_m = [m for m in MULTS if power[(al, m)] >= 0.80]
        lo = power[(al, MULTS[0])]
        if ok_m:
            m0 = ok_m[0]
            w(f"  {al:.0%} level: power only reaches {power[(al, m0)]:.0%} when the true coverage "
              f"is {m0:g} times nominal ({al*m0:.2%}); at a {MULTS[0]:g}x deviation power is only "
              f"{lo:.0%}, so failure to reject cannot be read as good calibration.")
        else:
            w(f"  {al:.0%} level: even when true coverage is {MULTS[-1]:g} times nominal, power is "
              f"only {power[(al, MULTS[-1])]:.0%} -- conclusions at this level cannot be relied on "
              f"at this sample size.")

    # -------------------------------------------------------- Summary ----
    w("")
    rule(); w("Section 6  CHECK summary"); rule()
    for c in CHECKS:
        w("  " + c)
    nf = sum(1 for c in CHECKS if c.startswith("[FAIL]"))
    w("")
    w(f"{len(CHECKS)} checks, {nf} failed." if nf else f"All {len(CHECKS)} checks passed.")

    p = os.path.join(a.outdir, "var_es_backtest.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()