"""
Performance, risk and factor attribution for Part 4's daily return series.

All inputs are daily simple returns of a book scaled to gross exposure 1
(long $0.50, short $0.50) unless stated. Annualisation uses 252 sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ANN = 252
NW_LAGS = 5

WINDOWS: dict[str, tuple[str, str]] = {
    "full 1996-2024": ("1996-01-01", "2024-12-31"),
    "1996-2006": ("1996-01-01", "2006-12-31"),
    "2007-2024": ("2007-01-01", "2024-12-31"),
    "post-2010": ("2010-01-04", "2024-12-31"),
    "last 18m": ("2023-07-03", "2024-12-31"),
    "last 12m": ("2024-01-01", "2024-12-31"),
}

STRESS: dict[str, tuple[str, str]] = {
    "Aug 2007 quant quake": ("2007-08-01", "2007-08-31"),
    "Sep-Oct 2008": ("2008-09-01", "2008-10-31"),
    "Mar 2020": ("2020-03-01", "2020-03-31"),
}


def nw_mean_test(r: pd.Series, lags: int = NW_LAGS) -> tuple[float, float]:
    """Newey-West t-statistic and two-sided p-value for the mean of `r`."""
    r = r.dropna()
    if len(r) < lags + 10:
        return np.nan, np.nan
    res = sm.OLS(r.to_numpy(), np.ones(len(r))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return float(res.tvalues[0]), float(res.pvalues[0])


def drawdown(r: pd.Series) -> pd.Series:
    """Drawdown of cumulative wealth (1 + r).cumprod() from its running peak.

    The peak starts at the initial capital of 1, so a loss on the first day
    of a series or window counts as a drawdown.
    """
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    return wealth / wealth.cummax().clip(lower=1.0) - 1.0


def max_dd_duration(r: pd.Series) -> int:
    """Longest run of consecutive sessions spent below the previous peak."""
    under = (drawdown(r) < 0).to_numpy()
    best = run = 0
    for u in under:
        run = run + 1 if u else 0
        best = max(best, run)
    return best


def summary_stats(r: pd.Series) -> dict[str, float]:
    """Headline statistics for one daily return series."""
    r = r.dropna()
    n = len(r)
    if n < 20:
        return {"n_days": n}
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else np.nan
    t_nw, p_nw = nw_mean_test(r)
    return {
        "n_days": n,
        "ann_mean": mu * ANN,
        "ann_vol": sd * np.sqrt(ANN),
        "sharpe": sr_d * np.sqrt(ANN),
        # Lo (2002), iid case: SE of the annualised Sharpe ratio
        "sharpe_se": np.sqrt((1.0 + 0.5 * sr_d**2) / n) * np.sqrt(ANN),
        "t_nw": t_nw,
        "p_nw": p_nw,
        "max_dd": float(drawdown(r).min()),
        "max_dd_days": max_dd_duration(r),
        "hit_rate": float((r > 0).mean()),
        "skew": float(stats.skew(r)),
        "excess_kurt": float(stats.kurtosis(r)),
    }


def window_slice(r: pd.Series, start: str, end: str) -> pd.Series:
    return r.loc[(r.index >= pd.Timestamp(start)) & (r.index <= pd.Timestamp(end))]


def var_backtest(
    r: pd.Series, var_level: float = 0.99, es_level: float = 0.975, window: int = 250
) -> dict[str, float]:
    """One-day-ahead historical VaR and ES, with Kupiec and Christoffersen tests.

    VaR and ES for day t use only the `window` returns ending at t-1. An
    exceedance is a day whose return is below -VaR. Kupiec tests whether the
    exceedance rate equals 1 - var_level; Christoffersen tests whether
    exceedances are independent from one day to the next.
    """
    r = r.dropna()
    q_var = r.rolling(window).quantile(1 - var_level).shift(1)
    es =r.rolling(window).apply(
        lambda x: x[x <= np.quantile(x, 1 - es_level)].mean(), raw=True
    ).shift(1)
    ok = q_var.notna()
    hits = (r[ok] < q_var[ok]).astype(int).to_numpy()
    n, x = len(hits), int(hits.sum())
    p = 1 - var_level
    if n < 2:
        return {"n_days": n, "var_level": var_level}

    def _ll(k: int, m: int, prob: float) -> float:
        """Binomial log-likelihood, using the 0*log(0) = 0 limit."""
        out = 0.0
        if m - k > 0:
            out += (m - k) * np.log(1 - prob) if prob < 1 else -np.inf
        if k > 0:
            out += k * np.log(prob) if prob > 0 else -np.inf
        return out

    lr_uc = -2 * (_ll(x, n, p) - _ll(x, n, x / n))
    # Christoffersen independence: transition counts of the hit sequence
    h0, h1 = hits[:-1], hits[1:]
    n00 = int(((h0 == 0) & (h1 == 0)).sum()); n01 = int(((h0 == 0) & (h1 == 1)).sum())
    n10 = int(((h0 == 1) & (h1 == 0)).sum()); n11 = int(((h0 == 1) & (h1 == 1)).sum())
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = _ll(n01 + n11, n00 + n01 + n10 + n11, pi)
    ll_alt = _ll(n01, n00 + n01, pi0) + _ll(n11, n10 + n11, pi1)
    lr_ind = -2 * (ll_null - ll_alt)
    return {
        "n_days": n,
        "var_level": var_level,
        "avg_var": float(-q_var[ok].mean()),
        "avg_es": float(-es[ok].mean()),
        "exceedances": x,
        "expected": n * p,
        "kupiec_lr": lr_uc,
        "kupiec_p": float(stats.chi2.sf(lr_uc, 1)) if np.isfinite(lr_uc) else np.nan,
        "christoffersen_lr": lr_ind,
        "christoffersen_p": float(stats.chi2.sf(lr_ind, 1)) if np.isfinite(lr_ind) else np.nan,
    }


def stress_table(r: pd.Series) -> pd.DataFrame:
    """Cumulative return, annualised vol and max drawdown in each stress window."""
    rows = []
    for name, (a, b) in STRESS.items():
        s = window_slice(r.dropna(), a, b)
        if len(s) == 0:
            continue
        rows.append({
            "period": name,
            "n_days": len(s),
            "cum_return": float((1 + s).prod() - 1),
            "ann_vol": float(s.std(ddof=1) * np.sqrt(ANN)),
            "max_dd": float(drawdown(s).min()),
            "worst_day": float(s.min()),
        })
    return pd.DataFrame(rows)


def attribution(r: pd.Series, regressors: pd.DataFrame, lags: int = NW_LAGS) -> dict[str, float]:
    """Newey-West time-series regression of `r` on `regressors`.

    Returns the annualised intercept and its t-statistic, each slope with its
    t-statistic, and R-squared.
    """
    df = pd.concat([r.rename("y"), regressors], axis=1, join="inner").dropna()
    if len(df) < 60:
        return {"n_days": len(df)}
    X = sm.add_constant(df.drop(columns="y"))
    res = sm.OLS(df["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out: dict[str, float] = {
        "n_days": len(df),
        "alpha_ann": float(res.params["const"] * ANN),
        "t_alpha": float(res.tvalues["const"]),
        "p_alpha": float(res.pvalues["const"]),
        "r2": float(res.rsquared),
    }
    for name in regressors.columns:
        out[f"b_{name}"] = float(res.params[name])
        out[f"t_{name}"] = float(res.tvalues[name])
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for a family of tests.

    Raises on a missing p-value: one NaN would otherwise sort first and force
    every adjusted p-value to 1.
    """
    bad = [k for k, v in pvalues.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"holm: non-finite p-values for {bad}")
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted
