"""Established risk predictors and historical portfolio variance (Methodology)."""


from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

OWN_FEATURES = ("own_rv_5", "own_rv_21", "own_rv_126")


MARKET_FEATURES = ("mkt_rv_126", "bear_24", "bear_x_var", "mkt_ret_36")


CORE_FEATURES = ("book_rv_126", "wml_beta_126", "bdp_ci", "disp_21", "mom_gap", "comom", "illiq")


FEATURE_COLUMNS = (*OWN_FEATURES, *CORE_FEATURES, *MARKET_FEATURES)


MACRO_FEATURES = ("log_vix", "baa10y", "t10y3m")


FEATURE_BLOCKS = {"HAR": OWN_FEATURES, "CORE": CORE_FEATURES,
                  "M": MARKET_FEATURES, "FRED": MACRO_FEATURES}


FEATURE_SETS = {
    "L1-own": OWN_FEATURES,
    "L1_without_M": tuple(c for c in FEATURE_COLUMNS if c not in MARKET_FEATURES),
    "L1": FEATURE_COLUMNS,
    "L1m": (*FEATURE_COLUMNS, *MACRO_FEATURES),
}


def rolling_mean(values: NDArray, window: int) -> NDArray:
    """Trailing arithmetic mean requiring every observation, with no future padding
    (Methodology)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or window < 1:
        raise ValueError("Rolling mean requires a vector and positive window")
    output = np.full(len(values), np.nan)
    if len(values) >= window:
        windows = np.lib.stride_tricks.sliding_window_view(values, window)
        output[window-1:] = windows.mean(axis=1)
    return output


def log_realized_variance(values: NDArray, window: int, horizon: int) -> NDArray:
    """log((horizon/window) sum last-window P²), Methodology."""
    variance = horizon * rolling_mean(np.asarray(values)**2, window)
    return np.log(variance, out=np.full(len(variance), np.nan), where=variance > 0)


def rolling_beta(returns: NDArray, market: NDArray, window: int) -> NDArray:
    """OLS slope with intercept: cov(P,Mkt−RF)/var(Mkt−RF), Methodology."""
    y, x = np.asarray(returns), np.asarray(market)
    cov = rolling_mean(x*y, window) - rolling_mean(x, window)*rolling_mean(y, window)
    variance = rolling_mean(x*x, window) - rolling_mean(x, window)**2
    return np.divide(cov, variance, out=np.full(len(x), np.nan), where=variance > 0)


def quantile_skewness(values: NDArray, quantiles: list[float]) -> float:
    """(q_high+q_low−2q_median)/(q_high−q_low), BDP approximation in Methodology."""
    if not np.isfinite(values).all():
        return float("nan")
    low, median, high = np.quantile(values, quantiles)
    return float((high + low - 2*median)/(high-low)) if high > low else float("nan")


def approximate_bdp(returns: NDArray, settings: dict[str, Any]) -> NDArray:
    """Daily RMS over 126 × quantile skewness over 252 (labeled approximation; Methodology)."""
    cfg = settings["features"]
    scale = np.sqrt(rolling_mean(np.asarray(returns)**2, cfg["approximation_scale_days"]))
    shape = np.full(len(returns), np.nan)
    window = cfg["approximation_shape_days"]
    for index in range(window-1, len(returns)):
        shape[index] = quantile_skewness(returns[index-window+1:index+1],
                                        cfg["approximation_quantiles"])
    return scale * shape


def average_residual_correlation(returns: NDArray, factors: NDArray) -> tuple[float, int]:
    """Balanced weekly excess-return FF3 residual correlation, Methodology.

    Columns are stocks; factors are Mkt−RF, SMB, HML, RF. All stocks use the
    same weeks and OLS intercept. Standardize using 1/T moments, then apply
    (n Var(mean(z_i))−1)/(n−1). Undefined zero-dispersion residuals are excluded
    and counted in the returned usable panel size. This is the pairwise,
    loser-only, no-industry-adjustment variant of Lou–Polk (2022), footnote 16.
    """
    panel, ff = np.asarray(returns, dtype=float), np.asarray(factors, dtype=float)
    if panel.ndim != 2 or ff.shape != (panel.shape[0], 4):
        raise ValueError("Weekly panel and four factors must align")
    if not np.isfinite(ff).all():
        return float("nan"), 0
    panel = panel[:, np.isfinite(panel).all(axis=0)]
    if panel.shape[1] < 2:
        return float("nan"), panel.shape[1]
    design = np.column_stack([np.ones(len(ff)), ff[:, :3]])
    excess = panel - ff[:, 3, None]
    residuals = excess - design @ np.linalg.lstsq(design, excess, rcond=None)[0]
    sd = residuals.std(axis=0)
    threshold = np.finfo(float).eps * max(design.shape) * np.max(np.abs(excess), axis=0)
    residuals = residuals[:, sd > threshold]
    sd = sd[sd > threshold]
    n = len(sd)
    if n < 2:
        return float("nan"), n
    standardized = (residuals - residuals.mean(axis=0)) / sd
    return float((n*np.var(standardized.mean(axis=1))-1)/(n-1)), n


def historical_book_variance(
    returns: NDArray, weights: NDArray, horizon: int,
) -> tuple[float, float]:
    """horizon × mean((omega' r)²), missing r=0; absolute-weight coverage (Methodology)."""
    returns, weights = np.asarray(returns), np.asarray(weights)
    if returns.ndim != 2 or returns.shape[1] != len(weights):
        raise ValueError("Historical returns and formation weights must align")
    total = np.abs(weights).sum()
    if total <= 0:
        return float("nan"), float("nan")
    observed = np.isfinite(returns)
    path = np.where(observed, returns, 0) @ weights
    variance = horizon * np.mean(path*path)
    coverage = np.mean(observed @ np.abs(weights) / total)
    return (float(np.log(variance)) if variance > 0 else float("nan"), float(coverage))


def completed_week_indices(calendar: list[date], origin: date) -> list[list[int]]:
    """Whole calendar weeks whose last scheduled trading date is ≤origin (Methodology)."""
    weeks: dict[date, list[int]] = {}
    for index, day in enumerate(calendar):
        monday = day - timedelta(days=day.weekday())
        weeks.setdefault(monday, []).append(index)
    return [indices for indices in weeks.values() if calendar[indices[-1]] <= origin
            and (calendar[indices[-1]] < calendar[-1]
                 or calendar[indices[-1]].weekday() == 4)]


def weekly_compound(panel: NDArray, groups: list[list[int]]) -> NDArray:
    """Compound daily stock returns or factor returns, retaining any missing value (Methodology)."""
    return np.array([np.prod(1 + panel[indices], axis=0) - 1 for indices in groups])


def own_features(books: pl.DataFrame, factors: pl.DataFrame, settings: dict) -> pl.DataFrame:
    """Return per-book trailing own features, retaining warmup nulls (Methodology)."""
    cfg, horizon = settings["features"], settings["targets"]["horizon"]
    frames = []
    for frame in books.sort("date").partition_by("series", maintain_order=True):
        frame = frame.join(factors.select("date", "mkt_rf"), on="date", how="left")
        y = frame["daily_return"].to_numpy()
        columns = [pl.Series(f"own_rv_{n}", log_realized_variance(y, n, horizon))
                   for n in cfg["own_rv_days"]]
        columns += [pl.Series("wml_beta_126", rolling_beta(y, frame["mkt_rf"].to_numpy(),
                                                        cfg["beta_days"])),
                    pl.Series("bdp_ci", approximate_bdp(y, settings))]
        frames.append(frame.select("date", "series").with_columns(columns))
    return pl.concat(frames).sort("date", "series")


def market_features(factors: pl.DataFrame, settings: dict) -> pl.DataFrame:
    """French-only trailing market risk and trends; RAW market = Mkt−RF+RF (Methodology)."""
    cfg = settings["features"]
    frame = factors.sort("date")
    market = frame["mkt_rf"].to_numpy()
    total = market + frame["rf"].to_numpy()
    variance = settings["targets"]["horizon"] * rolling_mean(market**2, cfg["market_rv_days"])
    bear_path = rolling_mean(np.log1p(total), cfg["bear_days"]) * cfg["bear_days"]
    bear = np.where(np.isfinite(bear_path), (bear_path < 0).astype(float), np.nan)
    trend = np.expm1(rolling_mean(np.log1p(total), cfg["market_return_days"])
                     * cfg["market_return_days"])
    return frame.select("date").with_columns(
        pl.Series("mkt_rv_126", np.log(variance)), pl.Series("bear_24", bear),
        pl.Series("bear_x_var", bear * variance), pl.Series("mkt_ret_36", trend),
    )


def macro_features(dates: pl.DataFrame, observations: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Backward observation-date joins for the three authorized FRED inputs (Methodology)."""
    frame = dates.select("date").unique().sort("date")
    for series, column, logarithm in (("VIXCLS", "log_vix", True),
                                     ("BAA10Y", "baa10y", False),
                                     ("T10Y3M", "t10y3m", False)):
        observed = observations[series].filter(pl.col("value").is_not_null()).sort("date")
        frame = frame.join_asof(observed.select("date", pl.col("value").alias(column)),
                               on="date", strategy="backward")
        if logarithm:
            frame = frame.with_columns(pl.col(column).log())
    return frame
