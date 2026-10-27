"""Shared monthly L2 observables from the RAW universe (Methodology)."""


from __future__ import annotations

import calendar as month_calendar
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.stats import rankdata

L2_COLUMNS = (
    "shape_mom_bowley", "shape_mom_moors", "shape_r1m_bowley", "shape_r1m_moors",
    "rho_ww", "rho_ll", "rho_wl", "rank_persist", "rank_ic",
)


@dataclass(frozen=True)
class BalancedPanel:
    """Complete, nonconstant columns standardized with population moments (Methodology)."""

    z: NDArray[np.float64]
    columns: NDArray[np.int64]
    complete_count: int
    missing_count: int
    zero_variance_count: int


def balanced_standardize(panel: NDArray[np.float64]) -> BalancedPanel:
    """z=(r−mean_t r)/sqrt(mean_t (r−mean_t r)²), complete columns only (Methodology).

    A constant column has undefined correlation and is omitted, separately
    from columns with missing observations. No pairwise filling is performed.
    """
    panel = np.asarray(panel, dtype=float)
    if panel.ndim != 2 or not panel.shape[0]:
        raise ValueError("A balanced panel requires a nonempty time axis")
    complete = np.isfinite(panel).all(axis=0)
    indices = np.flatnonzero(complete)
    values = panel[:, indices]
    centered = values - values.mean(axis=0)
    variance = np.mean(centered**2, axis=0)
    # The range check recognizes exactly constant floating-point columns even
    # if their repeated value rounds during the population-mean reduction.
    varying = (np.ptp(values, axis=0) > 0) & (variance > 0)
    return BalancedPanel(
        centered[:, varying] / np.sqrt(variance[varying]), indices[varying],
        int(complete.sum()), int((~complete).sum()), int((~varying).sum()),
    )


def within_correlation(z: NDArray[np.float64]) -> float:
    """(n Var_t(mean_i z_i)−1)/(n−1), with balanced 1/T standardized z (Methodology)."""
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Within-leg correlation requires a finite standardized panel")
    if z.shape[0] < 2 or z.shape[1] < 2:
        return float("nan")
    n = z.shape[1]
    return float((n * np.var(z.mean(axis=1)) - 1) / (n - 1))


def cross_correlation(winners: NDArray[np.float64], losers: NDArray[np.float64]) -> float:
    """Cov_t(mean_i z_W, mean_j z_L), balanced common dates and 1/T moments (Methodology)."""
    winners, losers = np.asarray(winners, dtype=float), np.asarray(losers, dtype=float)
    if (winners.ndim != 2 or losers.ndim != 2 or len(winners) != len(losers)
            or not np.isfinite(winners).all() or not np.isfinite(losers).all()):
        raise ValueError("Cross-leg correlation requires finite panels on the same dates")
    if len(winners) < 2 or not winners.shape[1] or not losers.shape[1]:
        return float("nan")
    w, loser = winners.mean(axis=1), losers.mean(axis=1)
    return float(np.mean((w-w.mean()) * (loser-loser.mean())))


def rank_correlation(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    """Spearman correlation with average tied ranks on finite paired observations (Methodology)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("Rank correlation requires aligned vectors")
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    a, b = rankdata(x[valid], method="average"), rankdata(y[valid], method="average")
    a, b = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.mean(a*a) * np.mean(b*b))
    return float(np.mean(a*b) / denominator) if denominator > 0 else float("nan")


def breadth(values: NDArray[np.float64]) -> float:
    """Mean sign(MOM) among finite nonzero observations; exact zeros excluded (Methodology)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Breadth requires a vector")
    selected = values[np.isfinite(values) & (values != 0)]
    return float(np.sign(selected).mean()) if len(selected) else float("nan")


def shape(
    values: NDArray[np.float64], bowley_quantiles: Sequence[float],
    moors_quantiles: Sequence[float], *, method: str,
) -> tuple[float, float]:
    """Bowley=(q75+q25−2q50)/IQR; Moors=(q87.5−q62.5+q37.5−q12.5)/IQR (Methodology).

    Use finite cross-sectional observations and the registered quantile method.
    Zero IQR gives two missing values rather than a denominator adjustment.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(bowley_quantiles) != 3 or len(moors_quantiles) != 4:
        raise ValueError("Shape requires a vector and the registered three/four quantiles")
    selected = values[np.isfinite(values)]
    if not len(selected):
        return float("nan"), float("nan")
    q25, q50, q75 = np.quantile(selected, bowley_quantiles, method=method)
    q125, q375, q625, q875 = np.quantile(selected, moors_quantiles, method=method)
    iqr = q75 - q25
    if iqr <= 0:
        return float("nan"), float("nan")
    return float((q75+q25-2*q50)/iqr), float((q875-q625+q375-q125)/iqr)


def monthly_features(
    days: Sequence[date], ids: NDArray[np.int64], returns: NDArray[np.float64],
    universe: pl.DataFrame, signals: pl.DataFrame, monthly_returns: pl.DataFrame,
    weights: pl.DataFrame, settings: dict[str, Any], *,
    formations: Sequence[date] | None = None,
) -> pl.DataFrame:
    """Compute each shared snapshot at its formation close, using no later row (Methodology).

    The caller supplies a bounded daily panel and the preceding formation's
    universe/signals. Current-month returns use calendar month-end keys, while
    feature dates remain trading month-ends. Leg membership is the pinned RAW
    primary formation, so momentum ties match the specified portfolios.
    """
    ids, returns = np.asarray(ids), np.asarray(returns, dtype=float)
    if (not days or list(days) != sorted(set(days)) or returns.shape != (len(days), len(ids))
            or not np.array_equal(ids, np.unique(ids))):
        raise ValueError("L2 needs sorted unique dates/IDs and an aligned return panel")
    cfg = settings["l2"]
    if cfg["leg_bins"] != settings["portfolios"]["primary_bins"]:
        raise ValueError("L2 leg bins must match the pinned primary RAW formation")
    window = cfg["correlation_days"]
    if window < 2:
        raise ValueError("L2 correlation windows require at least two observations")
    paired = universe.select("date", "permaticker").join(
        signals.select("date", "permaticker", "raw"), on=["date", "permaticker"],
        how="left", validate="1:1",
    ).sort("date", "permaticker")
    if paired["raw"].null_count() or not paired["raw"].is_finite().all():
        raise ValueError("Every L2 universe member requires its finite RAW signal")
    groups = {key[0]: frame for key, frame in paired.partition_by("date", as_dict=True).items()}
    periods = {(day.year, day.month): day for day in groups}
    if len(periods) != len(groups):
        raise ValueError("L2 expects exactly one formation per calendar month")
    leg_groups = {key[0]: frame for key, frame in weights.filter(
        pl.col("series") == "RAW", pl.col("construction") == settings["portfolios"]["primary"],
    ).partition_by("formation", as_dict=True).items()}
    monthly = {
        key[0]: dict(zip(frame["permaticker"].to_list(), frame["monthly_return"].to_list(),
                        strict=True))
        for key, frame in monthly_returns.partition_by("date", as_dict=True).items()
    }
    if monthly_returns.select(pl.struct("date", "permaticker").n_unique()).item() != len(
        monthly_returns
    ):
        raise ValueError("Duplicate stock-month returns")
    requested = (sorted(day for day in groups if days[0] <= day <= days[-1])
                 if formations is None else list(formations))
    if requested != sorted(set(requested)):
        raise ValueError("Requested L2 formations must be sorted and unique")
    day_indices = {day: i for i, day in enumerate(days)}
    id_indices = {int(identifier): i for i, identifier in enumerate(ids)}
    rows = []
    for formation in requested:
        if formation not in groups or formation not in day_indices:
            raise ValueError("Requested L2 formation is missing its universe or trading date")
        group = groups[formation]
        members, raw = group["permaticker"].to_list(), group["raw"].to_numpy()
        if any(identifier not in id_indices for identifier in members):
            raise ValueError("The L2 return panel omits a current universe member")
        month_end = date(formation.year, formation.month,
                         month_calendar.monthrange(formation.year, formation.month)[1])
        current_month = monthly.get(month_end, {})
        r1m = np.array([current_month.get(identifier) for identifier in members], dtype=float)
        mom_shape = shape(raw, cfg["bowley_quantiles"], cfg["moors_quantiles"],
                          method=cfg["quantile_method"])
        r1m_shape = shape(r1m, cfg["bowley_quantiles"], cfg["moors_quantiles"],
                          method=cfg["quantile_method"])
        leg = leg_groups.get(formation)
        if leg is None or leg["permaticker"].n_unique() != leg.height:
            raise ValueError("Missing or duplicate pinned RAW leg members")
        leg_ids = {
            "winner": leg.filter(pl.col("weight") > 0)["permaticker"].to_list(),
            "loser": leg.filter(pl.col("weight") < 0)["permaticker"].to_list(),
        }
        if any(not selected or not set(selected).issubset(members)
               for selected in leg_ids.values()):
            raise ValueError("RAW extreme legs must be nonempty subsets of the current universe")
        end = day_indices[formation] + 1
        panels = {}
        counts = {}
        for name, members_in_leg in leg_ids.items():
            columns = [id_indices[identifier] for identifier in members_in_leg]
            panel = (balanced_standardize(returns[end-window:end, columns]) if end >= window
                     else BalancedPanel(np.empty((end, 0)), np.empty(0, dtype=np.int64),
                                        0, len(columns), 0))
            panels[name] = panel.z
            counts.update({
                f"{name}_total": len(columns), f"{name}_complete": panel.complete_count,
                f"{name}_balanced": len(panel.columns), f"{name}_missing": panel.missing_count,
                f"{name}_zero_variance": panel.zero_variance_count,
            })
        previous_index = formation.year * 12 + formation.month - 2
        prior_year, prior_month = divmod(previous_index, 12)
        prior_date = periods.get((prior_year, prior_month + 1))
        prior_raw = ({} if prior_date is None else dict(zip(
            groups[prior_date]["permaticker"].to_list(), groups[prior_date]["raw"].to_list(),
            strict=True,
        )))
        intersection = np.array([i for i, identifier in enumerate(members)
                                 if identifier in prior_raw], dtype=int)
        lagged = np.array([prior_raw[members[i]] for i in intersection], dtype=float)
        rows.append({
            "date": formation,
            "shape_mom_bowley": mom_shape[0], "shape_mom_moors": mom_shape[1],
            "shape_r1m_bowley": r1m_shape[0], "shape_r1m_moors": r1m_shape[1],
            "rho_ww": within_correlation(panels["winner"]),
            "rho_ll": within_correlation(panels["loser"]),
            "rho_wl": cross_correlation(panels["winner"], panels["loser"]),
            "rank_persist": rank_correlation(lagged, raw[intersection]),
            "rank_ic": rank_correlation(lagged, r1m[intersection]), "breadth_m": breadth(raw),
            "universe_size": len(members), "shape_mom_count": int(np.isfinite(raw).sum()),
            "shape_r1m_count": int(np.isfinite(r1m).sum()),
            "breadth_count": int((np.isfinite(raw) & (raw != 0)).sum()),
            "rank_intersection_count": len(intersection),
            "rank_ic_count": int(np.isfinite(r1m[intersection]).sum()),
            "correlation_observations": min(end, window), **counts,
        })
    if not rows:
        raise ValueError("No L2 formations in the requested panel")
    return pl.DataFrame(rows).with_columns(pl.col(pl.Float64).fill_nan(None)).sort("date")
