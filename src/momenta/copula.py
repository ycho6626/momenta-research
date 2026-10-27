"""RAW-universe copula geometry and training-only centroids (Methodology)."""


from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

COORDINATES = ("mom", "r1m", "beta", "sigma")


CM_PAIRS = tuple(combinations(range(len(COORDINATES)), 2))


CM_COLUMNS = tuple(f"cm_{COORDINATES[i]}_{COORDINATES[j]}" for i, j in CM_PAIRS)


Q_COLUMNS = (*(f"{name}_sq" for name in CM_COLUMNS),
             *(f"{a}_x_{b}" for a, b in combinations(CM_COLUMNS, 2)))


OT_COLUMNS = tuple(f"ot_distance_{i+1}" for i in range(4))


def copula_cloud(values: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Average rank of each of four coordinates divided by n+1 on complete rows (Methodology)."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(COORDINATES):
        raise ValueError("A copula cloud requires MOM, current return, beta and return SD")
    complete = np.isfinite(values).all(axis=1)
    selected = values[complete]
    if len(selected) < 2:
        raise ValueError("A copula cloud requires at least two complete stocks")
    return rankdata(selected, method="average", axis=0)/(len(selected)+1), complete


def stock_beta_sigma(
    returns: NDArray[np.float64], market: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Beta=cov(r,Mkt−RF)/var(Mkt−RF), sigma=sqrt(var(r)), 1/T moments (Methodology).

    Require every daily stock return in the supplied window. Constant stock
    returns have beta=0 and sigma=0 and remain legitimate tied coordinates.
    """
    returns, market = np.asarray(returns, dtype=float), np.asarray(market, dtype=float)
    if (returns.ndim != 2 or market.shape != (len(returns),) or len(market) < 2
            or not np.isfinite(market).all()):
        raise ValueError("Stock beta requires aligned complete market observations")
    centered_market = market-market.mean()
    variance = np.mean(centered_market**2)
    if variance <= 0:
        raise ValueError("Stock beta requires nonconstant market returns")
    valid = np.isfinite(returns).all(axis=0)
    beta, sigma = np.full(returns.shape[1], np.nan), np.full(returns.shape[1], np.nan)
    centered = returns[:, valid]-returns[:, valid].mean(axis=0)
    beta[valid] = np.mean(centered*centered_market[:, None], axis=0)/variance
    sigma[valid] = np.sqrt(np.mean(centered**2, axis=0))
    constant = np.ptp(returns[:, valid], axis=0) == 0
    indices = np.flatnonzero(valid)[constant]
    beta[indices], sigma[indices] = 0., 0.
    return beta, sigma


def unit_directions(count: int, dimensions: int, seed: int) -> NDArray[np.float64]:
    """Normalize seeded isotropic Gaussian vectors to sample fixed unit directions (Methodology)."""
    if count < 1 or dimensions < 1:
        raise ValueError("Direction count and dimension must be positive")
    values = np.random.default_rng(seed).standard_normal((count, dimensions))
    return values/np.linalg.norm(values, axis=1, keepdims=True)


def sliced_quantiles(
    cloud: NDArray[np.float64], directions: NDArray[np.float64], grid_points: int,
    *, method: str,
) -> NDArray[np.float64]:
    """Q_l(p)=quantile(theta_l' u,p), p equally spaced on [0,1] (Methodology)."""
    cloud, directions = np.asarray(cloud, dtype=float), np.asarray(directions, dtype=float)
    if (cloud.ndim != 2 or not len(cloud) or directions.ndim != 2
            or cloud.shape[1] != directions.shape[1] or not len(directions)
            or not np.isfinite(cloud).all() or not np.isfinite(directions).all()
            or grid_points < 2):
        raise ValueError("Sliced quantiles require finite compatible clouds/directions and a grid")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1.):
        raise ValueError("Sliced-Wasserstein directions must have unit length")
    return np.quantile(cloud @ directions.T, np.linspace(0., 1., grid_points),
                       axis=0, method=method).T


def embedding(quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flatten Q/sqrt(LG), so Euclidean distance equals fixed-grid RMS SW2 (Methodology)."""
    quantiles = np.asarray(quantiles, dtype=float)
    if quantiles.ndim != 2 or not quantiles.size or not np.isfinite(quantiles).all():
        raise ValueError("A sliced-quantile embedding requires a finite nonempty matrix")
    return quantiles.ravel()/np.sqrt(quantiles.size)


def sw2_distance(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    """sqrt(mean((Q_a−Q_b)²)) across the common directions and grid (Methodology)."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.shape != second.shape:
        raise ValueError("SW2 requires the same directions and quantile grid")
    return float(np.linalg.norm(embedding(first)-embedding(second)))


def cm_moments(cloud: NDArray[np.float64]) -> NDArray[np.float64]:
    """Six Pearson correlations of the rank coordinates, i.e. Spearman moments (Methodology)."""
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != len(COORDINATES) or len(cloud) < 2:
        raise ValueError("Linear copula moments require a four-coordinate cloud")
    if not np.isfinite(cloud).all():
        raise ValueError("Linear copula moments require finite ranks")
    centered = cloud-cloud.mean(axis=0)
    scale = np.sqrt(np.mean(centered**2, axis=0))
    return np.array([np.mean(centered[:, i]*centered[:, j])/(scale[i]*scale[j])
                     if scale[i] > 0 and scale[j] > 0 else np.nan for i, j in CM_PAIRS])


def quadratic_moments(moments: NDArray[np.float64]) -> NDArray[np.float64]:
    """Six CM squares followed by the 15 distinct pairwise products (Methodology diagnostic)."""
    moments = np.asarray(moments, dtype=float)
    if moments.shape[-1:] != (len(CM_COLUMNS),):
        raise ValueError("The quadratic diagnostic requires exactly six copula moments")
    return np.concatenate([moments**2, np.stack(
        [moments[..., i]*moments[..., j] for i, j in combinations(range(len(CM_COLUMNS)), 2)],
        axis=-1,
    )], axis=-1)


def rank_weighted_value(
    signals: NDArray[np.float64], values: NDArray[np.float64], scale: float,
) -> float:
    """Σ c(rank MOM−mean rank)·value = N c cov(rank MOM,value), with 1/N (Methodology)."""
    signals, values = np.asarray(signals, dtype=float), np.asarray(values, dtype=float)
    if (signals.ndim != 1 or signals.shape != values.shape or not len(signals)
            or not np.isfinite(signals).all() or not np.isfinite(values).all()
            or not np.isfinite(scale)):
        raise ValueError("The rank-weighted identity requires aligned finite vectors and scale")
    ranks = rankdata(signals, method="average")
    return float((scale*(ranks-ranks.mean())) @ values)


@dataclass(frozen=True)
class CentroidFit:
    """Training-set centroids and their deterministic aligned labels (Methodology)."""

    centers: NDArray[np.float64]
    permutation: NDArray[np.int64]
    counts: NDArray[np.int64]
    inertia: float
    iterations: int

    def transform(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Distances to fixed training centroids; no refit or cross-month normalization
        (Methodology)."""
        values = np.asarray(values, dtype=float)
        if (values.ndim != 2 or values.shape[1] != self.centers.shape[1]
                or not np.isfinite(values).all()):
            raise ValueError("Centroid distances require finite embeddings in the fitted space")
        return cdist(values, self.centers)


def fit_centroids(
    train_embeddings: NDArray[np.float64], settings: dict[str, Any],
    previous_centroids: NDArray[np.float64] | None = None,
) -> CentroidFit:
    """Seeded KMeans on training embeddings only; Hungarian Euclidean alignment (Methodology).

    The caller keeps inner/full fit histories separate and supplies a previous
    fit learned no later than the current training cutoff. First-fit labels
    use lexicographic centroid order; alignment never changes distances.
    """
    values = np.asarray(train_embeddings, dtype=float)
    cfg = settings['ot']
    if (values.ndim != 2 or not np.isfinite(values).all()
            or len(values) < cfg['k_centroids']
            or len(np.unique(values, axis=0)) < cfg['k_centroids']):
        raise ValueError("KMeans requires enough distinct finite training-month embeddings")
    with threadpool_limits(limits=settings['walkforward']['blas_threads']):
        model = KMeans(
            n_clusters=cfg['k_centroids'], n_init=cfg['kmeans_n_init'], random_state=cfg['seed'],
            max_iter=cfg['kmeans_max_iter'], tol=cfg['kmeans_tolerance'], algorithm='lloyd',
        ).fit(values)
    centers = model.cluster_centers_
    if previous_centroids is None:
        order = np.array(sorted(range(len(centers)), key=lambda i: tuple(centers[i])), dtype=int)
    else:
        previous = np.asarray(previous_centroids, dtype=float)
        if previous.shape != centers.shape or not np.isfinite(previous).all():
            raise ValueError("Previous centroids must lie in the same embedding space")
        rows, columns = linear_sum_assignment(cdist(previous, centers))
        order = columns[np.argsort(rows)]
    aligned = centers[order]
    nearest = np.argmin(cdist(values, aligned), axis=1)
    return CentroidFit(aligned, order, np.bincount(nearest, minlength=len(centers)),
                       float(model.inertia_), int(model.n_iter_))


def fit_dated_centroids(
    monthly: pl.DataFrame, training_dates: Sequence[date], fit_cutoff: date,
    settings: dict[str, Any], previous_centroids: NDArray[np.float64] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fit only the unique eligible training months; transform without learning later rows
    (Methodology).

    The dated runner supplies months whose originating target windows finish
    before its full or inner cutoff. A repeated daily carried-forward value
    never increases that month's weight in the unsupervised centroid fit.
    """
    training_dates = sorted(set(training_dates))
    first = date.fromisoformat(settings['walkforward']['feature_start'])
    if not training_dates or training_dates[0] < first or training_dates[-1] >= fit_cutoff:
        raise ValueError("Centroid training months must follow feature_start and precede cutoff")
    monthly = monthly.sort('date')
    if monthly['date'].n_unique() != monthly.height:
        raise ValueError("Duplicate monthly copula embeddings")
    selected = monthly.filter(pl.col('date').is_in(training_dates))
    if selected['date'].to_list() != training_dates:
        raise ValueError("A requested centroid training month has no embedding")
    training = np.asarray(selected['embedding'].to_list(), dtype=float)
    fitted = fit_centroids(training, settings, previous_centroids)
    values = np.asarray(monthly['embedding'].to_list(), dtype=float)
    distances = fitted.transform(values)
    names = [f'ot_distance_{i+1}' for i in range(len(fitted.centers))]
    transformed = monthly.select('date').with_columns(
        [pl.Series(name, distances[:, i]) for i, name in enumerate(names)]
        + [pl.Series('nearest_centroid', distances.argmin(axis=1))],
    )
    return transformed, {
        'fit_cutoff': str(fit_cutoff), 'training_first': str(training_dates[0]),
        'training_last': str(training_dates[-1]), 'training_months': len(training_dates),
        'training_dates': [str(day) for day in training_dates],
        'centroids': fitted.centers.tolist(), 'alignment_permutation': fitted.permutation.tolist(),
        'occupancy_counts': fitted.counts.tolist(), 'inertia': fitted.inertia,
        'iterations': fitted.iterations, 'features': names,
    }


def separation_diagnostic(
    clouds: Sequence[NDArray[np.float64]], directions: NDArray[np.float64],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Compare 20 same-size half-cloud distances within/adjacent/across all months (Methodology).

    The globally common half size is floor(minimum cloud size/2). Each split
    independently subsamples two disjoint halves per month; unused points of
    larger clouds are discarded for that split. The fixed full-cloud ranks
    are retained. The ratio threshold is descriptive and never blocks H4.
    """
    cfg = settings['ot']
    clouds = [np.asarray(cloud, dtype=float) for cloud in clouds]
    if len(clouds) < 2 or min(len(cloud) for cloud in clouds) < 4:
        raise ValueError("Separation needs multiple months with at least four cloud points")
    half_size = min(len(cloud) for cloud in clouds)//2
    within, adjacent, all_pairs = [], [], []
    rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 3]))
    with threadpool_limits(limits=settings['walkforward']['blas_threads']):
        for _ in range(cfg['separation_splits']):
            first, second = [], []
            for cloud in clouds:
                indices = rng.permutation(len(cloud))[:2*half_size]
                for selected, destination in ((indices[:half_size], first),
                                               (indices[half_size:], second)):
                    destination.append(embedding(sliced_quantiles(
                        cloud[selected], directions, cfg['grid_points'],
                        method=cfg['quantile_method'],
                    )))
            a, b = np.asarray(first), np.asarray(second)
            squared = (np.sum(a*a, axis=1)[:, None] + np.sum(b*b, axis=1)[None, :]
                       - 2*(a @ b.T))
            distances = np.sqrt(np.maximum(squared, 0.))
            within.extend(distances.diagonal())
            adjacent.extend(distances[np.arange(len(clouds)-1), np.arange(1, len(clouds))])
            all_pairs.extend(distances[np.triu_indices(len(clouds), k=1)])
    medians = {name: float(np.median(values)) for name, values in (
        ('within', within), ('adjacent', adjacent), ('all_pairs', all_pairs),
    )}
    ratio = medians['adjacent']/medians['within'] if medians['within'] > 0 else None
    return {
        'months': len(clouds), 'half_size': half_size, 'splits': cfg['separation_splits'],
        'seed': cfg['seed'], 'medians': medians,
        'counts': {'within': len(within), 'adjacent': len(adjacent), 'all_pairs': len(all_pairs)},
        'adjacent_to_within_ratio': ratio, 'threshold': cfg['separation_ratio'],
        'threshold_passed': bool(ratio is not None and ratio >= cfg['separation_ratio']),
        'is_gate': False,
    }


def synthetic_diagnostics(settings: dict[str, Any]) -> dict[str, Any]:
    """Fixed n=1000 Spearman-shift gate and Gaussian-vs-t3 tail diagnostic (Methodology).

    Independent Gaussian null pairs supply a matched-size 95% SW2 noise
    threshold. The shift is an equicorrelated Gaussian copula whose every-pair
    population Spearman is 0.2. The tail alternative is a common-radial t3
    with identity correlation; it shares the Gaussian population Spearman0
    while retaining tail dependence. Every cloud is separately rank-transformed.
    """
    cfg = settings['ot']
    n, dimensions = cfg['synthetic_n'], len(COORDINATES)
    directions = unit_directions(cfg['directions'], dimensions, cfg['seed'])

    def distance(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
        quantiles = [sliced_quantiles(cloud, directions, cfg['grid_points'],
                                     method=cfg['quantile_method']) for cloud in (first, second)]
        return sw2_distance(*quantiles)

    null_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 0]))
    noise = []
    for _ in range(cfg['synthetic_noise_reps']):
        first = copula_cloud(null_rng.standard_normal((n, dimensions)))[0]
        second = copula_cloud(null_rng.standard_normal((n, dimensions)))[0]
        noise.append(distance(first, second))
    threshold = float(np.quantile(noise, 1-cfg['diagnostic_alpha'],
                                  method=cfg['quantile_method']))
    shift_rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 1]))
    rho = 2*np.sin(np.pi*cfg['synthetic_shift']/6)
    correlation = np.full((dimensions, dimensions), rho)
    np.fill_diagonal(correlation, 1.)
    before = copula_cloud(shift_rng.standard_normal((n, dimensions)))[0]
    after = copula_cloud(shift_rng.standard_normal((n, dimensions))
                         @ np.linalg.cholesky(correlation).T)[0]
    shifted_distance = distance(before, after)
    differences = cm_moments(after)-cm_moments(before)
    tail = []
    for replication in range(cfg['tail_reps']):
        rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], 2, replication]))
        gaussian = copula_cloud(rng.standard_normal((n, dimensions)))[0]
        student = rng.standard_normal((n, dimensions)) / np.sqrt(
            rng.chisquare(cfg['tail_df'], size=(n, 1))/cfg['tail_df'],
        )
        student = copula_cloud(student)[0]
        difference = cm_moments(student)-cm_moments(gaussian)
        observed = distance(gaussian, student)
        tail.append({'replication': replication, 'distance': observed,
                     'detected': observed > threshold,
                     'cm_difference': dict(zip(CM_COLUMNS, difference.tolist(), strict=True))})
    return {
        'n': n, 'seed': cfg['seed'], 'directions': cfg['directions'],
        'grid_points': cfg['grid_points'], 'null_reps': len(noise), 'noise_distances': noise,
        'noise_quantile': 1-cfg['diagnostic_alpha'], 'noise_threshold': threshold,
        'shift': {'population_spearman': cfg['synthetic_shift'], 'latent_rho': float(rho),
                  'realized_shifts': dict(zip(CM_COLUMNS, differences.tolist(), strict=True)),
                  'distance': shifted_distance, 'passed': shifted_distance > threshold},
        'tail': {'df': cfg['tail_df'], 'population_spearman': 0., 'reps': len(tail),
                 'detection_rate': float(np.mean([row['detected'] for row in tail])),
                 'is_gate': False, 'replications': tail},
    }
