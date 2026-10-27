"""Training-only monthly PCA/HMM states and causal probabilities (Methodology).

EM fitting uses hmmlearn 0.3.3; all reported probabilities use our own forward
recursion. https://hmmlearn.readthedocs.io/en/0.3.3/api.html#hmmlearn.hmm.GaussianHMM
"""


from __future__ import annotations

import json
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits


def _matrix(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError("Regimes require a nonempty finite matrix")
    return result


def fit_preprocessor(
    values: ArrayLike, settings: dict[str, Any], previous_components: ArrayLike | None = None,
) -> dict[str, Any]:
    """Fit population z scores, clip, center, then PCA by SVD on training only (Methodology).

    Orient each component toward its preceding fit's same component. With no
    preceding fit (or an exactly orthogonal reference), its largest loading is positive.
    """
    x = _matrix(values)
    pcs, clip = settings["hmm"]["pcs"], settings["walkforward"]["feature_clip_sd"]
    if min(x.shape) < pcs or pcs < 1 or not np.isfinite(clip) or clip <= 0:
        raise ValueError("Insufficient PCA training dimensions or invalid clipping")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = np.clip((x-mean)/scale, -clip, clip)
    center = z.mean(axis=0)
    _, singular, components = np.linalg.svd(z-center, full_matrices=False)
    components = components[:pcs]
    previous = None if previous_components is None else _matrix(previous_components)
    if previous is not None and previous.shape != components.shape:
        raise ValueError("PCA sign alignment requires matching component dimensions")
    for index, component in enumerate(components):
        reference = (float(component @ previous[index]) if previous is not None else 0.0)
        if reference == 0:
            reference = component[np.argmax(np.abs(component))]
        if reference < 0:
            components[index] *= -1
    return {"mean": mean.tolist(), "scale": scale.tolist(), "clip_sd": clip,
            "center": center.tolist(), "components": components.tolist(),
            "population_pc_variance": (singular[:pcs]**2/len(x)).tolist()}


def project(values: ArrayLike, fitted: dict[str, Any]) -> NDArray[np.float64]:
    """Apply the saved training moments, clip and PCA axes; learn nothing (Methodology)."""
    x = _matrix(values)
    z = np.clip((x-np.asarray(fitted["mean"]))/np.asarray(fitted["scale"]),
                -fitted["clip_sd"], fitted["clip_sd"])
    return (z-np.asarray(fitted["center"])) @ np.asarray(fitted["components"]).T


def stationary_distribution(transition: ArrayLike) -> NDArray[np.float64]:
    """Solve πP=π, Σπ=1; reject a nonunique stationary distribution (Methodology)."""
    p = _matrix(transition)
    k = len(p)
    if p.shape != (k, k) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Invalid HMM transition probabilities")
    design = np.vstack([p.T-np.eye(k), np.ones(k)])
    if np.linalg.matrix_rank(design) != k:
        raise ValueError("HMM stationary distribution is not unique")
    pi = np.linalg.lstsq(design, np.r_[np.zeros(k), 1.0], rcond=None)[0]
    if np.min(pi) < -1e-10 or not np.allclose(pi @ p, pi):
        raise ValueError("Invalid fitted stationary distribution")
    pi = np.maximum(pi, 0)
    return pi/pi.sum()


def gaussian_log_density(values: ArrayLike, means: ArrayLike, variances: ArrayLike) -> NDArray:
    """Diagonal Gaussian log density: −½Σ[log(2πv)+(x−μ)²/v] (Methodology)."""
    x, mu, variance = _matrix(values), _matrix(means), _matrix(variances)
    if mu.shape != variance.shape or x.shape[1] != mu.shape[1] or np.any(variance <= 0):
        raise ValueError("Invalid Gaussian emission dimensions or variances")
    return -.5*np.sum(np.log(2*np.pi*variance)[None, :, :]
                       +(x[:, None, :]-mu[None, :, :])**2/variance[None, :, :], axis=2)


def log_forward(
    log_emissions: ArrayLike, transition: ArrayLike, initial: ArrayLike,
) -> NDArray[np.float64]:
    """α_t ∝ density_t·(α_{t−1}P), starting from fitted start probabilities (Methodology).

    Log-sum-exp normalization avoids underflow. There is no backward recursion.
    """
    emissions, p = _matrix(log_emissions), _matrix(transition)
    initial = np.asarray(initial, dtype=float)
    k = emissions.shape[1]
    if (p.shape != (k, k) or initial.shape != (k,) or not np.isfinite(initial).all()
            or np.any(initial < 0) or np.any(p < 0)
            or not np.allclose(initial.sum(), 1) or not np.allclose(p.sum(axis=1), 1)):
        raise ValueError("Invalid forward probabilities")
    with np.errstate(divide="ignore"):
        log_p, prior = np.log(p), np.log(initial)
    result = np.empty_like(emissions)
    for index, density in enumerate(emissions):
        posterior = prior+density
        normalizer = logsumexp(posterior)
        if not np.isfinite(normalizer):
            raise ValueError("No possible HMM state at this observation")
        posterior -= normalizer
        result[index] = np.exp(posterior)
        prior = logsumexp(posterior[:, None]+log_p, axis=0)
    return result


def emission_probabilities(log_emissions: ArrayLike, stationary: ArrayLike) -> NDArray:
    """Emission-only p(k|x_t) ∝ stationary_k·density_k(x_t), without history (Methodology)."""
    emissions = _matrix(log_emissions)
    pi = np.asarray(stationary, dtype=float)
    if (pi.shape != (emissions.shape[1],) or np.any(pi < 0)
            or not np.isfinite(pi).all() or not np.isclose(pi.sum(), 1)):
        raise ValueError("Invalid emission-only stationary probabilities")
    with np.errstate(divide="ignore"):
        posterior = emissions+np.log(pi)
    return np.exp(posterior-logsumexp(posterior, axis=1)[:, None])


def fit_hmm(values: ArrayLike, settings: dict[str, Any], *, states: int = 2) -> dict[str, Any]:
    """Highest-likelihood converged diagonal Gaussian fit across fixed seeded restarts
    (Methodology)."""
    from hmmlearn.hmm import GaussianHMM

    x, cfg = _matrix(values), settings["hmm"]
    if states < 2 or len(x) <= states:
        raise ValueError("Insufficient HMM observations")
    attempts, best = [], None
    with threadpool_limits(limits=settings["walkforward"]["blas_threads"]):
        for restart in range(cfg["restarts"]):
            seed = int(cfg["seed"]+restart)
            record: dict[str, Any] = {"restart": restart, "seed": seed, "converged": False}
            try:
                model = GaussianHMM(
                    n_components=states, covariance_type="diag", random_state=seed,
                    n_iter=cfg["n_iter"], tol=cfg["tol"], min_covar=cfg["min_covar"],
                    **{name: cfg[name] for name in ("startprob_prior", "transmat_prior",
                       "means_prior", "means_weight", "covars_prior", "covars_weight")},
                ).fit(x)
                history = list(model.monitor_.history)
                gain = history[-1]-history[-2] if len(history) >= 2 else float("inf")
                likelihood = float(model.score(x))
                variances = np.diagonal(model.covars_, axis1=1, axis2=2)
                stationary = stationary_distribution(model.transmat_)
                valid = (np.isfinite(likelihood) and np.isfinite(variances).all()
                         and (variances > 0).all() and np.isfinite(model.means_).all())
                converged = bool(valid and model.monitor_.iter < cfg["n_iter"]
                                 and abs(gain) <= cfg["tol"])
                record.update(iterations=model.monitor_.iter,
                              final_gain=float(gain) if np.isfinite(gain) else None,
                              log_likelihood=likelihood if np.isfinite(likelihood) else None,
                              converged=converged,
                              status="converged" if converged else "nonconverged")
                if converged and (best is None or likelihood > best["log_likelihood"]):
                    order = np.argsort(model.means_[:, 0], kind="stable")
                    best = {"start": model.startprob_[order].tolist(),
                            "transition": model.transmat_[np.ix_(order, order)].tolist(),
                            "means": model.means_[order].tolist(),
                            "variances": variances[order].tolist(),
                            "stationary": stationary[order].tolist(),
                            "log_likelihood": likelihood, "selected_restart": restart}
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                record.update(status="numerical_failure", error=str(error))
            attempts.append(record)
    if best is None:
        raise ValueError(f"No converged HMM restart: {json.dumps(attempts, sort_keys=True)}")
    return {**best, "restarts": attempts}


def probabilities(values: ArrayLike, fit: dict[str, Any]) -> tuple[NDArray, NDArray]:
    """Return causal filtered and independent stationary-emission probabilities (Methodology)."""
    density = gaussian_log_density(values, fit["means"], fit["variances"])
    return (log_forward(density, fit["transition"], fit["start"]),
            emission_probabilities(density, fit["stationary"]))


def diagnostics(filtered: ArrayLike, fit: dict[str, Any], minimum: float) -> dict[str, Any]:
    """Training MAP occupancy, posterior occupancy and geometric mean durations (Methodology)."""
    probability = _matrix(filtered)
    states = probability.argmax(axis=1)
    occupancy = np.bincount(states, minlength=probability.shape[1])/len(states)
    diagonal = np.diag(fit["transition"])
    return {"occupancy": occupancy.tolist(), "posterior_occupancy": probability.mean(0).tolist(),
            "low_occupancy": (occupancy < minimum).tolist(),
            "mean_duration_months": [float(1/(1-p)) if p < 1 else None for p in diagonal],
            "infinite_duration": (diagonal == 1).tolist()}


def synthetic_recovery(settings: dict[str, Any]) -> tuple[dict[str, Any], pl.DataFrame]:
    """Fixed planted Markov chain with Euclidean/Mahalanobis separation 1.5 (Methodology).

    Independent simulation uses unit diagonal noise, state means ±separation/2
    on the first coordinate, and a symmetric persistent transition matrix.
    """
    cfg = settings["hmm"]
    random = np.random.default_rng(cfg["seed"])
    n, dimensions = cfg["synthetic_n"], cfg["synthetic_dims"]
    truth = np.empty(n, dtype=int)
    truth[0] = random.integers(2)
    for index in range(1, n):
        truth[index] = (truth[index-1] if random.random() < cfg["synthetic_persistence"]
                        else 1-truth[index-1])
    values = random.normal(size=(n, dimensions))
    values[:, 0] += (truth-.5)*cfg["synthetic_separation"]
    fit = fit_hmm(values, settings, states=2)
    filtered, emission = probabilities(values, fit)
    predicted = filtered.argmax(axis=1)
    confusion = np.array([[np.sum((truth == a) & (predicted == b)) for b in range(2)]
                          for a in range(2)])
    left, right = linear_sum_assignment(-confusion)
    order = right[np.argsort(left)]
    filtered, emission = filtered[:, order], emission[:, order]
    accuracy = float(np.mean(filtered.argmax(1) == truth))
    emission_accuracy = float(np.mean(emission.argmax(1) == truth))
    result = {"filtered_accuracy": accuracy, "emission_accuracy": emission_accuracy,
              "filtered_brier": float(np.mean((filtered[:, 1]-truth)**2)),
              "emission_brier": float(np.mean((emission[:, 1]-truth)**2)),
              "passed": accuracy >= cfg["synthetic_accuracy"] and accuracy > emission_accuracy,
              "fit": fit, "settings": cfg}
    return result, pl.DataFrame({"truth": truth, "filtered": filtered[:, 1],
                                 "emission": emission[:, 1]})
