"""Evaluation metrics, impulse response latency, and Pareto frontier tools."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FilterMetrics:
    mse: float
    nmse: float
    corr: float
    smoothness: float
    lag_centroid: float
    lag_cc: float
    predictive_score: float | None = None
    directional_accuracy: float = 0.0


def calc_smoothness(series: np.ndarray, u_scale: float = 1.0) -> float:
    """Calculate roughness / smoothness via normalized second-difference energy.

    Lower value = smoother series.
    """
    diff2 = np.diff(series, n=2)
    if len(diff2) == 0:
        return 0.0
    return float(np.mean(diff2**2)) / (u_scale**2)


def calc_lag_cc(true_target: np.ndarray, estimated: np.ndarray, max_lag: int = 30) -> float:
    """Estimate empirical tracking lag via cross-correlation peak.

    Convention: Positive lag indicates estimated series lags behind true_target.
    If estimated[t] ~ true_target[t - k] with k > 0, returns +k.
    """
    n = len(true_target)
    if n < max_lag * 2:
        return 0.0

    x = true_target - np.mean(true_target)
    y = estimated - np.mean(estimated)

    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for k in lags:
        if k < 0:
            # y leads x
            c = np.corrcoef(x[-k:], y[:k])[0, 1]
        elif k > 0:
            # y lags x by k
            c = np.corrcoef(x[:-k], y[k:])[0, 1]
        else:
            c = np.corrcoef(x, y)[0, 1]
        corrs.append(c if not np.isnan(c) else -1.0)

    best_idx = int(np.argmax(corrs))
    return float(lags[best_idx])


def calc_impulse_response_latency(
    F: np.ndarray,
    H: np.ndarray,
    K: np.ndarray,
    n_steps: int = 250,
) -> tuple[float, float, float]:
    """Compute linearized impulse-response metrics under frozen Kalman gain K.

    For an impulse input y_0 = 1, y_t = 0 (t >= 1):
        x_0 = K * 1
        x_k = (I - K H) F x_{k-1} = A x_{k-1}
        g_k = H x_k

    Returns:
        tau: centroid latency (positive = lag in bars, sum(k * g_k) / sum(g_k))
        noise_gain: sum (g_k)^2
        dc_gain: sum g_k (equals 1.0 for unit DC tracking)
    """
    n_states = F.shape[0]
    I_KH = np.eye(n_states) - K @ H
    A = I_KH @ F

    g = np.zeros(n_steps)
    x = K.copy()
    for k in range(n_steps):
        g[k] = float((H @ x)[0, 0])
        x = A @ x

    sum_g = float(np.sum(g))
    if abs(sum_g) > 1e-6:
        tau = float(np.sum(np.arange(n_steps) * g) / sum_g)
    else:
        tau = 0.0

    noise_gain = float(np.sum(g**2))
    return tau, noise_gain, sum_g


def calc_metrics(
    true_target: np.ndarray,
    estimated: np.ndarray,
    noise_var: float | None = None,
    innovations: np.ndarray | None = None,
    S_arr: np.ndarray | None = None,
    F: np.ndarray | None = None,
    H: np.ndarray | None = None,
    K_steady: np.ndarray | None = None,
    u_scale: float = 1.0,
) -> FilterMetrics:
    """Compute comprehensive metrics with validated lag and smoothness."""
    err = estimated - true_target
    mse = float(np.mean(err**2))

    if noise_var is None or noise_var <= 0:
        noise_var = float(np.var(err))
    nmse = mse / max(noise_var, 1e-9)

    corr_m = np.corrcoef(true_target, estimated)
    corr = float(corr_m[0, 1]) if not np.isnan(corr_m[0, 1]) else 0.0

    smoothness = calc_smoothness(estimated, u_scale=u_scale)
    lag_cc = calc_lag_cc(true_target, estimated)

    if F is not None and H is not None and K_steady is not None:
        tau_centroid, _, _ = calc_impulse_response_latency(F, H, K_steady)
    else:
        tau_centroid = lag_cc

    d_true = np.diff(true_target)
    d_est = np.diff(estimated)
    da = float(np.mean(np.sign(d_true) == np.sign(d_est))) if len(d_true) > 0 else 0.0

    pls = None
    if innovations is not None and S_arr is not None:
        valid = S_arr > 1e-12
        if np.any(valid):
            scores = -0.5 * (
                math.log(2.0 * math.pi)
                + np.log(S_arr[valid])
                + (innovations[valid] ** 2) / S_arr[valid]
            )
            pls = float(np.mean(scores))

    return FilterMetrics(
        mse=mse,
        nmse=nmse,
        corr=corr,
        smoothness=smoothness,
        lag_centroid=tau_centroid,
        lag_cc=lag_cc,
        predictive_score=pls,
        directional_accuracy=da,
    )


def calc_pareto_hypervolume(
    smoothness_vals: np.ndarray,
    lag_vals: np.ndarray,
    ref_s: float = 100.0,
    ref_lag: float = 30.0,
) -> float:
    """Compute dominated hypervolume of lower-left Pareto frontier.

    Lower S and lower Lag are preferred.
    """
    pts = sorted(zip(smoothness_vals, lag_vals), key=lambda p: p[0])
    valid_pts = [(s, lag) for s, lag in pts if s <= ref_s and lag <= ref_lag]
    if not valid_pts:
        return 0.0

    # Monotone lower-left envelope
    pareto = []
    min_lag = float("inf")
    for s, lag in valid_pts:
        if lag < min_lag:
            pareto.append((s, lag))
            min_lag = lag

    # Integrate area between pareto curve and (ref_s, ref_lag)
    area = 0.0
    for i in range(len(pareto)):
        s_curr, lag_curr = pareto[i]
        s_next = pareto[i + 1][0] if i + 1 < len(pareto) else ref_s
        width = s_next - s_curr
        height = max(ref_lag - lag_curr, 0.0)
        area += width * height

    return float(area)
