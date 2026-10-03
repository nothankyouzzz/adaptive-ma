"""Instrument estimators for Q/R channel separation.

Implements:
1. De-drifted Roll (1984) autocovariance estimator for observation noise R_t
2. Two-Scale Realized Variance (TSRV) for integrated variance IV_t (Q_t)
3. Normalized instrument signals with shrinkage and clamp diagnostics
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RollResult:
    noise_var: np.ndarray  # R_t series
    autocov: np.ndarray  # raw autocovariance series
    truncation_rate: float  # fraction of bars where autocov >= 0


def de_drifted_roll_estimator(
    prices: np.ndarray,
    ewma_decay: float = 0.98,
    r_floor: float = 1e-6,
) -> RollResult:
    """Causal, de-drifted Roll autocovariance estimator for observation noise R.

    Removes local return drift via EWMA before computing autocovariance:
        r_tilde_t = dp_t - EWMA(dp)_t
        Cov_t = decay * Cov_{t-1} + (1 - decay) * r_tilde_t * r_tilde_{t-1}
        R_t = max(-Cov_t, r_floor)
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    if n < 3:
        return RollResult(
            noise_var=np.full(n, r_floor),
            autocov=np.zeros(n),
            truncation_rate=0.0,
        )

    dp = np.diff(prices)
    dp_len = len(dp)

    # Local drift removal
    mean_dp = 0.0
    r_tilde = np.zeros(dp_len)
    for t in range(dp_len):
        mean_dp = ewma_decay * mean_dp + (1.0 - ewma_decay) * dp[t]
        r_tilde[t] = dp[t] - mean_dp

    cov_arr = np.zeros(n)
    r_arr = np.zeros(n)
    curr_cov = 0.0
    truncations = 0

    r_arr[0] = r_floor
    r_arr[1] = r_floor

    for t in range(1, dp_len):
        prod = r_tilde[t] * r_tilde[t - 1]
        curr_cov = ewma_decay * curr_cov + (1.0 - ewma_decay) * prod
        cov_arr[t + 1] = curr_cov
        if curr_cov < -r_floor:
            r_arr[t + 1] = -curr_cov
        else:
            r_arr[t + 1] = r_floor
            truncations += 1

    trunc_rate = truncations / max(dp_len - 1, 1)
    return RollResult(
        noise_var=r_arr,
        autocov=cov_arr,
        truncation_rate=float(trunc_rate),
    )


def tsrv_estimator(
    prices: np.ndarray,
    window: int = 150,
    k_subgrids: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Two-Scale Realized Variance (Zhang, Mykland, Ait-Sahalia 2005).

    Returns:
        iv_arr: integrated variance IV_t (proportional to Q)
        omega2_arr: microstructure noise variance omega^2_t (R)
    """
    prices = np.asarray(prices, dtype=float)
    n = len(prices)
    iv_arr = np.zeros(n)
    omega2_arr = np.zeros(n)

    if n < 5:
        return np.full(n, 1e-6), np.full(n, 1e-6)

    dp = np.diff(prices)

    for t in range(1, n):
        # Causal window up to t
        start_idx = max(0, t - window)
        sub_dp = dp[start_idx:t]
        m = len(sub_dp)
        if m < 4:
            iv_arr[t] = max(dp[t - 1] ** 2, 1e-6)
            omega2_arr[t] = 1e-6
            continue

        rv_all = float(np.sum(sub_dp**2))
        k = max(2, min(k_subgrids, int(math.ceil(m ** (2.0 / 3.0)))))

        rv_sub_sum = 0.0
        for i in range(k):
            grid_rets = []
            for j in range(i + k, m, k):
                grid_rets.append(np.sum(sub_dp[j - k : j]))
            if len(grid_rets) > 0:
                rv_sub_sum += float(np.sum(np.array(grid_rets) ** 2))

        rv_avg = rv_sub_sum / k
        n_bar = (m - k + 1) / k
        tsrv = rv_avg - (n_bar / m) * rv_all

        iv_arr[t] = max(tsrv, 1e-6)
        omega2_arr[t] = max((rv_all - tsrv) / max(2.0 * m, 1.0), 1e-6)

    iv_arr[0] = iv_arr[1]
    omega2_arr[0] = omega2_arr[1]
    return iv_arr, omega2_arr


def compute_instrument_signals(
    prices: np.ndarray,
    shrinkage: float = 0.5,
    c_lo: float = 0.1,
    c_hi: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compute normalized c_t (Q-channel) and d_t (R-channel) instrument multipliers.

    Uses strictly causal local EWMA normalization to prevent future leakage.
    """
    roll_res = de_drifted_roll_estimator(prices)
    iv_arr, _ = tsrv_estimator(prices)
    n = len(prices)

    # 1. R instrument from Roll with causal EWMA normalization
    r_raw = roll_res.noise_var
    mean_r_t = np.zeros(n)
    curr_m = max(r_raw[0], 1e-5)
    for t in range(n):
        curr_m = 0.98 * curr_m + 0.02 * r_raw[t]
        mean_r_t[t] = max(curr_m, 1e-6)
    r_norm = r_raw / mean_r_t

    log_d = (1.0 - shrinkage) * np.log(np.maximum(r_norm, 1e-4))
    d_raw = np.exp(log_d)
    d_clamped = np.clip(d_raw, c_lo, c_hi)

    # 2. Q instrument from TSRV IV with causal EWMA normalization
    mean_iv_t = np.zeros(n)
    curr_iv = max(iv_arr[0], 1e-5)
    for t in range(n):
        curr_iv = 0.98 * curr_iv + 0.02 * iv_arr[t]
        mean_iv_t[t] = max(curr_iv, 1e-6)
    q_norm = iv_arr / mean_iv_t

    log_c = (1.0 - shrinkage) * np.log(np.maximum(q_norm, 1e-4))
    c_raw = np.exp(log_c)
    c_clamped = np.clip(c_raw, c_lo, c_hi)

    clamp_hits_c = float(np.mean((c_raw < c_lo) | (c_raw > c_hi)))
    clamp_hits_d = float(np.mean((d_raw < c_lo) | (d_raw > c_hi)))

    diagnostics = {
        "roll_truncation_rate": roll_res.truncation_rate,
        "clamp_hits_c": clamp_hits_c,
        "clamp_hits_d": clamp_hits_d,
    }

    return c_clamped, d_clamped, diagnostics
