"""Tests for instrument estimators (Roll, TSRV) and impulse response metrics."""

from __future__ import annotations

import numpy as np

from bench.dgp import generate_roll_bounce
from bench.metrics import calc_impulse_response_latency
from instruments import de_drifted_roll_estimator, tsrv_estimator


def test_roll_estimator_accuracy_and_dedrifting():
    """Verify that de-drifted Roll estimator correctly recovers noise variance under drift."""
    spread = 1.0
    true_noise_var = 0.25
    s = generate_roll_bounce(n=3000, spread=spread, sigma_eff=0.01, seed=42)

    # 1. Pure bounce (no drift)
    res_clean = de_drifted_roll_estimator(s.price)
    est_var_clean = float(np.mean(res_clean.noise_var[-1000:]))
    assert abs(est_var_clean - true_noise_var) < 0.05

    # 2. Add strong price drift mu = 0.8 * spread per bar
    # Raw estimator without drift subtraction flips positive or underestimates
    drift = 0.8 * spread * np.arange(len(s.price))
    price_drifted = s.price + drift

    res_drifted = de_drifted_roll_estimator(price_drifted)
    est_var_drifted = float(np.mean(res_drifted.noise_var[-1000:]))
    assert abs(est_var_drifted - true_noise_var) < 0.08


def test_tsrv_integrated_variance_recovery():
    """Verify TSRV separates integrated variance from microstructure noise."""
    rng = np.random.default_rng(42)
    n = 2000
    sigma_w = 0.05
    sigma_eps = 0.3

    w = rng.normal(0, sigma_w, size=n)
    true_level = np.cumsum(w)
    eps = rng.normal(0, sigma_eps, size=n)
    p = true_level + eps

    iv_arr, omega2_arr = tsrv_estimator(p, window=200, k_subgrids=5)

    mean_iv_est = float(np.mean(iv_arr[-500:]))
    mean_omega2_est = float(np.mean(omega2_arr[-500:]))

    # Realized variance of returns per window
    expected_window_iv = 200 * (sigma_w**2)  # 200 * 0.0025 = 0.5
    expected_omega2 = sigma_eps**2  # 0.09

    assert abs(mean_iv_est - expected_window_iv) / expected_window_iv < 0.35
    assert abs(mean_omega2_est - expected_omega2) / expected_omega2 < 0.35


def test_impulse_response_dc_gains():
    """Check that linear impulse response has DC tracking gain = 1 and analytic latency (1-alpha)/alpha."""
    alpha = 0.181
    F = np.array([[1.0]])
    H = np.array([[1.0]])
    K = np.array([[alpha]])

    tau, noise_gain, dc_gain = calc_impulse_response_latency(F, H, K, n_steps=250)

    assert abs(dc_gain - 1.0) < 1e-4, f"DC signal gain must be 1.0, got {dc_gain}"
    tau_analytic = (1.0 - alpha) / alpha
    assert abs(tau - tau_analytic) < 1e-3, f"Tau {tau} does not match analytic {tau_analytic}"
    s_analytic = alpha / (2.0 - alpha)
    assert abs(noise_gain - s_analytic) < 1e-3, (
        f"Noise gain {noise_gain} does not match analytic {s_analytic}"
    )
