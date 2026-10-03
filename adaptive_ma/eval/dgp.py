"""Synthetic Data Generating Processes (DGPs) for moving average benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticSeries:
    price: np.ndarray
    true_target: np.ndarray
    imbalance: np.ndarray
    components: dict[str, np.ndarray]
    description: str


def generate_rw_noise(
    n: int = 1000,
    q: float = 0.04,
    sigma_eps: float = 1.0,
    seed: int = 42,
) -> SyntheticSeries:
    """B1: Standard Random Walk + White Noise.

    x_t = x_{t-1} + w_t,  w_t ~ N(0, q * sigma_eps^2)
    p_t = x_t + eps_t,    eps_t ~ N(0, sigma_eps^2)
    """
    rng = np.random.default_rng(seed)
    sigma_w = math.sqrt(q) * sigma_eps
    w = rng.normal(0, sigma_w, size=n)
    eps = rng.normal(0, sigma_eps, size=n)
    true_level = np.cumsum(w)
    price = true_level + eps
    imbalance = np.zeros(n)
    return SyntheticSeries(
        price=price,
        true_target=true_level,
        imbalance=imbalance,
        components={"level": true_level, "noise": eps},
        description=f"B1: Random Walk + Noise (q={q:.3f})",
    )


def generate_unobservable_two_rw(
    n: int = 1000,
    sigma_1: float = 0.05,
    sigma_2: float = 0.05,
    sigma_eps: float = 0.5,
    seed: int = 42,
) -> SyntheticSeries:
    """B2a: Two unobservable random walks + noise.

    d1_t = d1_{t-1} + w1_t
    d2_t = d2_{t-1} + w2_t
    p_t = d1_t + d2_t + eps_t
    """
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, sigma_1, size=n)
    w2 = rng.normal(0, sigma_2, size=n)
    d1 = np.cumsum(w1)
    d2 = np.cumsum(w2)
    eps = rng.normal(0, sigma_eps, size=n)
    true_sum = d1 + d2
    price = true_sum + eps
    return SyntheticSeries(
        price=price,
        true_target=true_sum,
        imbalance=np.zeros(n),
        components={"d1": d1, "d2": d2, "true_sum": true_sum, "noise": eps},
        description="B2a: Two Random Walks (Unobservable Split)",
    )


def generate_multiscale(
    n: int = 1500,
    sigma_mu: float = 0.05,
    sigma_beta: float = 0.0002,
    rho_c: float = 0.95,
    cycle_period: float = 60.0,
    sigma_c: float = 0.40,
    rho_h: float = 0.60,
    alpha_eff: float = 0.50,
    sigma_eps: float = 0.50,
    seed: int = 42,
) -> SyntheticSeries:
    """B3: Multiscale process (Macro LLT + Damped Swing Cycle + Micro AR(1) Impact + Noise).

    Rebalanced so macro drift, swing cycle, and micro impact have comparable energy.
    """
    rng = np.random.default_rng(seed)
    lam = 2.0 * math.pi / cycle_period

    # Macro local linear trend
    eta_mu = rng.normal(0, sigma_mu, size=n)
    eta_beta = rng.normal(0, sigma_beta, size=n)
    mu = np.zeros(n)
    beta = np.zeros(n)
    for t in range(1, n):
        beta[t] = beta[t - 1] + eta_beta[t]
        mu[t] = mu[t - 1] + beta[t - 1] + eta_mu[t]

    # Swing damped stochastic cycle
    eta_c = rng.normal(0, sigma_c, size=n)
    eta_c_star = rng.normal(0, sigma_c, size=n)
    c = np.zeros(n)
    c_star = np.zeros(n)
    cos_lam = math.cos(lam)
    sin_lam = math.sin(lam)
    for t in range(1, n):
        c[t] = rho_c * (cos_lam * c[t - 1] + sin_lam * c_star[t - 1]) + eta_c[t]
        c_star[t] = rho_c * (-sin_lam * c[t - 1] + cos_lam * c_star[t - 1]) + eta_c_star[t]

    # Micro order-flow impact AR(1)
    imbalance = rng.normal(0, 1.0, size=n)
    eta_h = rng.normal(0, 0.05, size=n)
    h = np.zeros(n)
    for t in range(1, n):
        h[t] = rho_h * h[t - 1] + alpha_eff * imbalance[t] + eta_h[t]

    eps = rng.normal(0, sigma_eps, size=n)

    # True target is macro + swing + micro (or macro + swing)
    true_target = mu + c + h
    price = true_target + eps

    return SyntheticSeries(
        price=price,
        true_target=true_target,
        imbalance=imbalance,
        components={
            "macro_mu": mu,
            "macro_beta": beta,
            "swing_cycle": c,
            "micro_impact": h,
            "noise": eps,
        },
        description="B3: Multiscale (LLT + Damped Cycle + Micro OFI)",
    )


def generate_garch_vol(
    n: int = 1500,
    omega: float = 0.0005,
    alpha: float = 0.1,
    beta: float = 0.85,
    sigma_eps: float = 0.5,
    seed: int = 42,
) -> SyntheticSeries:
    """B4: Fundamental volatility clustering (GARCH(1,1) level innovations)."""
    rng = np.random.default_rng(seed)
    sigma2 = np.zeros(n)
    w = np.zeros(n)
    unc_var = omega / max(1.0 - alpha - beta, 1e-4)
    sigma2[0] = unc_var

    z = rng.normal(0, 1.0, size=n)
    w[0] = math.sqrt(sigma2[0]) * z[0]

    for t in range(1, n):
        sigma2[t] = omega + alpha * (w[t - 1] ** 2) + beta * sigma2[t - 1]
        w[t] = math.sqrt(sigma2[t]) * z[t]

    true_level = np.cumsum(w)
    eps = rng.normal(0, sigma_eps, size=n)
    price = true_level + eps

    return SyntheticSeries(
        price=price,
        true_target=true_level,
        imbalance=np.zeros(n),
        components={"level": true_level, "sigma_fund": np.sqrt(sigma2), "noise": eps},
        description="B4: GARCH(1,1) Fundamental Volatility Clustering",
    )


def generate_noise_bursts(
    n: int = 1500,
    sigma_level: float = 0.05,
    sigma_eps_low: float = 0.3,
    sigma_eps_high: float = 2.0,
    burst_prob: float = 0.015,
    burst_duration: int = 10,
    seed: int = 42,
) -> SyntheticSeries:
    """B5: Microstructure noise bursts (episodic ~13% duty cycle)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, sigma_level, size=n)
    true_level = np.cumsum(w)

    in_burst = False
    dur = 0
    sigma_eps_t = np.zeros(n)
    for t in range(n):
        if not in_burst and rng.random() < burst_prob:
            in_burst = True
            dur = burst_duration
        if in_burst:
            sigma_eps_t[t] = sigma_eps_high
            dur -= 1
            if dur <= 0:
                in_burst = False
        else:
            sigma_eps_t[t] = sigma_eps_low

    eps = rng.normal(0, sigma_eps_t, size=n)
    price = true_level + eps

    return SyntheticSeries(
        price=price,
        true_target=true_level,
        imbalance=np.zeros(n),
        components={"level": true_level, "sigma_eps_t": sigma_eps_t, "noise": eps},
        description="B5: Observation Noise Bursts (Time-Varying R)",
    )


def generate_independent_vol_clusters(
    n: int = 2000,
    seed: int = 42,
) -> SyntheticSeries:
    """B6: Independent Fundamental Volatility (GARCH Q) & Microstructure Noise Bursts (R).

    The headline identification benchmark.
    """
    s_garch = generate_garch_vol(n=n, seed=seed)
    s_burst = generate_noise_bursts(n=n, seed=seed + 100)

    true_level = s_garch.components["level"]
    sigma_fund = s_garch.components["sigma_fund"]
    sigma_eps_t = s_burst.components["sigma_eps_t"]
    rng = np.random.default_rng(seed + 200)
    eps = rng.normal(0, sigma_eps_t, size=n)
    price = true_level + eps

    return SyntheticSeries(
        price=price,
        true_target=true_level,
        imbalance=np.zeros(n),
        components={
            "level": true_level,
            "sigma_fund": sigma_fund,
            "sigma_eps_t": sigma_eps_t,
            "noise": eps,
        },
        description="B6: Independent Fundamental Vol (Q) & Noise Bursts (R)",
    )


def generate_roll_bounce(
    n: int = 2000,
    sigma_eff: float = 0.02,
    spread: float = 1.0,
    seed: int = 42,
) -> SyntheticSeries:
    """B9: Pure Roll Bid-Ask Bounce (Null Control for 'Not an Alpha').

    m_t = m_{t-1} + w_t (efficient price random walk)
    p_t = m_t + 0.5 * spread * q_t, where q_t in {-1, +1} is trade sign.
    """
    rng = np.random.default_rng(seed)
    w = rng.normal(0, sigma_eff, size=n)
    m = np.cumsum(w)
    trade_signs = rng.choice([-1.0, 1.0], size=n)
    noise = 0.5 * spread * trade_signs
    price = m + noise

    return SyntheticSeries(
        price=price,
        true_target=m,
        imbalance=np.zeros(n),
        components={"efficient_price": m, "trade_signs": trade_signs, "noise": noise},
        description="B9: Pure Roll Bid-Ask Bounce (Null Alpha Control)",
    )
