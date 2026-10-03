"""Multiscale State-Space Moving Average models (Unified Specification).

Uses kalman_core.py engine with P1 scale equivariance and P2 knob separation.
"""

from __future__ import annotations

import math

import numpy as np

from kalman_core import StateSpaceSpec, run_filter
from params import MultiscaleParams


def build_spec_2state(
    rho: float = 0.8,
    alpha_eff: float = 0.0,
    sigma_level: float = 0.1,
    sigma_exc: float = 0.01,
    sigma_eps: float = 1.0,
) -> StateSpaceSpec:
    """Baseline 2-state specification [level, excitation]."""
    F = np.array([[1.0, 0.0], [0.0, float(rho)]])
    B = np.array([[0.0], [float(alpha_eff)]])
    H = np.array([[1.0, 1.0]])
    Q0 = np.diag([float(sigma_level) ** 2, float(sigma_exc) ** 2])
    R0 = np.array([[float(sigma_eps) ** 2]])
    return StateSpaceSpec(F, B, H, Q0, R0, unstable_dim=1, output_indices=[0, 1])


def build_spec_3state(
    rho_m: float = 0.96,
    rho_h: float = 0.6,
    alpha_eff: float = 0.4,
    sigma_d: float = 0.05,
    sigma_m: float = 0.1,
    sigma_h: float = 0.1,
    sigma_eps: float = 0.5,
) -> StateSpaceSpec:
    """3-state specification [macro RW, swing OU, micro AR1]."""
    F = np.diag([1.0, float(rho_m), float(rho_h)])
    B = np.array([[0.0], [0.0], [float(alpha_eff)]])
    H = np.array([[1.0, 1.0, 1.0]])
    Q0 = np.diag([float(sigma_d) ** 2, float(sigma_m) ** 2, float(sigma_h) ** 2])
    R0 = np.array([[float(sigma_eps) ** 2]])
    return StateSpaceSpec(F, B, H, Q0, R0, unstable_dim=1, output_indices=[0, 1, 2])


def build_spec_5state(
    params: MultiscaleParams | None = None,
    cycle_period: float = 45.0,
    rho_c: float = 0.96,
    rho_h: float = 0.6,
    alpha_eff: float = 0.4,
    sigma_mu: float = 0.05,
    sigma_beta: float = 0.002,
    sigma_c: float = 0.15,
    sigma_h: float = 0.1,
    sigma_eps: float = 0.5,
) -> StateSpaceSpec:
    """5-state Harvey specification [mu, beta, c, c*, h]."""
    if params is not None:
        from estimate import build_spec_from_params

        return build_spec_from_params(params)

    lam = 2.0 * math.pi / max(cycle_period, 2.0)
    cos_l = math.cos(lam)
    sin_l = math.sin(lam)

    F = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, rho_c * cos_l, rho_c * sin_l, 0.0],
            [0.0, 0.0, -rho_c * sin_l, rho_c * cos_l, 0.0],
            [0.0, 0.0, 0.0, 0.0, rho_h],
        ]
    )
    B = np.array([[0.0], [0.0], [0.0], [0.0], [alpha_eff]])
    H = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    Q0 = np.diag(
        [
            sigma_mu**2,
            sigma_beta**2,
            sigma_c**2,
            sigma_c**2,
            sigma_h**2,
        ]
    )
    R0 = np.array([[sigma_eps**2]])
    return StateSpaceSpec(F, B, H, Q0, R0, unstable_dim=2, output_indices=[0, 2, 4])


def build_spec_5state_ct(
    cycle_period_hours: float = 72.0,
    rho_c: float = 0.96,
    rho_h: float = 0.70,
    alpha_eff: float = 0.40,
    sigma_mu: float = 0.08,
    sigma_beta: float = 0.001,
    sigma_c: float = 0.40,
    sigma_h: float = 0.15,
    sigma_eps: float = 0.50,
):
    """Continuous-Time 5-state specification with physical time constants."""
    from kalman_core import ContinuousTimeSpec

    lam = 2.0 * math.pi / max(cycle_period_hours, 2.0)
    A = np.array(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, math.log(max(rho_c, 1e-6)), lam, 0.0],
            [0.0, 0.0, -lam, math.log(max(rho_c, 1e-6)), 0.0],
            [0.0, 0.0, 0.0, 0.0, math.log(max(rho_h, 1e-6))],
        ]
    )
    Qc = np.diag(
        [
            sigma_mu**2,
            sigma_beta**2,
            sigma_c**2,
            sigma_c**2,
            sigma_h**2,
        ]
    )
    B = np.array([[0.0], [0.0], [0.0], [0.0], [alpha_eff]])
    H = np.array([[1.0, 0.0, 1.0, 0.0, 1.0]])
    R0 = np.array([[sigma_eps**2]])
    return ContinuousTimeSpec(A, Qc, B, H, R0, unstable_dim=2, output_indices=[0, 2, 4])


def calc_observability(spec: StateSpaceSpec) -> tuple[int, float]:
    """Compute rank and condition number of observability matrix [H; HF; ...]."""
    n = spec.n_states
    obs_mat = np.zeros((n, n))
    HFk = spec.H.copy()
    for i in range(n):
        obs_mat[i, :] = HFk[0, :]
        HFk = HFk @ spec.F
    rank = int(np.linalg.matrix_rank(obs_mat))
    cond = float(np.linalg.cond(obs_mat))
    return rank, cond


def filter_multiscale(
    price: np.ndarray,
    imbalance: np.ndarray | None = None,
    spec: StateSpaceSpec | None = None,
    xi: float | np.ndarray = 0.0,
    c: float | np.ndarray = 1.0,
    **kwargs,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Run multiscale filter via kalman_core engine."""
    if spec is None:
        spec = build_spec_5state(**kwargs)

    out = run_filter(spec, price, imbalance, xi=xi, c=c)
    rank, cond = calc_observability(spec)

    diagnostics = {
        "innovations": out.innovations,
        "S": out.S,
        "total_loglik": out.total_loglik,
        "loglik_per_bar": out.loglik_per_bar,
        "K_traj": out.K_traj,
        "state_traj": out.state_traj,
        "obs_rank": rank,
        "obs_cond": cond,
    }
    return out.ma, diagnostics
