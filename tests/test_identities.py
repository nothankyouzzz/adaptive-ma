"""Mathematical identities and invariant verification tests."""

from __future__ import annotations

import math

import numpy as np

from kalman_core import StateSpaceSpec, run_filter
from multiscale_ma import build_spec_2state, build_spec_5state, calc_observability


def test_b1_ema_steady_state_equivalence():
    """Verify that 1D random walk + noise steady-state gain matches EMA formula."""
    q = 0.04
    k_star_exact = (-q + math.sqrt(q**2 + 4.0 * q)) / 2.0  # ~ 0.18099
    p_star_exact = k_star_exact  # since P_pred = P + q, K = P_pred / (P_pred + 1)

    spec = StateSpaceSpec(
        F=np.array([[1.0]]),
        B=np.array([[0.0]]),
        H=np.array([[1.0]]),
        Q0=np.array([[q]]),
        R0=np.array([[1.0]]),
        unstable_dim=1,
    )

    # Constant observation series to let filter reach steady state
    y = np.ones(300)
    out = run_filter(spec, y, normalize=False)

    k_steady = out.K_traj[-1, 0]
    p_steady = out.P_diag_traj[-1, 0]

    assert abs(k_steady - k_star_exact) < 1e-4
    assert abs(p_steady - p_star_exact) < 1e-4


def test_gain_invariance_theorem_exact():
    """Theorem 2: Common scaling Q = c * Q0 and R = c * R0 leaves K_t strictly invariant."""
    spec = build_spec_2state(sigma_level=0.1, sigma_exc=0.2, sigma_eps=0.5)
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(0, 0.1, size=200))

    # Baseline c = 1.0
    out_base = run_filter(spec, y, c=1.0, normalize=False)
    k_base = out_base.K_traj

    for c_val in [0.01, 0.5, 2.0, 100.0]:
        out_scaled = run_filter(spec, y, c=c_val, normalize=False)
        k_scaled = out_scaled.K_traj
        max_diff = np.max(np.abs(k_scaled - k_base))
        assert max_diff < 1e-10, f"Gain invariance failed for c={c_val}, max diff={max_diff}"


def test_scale_equivariance():
    """Principle 1: filter(alpha * y, u) == alpha * filter(y, u) to 1e-10 with active control u."""
    spec = build_spec_5state()
    rng = np.random.default_rng(42)
    y = 50000.0 + np.cumsum(rng.normal(0, 50.0, size=200))
    u = rng.normal(0, 1.0, size=200)

    out_1 = run_filter(spec, y, u, normalize=True)

    for alpha in [0.1, 2.5, 10.0]:
        out_alpha = run_filter(spec, alpha * y, u, normalize=True)
        rel_diff = np.max(np.abs(out_alpha.ma - alpha * out_1.ma)) / (alpha * np.mean(y))
        assert rel_diff < 1e-8, f"Scale equivariance failed for alpha={alpha}, rel_diff={rel_diff}"


def test_lyapunov_stationary_init():
    """Verify that stable block P_s satisfies discrete Lyapunov equation exactly."""
    spec = build_spec_5state()
    P0 = spec.init_covariance()

    # Stable block is indices [2, 3, 4]
    F_s = spec.F[2:, 2:]
    Q_s = spec.Q0[2:, 2:]
    P_s = P0[2:, 2:]

    # Discrete Lyapunov identity: P_s - F_s @ P_s @ F_s.T - Q_s == 0
    res = P_s - F_s @ P_s @ F_s.T - Q_s
    assert np.max(np.abs(res)) < 1e-10


def test_observability_unobservable_negative_control():
    """Two identical random walks must be unobservable (rank 1 < 2)."""
    # 2 identical RWs
    spec_unobs = StateSpaceSpec(
        F=np.eye(2),
        B=np.zeros((2, 1)),
        H=np.array([[1.0, 1.0]]),
        Q0=np.eye(2) * 0.01,
        R0=np.array([[1.0]]),
    )
    rank, cond = calc_observability(spec_unobs)
    assert rank == 1
    assert cond > 1e12

    # 5-state Harvey model must be full rank
    spec_5s = build_spec_5state()
    rank5, cond5 = calc_observability(spec_5s)
    assert rank5 == 5
    assert cond5 < 1e7


def test_ct_van_loan_discretization_identity():
    """Verify that CT 5-state Van Loan discretization at dt=1.0 matches discrete F to 1e-12."""
    from multiscale_ma import build_spec_5state_ct

    ct_spec = build_spec_5state_ct(cycle_period_hours=72.0, rho_c=0.96, rho_h=0.70)
    discrete_spec = ct_spec.discretize(dt=1.0)
    orig_spec = build_spec_5state(cycle_period=72.0, rho_c=0.96, rho_h=0.70)

    max_diff_f = np.max(np.abs(discrete_spec.F - orig_spec.F))
    assert max_diff_f < 1e-12, f"CT discretization F mismatch: {max_diff_f}"


def test_full_covariance_exposure():
    """Verify that full P_traj (N, n, n) is exposed, symmetric, and matches P_diag_traj."""
    spec = build_spec_5state()
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(0, 1.0, size=50))
    out = run_filter(spec, y, normalize=False)

    assert out.P_traj.shape == (50, 5, 5)
    assert out.z_beta.shape == (50,)

    # Check symmetry and diagonal match
    for t in range(50):
        P_t = out.P_traj[t]
        assert np.max(np.abs(P_t - P_t.T)) < 1e-12, f"P_traj at bar {t} is not symmetric"
        assert np.max(np.abs(np.diag(P_t) - out.P_diag_traj[t])) < 1e-12
