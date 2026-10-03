"""Generic State-Space Kalman Filter Engine.

Implements:
1. P1: Scale equivariance via robust input normalization
2. P2: Orthogonal knob separation: Q(xi, c) = c * e^{+xi} * Q_0, R(xi, c) = c * e^{-xi} * R_0
3. Stationary Lyapunov initialization for stable blocks (cycle, AR1)
4. Diffuse initialization for unit-root blocks (LLT)
5. Joseph-form covariance update with strict symmetrization
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg


@dataclass
class SSMOutput:
    state_traj: np.ndarray  # (N, n_states)
    y_pred: np.ndarray  # (N,)
    ma: np.ndarray  # (N,)
    innovations: np.ndarray  # (N,)
    S: np.ndarray  # (N,)
    K_traj: np.ndarray  # (N, n_states)
    loglik_per_bar: np.ndarray  # (N,)
    total_loglik: float
    P_diag_traj: np.ndarray  # (N, n_states)
    P_traj: np.ndarray  # (N, n_states, n_states) full covariance trajectory
    z_beta: np.ndarray  # (N,) t-ratio beta_t / sqrt(P_beta_beta)
    u_scale: float


def van_loan_discretization(A: np.ndarray, Qc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Van Loan (1978) closed-form matrix fraction discretization of continuous-time SDE:

        dx = A x dt + dW,  E[dW dW^T] = Qc dt

    Returns:
        F(dt) = expm(A * dt)
        Q(dt) = int_0^dt expm(A * s) Qc expm(A^T * s) ds
    """
    n = A.shape[0]
    dt = float(dt)
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A * dt
    M[:n, n:] = Qc * dt
    M[n:, n:] = A.T * dt

    E = scipy.linalg.expm(M)
    E22 = E[n:, n:]
    E12 = E[:n, n:]

    F_dt = E22.T
    Q_dt = F_dt @ E12
    Q_dt = 0.5 * (Q_dt + Q_dt.T)
    return F_dt, Q_dt


class ContinuousTimeSpec:
    """Continuous-Time State-Space Specification with physical time constants.

    Allows exact, invariant discretization to any time step dt (e.g. 1s, 1m, 1h).
    """

    def __init__(
        self,
        A: np.ndarray,
        Qc: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        R0: np.ndarray,
        unstable_dim: int = 0,
        output_indices: list[int] | None = None,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.Qc = np.asarray(Qc, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.R0 = np.asarray(R0, dtype=float)
        self.unstable_dim = int(unstable_dim)
        self.n_states = self.A.shape[0]
        self.output_indices = output_indices

    def discretize(self, dt: float = 1.0) -> StateSpaceSpec:
        """Discretize continuous-time model to step dt using Van Loan's method."""
        F_dt, Q_dt = van_loan_discretization(self.A, self.Qc, dt)
        B_dt = self.B * float(dt)
        return StateSpaceSpec(
            F=F_dt,
            B=B_dt,
            H=self.H,
            Q0=Q_dt,
            R0=self.R0,
            unstable_dim=self.unstable_dim,
            output_indices=self.output_indices,
        )


class StateSpaceSpec:
    """State-space model specification (F, B, H, Q0, R0, stable_indices)."""

    def __init__(
        self,
        F: np.ndarray,
        B: np.ndarray,
        H: np.ndarray,
        Q0: np.ndarray,
        R0: np.ndarray,
        unstable_dim: int = 0,
        output_indices: list[int] | None = None,
    ) -> None:
        self.F = np.asarray(F, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q0 = np.asarray(Q0, dtype=float)
        self.R0 = np.asarray(R0, dtype=float)
        self.unstable_dim = int(unstable_dim)
        self.n_states = self.F.shape[0]
        if output_indices is None:
            # By default all states with non-zero H contribute to MA
            self.output_indices = [i for i in range(self.n_states) if abs(self.H[0, i]) > 1e-9]
        else:
            self.output_indices = output_indices

    def init_covariance(self, kappa_diffuse: float = 1e6) -> np.ndarray:
        """Initialize P0: Lyapunov for stable block, diffuse for unstable block."""
        P0 = np.zeros((self.n_states, self.n_states))
        d = self.unstable_dim

        # Unstable block: diffuse
        if d > 0:
            P0[:d, :d] = np.eye(d) * kappa_diffuse

        # Stable block: discrete Lyapunov P_s = F_s P_s F_s^T + Q_s
        if d < self.n_states:
            F_s = self.F[d:, d:]
            Q_s = self.Q0[d:, d:]
            try:
                P_s = scipy.linalg.solve_discrete_lyapunov(F_s, Q_s)
                P0[d:, d:] = 0.5 * (P_s + P_s.T)
            except Exception:
                # Fallback if Lyapunov fails (e.g. near unit root)
                P0[d:, d:] = np.diag(np.diag(Q_s) / np.maximum(1.0 - np.diag(F_s) ** 2, 1e-4))

        return P0


def run_filter(
    spec: StateSpaceSpec,
    price: np.ndarray,
    imbalance: np.ndarray | None = None,
    xi: float | np.ndarray = 0.0,
    c: float | np.ndarray = 1.0,
    u_scale: float | None = None,
    burn_in: int = 15,
    normalize: bool = True,
) -> SSMOutput:
    """Run state space Kalman filter with P2 knob separation.

    Parameters
    ----------
    spec : StateSpaceSpec
    price : 1D array of observed prices
    imbalance : 1D array of order flow controls (or None)
    xi : float or 1D array (responsiveness knob, scales Q/R)
    c : float or 1D array (calibration knob, scales total variance)
    u_scale : float, normalization scale (if None, estimated via MAD)
    burn_in : int, number of initial bars to exclude from total log-likelihood
    normalize : bool, if True apply scale equivariance transform
    """
    price = np.asarray(price, dtype=float)
    n = len(price)
    if imbalance is None:
        imbalance = np.zeros(n)
    imbalance = np.asarray(imbalance, dtype=float)

    if u_scale is None:
        from params import calc_u_scale

        u_scale = calc_u_scale(price) if normalize else 1.0

    p0 = price[0] if (normalize and n > 0) else 0.0
    y_norm = (price - p0) / u_scale if normalize else price.copy()
    u_norm = np.asarray(imbalance, dtype=float).copy()

    # Expand xi and c if scalar
    if isinstance(xi, (int, float, np.number)):
        xi_arr = np.full(n, float(xi))
    else:
        xi_arr = np.asarray(xi, dtype=float)

    if isinstance(c, (int, float, np.number)):
        c_arr = np.full(n, float(c))
    else:
        c_arr = np.asarray(c, dtype=float)

    n_states = spec.n_states
    x = np.zeros((n_states, 1))
    if spec.unstable_dim > 0 and n > 0:
        # Warm-start the level mean state to y_norm[0]
        x[0, 0] = y_norm[0]

    c_0 = float(c_arr[0]) if len(c_arr) > 0 else 1.0
    P = spec.init_covariance() * c_0

    state_traj = np.zeros((n, n_states))
    y_pred_arr = np.zeros(n)
    innovations = np.zeros(n)
    S_arr = np.zeros(n)
    K_traj = np.zeros((n, n_states))
    loglik_arr = np.zeros(n)
    P_diag_traj = np.zeros((n, n_states))
    P_traj = np.zeros((n, n_states, n_states))
    z_beta_arr = np.zeros(n)

    for t in range(n):
        # Time-varying Q and R from P2 knobs
        c_t = float(c_arr[t])
        xi_t = float(xi_arr[t])
        exp_xi = math.exp(np.clip(xi_t, -15.0, 15.0))

        Q_t = c_t * exp_xi * spec.Q0
        R_val = float(c_t * (1.0 / exp_xi) * spec.R0[0, 0])
        R_val = max(R_val, 1e-12)

        # 1. Predict
        u_val = u_norm[t]
        x_pred = spec.F @ x + spec.B * u_val
        P_pred = spec.F @ P @ spec.F.T + Q_t

        # 2. Innovation
        y_pred = float((spec.H @ x_pred)[0, 0])
        S_val = float((spec.H @ P_pred @ spec.H.T)[0, 0] + R_val)
        S_val = max(S_val, 1e-12)

        nu = float(y_norm[t] - y_pred)
        K = P_pred @ spec.H.T / S_val

        # 3. State update
        x = x_pred + K * nu

        # 4. Joseph-form covariance update
        I_KH = np.eye(n_states) - K @ spec.H
        P_post = I_KH @ P_pred @ I_KH.T + (K * R_val) @ K.T
        P = 0.5 * (P_post + P_post.T)

        # Per-bar log-likelihood
        ll = -0.5 * (math.log(2.0 * math.pi) + math.log(S_val) + (nu**2) / S_val)

        state_traj[t] = x.ravel()
        y_pred_arr[t] = y_pred
        innovations[t] = nu
        S_arr[t] = S_val
        K_traj[t] = K.ravel()
        loglik_arr[t] = ll
        P_diag_traj[t] = np.diag(P)
        P_traj[t] = P.copy()
        if n_states > 1:
            z_beta_arr[t] = float(x[1, 0] / max(math.sqrt(max(P[1, 1], 1e-12)), 1e-8))

    # Accumulate log-likelihood after burn-in
    valid_mask = np.zeros(n, dtype=bool)
    if n > burn_in:
        valid_mask[burn_in:] = True
    total_loglik = float(np.sum(loglik_arr[valid_mask]))

    # Scale back to original units
    if normalize:
        state_traj_unnorm = state_traj * u_scale
        # First state had p0 offset
        if spec.unstable_dim > 0:
            state_traj_unnorm[:, 0] += p0
        y_pred_unnorm = y_pred_arr * u_scale + p0
        innovations_unnorm = innovations * u_scale
        S_unnorm = S_arr * (u_scale**2)
    else:
        state_traj_unnorm = state_traj
        y_pred_unnorm = y_pred_arr
        innovations_unnorm = innovations
        S_unnorm = S_arr

    # Moving average output: sum of designated output components
    ma_unnorm = np.sum(state_traj_unnorm[:, spec.output_indices], axis=1)

    return SSMOutput(
        state_traj=state_traj_unnorm,
        y_pred=y_pred_unnorm,
        ma=ma_unnorm,
        innovations=innovations_unnorm,
        S=S_unnorm,
        K_traj=K_traj,
        loglik_per_bar=loglik_arr,
        total_loglik=total_loglik,
        P_diag_traj=P_diag_traj,
        P_traj=P_traj,
        z_beta=z_beta_arr,
        u_scale=u_scale,
    )
