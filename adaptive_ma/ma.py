"""A model-based adaptive moving average (state-space Kalman filter).

Instead of a hand-picked SMA/EMA window, this estimates a latent "fair value"
level from a small state-space model:

    price_t = level_t + excitation_t + noise_t

    level_t      = level_{t-1}      + lam_perm * softthr(u_t, c) + w_level
    excitation_t = rho * excitation_{t-1} + alpha_eff * u_t       + w_exc

    u_t = signed order-flow imbalance (observable; optional)
    softthr(u, c) = sign(u) * max(|u| - c, 0)

The filter output ``level + excitation`` is the adaptive moving average.

``AdaptiveKalmanMA`` additionally lets large order-flow shocks feed a
*permanent* term into the level and lets the level's process noise grow with
shock size, so the MA can absorb regime shifts instead of lagging them.

.. warning::
   This is a denoising / decomposition tool, NOT an alpha generator. The
   residual ``price - (level + excitation)`` is mostly observation noise and
   should not be traded as a mean-reversion signal (that apparent edge is a
   mechanical noise bounce). See the README.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


def _softthr(u: float, c: float) -> float:
    return float(np.sign(u) * max(abs(u) - c, 0.0))


class StepResult(NamedTuple):
    level: float
    excitation: float
    innovation: float
    S: float
    loglik: float
    K: np.ndarray


class FilterResult(NamedTuple):
    level: np.ndarray
    excitation: np.ndarray
    residual: np.ndarray
    innovations: np.ndarray
    S: np.ndarray
    loglik: np.ndarray
    K: np.ndarray


class KalmanMA:
    """Fixed state-space moving average.

    State is ``[level, excitation]``. The level is a random walk (the slow
    fair-value component); the excitation is an AR(1) driven by the signed
    order-flow imbalance (the transient impact component).
    """

    def __init__(
        self,
        rho: float,
        alpha_eff: float,
        sigma_level: float = 1e-2,
        sigma_exc: float = 0.2,
        sigma_eps: float = 1.0,
    ) -> None:
        self.rho = float(rho)
        self.alpha_eff = float(alpha_eff)
        self.sigma_level = float(sigma_level)
        self.sigma_exc = float(sigma_exc)
        self.sigma_eps = float(sigma_eps)
        self.F = np.array([[1.0, 0.0], [0.0, self.rho]])
        self.B = np.array([[0.0], [self.alpha_eff]])
        self.H = np.array([[1.0, 1.0]])
        self.Q = np.diag([self.sigma_level**2, self.sigma_exc**2])
        self.R = np.array([[self.sigma_eps**2]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.last_result: StepResult | None = None

    def step_full(self, u: float, y: float) -> StepResult:
        """Run one filter step returning complete diagnostics."""
        x_pred = self.F @ self.x + self.B * u
        P_pred = self.F @ self.P @ self.F.T + self.Q

        y_pred = float((self.H @ x_pred)[0, 0])
        S_val = float((self.H @ P_pred @ self.H.T + self.R)[0, 0])
        K = P_pred @ self.H.T / S_val

        innovation = float(y - y_pred)
        self.x = x_pred + K * innovation

        # Joseph-form covariance update with symmetrization
        I_KH = np.eye(2) - K @ self.H
        P_post = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (P_post + P_post.T)

        loglik = -0.5 * (
            math.log(2.0 * math.pi)
            + math.log(max(S_val, 1e-12))
            + (innovation**2) / max(S_val, 1e-12)
        )
        res = StepResult(
            level=float(self.x[0, 0]),
            excitation=float(self.x[1, 0]),
            innovation=innovation,
            S=S_val,
            loglik=loglik,
            K=K.copy(),
        )
        self.last_result = res
        return res

    def step(self, u: float, y: float) -> tuple[float, float]:
        """Run one filter step; returns ``(level, excitation)``."""
        res = self.step_full(u, y)
        return res.level, res.excitation


class AdaptiveKalmanMA:
    """Adaptive MA: large shocks permanently shift the level.

    Adds two mechanisms on top of ``KalmanMA``:

    * ``lam_perm * softthr(u_t, c)`` feeds a permanent term into the level,
      so large shocks change the trend while small shocks only excite
      transiently.
    * the level's process noise grows with shock size,
      ``sigma_level_t = sigma_level + gamma * |softthr(u_t, c)|``.
    """

    def __init__(
        self,
        rho: float,
        alpha_eff: float,
        lam_perm: float = 0.0,
        threshold: float = 0.0,
        sigma_level: float = 1e-2,
        sigma_exc: float = 0.2,
        sigma_eps: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        self.rho = float(rho)
        self.alpha_eff = float(alpha_eff)
        self.lam_perm = float(lam_perm)
        self.threshold = float(threshold)
        self.gamma = float(gamma)
        self.sigma_level = float(sigma_level)
        self.sigma_exc = float(sigma_exc)
        self.sigma_eps = float(sigma_eps)
        self.F = np.array([[1.0, 0.0], [0.0, self.rho]])
        self.H = np.array([[1.0, 1.0]])
        self.R = np.array([[self.sigma_eps**2]])
        self.x = np.zeros((2, 1))
        self.P = np.eye(2)
        self.last_result: StepResult | None = None

    def step_full(self, u: float, y: float) -> StepResult:
        """Run one filter step returning complete diagnostics."""
        g = _softthr(u, self.threshold)
        B = np.array([[self.lam_perm, 0.0], [0.0, self.alpha_eff]])
        u_vec = np.array([[g], [u]])
        sigma_level_t = self.sigma_level + self.gamma * abs(g)
        Q = np.diag([sigma_level_t**2, self.sigma_exc**2])

        x_pred = self.F @ self.x + B @ u_vec
        P_pred = self.F @ self.P @ self.F.T + Q

        y_pred = float((self.H @ x_pred)[0, 0])
        S_val = float((self.H @ P_pred @ self.H.T + self.R)[0, 0])
        K = P_pred @ self.H.T / S_val

        innovation = float(y - y_pred)
        self.x = x_pred + K * innovation

        # Joseph-form covariance update with symmetrization
        I_KH = np.eye(2) - K @ self.H
        P_post = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        self.P = 0.5 * (P_post + P_post.T)

        loglik = -0.5 * (
            math.log(2.0 * math.pi)
            + math.log(max(S_val, 1e-12))
            + (innovation**2) / max(S_val, 1e-12)
        )
        res = StepResult(
            level=float(self.x[0, 0]),
            excitation=float(self.x[1, 0]),
            innovation=innovation,
            S=S_val,
            loglik=loglik,
            K=K.copy(),
        )
        self.last_result = res
        return res

    def step(self, u: float, y: float) -> tuple[float, float]:
        """Run one filter step; returns ``(level, excitation)``."""
        res = self.step_full(u, y)
        return res.level, res.excitation


def filter_price(price, imbalance=None, adaptive: bool = True, return_full: bool = False, **kwargs):
    """Run the filter over a series and return the decomposition.

    Parameters
    ----------
    price : array_like
        Price series.
    imbalance : array_like or None
        Signed order-flow imbalance, same length as ``price``. If ``None``,
        zeros are used (the excitation is then driven only by its own noise).
    adaptive : bool
        Use :class:`AdaptiveKalmanMA` (``True``) or :class:`KalmanMA`
        (``False``).
    return_full : bool
        If ``True``, return :class:`FilterResult` containing innovations, S,
        loglik, and K.
    **kwargs
        Passed to the filter constructor (e.g. ``rho``, ``alpha_eff``).

    Returns
    -------
    level, excitation, residual : ndarray (or FilterResult if return_full=True)
        The moving average is ``level + excitation``.
    """
    price = np.asarray(price, dtype=float)
    n = len(price)
    if imbalance is None:
        imbalance = np.zeros(n)
    imbalance = np.asarray(imbalance, dtype=float)

    cls = AdaptiveKalmanMA if adaptive else KalmanMA
    kf = cls(**kwargs)
    if n > 0:
        kf.x[0, 0] = price[0]

    level = np.zeros(n)
    excitation = np.zeros(n)
    innovations = np.zeros(n)
    S_arr = np.zeros(n)
    loglik_arr = np.zeros(n)
    K_arr = np.zeros((n, 2))

    for t in range(n):
        res = kf.step_full(imbalance[t], price[t])
        level[t] = res.level
        excitation[t] = res.excitation
        innovations[t] = res.innovation
        S_arr[t] = res.S
        loglik_arr[t] = res.loglik
        K_arr[t] = res.K.ravel()

    residual = price - (level + excitation)

    if return_full:
        return FilterResult(
            level=level,
            excitation=excitation,
            residual=residual,
            innovations=innovations,
            S=S_arr,
            loglik=loglik_arr,
            K=K_arr,
        )
    return level, excitation, residual
