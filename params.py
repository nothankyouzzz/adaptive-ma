"""Parameter transformations, bounds, ordering constraints, and scale equivariance."""

from __future__ import annotations

import math

import numpy as np


def calc_u_scale(price: np.ndarray) -> float:
    """Robust scale anchor: 1.4826 * MAD(diff(price))."""
    dp = np.diff(price)
    if len(dp) == 0:
        return 1.0
    med = np.median(dp)
    mad = np.median(np.abs(dp - med))
    scale = 1.4826 * mad
    return float(max(scale, 1e-8))


def sigma_L(x: float | np.ndarray) -> float | np.ndarray:
    """Logistic sigmoid function with numerical clipping."""
    x_clipped = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def logit(p: float | np.ndarray) -> float | np.ndarray:
    """Logit function."""
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p_clipped / (1.0 - p_clipped))


class MultiscaleParams:
    """Transformed parameter container enforcing identifiability constraints."""

    LAMBDA_LO = 0.10
    LAMBDA_HI = math.pi - 0.10

    def __init__(
        self,
        log_sigma_eps: float = 0.0,
        phi_mu: float = -1.0,
        phi_beta: float = -3.0,
        phi_c: float = 0.0,
        phi_h: float = 0.0,
        logit_rho_h: float = 0.5,
        logit_rho_c_delta: float = 2.0,
        logit_lambda: float = 0.0,
        log_alpha_rel: float = 0.0,
    ) -> None:
        self.log_sigma_eps = float(log_sigma_eps)
        self.phi_mu = float(phi_mu)
        self.phi_beta = float(phi_beta)
        self.phi_c = float(phi_c)
        self.phi_h = float(phi_h)
        self.logit_rho_h = float(logit_rho_h)
        self.logit_rho_c_delta = float(logit_rho_c_delta)
        self.logit_lambda = float(logit_lambda)
        self.log_alpha_rel = float(log_alpha_rel)

    @classmethod
    def from_vector(cls, theta: np.ndarray) -> MultiscaleParams:
        return cls(
            log_sigma_eps=theta[0],
            phi_mu=theta[1],
            phi_beta=theta[2],
            phi_c=theta[3],
            phi_h=theta[4],
            logit_rho_h=theta[5],
            logit_rho_c_delta=theta[6],
            logit_lambda=theta[7],
            log_alpha_rel=theta[8] if len(theta) > 8 else 0.0,
        )

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.log_sigma_eps,
                self.phi_mu,
                self.phi_beta,
                self.phi_c,
                self.phi_h,
                self.logit_rho_h,
                self.logit_rho_c_delta,
                self.logit_lambda,
                self.log_alpha_rel,
            ]
        )

    @property
    def sigma_eps(self) -> float:
        return math.exp(self.log_sigma_eps)

    @property
    def sigma_mu(self) -> float:
        return self.sigma_eps * math.exp(self.phi_mu)

    @property
    def sigma_beta(self) -> float:
        return self.sigma_eps * math.exp(self.phi_beta)

    @property
    def sigma_c(self) -> float:
        return self.sigma_eps * math.exp(self.phi_c)

    @property
    def sigma_h(self) -> float:
        return self.sigma_eps * math.exp(self.phi_h)

    @property
    def rho_h(self) -> float:
        return float(sigma_L(self.logit_rho_h))

    @property
    def rho_c(self) -> float:
        # Enforces rho_h < rho_c < 1
        delta = float(sigma_L(self.logit_rho_c_delta))
        return self.rho_h + (1.0 - self.rho_h) * delta

    @property
    def cycle_lambda(self) -> float:
        delta = float(sigma_L(self.logit_lambda))
        return self.LAMBDA_LO + (self.LAMBDA_HI - self.LAMBDA_LO) * delta

    @property
    def alpha_eff(self) -> float:
        return self.sigma_eps * math.exp(self.log_alpha_rel)
