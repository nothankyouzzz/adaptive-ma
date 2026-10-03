"""Instrument-Driven Volatility Adaptive Moving Average.

Implements the 4-arm experimental design:
1. 'fixed': Constant Q and R baseline (c_t = 1, d_t = 1)
2. 'naive_rv': Common scaling (c_t = c_hat, d_t = c_hat) -> Gain Invariance test
3. 'instrument': True decoupled channels (c_t = c_hat, d_t = d_hat)
4. 'swapped': Attribution negative control (c_t = d_hat, d_t = c_hat)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instruments import compute_instrument_signals
from kalman_core import StateSpaceSpec, run_filter
from multiscale_ma import build_spec_2state


@dataclass
class VolFilterResult:
    ma: np.ndarray
    c_arr: np.ndarray
    d_arr: np.ndarray
    xi_arr: np.ndarray
    innovations: np.ndarray
    S: np.ndarray
    total_loglik: float
    diagnostics: dict[str, float]


def filter_volatility_instrument(
    price: np.ndarray,
    imbalance: np.ndarray | None = None,
    mode: str = "instrument",
    spec: StateSpaceSpec | None = None,
    kappa: float = 1.0,
    xi_bar: float = 0.0,
    shrinkage: float = 0.5,
    c_lo: float = 0.1,
    c_hi: float = 10.0,
) -> VolFilterResult:
    """Run instrument-driven volatility filter with 4 experimental arms.

    Parameters
    ----------
    price : 1D array of prices
    imbalance : 1D array of order flow
    mode : 'fixed', 'naive_rv', 'instrument', 'swapped'
    spec : StateSpaceSpec (if None, defaults to build_spec_2state())
    kappa : sensitivity multiplier on instrument difference signal
    xi_bar : baseline responsiveness knob
    shrinkage : shrinkage factor on instruments toward mean
    c_lo, c_hi : clamp bounds on instruments
    """
    price = np.asarray(price, dtype=float)
    n = len(price)
    if spec is None:
        spec = build_spec_2state()

    c_hat, d_hat, diag = compute_instrument_signals(
        price, shrinkage=shrinkage, c_lo=c_lo, c_hi=c_hi
    )

    if mode == "fixed":
        c_eff = np.ones(n)
        d_eff = np.ones(n)
    elif mode == "naive_rv":
        # Scale both channels identically by c_hat (Gain Invariance Theorem test)
        c_eff = c_hat.copy()
        d_eff = c_hat.copy()
    elif mode == "swapped":
        # Swapped attribution control: feed noise instrument to Q and IV instrument to R
        c_eff = d_hat.copy()
        d_eff = c_hat.copy()
    else:  # 'instrument'
        c_eff = c_hat.copy()
        d_eff = d_hat.copy()

    # In kalman_core:
    # Q(xi, c) = c * exp(xi) * Q0
    # R(xi, c) = c * exp(-xi) * R0
    # To match c_eff on Q and d_eff on R:
    # c_eff = c * exp(xi)
    # d_eff = c * exp(-xi)
    # => c_eff * d_eff = c^2 => c = sqrt(c_eff * d_eff)
    # => c_eff / d_eff = exp(2 * xi) => xi = 0.5 * (log(c_eff) - log(d_eff))
    c_series = np.sqrt(np.maximum(c_eff * d_eff, 1e-12))
    log_diff = 0.5 * (np.log(np.maximum(c_eff, 1e-6)) - np.log(np.maximum(d_eff, 1e-6)))

    # Apply kappa modulation on the difference signal
    xi_series = xi_bar + kappa * log_diff

    out = run_filter(spec, price, imbalance, xi=xi_series, c=c_series)

    return VolFilterResult(
        ma=out.ma,
        c_arr=c_eff,
        d_arr=d_eff,
        xi_arr=xi_series,
        innovations=out.innovations,
        S=out.S,
        total_loglik=out.total_loglik,
        diagnostics=diag,
    )
