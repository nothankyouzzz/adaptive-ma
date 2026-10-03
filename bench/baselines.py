"""Baseline moving averages with Train/Val hyperparameter selection."""

from __future__ import annotations

import numpy as np


def calc_sma(price: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple Moving Average."""
    n = len(price)
    out = np.zeros(n)
    if window <= 0:
        return price.copy()
    window = min(window, n)

    cumsum = np.cumsum(np.insert(price, 0, 0.0))
    for i in range(window):
        out[i] = cumsum[i + 1] / (i + 1)
    out[window - 1 :] = (cumsum[window:] - cumsum[:-window]) / window
    return out


def calc_ema(
    price: np.ndarray, alpha: float | None = None, window: int | None = None
) -> np.ndarray:
    """Exponential Moving Average."""
    n = len(price)
    out = np.zeros(n)
    if n == 0:
        return out
    if alpha is None:
        if window is None or window <= 1:
            alpha = 1.0
        else:
            alpha = 2.0 / (window + 1.0)

    out[0] = price[0]
    for t in range(1, n):
        out[t] = alpha * price[t] + (1.0 - alpha) * out[t - 1]
    return out


def calc_kama(
    price: np.ndarray,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30,
) -> np.ndarray:
    """Kaufman's Adaptive Moving Average (KAMA)."""
    n = len(price)
    out = np.zeros(n)
    if n == 0:
        return out

    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)

    out[0] = price[0]
    for t in range(1, n):
        if t < er_period:
            out[t] = price[t]
            continue

        change = abs(price[t] - price[t - er_period])
        volatility = np.sum(np.abs(np.diff(price[t - er_period : t + 1])))
        if volatility == 0:
            er = 0.0
        else:
            er = change / volatility

        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        out[t] = out[t - 1] + sc * (price[t] - out[t - 1])

    return out


def tune_ema_on_train(
    train_price: np.ndarray,
    target_smoothness: float | None = None,
    criterion: str = "smoothness",
) -> float:
    """Select optimal alpha on training split."""
    alphas = np.logspace(-3, 0, 150)
    from bench.metrics import calc_smoothness

    if target_smoothness is not None:
        # Match target smoothness
        best_alpha = 0.1
        best_diff = float("inf")
        for a in alphas:
            ema = calc_ema(train_price, alpha=a)
            s = calc_smoothness(ema)
            diff = abs(s - target_smoothness)
            if diff < best_diff:
                best_diff = diff
                best_alpha = a
        return float(best_alpha)
    else:
        # One-step ahead prediction error
        best_alpha = 0.1
        best_err = float("inf")
        for a in alphas:
            ema = calc_ema(train_price, alpha=a)
            # One step ahead prediction: ema[t-1] predicts price[t]
            err = np.mean((ema[:-1] - train_price[1:]) ** 2)
            if err < best_err:
                best_err = err
                best_alpha = a
        return float(best_alpha)
