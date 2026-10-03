"""Model-based adaptive moving averages from state-space Kalman filters.

Public API
----------
``filter_price``
    Run the filter over a price series and return ``(level, excitation, residual)``.
``KalmanMA`` / ``AdaptiveKalmanMA``
    Fixed and shock-adaptive 2-state filters.
``FilterResult`` / ``StepResult``
    NamedTuple diagnostics returned by the filters.

The submodules ``core``, ``params``, ``multiscale``, ``estimate``,
``instruments``, ``volatility`` and the ``eval`` package expose the richer
multiscale state-space, fitting, microstructure-noise and evaluation tooling.
"""

from .ma import AdaptiveKalmanMA, FilterResult, KalmanMA, StepResult, filter_price

__version__ = "0.3.0"

__all__ = [
    "AdaptiveKalmanMA",
    "FilterResult",
    "KalmanMA",
    "StepResult",
    "filter_price",
]
