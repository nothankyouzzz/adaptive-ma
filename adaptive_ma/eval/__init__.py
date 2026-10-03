"""Honest-evaluation toolkit: metrics, synthetic DGPs, baselines, data loaders.

Submodules are imported explicitly (``adaptive_ma.eval.metrics`` etc.) so that a
numpy-only install never pulls in the optional ``eval`` dependencies (pandas,
requests). Only :mod:`adaptive_ma.eval.data` needs pandas/requests.
"""

__all__ = ["baselines", "data", "dgp", "metrics"]
