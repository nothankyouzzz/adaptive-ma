"""Public API back-compat, package import paths, and end-to-end invariants."""

from __future__ import annotations

import numpy as np

import adaptive_ma
from adaptive_ma import filter_price
from adaptive_ma.core import run_filter
from adaptive_ma.estimate import build_spec_from_params, fit_multiscale_mle
from adaptive_ma.eval.dgp import generate_multiscale
from adaptive_ma.multiscale import build_spec_5state, filter_multiscale


def test_backcompat_filter_price():
    """``from adaptive_ma import filter_price`` still returns the 3-tuple API."""
    rng = np.random.default_rng(0)
    price = np.cumsum(rng.normal(0, 0.05, 200))
    level, excitation, residual = filter_price(price, adaptive=True, rho=0.8, alpha_eff=0.5)

    assert level.shape == price.shape
    assert excitation.shape == price.shape
    assert residual.shape == price.shape
    assert np.all(np.isfinite(level + excitation))


def test_public_names_exported():
    for name in ["filter_price", "KalmanMA", "AdaptiveKalmanMA", "FilterResult", "StepResult"]:
        assert hasattr(adaptive_ma, name), f"missing public export {name}"


def test_multiscale_filter_fit_roundtrip_finite():
    series = generate_multiscale(n=150, seed=3)

    ma, diagnostics = filter_multiscale(series.price, series.imbalance, spec=build_spec_5state())
    assert ma.shape == series.price.shape
    assert np.all(np.isfinite(ma))
    assert np.isfinite(diagnostics["total_loglik"])

    fit = fit_multiscale_mle(series.price, series.imbalance, n_restarts=1, seed=3)
    assert fit.converged
    out = run_filter(build_spec_from_params(fit.params), series.price, series.imbalance)
    assert np.all(np.isfinite(out.ma))


def test_multiscale_output_is_causal():
    """A bar's output must not depend on future bars."""
    series = generate_multiscale(n=200, seed=5)
    spec = build_spec_5state()
    full = run_filter(spec, series.price, series.imbalance)

    k = 120
    prefix = run_filter(spec, series.price[:k], series.imbalance[:k], u_scale=full.u_scale)
    np.testing.assert_allclose(prefix.ma, full.ma[:k], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(prefix.state_traj, full.state_traj[:k], rtol=0.0, atol=1e-10)


def test_scale_equivariance_through_package_api():
    series = generate_multiscale(n=200, seed=11)
    price = series.price + 50000.0
    spec = build_spec_5state()
    base = run_filter(spec, price, series.imbalance)

    for alpha in (0.1, 3.0, 10.0):
        scaled = run_filter(spec, alpha * price, series.imbalance)
        rel_diff = np.max(np.abs(scaled.ma - alpha * base.ma)) / (alpha * np.mean(price))
        assert rel_diff < 1e-8, f"scale equivariance failed for alpha={alpha}: {rel_diff}"


def test_eval_metrics_imports_without_optional_deps():
    from adaptive_ma.eval import metrics

    assert hasattr(metrics, "calc_metrics")
    assert hasattr(metrics, "calc_impulse_response_latency")
