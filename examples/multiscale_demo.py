"""Offline demo: multiscale 5-state filter and MLE fit on synthetic data.

Runs entirely on the synthetic generators in ``adaptive_ma.eval.dgp`` -- no
network access and no cached market data required.
"""

import numpy as np

from adaptive_ma.estimate import build_spec_from_params, fit_multiscale_mle
from adaptive_ma.eval.dgp import generate_multiscale
from adaptive_ma.multiscale import build_spec_5state, calc_observability, filter_multiscale


def main() -> None:
    series = generate_multiscale(n=1000, seed=7)
    spec = build_spec_5state()
    rank, cond = calc_observability(spec)
    print(f"5-state spec: obs_rank={rank} obs_cond={cond:.3g}")

    ma, diagnostics = filter_multiscale(series.price, series.imbalance, spec=spec)
    rmse = float(np.sqrt(np.mean((ma - series.true_target) ** 2)))
    print(f"filter: total_loglik={diagnostics['total_loglik']:.1f} rmse={rmse:.4f}")

    fit = fit_multiscale_mle(series.price[:600], series.imbalance[:600], n_restarts=2, seed=7)
    refit_spec = build_spec_from_params(fit.params)
    refit_rank, _ = calc_observability(refit_spec)
    print(
        f"fit: train_loglik={fit.train_loglik:.1f} converged={fit.converged} "
        f"multi_start_spread={fit.multi_start_spread:.4g} obs_rank={refit_rank}"
    )
    print(f"fitted rho_c={fit.params.rho_c:.3f} rho_h={fit.params.rho_h:.3f}")


if __name__ == "__main__":
    main()
