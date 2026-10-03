# adaptive-ma

A model-based adaptive moving average toolkit: state-space Kalman filters that estimate latent fair-value levels and structural trends, replacing heuristic window tuning with statistical models.

Includes 2-state adaptive filters, a 5-state multiscale structural model, empirical microstructure-noise instruments, and an honest-evaluation harness.

---

## Why Model-Based Moving Averages?

An exponential moving average (EMA) is the mathematically optimal filter for a *random walk + white noise* model:

$$x_t = x_{t-1} + w_t, \qquad p_t = x_t + \varepsilon_t$$

Under constant noise variances $w_t \sim \mathcal{N}(0, \sigma_w^2)$ and $\varepsilon_t \sim \mathcal{N}(0, \sigma_\varepsilon^2)$, the steady-state Kalman gain $K$ is exactly the EMA smoothing constant:

$$q = \frac{\sigma_w^2}{\sigma_\varepsilon^2}, \qquad K = \frac{-q + \sqrt{q^2 + 4q}}{2}$$

Selecting an EMA window is therefore *isomorphic* to picking an assumed signal-to-noise ratio $q$ by hand.

`adaptive-ma` makes the underlying time-series model explicit:

| Indicator | Adaptivity Source | Statistical Model? | Causal? |
|---|---|---|---|
| **KAMA / AMA** | Efficiency ratio heuristic | No — heuristic formula | Yes |
| **VIDYA** | Momentum oscillator (CMO) | No — heuristic formula | Yes |
| **FRAMA** | Fractal dimension | No — heuristic formula | Yes |
| **Centered MA** | Centered window convolution | No — smoothing heuristic | **No — leaks future data** |
| **`adaptive-ma` (2-state)** | Dynamic Kalman gain + shock-dependent noise | **Yes — state-space model** | **Yes — strictly causal** |
| **`adaptive-ma` (5-state)** | Harvey structural state-space decomposition | **Yes — continuous-time SDE** | **Yes — strictly causal** |

In short: heuristic adaptive MAs are clever window pickers; `adaptive-ma` is a principled statistical filter.

---

## Two Moving-Average Models

### 1. 2-State Adaptive Kalman MA (`adaptive_ma.ma`)

Decomposes price into a slow fair-value level $d_t$ and a mean-reverting transient impact $h_t$:

$$p_t = d_t + h_t + \varepsilon_t$$

$$d_t = d_{t-1} + \lambda_{\mathrm{perm}} \operatorname{softthr}(u_t, c) + \eta_t$$

$$h_t = \rho\, h_{t-1} + \alpha_{\mathrm{eff}}\, u_t + \xi_t$$

where $u_t$ is observable signed order-flow imbalance (optional; zeros if unavailable), and $\operatorname{softthr}(u, c) = \operatorname{sign}(u) \max(|u| - c, 0)$.

`AdaptiveKalmanMA` dynamically widens the level's process noise when large shocks arrive ($\sigma_{d,t} = \sigma_d + \gamma |\operatorname{softthr}(u_t, c)|$), enabling the moving average to track structural shifts promptly without lagging.

### 2. 5-State Multiscale Structural Model (`adaptive_ma.multiscale`)

Formulated as a continuous-time stochastic differential equation (SDE) and discretized via Van Loan matrix exponentials. The state vector:

$$x(t) = \begin{bmatrix} \mu(t) & \beta(t) & c(t) & c^*(t) & h(t) \end{bmatrix}^\top$$

simultaneously isolates:
- **$\mu(t)$ (Secular Level)**: Slow non-stationary trend.
- **$\beta(t)$ (Drift Velocity)**: Local slope / slope velocity of $\mu(t)$.
- **$c(t), c^*(t)$ (Damped Swing Cycle)**: Stochastic harmonic cycle with fundamental period $\tau_{\text{cycle}}$ and damping $\rho_c$.
- **$h(t)$ (Micro Transient Impact)**: Fast mean-reverting AR(1) driven by order flow $u_t$.

The moving average is the combined latent trend: $\hat{y}_{\text{MA}} = \hat{\mu} + \hat{c} + \hat{h}$.

See [docs/multiscale.md](docs/multiscale.md) for the continuous-time SDE specification, Van Loan discretization proofs, scale equivariance, and discrete Lyapunov stationary initialization.

---

## Decoupled Volatility & Noise Instruments

Standard indicators scale smoothing speed using a single volatility index (e.g. ATR or realized variance). However, the **Gain Invariance Theorem** proves that scaling process noise $Q$ and observation noise $R$ proportionally leaves the Kalman gain $K_t$ unchanged.

`adaptive-ma` extracts two empirical microstructure instruments to separate these channels:
- **De-drifted Roll (1984) autocovariance**: identifies observation noise variance $R_t$ (bid-ask bounce) while removing trending return bias.
- **Two-Scale Realized Variance (TSRV)**: identifies noise-corrected integrated variance $Q_t$.

These instruments independently drive orthogonal $Q$ and $R$ knobs, allowing the filter to speed up during fundamental volatility expansions and slow down during microstructure noise bursts.

See [docs/instruments.md](docs/instruments.md) for theory and implementation.

---

## Installation

### Core Package

The core package requires only `numpy>=1.20` and `scipy>=1.10`:

```bash
pip install git+https://github.com/nothankyouzzz/adaptive-ma.git
```

### With Evaluation Harness (`eval` extra)

Includes `pandas`, `requests`, and `matplotlib` for synthetic benchmarking and cached data loading:

```bash
pip install "adaptive-ma[eval] @ git+https://github.com/nothankyouzzz/adaptive-ma.git"
```

---

## Quickstarts

### 1. 2-State Adaptive MA

The classic 3-tuple public API (`level, excitation, residual`):

```python
import numpy as np
from adaptive_ma import filter_price

# Generate synthetic price series
rng = np.random.default_rng(42)
price = 100.0 + np.cumsum(rng.normal(0, 0.2, 500))
imbalance = rng.normal(0, 1.0, 500)  # optional signed order flow

level, excitation, residual = filter_price(
    price,
    imbalance,
    adaptive=True,
    rho=0.8,
    alpha_eff=0.5,
    sigma_level=0.05,
    sigma_exc=0.1,
    sigma_eps=0.5,
)

# The adaptive moving average is the sum of fair-value level and transient excitation
ma = level + excitation
```

### 2. 5-State Multiscale State-Space Filter

Run the 5-state Harvey model with automatic scale equivariance:

```python
import numpy as np
from adaptive_ma.multiscale import build_spec_5state, filter_multiscale
from adaptive_ma.eval.dgp import generate_multiscale

# Generate synthetic multiscale benchmark (macro trend + swing cycle + micro impact)
series = generate_multiscale(n=1000, seed=42)

# Build 5-state specification and run filter
spec = build_spec_5state(cycle_period=45.0, rho_c=0.96, rho_h=0.60)
ma, diag = filter_multiscale(series.price, series.imbalance, spec=spec)

print(f"Log-likelihood: {diag['total_loglik']:.2f}")
print(f"Observability rank: {diag['obs_rank']}/5")
print(f"State trajectory shape: {diag['state_traj'].shape}")
```

### 3. Decoupled Volatility Instrument Filtering

Adapt smoothing speed using empirical microstructure noise and integrated variance:

```python
from adaptive_ma.volatility import filter_volatility_instrument
from adaptive_ma.eval.dgp import generate_independent_vol_clusters

# Series with independent fundamental volatility shifts and noise bursts
series = generate_independent_vol_clusters(n=1500, seed=42)

# Filter with decoupled Q and R instruments
res = filter_volatility_instrument(series.price, mode="instrument")

print(f"MA shape: {res.ma.shape}")
print(f"Mean responsiveness knob xi: {res.xi_arr.mean():.3f}")
```

---

## Public API Overview

| Module | Key Functions / Classes | Description |
|---|---|---|
| `adaptive_ma` | `filter_price`, `KalmanMA`, `AdaptiveKalmanMA`, `FilterResult`, `StepResult` | Backward-compatible 2-state adaptive Kalman moving average |
| `adaptive_ma.multiscale` | `build_spec_5state`, `build_spec_5state_ct`, `build_spec_2state`, `build_spec_3state`, `calc_observability`, `filter_multiscale` | 5-state continuous and discrete state-space models |
| `adaptive_ma.core` | `StateSpaceSpec`, `ContinuousTimeSpec`, `van_loan_discretization`, `run_filter`, `SSMOutput` | Core Kalman filter engine with scale equivariance (P1) and knob separation (P2) |
| `adaptive_ma.params` | `MultiscaleParams`, `calc_u_scale` | Constrained parameter transformations and robust MAD scale estimation |
| `adaptive_ma.estimate` | `fit_multiscale_mle`, `build_spec_from_params`, `FitResult` | Penalized multi-start MLE parameter estimation |
| `adaptive_ma.instruments` | `de_drifted_roll_estimator`, `tsrv_estimator`, `compute_instrument_signals`, `RollResult` | Microstructure noise estimators (Roll autocovariance and TSRV) |
| `adaptive_ma.volatility` | `filter_volatility_instrument`, `VolFilterResult` | 4-arm decoupled volatility adaptive filter (`instrument`, `fixed`, `naive_rv`, `swapped`) |
| `adaptive_ma.eval.metrics` | `calc_metrics`, `calc_smoothness`, `calc_lag_cc`, `calc_impulse_response_latency`, `calc_pareto_hypervolume`, `FilterMetrics` | Iso-smoothness, tracking lag, and impulse-response metrics |
| `adaptive_ma.eval.dgp` | `generate_rw_noise`, `generate_multiscale`, `generate_garch_vol`, `generate_noise_bursts`, `generate_independent_vol_clusters`, `generate_roll_bounce` | Synthetic benchmark data generating processes (B1-B9 subset) |
| `adaptive_ma.eval.baselines` | `calc_sma`, `calc_ema`, `calc_kama`, `tune_ema_on_train` | Benchmark moving averages and train-partition hyperparameter tuning |
| `adaptive_ma.eval.data` | `load_btc_dataset`, `ensure_btc_data_cached`, `resolve_cache_dir` | Cached Binance 1m data loader with SHA256 integrity checks |

---

## In-Depth Documentation

- [docs/multiscale.md](docs/multiscale.md): Continuous-time SDE formulation, Van Loan discretization, scale equivariance, discrete Lyapunov initialization, observability proofs, and MLE fitting.
- [docs/instruments.md](docs/instruments.md): De-drifted Roll estimator, Two-Scale Realized Variance (TSRV), and the Gain Invariance Theorem.
- [docs/evaluation.md](docs/evaluation.md): Honest moving average evaluation, iso-smoothness comparisons, causal lag metrics, synthetic DGPs, and the bid-ask bounce trap.

---

## Important Methodological Note: The Residual is Not an Alpha

The filter residual:

$$e_t = p_t - \hat{y}_t$$

predominantly reflects microstructure observation noise (e.g. bid-ask bounce). Naively trading this residual as a mean-reversion signal creates a strong **spurious** backtest edge because the observation noise mechanically reverses on the subsequent transaction.

In live execution, crossing the bid-ask spread and paying exchange fees eliminates this apparent edge entirely. This library is designed for denoising, state tracking, and trend decomposition—not as a standalone trading signal generator. See [docs/evaluation.md](docs/evaluation.md) for detailed analysis and synthetic null controls.

---

## Development & Testing

The repository uses `uv` for environment management:

```bash
# Install all dependencies and extras
uv sync --all-extras

# Run the complete test suite
uv run pytest -v

# Code quality checks
uv run ruff check .
uv run ruff format --check .

# Run runnable examples
uv run python examples/adaptive_ma_demo.py
uv run python examples/multiscale_demo.py
```

---

## License

MIT
