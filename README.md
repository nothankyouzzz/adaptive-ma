# adaptive-ma

A model-based adaptive moving average: a small state-space model + Kalman
filter that estimates a latent "fair value" level, instead of a hand-picked
SMA/EMA window.

## Why

An exponential moving average (EMA) is the optimal filter for a *random walk +
noise* model:

$$x_t = x_{t-1} + w_t, \qquad p_t = x_t + \varepsilon_t$$

For this model the steady-state Kalman gain is exactly the EMA smoothing
constant:

$$q = \frac{\sigma_w^2}{\sigma_\varepsilon^2}, \qquad k = \frac{-q + \sqrt{q^2 + 4q}}{2}$$

So picking an EMA window is *equivalent* to picking the signal-to-noise ratio
$q$ by hand. This library writes the model down explicitly instead: the
"window" is replaced by interpretable parameters, the gain adapts over time,
and the level can respond to order flow and to regime shifts.

## How is this different from KAMA / VIDYA / AMA?

Most "adaptive moving averages" (Kaufman's KAMA/AMA, VIDYA, FRAMA, Hull MA,
etc.) adapt a smoothing constant using a hand-picked market statistic:

| indicator | adaptivity source | statistical model? |
|---|---|---|
| KAMA / AMA | efficiency ratio | no — heuristic |
| VIDYA | momentum oscillator (CMO) | no — heuristic |
| FRAMA | fractal dimension | no — heuristic |
| Hull MA | weighted MAs to reduce lag | no — heuristic |
| **this library** | Kalman gain + state-dependent noise | **yes — state-space model** |

Those indicators encode useful market intuition, but the adaptation rule
itself is chosen by trial and error: there is no underlying model and no
argument for why that particular statistic should set the window.

Here the adaptivity falls out of the math instead of being bolted on:

- the gain $K_t$ is the *optimal* filter gain for the assumed dynamics,
  updated every step from the error covariance, not a hand-tuned rule;
- in `AdaptiveKalmanMA`, large shocks feed a *permanent* term into the level
  and widen the level's process noise — a mechanism that maps to a concrete
  story (large informed flow shifts the trend) rather than an arbitrary
  formula.

In short: heuristic adaptive MAs are "clever window pickers"; this is "a
model with an optimal filter".

## Model

Price is decomposed as

$$p_t = d_t + h_t + \varepsilon_t$$

where $x_t = [d_t,\ h_t]^\top$ is the state, $d_t$ is the slow fair-value level
(drift) and $h_t$ is the self-exciting (mean-reverting) impact:

$$d_t = d_{t-1} + \lambda_{\mathrm{perm}}\, \mathrm{softthr}(u_t, c) + \eta_t$$

$$h_t = \rho\, h_{t-1} + \alpha_{\mathrm{eff}}\, u_t + \xi_t$$

Here $u_t$ is the signed order-flow imbalance (optional; zeros if unavailable)
and

$$\mathrm{softthr}(u, c) = \mathrm{sign}(u)\cdot\max(|u|-c,\, 0)$$

The moving average is $d_t + h_t$.

`AdaptiveKalmanMA` adds two mechanisms on top of the fixed model:

- large shocks ($|u_t| > c$) feed a *permanent* term
  $\lambda_{\mathrm{perm}}\, \mathrm{softthr}(u_t, c)$ into the level;
- the level's process noise grows with shock size,

$$\sigma_{d,t} = \sigma_d + \gamma\, |\mathrm{softthr}(u_t, c)|$$

so the MA absorbs regime shifts instead of lagging them.

## Kalman filter

The state-space form is

$$x_t = F x_{t-1} + B u_t + w_t, \qquad w_t \sim N(0, Q_t)$$

$$p_t = H x_t + \varepsilon_t, \qquad \varepsilon_t \sim N(0, R)$$

with

$$F = \begin{bmatrix} 1 & 0 \\\\ 0 & \rho \end{bmatrix}, \qquad H = \begin{bmatrix} 1 & 1 \end{bmatrix}$$

The filter alternates predict and update steps:

$$\hat{x}_{t\mid t-1} = F \hat{x}_{t-1} + B u_t, \qquad P_{t\mid t-1} = F P_{t-1} F^\top + Q_t$$

$$K_t = P_{t\mid t-1} H^\top \left( H P_{t\mid t-1} H^\top + R \right)^{-1}$$

$$\hat{x}_t = \hat{x}_{t\mid t-1} + K_t \left( p_t - H \hat{x}_{t\mid t-1} \right)$$

The Kalman gain $K_t$ adapts each step; in steady state (fixed $Q, R$) it
converges to a constant, recovering the EMA case above.

## Important: not an alpha

The residual $p_t - (d_t + h_t)$ is mostly observation noise. Trading it as a
mean-reversion signal produces a large *spurious* backtest edge (the current
noise term mechanically reverses next bar, i.e. bid-ask bounce). This is a
denoising / decomposition tool, not a standalone tradeable signal.

## Install

```
pip install numpy    # only dependency
```

Then copy `adaptive_ma.py` into your project, or:

```
pip install git+https://github.com/nothankyouzzz/adaptive-ma.git
```

## Usage

```python
import numpy as np
from adaptive_ma import filter_price

price = ...        # 1-D array
imbalance = ...    # optional signed order flow (same length); None -> zeros

level, excitation, residual = filter_price(
    price, imbalance, adaptive=True,
    rho=0.8, alpha_eff=0.5, sigma_eps=0.5,
)
ma = level + excitation
```

See `example.py` for a complete synthetic demo.

## Parameters

| name | meaning |
|---|---|
| `rho` | excitation persistence (0..1); decay rate of the transient impact |
| `alpha_eff` | how much a unit of imbalance moves the excitation |
| `lam_perm` | permanent impact of large shocks on the level |
| `threshold` | shock-size threshold `c` for the permanent term |
| `sigma_level` | level process noise (how fast the level can move) |
| `sigma_exc` | excitation process noise |
| `sigma_eps` | observation noise |
| `gamma` | how much `sigma_level` grows with shock size |

## License

MIT
