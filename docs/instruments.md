# Microstructure Noise Instruments & Decoupled Volatility Filtering

Moving averages often attempt to adapt their smoothing speed based on market volatility. However, classical technical indicators (such as KAMA or volatility-adjusted EMAs) make a fundamental modeling error: they treat all price variance as identical.

In financial markets, observed price changes stem from two physically distinct mechanisms:
1. **Fundamental Process Innovations ($Q$)**: Genuine shifts in the asset's latent efficient value. When fundamental variance increases, the filter should speed up to track the trend without lag.
2. **Microstructure Observation Noise ($R$)**: Frictions including order routing delays, bid-ask bounce, liquidity replenishment gaps, and exchange discreteness. When microstructure noise increases, the filter should slow down to avoid chasing false bounces.

If a moving average scales both $Q$ and $R$ uniformly by a single volatility index (e.g. realized volatility or ATR), the **Gain Invariance Theorem** dictates that the Kalman gain $K_t$ remains unchanged—the filter's tracking responsiveness fails to adapt.

To achieve genuine volatility adaptivity, the toolkit extracts two separate empirical instruments:
- **De-drifted Roll (1984) autocovariance**: provides a clean estimator for observation noise variance $R_t$.
- **Two-Scale Realized Variance (TSRV)**: provides a noise-corrected estimator for integrated variance $Q_t$.

---

## 1. De-drifted Roll Noise Estimator ($R_t$)

### The Classical Roll Model

In Roll's (1984) microstructure model, the observed transaction price $p_t$ fluctuates around the latent efficient price $m_t$ due to the bid-ask bounce:

$$p_t = m_t + \frac{s}{2} q_t$$

where $s > 0$ is the effective bid-ask spread and $q_t \in \{-1, +1\}$ is an independent trade direction indicator (buy vs. sell). Assuming $m_t$ is a random walk with innovations $w_t \sim \mathcal{N}(0, \sigma_w^2)$:

$$\Delta p_t = w_t + \frac{s}{2}(q_t - q_{t-1})$$

The lag-1 autocovariance of price increments is:

$$\operatorname{Cov}(\Delta p_t, \Delta p_{t-1}) = -\frac{s^2}{4} = -\sigma_\varepsilon^2$$

Thus, the microstructure noise variance is directly identified by the negative lag-1 autocovariance:

$$R_t = -\operatorname{Cov}(\Delta p_t, \Delta p_{t-1})$$

### The Drift Bias Pitfall

In real markets, asset prices exhibit local trending drift $\mu_t = \mathbb{E}[\Delta p_t] \ne 0$. Without correction, the uncentered second moment satisfies:

$$\mathbb{E}[(\Delta p_t)(\Delta p_{t-1})] = \mu_t^2 - \sigma_\varepsilon^2$$

If the local drift magnitude $|\mu_t|$ exceeds $\sigma_\varepsilon$, the raw autocovariance becomes positive. In classical implementations, this causes the Roll estimator to fail and truncate to zero or its noise floor, blinding the filter to actual noise conditions during strong trends.

### De-drifted Implementation

`adaptive_ma.instruments.de_drifted_roll_estimator` removes local drift causally before computing running autocovariance:

1. **Local Drift Tracking**: Update an exponentially weighted moving average (EWMA) of price returns:
   $$\bar{r}_t = \lambda_{\text{decay}} \bar{r}_{t-1} + (1 - \lambda_{\text{decay}}) \Delta p_t$$
2. **De-drifted Returns**: Subtract the estimated local drift:
   $$\tilde{r}_t = \Delta p_t - \bar{r}_t$$
3. **Causal Autocovariance**:
   $$\operatorname{Cov}_t = \lambda_{\text{decay}} \operatorname{Cov}_{t-1} + (1 - \lambda_{\text{decay}}) \tilde{r}_t \tilde{r}_{t-1}$$
4. **Noise Variance Extraction**:
   $$R_t = \max(-\operatorname{Cov}_t,\, R_{\text{floor}})$$

The returned `RollResult` provides the estimated noise variance series, the raw autocovariance, and the `truncation_rate` (the proportion of bars where $\operatorname{Cov}_t \ge -R_{\text{floor}}$).

```python
from adaptive_ma.instruments import de_drifted_roll_estimator

roll_res = de_drifted_roll_estimator(prices, ewma_decay=0.98, r_floor=1e-6)
print(f"Mean estimated R: {roll_res.noise_var.mean():.4f}")
print(f"Truncation rate: {roll_res.truncation_rate:.1%}")
```

---

## 2. Two-Scale Realized Variance (TSRV) for Integrated Variance ($Q_t$)

### Why Standard Realized Variance Fails

High-frequency realized volatility (sum of squared returns $\sum \Delta p_i^2$) is biased by microstructure noise. As sampling frequency increases, standard realized variance diverges:

$$\operatorname{RV} \xrightarrow{N \to \infty} \operatorname{IV} + 2 N \sigma_\varepsilon^2$$

where $\operatorname{IV} = \int_0^T \sigma^2(t)\, dt$ is the true continuous integrated variance and $N$ is the number of return intervals.

### The Two-Scale Estimator

Zhang, Mykland, and Aït-Sahalia (2005) introduced the Two-Scale Realized Variance (TSRV) to consistently separate integrated variance from noise:

1. Subsample returns across $K$ interleaved subgrids of frequency $K$.
2. Compute the average realized variance across the slow subgrids:
   $$\operatorname{RV}_{\text{avg}} = \frac{1}{K} \sum_{i=1}^K \operatorname{RV}^{(i)}$$
3. Compute the full-grid realized variance $\operatorname{RV}_{\text{all}}$.
4. The debiased integrated variance estimator is:
   $$\widehat{\operatorname{IV}} = \operatorname{RV}_{\text{avg}} - \frac{\bar{n}}{m} \operatorname{RV}_{\text{all}}$$
   where $m$ is the window size, $\bar{n} = (m - K + 1) / K$, and $K \sim m^{2/3}$.

In `adaptive_ma.instruments.tsrv_estimator`, this calculation is performed in a rolling, causal window:

```python
from adaptive_ma.instruments import tsrv_estimator

# Computes rolling integrated variance (Q channel) and noise variance (R channel)
iv_arr, omega2_arr = tsrv_estimator(prices, window=150, k_subgrids=15)
```

---

## 3. Decoupled Q/R Volatility Filter

The functions in `adaptive_ma.instruments` and `adaptive_ma.volatility` combine the two instruments to drive the state-space Kalman filter.

### Signal Normalization and Shrinkage

The raw estimates $\widehat{\operatorname{IV}}_t$ and $\widehat{R}_t$ possess different units and sampling variances. `compute_instrument_signals` normalizes both via causal EWMA baselines:

$$\tilde{c}_t = \frac{\widehat{\operatorname{IV}}_t}{\operatorname{EWMA}(\widehat{\operatorname{IV}})_t}, \qquad \tilde{d}_t = \frac{\widehat{R}_t}{\operatorname{EWMA}(\widehat{R})_t}$$

Shrinkage towards the unit mean is applied in log space:

$$\ln \hat{c}_t = (1 - s) \ln \tilde{c}_t, \qquad \ln \hat{d}_t = (1 - s) \ln \tilde{d}_t$$

and values are clamped to robust operating bounds $[c_{\text{lo}}, c_{\text{hi}}]$ (default $[0.1, 10.0]$).

### Mapping to Orthogonal P2 Knobs

In the core filter engine:

$$Q(\xi_t, c_t) = c_t \cdot e^{+\xi_t} Q_0, \qquad R(\xi_t, c_t) = c_t \cdot e^{-\xi_t} R_0$$

To match effective multipliers $\hat{c}_t$ on $Q$ and $\hat{d}_t$ on $R$:

$$\hat{c}_t = c_t \cdot e^{\xi_t}, \qquad \hat{d}_t = c_t \cdot e^{-\xi_t}$$

Solving for the orthogonal knobs yields:

$$c_t = \sqrt{\hat{c}_t \hat{d}_t}$$

$$\xi_t = \bar{\xi} + \frac{\kappa}{2} \left( \ln \hat{c}_t - \ln \hat{d}_t \right)$$

where $\kappa$ is a sensitivity multiplier (default $1.0$).

### Filter Dynamics Under Market Regimes

- **Fundamental Volatility Surge ($\hat{c}_t \gg \hat{d}_t$)**:
  $\ln \hat{c}_t - \ln \hat{d}_t > 0 \implies \xi_t > 0$.
  The signal-to-noise ratio $Q/R$ expands, the Kalman gain increases, and the moving average rapidly tracks the breakout.
- **Microstructure Noise Burst ($\hat{d}_t \gg \hat{c}_t$)**:
  $\ln \hat{c}_t - \ln \hat{d}_t < 0 \implies \xi_t < 0$.
  The signal-to-noise ratio shrinks, the Kalman gain drops, and the moving average maintains smooth filtering without chasing noise.
- **Proportional Volatility Surge ($\hat{c}_t \approx \hat{d}_t$)**:
  $\xi_t \approx \bar{\xi}$.
  The filter responsiveness remains steady, while $c_t$ widens the innovation covariance $S_t$.

---

## 4. Experimental Arms & Attribution Control

`adaptive_ma.volatility.filter_volatility_instrument` provides four experimental modes to scientifically test attribution:

| Mode | Q Multiplier | R Multiplier | Purpose |
|---|---|---|---|
| `'fixed'` | $1.0$ | $1.0$ | Baseline unadapted filter |
| `'naive_rv'` | $\hat{c}_t$ | $\hat{c}_t$ | **Gain Invariance Test**: proves common scaling does not change Kalman gain |
| `'instrument'` | $\hat{c}_t$ | $\hat{d}_t$ | **True decoupled model**: independent $Q$ and $R$ channel adaptation |
| `'swapped'` | $\hat{d}_t$ | $\hat{c}_t$ | **Attribution Negative Control**: inverts signals to verify causal attribution |

### Python Example

```python
import numpy as np
from adaptive_ma.volatility import filter_volatility_instrument
from adaptive_ma.eval.dgp import generate_independent_vol_clusters

# Generate synthetic series with independent fundamental vol (Q) and noise bursts (R)
series = generate_independent_vol_clusters(n=1500, seed=42)

# Run decoupled instrument filter
res_inst = filter_volatility_instrument(series.price, mode="instrument")

# Run fixed baseline
res_fixed = filter_volatility_instrument(series.price, mode="fixed")

# Compare tracking error to true latent level
rmse_inst = np.sqrt(np.mean((res_inst.ma - series.true_target) ** 2))
rmse_fixed = np.sqrt(np.mean((res_fixed.ma - series.true_target) ** 2))

print(f"Decoupled instrument RMSE: {rmse_inst:.4f}")
print(f"Fixed baseline RMSE:       {rmse_fixed:.4f}")
print(f"Average responsiveness knob xi: {res_inst.xi_arr.mean():.3f}")
```
