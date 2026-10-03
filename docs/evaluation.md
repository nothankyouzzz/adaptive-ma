# Honest Evaluation of Moving Averages

Evaluating a moving average or filter seems straightforward at first glance: compute Mean Squared Error (MSE) or tracking error against price. However, standard evaluation protocols in technical analysis and algorithmic research frequently suffer from systematic flaws that produce misleading results.

The `adaptive_ma.eval` module provides a rigorous evaluation framework designed to address these methodological traps.

---

## 1. The Core Traps of Moving Average Evaluation

### Trap 1: The Roughness Mismatch (Comparing Apples to Oranges)

A moving average with a smaller smoothing window will always track raw price closer (lower MSE) than an MA with a wider window. However, it achieves lower MSE simply by passing high-frequency noise through to the output.

Comparing two moving averages at different roughness levels is mathematically invalid:
- An unfiltered identity filter $\hat{y}_t = p_t$ has $\text{MSE} = 0$, but provides zero noise reduction.
- A flat constant line $\hat{y}_t = \bar{p}$ has high MSE, but zero roughness.

**The Solution: Iso-Smoothness (Matched-Roughness) Comparison.**
To compare filters honestly:
1. Define a rigorous roughness metric: normalized second-difference energy $\mathcal{S}(y)$:
   $$\mathcal{S}(y) = \frac{1}{u_{\text{scale}}^2} \frac{1}{N - 2} \sum_{t=2}^N \left( \Delta^2 y_t \right)^2, \qquad \Delta^2 y_t = y_t - 2 y_{t-1} + y_{t-2}$$
   Lower values indicate a smoother output.
2. Tune baseline parameters (e.g. the smoothing factor $\alpha$ of an EMA) on a dedicated training set so that its roughness exactly matches the candidate filter's roughness.
3. Compare tracking lag, MSE, or predictive score only after roughness has been matched.

### Trap 2: Future Leakage (Non-Causal Filters)

Many smoothing algorithms (such as centered moving averages, Savitzky-Golay filters, or bidirectional spline smoothers) evaluate a point $y_t$ using future observations $p_{t+k}$ ($k > 0$). In historical backtests, non-causal filters appear to have zero lag and near-zero error.

However, in online real-time operation, future data is unavailable. All filters in this toolkit are strictly causal:

$$\hat{x}_t = \mathcal{F}(p_{\le t}, u_{\le t})$$

The test suite enforces this property explicitly: truncating the price series at bar $k$ yields identical output for bars $1 \dots k$ as running on the full series.

### Trap 3: In-Sample Hyperparameter Overfitting

Tuning smoothing parameters (such as EMA windows or KAMA periods) on the full evaluation dataset leads to optimistic tracking metrics. The toolkit provides `adaptive_ma.eval.baselines.tune_ema_on_train` to select baseline parameters strictly on training splits before out-of-sample testing.

---

## 2. Explicit Warning: The Filter Residual is NOT an Alpha

A frequent temptation when working with state-space filters is to treat the filter residual:

$$e_t = p_t - \hat{y}_t$$

as a mean-reversion signal (e.g. going short when $e_t > 0$ and long when $e_t < 0$).

In naive friction-free simulations, this heuristic exhibits deceptively high apparent predictive correlation.

### Why This Edge is an Illusion: The Bid-Ask Bounce

Under Roll's (1984) model, observed price is $p_t = m_t + \varepsilon_t$, where $m_t$ is the latent efficient price and $\varepsilon_t = \pm \frac{s}{2}$ is microstructure observation noise (bid-ask bounce).

The one-step price return is:

$$\Delta p_{t+1} = \Delta m_{t+1} + \varepsilon_{t+1} - \varepsilon_t$$

Notice the mechanical presence of $-\varepsilon_t$:
- If a trade occurred at the ask ($p_t = m_t + s/2$), the residual $e_t$ is positive.
- The next transaction has probability $0.5$ of being at the bid ($p_{t+1} = m_{t+1} - s/2$), creating a return of $\Delta p_{t+1} \approx -s$.

Regressing $\Delta p_{t+1}$ on $-e_t$ yields a statistically significant positive slope purely due to the negative serial correlation of observation noise. The efficient price $m_t$ did not move predictably at all.

In live execution, attempting to capture this residual requires crossing the bid-ask spread and paying trading fees. The mechanical bounce precisely equals the half-spread, ensuring guaranteed negative expectation net of frictions.

**The filter residual is a diagnostic tool for model fit and noise estimation, not a standalone tradeable alpha signal.**

---

## 3. Metrics in `adaptive_ma.eval.metrics`

The evaluation module implements standardized metrics for filter performance:

### Roughness / Smoothness

`calc_smoothness(series, u_scale=1.0)` computes normalized second-difference energy. Normalization by $u_{\text{scale}}^2$ ensures that roughness remains invariant to price scaling.

### Tracking Lag Metrics

1. **Empirical Cross-Correlation Lag (`calc_lag_cc`)**:
   Computes the lag $k^*$ that maximizes the cross-correlation between the true latent target and the estimated filter output:
   $$k^* = \arg\max_k \operatorname{Corr}\left(x_t,\, \hat{y}_{t+k}\right)$$
   A positive value indicates the filter lags behind the true signal.
2. **Impulse Response Centroid Latency (`calc_impulse_response_latency`)**:
   Under a steady-state Kalman gain $K$, the filter's closed-loop state dynamics for an impulse input $y_0 = 1, y_{t \ge 1} = 0$ are:
   $$x_k = (I - K H) F x_{k-1}, \qquad g_k = H x_k$$
   - **Centroid Latency $\tau$**:
     $$\tau = \frac{\sum_{k=0}^\infty k \cdot g_k}{\sum_{k=0}^\infty g_k}$$
     For an exponential moving average with parameter $\alpha$, the analytic latency is exactly $\tau = (1 - \alpha) / \alpha$.
   - **Noise Gain**: $\sum_{k=0}^\infty g_k^2$ (for EMA: $\frac{\alpha}{2 - \alpha}$).
   - **DC Gain**: $\sum_{k=0}^\infty g_k$ (must equal $1.0$ for unbiased tracking).

### Comprehensive Metric Suite (`calc_metrics`)

Returns a `FilterMetrics` dataclass with:
- `mse`: Mean squared error against true target.
- `nmse`: Normalized mean squared error (MSE divided by target noise variance).
- `corr`: Pearson correlation coefficient.
- `smoothness`: Normalized second-difference energy $\mathcal{S}$.
- `lag_centroid`: Centroid latency from impulse response (or cross-correlation).
- `lag_cc`: Empirical cross-correlation lag.
- `predictive_score`: One-step-ahead Gaussian log predictive score $\frac{1}{N}\sum \ln p(y_t \mid y_{<t})$.
- `directional_accuracy`: Fraction of bars where sign of filter change matches sign of target change.

### Pareto Frontier Analysis (`calc_pareto_hypervolume`)

Filters face an inherent trade-off between smoothness $\mathcal{S}$ and tracking lag $\tau$. `calc_pareto_hypervolume` computes the dominated area on the $(\mathcal{S}, \tau)$ plane against a reference threshold, quantifying overall filter efficiency across multiple parameter settings.

---

## 4. Synthetic Data Generating Processes (`adaptive_ma.eval.dgp`)

Synthetic benchmarks allow rigorous testing because the ground truth latent state is known exactly:

| DGP Function | Benchmark | Mathematical Process | Purpose |
|---|---|---|---|
| `generate_rw_noise` | B1 | $x_t = x_{t-1} + w_t, \quad p_t = x_t + \varepsilon_t$ | Baseline Random Walk + Noise; optimal filter is EMA |
| `generate_unobservable_two_rw` | B2a | $p_t = d_{1,t} + d_{2,t} + \varepsilon_t$ (two RWs) | Observability failure negative control |
| `generate_multiscale` | B3 | Macro trend + damped cycle + micro AR(1) | Multiscale structural decomposition test |
| `generate_garch_vol` | B4 | GARCH(1,1) process noise $Q_t$, constant $R$ | Fundamental volatility clustering test |
| `generate_noise_bursts` | B5 | Constant $Q$, episodic bursts in $R_t$ | Microstructure noise burst test |
| `generate_independent_vol_clusters` | B6 | Independent GARCH $Q_t$ + noise bursts $R_t$ | Headline decoupled instrument test |
| `generate_roll_bounce` | B9 | $p_t = m_t + \frac{s}{2} q_t, \quad q_t \in \{-1, +1\}$ | Pure Roll bid-ask bounce null control |

---

## 5. End-to-End Evaluation Workflow

Here is a complete, self-contained example demonstrating an honest iso-smoothness evaluation:

```python
import numpy as np
from adaptive_ma.eval.dgp import generate_multiscale
from adaptive_ma.eval.baselines import calc_ema, tune_ema_on_train
from adaptive_ma.eval.metrics import calc_metrics, calc_smoothness
from adaptive_ma.multiscale import build_spec_5state, filter_multiscale

# 1. Generate synthetic benchmark series
series = generate_multiscale(n=1200, seed=42)
price = series.price
true_target = series.true_target

# Train / Test split (600 bars train, 600 bars test)
train_p, test_p = price[:600], price[600:]
train_true, test_true = true_target[:600], true_target[600:]
train_u, test_u = series.imbalance[:600], series.imbalance[600:]

# 2. Run multiscale filter on test set
spec = build_spec_5state()
ma_test, diag = filter_multiscale(test_p, test_u, spec=spec)
target_smoothness = calc_smoothness(ma_test)

# 3. Honest baseline: tune EMA alpha on TRAIN set to match candidate smoothness
optimal_alpha = tune_ema_on_train(train_p, target_smoothness=target_smoothness)

# 4. Evaluate matched-roughness EMA on TEST set
ema_test = calc_ema(test_p, alpha=optimal_alpha)

# 5. Compute metrics
m_multiscale = calc_metrics(test_true, ma_test)
m_ema = calc_metrics(test_true, ema_test)

print(f"Target Smoothness (Multiscale): {m_multiscale.smoothness:.4e}")
print(f"Matched Smoothness (EMA):       {m_ema.smoothness:.4e}")
print("--- Iso-Smoothness Performance ---")
print(f"Multiscale RMSE: {np.sqrt(m_multiscale.mse):.4f} | Lag CC: {m_multiscale.lag_cc:.1f} bars")
print(f"Matched EMA RMSE: {np.sqrt(m_ema.mse):.4f} | Lag CC: {m_ema.lag_cc:.1f} bars")
```
