# Multiscale State-Space Moving Average

The multiscale model decomposes an observed price series into distinct physical frequency bands using a continuous-time structural state-space model discretized via matrix exponentials.

Unlike heuristic multi-timeframe moving averages that apply several window lengths in parallel, the state-space formulation enforces dynamic consistency across all scales through a single unified Kalman filter.

---

## 1. Continuous-Time SDE Specification

Let $p(t)$ denote the observable price at continuous time $t \ge 0$. The latent state vector is five-dimensional:

$$x(t) = \begin{bmatrix} \mu(t) \\ \beta(t) \\ c(t) \\ c^*(t) \\ h(t) \end{bmatrix}$$

where:
- $\mu(t) \in \mathbb{R}$: the **secular level** (slow macro trend).
- $\beta(t) \in \mathbb{R}$: the **drift velocity** (slope of $\mu(t)$).
- $c(t), c^*(t) \in \mathbb{R}^2$: the **damped stochastic cycle** (intermediate swing band) with fundamental frequency $\lambda = 2\pi / \tau_{\text{cycle}}$ and damping factor $\rho_c \in (0, 1)$. Here $c^*(t)$ is the orthogonal conjugate state required for harmonic rotation.
- $h(t) \in \mathbb{R}$: the **micro transient impact** (fast order-flow excitation) with persistence $\rho_h \in (0, 1)$, excited by signed order-flow imbalance $u(t)$.

The continuous-time system is governed by the linear stochastic differential equation (SDE):

$$dx(t) = A x(t)\, dt + B u(t)\, dt + dW(t)$$

with continuous drift matrix $A$:

$$A = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 \\
0 & 0 & \ln \rho_c & \lambda & 0 \\
0 & 0 & -\lambda & \ln \rho_c & 0 \\
0 & 0 & 0 & 0 & \ln \rho_h
\end{bmatrix}$$

control input vector $B = [0, 0, 0, 0, \alpha_{\text{eff}}]^\top$, and Wiener process covariance:

$$\mathbb{E}[dW(t)\, dW(t)^\top] = Q_c\, dt, \qquad Q_c = \operatorname{diag}(\sigma_\mu^2, \sigma_\beta^2, \sigma_c^2, \sigma_c^2, \sigma_h^2)$$

The continuous observation equation is:

$$p(t) = H x(t) + \varepsilon(t), \qquad H = \begin{bmatrix} 1 & 0 & 1 & 0 & 1 \end{bmatrix}, \qquad \varepsilon(t) \sim \mathcal{N}(0, \sigma_\varepsilon^2)$$

The moving average estimate corresponds to the filtered latent trend:

$$\hat{y}_{\text{MA}}(t) = \hat{\mu}(t) + \hat{c}(t) + \hat{h}(t) = H \hat{x}(t)$$

---

## 2. Van Loan Matrix-Fraction Discretization

To evaluate the continuous SDE at discrete sample intervals $\Delta t$ (e.g. 1 second, 1 minute, 1 hour) without Euler discretization bias, we employ Van Loan's (1978) matrix fraction method.

We construct the $2n \times 2n$ block matrix ($n = 5$):

$$M = \begin{bmatrix} -A \Delta t & Q_c \Delta t \\ 0 & A^\top \Delta t \end{bmatrix}$$

and compute its matrix exponential:

$$E = \exp(M) = \begin{bmatrix} E_{11} & E_{12} \\ 0 & E_{22} \end{bmatrix}$$

The exact discrete transition matrix $F(\Delta t)$ and process noise covariance $Q(\Delta t)$ are given by:

$$F(\Delta t) = E_{22}^\top = \exp(A \Delta t)$$

$$Q(\Delta t) = F(\Delta t) E_{12} = \int_0^{\Delta t} \exp(A s)\, Q_c\, \exp(A^\top s)\, ds$$

The discrete control matrix is $B(\Delta t) = B \Delta t$. The matrix $Q(\Delta t)$ is explicitly symmetrized: $Q \leftarrow \frac{1}{2}(Q + Q^\top)$.

This discretization guarantees exact mathematical invariance across varying sampling intervals $\Delta t$.

In Python, this is provided by `adaptive_ma.multiscale.build_spec_5state_ct`:

```python
from adaptive_ma.multiscale import build_spec_5state_ct

# Continuous-time specification with physical time constants
ct_spec = build_spec_5state_ct(cycle_period_hours=72.0, rho_c=0.96, rho_h=0.70)

# Discretize exactly to 1-hour or 1-minute steps
spec_1h = ct_spec.discretize(dt=1.0)
spec_1m = ct_spec.discretize(dt=1.0 / 60.0)
```

---

## 3. Principle 1: Scale Equivariance

Financial asset prices span several orders of magnitude (e.g. BTC at $60,000 vs an equity at $25.00). If state variances and priors are fixed in absolute currency units, filter dynamics distort when price levels shift.

The toolkit enforces **Scale Equivariance**:

$$\operatorname{filter}(\alpha \cdot p, u) = \alpha \cdot \operatorname{filter}(p, u) \qquad \forall \alpha > 0$$

### Normalization Pipeline

1. **Scale Anchor**: Compute robust scale estimate from return median absolute deviation:
   $$u_{\text{scale}} = 1.4826 \cdot \operatorname{MAD}(\Delta p)$$
2. **Dimensionless Price**: Normalize raw prices relative to origin $p_0$:
   $$\tilde{p}_t = \frac{p_t - p_0}{u_{\text{scale}}}$$
3. **Filtering**: Run the Kalman filter in normalized dimensionless coordinates. The initial level state is warm-started to $\tilde{p}_0 = 0$.
4. **Denormalization**: Project state trajectories and observation variance back to nominal units:
   $$\hat{x}_t \leftarrow \hat{x}_t \cdot u_{\text{scale}}, \qquad \hat{\mu}_t \leftarrow \hat{\mu}_t \cdot u_{\text{scale}} + p_0$$
   $$S_t \leftarrow S_t \cdot u_{\text{scale}}^2$$

> **Note on State Covariances (`P_traj`, `P_diag_traj`):**
> `adaptive_ma.core.run_filter` denormalizes `state_traj`, `y_pred`, `innovations`, and innovation variance `S`. The state error covariance trajectory arrays `P_traj` and `P_diag_traj` are returned directly in normalized (dimensionless) units. If you are constructing nominal confidence or uncertainty bands for state $i$, scale the standard deviation by $u_{\text{scale}}$:
> $$\sigma_{\text{nominal}, i}(t) = u_{\text{scale}} \cdot \sqrt{P_{t, ii}}$$

The test suite verifies this identity to $< 10^{-8}$ relative difference across scale shifts from $0.1\times$ to $10\times$.

---

## 4. Principle 2: Double-Knob Q/R Decomposition

In standard Kalman filters, process noise $Q$ and observation noise $R$ are often tuned simultaneously. However, they possess an inherent symmetry:

> **Theorem (Gain Invariance):**
> Scaling both $Q$ and $R$ by a common scalar factor $c > 0$ leaves the discrete Kalman gain trajectory $K_t$ strictly invariant for all $t$:
> $$K_t(c Q_0, c R_0) = K_t(Q_0, R_0)$$
> Only the innovation variance $S_t$ scales proportionally: $S_t(c Q_0, c R_0) = c S_t(Q_0, R_0)$.

Consequently, dynamic filter adaptation cannot rely on a single volatility scale. The toolkit factorizes the parameter space into two orthogonal knobs:

$$Q(\xi_t, c_t) = c_t \cdot e^{+\xi_t} Q_0, \qquad R(\xi_t, c_t) = c_t \cdot e^{-\xi_t} R_0$$

- **Calibration Knob** $c_t > 0$: Adjusts overall variance level and likelihood calibration without changing the filter's tracking responsiveness or gain.
- **Responsiveness Knob** $\xi_t \in \mathbb{R}$: Modulates the signal-to-noise ratio $Q / R \propto e^{2\xi_t}$, shifting filter weight between immediate responsiveness ($\xi_t > 0$) and aggressive smoothing ($\xi_t < 0$).

---

## 5. Discrete Lyapunov Stationary Initialization

The 5-state model decomposes into:
- An **unstable block** of dimension $d = 2$ ($[\mu_t, \beta_t]$), representing integrated unit-root dynamics.
- A **stable block** of dimension $n - d = 3$ ($[c_t, c^*_t, h_t]$), with transition matrix $F_s$ having spectral radius $\rho(F_s) < 1$.

Arbitrary covariance initialization (e.g. $P_0 = I$) introduces transient filter bias during early bars. To eliminate initial transient distortion:

1. **Unstable Block**: Initialized diffusely with variance $\kappa_{\text{diffuse}} = 10^6$:
   $$P_0[:2, :2] = \kappa_{\text{diffuse}} I_2$$
2. **Stable Block**: Initialized from the exact ergodic stationary covariance by solving the discrete algebraic Lyapunov equation:
   $$P_s = F_s P_s F_s^\top + Q_s$$
   solved via `scipy.linalg.solve_discrete_lyapunov`.

At $t = 0$, the cycle and transient components start in their true stationary distribution, removing burn-in distortion for the oscillatory states.

---

## 6. Observability Caveat

A linear state-space system $(F, H)$ is observable if and only if its observability matrix:

$$\mathcal{O} = \begin{bmatrix} H \\ H F \\ H F^2 \\ \vdots \\ H F^{n-1} \end{bmatrix}$$

has full rank $n$.

### Why Mathematical Structure Matters

Consider a naive model attempting to split price into two independent random walks:

$$p_t = d_{1,t} + d_{2,t} + \varepsilon_t, \qquad d_{i,t} = d_{i,t-1} + w_{i,t}$$

Here $F = I_2$ and $H = [1, 1]$. The observability matrix is:

$$\mathcal{O} = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$$

which has rank $1 < 2$ and condition number $\approx \infty$. It is impossible for any filter to determine which random walk moved; the states are mathematically unobservable.

In the 5-state Harvey model:
- $\mu_t$ has unit root order 2 ($F_{12} = 1$).
- $c_t, c^*_t$ oscillate at harmonic frequency $\lambda$ with damping $\rho_c$.
- $h_t$ is an AR(1) with decay $\rho_h \ne \rho_c$.

Because each component occupies a distinct frequency signature, the observability matrix $\mathcal{O}$ has full rank $5$ and condition number $< 10^7$.

The function `calc_observability(spec)` computes both rank and condition number:

```python
from adaptive_ma.multiscale import build_spec_5state, calc_observability

spec = build_spec_5state()
rank, cond = calc_observability(spec)
print(f"Observability rank: {rank}/5, condition number: {cond:.2e}")
```

---

## 7. Model Fitting with `adaptive_ma.estimate`

Parameters of the 5-state model can be calibrated to historical data using penalized Maximum Likelihood Estimation (MLE) via `adaptive_ma.estimate.fit_multiscale_mle`.

### Identifiability Constraints

Unconstrained optimization can lead to parameter pile-up (e.g. $\rho_c \to 1$ or $\lambda \to 0$). The `MultiscaleParams` class enforces bounds via smooth transforms:
- Decay ordering: $\rho_h = \sigma_L(\text{logit\_rho\_h})$ and $\rho_c = \rho_h + (1 - \rho_h) \sigma_L(\text{logit\_rho\_c\_delta})$, guaranteeing $0 < \rho_h < \rho_c < 1$.
- Frequency bounds: $\lambda \in [0.10, \pi - 0.10]$ (corresponding to cycles between $\sim 2.1$ and $\sim 62$ bars).
- Variance ratios: parameterized as log ratios relative to $\sigma_\varepsilon$.

### Multi-Start Optimization

`fit_multiscale_mle` minimizes negative diffuse log-likelihood with a weakly informative Gaussian prior ($\sigma = 3.0$ on transformed parameters) using SciPy's L-BFGS-B:

```python
import numpy as np
from adaptive_ma.estimate import fit_multiscale_mle, build_spec_from_params
from adaptive_ma.multiscale import filter_multiscale
from adaptive_ma.eval.dgp import generate_multiscale

# Generate synthetic benchmark data
series = generate_multiscale(n=1000, seed=42)

# Fit parameters on training partition (e.g. first 700 bars)
train_p = series.price[:700]
train_u = series.imbalance[:700]
val_p = series.price[700:]
val_u = series.imbalance[700:]

fit_result = fit_multiscale_mle(
    train_price=train_p,
    train_imbalance=train_u,
    val_price=val_p,
    val_imbalance=val_u,
    n_restarts=3,
    seed=42,
)

print(f"Converged: {fit_result.converged}")
print(f"Train loglik: {fit_result.train_loglik:.2f}, Val loglik: {fit_result.val_loglik:.2f}")
print(f"Fitted rho_c: {fit_result.params.rho_c:.3f}, rho_h: {fit_result.params.rho_h:.3f}")

# Run filter with the fitted specification
fitted_spec = build_spec_from_params(fit_result.params)
ma, diag = filter_multiscale(series.price, series.imbalance, spec=fitted_spec)
```
