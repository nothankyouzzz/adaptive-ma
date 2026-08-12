"""Quick demo: synthetic price -> adaptive moving average."""
import numpy as np

from adaptive_ma import filter_price


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    rho = 0.8
    alpha_eff = 0.5

    # ground truth consistent with the model:
    #   level = random walk, excitation = AR(1) driven by order flow
    level = np.cumsum(rng.normal(0, 0.05, n))
    imbalance = rng.normal(0, 1.0, n)                 # signed order flow
    excitation = np.zeros(n)
    for t in range(1, n):
        excitation[t] = rho * excitation[t - 1] + alpha_eff * imbalance[t]
    price = level + excitation + rng.normal(0, 0.5, n)

    lvl, exc, residual = filter_price(
        price, imbalance, adaptive=True,
        rho=rho, alpha_eff=alpha_eff,
        sigma_level=0.05, sigma_exc=0.01, sigma_eps=0.5,
    )
    ma = lvl + exc

    true = level + excitation
    print(f"corr(ma, true level+excitation) = {np.corrcoef(ma, true)[0, 1]:.3f}")
    print(f"residual std = {residual.std():.3f}  (noise std = 0.5)")


if __name__ == "__main__":
    main()
