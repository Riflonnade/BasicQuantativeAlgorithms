from typing import Tuple
import numpy as np
from .models import GBM

def simulate_gbm(mu: float, sigma: float, S0: float, T: float, N: int, M: int,
                 method: str = "exact", random_state=None) -> Tuple[np.ndarray, np.ndarray]:
    """Simule un GBM avec méthode 'exact' (solution fermée) ou 'euler'."""
    gbm = GBM(mu=mu, sigma=sigma, S0=S0)
    if method == "exact":
        return gbm.simulate(T=T, N=N, M=M)
    elif method == "euler":
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        rng = np.random.default_rng(random_state)
        S = np.empty((M, N + 1), dtype=float)
        S[:, 0] = S0
        for k in range(N):
            dW = rng.normal(0.0, np.sqrt(dt), size=M)
            S[:, k + 1] = S[:, k] + mu * S[:, k] * dt + sigma * S[:, k] * dW
        return t, S
    else:
        raise ValueError("method doit être 'exact' ou 'euler'.")
