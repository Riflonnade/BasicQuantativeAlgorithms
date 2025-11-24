import numpy as np
from .base_model import PathModel

class GBM(PathModel):
    """Mouvement brownien géométrique (Black–Scholes)
    dS_t = μ S_t dt + σ S_t dW_t
    """

    def __init__(self, mu: float, sigma: float, S0: float):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.S0 = float(S0)

    def simulate(self, T: float, N: int, M: int, random_state=None):
        """Simule M trajectoires de longueur N sur [0,T]."""
        rng = np.random.default_rng(random_state)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)
        dW = rng.normal(0.0, np.sqrt(dt), size=(M, N))
        W = np.cumsum(dW, axis=1)
        W = np.concatenate([np.zeros((M, 1)), W], axis=1)
        drift = (self.mu - 0.5 * self.sigma ** 2) * t
        S = self.S0 * np.exp(drift + self.sigma * W)
        return t, S
