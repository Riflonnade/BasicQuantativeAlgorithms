import numpy as np
from .base_model import PathModel

class VarianceGamma(PathModel):
    def __init__(self, theta: float, sigma: float, nu: float, S0: float):
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.nu = float(nu)
        self.S0 = float(S0)
    def simulate(self, T: float, N: int, M: int, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)

        shape = dt / self.nu
        scale = self.nu

        X = np.zeros((M, N + 1), dtype=float)
        for k in range(N):
            dG = rng.gamma(shape=shape, scale=scale, size=M)
            dW = rng.normal(0.0, 1.0, size=M)
            dX = self.theta * dG + self.sigma * np.sqrt(dG) * dW
            X[:, k + 1] = X[:, k] + dX

        S = self.S0 * np.exp(X)
        return t, S