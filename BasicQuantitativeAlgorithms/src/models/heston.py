import numpy as np
from .base_model import PathModel

class Heston(PathModel):
    def __init__(self, kappa: float, theta: float, xi: float, rho: float, v0: float, S0: float, mu: float = 0.0):
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.S0 = float(S0)
        self.mu = float(mu)
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError("rho doit être dans [-1, 1].")

    def simulate(self, T: float, N: int, M: int, random_state=None):
        rng = np.random.default_rng(random_state)
        dt = T / N
        t = np.linspace(0.0, T, N + 1)

        S = np.empty((M, N + 1), dtype=float)
        v = np.empty((M, N + 1), dtype=float)
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        sqrt_dt = np.sqrt(dt)
        rho = self.rho
        xi = self.xi
        kappa = self.kappa
        theta = self.theta

        for k in range(N):
            Z1 = rng.normal(0.0, 1.0, size=M)
            Z2 = rng.normal(0.0, 1.0, size=M)
            dW1 = sqrt_dt * Z1
            dW2 = sqrt_dt * (rho * Z1 + np.sqrt(max(1.0 - rho ** 2, 0.0)) * Z2)

            v_prev = np.maximum(v[:, k], 0.0)
            v_new = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * dW2
            v_new = np.maximum(v_new, 0.0)

            S_prev = S[:, k]
            S_new = S_prev + self.mu * S_prev * dt + np.sqrt(v_prev) * S_prev * dW1

            v[:, k + 1] = v_new
            S[:, k + 1] = S_new

        return t, S