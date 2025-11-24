import numpy as np
from src.models import GeometricBrownianMotion
from src.stats import empirical_log_return_stats, gbm_theoretical_log_return_moments

def test_empirical_vs_theoretical_close():
    mu, sigma, S0, T, N, M = 0.05, 0.2, 100.0, 1.0, 252, 15000
    gbm = GeometricBrownianMotion(mu=mu, sigma=sigma, S0=S0)
    t, S = gbm.simulate(T=T, N=N, M=M)
    emp_mean, emp_var = empirical_log_return_stats(S, S0)
    th_mean, th_var = gbm_theoretical_log_return_moments(mu, sigma, T)
    assert abs(emp_mean - th_mean) < 0.03
    assert abs(emp_var - th_var) < 0.03
