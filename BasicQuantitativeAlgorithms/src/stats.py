import numpy as np

def log_return_moments(S: np.ndarray, S0: float):
    ST = S[:, -1]
    lr = np.log(ST / S0)
    return float(np.mean(lr)), float(np.var(lr))

def var_cvar(returns: np.ndarray, alpha: float = 0.95):
    q = float(np.quantile(returns, 1 - alpha))
    tail = returns[returns <= q]
    cvar = float(np.mean(tail)) if tail.size else q
    return q, cvar

def empirical_log_return_stats(S: np.ndarray, S0: float):
    """Retourne la moyenne et variance empirique des log-returns."""
    return log_return_moments(S, S0)

def gbm_theoretical_log_return_moments(mu: float, sigma: float, T: float):
    """Retourne les moments théoriques de log(S_T / S_0) pour le GBM."""
    mean = (mu - 0.5 * sigma**2) * T
    var = sigma**2 * T
    return mean, var
