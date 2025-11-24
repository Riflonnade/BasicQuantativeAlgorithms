import numpy as np

def var_cvar_from_prices(prices: np.ndarray, horizon: int = 1, alpha: float = 0.95):
    """
    VaR/CVaR empiriques sur horizon (en pas) à partir d'une série de prix.
    Retourne (VaR, CVaR) sur les rendements.
    """
    prices = np.asarray(prices, dtype=float)
    ret = np.diff(np.log(prices), n=horizon)
    q = float(np.quantile(ret, 1 - alpha))
    tail = ret[ret <= q]
    cvar = float(np.mean(tail)) if tail.size else q
    return q, cvar