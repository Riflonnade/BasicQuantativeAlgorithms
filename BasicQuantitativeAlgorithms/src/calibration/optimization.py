import numpy as np
from scipy.optimize import minimize
from pricing.black_scholes import bs_call


# -------------------------------------------------------------------------
# 1. Calibration de volatilité BS par moindres carrés
# -------------------------------------------------------------------------
def fit_bs_vol_to_call_price(S0: float, K: float, T: float, r: float,
                             market_price: float, sigma0: float = 0.2):
    """
    Calibre sigma de Black–Scholes pour reproduire un prix d'option cible.
    Utilise les moindres carrés : minimise (BS - market_price)^2.
    """
    def obj(sig):
        sig = float(np.maximum(sig, 1e-8))
        price = bs_call(S0, K, T, r, sig)
        return (price - market_price) ** 2

    res = minimize(lambda x: obj(x[0]),
                   x0=[sigma0],
                   method="Nelder-Mead")

    sigma_hat = float(max(res.x[0], 1e-8))
    return sigma_hat, res


# -------------------------------------------------------------------------
# 2. Calibration générique (maximum de vraisemblance, moments, etc.)
# -------------------------------------------------------------------------
def calibrate_model(objective,
                    theta0,
                    method: str = "Nelder-Mead",
                    bounds=None):
    """
    Calibre un modèle en minimisant une fonction objectif générale.
    (ex : -log-likelihood, distance de moments, etc.)

    Parameters
    ----------
    objective : callable(theta) -> float
        Fonction à minimiser (ex: -log-vraisemblance)
    theta0 : array_like
        Point de départ
    method : str
        Méthode d'optimisation scipy (Nelder-Mead, L-BFGS-B, Powell…)
    bounds : list of tuple or None
        Bornes éventuelles (pour L-BFGS-B)

    Returns
    -------
    theta_hat : ndarray
        Paramètres calibrés
    res : OptimizeResult
        Résultat scipy complet
    """

    res = minimize(objective,
                   x0=np.asarray(theta0, dtype=float),
                   method=method,
                   bounds=bounds)

    return res.x, res
