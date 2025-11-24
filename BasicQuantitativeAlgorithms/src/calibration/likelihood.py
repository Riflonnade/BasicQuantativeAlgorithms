import numpy as np

def gbm_mle_from_prices(prices: np.ndarray, dt: float):
    """
    Estimateurs MLE fermés pour un GBM à partir d'une série de prix S_t.
    Retourne (mu_hat, sigma_hat).
    """
    prices = np.asarray(prices, dtype=float)
    log_ret = np.diff(np.log(prices))
    mu_hat = (np.mean(log_ret) + 0.5 * np.var(log_ret)) / dt
    sigma_hat = np.sqrt(np.var(log_ret) / dt)
    return float(mu_hat), float(sigma_hat)


def log_likelihood(theta, log_returns: np.ndarray) -> float:
    """
    Log-vraisemblance d'un GBM pour une série de rendements log.

    Paramètres
    ----------
    theta : (mu, sigma)
        mu : drift (par pas de temps)
        sigma : volatilité (par pas de temps, > 0)
    log_returns : array_like
        R_i = log(S_{i+1} / S_i) pour i = 0,...,n-1 (on suppose dt = 1).

    Retour
    ------
    float
        Valeur de la log-vraisemblance.
    """
    log_returns = np.asarray(log_returns, dtype=float)
    mu, sigma = float(theta[0]), float(theta[1])

    if sigma <= 0:
        # sigma non valide -> -inf pour que l'optimiseur évite cette zone
        return -np.inf

    n = log_returns.size
    m = mu - 0.5 * sigma**2          # espérance par pas
    v = sigma**2                     # variance par pas

    resid = log_returns - m
    ll = -0.5 * n * np.log(2 * np.pi * v) - 0.5 * np.sum(resid**2) / v
    return float(ll)
