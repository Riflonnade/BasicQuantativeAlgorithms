import numpy as np


# -------------------------------------------------------------------------
# 1. Moments empiriques sur une série de rendements log
# -------------------------------------------------------------------------
def empirical_moments(logreturns: np.ndarray):
    """
    Calcule les 4 premiers moments empiriques d'une série de log-rendements :
    moyenne, variance, asymétrie, kurtosis.
    """
    lr = np.asarray(logreturns, dtype=float)
    mu = np.mean(lr)
    var = np.var(lr)
    std = np.sqrt(var)

    skew = np.mean(((lr - mu) / std) ** 3)
    kurt = np.mean(((lr - mu) / std) ** 4) - 3  # kurtosis centrée

    return mu, var, skew, kurt


# -------------------------------------------------------------------------
# 2. Moments théoriques approx. du modèle Heston
# -------------------------------------------------------------------------
def heston_theoretical_moments(theta):
    """
    Moments théoriques approchés du Heston pour un pas de temps court (dt = 1).

    Paramètres
    ----------
    theta = (kappa, theta_bar, xi, rho, v0)

    Retour
    ------
    (mean, variance, skewness, kurtosis)
    """

    kappa, theta_bar, xi, rho, v0 = theta

    # 1) mean approx. : E[log(S)] ≈ ( -0.5 E[v] )
    mean = -0.5 * theta_bar

    # 2) variance approx. : Var(log(S)) ≈ E[v]
    variance = theta_bar

    # 3) skew approximation : dépend de rho et xi
    skew = rho * xi / np.sqrt(variance + 1e-12)

    # 4) kurtosis approx : dépend du vol-of-vol
    kurt = 3 + (xi**2) / (kappa + 1e-12)

    return mean, variance, skew, kurt


# -------------------------------------------------------------------------
# 3. Fonction objectif (distance entre moments)
# -------------------------------------------------------------------------
def moments_objective(theta, logreturns):
    """
    Distance quadratique entre moments empiriques et moments Heston.
    Utilisée pour calibrer theta par moindres carrés.
    """
    emp = np.array(empirical_moments(logreturns))
    theo = np.array(heston_theoretical_moments(theta))

    return np.sum((emp - theo)**2)


# -------------------------------------------------------------------------
# 4. Calibration directe par moindres carrés (utilise moments_objective)
# -------------------------------------------------------------------------
from scipy.optimize import minimize

def calibrate_heston_moments(logreturns, theta0):
    """
    Calibre un modèle Heston en minimisant la distance des moments.
    """
    res = minimize(lambda th: moments_objective(th, logreturns),
                   x0=np.asarray(theta0, dtype=float),
                   method="Nelder-Mead")

    return res.x, res


import numpy as np

def gbm_empirical_moments(logreturns):
    """
    Moments empiriques pour un GBM :
    mean ≈ mu - 0.5 sigma²
    var ≈ sigma²
    """
    m1 = np.mean(logreturns)
    m2 = np.var(logreturns)
    return np.array([m1, m2])


def gbm_theoretical_moments(theta):
    """
    Moments théoriques d'un GBM pour un pas dt=1.
    theta = (mu, sigma)
    """
    mu, sigma = theta
    m1 = mu - 0.5 * sigma**2
    m2 = sigma**2
    return np.array([m1, m2])


def gbm_moments_objective(theta, logreturns):
    """
    Distance quadratique entre moments empiriques et moments théoriques (GBM).
    """
    emp = gbm_empirical_moments(logreturns)
    theo = gbm_theoretical_moments(theta)
    return np.sum((emp - theo)**2)
