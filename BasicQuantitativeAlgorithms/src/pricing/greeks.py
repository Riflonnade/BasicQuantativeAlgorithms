from .black_scholes import bs_call, bs_put

def greeks_fd(S0: float, K: float, T: float, r: float, sigma: float, payoff="call", eps=1e-4):
    """
    Grecques par différences finies sur Black–Scholes (Delta, Gamma, Vega, Theta, Rho).
    """
    price_fun = bs_call if payoff == "call" else bs_put
    # Delta & Gamma (variation S0)
    p0 = price_fun(S0, K, T, r, sigma)
    p_up = price_fun(S0 * (1 + eps), K, T, r, sigma)
    p_dn = price_fun(S0 * (1 - eps), K, T, r, sigma)
    delta = (p_up - p_dn) / (2 * S0 * eps)
    gamma = (p_up - 2 * p0 + p_dn) / (S0 * eps) ** 2

    # Vega (variation sigma)
    ps_up = price_fun(S0, K, T, r, sigma * (1 + eps))
    ps_dn = price_fun(S0, K, T, r, sigma * (1 - eps))
    vega = (ps_up - ps_dn) / (2 * sigma * eps)

    # Rho (variation r)
    pr_up = price_fun(S0, K, T, r * (1 + eps), sigma)
    pr_dn = price_fun(S0, K, T, r * (1 - eps), sigma)
    rho = (pr_up - pr_dn) / (2 * r * eps) if r != 0 else (pr_up - pr_dn) / (2 * eps)

    # Theta (variation T)
    pt_up = price_fun(S0, K, T * (1 + eps), r, sigma)
    pt_dn = price_fun(S0, K, T * (1 - eps), r, sigma)
    theta = (pt_dn - pt_up) / (2 * T * eps)

    return {"delta": delta, "gamma": gamma, "vega": vega, "rho": rho, "theta": theta}
