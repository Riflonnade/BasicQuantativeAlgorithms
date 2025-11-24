import math

def _N(x: float) -> float:
    # CDF de la loi normale standard via erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, S0 - K)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S0 * _N(d1) - K * math.exp(-r * T) * _N(d2)

def bs_put(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(0.0, K - S0)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) - S0 + bs_call(S0, K, T, r, sigma)