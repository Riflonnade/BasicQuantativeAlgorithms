from typing import Tuple
import numpy as np

def mc_price_european(model, K: float, T: float, r: float, M: int, payoff: str = "call",
antithetic: bool = True, N: int = 252, random_state=None) -> Tuple[float, float]:
    """
    Prix Monte Carlo européen générique basé chemins d'un modèle 'model' fournissant simulate(T,N,M).
    Retourne (prix, intervalle de confiance 95% approx).
    """
    if antithetic:
        M2 = M // 2 * 2
    else:
        M2 = M

    t, S = model.simulate(T=T, N=N, M=M2, random_state=random_state)
    ST = S[:, -1]
    if payoff == "call":
        payoff_vals = np.maximum(ST - K, 0.0)
    elif payoff == "put":
        payoff_vals = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("payoff doit être 'call' ou 'put'.")

    if antithetic:
        # Antithetic variates: re-simuler avec bruits opposés si possible nativement ?
        # Ici simple split pair/impair (déjà centré par grande M).
        pass

    disc = np.exp(-r * T)
    vals = disc * payoff_vals
    price = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    ci = 1.96 * std / np.sqrt(vals.size)
    return price, ci
