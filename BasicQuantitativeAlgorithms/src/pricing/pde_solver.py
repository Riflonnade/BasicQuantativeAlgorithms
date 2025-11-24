import numpy as np

def crank_nicolson_bs(Smax: float, K: float, T: float, r: float, sigma: float,
                      M_S: int = 200, M_T: int = 200, option: str = "call"):
    """
    Résout l'EDP de Black–Scholes par Crank–Nicolson pour un call/put européen.
    Retourne (S_grid, V_grid) = valeurs au temps 0 sur la grille des spots.
    """
    dt = T / M_T
    dS = Smax / M_S
    S = np.linspace(0.0, Smax, M_S + 1)

    # Condition terminale à t = T
    if option == "call":
        V = np.maximum(S - K, 0.0)
    else:
        V = np.maximum(K - S, 0.0)

    # Marche en temps inverse
    for n in range(M_T, 0, -1):
        t = (n - 1) * dt

        # Indices internes i = 1,...,M_S-1 (M_S-1 points)
        i = np.arange(1, M_S)

        alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
        beta  = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

        # Coeffs matrice A (LHS, temps n-1)
        diag_A     = 1.0 - beta                # taille M_S-1
        off_down_A = -alpha[1:]                # taille M_S-2
        off_up_A   = -gamma[:-1]               # taille M_S-2

        # Coeffs RHS (temps n)
        D = 1.0 + beta                          # taille M_S-1
        E = alpha                               # taille M_S-1
        F = gamma                               # taille M_S-1

        # Valeurs internes de V
        V_i   = V[1:-1]                         # taille M_S-1
        V_im1 = V[0:-2]                         # taille M_S-1
        V_ip1 = V[2:]                           # taille M_S-1

        # Terme droit
        rhs = D * V_i + E * V_im1 + F * V_ip1   # taille M_S-1

        # Conditions aux bornes à l’instant t
        if option == "call":
            V0   = 0.0
            Vmax = Smax - K * np.exp(-r * (T - t))
        else:
            V0   = K * np.exp(-r * (T - t))
            Vmax = 0.0

        # Ajout des contributions de bord
        rhs[0]  += E[0]    * V0
        rhs[-1] += F[-1]   * Vmax

        # Construction de la matrice tridiagonale
        N_int = M_S - 1
        A_mat = np.zeros((N_int, N_int))
        np.fill_diagonal(A_mat, diag_A)
        np.fill_diagonal(A_mat[1:], off_down_A)
        np.fill_diagonal(A_mat[:, 1:], off_up_A)

        # Résolution pour V^{n-1}_i
        V[1:-1] = np.linalg.solve(A_mat, rhs)

        # Bords explicites
        V[0]  = V0
        V[-1] = Vmax

    # V est maintenant la solution à t=0
    return S, V
