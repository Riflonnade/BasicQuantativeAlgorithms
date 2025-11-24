# Simulation de modèles stochastiques de prix d’actifs — Projet GBM

Ce dépôt présente une **implémentation propre, testée et reproductible** de la simulation d’un actif financier sous le **mouvement brownien géométrique (GBM / modèle de Black–Scholes)**, avec :
- une **simulation exacte** via la solution fermée,
- une **approximation d’Euler–Maruyama** (utile pédagogiquement),
- une **validation statistique** (moments empiriques vs théoriques),
- des **tests unitaires** (`pytest`),
- des **notebooks** pour visualiser et documenter la démarche.

> Objectif : montrer la capacité à passer de la **formulation mathématique** à une **implémentation Python** rigoureuse,
> afin de valoriser une candidature en **recherche en finance quantitative**.

---

## 1. Modèle mathématique et propriétés

### 1.1. Équation différentielle stochastique (EDS)

On modélise le prix S_t par :

`dS_t = μ S_t dt + σ S_t dW_t`,

où μ est le drift, σ la volatilité et W_t un mouvement brownien standard.

### 1.2. Solution fermée (simulation exacte)

Cette EDS admet la solution explicite :

`S_t = S_0 * exp( (μ - 1/2 σ^2) t + σ W_t )`.

Il s’ensuit que le **log-rendement** log(S_t/S_0) est gaussien :

`log(S_t/S_0) ~ Normal( (μ - 1/2 σ^2) t , σ^2 t )`.

### 1.3. Schéma d’Euler–Maruyama (approximation)

Le schéma discret pour un pas Δt est :

`S_{k+1} = S_k + μ S_k Δt + σ S_k ΔW_k`, avec `ΔW_k ~ Normal(0, Δt)`.

Ce schéma sert de **point de comparaison** et illustre la simulation d’EDS lorsque la solution fermée n’est pas disponible (p.ex. Heston).

---

## 2. Démarche scientifique & validation

1. **Simulation exacte** : génération de W_t par sommes d’incréments gaussiens, puis application de la solution fermée.
2. **Approximation Euler** : intégration étape par étape via les incréments browniens.
3. **Validation statistique** : on compare, sur un grand nombre de trajectoires, les moments **empiriques** de log(S_T/S_0) aux valeurs **théoriques** :

- E[log(S_T/S_0)] = (μ - 1/2 σ^2) T
- Var[log(S_T/S_0)] = σ^2 T

4. **Tests** : `pytest` vérifie la proximité (tolérances configurées) et la stabilité numérique.

**Complexité** : la simulation naïve coûte O(MN) pour M trajectoires et N pas de temps.

---

## 3. Organisation du dépôt

```
.
├── README.md
├── requirements.txt
├── data/
├── notebooks/
│   ├── 01_simulation_examples.ipynb   # démonstration GBM
│   └── 02_validation.ipynb            # (ajouté) validation moments
├── src/
│   ├── __init__.py
│   ├── models.py                      # GeometricBrownianMotion (solution exacte)
│   ├── plotting.py                    # visualisations simples
│   ├── simulation.py                  # wrappers (exact / euler)
│   └── stats.py                       # moments théoriques / stats empiriques
└── tests/
    ├── test_models.py                 # (fourni initialement)
    └── test_stats.py                  # (ajout) validation moments
```

---

## 4. Installation & exécution

Créer un venv puis :

```bash
pip install -r requirements.txt
pytest
```

Ouvrir les notebooks :

```bash
jupyter notebook notebooks/01_simulation_examples.ipynb
jupyter notebook notebooks/02_validation.ipynb
```

---

## 5. Utilisation rapide (code)

### 5.1. Simulation exacte
```python
from src.models import GeometricBrownianMotion
from src.plotting import plot_paths

gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2, S0=100)
t, S = gbm.simulate(T=1.0, N=252, M=10000)
plot_paths(t, S[:30], title="GBM — trajectoires (solution exacte)")
```

### 5.2. Euler–Maruyama
```python
from src.simulation import simulate_gbm
t, S_euler = simulate_gbm(mu=0.05, sigma=0.2, S0=100, T=1.0, N=252, M=10000, method="euler", random_state=0)
```

### 5.3. Validation des moments
```python
from src.stats import empirical_log_return_stats, gbm_theoretical_log_return_moments

emp_mean, emp_var = empirical_log_return_stats(S, S0=100)
th_mean, th_var = gbm_theoretical_log_return_moments(mu=0.05, sigma=0.2, T=1.0)
print("Empirical mean:", emp_mean, "| Theoretical mean:", th_mean)
print("Empirical var :", emp_var,  "| Theoretical var :", th_var)
```

---

## 6. Bonnes pratiques
- Reproductibilité (seeds / `random_state`)
- Tests unitaires et tolérances explicites
- Séparation claire du code (modèles / stats / plotting / tests / notebooks)

---

## 7. Extensions possibles
- Modèle de Heston (variance stochastique, browniens corrélés)
- Variance Gamma (processus de Lévy)
- Pricing d’options (Monte Carlo vs formule de Black–Scholes)
- Calibration sur données de marché (MLE / moments / moindres carrés)
- Étude de convergence des schémas numériques

---

## 8. Licence & contact
Projet éducatif. Contributions bienvenues via issues / PR.
