# Projet d’Implémentation de méthodes Quantitatives élémentaires

## 1. Présentation du dépot
Pour tout ce qui concerne la démarche et les techniques mathématiques utilisées merci de se référer au rapport.

## 2. Organisation du dépôt

```
.
├── README.md
├── requirements.txt
├── setup.cfg
├── .gitignore
├── data/                          # Données de marché (fichiers CSV)
├── notebooks/                     # Démonstrations et analyses
│   ├── 01_overview_and_data.ipynb
│   ├── 02_models_simulation.ipynb
│   ├── 03_pricing_comparison.ipynb
│   └── 04_calibration_and_risk.ipynb
├── src/                           # Code source du projet
│   ├── models/                    # Modèles stochastiques
│   │   ├── base_model.py          # Classe parente abstraite
│   │   ├── gbm.py                 # Geometric Brownian Motion
│   │   ├── heston.py              # Modèle de Heston
│   │   └── variance_gamma.py      # Modèle Variance Gamma
│   ├── pricing/                   # Méthodes de valorisation
│   │   ├── black_scholes.py       # Formules analytiques
│   │   └── monte_carlo.py         # Simulations de Monte-Carlo
│   ├── plotting.py                # Utilitaires graphiques
│   ├── simulation.py              # Wrappers de simulation (Exact / Euler)
│   ├── stats.py                   # Statistiques et moments (théoriques/empiriques)
│   └── utils.py                   # Fonctions auxiliaires
└── tests/                         # Validation et tests unitaires
    └── test_stats.py
```

---

## 3. Installation & exécution

Créer un venv puis :

```bash
pip install -r requirements.txt
```

Ouvrir les notebooks :

```bash
jupyter notebook notebooks/01_simulation_examples.ipynb
jupyter notebook notebooks/02_validation.ipynb
```

---

