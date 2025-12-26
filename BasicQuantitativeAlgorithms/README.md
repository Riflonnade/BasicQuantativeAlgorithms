# Projet d’Implémentation de méthodes Quantitatives élémentaires

## 1. Présentation du dépot
Pour tout ce qui concerne la démarche et les techniques mathématiques utilisées merci de se référer au rapport.

## 2. Organisation du dépôt

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

