# 8INF852 — TP2 — Évolution discrète : recombinaison vs modélisation probabiliste
**Étudiant :** AITELOURF Youssef
**Session :** Hiver 2026

---

## Description

Ce projet compare deux philosophies d'algorithmes évolutionnaires discrets pour
faire évoluer des mots qui sonnent français sans être de vrais mots :

- **GA** (Algorithme Génétique) — recombinaison explicite avec crossover & mutation
- **UMDA** (EDA) — modélisation probabiliste des distributions marginales par position

La « franciosité » est mesurée via un modèle Markovien d'ordre 2 entraîné sur
Lexique383 (~150 000 mots français), et utilisé exclusivement comme boîte noire.

---

## Installation

```bash
pip install -r requirements.txt
```

Le dictionnaire et le modèle de langage sont téléchargés et mis en cache
automatiquement au premier lancement.

---

## Utilisation

```bash
# Toutes les expériences (recommandé)
python main.py

# Mode démonstration (une exécution)
python main.py --mode run_ga
python main.py --mode run_eda

# Expériences individuelles
python main.py --mode exp_crossover    # i.  Comparaison des croisements
python main.py --mode exp_etalon       # ii. Effet de l'étalon
python main.py --mode exp_losers       # iii. Effet des losers
python main.py --mode exp_ga_vs_eda    # iv. GA vs UMDA

# Options
python main.py --config config.yaml    # fichier de config (défaut)
python main.py --verbosity 2           # mode très verbeux
python main.py --output-dir mes_resultats/
python main.py --force-rebuild         # reconstruire dict + modèle
```

---

## Structure du projet

```
├── problem.py      Dictionnaire, modèle Markovien, perplexité, fonction objectif
├── algorithms.py   GA (GeneticAlgorithm) + EDA (UMDA)
├── experiments.py  Protocole Monte-Carlo et 4 expériences du barème
├── analysis.py     Statistiques descriptives et graphiques
├── main.py         Point d'entrée CLI
├── config.yaml     Tous les paramètres (modifiable sans toucher au code)
├── requirements.txt
└── lexique/        Cache auto-généré (dictionnaire + modèle)
```

---

## Paramètres configurables (config.yaml)

| Section | Paramètre | Description |
|---------|-----------|-------------|
| `general` | `verbosity` | 0/1/2 |
| `general` | `random_seeds` | Seeds PCG64 pour chaque run |
| `problem` | `dict_penalty` | Pénalité mot existant |
| `problem` | `repeat_penalty` | Pénalité répétitions > 3 |
| `ga` | `use_elitism/etalon/losers/reseeds` | Toggles des mécanismes de diversité |
| `ga` | `n_elites/losers/reseeds` | Taille de chaque mécanisme |
| `ga` | `pm`, `pc` | Probabilités de mutation et croisement |
| `ga` | `crossover` | `"single_point"` ou `"uniform"` |
| `eda` | `selection_ratio` | Fraction de sélection (top-k%) |
| `eda` | `smoothing` | Lissage de Laplace |
| `experiments` | `n_runs` | Nombre de runs Monte-Carlo |

---

## Sorties

Les résultats sont générés dans `results/` :

```
results/
├── exp_crossover/
│   ├── exp_crossover.csv           Historique brut
│   ├── exp_crossover.pkl           Pickle pandas
│   ├── exp_crossover_convergence.png
│   ├── exp_crossover_boxplot.png
│   ├── exp_crossover_diversity.png
│   ├── exp_crossover_stats.csv     Médiane, Q1, Q3, IQR, std
│   └── annexe_mots_*.csv           Meilleurs mots générés
├── exp_etalon/  ...
├── exp_losers/  ...
├── exp_ga_vs_eda/  ...
└── annexe/
    └── annexe_mots_*.csv           Top-500 mots par algorithme (annexe rapport)
```
