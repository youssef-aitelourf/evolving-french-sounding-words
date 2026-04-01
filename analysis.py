"""
analysis.py — Analyse statistique et visualisation des résultats expérimentaux.

Fournit :
  - normalize_to_budget()      : alignement des runs sur des checkpoints d'évaluations
  - convergence_stats()        : médiane, Q1, Q3, min, max par checkpoint
  - final_stats()              : statistiques descriptives des fitness finales
  - plot_convergence()         : courbes de convergence avec bandes IQR
  - plot_boxplot_final()       : boîtes à moustaches des fitness finales
  - plot_diversity()           : évolution de la diversité
  - save_best_words_annex()    : tableau des 500 meilleurs mots (annexe obligatoire)
  - run_full_analysis()        : pipeline complet pour une expérience

Convention : les DataFrames sont ceux produits par experiments.run_monte_carlo().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style global
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_COLORS = plt.cm.tab10.colors  # palette cohérente


# ─────────────────────────────────────────────────────────────────────────────
# Alignement sur le budget d'évaluations
# ─────────────────────────────────────────────────────────────────────────────

def normalize_to_budget(
    df: pd.DataFrame,
    n_checkpoints: int = 100,
) -> pd.DataFrame:
    """Aligne les historiques de tous les runs sur des checkpoints uniformes.

    Problème : différentes configurations peuvent avoir des eval_count légèrement
    différents par génération. Cette fonction résout le problème en créant
    n_checkpoints réguliers en termes d'évaluations et en faisant un forward-fill
    de la meilleure fitness connue à chaque checkpoint.

    Args:
        df            : DataFrame brut de run_monte_carlo()
        n_checkpoints : nombre de points d'alignement

    Returns:
        DataFrame avec colonnes : run, algo, checkpoint_eval, best_fitness, diversity
    """
    max_eval = int(df["eval_count"].max())
    checkpoints = np.linspace(0, max_eval, n_checkpoints + 1, dtype=int)[1:]

    rows: list[dict] = []
    for (run, algo), group in df.groupby(["run", "algo"]):
        g = group.sort_values("eval_count")
        eval_arr = g["eval_count"].to_numpy()
        fit_arr = g["best_fitness"].to_numpy()
        div_arr = g["diversity"].to_numpy()

        for cp in checkpoints:
            # Dernier enregistrement avant ou au checkpoint
            mask = eval_arr <= cp
            if not mask.any():
                continue
            idx = int(np.where(mask)[0][-1])
            rows.append(
                {
                    "run": run,
                    "algo": algo,
                    "checkpoint_eval": int(cp),
                    "best_fitness": float(fit_arr[idx]),
                    "diversity": float(div_arr[idx]),
                }
            )

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Statistiques
# ─────────────────────────────────────────────────────────────────────────────

def convergence_stats(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Calcule médiane, Q1, Q3, min, max par (algo, checkpoint).

    Args:
        df_norm : DataFrame normalisé (sortie de normalize_to_budget)

    Returns:
        DataFrame agrégé trié par algo + checkpoint.
    """
    stats = (
        df_norm.groupby(["algo", "checkpoint_eval"], as_index=False)
        .agg(
            median_fit=("best_fitness", "median"),
            q1_fit=("best_fitness", lambda x: float(np.quantile(x, 0.25))),
            q3_fit=("best_fitness", lambda x: float(np.quantile(x, 0.75))),
            min_fit=("best_fitness", "min"),
            max_fit=("best_fitness", "max"),
            median_div=("diversity", "median"),
            q1_div=("diversity", lambda x: float(np.quantile(x, 0.25))),
            q3_div=("diversity", lambda x: float(np.quantile(x, 0.75))),
        )
        .sort_values(["algo", "checkpoint_eval"])
    )
    return stats


def final_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques descriptives des fitness finales (dernier enregistrement par run).

    Returns:
        DataFrame avec colonnes : algo, n_runs, best, median, q1, q3, iqr, std, worst
    """
    final = (
        df.sort_values(["algo", "run", "eval_count"])
        .groupby(["algo", "run"], as_index=False)
        .tail(1)
    )
    table = (
        final.groupby("algo")["best_fitness"]
        .agg(
            n_runs="count",
            best="min",
            median="median",
            q1=lambda x: float(np.quantile(x, 0.25)),
            q3=lambda x: float(np.quantile(x, 0.75)),
            std="std",
            worst="max",
        )
        .reset_index()
    )
    table["iqr"] = table["q3"] - table["q1"]
    table = table[["algo", "n_runs", "best", "median", "q1", "q3", "iqr", "std", "worst"]]
    return table.round(4)


# ─────────────────────────────────────────────────────────────────────────────
# Graphiques
# ─────────────────────────────────────────────────────────────────────────────

def plot_convergence(
    stats_df: pd.DataFrame,
    title: str,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """Courbes de convergence : médiane + bande IQR par algorithme.

    L'axe x est le nombre d'évaluations de la fonction objectif (budget).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (algo, data) in enumerate(stats_df.groupby("algo")):
        c = _COLORS[idx % len(_COLORS)]
        ax.plot(
            data["checkpoint_eval"],
            data["median_fit"],
            label=f"{algo} (médiane)",
            color=c,
            linewidth=2,
        )
        ax.fill_between(
            data["checkpoint_eval"],
            data["q1_fit"],
            data["q3_fit"],
            alpha=0.2,
            color=c,
            label=f"{algo} (IQR)",
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Nombre d'évaluations de la fonction objectif", fontsize=12)
    ax.set_ylabel("Meilleure fitness (perplexité + pénalités)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_boxplot_final(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Boîtes à moustaches des fitness finales par algorithme."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dernier enregistrement par (algo, run)
    final = (
        df.sort_values(["algo", "run", "eval_count"])
        .groupby(["algo", "run"], as_index=False)
        .tail(1)
    )

    algos = sorted(final["algo"].unique())
    data_per_algo = [final[final["algo"] == a]["best_fitness"].to_numpy() for a in algos]

    fig, ax = plt.subplots(figsize=(max(6, len(algos) * 2.5), 5))
    bp = ax.boxplot(
        data_per_algo,
        labels=algos,
        patch_artist=True,
        notch=False,
        medianprops={"color": "black", "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], _COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Fitness finale (perplexité + pénalités)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_diversity(
    stats_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """Évolution de la diversité (fraction de mots uniques) par algorithme."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, (algo, data) in enumerate(stats_df.groupby("algo")):
        c = _COLORS[idx % len(_COLORS)]
        ax.plot(
            data["checkpoint_eval"],
            data["median_div"],
            label=f"{algo} (médiane)",
            color=c,
            linewidth=2,
        )
        ax.fill_between(
            data["checkpoint_eval"],
            data["q1_div"],
            data["q3_div"],
            alpha=0.2,
            color=c,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Nombre d'évaluations de la fonction objectif", fontsize=12)
    ax.set_ylabel("Diversité (fraction mots uniques)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Annexe — Meilleurs mots générés
# ─────────────────────────────────────────────────────────────────────────────

def save_best_words_annex(
    pools: dict[str, dict[str, float]],
    dictionary_set: set[str],
    output_dir: Path,
    top_n: int = 500,
) -> dict[str, Path]:
    """Sauvegarde les top_n meilleurs mots générés par algorithme (annexe obligatoire).

    Utilise le word_pool accumulé lors des runs Monte-Carlo (top-10 par génération,
    toutes exécutions confondues). Ce pool est bien plus riche que le seul suivi du
    meilleur global et permet d'atteindre les 500 mots requis.

    Pour chaque algorithme :
      - Filtre les mots existants dans le dictionnaire (pénalité douce → peuvent
        être présents dans le pool mais ne doivent pas figurer dans l'annexe)
      - Trie par fitness ascendante (meilleur = plus petit = plus français)
      - Sauvegarde en CSV (colonnes : rang, mot, fitness)

    Returns:
        Dictionnaire {algo: chemin_csv}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for algo, pool in pools.items():
        rows = [
            {"mot": w, "fitness": f}
            for w, f in pool.items()
            if w not in dictionary_set
        ]
        word_df = (
            pd.DataFrame(rows)
            .sort_values("fitness")
            .reset_index(drop=True)
            .head(top_n)
        )
        word_df.insert(0, "rang", range(1, len(word_df) + 1))

        csv_path = output_dir / f"annexe_mots_{algo}.csv"
        word_df.to_csv(csv_path, index=False, encoding="utf-8")
        paths[str(algo)] = csv_path

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline complet
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    dfs: dict[str, pd.DataFrame],
    experiment_name: str,
    output_dir: Path,
    n_checkpoints: int = 100,
    log_scale: bool = True,
) -> dict[str, Path]:
    """Pipeline d'analyse complet pour une expérience.

    Produit :
      - Courbe de convergence (médiane + IQR)
      - Boîte à moustaches des fitness finales
      - Courbe de diversité
      - Tableau de statistiques descriptives (CSV)

    Returns:
        Dictionnaire {type_sortie: chemin} des fichiers produits.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    combined = pd.concat(list(dfs.values()), ignore_index=True)

    # Alignement sur le budget
    df_norm = normalize_to_budget(combined, n_checkpoints)
    stats = convergence_stats(df_norm)

    # 1. Convergence
    conv_path = output_dir / f"{experiment_name}_convergence.png"
    plot_convergence(
        stats,
        title=f"Convergence — {experiment_name.replace('_', ' ').title()}",
        output_path=conv_path,
        log_scale=log_scale,
    )
    paths["convergence"] = conv_path

    # 2. Boîtes à moustaches
    box_path = output_dir / f"{experiment_name}_boxplot.png"
    plot_boxplot_final(
        combined,
        title=f"Distribution des fitness finales — {experiment_name.replace('_', ' ').title()}",
        output_path=box_path,
    )
    paths["boxplot"] = box_path

    # 3. Diversité
    div_path = output_dir / f"{experiment_name}_diversity.png"
    plot_diversity(
        stats,
        title=f"Diversité — {experiment_name.replace('_', ' ').title()}",
        output_path=div_path,
    )
    paths["diversity"] = div_path

    # 4. Statistiques descriptives
    fstats = final_stats(combined)
    stats_path = output_dir / f"{experiment_name}_stats.csv"
    fstats.to_csv(stats_path, index=False, encoding="utf-8")
    paths["stats"] = stats_path

    return paths
