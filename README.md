# Evolving French-sounding words

**Genetic algorithms vs. estimation-of-distribution — without ever putting real French words in the output.**

Two evolutionary approaches compete to invent strings that *feel* French (low perplexity under a character-level Markov model) while staying **out of the dictionary**. The code runs a full Monte Carlo study: crossover operators, an “étalon” (champion breeding) operator, deliberate retention of the **worst** individuals for diversity, and a head-to-head against **UMDA** (univariate marginal distribution algorithm).

Write-up with figures and interpretation: **[What If an AI Invented French Words?](https://medium.com/@youssefaitelourf/what-if-an-ai-invented-french-words-51c37fb6e43c)** (Medium)

---

## What this repo contains

| Piece | Role |
|--------|------|
| **Fitness** | Second-order Markov (trigram) model trained on [Lexique383](http://www.lexique.org/) (~125k lemmas). Perplexity + soft dictionary penalty + validity penalties. |
| **GA** | Tournament selection, single-point or uniform crossover, mutation, elitism, optional étalon and “losers” pool. |
| **UMDA** | Selects top fraction, fits per-position character marginals (+ length), samples new population. |
| **Experiments** | Four studies: crossover type, étalon on/off, losers on/off, best GA vs UMDA — each **10 seeds**, PCG64, comparable evaluation budgets. |
| **Outputs** | CSV histories, stats, convergence / boxplot / diversity plots under `results/`. |

### Headline results (see the article for full analysis)

- Keeping the **five worst** solutions each generation improved median fitness by **~8%** vs discarding them; diversity collapsed without them.
- **GA** beat **UMDA** by a large margin on median and best fitness — the fitness is explicitly **Markovian** (dependencies between adjacent characters); UMDA’s independence assumption does not match that structure.
- Single-point vs uniform crossover: **same** median best fitness in this setup; uniform crossover was slightly **more consistent** (lower IQR).

---

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

The lexicon and language model are built or loaded from `lexique/` on first run.

### Useful CLI modes

```bash
python main.py --mode run_ga          # single GA demo
python main.py --mode run_eda         # single UMDA demo
python main.py --mode exp_crossover   # crossover comparison
python main.py --mode exp_etalon      # étalon on/off
python main.py --mode exp_losers      # losers on/off
python main.py --mode exp_ga_vs_eda   # GA vs UMDA
python main.py --config config.yaml
python main.py --output-dir my_results/
python main.py --force-rebuild        # rebuild dict + Markov model
```

---

## Project layout

```
problem.py      Dictionary, Markov model, perplexity, objective
algorithms.py   GeneticAlgorithm + UMDA
experiments.py  Monte Carlo harness and the four experiments
analysis.py     Summaries and plots
main.py         CLI entry
config.yaml     Hyperparameters (no code edits required)
lexique/        Cached lexicon + model (auto-generated)
results/        Run outputs (CSV, PNG, pickles)
```

---

## Configuration

All tunables live in `config.yaml` (verbosity, seeds, GA toggles, crossover type, EDA selection ratio, experiment repeat count, penalties, etc.).

---

## Reproducibility

Experiments use fixed **PCG64** seeds from config; reported runs use **10 seeds** per configuration unless you change `experiments.n_runs`. Figures and tables in the Medium piece correspond to this pipeline.

---

## License / attribution

Lexique383 is an external lexical resource; cite Lexique.org if you reuse their data. This repository is the companion code for the Medium article above.
