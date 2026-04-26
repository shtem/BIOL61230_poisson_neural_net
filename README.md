# BIOL61230 — Poisson Neural Network for Neural Response Prediction

MSc Bioinformatics dissertation project. Predicts spike counts from mouse dorsal lateral geniculate nucleus (dLGN) neurons using a suite of Poisson regression models, including transfer-learning neural networks trained jointly across all cells.

---

## Overview

Visual stimuli and 3D body-tracking covariates (position, velocity, tilt) recorded simultaneously with single-unit activity from 100 dLGN neurons across two recording sessions. The core question is whether a shared neural network feature extractor — trained across all cells — improves per-cell spike count prediction over independent baselines.

**Evaluation metric:** Poisson pseudo-R² = `1 − (D_model / D_null)`, where D is deviance. Values above 0 indicate performance better than the null (mean-rate) model.

---

## Models

| Model | Description |
|---|---|
| GLM | Poisson GLM with grid-searched L2 regularisation |
| XGBoost | Gradient-boosted trees with Poisson loss |
| NN-PerCell-{MLP,CNN,RNN} | Independent neural networks fit separately per cell |
| NN-DeepSharedShallowHead-TL-{MLP,CNN,RNN} | Shared deep extractor + shallow per-cell output head |
| NN-DeepSharedDeepHead-TL-{MLP,CNN,RNN} | Shared deep extractor + deep per-cell output head |
| NN-ShallowSharedDeepHead-TL-{MLP,CNN,RNN} | Shared shallow extractor + deep per-cell output head |

Transfer-learning (TL) models share a feature extractor across all cells and attach a separate output head per cell. The three extractor types (MLP, CNN, RNN) and three head-depth combinations give 9 TL variants.

---

## Repository Structure

```
src/
  get_data.py              — Data loading, imputation, train/val/test splitting
  visualisation.py         — All plotting functions and journal figure helpers
  clustering_tools.py      — Correlation- and GLM-based clustering utilities
  train/
    training.py            — run_experiment() orchestrator (cache, fit, save, plot)
    evaluate.py            — pseudo_r2() and related metrics
    hyperparam_search.py   — Grid search and cross-validation helpers
    io.py                  — save_model() / load_model() / save_plot()
    utils.py               — Shared tensor/device utilities
    poisson_baseline/
      baseline_main.py     — GLM and XGBoost fitting functions
    poisson_nn/
      nn_models.py         — PyTorch model definitions (per-cell and TL architectures)
      nn_training.py       — Trainers (per-cell and transfer-learning)
      nn_main.py           — fit_poisson_nn() and fit_poisson_nn_transfer_learning()
  stats/
    batch_statistical_analysis.py        — Per-batch summary statistics and Wilcoxon tests
    cross_batch_statistics_aggregation.py — Combined statistics across all batches

notebooks/
  neural_network_simulated.ipynb         — Full pipeline on synthetic data (validation)
  neural_network_real_batch{0-3}.ipynb   — Per-batch training on real dLGN data
  results_figures.ipynb                  — Generates all dissertation figures
  results_verification.ipynb             — Sanity checks on saved results
  data_playbook.ipynb                    — Data exploration

resources/
  data/
    simulated/             — Synthetic .mat files
    real/                  — Real .mat file
  models/
    simulated/             — Fitted model pickles (.pkl)
    real/batch_{0-3}/      — Per-batch model pickles
  results/
    simulated/             — Per-model training curves and scatter plots
    real/batch_{0-3}/      — Per-batch diagnostic plots
    real/combined/         — Cross-batch aggregate CSVs and Wilcoxon tables
    figures/               — Final dissertation figures (PNG + PDF)
```

---

## Data

**Real data** (`Temi_Data.mat`, gitignored): 100 dLGN cells recorded across two sessions. Covariates include visual stimulus parameters and 3D body-tracking signals. Training uses a temporal 70 / 15 / 15 train / val / test split per cell to preserve time-series structure. NaN values from 3D tracking failures are imputed (mean strategy) using only training-bin statistics.

**Simulated data** (`resources/data/simulated/`): Four synthetic datasets used to validate the full pipeline before running on real recordings.

Cells are processed in four batches of 25:

| Batch | Cells | Session |
|---|---|---|
| 0 | 1–25 | Session 1 |
| 1 | 26–50 | Session 1 |
| 2 | 51–75 | Sessions 1–2 |
| 3 | 76–100 | Session 2 |

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. GPU training is supported automatically if CUDA is available.

---

## Running the Pipeline

**Simulated data (end-to-end validation):**

Open and run `notebooks/neural_network_simulated.ipynb`.

**Real data (per-batch training):**

Run each batch notebook in order:
```
notebooks/neural_network_real_batch0.ipynb
notebooks/neural_network_real_batch1.ipynb
notebooks/neural_network_real_batch2.ipynb
notebooks/neural_network_real_batch3.ipynb
```

Each notebook saves fitted models to `resources/models/real/batch_{N}/` and per-model diagnostic plots to `resources/results/real/batch_{N}/`. Completed models are cached — re-running a cell with an existing `.pkl` loads from cache rather than refitting.

**Cross-batch statistics:**

After all four batches are complete, run:
```bash
python src/stats/cross_batch_statistics_aggregation.py
```

This produces combined summary tables and Wilcoxon signed-rank test results in `resources/results/real/combined/`.

**Dissertation figures:**

Open and run `notebooks/results_figures.ipynb`. All figures are saved as PNG (300 DPI) and PDF to `resources/results/figures/`.

---

## Statistical Analysis

Model comparisons use one-sided Wilcoxon signed-rank tests (alternative: model > baseline) applied to per-cell pseudo-R² distributions. No multiple-testing correction is applied: hypotheses are pre-specified, directional, and the tests are correlated (same cells).
