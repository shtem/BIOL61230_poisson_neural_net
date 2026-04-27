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

Transfer-learning (TL) models share a single feature extractor across all cells and attach a separate output head per cell. The three extractor types (MLP, CNN, RNN) and three head-depth combinations give 9 TL variants. All models output Poisson-distributed spike counts via a Softplus activation on the final layer.

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
  data_playbook.ipynb                    — Data exploration and cell clustering

resources/
  data/
    simulated/             — Synthetic .mat files
    real/                  — real_data.mat
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

**Real data** (`resources/data/real/real_data.mat`): 100 dLGN cells recorded across two sessions. The full `.mat` file contains 247 cells and 5 recordings; only cells 1–100 are used for model training. Covariates include visual stimulus parameters (orientation, ON/OFF responses) and 3D body-tracking signals (tilt, angular and linear head speed). The full covariate list is defined in `COVARIATE_NAMES_REAL` in `src/visualisation.py`.

**Simulated data** (`resources/data/simulated/`): Four synthetic datasets (`test1–4.mat`) with 5 cells each and known ground-truth spike rates, used to validate the full pipeline end-to-end before running on real recordings.

### Data splits

All splits are **temporal** (no shuffling) to preserve the time-series structure of spike trains:

| Split | Fraction | Purpose |
|---|---|---|
| Train | 70% | Model fitting |
| Val | 15% | Early stopping and hyperparameter selection |
| Test | 15% | Final evaluation only |

NaN values (from 3D tracking failures) are imputed using a mean strategy. The imputer is fitted on training bins only to prevent data leakage.

### Batch breakdown

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

Requires Python 3.10+. GPU training is supported automatically when CUDA is available — the device is resolved once at import time in `src/train/utils.py`.

---

## Running the Pipeline

**1. Validate on simulated data (optional but recommended first):**

Open and run `notebooks/neural_network_simulated.ipynb`.

**2. Train on real data (per-batch):**

Run each batch notebook in order. They are independent and can be run in separate sessions:

```
notebooks/neural_network_real_batch0.ipynb
notebooks/neural_network_real_batch1.ipynb
notebooks/neural_network_real_batch2.ipynb
notebooks/neural_network_real_batch3.ipynb
```

Each notebook saves fitted models to `resources/models/real/batch_{N}/` as `.pkl` files and writes diagnostic plots to `resources/results/real/batch_{N}/`. **Models are cached** — re-running a cell skips training if the `.pkl` already exists. To force retraining, set `FORCE_EXPERIMENTS = True` at the top of the notebook.

**3. Aggregate statistics across all batches:**

```bash
python src/stats/cross_batch_statistics_aggregation.py
```

Reads the per-batch `.pkl` files and writes combined summary CSVs and Wilcoxon test tables to `resources/results/real/combined/`.

**4. Generate dissertation figures:**

Open and run `notebooks/results_figures.ipynb`. All figures are saved as PNG (300 DPI) and PDF to `resources/results/figures/`.

---

## Key Design Decisions

**Hyperparameter search:** Grid search is run per cell for GLM, XGBoost, and per-cell NNs; a single shared grid search (aggregated across cells by median pseudo-R²) is used for TL models, since their shared extractor must be trained on all cells jointly.

**Transfer learning training:** The shared extractor and all cell heads are trained end-to-end in a single pass. Each batch presents all cells; gradients flow through both shared and head components simultaneously.

**Model caching:** `run_experiment()` checks for an existing `.pkl` before fitting. This makes it safe to re-run notebooks after interruptions without losing completed work.

---

## Statistical Analysis

Model comparisons use one-sided Wilcoxon signed-rank tests (alternative: model > baseline) applied to per-cell pseudo-R² distributions. No multiple-testing correction is applied: hypotheses are pre-specified, directional, and the tests are correlated across models (same 100 cells).
