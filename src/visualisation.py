import numpy as np
import matplotlib.pyplot as plt


def _set_journal_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.8,
        }
    )


def compute_psth(y, bins_per_trial=25, smooth_sigma=None):
    """ ""
    Compute a peri-stimulus time histogram (PSTH) from spike data.
    PSTH is computed by averaging spike counts across trials for each time bin.

    :param y: Array of spike counts for each time bin
    :param bins_per_trial: Number of bins per trial
    :param smooth_sigma: Optional standard deviation for Gaussian smoothing

    :return: Array of PSTH values for each bin
    """
    n_time = len(y)
    n_trials = n_time // bins_per_trial

    # reshape into (trials, bins)
    y_trials = y[: n_trials * bins_per_trial].reshape(n_trials, bins_per_trial)

    # average across trials
    psth = y_trials.mean(axis=0)

    # optional Gaussian smoothing
    if smooth_sigma is not None:
        from scipy.ndimage import gaussian_filter1d

        psth = gaussian_filter1d(psth, sigma=smooth_sigma)

    return psth


def plot_psth(psth_true, psth_pred, title="PSTH Comparison"):
    """
    Plot peri-stimulus time histogram (PSTH) comparison.

    :param psth_true: Array of true PSTH values
    :param psth_pred: Array of predicted PSTH values
    :param title: Title for the plot

    :return: Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(psth_true, label="Actual", linewidth=2)
    ax.plot(psth_pred, label="Predicted", linewidth=2)

    ax.set_xlabel("Time bin within trial")
    ax.set_ylabel("Firing rate (spikes/bin)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.close(fig)

    return fig


def plot_ytrue_vs_ypred(
    y_true, y_pred, title="", xlabel="True spike count", ylabel="Predicted spike count"
):
    _set_journal_style()
    fig, ax = plt.subplots(figsize=(4, 4))

    ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    plt.close(fig)

    return fig


def compare_models_for_cell(glm_results, xgb_results, nn_results, cell, split="test"):
    _set_journal_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True, sharey=True)

    for ax, (results, name) in zip(
        axes,
        [
            (glm_results, "GLM"),
            (xgb_results, "XGBoost"),
            (nn_results, "Neural Network"),
        ],
    ):
        y_true = results[cell][f"y_{split}"]
        y_pred = results[cell][f"y_pred_{split}"]

        ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolor="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_title(f"{name} — Cell {cell}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    fig.suptitle(f"Model comparison — Cell {cell} ({split})", y=1.02)
    fig.tight_layout()

    plt.close(fig)

    return fig


def compare_r2_across_cells(glm_results, xgb_results, nn_results, split="test"):
    _set_journal_style()
    cells = sorted(glm_results.keys())
    glm_r2 = [glm_results[c][split]["pseudo_r2"] for c in cells]
    xgb_r2 = [xgb_results[c][split]["pseudo_r2"] for c in cells]
    nn_r2 = [nn_results[c][split]["pseudo_r2"] for c in cells]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(cells, glm_r2, label="GLM", marker="o")
    ax.plot(cells, xgb_r2, label="XGBoost", marker="o")
    ax.plot(cells, nn_r2, label="Neural Network", marker="o")

    ax.set_xlabel("Cell ID")
    ax.set_ylabel(f"{split.capitalize()} pseudo-R²")
    ax.set_title("Model performance across cells")
    ax.legend()
    fig.tight_layout()

    plt.close(fig)

    return fig


def plot_training_curves(train_losses=None, val_losses=None, title="Training curves"):
    if train_losses is None:
        return None

    _set_journal_style()
    fig, ax = plt.subplots(figsize=(4, 3))

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")

    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Val loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Poisson NLL")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    plt.close(fig)
    return fig
