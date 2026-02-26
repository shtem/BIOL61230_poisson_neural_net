import numpy as np
import matplotlib.pyplot as plt


def _set_journal_style():
    """Apply consistent publication-style formatting to matplotlib figures.

    This internal helper sets a seaborn-based whitegrid style and adjusts
    rcParams for DPI, font sizes, line widths, and removes top/right spines
    so that plots look clean in articles or presentations.
    """
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
    """Compute a peri-stimulus time histogram (PSTH).

    The input ``y`` contains spike counts arranged sequentially across trials.
    This routine partitions the vector into trials of length ``bins_per_trial``
    and then averages across the trial dimension.  A Gaussian smoother may be
    applied optionally.

    Parameters
    ----------
    y : ndarray, shape (n_time,)
        Spike counts ordered by time bin.
    bins_per_trial : int, optional
        Number of bins per trial (default 25).
    smooth_sigma : float or None, optional
        Standard deviation for a 1d Gaussian filter; if ``None`` no smoothing
        is performed.

    Returns
    -------
    ndarray
        PSTH values with length ``bins_per_trial``.
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
    """Create a comparison plot of two PSTH traces.

    The figure is closed before returning so that calling code may save or
    display it without creating duplicate windows.

    Parameters
    ----------
    psth_true : ndarray
        Ground truth PSTH values.
    psth_pred : ndarray
        Model-generated PSTH values.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the PSTH comparison.
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
    """Scatter plot comparing true versus predicted spike counts.

    A unity line is added for reference.  The journal style is applied to
    ensure consistent aesthetics.

    Parameters
    ----------
    y_true : ndarray
    y_pred : ndarray
    title : str, optional
    xlabel : str, optional
    ylabel : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
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


def compare_models_for_cell(
    glm_results, xgb_results, nn_results, tl_results, cell, split="test"
):
    """Scatter-plot grid comparing model predictions for a single cell.

    Parameters
    ----------
    glm_results, xgb_results, nn_results, tl_results : dict
        Results dictionaries keyed by cell containing predictions and metrics.
    cell : int
        Cell identifier to plot.
    split : str, optional
        Dataset split to use ("train","val","test").

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    model_list = [
        (glm_results, "GLM"),
        (xgb_results, "XGBoost"),
        (nn_results, "Neural Network"),
        (tl_results, "Transfer Learning NN"),
    ]

    for ax, (results, name) in zip(axes, model_list):

        # Transfer learning uses y_test / y_pred_test only
        if name == "Transfer Learning NN":
            y_true = results[cell]["y_test"]
            y_pred = results[cell]["y_pred_test"]
        else:
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


def compare_r2_across_cells(
    glm_results, xgb_results, nn_results, tl_results, split="test"
):
    """Line plot of pseudo‑R² values across cells for each model.

    Parameters
    ----------
    glm_results, xgb_results, nn_results, tl_results : dict
        Results dictionaries keyed by cell containing metrics.
    split : str, optional
        Dataset split to use when extracting pseudo‑R².

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()
    cells = sorted(glm_results.keys())
    glm_r2 = [glm_results[c][split]["pseudo_r2"] for c in cells]
    xgb_r2 = [xgb_results[c][split]["pseudo_r2"] for c in cells]
    nn_r2 = [nn_results[c][split]["pseudo_r2"] for c in cells]
    tl_r2 = [tl_results[c][split]["pseudo_r2"] for c in cells]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(cells, glm_r2, label="GLM", marker="o")
    ax.plot(cells, xgb_r2, label="XGBoost", marker="o")
    ax.plot(cells, nn_r2, label="Neural Network", marker="o")
    ax.plot(cells, tl_r2, label="Transfer Learning NN", marker="o")

    ax.set_xlabel("Cell ID")
    ax.set_ylabel(f"{split.capitalize()} pseudo-R²")
    ax.set_title("Model performance across cells")
    ax.legend()
    fig.tight_layout()

    plt.close(fig)

    return fig


def plot_training_curves(train_losses=None, val_losses=None, title="Training curves"):
    """Plot training (and optionally validation) loss over epochs.

    If ``train_losses`` is ``None`` the function returns immediately.  The plot
    is closed before return so that callers can save the figure without
    displaying it.

    Parameters
    ----------
    train_losses : sequence or None
        Training loss values per epoch.
    val_losses : sequence or None, optional
        Validation loss values per epoch.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure or None
        Figure object when ``train_losses`` is provided, otherwise ``None``.
    """
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
