import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torchview import draw_graph
from itertools import combinations


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
    model_results_list,
    cell,
    split="test",
):
    """Scatter-plot grid comparing predictions for one cell across many models.

    Parameters
    ----------
    model_results_list : list of tuples
        Each tuple contains ``(results_dict, model_name)`` where
        ``results_dict`` is the dictionary returned by any of the
        :mod:`fit_*` wrappers.  ``model_name`` will be used for titles and
        file labels.
    cell : int
        Identifier of the cell to visualise.
    split : str, optional
        Which data split to plot (``"train"``, ``"val"`` or ``"test"``).

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing a grid of scatter panels, one per model.
    """
    _set_journal_style()

    n_models = len(model_results_list)
    # choose grid shape automatically (square-ish)
    cols = int(np.ceil(np.sqrt(n_models)))
    rows = int(np.ceil(n_models / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3.5 * rows), sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)  # flatten even if single row/col

    for ax, (results, name) in zip(axes, model_results_list):
        # decide which y arrays to use
        if "y_test" in results[cell]:
            y_true = results[cell].get(f"y_{split}", results[cell]["y_test"])
            y_pred = results[cell].get(f"y_pred_{split}", results[cell]["y_pred_test"])
        else:
            y_true = results[cell][f"y_{split}"]
            y_pred = results[cell][f"y_pred_{split}"]

        ax.scatter(y_true, y_pred, alpha=0.4, s=10, edgecolor="none")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_title(f"{name} — Cell {cell}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    # hide any unused axes
    for ax in axes[n_models:]:
        ax.axis("off")

    fig.suptitle(f"Model comparison — Cell {cell} ({split})", y=1.02)
    fig.tight_layout()

    plt.close(fig)
    return fig


def compare_r2_across_cells(
    model_results_list,
    split="test",
):
    """Line plot of pseudo‑R² values across cells for an arbitrary set of models.

    Parameters
    ----------
    model_results_list : list of tuples
        ``(results_dict, model_name)`` pairs for each model to include.
    split : str, optional
        Which split to use when extracting pseudo‑R².

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()

    # assume all dicts share the same cell keys
    first_results = model_results_list[0][0]
    cells = sorted(first_results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    cmap = plt.get_cmap("tab20")
    n_models = len(model_results_list)
    colours = [cmap(i % 20) for i in range(n_models)]

    for (results, name), colour in zip(model_results_list, colours):
        r2 = [results[c][split]["pseudo_r2"] for c in cells]
        ax.plot(cells, r2, label=name, marker="o", color=colour)

    ax.set_xlabel("Cell ID")
    ax.set_ylabel(f"{split.capitalize()} pseudo-R²")
    ax.set_title("Model performance across cells")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    plt.close(fig)
    return fig


def compare_models_pairwise_r2(
    model_results_list,
    model_x_name,
    model_y_name,
    split="test",
    title=None,
):
    """
    Scatter plot comparing pseudo-R² across all cells for two models.

    Parameters
    ----------
    model_results_list : list of tuples
        (results_dict, model_name) pairs.
    model_x_name : str
        Name of the model to use on the x-axis.
    model_y_name : str
        Name of the model to use on the y-axis.
    split : str, optional
        Which split to use ("train", "val", "test").
    title : str, optional
        Plot title. If None, a default is constructed.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()

    # Convert list → dict for easy lookup
    results_dict = {name: res for res, name in model_results_list}

    if model_x_name not in results_dict or model_y_name not in results_dict:
        raise ValueError("Model names must match entries in model_results_list")

    res_x = results_dict[model_x_name]
    res_y = results_dict[model_y_name]

    cells = sorted(res_x.keys())

    r2_x = [res_x[c][split]["pseudo_r2"] for c in cells]
    r2_y = [res_y[c][split]["pseudo_r2"] for c in cells]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    ax.scatter(r2_x, r2_y, alpha=0.7, s=25, edgecolor="none")

    # unity line
    lims = [
        min(min(r2_x), min(r2_y)),
        max(max(r2_x), max(r2_y)),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)

    ax.set_xlabel(f"{model_x_name} pseudo-R²")
    ax.set_ylabel(f"{model_y_name} pseudo-R²")

    if title is None:
        title = f"Pseudo-R² comparison: {model_x_name} vs {model_y_name}"
    ax.set_title(title)

    ax.set_aspect("equal", adjustable="box")
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


def journal_plot_pack(
    model_results_list,
    cells,
    split="test",
    base_dir="resources/results/journal",
):
    """Generate per-cell comparison plots for a set of models and save them.

    This convenience routine iterates over the provided ``cells`` list,
    produces a comparison figure for each cell using
    :func:`compare_models_for_cell`, generates a cross-cell R² plot via
    :func:`compare_r2_across_cells`, and writes all figures to disk.
    :func:`compare_models_pairwise_r2` is also used to create scatter plots comparing
    pseudo-R² values for each pair of models.  The resulting images are saved in a
    subdirectory of ``base_dir`` named after the model comparison (e.g. "NN-MLP vs XGBoost")
    Returns a list of file paths that were created.

    Parameters
    ----------
    model_results_list : list of tuples
        ``(results_dict, model_name)`` pairs, as accepted by
        :func:`compare_models_for_cell` and :func:`compare_r2_across_cells`.
    cells : sequence
        Cell identifiers to plot per-cell comparisons for.
    split : str, optional
        Dataset split name to supply to the comparison functions.
    base_dir : str or Path, optional
        Directory under which the plots are saved.

    Returns
    -------
    list of pathlib.Path
        Paths to all saved image files (per-cell + R² summary).
    """
    from src.train.io import save_plot

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for cell in cells:
        fig = compare_models_for_cell(model_results_list, cell, split=split)
        fname = f"cell_{cell}.png"
        path = save_plot(fig, "journal", fname, base_dir=base_dir)
        saved.append(path)

    # 2. Cross-cell R² summary plot
    fig_r2 = compare_r2_across_cells(model_results_list, split=split)
    path_r2 = save_plot(fig_r2, "journal", "r2_summary.png", base_dir=base_dir)
    saved.append(path_r2)

    # 3. All pairwise pseudo-R² scatter plots
    name_pairs = combinations([name for _, name in model_results_list], 2)

    for nameA, nameB in name_pairs:
        fig = compare_models_pairwise_r2(
            model_results_list,
            model_x_name=nameA,
            model_y_name=nameB,
            split=split,
        )
        fname = f"pairwise_{nameA}_vs_{nameB}.png".replace(" ", "_")
        path = save_plot(fig, "journal", fname, base_dir=base_dir)
        saved.append(path)

    return saved


# ------------------------------------------------------------------
# Neural network architecture visualization
# ------------------------------------------------------------------


def plot_nn_architecture(model, model_name: str, base_dir="resources/results"):
    """
    Visualise the *actual* PyTorch module architecture (layers, shapes, hierarchy)
    using torchview instead of torchviz.
    """
    # infer input dimension
    if hasattr(model, "input_dim"):
        input_dim = model.input_dim
    else:
        for m in model.modules():
            if hasattr(m, "in_features"):
                input_dim = m.in_features
                break
        else:
            raise ValueError("could not determine input dimension")

    # detect extractor input type
    # Try model.input_type
    if hasattr(model, "input_type"):
        input_type = model.input_type
    # Try extractor.input_type
    elif hasattr(model, "extractor") and hasattr(model.extractor, "input_type"):
        input_type = model.extractor.input_type
    # Fallback
    else:
        input_type = "flat"

    if input_type == "flat":
        input_size = (1, input_dim)
    elif input_type == "sequence":
        input_size = (1, 1, input_dim)
    elif input_type == "image":
        H = int(input_dim**0.5)
        input_size = (1, 1, H, H)
    else:
        input_size = (1, input_dim)

    # build architecture graph (forward graph, not autograd)
    graph = draw_graph(
        model,
        input_size=input_size,
        expand_nested=True,  # show submodules
        save_graph=False,
    )

    # save to disk
    out_dir = Path(base_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "architecture"

    graph.visual_graph.render(str(out_path), format="png")
    return out_path.with_suffix(".png")
