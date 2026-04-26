import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from torchview import draw_graph
from src.get_data import get_trial_slice


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
            "font.family": "DejaVu Sans",
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
    sort_by=None,
    figsize=None,
):
    """Line plot of pseudo-R² values across cells for multiple models.

    Most useful for identifying which individual cells are consistently well
    or poorly predicted across all models. For a cleaner population-level
    comparison, use plot_r2_comparison_boxplot instead.

    Parameters
    ----------
    model_results_list : list of tuples
        (results_dict, model_name) pairs.
    split : str, optional
        Data split to use. Default "test".
    sort_by : str or None, optional
        If provided, sort cells by the pseudo-R² of this named model
        (ascending) before plotting. This makes cross-model patterns much
        clearer — cells that are hard for one model tend to be hard for all.
        Example: sort_by="GLM" to rank cells by GLM performance.
    figsize : tuple or None, optional
        Override figure size.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()

    first_results = model_results_list[0][0]
    cells = sorted(first_results.keys())

    # optionally sort cells by a reference model's performance
    if sort_by is not None:
        ref = {name: res for res, name in model_results_list}.get(sort_by)
        if ref is not None:
            cells = sorted(cells, key=lambda c: ref[c][split]["pseudo_r2"])

    if figsize is None:
        figsize = (max(10, len(cells) * 0.5), 5)

    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.get_cmap("tab10")
    for i, (results, name) in enumerate(model_results_list):
        r2 = [results[c][split]["pseudo_r2"] for c in cells]
        ax.plot(
            range(len(cells)),
            r2,
            label=name,
            marker="o",
            markersize=4,
            linewidth=1.2,
            color=cmap(i % 10),
            alpha=0.85,
        )

    # chance reference line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0, label="Chance")

    # use cell rank on x-axis (not raw cell ID) so spacing is even
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([str(c) for c in cells], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel(
        f"Cell ID{' (sorted by GLM performance)' if sort_by else ''}", fontsize=11
    )
    ax.set_ylabel(f"{split.capitalize()} pseudo-R²", fontsize=11)
    ax.set_title("Model performance across individual cells", fontsize=12)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

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


"""
New visualisation functions to append to visualisation.py
----------------------------------------------------------
Add these two functions to the end of your existing visualisation.py file.
They follow the same style conventions as the existing code (_set_journal_style,
plt.close(fig) before return, consistent docstrings).

Functions added:
    plot_r2_comparison_boxplot  -- grouped box plot comparing pseudo-R² across models
    plot_r2_histogram           -- histogram of pseudo-R² values (like paper Fig 3E)
    plot_covariate_trial        -- time-series of covariates + spikes for one trial
"""


# ---------------------------------------------------------------------------
# Figure 3 / 4 / 5 helper  ―  model performance comparison box plot
# ---------------------------------------------------------------------------


def plot_r2_comparison_boxplot(
    model_results_list,
    split="test",
    title=None,
    figsize=None,
    show_points=True,
    chance_line=True,
    color_palette=None,
    rotate_labels=True,
):
    """Grouped box plot comparing pseudo-R² distributions across models.

    This is the main figure function for comparing model performance in the
    report. For each model in ``model_results_list`` it draws a box showing
    the interquartile range of pseudo-R² across all cells, with the median
    marked, and (optionally) individual cell values overlaid as jittered
    dots so the reader can see the full distribution without losing
    individual-cell information.

    A horizontal dashed line at y=0 marks chance-level performance (a model
    predicting the mean firing rate). Cells with negative pseudo-R² are
    therefore performing *worse* than the null model, which is an important
    reference point.

    Parameters
    ----------
    model_results_list : list of tuples
        Each tuple is ``(results_dict, model_name)`` exactly as used by
        ``compare_r2_across_cells``. Models are plotted left-to-right in
        list order.
    split : str, optional
        Which data split to use (``"train"``, ``"val"`` or ``"test"``).
        Default ``"test"``.
    title : str or None, optional
        Figure title. If ``None``, a sensible default is used.
    figsize : tuple or None, optional
        Override the default figure size. Default scales with number of models.
    show_points : bool, optional
        If ``True``, overlay individual cell pseudo-R² values as jittered
        scatter points. Default ``True``.
    chance_line : bool, optional
        If ``True``, draw a dashed horizontal line at y = 0 labelled
        "Chance". Default ``True``.
    color_palette : list or None, optional
        List of matplotlib colour strings, one per model. If ``None``, the
        tab10 palette is used automatically.
    rotate_labels : bool, optional
        If ``True``, rotate x-axis labels 30° for readability when model
        names are long. Default ``True``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure ready to save or display.

    Notes
    -----
    The function uses ``matplotlib.axes.Axes.boxplot`` directly rather than
    seaborn so that it stays dependency-light and is easy to adjust.
    """
    _set_journal_style()

    n_models = len(model_results_list)

    # --- collect per-model pseudo-R² arrays ---
    model_names = []
    r2_arrays = []

    for results, name in model_results_list:
        cells = sorted(results.keys())
        r2 = np.array([results[c][split]["pseudo_r2"] for c in cells])
        model_names.append(name)
        r2_arrays.append(r2)

    # --- figure geometry ---
    if figsize is None:
        # scale width with number of models, minimum 6 inches
        width = max(6, n_models * 1.1)
        figsize = (width, 5)

    fig, ax = plt.subplots(figsize=figsize)

    # --- colour palette ---
    if color_palette is None:
        cmap = plt.get_cmap("tab10")
        color_palette = [cmap(i % 10) for i in range(n_models)]

    # --- draw boxes ---
    positions = np.arange(1, n_models + 1)

    bp = ax.boxplot(
        r2_arrays,
        positions=positions,
        widths=0.5,
        patch_artist=True,  # filled boxes
        showfliers=False,  # hide outlier markers; we show all points
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        boxprops=dict(linewidth=1.2),
    )

    # colour each box to match the scatter points below
    for patch, colour in zip(bp["boxes"], color_palette):
        patch.set_facecolor(colour)
        patch.set_alpha(0.45)

    # --- jittered individual cell points ---
    if show_points:
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        for pos, r2, colour in zip(positions, r2_arrays, color_palette):
            # small horizontal jitter so overlapping points separate visually
            jitter = rng.uniform(-0.18, 0.18, size=len(r2))
            ax.scatter(
                pos + jitter,
                r2,
                color=colour,
                alpha=0.65,
                s=18,
                edgecolors="none",
                zorder=3,  # draw on top of box
            )

    # --- chance-level reference line ---
    if chance_line:
        ax.axhline(
            y=0,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            label="Chance (null model)",
            zorder=1,
        )
        ax.legend(fontsize=9, loc="upper left")

    # --- axes labels and formatting ---
    ax.set_xticks(positions)
    ax.set_xticklabels(
        model_names,
        rotation=30 if rotate_labels else 0,
        ha="right" if rotate_labels else "center",
        fontsize=9,
    )

    ax.set_ylabel(f"{split.capitalize()} pseudo-R²", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)

    if title is None:
        title = f"Model comparison — {split} pseudo-R² across cells"
    ax.set_title(title, fontsize=12, pad=10)

    # add a subtle grid on the y-axis only; the box plot already has structure
    ax.yaxis.grid(True, alpha=0.35, linestyle=":")
    ax.set_axisbelow(True)

    # tight y-limits with a small margin so points aren't clipped
    all_r2 = np.concatenate(r2_arrays)
    y_pad = (all_r2.max() - all_r2.min()) * 0.08
    ax.set_ylim(all_r2.min() - y_pad, all_r2.max() + y_pad)

    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Bonus helper  ―  pseudo-R² histogram  (mirrors paper Figure 3E style)
# ---------------------------------------------------------------------------


def plot_r2_histogram(
    model_results_list,
    split="test",
    bins=20,
    title=None,
    figsize=(5, 3.5),
):
    """Overlaid histogram of pseudo-R² values across cells for multiple models.

    This reproduces the style of Figure 3E in Ebrahimi et al. (2025), making
    the comparison to the paper's GLM baseline visually direct. Each model is
    drawn as a semi-transparent histogram so overlaps are visible.

    Parameters
    ----------
    model_results_list : list of tuples
        ``(results_dict, model_name)`` pairs.
    split : str, optional
        Data split to use. Default ``"test"``.
    bins : int, optional
        Number of histogram bins. Default 20.
    title : str or None, optional
        Figure title.
    figsize : tuple, optional
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_journal_style()
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap("tab10")

    for i, (results, name) in enumerate(model_results_list):
        cells = sorted(results.keys())
        r2 = np.array([results[c][split]["pseudo_r2"] for c in cells])
        ax.hist(
            r2,
            bins=bins,
            histtype="step",  # CHANGED: outline only, no fill
            linewidth=2.0,  # thick enough to see clearly
            color=cmap(i % 10),
            label=name,
        )
        # add a subtle fill at very low alpha for easier reading
        ax.hist(
            r2,
            bins=bins,
            histtype="stepfilled",
            alpha=0.12,  # very subtle fill
            color=cmap(i % 10),
        )

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1.0, label="Chance")
    ax.set_xlabel(f"{split.capitalize()} pseudo-R²", fontsize=11)
    ax.set_ylabel("Number of cells", fontsize=11)
    if title is None:
        title = f"Distribution of pseudo-R² across cells ({split})"
    ax.set_title(title, fontsize=12)

    # place legend outside the plot area to avoid overlap with bars
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 1 helper  ―  covariate time-series + spike counts for one trial
# ---------------------------------------------------------------------------

# Canonical covariate names matching the real data X matrix.
# Index 0-4 are the core motion/tilt covariates; 5-8 are positional;
# 9-13 are the visual response covariates (visual stimulation dataset).
COVARIATE_NAMES_REAL = [
    "Tilt angle",  # 0
    "aPM (ang. passive)",  # 1
    "lPM (lin. passive)",  # 2
    "aAM (ang. active)",  # 3
    "lAM (lin. active)",  # 4
    "Sin(Ori)",  # 5
    "Cos(Ori)",  # 6
    "DistRot",  # 7
    "TiltL",  # 8
    "TiltR",  # 9
    "ON",  # 10
    "ON_fast",  # 11
    "OFF_fast",  # 12
    "ON_slow",  # 13
]

COVARIATE_NAMES_SIMULATED = [
    "Tilt angle",  # 0
    "aPM (ang. passive)",  # 1
    "aAM (ang. active)",  # 2
    "lPM (lin. passive)",  # 3
    "lAM (lin. active)",  # 4
]


def plot_covariate_trial(
    X,
    Y,
    cell_ids,
    cell_idx,
    trial_idx,
    covariate_names=None,
    trials_per_cell=400,
    bin_duration_ms=50,
    highlight_motion=True,
    figsize=None,
    title=None,
    show_only_indices=None,
    show_legends=False,
):
    """Publication-quality plot of covariates and spike counts for one trial.

    Plots each covariate as a separate time-series panel, with the cell's
    spike counts shown as a stem plot in the bottom panel. This is the key
    data-overview figure that establishes what the inputs and outputs look
    like for a reader unfamiliar with the dataset.

    The time axis is converted from bin indices to milliseconds using
    ``bin_duration_ms`` so the x-axis is interpretable without knowledge
    of the binning scheme.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_time)
        Full feature matrix for all cells.
    Y : ndarray, shape (n_time,)
        Full spike count vector for all cells.
    cell_ids : ndarray, shape (n_time,)
        Cell identifier for each time bin.
    cell_idx : int
        Which cell to visualise (must be in ``np.unique(cell_ids)``).
    trial_idx : int
        Zero-based trial index within the cell (0 to trials_per_cell - 1).
    covariate_names : list of str or None, optional
        Names for each row of X. If ``None``, generic labels ``Cov 0``,
        ``Cov 1`` ... are used. Use ``COVARIATE_NAMES_REAL`` or
        ``COVARIATE_NAMES_SIMULATED`` as a convenience.
    trials_per_cell : int, optional
        Number of trials per cell. Used to compute bin boundaries.
        Default 400 (real data). Use 120 for simulated data.
    bin_duration_ms : int, optional
        Duration of each time bin in milliseconds. Default 50 ms.
    highlight_motion : bool, optional
        If ``True``, shade the motion covariate panels (indices 0-4) with a
        very light background to visually group them. Default ``True``.
    figsize : tuple or None, optional
        Override figure size. Default scales with number of features.
    title : str or None, optional
        Suptitle for the figure. If ``None``, "Cell {cell_idx}, Trial
        {trial_idx}" is used.
    show_legends : bool, optional
        If ``True``, add legends to each panel. Default ``False`` for a cleaner

    Returns
    -------
    matplotlib.figure.Figure
        Figure with n_features + 1 stacked panels sharing the x-axis.

    """
    _set_journal_style()

    trial_slice = get_trial_slice(cell_idx, trial_idx, cell_ids, trials_per_cell)

    X_trial = X[:, trial_slice]  # (n_features, bins_per_trial)
    Y_trial = Y[trial_slice]  # (bins_per_trial,)

    n_features = X_trial.shape[0]
    n_bins = X_trial.shape[1]

    # convert bin indices to milliseconds for the time axis
    time_ms = np.arange(n_bins) * bin_duration_ms

    # --- covariate names ---
    if covariate_names is None:
        covariate_names = [f"Cov {i}" for i in range(n_features)]
    # safety: truncate or pad if lengths differ
    covariate_names = list(covariate_names)[:n_features]
    while len(covariate_names) < n_features:
        covariate_names.append(f"Cov {len(covariate_names)}")

    # --- filter to requested covariates only ---
    # If show_only_indices is None, show all. Otherwise restrict to the
    # specified rows. This is important for real data where positional and
    # visual covariates may be constant within a single trial.
    if show_only_indices is not None:
        show_only_indices = list(show_only_indices)
        X_trial = X_trial[show_only_indices, :]
        covariate_names = [covariate_names[i] for i in show_only_indices]

    n_features = X_trial.shape[0]  # recompute after filtering

    # --- figure layout ---
    n_rows = n_features + 1  # one panel per covariate + spike panel
    if figsize is None:
        figsize = (8, 1.6 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.08},  # tight vertical spacing
    )

    # --- colour scheme ---
    # Motion covariates (0-4) in a warm amber; positional (5-9) in teal;
    # visual (10-13) in indigo. This matches the three logical groups.
    def _cov_colour(idx):
        if idx <= 4:
            return "#E07B39"  # amber — motion covariates
        elif idx <= 9:
            return "#2A9D8F"  # teal  — positional covariates
        else:
            return "#5E4FA2"  # indigo — visual covariates

    # --- covariate panels ---
    for i in range(n_features):
        ax = axes[i]
        colour = _cov_colour(i)

        ax.plot(time_ms, X_trial[i], color=colour, linewidth=1.2)

        # optional shading to group motion covariates
        if highlight_motion and i <= 4:
            ax.set_facecolor("#FFF8F2")  # very light amber wash

        # y-axis label: abbreviated name on the right to save horizontal space
        ax.set_ylabel(
            covariate_names[i],
            fontsize=8,
            rotation=90,  # vertical
            labelpad=9,
            ha="center",
            va="center",
        )

        # minimal y-axis: just show the zero line for reference
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.tick_params(axis="y", labelsize=7, rotation=45)
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # --- spike count panel (bottom) ---
    ax_spk = axes[-1]
    # use a stem plot: vertical lines from zero up to each spike count
    # markerline, stemlines, baseline
    markerline, stemlines, baseline = ax_spk.stem(
        time_ms, Y_trial, linefmt="C3-", markerfmt="C3.", basefmt="gray"
    )
    plt.setp(stemlines, linewidth=0.9)
    plt.setp(markerline, markersize=3)

    ax_spk.set_ylabel(
        "Spikes", fontsize=8, rotation=90, labelpad=10, ha="center", va="center"
    )
    ax_spk.set_xlabel(f"Time (ms)", fontsize=10, labelpad=10, ha="center")
    ax_spk.tick_params(axis="y", labelsize=7)
    ax_spk.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))

    # --- title and cosmetics ---
    if title is None:
        title = f"Cell {cell_idx}, Trial {trial_idx}"

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#E07B39", alpha=0.8, label="Motion (aPM, lPM, aAM, lAM, Tilt)"
        ),
        Patch(
            facecolor="#2A9D8F", alpha=0.8, label="Positional (Ori, DistRot, TiltL/R)"
        ),
        Patch(facecolor="#5E4FA2", alpha=0.8, label="Visual (ON, ON/OFF fast/slow)"),
    ]

    # Title first, then legend horizontally below it
    fig.suptitle(title, fontsize=11, y=0.975, x=0.57)

    if show_legends:
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.965),  # centred horizontally, just below title
            ncol=3,  # horizontal — one row, three columns
            fontsize=7.5,
            framealpha=0.9,
            handlelength=1.2,
            columnspacing=1.0,
            borderaxespad=0,
        )

    # More space on left for y-axis labels, top space for title + legend
    fig.subplots_adjust(
        left=0.15,  # space for y-axis labels
        right=0.97,  # small right margin
        top=0.93,  # space for title
        bottom=0.07,  # space for x-axis label
        hspace=0.08,  # tight vertical spacing between panels
    )
    plt.close(fig)
    return fig


def journal_plot_pack(
    model_results_list,
    cells,
    split="test",
    base_dir="resources/results/journal",
    pairwise_pairs=None,
    example_cells=None,
):
    """Generate a curated set of report-ready figures and save them to disk.

    Rather than generating every possible pairwise comparison, this function
    produces a focused set of figures that map directly onto the planned report
    figures. The caller controls which pairwise comparisons and which example
    cells to include, keeping the output manageable.

    Parameters
    ----------
    model_results_list : list of tuples
        (results_dict, model_name) pairs for all models to include.
    cells : sequence
        All cell IDs present in the results (used for box plot and histogram).
    split : str, optional
        Data split to evaluate on. Default "test".
    base_dir : str or Path, optional
        Root directory for saved figures.
    pairwise_pairs : list of (str, str) or None, optional
        Specific model name pairs to generate pairwise scatter plots for.
        If None, defaults to comparing the first model (typically GLM)
        against every other model — much more focused than all pairs.
    example_cells : list of int or None, optional
        Cell IDs to generate per-cell scatter grids for. If None, uses
        only the first cell. Keep this to 2-3 cells maximum.

    Returns
    -------
    list of pathlib.Path
        Paths to all saved image files.
    """
    from src.train.io import save_plot

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # 1. Box plot: full model comparison (the main performance figure)
    fig_box = plot_r2_comparison_boxplot(model_results_list, split=split)
    saved.append(save_plot(fig_box, "journal", "r2_boxplot.png", base_dir=base_dir))

    # 2. Histogram: pseudo-R² distribution (mirrors paper Fig 3E)
    fig_hist = plot_r2_histogram(model_results_list, split=split)
    saved.append(save_plot(fig_hist, "journal", "r2_histogram.png", base_dir=base_dir))

    # 3. Line plot: kept for exploratory use, not the report figure
    fig_line = compare_r2_across_cells(model_results_list, split=split)
    saved.append(save_plot(fig_line, "journal", "r2_line.png", base_dir=base_dir))

    # 4. Pairwise scatter: only the specified pairs (or GLM vs all others)
    if pairwise_pairs is None:
        # default: compare the first model against every other
        baseline_name = model_results_list[0][1]
        pairwise_pairs = [(baseline_name, name) for _, name in model_results_list[1:]]

    for name_x, name_y in pairwise_pairs:
        fig_pw = compare_models_pairwise_r2(
            model_results_list,
            model_x_name=name_x,
            model_y_name=name_y,
            split=split,
        )
        fname = f"pairwise_{name_x}_vs_{name_y}.png".replace(" ", "_")
        saved.append(save_plot(fig_pw, "journal", fname, base_dir=base_dir))

    # 5. Per-cell scatter grids: only for specified example cells
    if example_cells is None:
        example_cells = [cells[0]]  # just one cell by default

    for cell in example_cells:
        fig_cell = compare_models_for_cell(model_results_list, cell, split=split)
        saved.append(
            save_plot(fig_cell, "journal", f"cell_{cell}.png", base_dir=base_dir)
        )

    return saved


# ------------------------------------------------------------------
# Neural network architecture visualisation
# ------------------------------------------------------------------


def plot_nn_architecture(model, model_name: str, base_dir="resources/results"):
    """
    Generate and save a visual diagram of a PyTorch model's architecture using
    ``torchview``. This utility traces the model's forward graph, expands nested
    submodules, and renders a clean hierarchy of layers and tensor shapes.

    The function attempts to infer the appropriate input shape by inspecting
    ``model.input_type`` or, if present, the ``input_type`` attribute of a
    contained extractor. For flat models, the input is treated as a single
    feature vector; for sequence models, a (batch, seq_len, features) tensor is
    constructed; and for image models, a square spatial layout is assumed.

    The resulting architecture diagram is written to a subdirectory named after
    ``model_name`` inside ``base_dir``. The file is saved as ``architecture.png``.

    Parameters
    ----------
    model : torch.nn.Module
        The model to visualise. May contain nested extractors or shared modules.
    model_name : str
        Name used to create the output directory and filename.
    base_dir : str or Path, optional
        Root directory under which the architecture image is saved.

    Returns
    -------
    pathlib.Path
        Path to the saved ``architecture.png`` file.
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
