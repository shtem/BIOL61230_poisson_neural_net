import matplotlib.pyplot as plt


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
