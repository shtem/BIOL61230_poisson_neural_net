from sklearn.metrics import mean_poisson_deviance
import matplotlib.pyplot as plt
import numpy as np


def pseudo_r2(y_true, y_pred):
    """
    Calculate pseudo-R2 for Poisson regression model predictions.
    R2_pseudo = 1 - (deviance_model / deviance_null)
    where deviance_null is the deviance of a null model that
    predicts the mean firing rate

    :param y_true: Array of true spike counts
    :param y_pred: Array of predicted spike counts

    :return: Pseudo-R2 value (between 0 and 1, higher is better)
    """
    # null model predicts mean firing rate
    y_null = np.full_like(y_true, y_true.mean())

    dev_model = mean_poisson_deviance(y_true, y_pred)
    dev_null = mean_poisson_deviance(y_true, y_null)

    return 1 - dev_model / dev_null


def poisson_log_likelihood(y_true, y_pred):
    """
    Compute the log-likelihood for a Poisson distribution.
    logL = sum(y_true * log(y_pred) - y_pred)
    where y_pred is the predicted firing rate and y_true
    is the observed spike count.

    :param y_true: Array of true spike counts
    :param y_pred: Array of predicted spike counts

    :return: Log-likelihood value (higher is better)
    """
    # avoid log(0)
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, None)

    ll = np.sum(y_true * np.log(y_pred) - y_pred)
    return ll


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


def evaluate_poisson_model(y_true, y_pred, bins_per_trial=25, smooth_sigma=1.0):
    """
    Comprehensive evaluation of Poisson model predictions,
    including pseudo-R2, log-likelihood, deviance, and PSTH comparison.

    :param y_true: Array of true spike counts
    :param y_pred: Array of predicted spike counts
    :param bins_per_trial: Number of bins per trial for PSTH computation
    :param smooth_sigma: Standard deviation for Gaussian smoothing of PSTH

    :return: Dictionary of evaluation metrics and PSTH figure
    """
    # core metrics
    r2 = pseudo_r2(y_true, y_pred)
    ll = poisson_log_likelihood(y_true, y_pred)
    dev = mean_poisson_deviance(y_true, y_pred)

    # PSTHs
    psth_true = compute_psth(
        y_true, bins_per_trial=bins_per_trial, smooth_sigma=smooth_sigma
    )
    psth_pred = compute_psth(
        y_pred, bins_per_trial=bins_per_trial, smooth_sigma=smooth_sigma
    )

    # PSTH figure
    fig = plot_psth(psth_true, psth_pred, title="PSTH Comparison")

    return {
        "pseudo_r2": r2,
        "log_likelihood": ll,
        "deviance": dev,
        "psth_true": psth_true,
        "psth_pred": psth_pred,
        "psth_fig": fig,
    }
