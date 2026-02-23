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


def evaluate_poisson_model(y_true, y_pred):
    """
    Comprehensive evaluation of Poisson model predictions,
    including pseudo-R2, log-likelihood, and deviance.

    :param y_true: Array of true spike counts
    :param y_pred: Array of predicted spike counts
    :param bins_per_trial: Number of bins per trial for PSTH computation
    :param smooth_sigma: Standard deviation for Gaussian smoothing of PSTH

    :return: Dictionary of evaluation metrics and PSTH figure
    """
    # Ensure strictly positive predictions
    y_pred = np.clip(y_pred, 1e-8, None)

    # core metrics
    r2 = pseudo_r2(y_true, y_pred)
    ll = poisson_log_likelihood(y_true, y_pred)
    dev = mean_poisson_deviance(y_true, y_pred)

    return {
        "pseudo_r2": r2,
        "log_likelihood": ll,
        "deviance": dev,
    }
