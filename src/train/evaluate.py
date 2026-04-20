import numpy as np
from sklearn.metrics import mean_poisson_deviance


def pseudo_r2(y_true, y_pred):
    """
    Compute the pseudo-R² for Poisson regression.

    The pseudo-R² statistic is defined as::

        R2_pseudo = 1 - (deviance_model / deviance_null)

    where ``deviance_null`` is the mean Poisson deviance of a null model that
    always predicts the mean of ``y_true``. This measure behaves similarly to
    the classical R² in linear regression but is appropriate when the response
    follows a Poisson distribution.

    Parameters
    ----------
    y_true : array-like
        Observed spike counts (non-negative integers).
    y_pred : array-like
        Predicted firing rates (non-negative floats).

    Returns
    -------
    float
        Pseudo-R² value. Values closer to 1 indicate better fit; the statistic
        can be negative if the model performs worse than the null model.
    """
    # construct null prediction array equal to the mean of the true values
    y_null = np.full_like(y_true, y_true.mean())
    y_pred = np.clip(y_pred, 1e-8, None)

    dev_model = mean_poisson_deviance(y_true, y_pred)
    dev_null = mean_poisson_deviance(y_true, y_null)

    return 1 - dev_model / dev_null


def poisson_log_likelihood(y_true, y_pred):
    """
    Calculate the log-likelihood under a Poisson model.

    The log-likelihood for each observation is::

        logL_i = y_true_i * log(y_pred_i) - y_pred_i

    where ``y_pred`` represents the predicted rate and ``y_true`` the
    observed count.  The total log-likelihood is the sum across samples. This
    quantity is useful for model comparison and can be negative.

    Parameters
    ----------
    y_true : array-like
        Observed counts.
    y_pred : array-like
        Predicted rates; values are clipped to avoid taking ``log(0)``.

    Returns
    -------
    float
        Sum of log-likelihoods over the provided observations.
    """
    # avoid log(0) by forcing a small positive lower bound on predictions
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, None)

    ll = np.sum(y_true * np.log(y_pred) - y_pred)
    return ll


def evaluate_poisson_model(y_true, y_pred):
    """
    Evaluate Poisson regression predictions with standard metrics.

    This wrapper combines the fundamental evaluation utilities defined above
    and returns a dictionary so that results can be stored alongside models or
    displayed in summaries.

    Parameters
    ----------
    y_true : array-like
        Ground-truth spike counts.
    y_pred : array-like
        Model predictions (rates).  Values are clipped to ensure positivity.

    Returns
    -------
    dict
        Contains the keys ``"pseudo_r2"``, ``"log_likelihood"`` and
        ``"deviance"`` with the corresponding numeric results.
    """
    # clip to avoid zero or negative predictions which break evaluation
    y_pred = np.clip(y_pred, 1e-8, None)

    # compute each metric using helper functions
    r2 = pseudo_r2(y_true, y_pred)
    ll = poisson_log_likelihood(y_true, y_pred)
    dev = mean_poisson_deviance(y_true, y_pred)

    return {
        "pseudo_r2": r2,
        "log_likelihood": ll,
        "deviance": dev,
    }
