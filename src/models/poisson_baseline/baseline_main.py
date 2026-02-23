import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor
from src.models.utils import fit_model_per_cell


def fit_poisson_glm(
    X,
    Y,
    cell_ids,
    alpha=0.0,
    train_frac=0.7,
    val_frac=0.15,
):
    """
    Fit a Poisson GLM baseline model for each cell.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the target values (spike counts)
    :param cell_ids: Array of all cell IDs
    :param train_frac: Fraction of samples to use for training (default 0.7)
    :param val_frac: Fraction of samples to use for validation (default 0.15)
    :param alpha: Regularization strength (default 0.0)

    :return: Dictionary containing fitted models, coefficients, and performance metrics for each cell
    """
    glm_kwargs = {
        "alpha": alpha,
        "max_iter": 2000,
    }

    return fit_model_per_cell(
        X,
        Y,
        cell_ids,
        model_class=PoissonRegressor,
        model_kwargs=glm_kwargs,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def fit_poisson_xgboost(X, Y, cell_ids, train_frac=0.7, val_frac=0.15, **kwargs):
    """
    Fit a Poisson XGBoost baseline model for each cell.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the target values (spike counts)
    :param cell_ids: Array of all cell IDs
    :param train_frac: Fraction of samples to use for training (default 0.7)
    :param val_frac: Fraction of samples to use for validation (default 0.15)
    :param kwargs: Additional keyword arguments to be passed to the XGBRegressor constructor

    :return: Dictionary containing fitted models, coefficients, and performance metrics for each cell
    """
    default_params = dict(
        objective="count:poisson",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
    )
    default_params.update(kwargs)

    return fit_model_per_cell(
        X,
        Y,
        cell_ids,
        model_class=XGBRegressor,
        model_kwargs=default_params,
        train_frac=train_frac,
        val_frac=val_frac,
    )
