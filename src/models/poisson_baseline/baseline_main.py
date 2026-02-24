import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor
from src.models.utils import fit_model_per_cell
from src.models.hyperparam_search import grid_search_per_cell


def fit_poisson_glm(
    X,
    Y,
    cell_ids,
    alpha=0.0,
    alpha_grid=None,
    grid_search=False,
    train_frac=0.7,
    val_frac=0.15,
    k_folds=3,
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

    if not grid_search:
        glm_kwargs = {"alpha": alpha, "max_iter": 2000}

        results = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=PoissonRegressor,
            model_kwargs=glm_kwargs,
            train_frac=train_frac,
            val_frac=val_frac,
        )

        return {
            "results": results,
            "best_params": {cell: {"alpha": alpha} for cell in np.unique(cell_ids)},
            "all_scores": None,
        }

    # -------------------------
    # MODE B — PER-CELL GRID SEARCH
    # -------------------------
    if alpha_grid is None:
        alpha_grid = [0.0, 0.01, 0.1, 1.0]

    model_param_grid = {"alpha": alpha_grid}

    gs = grid_search_per_cell(
        X,
        Y,
        cell_ids,
        model_class=PoissonRegressor,
        model_param_grid=model_param_grid,
        trainer_param_grid=None,  # GLM has no trainer params
        k_folds=k_folds,
    )

    best_params = gs["best_params"]

    # Fit final models per cell
    final_results = {}
    for cell in np.unique(cell_ids):
        glm_kwargs = {
            "alpha": best_params[cell]["model_params"]["alpha"],
            "max_iter": 2000,
        }

        final_results[cell] = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=PoissonRegressor,
            model_kwargs=glm_kwargs,
            train_frac=train_frac,
            val_frac=val_frac,
        )[cell]

    return {
        "results": final_results,
        "best_params": best_params,
        "all_scores": gs["all_scores"],
    }


def fit_poisson_xgboost(
    X,
    Y,
    cell_ids,
    param_grid=None,
    grid_search=False,
    train_frac=0.7,
    val_frac=0.15,
    k_folds=3,
    **kwargs,
):
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

    # -------------------------
    # MODE A — GLOBAL PARAMS
    # -------------------------
    if not grid_search:
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

        results = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=XGBRegressor,
            model_kwargs=default_params,
            train_frac=train_frac,
            val_frac=val_frac,
        )

        return {
            "results": results,
            "best_params": {cell: default_params for cell in np.unique(cell_ids)},
            "all_scores": None,
        }

    # -------------------------
    # MODE B — PER-CELL GRID SEARCH
    # -------------------------
    if param_grid is None:
        param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [200, 300],
        }

    model_param_grid = param_grid

    gs = grid_search_per_cell(
        X,
        Y,
        cell_ids,
        model_class=XGBRegressor,
        model_param_grid=model_param_grid,
        trainer_param_grid=None,  # XGB has no trainer params
        k_folds=k_folds,
    )

    best_params = gs["best_params"]

    final_results = {}
    for cell in np.unique(cell_ids):
        params = best_params[cell]["model_params"].copy()
        params.update(
            dict(
                objective="count:poisson",
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
            )
        )

        final_results[cell] = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=XGBRegressor,
            model_kwargs=params,
            train_frac=train_frac,
            val_frac=val_frac,
        )[cell]

    return {
        "results": final_results,
        "best_params": best_params,
        "all_scores": gs["all_scores"],
    }
