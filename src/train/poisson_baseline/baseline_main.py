import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import PoissonRegressor
from src.train.training import fit_model_per_cell
from src.train.hyperparam_search import grid_search_per_cell
from src.get_data import prepare_cellwise_datasets, flatten_cellwise_data


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
    verbose=False,
):
    """Baseline per-cell Poisson generalised linear model.

    Two operation modes are supported:

    * ``grid_search=False``: fit a single GLM per cell using supplied ``alpha``.
    * ``grid_search=True``: perform per-cell k-fold CV over a set of ``alpha``
      values and refit the best model.

    Parameters
    ----------
    X : ndarray
        Feature matrix, columns = time bins.
    Y : ndarray
        Response vector of spike counts.
    cell_ids : ndarray
        Cell identifiers for each time bin.
    alpha : float
        Regularisation strength for PoissonRegressor when not tuning.
    alpha_grid : list or None
        List of candidate alphas used during grid search.
    grid_search : bool
        Whether to tune ``alpha`` per cell.
    train_frac, val_frac : float
        Train/validation split proportions.
    k_folds : int
        Number of folds for cross-validation during grid search.
    verbose : bool, optional
        If True, print progress information during splitting and grid search.

    Returns
    -------
    dict
        ``{"results": ..., "best_params": ..., "all_scores": ...}``.
    """

    # split data into train/val/test sets per cell; validation only required if
    # performing grid search.
    Xtr, Ytr, Xv, Yv, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        use_val=grid_search,  # only use val if grid search is on
    )
    if verbose:
        print(
            f"GLM dataset splits: train {train_frac}, val {val_frac}, use_val {grid_search}"
        )
    # flatten structure for search utilities
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

    # -------------------------
    # MODE A — fixed alpha for all cells
    # -------------------------
    if not grid_search:
        glm_kwargs = {"alpha": alpha, "max_iter": 2000}

        results = fit_model_per_cell(
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=PoissonRegressor,
            model_kwargs=glm_kwargs,
        )

        return {
            "results": results,
            "best_params": {cell: {"alpha": alpha} for cell in np.unique(cell_ids)},
            "all_scores": None,
        }

    # -------------------------
    # MODE B — PER-CELL GRID SEARCH
    # -------------------------
    # prepare alpha candidates if none provided
    if alpha_grid is None:
        alpha_grid = [0.0, 0.01, 0.1, 1.0]

    model_param_grid = {"alpha": alpha_grid}

    # perform grid search separately for each cell
    if verbose:
        print("Starting GLM per-cell grid search", model_param_grid)
    gs = grid_search_per_cell(
        Xtr_flat,
        Ytr_flat,
        cell_ids_tr_flat,
        model_class=PoissonRegressor,
        model_param_grid=model_param_grid,
        trainer_param_grid=None,  # GLM has no trainer params
        k_folds=k_folds,
        verbose=verbose,
    )

    best_params = gs["best_params"]

    # Fit final models per cell using selected alpha values
    final_results = {}
    for cell in np.unique(cell_ids):
        glm_kwargs = {
            "alpha": best_params[cell]["model_params"]["alpha"],
            "max_iter": 2000,
        }

        final_results[cell] = fit_model_per_cell(
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=PoissonRegressor,
            model_kwargs=glm_kwargs,
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
    verbose=False,
    **kwargs,
):
    """Baseline per-cell XGBoost with Poisson objective.

    Works similarly to ``fit_poisson_glm`` but uses an XGBRegressor.  When
    ``grid_search`` is enabled it performs per-cell cross-validated tuning on
    the hyperparameters supplied via ``param_grid``.

    Parameters
    ----------
    X, Y, cell_ids : array-like
        Data matrices and identifiers as in ``fit_poisson_glm``.
    param_grid : dict or None
        Hyperparameter grid for ``XGBRegressor``.
    grid_search : bool
        If True, runs tuning; otherwise uses default/global params.
    verbose : bool, optional
        If True, print progress of splits/grid search.
    kwargs :
        Extra parameters forwarded to ``XGBRegressor`` constructor when not
        tuning.

    Returns
    -------
    dict
        Same structure as ``fit_poisson_glm``.
    """

    # prepare datasets as before
    Xtr, Ytr, Xv, Yv, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        use_val=grid_search,
    )
    if verbose:
        print(
            f"XGBoost dataset splits: train {train_frac}, val {val_frac}, use_val {grid_search}"
        )
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

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
        # allow caller to override defaults
        default_params.update(kwargs)

        results = fit_model_per_cell(
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=XGBRegressor,
            model_kwargs=default_params,
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

    # tune parameters for each cell independently
    if verbose:
        print("Starting XGBoost per-cell grid search", model_param_grid)
    gs = grid_search_per_cell(
        Xtr_flat,
        Ytr_flat,
        cell_ids_tr_flat,
        model_class=XGBRegressor,
        model_param_grid=model_param_grid,
        trainer_param_grid=None,  # XGB has no trainer params
        k_folds=k_folds,
        verbose=verbose,
    )

    best_params = gs["best_params"]

    # re-fit final models using chosen params and fixed bookkeeping settings
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
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=XGBRegressor,
            model_kwargs=params,
        )[cell]

    return {
        "results": final_results,
        "best_params": best_params,
        "all_scores": gs["all_scores"],
    }
