import numpy as np
from sklearn.linear_model import PoissonRegressor
from xgboost import XGBRegressor
from src.get_data import split_cell_data
from src.eval import evaluate_poisson_model


def fit_model_per_cell(
    X, Y, cell_ids, model_class, model_kwargs=None, train_frac=0.7, val_frac=0.15
):
    """
    Generic function to fit ANY model per cell using the same pipeline.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the target values (spike counts)
    :param cell_ids: Array of all cell IDs
    :param model_class: The class of the model to be fitted (e.g., PoissonRegressor, XGBRegressor)
    :param model_kwargs: Dictionary of keyword arguments to be passed to the model constructor
    :param train_frac: Fraction of samples to use for training (default 0.7)
    :param val_frac: Fraction of samples to use for validation (default 0.15)

    :return: Dictionary containing fitted models, coefficients, and performance metrics for each cell
    """
    if model_kwargs is None:
        model_kwargs = {}

    results = {}
    splits = split_cell_data(cell_ids, train_frac, val_frac)

    for cell, s in splits.items():
        train_idx = s["train_idx"]
        val_idx = s["val_idx"]
        test_idx = s["test_idx"]

        # prepare data for this cell
        # scikit's PoissonRegressor expects shape (n_samples, n_features) for X and (n_samples,) for y
        # we need to transpose X to get shape (n_time_bins, n_features)
        X_train = X[:, train_idx].T
        y_train = Y[train_idx]

        X_val = X[:, val_idx].T
        y_val = Y[val_idx]

        X_test = X[:, test_idx].T
        y_test = Y[test_idx]

        # instantiate model
        model = model_class(**model_kwargs)

        # fit
        model.fit(X_train, y_train)

        # predict
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # evaluate
        train_eval = evaluate_poisson_model(y_train, y_pred_train)
        val_eval = evaluate_poisson_model(y_val, y_pred_val)
        test_eval = evaluate_poisson_model(y_test, y_pred_test)

        results[cell] = {
            "model": model,
            "train": train_eval,
            "val": val_eval,
            "test": test_eval,
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred_test": y_pred_test,
        }

    return results


def fit_poisson_glm(X, Y, cell_ids, alpha=0.0, train_frac=0.7, val_frac=0.15):
    """
    Fit a Poisson GLM baseline model for each cell.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the target values (spike counts)
    :param cell_ids: Array of all cell IDs
    :param alpha: Regularization strength (default 0.0)
    :param train_frac: Fraction of samples to use for training (default 0.7)
    :param val_frac: Fraction of samples to use for validation (default 0.15)

    :return: Dictionary containing fitted models, coefficients, and performance metrics for each cell
    """
    return fit_model_per_cell(
        X,
        Y,
        cell_ids,
        model_class=PoissonRegressor,
        model_kwargs={"alpha": alpha, "max_iter": 2000},
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


def summarise_model_results(results, model_name="Model"):
    """
    Print a summary of model performance metrics for each cell.

    :param results: Dictionary containing fitted models and performance metrics for each cell
    :param model_name: Name of the model to be displayed in the summary (default "Model")
    """

    print(f"\n===== {model_name} Summary =====")

    for cell, info in results.items():
        train = info["train"]
        val = info["val"]
        test = info["test"]

        print(f"\n--- Cell {cell} ---")
        print(f"Train pseudo-R²:       {train['pseudo_r2']:.4f}")
        print(f"Val pseudo-R²:         {val['pseudo_r2']:.4f}")
        print(f"Test pseudo-R²:        {test['pseudo_r2']:.4f}")

        print(f"Train log-likelihood:  {train['log_likelihood']:.2f}")
        print(f"Val log-likelihood:    {val['log_likelihood']:.2f}")
        print(f"Test log-likelihood:   {test['log_likelihood']:.2f}")

        print(f"Train deviance:        {train['deviance']:.2f}")
        print(f"Val deviance:          {val['deviance']:.2f}")
        print(f"Test deviance:         {test['deviance']:.2f}")

    print("\n===== End of Summary =====\n")
