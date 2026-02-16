from src.get_data import split_cell_data
from src.eval import evaluate_poisson_model


def fit_model_per_cell(
    X,
    Y,
    cell_ids,
    model_class,
    model_kwargs=None,
    train_frac=0.7,
    val_frac=0.15,
    scaler=None,
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

        # Optional Scaling
        if scaler is not None:
            scaler_instance = scaler()
            X_train_s = scaler_instance.fit_transform(X_train)
            X_val_s = scaler_instance.transform(X_val)
            X_test_s = scaler_instance.transform(X_test)
        else:
            scaler_instance = None
            X_train_s, X_val_s, X_test_s = X_train, X_val, X_test

        # Instantiate model
        model = model_class(**model_kwargs)

        # Fit model
        model.fit(X_train_s, y_train)

        # Predict
        y_pred_train = model.predict(X_train_s)
        y_pred_val = model.predict(X_val_s)
        y_pred_test = model.predict(X_test_s)

        # Evaluate
        train_eval = evaluate_poisson_model(y_train, y_pred_train)
        val_eval = evaluate_poisson_model(y_val, y_pred_val)
        test_eval = evaluate_poisson_model(y_test, y_pred_test)

        results[cell] = {
            "model": model,
            "scaler": scaler_instance,
            # evaluation
            "train": train_eval,
            "val": val_eval,
            "test": test_eval,
            # predictions
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred_test": y_pred_test,
            # true values
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            # indices
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

    return results


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
