from src.get_data import split_cell_data
from src.models.evaluate import evaluate_poisson_model


def fit_model_per_cell(
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    model_class,
    model_kwargs=None,
    scaler=None,
    custom_train_fn=None,
):
    results = {}

    for cell in X_train.keys():
        Xtr = X_train[cell]
        ytr = Y_train[cell]
        Xv = X_val[cell] if X_val is not None else None
        yv = Y_val[cell] if Y_val is not None else None
        Xte = X_test[cell]
        yte = Y_test[cell]

        # scaling
        if scaler is not None:
            sc = scaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xv_s = sc.transform(Xv) if Xv is not None else None
            Xte_s = sc.transform(Xte)
        else:
            sc = None
            Xtr_s, Xv_s, Xte_s = Xtr, Xv, Xte

        model = model_class(**(model_kwargs or {}))

        if custom_train_fn is None:
            model.fit(Xtr_s, ytr)
        else:
            model = custom_train_fn(model, Xtr_s, ytr, Xv_s, yv)

        # predictions
        y_pred_train = model.predict(Xtr_s)
        y_pred_val = model.predict(Xv_s) if Xv_s is not None else None
        y_pred_test = model.predict(Xte_s)

        # evaluation
        train_eval = evaluate_poisson_model(ytr, y_pred_train)
        val_eval = evaluate_poisson_model(yv, y_pred_val) if yv is not None else None
        test_eval = evaluate_poisson_model(yte, y_pred_test)

        results[cell] = {
            "model": model,
            "scaler": sc,
            "train": train_eval,
            "val": val_eval,
            "test": test_eval,
            "y_pred_train": y_pred_train,
            "y_pred_val": y_pred_val,
            "y_pred_test": y_pred_test,
            "y_train": ytr,
            "y_val": yv,
            "y_test": yte,
        }

        # optional training curves (for NN etc.)
        if hasattr(model, "train_losses"):
            results[cell]["train_losses"] = model.train_losses
        if hasattr(model, "val_losses"):
            results[cell]["val_losses"] = model.val_losses

    return results


def summarise_model_results(results, model_name="Model"):
    """
    Print a summary of model performance metrics for each cell.

    :param results: Dictionary containing fitted models and performance metrics for each cell
    :param model_name: Name of the model to be displayed in the summary (default "Model")
    """
    print(f"\n===== {model_name} Summary =====")

    for cell, info in results.items():
        print(f"\n--- Cell {cell} ---")

        # Train metrics
        train = info.get("train")
        if train is not None:
            print(f"Train pseudo-R²:       {train['pseudo_r2']:.4f}")
            print(f"Train log-likelihood:  {train['log_likelihood']:.2f}")
            print(f"Train deviance:        {train['deviance']:.2f}")
        else:
            print("Train metrics:         (not available)")

        # Val metrics
        val = info.get("val")
        if val is not None:
            print(f"Val pseudo-R²:         {val['pseudo_r2']:.4f}")
            print(f"Val log-likelihood:    {val['log_likelihood']:.2f}")
            print(f"Val deviance:          {val['deviance']:.2f}")
        else:
            print("Val metrics:           (not available)")

        # Test metrics (always available)
        test = info["test"]
        print(f"Test pseudo-R²:        {test['pseudo_r2']:.4f}")
        print(f"Test log-likelihood:   {test['log_likelihood']:.2f}")
        print(f"Test deviance:         {test['deviance']:.2f}")

    print("\n===== End of Summary =====\n")
