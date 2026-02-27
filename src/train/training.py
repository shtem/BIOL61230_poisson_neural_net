from src.train.evaluate import evaluate_poisson_model


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
    """Fit separate models for each cell and evaluate their performance.

    Convenience routine used by higher‑level wrappers to train one estimator
    per cell.  For each cell id in ``X_train`` the supplied ``model_class``
    is instantiated (with ``model_kwargs``), optionally wrapped with a
    scaler, and fitted using either its own ``fit`` method or a ``custom_train_fn``.
    Predictions are generated on train/validation/test splits and metrics are
    computed with :func:`src.train.evaluate.evaluate_poisson_model`.

    Parameters
    ----------
    X_train : dict
        Mapping from cell id to training feature arrays.
    Y_train : dict
        Mapping from cell id to training target vectors.
    X_val : dict or None
        Validation feature arrays keyed by cell, or ``None`` if not used.
    Y_val : dict or None
        Validation target vectors, or ``None``.
    X_test : dict
        Test feature arrays keyed by cell.
    Y_test : dict
        Test target vectors keyed by cell.
    model_class : type
        Class of model to instantiate for each cell.  Must implement
        ``fit`` and ``predict``.
    model_kwargs : dict, optional
        Keyword arguments forwarded to the model constructor.
    scaler : callable or None, optional
        Factory that returns a scaler instance (e.g. :class:`sklearn.preprocessing.StandardScaler`).
        If provided the scaler is fitted on ``X_train`` and applied to all
        splits; the fitted scaler is stored in the results for each cell.
    custom_train_fn : callable or None, optional
        If supplied, called instead of ``model.fit``.  The function should
        have signature ``(model, Xtr, ytr, Xval, yval)`` and return a trained
        model instance.

    Returns
    -------
    dict
        Dictionary keyed by cell id.  Each value is itself a dictionary
        containing:

        * ``model`` – the fitted estimator
        * ``scaler`` – the fitted scaler (or ``None``)
        * ``train``/``val``/``test`` – evaluation metrics dictionaries
        * ``y_pred_*`` / ``y_*`` – predictions and true responses
        * optionally ``train_losses``/``val_losses`` if present on the model
    """
    results = {}

    for cell in X_train.keys():
        Xtr = X_train[cell]
        ytr = Y_train[cell]
        Xv = X_val[cell] if X_val is not None else None
        yv = Y_val[cell] if Y_val is not None else None
        Xte = X_test[cell]
        yte = Y_test[cell]

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

        y_pred_train = model.predict(Xtr_s)
        y_pred_val = model.predict(Xv_s) if Xv_s is not None else None
        y_pred_test = model.predict(Xte_s)

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

        if hasattr(model, "train_losses"):
            results[cell]["train_losses"] = model.train_losses
        if hasattr(model, "val_losses"):
            results[cell]["val_losses"] = model.val_losses

    return results


def summarise_model_results(results, model_name="Model"):
    """Print a human‑readable summary of per‑cell evaluation metrics.

    Parameters
    ----------
    results : dict
        Output from :func:`fit_model_per_cell` (or any function returning the
        same structure).  Keys are cell ids and values are dictionaries
        containing metric information under the keys ``"train"``,
        ``"val"``, and ``"test"``.
    model_name : str, optional
        Label to display in the printed headings (default ``"Model"``).

    Returns
    -------
    None
        This function only prints to standard output; it does not return a
        value.
    """
    print(f"\n===== {model_name} Summary =====")

    for cell, info in results.items():
        print(f"\n--- Cell {cell} ---")

        train = info.get("train")
        if train is not None:
            print(f"Train pseudo-R²:       {train['pseudo_r2']:.4f}")
            print(f"Train log-likelihood:  {train['log_likelihood']:.2f}")
            print(f"Train deviance:        {train['deviance']:.2f}")
        else:
            print("Train metrics:         (not available)")

        val = info.get("val")
        if val is not None:
            print(f"Val pseudo-R²:         {val['pseudo_r2']:.4f}")
            print(f"Val log-likelihood:    {val['log_likelihood']:.2f}")
            print(f"Val deviance:          {val['deviance']:.2f}")
        else:
            print("Val metrics:           (not available)")

        test = info.get("test")
        if test is None:
            print("Warning: test metrics missing for this cell")
        else:
            print(f"Test pseudo-R²:        {test['pseudo_r2']:.4f}")
            print(f"Test log-likelihood:   {test['log_likelihood']:.2f}")
            print(f"Test deviance:         {test['deviance']:.2f}")

    print("\n===== End of Summary =====\n")
