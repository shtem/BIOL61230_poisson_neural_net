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
    """Fit individual models for each cell and evaluate their performance.

    A common pattern in neural and Poisson modelling is to treat each cell
    (e.g. a neuron) separately. This helper loops over the keys in the
    training dictionaries, fits a fresh model per cell, optionally scales the
    features, and computes predictions and evaluation metrics on training,
    validation and test splits.

    Parameters
    ----------
    X_train : dict
        Mapping from cell identifier to training feature array.
    Y_train : dict
        Corresponding training target counts per cell.
    X_val : dict or None
        Validation features (same keys as ``X_train``) or ``None`` if not
        using a validation set.
    Y_val : dict or None
        Validation targets, required if ``X_val`` is provided.
    X_test : dict
        Test features keyed by cell.
    Y_test : dict
        Test targets keyed by cell.
    model_class : class
        A class or callable returning an unfitted estimator instance when
        invoked with keyword arguments from ``model_kwargs``.
    model_kwargs : dict, optional
        Arguments forwarded to ``model_class`` constructor. Defaults to empty
        dict.
    scaler : callable or None, optional
        If provided, should be a scaler factory (e.g. ``StandardScaler``).
        A fresh scaler is fitted on each cell's training set and reused to
        transform validation/test sets. If ``None`` no scaling is applied.
    custom_train_fn : callable or None, optional
        If given, this function is used in place of ``model.fit``. Its
        signature should be ``custom_train_fn(model, X_tr, y_tr, X_val,
        y_val)`` and it must return the trained model instance.

    Returns
    -------
    dict
        Results indexed by cell. Each entry is itself a dictionary containing
        the fitted ``model``, ``scaler`` (or ``None``), raw predictions on each
        split, evaluation dictionaries under ``train``, ``val`` and ``test``
        and the original targets. If the model produced ``train_losses`` or
        ``val_losses`` attributes they are also included.
    """
    results = {}

    # loop through each cell's data and handle scaling, fitting, predicting
    for cell in X_train.keys():
        Xtr = X_train[cell]
        ytr = Y_train[cell]
        Xv = X_val[cell] if X_val is not None else None
        yv = Y_val[cell] if Y_val is not None else None
        Xte = X_test[cell]
        yte = Y_test[cell]

        # apply scaling per cell if a scaler factory was provided
        if scaler is not None:
            sc = scaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xv_s = sc.transform(Xv) if Xv is not None else None
            Xte_s = sc.transform(Xte)
        else:
            sc = None
            Xtr_s, Xv_s, Xte_s = Xtr, Xv, Xte

        # instantiate and train the model
        model = model_class(**(model_kwargs or {}))

        if custom_train_fn is None:
            model.fit(Xtr_s, ytr)
        else:
            # allow caller to override the training procedure (e.g. for NNs)
            model = custom_train_fn(model, Xtr_s, ytr, Xv_s, yv)

        # obtain predictions on all splits
        y_pred_train = model.predict(Xtr_s)
        y_pred_val = model.predict(Xv_s) if Xv_s is not None else None
        y_pred_test = model.predict(Xte_s)

        # compute evaluation metrics using helper from evaluate.py
        train_eval = evaluate_poisson_model(ytr, y_pred_train)
        val_eval = evaluate_poisson_model(yv, y_pred_val) if yv is not None else None
        test_eval = evaluate_poisson_model(yte, y_pred_test)

        # store everything in results dictionary for later inspection
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

        # some models (neural nets) record loss curves during training
        if hasattr(model, "train_losses"):
            results[cell]["train_losses"] = model.train_losses
        if hasattr(model, "val_losses"):
            results[cell]["val_losses"] = model.val_losses

    return results


def summarise_model_results(results, model_name="Model"):
    """Display a human-readable performance summary for each cell.

    The ``results`` dictionary is expected to be in the same format returned by
    :func:`fit_model_per_cell`: each key is a cell identifier and each value is
    itself a dictionary containing evaluation metric dictionaries under the
    keys ``"train"``, ``"val"`` and ``"test"``. This function iterates
    through the cells and prints pseudo‑R², log‑likelihood and deviance for
    each available split. Missing training or validation metrics are handled
    gracefully.

    Parameters
    ----------
    results : dict
        Mapping of cell identifiers to metric dictionaries. Usually the output
        of :func:`fit_model_per_cell` or similar.
    model_name : str, optional
        Label used in the heading of the summary output (default ``"Model"``).

    Notes
    -----
    The function prints directly to standard output and does not return any
    value. It is intended for quick inspection during exploratory analyses or
    debugging.
    """
    print(f"\n===== {model_name} Summary =====")

    for cell, info in results.items():
        print(f"\n--- Cell {cell} ---")

        # retrieve training metrics if they were computed
        train = info.get("train")
        if train is not None:
            print(f"Train pseudo-R²:       {train['pseudo_r2']:.4f}")
            print(f"Train log-likelihood:  {train['log_likelihood']:.2f}")
            print(f"Train deviance:        {train['deviance']:.2f}")
        else:
            print("Train metrics:         (not available)")

        # retrieve validation metrics if they exist
        val = info.get("val")
        if val is not None:
            print(f"Val pseudo-R²:         {val['pseudo_r2']:.4f}")
            print(f"Val log-likelihood:    {val['log_likelihood']:.2f}")
            print(f"Val deviance:          {val['deviance']:.2f}")
        else:
            print("Val metrics:           (not available)")

        # test metrics are expected to always be present; raise if not
        test = info.get("test")
        if test is None:
            print("Warning: test metrics missing for this cell")
        else:
            print(f"Test pseudo-R²:        {test['pseudo_r2']:.4f}")
            print(f"Test log-likelihood:   {test['log_likelihood']:.2f}")
            print(f"Test deviance:         {test['deviance']:.2f}")

    print("\n===== End of Summary =====\n")
