import torch
import optuna
import itertools
import numpy as np
from sklearn.model_selection import KFold
from src.train.utils import _to_tensor
from src.train.evaluate import pseudo_r2
from src.get_data import prepare_cellwise_datasets
from src.train.poisson_nn.nn_training import TransferLearningTrainer

# Add device handling at top
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _expand_param_grid(param_grid):
    """
    Generate all parameter combinations from a grid specification.

    Parameters
    ----------
    param_grid : dict
        Mapping from parameter name to a list of possible values. All values
        should be iterable. The function returns the Cartesian product of the
        value lists, with each combination represented as a dict.

    Returns
    -------
    List[dict]
        Each entry is a mapping from parameter names to one choice from the
        corresponding list.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = itertools.product(*values)  # cartesian product of value lists
    return [dict(zip(keys, combo)) for combo in combos]


def _freeze_params(params):
    """
    Produce a hashable representation of a parameter dict.

    The returned tuple consists of sorted ``(key, value)`` pairs. Sorting
    ensures that the ordering is deterministic, so two dicts with the same
    contents produce identical tuples even if their insertion order differs.
    This is useful when using parameter sets as keys in caching dictionaries.

    Parameters
    ----------
    params : dict
        Hyperparameter mapping.

    Returns
    -------
    tuple
        Sorted tuple of key/value pairs.
    """
    return tuple(sorted(params.items(), key=lambda x: x[0]))


def cross_validate_model_per_cell(
    X,
    Y,
    cell_ids,
    model_class,
    model_kwargs,
    k_folds,
    scaler=None,
    custom_train_fn=None,
    trainer_params=None,
    verbose=False,
):
    """
    Perform k-fold cross-validation separately for each cell.

    This helper rearranges the full dataset by cell, then for each cell
    performs ``k_folds`` splits and fits the provided ``model_class`` on the
    training portion.  Evaluation uses the pseudo-R² score defined in
    :mod:`src.train.evaluate`.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_samples)
        Shared feature matrix where columns correspond to samples.
    Y : ndarray, shape (n_samples,)
        Response vector of counts.
    cell_ids : ndarray, shape (n_samples,)
        Integer/label identifying which cell produced each sample.
    model_class : callable
        Constructor for the estimator to evaluate. Will be called with
        ``**model_kwargs``.
    model_kwargs : dict
        Keyword arguments for ``model_class``.
    k_folds : int
        Number of folds for cross-validation.
    scaler : callable or None, optional
        Scaler factory (e.g. ``StandardScaler``) applied to train/val sets.
    custom_train_fn : callable or None, optional
        If provided, used in place of ``model.fit``. It should accept
        ``(model, X_train, y_train, X_val, y_val, **trainer_params)`` and
        return a trained model.
    trainer_params : dict or None
        Additional parameters passed to ``custom_train_fn``.
    verbose : bool, optional
        If True, print progress information for each cell and fold.

    Returns
    -------
    dict
        Mapping from each cell ID to its mean cross-validated pseudo-R² score.
    """
    if trainer_params is None:
        trainer_params = {}

    unique_cells = np.unique(cell_ids)
    scores = {}

    # iterate through cells and perform internal k-fold procedure
    for cell in unique_cells:
        if verbose:
            print(f"Cross-validating cell {cell}")
        Xc = X[:, cell_ids == cell].T  # features for this cell (samples x features)
        yc = Y[cell_ids == cell]

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(KFold(k_folds).split(Xc)):
            if verbose:
                print(f"  Fold {fold_idx+1}/{k_folds}")
            # split data for this fold
            X_train, X_val = Xc[train_idx], Xc[val_idx]
            y_train, y_val = yc[train_idx], yc[val_idx]

            # optional scaling within the cell
            if scaler is not None:
                s = scaler()
                X_train = s.fit_transform(X_train)
                X_val = s.transform(X_val)

            model = model_class(**model_kwargs)

            if custom_train_fn is None:
                model.fit(X_train, y_train)
            else:
                # custom training routine could encapsulate early stopping, etc.
                model = custom_train_fn(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    **trainer_params,
                )

            y_pred = model.predict(X_val)
            y_pred = np.clip(y_pred, 1e-8, None)  # avoid zero predictions
            score = pseudo_r2(y_val, y_pred)
            fold_scores.append(score)

        scores[cell] = np.mean(fold_scores)

    return scores


def grid_search_per_cell(
    X,
    Y,
    cell_ids,
    model_class,
    model_param_grid,
    trainer_param_grid=None,
    k_folds=3,
    scaler=None,
    custom_train_fn=None,
    verbose=False,
):
    """
    Perform grid search on both model and trainer hyperparameters per cell.

    This routine coordinates a full-factorial search over two sets of
    hyperparameters: those belonging to the model itself and those associated
    with a custom training procedure (e.g. neural network trainer settings).
    Each candidate pair of parameter dictionaries is evaluated by calling
    :func:`cross_validate_model_per_cell` and the mean pseudo-R² is recorded for
    every cell. The best parameter tuple is then chosen independently for each
    cell.

    Parameters
    ----------
    X, Y, cell_ids
        Data in the same format expected by :func:`cross_validate_model_per_cell`.
    model_class : callable
        Constructor for the base estimator. Note that during evaluation the
        actual constructor called is wrapped so that it ignores ``**kw`` and
        instead uses the current ``model_params`` combination.
    model_param_grid : dict
        Hyperparameter grid for the model. Values should be lists.
    trainer_param_grid : dict or None, optional
        Hyperparameter grid for the trainer. If ``None`` only model parameters
        are varied.
    k_folds : int, optional
        Number of cross-validation folds (default 3).
    scaler : callable or None, optional
        Feature scaler factory applied per cell.
    custom_train_fn : callable or None, optional
        If provided, will be passed through to the CV routine.
    verbose : bool, optional
        If True, print progress messages during the search.

    Returns
    -------
    dict
        ``{"best_params": {...}, "all_scores": {...}}``.  ``best_params``
        maps each cell to the optimal model/tester parameter set; ``all_scores``
        contains the full CV results keyed by frozen parameter tuples.
    """

    def objective(trial):
        # Sample params
        mp = {}
        for k, v in model_param_grid.items():
            mp[k] = trial.suggest_categorical(k, v)
        tp = {}
        if trainer_param_grid:
            for k, v in trainer_param_grid.items():
                if k == "l1_lambda":
                    tp[k] = trial.suggest_float(
                        k, min(v), max(v), log=False
                    )  # Changed to log=False
                else:
                    tp[k] = trial.suggest_categorical(k, v)

        def _model_class_wrapper(**kw):
            return model_class(**mp)

        scores = cross_validate_model_per_cell(
            X,
            Y,
            cell_ids,
            _model_class_wrapper,
            {},
            k_folds,
            scaler,
            custom_train_fn,
            tp,
            verbose,
        )

        # Return mean across cells
        return np.mean(list(scores.values()))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed

    best_params_optuna = study.best_params
    model_params = {
        k: v for k, v in best_params_optuna.items() if k in model_param_grid
    }
    trainer_params = {
        k: v for k, v in best_params_optuna.items() if k in (trainer_param_grid or {})
    }

    # To keep API, set best_params for each cell to the global best
    unique_cells = np.unique(cell_ids)
    best_params = {}
    for cell in unique_cells:
        best_params[cell] = {
            "model_params": model_params,
            "trainer_params": trainer_params,
        }

    # For all_scores, run with best params to populate
    frozen = (_freeze_params(model_params), _freeze_params(trainer_params))

    def _model_class_wrapper(**kw):
        return model_class(**model_params)

    scores = cross_validate_model_per_cell(
        X,
        Y,
        cell_ids,
        _model_class_wrapper,
        {},
        k_folds,
        scaler,
        custom_train_fn,
        trainer_params,
        verbose,
    )
    all_scores = {frozen: scores}

    return {
        "best_params": best_params,
        "all_scores": all_scores,
    }


def grid_search_transfer_learning(
    X,
    Y,
    cell_ids,
    model_class,
    model_param_grid,
    trainer_param_grid,
    scaler=None,
    verbose=False,
    agg_method="median",
):
    """
    Grid search hyperparameters for a transfer-learning architecture.

    This routine evaluates each candidate combination on a per-cell
    validation fold and then aggregates the per-cell pseudo-R² scores to
    determine which set of hyperparameters is best.  By default the
    **median** pseudo-R² across cells is used, which reduces the influence of
    a few poorly-predicted cells.  If you prefer to maximise the overall
    average performance you may pass ``agg_method="mean"``.

    Parameters
    ----------
    X, Y, cell_ids : array-like
        Full dataset analogous to other helpers.
    model_class : callable
        Constructor for the transfer-learning model; it must accept
        ``n_features`` and ``n_cells`` along with additional parameters.
    model_param_grid : dict
        Grid of parameters to pass to ``model_class``.
    trainer_param_grid : dict
        Grid of parameters for ``TransferLearningTrainer``.
    scaler : callable or None, optional
        If provided, each cell's features in the training split are scaled.
    verbose : bool, optional
        If True, print progress messages while evaluating parameter
        combinations and aggregate scores.
    agg_method : {"median", "mean"}, optional
        How to aggregate per-cell validation scores when selecting the best
        hyperparameters.  ``"median"`` is more robust to outliers, while
        ``"mean"`` maximises overall average performance.

    Returns
    -------
    dict
        Same structure as ``grid_search_per_cell``: best parameters and all
        observed scores.
    """
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    # ------------------------------------------------------------
    # Use a proper train/val split (no test) for hyperparam search
    # ------------------------------------------------------------
    # Each cell is split independently; the validation data will be used to
    # score hyperparameter combinations, leaving a separate test set untouched
    # until final evaluation.
    Xtr, Ytr, Xv, Yv, _, _ = prepare_cellwise_datasets(
        X, Y, cell_ids, train_frac=0.7, val_frac=0.15, use_val=True
    )
    if scaler is not None:
        for cell in unique_cells:
            sc = scaler()
            Xtr[cell] = sc.fit_transform(Xtr[cell])
            Xv[cell] = sc.transform(Xv[cell])
    X_cells_train = [_to_tensor(Xtr[cell], device) for cell in unique_cells]
    Y_cells_train = [_to_tensor(Ytr[cell], device) for cell in unique_cells]
    X_cells_val = [_to_tensor(Xv[cell], device) for cell in unique_cells]
    Y_cells_val = Y_cells_val  # Keep NumPy

    def objective(trial):
        # Sample params
        mp = {}
        for k, v in model_param_grid.items():
            if isinstance(v[0], tuple):
                mp[k] = trial.suggest_categorical(k, v)
            else:
                mp[k] = trial.suggest_categorical(k, v)
        tp = {}
        if trainer_param_grid:
            for k, v in trainer_param_grid.items():
                if k == "l1_lambda":
                    tp[k] = trial.suggest_float(
                        k, min(v), max(v), log=False
                    )  # Changed to log=False
                else:
                    tp[k] = trial.suggest_categorical(k, v)

        # Train on the training split
        model = model_class(n_features=n_features, n_cells=n_cells, **mp).to(device)
        trainer = TransferLearningTrainer(**tp)
        out = trainer.train(model, X_cells_train, Y_cells_train)
        model = out[0] if isinstance(out, tuple) else out

        # Evaluate on the validation split, cell-by-cell
        # each head is scored separately so we can inspect per-cell
        # performance later; this also allows robust aggregation.
        cell_scores = {}
        for ci, cell in enumerate(unique_cells):
            model.eval()
            with torch.no_grad():
                y_pred = model(X_cells_val[ci], ci).cpu().numpy()
            score = pseudo_r2(Y_cells_val[ci], y_pred)
            cell_scores[cell] = score

        # Free memory
        del model
        torch.cuda.empty_cache()

        scores_arr = np.array(list(cell_scores.values()))
        agg = np.median(scores_arr) if agg_method == "median" else np.mean(scores_arr)
        return agg

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_params_optuna = study.best_params
    best_model_params = {
        k: v for k, v in best_params_optuna.items() if k in model_param_grid
    }
    best_trainer_params = {
        k: v for k, v in best_params_optuna.items() if k in trainer_param_grid
    }

    # Populate all_scores by re-running best params
    frozen = (_freeze_params(best_model_params), _freeze_params(best_trainer_params))
    model = model_class(n_features=n_features, n_cells=n_cells, **best_model_params).to(
        device
    )
    trainer = TransferLearningTrainer(**best_trainer_params)
    out = trainer.train(model, X_cells_train, Y_cells_train)
    model = out[0] if isinstance(out, tuple) else out
    cell_scores = {}
    for ci, cell in enumerate(unique_cells):
        model.eval()
        with torch.no_grad():
            y_pred = model(X_cells_val[ci], ci).cpu().numpy()
        score = pseudo_r2(Y_cells_val[ci], y_pred)
        cell_scores[cell] = score
    all_scores = {frozen: cell_scores}

    return {
        "best_params": {
            "model_params": best_model_params,
            "trainer_params": best_trainer_params,
        },
        "all_scores": all_scores,
    }
