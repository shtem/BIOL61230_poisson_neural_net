import torch
import itertools
import numpy as np
from sklearn.model_selection import KFold
from src.models.evaluate import pseudo_r2
from src.get_data import prepare_cellwise_datasets
from src.models.poisson_nn.nn_training import TransferLearningTrainer


def _expand_param_grid(param_grid):
    """Generate all parameter combinations from a grid specification.

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
    """Produce a hashable representation of a parameter dict.

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
):
    """Perform k-fold cross-validation separately for each cell.

    This helper rearranges the full dataset by cell, then for each cell
    performs ``k_folds`` splits and fits the provided ``model_class`` on the
    training portion.  Evaluation uses the pseudo‑R² score defined in
    :mod:`src.models.evaluate`.

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

    Returns
    -------
    dict
        Mapping from each cell ID to its mean cross-validated pseudo‑R² score.
    """
    if trainer_params is None:
        trainer_params = {}

    unique_cells = np.unique(cell_ids)
    scores = {}

    # iterate through cells and perform internal k-fold procedure
    for cell in unique_cells:
        Xc = X[:, cell_ids == cell].T  # features for this cell (samples x features)
        yc = Y[cell_ids == cell]

        fold_scores = []

        for train_idx, val_idx in KFold(k_folds).split(Xc):
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
):
    """Perform grid search on both model and trainer hyperparameters per cell.

    This routine coordinates a full-factorial search over two sets of
    hyperparameters: those belonging to the model itself and those associated
    with a custom training procedure (e.g. neural network trainer settings).
    Each candidate pair of parameter dictionaries is evaluated by calling
    :func:`cross_validate_model_per_cell` and the mean pseudo‑R² is recorded for
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

    Returns
    -------
    dict
        ``{"best_params": {...}, "all_scores": {...}}``.  ``best_params``
        maps each cell to the optimal model/tester parameter set; ``all_scores``
        contains the full CV results keyed by frozen parameter tuples.
    """

    # Expand model params
    model_param_combos = _expand_param_grid(model_param_grid)

    # Expand trainer params (or use empty dict)
    if trainer_param_grid is None:
        trainer_param_combos = [{}]
    else:
        trainer_param_combos = _expand_param_grid(trainer_param_grid)

    # Combine both parameter sets into a list of pairs
    all_param_combos = []
    for mp in model_param_combos:
        for tp in trainer_param_combos:
            all_param_combos.append((mp, tp))

    all_scores = {}

    # Evaluate each combination via cross-validation
    for model_params, trainer_params in all_param_combos:

        frozen = (
            _freeze_params(model_params),
            _freeze_params(trainer_params),
        )

        # wrapper ignores kw because grid search handles parameters separately
        def _model_class_wrapper(**kw):
            return model_class(**model_params)

        scores = cross_validate_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=_model_class_wrapper,
            model_kwargs={},
            k_folds=k_folds,
            scaler=scaler,
            custom_train_fn=custom_train_fn,
            trainer_params=trainer_params,
        )

        all_scores[frozen] = scores

    # Select best params/cv score for each cell separately
    best_params = {}
    unique_cells = np.unique(cell_ids)

    for cell in unique_cells:
        best_score = -np.inf
        best_combo = None

        for frozen, cell_scores in all_scores.items():
            score = cell_scores[cell]
            if score > best_score:
                best_score = score
                best_combo = frozen

        model_params = dict(best_combo[0])
        trainer_params = dict(best_combo[1])

        best_params[cell] = {
            "model_params": model_params,
            "trainer_params": trainer_params,
        }

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
):
    """Grid search hyperparameters for a transfer‑learning architecture.

    This function differs from ``grid_search_per_cell`` in that it handles a
    special transfer-learning model trained jointly across cells.  We only use
    the training portion of the data (no validation/test) to avoid leakage;
    the final evaluation happens on the same split that will be used by the
    production model.

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

    Returns
    -------
    dict
        Same structure as ``grid_search_per_cell``: best parameters and all
        observed scores.
    """

    model_param_combos = _expand_param_grid(model_param_grid)
    trainer_param_combos = _expand_param_grid(trainer_param_grid)

    all_scores = {}

    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    # ------------------------------------------------------------
    # Use SAME train split as final TL model
    # ------------------------------------------------------------
    Xtr, Ytr, _, _, _, _ = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=0.85,
        val_frac=0.0,
        use_val=False,
    )

    # Optional scaling per cell on training-only data
    if scaler is not None:
        for cell in unique_cells:
            sc = scaler()
            Xtr[cell] = sc.fit_transform(Xtr[cell])

    # Convert dictionary to ordered lists for consistent indexing
    X_cells = [Xtr[cell] for cell in unique_cells]
    Y_cells = [Ytr[cell] for cell in unique_cells]

    # ------------------------------------------------------------
    # Evaluate each combination
    # ------------------------------------------------------------
    for mp in model_param_combos:
        for tp in trainer_param_combos:

            frozen = (_freeze_params(mp), _freeze_params(tp))

            # Build model + trainer instances
            model = model_class(n_features=n_features, n_cells=n_cells, **mp)
            trainer = TransferLearningTrainer(**tp)

            # Train the transfer-learning model
            out = trainer.train(model, X_cells, Y_cells)
            model = out[0] if isinstance(out, tuple) else out

            # Evaluate on the same training split, cell-by-cell
            cell_scores = []
            for ci, cell in enumerate(unique_cells):
                Xc = torch.tensor(X_cells[ci], dtype=torch.float32)
                model.eval()
                with torch.no_grad():
                    y_pred = model(Xc, ci).cpu().numpy()
                y_true = Y_cells[ci]
                score = pseudo_r2(y_true, y_pred)
                cell_scores.append(score)

            all_scores[frozen] = np.mean(cell_scores)

    # Select best params based on averaged score across cells
    best_combo = max(all_scores, key=all_scores.get)
    best_model_params = dict(best_combo[0])
    best_trainer_params = dict(best_combo[1])

    return {
        "best_params": {
            "model_params": best_model_params,
            "trainer_params": best_trainer_params,
        },
        "all_scores": all_scores,
    }
