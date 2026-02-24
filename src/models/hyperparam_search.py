import itertools
import torch
import numpy as np
from sklearn.model_selection import KFold
from src.models.evaluate import pseudo_r2
from src.models.poisson_nn.nn_training import TransferLearningTrainer


def _expand_param_grid(param_grid):
    """
    Convert a dict of lists into a list of dicts (cartesian product).
    Example:
        {"alpha": [0.1, 1.0], "penalty": ["l2", "l1"]}
    becomes:
        [
            {"alpha": 0.1, "penalty": "l2"},
            {"alpha": 0.1, "penalty": "l1"},
            {"alpha": 1.0, "penalty": "l2"},
            {"alpha": 1.0, "penalty": "l1"},
        ]
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = itertools.product(*values)
    return [dict(zip(keys, combo)) for combo in combos]


def _freeze_params(params):
    """
    Convert a dict of params into a sorted tuple of (key, value) pairs.
    This makes it hashable and stable for use as a dictionary key.
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
    """
    Per-cell k-fold cross-validation using pseudo-R2 as score.

    Returns: dict[cell_id] -> mean CV pseudo-R2
    """
    if trainer_params is None:
        trainer_params = {}

    unique_cells = np.unique(cell_ids)
    scores = {}

    for cell in unique_cells:
        Xc = X[:, cell_ids == cell].T
        yc = Y[cell_ids == cell]

        fold_scores = []

        for train_idx, val_idx in KFold(k_folds).split(Xc):
            X_train, X_val = Xc[train_idx], Xc[val_idx]
            y_train, y_val = yc[train_idx], yc[val_idx]

            if scaler is not None:
                s = scaler()
                X_train = s.fit_transform(X_train)
                X_val = s.transform(X_val)

            model = model_class(**model_kwargs)

            if custom_train_fn is None:
                model.fit(X_train, y_train)
            else:
                model = custom_train_fn(
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    **trainer_params,
                )

            y_pred = model.predict(X_val)
            y_pred = np.clip(y_pred, 1e-8, None)  # ensure strictly positive
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
    """
    Unified grid search for models with separate model + trainer hyperparameters.

    model_param_grid: dict of model hyperparameters
    trainer_param_grid: dict of trainer hyperparameters (optional)
    """

    # Expand model params
    model_param_combos = _expand_param_grid(model_param_grid)

    # Expand trainer params (or use empty dict)
    if trainer_param_grid is None:
        trainer_param_combos = [{}]
    else:
        trainer_param_combos = _expand_param_grid(trainer_param_grid)

    # Combine both
    all_param_combos = []
    for mp in model_param_combos:
        for tp in trainer_param_combos:
            all_param_combos.append((mp, tp))

    all_scores = {}

    # Evaluate each combination
    for model_params, trainer_params in all_param_combos:

        frozen = (
            _freeze_params(model_params),
            _freeze_params(trainer_params),
        )

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

    # Select best params per cell
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
    """
    Grid search for transfer-learning models.
    One shared model, one shared trainer.
    """

    model_param_combos = _expand_param_grid(model_param_grid)
    trainer_param_combos = _expand_param_grid(trainer_param_grid)

    all_scores = {}

    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    # Prepare per-cell data
    X_cells = []
    Y_cells = []
    for cell in unique_cells:
        idx = np.where(cell_ids == cell)[0]
        Xc = X[:, idx].T
        Yc = Y[idx]
        if scaler is not None:
            sc = scaler()
            Xc = sc.fit_transform(Xc)
        X_cells.append(Xc)
        Y_cells.append(Yc)

    # Evaluate each combination
    for mp in model_param_combos:
        for tp in trainer_param_combos:

            frozen = (_freeze_params(mp), _freeze_params(tp))

            # Build model + trainer
            model = model_class(n_features=n_features, n_cells=n_cells, **mp)
            trainer = TransferLearningTrainer(**tp)

            # Train
            out = trainer.train(model, X_cells, Y_cells)
            # Handle (model, train_losses) return format
            if isinstance(out, tuple):
                model = out[0]
            else:
                model = out

            # Evaluate
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

    # Select best params
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
