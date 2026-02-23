import itertools
import numpy as np
from src.models.utils import cross_validate_model_per_cell


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


def grid_search_per_cell(
    X,
    Y,
    cell_ids,
    model_class,
    param_grid,
    k_folds=3,
    scaler=None,
):
    """
    Generic per-cell grid search for ANY model.

    Parameters
    ----------
    X : array (n_features, n_time_bins)
    Y : array (n_time_bins,)
    cell_ids : array (n_time_bins,)
    model_class : callable
        A constructor or lambda that returns a model instance.
    param_grid : dict
        Keys = hyperparameter names
        Values = list of possible values
    k_folds : int
        Number of CV folds
    scaler : optional
        A scaler class (e.g., StandardScaler)

    Returns
    -------
    {
        "best_params": {cell_id: {...}},
        "all_scores": {
            (("alpha", 0.1), ("penalty", "l2")): {cell_id: score, ...},
            ...
        }
    }
    """

    # Expand grid into list of param dicts
    param_combinations = _expand_param_grid(param_grid)

    # Storage for all scores
    all_scores = {}

    # Evaluate each hyperparameter combination
    for params in param_combinations:
        frozen = _freeze_params(params)

        # Wrap model_class so CV can instantiate it with params
        def _model_class_wrapper(**kw):
            return model_class(**params)

        scores = cross_validate_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=_model_class_wrapper,
            model_kwargs={},
            k_folds=k_folds,
            scaler=scaler,
        )

        all_scores[frozen] = scores

    # Now pick best params per cell
    best_params = {}
    unique_cells = np.unique(cell_ids)

    for cell in unique_cells:
        best_score = -np.inf
        best_param_set = None

        for frozen_params, cell_scores in all_scores.items():
            score = cell_scores[cell]
            if score > best_score:
                best_score = score
                best_param_set = frozen_params

        # Convert frozen tuple back to dict
        best_params[cell] = dict(best_param_set)

    return {
        "best_params": best_params,
        "all_scores": all_scores,
    }
