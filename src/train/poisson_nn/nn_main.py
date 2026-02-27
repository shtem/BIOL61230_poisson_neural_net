import torch
import numpy as np
from src.train.poisson_nn.nn_models import PoissonNN, SharedHiddenPoissonNN
from src.train.poisson_nn.nn_training import PoissonTrainer, TransferLearningTrainer
from src.train.training import fit_model_per_cell
from src.train.evaluate import evaluate_poisson_model
from src.get_data import prepare_cellwise_datasets, flatten_cellwise_data
from src.train.hyperparam_search import (
    grid_search_per_cell,
    grid_search_transfer_learning,
)


# ------------------------------------------------------------
# Unified training wrapper (works for both trainers)
# ------------------------------------------------------------
def run_trainer(trainer, model, Xtr, ytr=None, Xv=None, yv=None):
    """Dispatch call to the appropriate trainer and normalize outputs.

    Both trainer classes return slightly different tuples.  This helper inspects
    the ``trainer`` instance, calls its ``train`` method with the correct
    signature, then attaches ``train_losses`` and ``val_losses`` attributes to
    the returned model for later inspection.

    Parameters
    ----------
    trainer : PoissonTrainer or TransferLearningTrainer
    model : nn.Module
        Untrained model instance.
    Xtr : array-like or list
        Training data; format depends on trainer type (flat vs per-cell list).
    ytr : array-like or list, optional
        Training targets.
    Xv, yv : array-like, optional
        Validation sets used only by ``PoissonTrainer``.
    """

    # --- Transfer Learning Trainer ---
    if isinstance(trainer, TransferLearningTrainer):
        # TL trainer expects Xtr/ytr to be lists of cell-specific data
        out = trainer.train(model, Xtr, ytr)  # Xtr = X_cells, ytr = Y_cells

        # TL returns (model, train_losses) or just model
        if isinstance(out, tuple):
            model, train_losses = out
            model.train_losses = train_losses
            model.val_losses = None
        else:
            model = out
            model.train_losses = None
            model.val_losses = None

        return model

    # --- Per-cell PoissonTrainer ---
    else:
        # standard trainer uses explicit train/val splits
        out = trainer.train(model, Xtr, ytr, Xv, yv)

        # PoissonTrainer returns (model, train_losses, val_losses)
        if isinstance(out, tuple) and len(out) == 3:
            model, train_losses, val_losses = out
            model.train_losses = train_losses
            model.val_losses = val_losses
        else:
            model = out
            model.train_losses = None
            model.val_losses = None

        return model


# ------------------------------------------------------------
# Per-cell PoissonNN
# ------------------------------------------------------------
def fit_poisson_nn(
    X,
    Y,
    cell_ids,
    grid_search=False,
    model_param_grid=None,
    trainer_param_grid=None,
    hidden_sizes=[32, 16],
    k_folds=3,
    lr=1e-3,
    epochs=100,
    weight_decay=1e-4,
    l1_lambda=0.0,
    batch_size="auto",
    patience=10,
    train_frac=0.7,
    val_frac=0.15,
    scaler=None,
    verbose=False,
):
    """Train independent Poisson neural nets for each cell.

    Two modes are supported:

    * ``grid_search=False`` - use fixed hyperparameters for every cell.
    * ``grid_search=True`` - perform per-cell K-fold cross-validated grid
      search over model and trainer parameters, then refit best models.

    Parameters
    ----------
    X : ndarray (n_features, n_samples)
    Y : ndarray (n_samples,)
    cell_ids : ndarray (n_samples,)
        Identifiers mapping each sample to a cell.
    grid_search : bool, optional
        Whether to tune hyperparameters per cell.
    model_param_grid, trainer_param_grid : dict, optional
        Parameter grids passed to ``grid_search_per_cell`` when tuning.
    hidden_sizes : list, optional
        Default architecture if not tuning.
    k_folds : int
        Number of folds for cross-validation during grid search.
    lr, epochs, weight_decay, l1_lambda, batch_size, patience :
        Default trainer hyperparameters.
    train_frac, val_frac : float
        Fractions of data allocated to training and validation sets.
    scaler : callable or None
        Optional feature scaler applied independently per cell.
    verbose : bool, optional
        If True, print dataset splits and hyperparameter search progress.

    Returns
    -------
    dict
        Contains fitted ``results`` (by cell), ``best_params`` and
        ``all_scores`` from grid search (if performed).
    """

    # ------------------------------------------------------------
    # 1. Prepare datasets (validation only if grid_search=True)
    # ------------------------------------------------------------
    use_val = True
    if verbose:
        print(
            f"Preparing datasets with train_frac={train_frac}, val_frac={val_frac}, use_val={use_val}"
        )

    Xtr, Ytr, Xv, Yv, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        use_val=use_val,
    )
    # flatten for grid search routines which expect 2d data
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

    # ------------------------------------------------------------
    # MODE A — NO GRID SEARCH
    # ------------------------------------------------------------
    if not grid_search:

        # use provided grids or fall back to defaults
        model_params = model_param_grid or {"hidden_sizes": hidden_sizes}
        trainer_params = trainer_param_grid or {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
            "patience": patience,
        }

        # factory to create new model instances per cell
        def model_class(**kw):
            return PoissonNN(n_features=X.shape[0], **model_params)

        # training wrapper that ignores grid search args
        def train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c, **tp):
            trainer = PoissonTrainer(**trainer_params)
            return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

        # fit each cell separately using fit_model_per_cell helper
        results = fit_model_per_cell(
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=model_class,
            model_kwargs={},
            scaler=scaler,
            custom_train_fn=train_fn,
        )

        return {
            "results": results,
            "best_params": {
                cell: {"model_params": model_params, "trainer_params": trainer_params}
                for cell in np.unique(cell_ids)
            },
            "all_scores": None,
        }

    # ------------------------------------------------------------
    # MODE B — GRID SEARCH
    # ------------------------------------------------------------
    # if no grids provided, create trivial grids that include default values
    if model_param_grid is None:
        model_param_grid = {"hidden_sizes": [hidden_sizes]}

    if trainer_param_grid is None:
        trainer_param_grid = {
            "lr": [lr],
            "epochs": [epochs],
            "weight_decay": [weight_decay],
            "l1_lambda": [l1_lambda],
            "batch_size": [batch_size],
            "patience": [patience],
        }

    # wrapper used during cross-validation to instantiate trainers with
    # varying hyperparameters
    def gs_train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c, **tp):
        trainer = PoissonTrainer(**tp)
        return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

    # perform per-cell cross-validated grid search
    if verbose:
        print("Starting per-cell grid search")
        print("model_param_grid=", model_param_grid)
        print("trainer_param_grid=", trainer_param_grid)
    gs = grid_search_per_cell(
        Xtr_flat,
        Ytr_flat,
        cell_ids_tr_flat,
        model_class=lambda **kw: PoissonNN(n_features=X.shape[0], **kw),
        model_param_grid=model_param_grid,
        trainer_param_grid=trainer_param_grid,
        k_folds=k_folds,
        scaler=scaler,
        custom_train_fn=gs_train_fn,
        verbose=verbose,
    )

    best_params = gs["best_params"]
    if verbose:
        print("Per-cell grid search best_params:", best_params)

    # ------------------------------------------------------------
    # Fit final models using best hyperparameters
    # ------------------------------------------------------------
    final_results = {}

    for cell in np.unique(cell_ids):
        mp = best_params[cell]["model_params"]
        tp = best_params[cell]["trainer_params"]

        def final_train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c):
            trainer = PoissonTrainer(**tp)
            return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

        # refit with optimal parameters for each cell and store only that cell's
        # set of results
        final_results[cell] = fit_model_per_cell(
            Xtr,
            Ytr,
            Xv,
            Yv,
            Xte,
            Yte,
            model_class=lambda **kw: PoissonNN(n_features=X.shape[0], **mp),
            model_kwargs={},
            scaler=scaler,
            custom_train_fn=final_train_fn,
        )[cell]

    return {
        "results": final_results,
        "best_params": best_params,
        "all_scores": gs["all_scores"],
    }


# ------------------------------------------------------------
# Transfer Learning PoissonNN
# ------------------------------------------------------------
def fit_poisson_nn_transfer_learning(
    X,
    Y,
    cell_ids,
    grid_search=False,
    model_param_grid=None,
    trainer_param_grid=None,
    hidden_sizes=[16],
    lr=1e-3,
    epochs=100,
    weight_decay=1e-4,
    l1_lambda=0.0,
    batch_size="auto",
    patience=10,
    scaler=None,
    verbose=False,
):
    """Train a shared‑hidden PoissonNN across multiple cells (transfer learning).

    Each cell has its own output head but the hidden layers are shared.  This
    function supports an optional hyperparameter grid search over the same
    parameters as ``fit_poisson_nn``.  Because the transfer model is trained
    jointly, we only split into train/test (no validation) to avoid leakage.

    Parameters
    ----------
    X, Y, cell_ids : array-like
        Input data as in ``fit_poisson_nn``.
    grid_search : bool
        Whether to perform grid search using ``grid_search_transfer_learning``.
    model_param_grid, trainer_param_grid : dict, optional
        If ``grid_search`` is True, these define the search space.
    hidden_sizes : list
        Default hidden layer sizes when not tuning.
    Other arguments :
        Trainer hyperparameters (lr, epochs, etc.) passed to
        ``TransferLearningTrainer``.
    scaler : callable or None
        Optional per-cell feature scaler.
    verbose : bool, optional
        If True, print dataset information and grid-search progress.

    Returns
    -------
    dict
        Includes ``results`` keyed by cell, ``best_params`` and optionally
        ``all_scores`` from grid search.  When ``grid_search`` is True the
        search is restricted to the 85% training split to prevent information
        from the final test set leaking into hyperparameter selection.
    """
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    if verbose:
        print(f"TL fit called with {len(unique_cells)} cells, {n_features} features")

    # ------------------------------------------------------------
    # Proper train/test split for TL (no val)
    # ------------------------------------------------------------
    # We reserve 15% of the data as a final test set that is **never**
    # touched during hyperparameter tuning.  Grid search will only see the
    # remaining 85% (see below) to prevent any leakage of test-set samples.
    Xtr, Ytr, _, _, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=0.85,
        val_frac=0.0,
        use_val=False,
    )

    # Optional scaling per cell (fit on train, transform train+test)
    if scaler is not None:
        for cell in unique_cells:
            sc = scaler()
            Xtr[cell] = sc.fit_transform(Xtr[cell])
            Xte[cell] = sc.transform(Xte[cell])

    # Convert dict → list in cell order for trainers which expect lists
    X_cells_train = [Xtr[cell] for cell in unique_cells]
    Y_cells_train = [Ytr[cell] for cell in unique_cells]

    X_cells_test = [Xte[cell] for cell in unique_cells]
    Y_cells_test = [Yte[cell] for cell in unique_cells]

    # ------------------------------------------------------------
    # MODE A — NO GRID SEARCH
    # ------------------------------------------------------------
    if not grid_search:

        # fixed hyperparameters across the board
        model_params = {"hidden_sizes": hidden_sizes}
        trainer_params = {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
            "patience": patience,
        }

        # instantiate shared-hidden model and train it on all cells simultaneously
        model = SharedHiddenPoissonNN(n_features, hidden_sizes, n_cells)
        trainer = TransferLearningTrainer(**trainer_params)
        model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

        # Evaluate on held-out test set cell-by-cell
        results = {}
        for ci, cell in enumerate(unique_cells):
            Xc_test = torch.tensor(X_cells_test[ci], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                y_pred = model(Xc_test, ci).cpu().numpy()
            y_true = Y_cells_test[ci]
            metrics = evaluate_poisson_model(y_true, y_pred)

            results[cell] = {
                "model": model,
                "cell_head_index": ci,
                "y_test": y_true,
                "y_pred_test": y_pred,
                "test": metrics,
                "train": None,
                "val": None,
                "train_losses": getattr(model, "train_losses", None),
                "val_losses": getattr(model, "val_losses", None),
            }

        return {
            "results": results,
            "best_params": {
                "model_params": model_params,
                "trainer_params": trainer_params,
            },
            "all_scores": None,
        }

    # ------------------------------------------------------------
    # MODE B — GRID SEARCH
    # ------------------------------------------------------------
    # Grid search should operate only on the *training* portion defined above.
    # To achieve this we flatten the per‑cell dictionaries returned by the
    # initial split and pass them to the search routine. The grid search itself
    # will perform its own inner train/validation split on this subset.
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

    # perform a joint grid search over model and trainer parameters using
    # training-only data to avoid leakage from the held‑out test set
    if verbose:
        print("Starting TL grid search on flattened training set")
        print("model_param_grid=", model_param_grid)
        print("trainer_param_grid=", trainer_param_grid)
        print(f"training set size: {Xtr_flat.shape}, labels: {cell_ids_tr_flat.shape}")
    gs = grid_search_transfer_learning(
        Xtr_flat,
        Ytr_flat,
        cell_ids_tr_flat,
        model_class=lambda n_features, n_cells, **kw: SharedHiddenPoissonNN(
            n_features, kw["hidden_sizes"], n_cells
        ),
        model_param_grid=model_param_grid,
        trainer_param_grid=trainer_param_grid,
        scaler=scaler,
        verbose=verbose,
    )

    best_mp = gs["best_params"]["model_params"]
    best_tp = gs["best_params"]["trainer_params"]
    if verbose:
        print("Grid-search returned best_params:", gs["best_params"])
        # optionally inspect all_scores dictionary
        from pprint import pprint

        pprint(gs.get("all_scores", {}))

    # Fit final TL model on train split using found hyperparameters
    model = SharedHiddenPoissonNN(n_features, best_mp["hidden_sizes"], n_cells)
    trainer = TransferLearningTrainer(**best_tp)
    model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

    # Evaluate per cell on test split; behavior same as mode A
    results = {}
    for ci, cell in enumerate(unique_cells):
        Xc_test = torch.tensor(X_cells_test[ci], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(Xc_test, ci).cpu().numpy()
        y_true = Y_cells_test[ci]
        metrics = evaluate_poisson_model(y_true, y_pred)

        results[cell] = {
            "model": model,
            "cell_head_index": ci,
            "y_test": y_true,
            "y_pred_test": y_pred,
            "test": metrics,
            "train": None,
            "val": None,
            "train_losses": getattr(model, "train_losses", None),
            "val_losses": getattr(model, "val_losses", None),
        }

    return {
        "results": results,
        "best_params": gs["best_params"],
        "all_scores": gs["all_scores"],
    }
