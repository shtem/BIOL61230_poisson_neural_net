import torch
import torch.nn as nn
import numpy as np
from src.train.poisson_nn.nn_models import (
    PoissonNN,
    SharedHiddenPoissonNN,
    SharedFirstLayerPoissonNN,
    SharedNonlinearHeadsPoissonNN,
)
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
# Model factory: creates the correct TL architecture
# ------------------------------------------------------------
def make_model(model_type, n_features, n_cells, **mp):
    """
    Construct a transfer‑learning Poisson neural network architecture.

    This factory function centralises model creation for all supported
    transfer‑learning architectures. It ensures that the rest of the training
    and grid‑search pipeline can remain unchanged while allowing flexible
    selection between different shared‑representation designs.

    Parameters
    ----------
    model_type : {"shared_hidden", "shared_nonlinear_heads", "shared_first_layer"}
        Specifies which architecture to instantiate:

        - "shared_hidden":
            Deep shared feature extractor with simple linear per‑cell heads.
            Expects `mp["hidden_sizes"]` (list of ints).

        - "shared_nonlinear_heads":
            Shallow shared trunk with nonlinear per‑cell MLP heads.
            Expects:
                `mp["shared_sizes"]` (list of ints)
                `mp["head_sizes"]`   (list of ints)

        - "shared_first_layer":
            Only the first layer is shared; all deeper layers are per‑cell.
            Expects:
                `mp["shared_dim"]`  (int)
                `mp["head_sizes"]`  (list of ints)

    n_features : int
        Number of input features for each sample.

    n_cells : int
        Number of cells, used to create the appropriate number of output heads.

    **mp : dict
        Model‑specific hyperparameters required by the chosen architecture.
        Missing or incompatible keys will raise a KeyError or ValueError.

    Returns
    -------
    nn.Module
        A fully constructed Poisson neural network model matching the requested
        architecture and ready for training with the transfer‑learning trainer.
    """
    if model_type == "shared_hidden":
        return SharedHiddenPoissonNN(
            n_features=n_features,
            hidden_sizes=mp["hidden_sizes"],
            n_cells=n_cells,
        )

    elif model_type == "shared_nonlinear_heads":
        return SharedNonlinearHeadsPoissonNN(
            n_features=n_features,
            shared_sizes=mp["shared_sizes"],
            head_sizes=mp["head_sizes"],
            n_cells=n_cells,
        )

    elif model_type == "shared_first_layer":
        return SharedFirstLayerPoissonNN(
            n_features=n_features,
            shared_dim=mp["shared_dim"],
            head_sizes=mp["head_sizes"],
            n_cells=n_cells,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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
        If True, print dataset splits and hyperparameter search progress.  For
        PyTorch models a summary of the network architecture will also be
        printed (requires the ``torchsummary`` package).

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
    n_features = X.shape[0]
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

        # when verbose, print a torchsummary of the architecture once
        if verbose:
            try:
                from torchsummary import summary

                tmp = PoissonNN(n_features=n_features, **model_params)
                print("\n=== Network architecture summary ===")
                summary(tmp, (n_features,))
                print("=== end architecture summary ===\n")
            except ImportError:
                print("torchsummary not installed; skipping architecture summary")

        # factory to create new model instances per cell
        def model_class(**kw):
            return PoissonNN(n_features=n_features, **model_params)

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

        # MODE A returns immediately after fitting, so we can still summarise
        if verbose and results:
            first = sorted(results.keys())[0]
            model = results[first]["model"]
            try:
                from torchsummary import summary

                print("\n=== Final network architecture summary (first cell) ===")
                summary(model, (n_features,))
                print("=== end architecture summary ===\n")
            except ImportError:
                pass

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

    # if verbose we will print a summary for the best model at the end;
    # the results object is filled later so we'll handle that after the search

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
        model_class=lambda **kw: PoissonNN(n_features=n_features, **kw),
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
            model_class=lambda **kw: PoissonNN(n_features=n_features, **mp),
            model_kwargs={},
            scaler=scaler,
            custom_train_fn=final_train_fn,
        )[cell]

    # after building final_results, optionally print summary for first cell
    if verbose and final_results:
        first = sorted(final_results.keys())[0]
        model = final_results[first]["model"]
        try:
            from torchsummary import summary

            print("\n=== Final TL network architecture summary (first cell) ===")
            summary(model, (n_features,))
            print("=== end architecture summary ===\n")
        except ImportError:
            pass

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
    model_params=None,
    hidden_sizes=[16],
    lr=1e-3,
    epochs=100,
    weight_decay=1e-4,
    l1_lambda=0.0,
    batch_size="auto",
    patience=10,
    scaler=None,
    verbose=False,
    agg_method="median",
    model_type="shared_hidden",
):
    """
    Train a multi-cell Poisson neural network using transfer learning.

    This function supports three architectures:
    - "shared_hidden": deep shared trunk + linear per-cell heads
    - "shared_nonlinear_heads": shallow shared trunk + nonlinear per-cell heads
    - "shared_first_layer": only the first layer shared, deeper layers per-cell

    Two training modes are supported:
    - No grid search: model hyperparameters are passed via `model_params`
    - Grid search: model hyperparameters are passed via `model_param_grid`

    The function performs:
    - 85/15 train/test split (no validation for final training)
    - optional per-cell feature scaling
    - per-cell evaluation on the held-out test set
    - optional grid search using a separate 70/15/15 split

    Parameters
    ----------
    X, Y, cell_ids : array-like
        Full dataset in cell-wise format.
    grid_search : bool
        Whether to run transfer-learning grid search.
    model_param_grid : dict or None
        Hyperparameter grid for model (used only when grid_search=True).
    trainer_param_grid : dict or None
        Hyperparameter grid for trainer (used only when grid_search=True).
    model_params : dict or None
        Direct model parameters for no-grid-search mode.
    hidden_sizes : list
        Default shared hidden sizes for "shared_hidden".
    lr, epochs, weight_decay, l1_lambda, batch_size, patience : trainer settings
    scaler : callable or None
        Optional per-cell feature scaler.
    verbose : bool
        Print architecture summary and progress.
    agg_method : {"median", "mean"}
        Aggregation method for TL grid search.
    model_type : {"shared_hidden", "shared_nonlinear_heads", "shared_first_layer"}
        Selects which TL architecture to use.

    Returns
    -------
    dict
        {
            "results": per-cell evaluation,
            "best_params": best model/trainer params,
            "all_scores": grid search scores or None
        }
    """

    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    # ------------------------------------------------------------
    # Optional architecture summary
    # ------------------------------------------------------------
    if verbose:
        print(f"TL fit called with {n_cells} cells, {n_features} features")

        try:
            from torchsummary import summary

            # Build a temporary model for summary
            tmp = make_model(
                model_type,
                n_features,
                n_cells,
                hidden_sizes=hidden_sizes,
                shared_sizes=hidden_sizes,
                head_sizes=[32, 16],
                shared_dim=hidden_sizes[0],
            )

            class _Wrapper(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x, 0)

            print("\n=== TL network architecture summary ===")
            summary(_Wrapper(tmp), (n_features,))
            print("=== end architecture summary ===\n")

        except ImportError:
            print("torchsummary not installed; skipping architecture summary")

    # ------------------------------------------------------------
    # Train/test split (no validation here)
    # ------------------------------------------------------------
    Xtr, Ytr, _, _, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=0.85,
        val_frac=0.0,
        use_val=False,
    )

    # Optional per-cell scaling
    if scaler is not None:
        for cell in unique_cells:
            sc = scaler()
            Xtr[cell] = sc.fit_transform(Xtr[cell])
            Xte[cell] = sc.transform(Xte[cell])

    # Convert dict → list for trainer
    X_cells_train = [Xtr[cell] for cell in unique_cells]
    Y_cells_train = [Ytr[cell] for cell in unique_cells]
    X_cells_test = [Xte[cell] for cell in unique_cells]
    Y_cells_test = [Yte[cell] for cell in unique_cells]

    # ------------------------------------------------------------
    # MODE A — NO GRID SEARCH
    # ------------------------------------------------------------
    if not grid_search:

        # If no model_params provided, fall back to defaults
        if model_params is None:
            model_params = {
                "hidden_sizes": hidden_sizes,
                "shared_sizes": hidden_sizes,
                "head_sizes": [32, 16],
                "shared_dim": hidden_sizes[0],
            }

        trainer_params = {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
            "patience": patience,
        }

        model = make_model(model_type, n_features, n_cells, **model_params)
        trainer = TransferLearningTrainer(**trainer_params)
        model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

        # Evaluate on test set
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
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

    gs = grid_search_transfer_learning(
        Xtr_flat,
        Ytr_flat,
        cell_ids_tr_flat,
        model_class=lambda n_features, n_cells, **kw: make_model(
            model_type, n_features, n_cells, **kw
        ),
        model_param_grid=model_param_grid,
        trainer_param_grid=trainer_param_grid,
        scaler=scaler,
        verbose=verbose,
        agg_method=agg_method,
    )

    best_mp = gs["best_params"]["model_params"]
    best_tp = gs["best_params"]["trainer_params"]

    model = make_model(model_type, n_features, n_cells, **best_mp)
    trainer = TransferLearningTrainer(**best_tp)
    model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

    # Evaluate on test set
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
