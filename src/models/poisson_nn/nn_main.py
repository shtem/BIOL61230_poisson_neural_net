from src.models.poisson_nn.nn_models import PoissonNN, SharedHiddenPoissonNN
from src.models.poisson_nn.nn_training import PoissonTrainer, TransferLearningTrainer
from src.models.utils import fit_model_per_cell
from src.models.evaluate import evaluate_poisson_model
from src.get_data import prepare_cellwise_datasets, flatten_cellwise_data
from src.models.hyperparam_search import (
    grid_search_per_cell,
    grid_search_transfer_learning,
)
import numpy as np
import torch


# ============================================================
# Unified training wrapper (works for both trainers)
# ============================================================
def run_trainer(trainer, model, Xtr, ytr=None, Xv=None, yv=None):
    """
    Unified wrapper for both PoissonTrainer and TransferLearningTrainer.
    Detects which signature to use based on the trainer type.
    """

    # --- Transfer Learning Trainer ---
    if isinstance(trainer, TransferLearningTrainer):
        out = trainer.train(model, Xtr, ytr)  # Xtr = X_cells, ytr = Y_cells

        # TL returns (model, train_losses)
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


# ============================================================
# Per-cell PoissonNN
# ============================================================
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
):
    """
    Fit per-cell PoissonNN models, optionally with per-cell grid search.
    """

    # ============================================================
    # 1. Prepare datasets (validation only if grid_search=True)
    # ============================================================
    use_val = grid_search

    Xtr, Ytr, Xv, Yv, Xte, Yte = prepare_cellwise_datasets(
        X,
        Y,
        cell_ids,
        train_frac=train_frac,
        val_frac=val_frac,
        use_val=use_val,
    )
    Xtr_flat, Ytr_flat, cell_ids_tr_flat = flatten_cellwise_data(Xtr, Ytr)

    # ============================================================
    # MODE A — GLOBAL PARAMS (no grid search)
    # ============================================================
    if not grid_search:

        model_params = model_param_grid or {"hidden_sizes": hidden_sizes}
        trainer_params = trainer_param_grid or {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
            "patience": patience,
        }

        def model_class(**kw):
            return PoissonNN(n_features=X.shape[0], **model_params)

        def train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c, **tp):
            trainer = PoissonTrainer(**trainer_params)
            return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

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

    # ============================================================
    # MODE B — GRID SEARCH
    # ============================================================
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

    def gs_train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c, **tp):
        trainer = PoissonTrainer(**tp)
        return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

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
    )

    best_params = gs["best_params"]

    # ============================================================
    # Fit final models using best hyperparameters
    # ============================================================
    final_results = {}

    for cell in np.unique(cell_ids):
        mp = best_params[cell]["model_params"]
        tp = best_params[cell]["trainer_params"]

        def final_train_fn(model, Xtr_c, ytr_c, Xv_c, yv_c):
            trainer = PoissonTrainer(**tp)
            return run_trainer(trainer, model, Xtr_c, ytr_c, Xv_c, yv_c)

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


# ============================================================
# Transfer Learning PoissonNN
# ============================================================
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
):
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_features = X.shape[0]

    # ============================================================
    # Proper train/test split for TL (no val)
    # ============================================================
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

    # Convert dict → list in cell order
    X_cells_train = [Xtr[cell] for cell in unique_cells]
    Y_cells_train = [Ytr[cell] for cell in unique_cells]

    X_cells_test = [Xte[cell] for cell in unique_cells]
    Y_cells_test = [Yte[cell] for cell in unique_cells]

    # ============================================================
    # MODE A — NO GRID SEARCH
    # ============================================================
    if not grid_search:

        model_params = {"hidden_sizes": hidden_sizes}
        trainer_params = {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
            "patience": patience,
        }

        model = SharedHiddenPoissonNN(n_features, hidden_sizes, n_cells)
        trainer = TransferLearningTrainer(**trainer_params)
        model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

        # Evaluate on held-out test set
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

    # ============================================================
    # MODE B — GRID SEARCH
    # ============================================================
    gs = grid_search_transfer_learning(
        X,
        Y,
        cell_ids,
        model_class=lambda n_features, n_cells, **kw: SharedHiddenPoissonNN(
            n_features, kw["hidden_sizes"], n_cells
        ),
        model_param_grid=model_param_grid,
        trainer_param_grid=trainer_param_grid,
        scaler=scaler,
    )

    best_mp = gs["best_params"]["model_params"]
    best_tp = gs["best_params"]["trainer_params"]

    # Fit final TL model on train split
    model = SharedHiddenPoissonNN(n_features, best_mp["hidden_sizes"], n_cells)
    trainer = TransferLearningTrainer(**best_tp)
    model = run_trainer(trainer, model, X_cells_train, Y_cells_train)

    # Evaluate per cell on test split
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
