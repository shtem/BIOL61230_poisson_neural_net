from src.models.poisson_nn.nn_models import PoissonNN, SharedHiddenPoissonNN
from src.models.poisson_nn.nn_training import PoissonTrainer, TransferLearningTrainer
from src.models.utils import fit_model_per_cell
from src.models.evaluate import evaluate_poisson_model
from src.models.hyperparam_search import (
    grid_search_per_cell,
    grid_search_transfer_learning,
)
import numpy as np
import torch


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

    # -------------------------
    # MODE A — GLOBAL PARAMS
    # -------------------------
    if not grid_search:

        # Default model + trainer params
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

        def train_fn(model, Xtr, ytr, Xv, yv, **tp):
            trainer = PoissonTrainer(**trainer_params)
            model, tl, vl = trainer.train(model, Xtr, ytr, Xv, yv)
            model.train_losses = tl
            model.val_losses = vl
            return model

        results = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=model_class,
            model_kwargs={},
            train_frac=train_frac,
            val_frac=val_frac,
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

    # -------------------------
    # MODE B — GRID SEARCH
    # -------------------------

    # Ensure grids exist
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

    gs = grid_search_per_cell(
        X,
        Y,
        cell_ids,
        model_class=lambda **kw: PoissonNN(n_features=X.shape[0], **kw),
        model_param_grid=model_param_grid,
        trainer_param_grid=trainer_param_grid,
        k_folds=k_folds,
        scaler=scaler,
        custom_train_fn=lambda model, Xtr, ytr, Xv, yv, **tp: PoissonTrainer(
            **tp
        ).train(model, Xtr, ytr, Xv, yv)[0],
    )

    best_params = gs["best_params"]

    # Fit final models per cell
    final_results = {}
    for cell in np.unique(cell_ids):
        mp = best_params[cell]["model_params"]
        tp = best_params[cell]["trainer_params"]

        model = PoissonNN(n_features=X.shape[0], **mp)
        trainer = PoissonTrainer(**tp)

        final_results[cell] = fit_model_per_cell(
            X,
            Y,
            cell_ids,
            model_class=lambda **kw: PoissonNN(n_features=X.shape[0], **mp),
            model_kwargs={},
            train_frac=train_frac,
            val_frac=val_frac,
            scaler=scaler,
            custom_train_fn=lambda m, Xtr, ytr, Xv, yv: trainer.train(
                m, Xtr, ytr, Xv, yv
            )[0],
        )[cell]

    return {
        "results": final_results,
        "best_params": best_params,
        "all_scores": gs["all_scores"],
    }


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
    scaler=None,
):
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

    # -------------------------
    # MODE A — GLOBAL PARAMS
    # -------------------------
    if not grid_search:
        model_params = {"hidden_sizes": hidden_sizes}
        trainer_params = {
            "lr": lr,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "l1_lambda": l1_lambda,
            "batch_size": batch_size,
        }

        model = SharedHiddenPoissonNN(n_features, hidden_sizes, n_cells)
        trainer = TransferLearningTrainer(**trainer_params)
        model = trainer.train(model, X_cells, Y_cells)

        results = {}
        for ci, cell in enumerate(unique_cells):
            Xc = torch.tensor(X_cells[ci], dtype=torch.float32)
            model.eval()
            with torch.no_grad():
                y_pred = model(Xc, ci).cpu().numpy()
            y_true = Y_cells[ci]
            metrics = evaluate_poisson_model(y_true, y_pred)
            results[cell] = {
                "model": model,
                "cell_head_index": ci,
                "y_test": y_true,
                "y_pred_test": y_pred,
                "test": metrics,
            }

        return {
            "results": results,
            "best_params": {
                "model_params": model_params,
                "trainer_params": trainer_params,
            },
            "all_scores": None,
        }

    # -------------------------
    # MODE B — GRID SEARCH
    # -------------------------
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

    # Fit final model
    model = SharedHiddenPoissonNN(n_features, best_mp["hidden_sizes"], n_cells)
    trainer = TransferLearningTrainer(**best_tp)
    model = trainer.train(model, X_cells, Y_cells)

    # Evaluate per cell
    results = {}
    for ci, cell in enumerate(unique_cells):
        Xc = torch.tensor(X_cells[ci], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(Xc, ci).cpu().numpy()
        y_true = Y_cells[ci]
        metrics = evaluate_poisson_model(y_true, y_pred)
        results[cell] = {
            "model": model,
            "cell_head_index": ci,
            "y_test": y_true,
            "y_pred_test": y_pred,
            "test": metrics,
        }

    return {
        "results": results,
        "best_params": gs["best_params"],
        "all_scores": gs["all_scores"],
    }
