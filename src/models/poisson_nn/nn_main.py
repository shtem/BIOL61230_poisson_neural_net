from src.models.poisson_nn.nn_models import PoissonNN, SharedHiddenPoissonNN
from src.models.poisson_nn.nn_training import PoissonTrainer, TransferLearningTrainer
from src.models.utils import fit_model_per_cell
from src.models.evaluate import evaluate_poisson_model
import numpy as np
import torch


def fit_poisson_nn(
    X,
    Y,
    cell_ids,
    train_frac=0.7,
    val_frac=0.15,
    scaler=None,
    hidden_sizes=[32, 16],
    lr=1e-3,
    epochs=100,
    weight_decay=1e-4,
    l1_lambda=0.0,
    batch_size="auto",
    patience=10,
):
    """
    Fit PoissonNN per cell using the new trainer.
    """

    n_features = X.shape[0]

    # Build model class with model hyperparameters only
    def model_class(**kw):
        return PoissonNN(n_features, hidden_sizes=hidden_sizes)

    # Build training function with trainer hyperparameters only
    def train_fn(model, X_train, y_train, X_val, y_val):
        trainer = PoissonTrainer(
            lr=lr,
            epochs=epochs,
            weight_decay=weight_decay,
            l1_lambda=l1_lambda,
            batch_size=batch_size,
            patience=patience,
        )
        model, train_losses, val_losses = trainer.train(
            model, X_train, y_train, X_val, y_val
        )
        model.train_losses = train_losses
        model.val_losses = val_losses
        return model

    return fit_model_per_cell(
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


def fit_poisson_nn_transfer_learning(
    X,
    Y,
    cell_ids,
    hidden_sizes=[32, 16],
    lr=1e-3,
    epochs=100,
    weight_decay=1e-4,
    l1_lambda=0.0,
    batch_size="auto",
    scaler=None,
):
    """
    Fit transfer-learning NN with shared hidden layers.
    """
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

    model = SharedHiddenPoissonNN(n_features, hidden_sizes, n_cells)
    trainer = TransferLearningTrainer(
        lr=lr,
        epochs=epochs,
        weight_decay=weight_decay,
        l1_lambda=l1_lambda,
        batch_size=batch_size,
    )
    model = trainer.train(model, X_cells, Y_cells)

    results = {}
    for ci, cell in enumerate(unique_cells):
        Xc = torch.tensor(X_cells[ci], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(Xc, ci).numpy()
        y_true = Y_cells[ci]
        metrics = evaluate_poisson_model(y_true, y_pred)
        results[cell] = {
            "model": model,
            "cell_head_index": ci,
            "y_test": y_true,
            "y_pred_test": y_pred,
            "test": metrics,
        }

    return results
