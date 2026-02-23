import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PoissonTrainer:
    """
    Generic trainer for Poisson neural networks.
    Supports:
    - full batch or mini-batch
    - L1 and L2 regularisation
    - validation set
    - early stopping
    """

    def __init__(
        self,
        lr=1e-3,
        epochs=100,
        weight_decay=1e-4,
        l1_lambda=0.0,
        batch_size="auto",
        patience=10,
    ):
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size
        self.patience = patience

    def _get_batch_size(self, n_samples):
        if self.batch_size == "auto":
            return None if n_samples < 2000 else 128
        return self.batch_size

    def _l1_penalty(self, model):
        return sum(p.abs().sum() for p in model.parameters())

    def train(self, model, X_train, y_train, X_val, y_val):
        """
        Train model with validation set for early stopping.
        """
        device = torch.device("cpu")
        model.to(device)

        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

        batch_size = self._get_batch_size(len(X_train))
        if batch_size is None:
            train_loader = [(X_train, y_train)]
        else:
            ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        criterion = nn.PoissonNLLLoss(log_input=False)
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        for _ in range(self.epochs):
            # Training
            model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                optimiser.zero_grad()
                preds = model(xb)
                loss = criterion(preds, yb)
                if self.l1_lambda > 0:
                    loss = loss + self.l1_lambda * self._l1_penalty(model)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            train_losses.append(epoch_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()

            val_losses.append(val_loss)

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return model, train_losses, val_losses


class TransferLearningTrainer:
    """
    Trainer for SharedHiddenPoissonNN.
    """

    def __init__(
        self,
        lr=1e-3,
        epochs=100,
        weight_decay=1e-4,
        l1_lambda=0.0,
        batch_size="auto",
    ):
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size

    def _get_batch_size(self, n_samples):
        if self.batch_size == "auto":
            return None if n_samples < 2000 else 128
        return self.batch_size

    def _l1_penalty(self, model):
        return sum(p.abs().sum() for p in model.parameters())

    def train(self, model, X_cells, Y_cells):
        """
        Joint training across all cells.
        X_cells, Y_cells: lists of arrays per cell
        """
        device = torch.device("cpu")
        model.to(device)

        # Convert to tensors
        X_cells = [torch.tensor(x, dtype=torch.float32, device=device) for x in X_cells]
        Y_cells = [torch.tensor(y, dtype=torch.float32, device=device) for y in Y_cells]

        criterion = nn.PoissonNLLLoss(log_input=False)
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        for _ in range(self.epochs):
            model.train()
            optimiser.zero_grad()

            total_loss = 0.0
            for ci, (Xc, Yc) in enumerate(zip(X_cells, Y_cells)):
                preds = model(Xc, ci)
                loss = criterion(preds, Yc)
                total_loss += loss

            if self.l1_lambda > 0:
                total_loss = total_loss + self.l1_lambda * self._l1_penalty(model)

            total_loss.backward()
            optimiser.step()

        return model
