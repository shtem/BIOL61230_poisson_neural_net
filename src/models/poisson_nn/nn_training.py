import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer(ABC):
    """Abstract training utility for Poisson neural networks.

    Handles common configuration such as optimizer settings, device selection,
    batch size heuristic, and optional L1 regularization. Concrete trainers must
    implement the ``train`` method that accepts a model and appropriate data
    and returns a trained model (plus any diagnostics).
    """

    def __init__(
        self,
        lr=1e-3,
        epochs=100,
        weight_decay=1e-4,
        l1_lambda=0.0,
        batch_size="auto",
        device=None,
    ):
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.batch_size = batch_size

        # Automatically choose GPU if available unless overridden
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _get_batch_size(self, n_samples):
        """Decide batch size based on dataset size and user preference.

        Using "auto" returns ``None`` (full-batch) for small datasets and 128 for
        larger ones. Returning ``None`` signals the caller to bypass DataLoader
        and use the entire set in one go.
        """
        if self.batch_size == "auto":
            return None if n_samples < 2000 else 128
        return self.batch_size

    def _l1_penalty(self, model):
        """Compute L1 norm over all parameters for sparsity regularization."""
        return sum(p.abs().sum() for p in model.parameters())

    @abstractmethod
    def train(self, *args, **kwargs):
        """Train a model. Must be implemented by subclasses."""
        pass


class PoissonTrainer(BaseTrainer):
    """Trainer for single-cell Poisson neural networks.

    Implements standard epoch-based training with optional early stopping
    monitored on a separate validation set. Loss is Poisson negative log
    likelihood, and both L2 (via optimizer weight decay) and optional L1
    penalties are supported.
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
        super().__init__(lr, epochs, weight_decay, l1_lambda, batch_size)
        self.patience = patience

    def train(self, model, X_train, y_train, X_val, y_val):
        """Fit ``model`` on training data, using ``X_val`` for early stopping.

        Parameters
        ----------
        model : nn.Module
            Poisson model implementing ``forward``.
        X_train, y_train : array-like
            Training features and targets.
        X_val, y_val : array-like
            Validation set used to track loss and drive early stopping.

        Returns
        -------
        tuple
            ``(best_model, train_losses, val_losses)`` where ``best_model`` is
            the model state with lowest validation loss.
        """
        # move model parameters to target device (CPU/GPU)
        model.to(self.device)

        # convert inputs to tensors and transfer to chosen device so that the
        # data lives in the same memory space as the model weights; this is
        # required before any forward/backward passes can occur
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        # determine minibatch size from heuristic or user override
        batch_size = self._get_batch_size(len(X_train))
        if batch_size is None:
            # for small datasets we perform full-batch gradient descent (one
            # weight update per epoch) for stability and simplicity
            train_loader = [(X_train, y_train)]
        else:
            # DataLoader handles shuffling and batching
            ds = TensorDataset(X_train, y_train)
            train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # define loss function appropriate for Poisson-distributed counts
        criterion = nn.PoissonNLLLoss(log_input=False)
        # Adam optimizer with optional L2 weight decay for regularization
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # bookkeeping for early stopping; track best validation loss seen so
        # far and corresponding model weights
        best_val = float("inf")
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        # main epoch loop: each iteration performs training on the entire
        # training set (possibly split into minibatches) and then evaluates on
        # the validation set
        for _ in range(self.epochs):
            model.train()  # set dropout/batchnorm to training mode if present
            epoch_loss = 0.0

            # iterate through minibatches
            for xb, yb in train_loader:
                optimiser.zero_grad()  # clear accumulated gradients
                preds = model(xb)  # forward pass: compute model outputs
                loss = criterion(preds, yb)  # compute Poisson NLL loss
                if self.l1_lambda > 0:
                    # include L1 penalty on weights if requested, scaled by lambda
                    loss = loss + self.l1_lambda * self._l1_penalty(model)
                loss.backward()  # backpropagate gradients
                optimiser.step()  # update weights using optimizer
                epoch_loss += loss.item()  # accumulate scalar loss

            # record average training loss for this epoch
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)

            # evaluate on validation set without tracking gradients
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()

            val_losses.append(val_loss)

            # early stopping logic: keep best model by validation loss
            if val_loss < best_val:
                best_val = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # restore best model state if available
        if best_state is not None:
            model.load_state_dict(best_state)

        return model, train_losses, val_losses


class TransferLearningTrainer(BaseTrainer):
    """Trainer for transfer-learning models that share parameters across cells.

    During training the network is evaluated on each cell's data in turn and
    gradients accumulate before the optimizer step.  The early stopping criterion
    monitors the total loss summed over all cells.
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
        super().__init__(lr, epochs, weight_decay, l1_lambda, batch_size)
        self.patience = patience

    def train(self, model, X_cells, Y_cells):
        """Train a transfer-learning ``model`` using per-cell datasets.

        Parameters
        ----------
        model : nn.Module
            Should accept inputs ``(X, cell_index)`` in its forward.
        X_cells : list of array-like
            Feature arrays for each cell.
        Y_cells : list of array-like
            Corresponding target arrays.

        Returns
        -------
        tuple
            ``(best_model, train_losses)`` where ``train_losses`` contains the
            summed loss at each epoch.
        """
        model.to(self.device)

        # convert lists of arrays to tensors on device
        X_cells = [
            torch.tensor(x, dtype=torch.float32, device=self.device) for x in X_cells
        ]
        Y_cells = [
            torch.tensor(y, dtype=torch.float32, device=self.device) for y in Y_cells
        ]

        criterion = nn.PoissonNLLLoss(log_input=False)
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_losses = []
        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        # epoch loop
        for _ in range(self.epochs):
            # switch to training mode and zero out gradients before epoch
            model.train()
            optimiser.zero_grad()

            total_loss = 0.0

            # loop over each cell's dataset; gradients from each cell accumulate
            # in the model parameters before the optimizer step at epoch end
            for ci, (Xc, Yc) in enumerate(zip(X_cells, Y_cells)):
                # determine batch size for this cell's data
                batch_size = self._get_batch_size(len(Xc))
                if batch_size is None:
                    # full batch for small cell-specific dataset
                    loader = [(Xc, Yc)]
                else:
                    ds = TensorDataset(Xc, Yc)
                    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

                # iterate through minibatches for this cell
                for xb, yb in loader:
                    preds = model(xb, ci)  # forward pass including cell index
                    loss = criterion(preds, yb)  # compute loss for this batch
                    if self.l1_lambda > 0:
                        # add sparsity regularization term, scaled by lambda
                        loss = loss + self.l1_lambda * self._l1_penalty(model)
                    total_loss += loss

            # after accumulating loss over all cells and batches, backpropagate
            total_loss.backward()
            optimiser.step()  # update shared parameters

            # record the combined loss (scalar) for the epoch
            train_losses.append(total_loss.item())

            # early stopping check based on total loss history
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        return model, train_losses
