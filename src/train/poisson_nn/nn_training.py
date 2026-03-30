import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer(ABC):
    """
    Abstract base class for training Poisson neural network models.

    This class centralises common training configuration such as learning rate,
    number of epochs, weight decay, optional L1 regularisation, device
    selection, and batch-size heuristics. Concrete trainer subclasses implement
    the actual optimisation loop in ``train`` while reusing the shared utilities
    provided here.

    The trainer automatically selects a GPU if available (unless a device is
    explicitly provided) and exposes helper methods for computing L1 penalties
    and determining an appropriate batch size based on dataset size. Returning
    ``None`` from ``_get_batch_size`` signals that the caller should perform
    full-batch training without a DataLoader.

    This abstraction allows different training strategies (full-batch,
    mini-batch, curriculum learning, multi-cell training, etc.) to share a
    consistent interface and configuration structure.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimiser.
    epochs : int, optional
        Number of training epochs.
    weight_decay : float, optional
        L2 regularisation strength passed to the optimiser.
    l1_lambda : float, optional
        Coefficient for optional L1 regularisation applied manually via
        ``_l1_penalty``.
    batch_size : int or "auto", optional
        Batch size to use. If ``"auto"``, the trainer chooses full-batch for
        small datasets and 128 for larger ones.
    device : str or torch.device, optional
        Device on which to run training. If ``None``, the trainer selects GPU if
        available, otherwise CPU.

    Attributes
    ----------
    device : torch.device
        The device used for training and inference.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    weight_decay : float
        L2 regularisation strength.
    l1_lambda : float
        L1 regularisation coefficient.
    batch_size : int or "auto"
        Batch size configuration.

    Notes
    -----
    Subclasses must implement the ``train`` method, which should accept a model
    and training data, perform optimisation, and return the trained model along
    with any diagnostics (loss curves, metrics, etc.).
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

        # Automatically choose GPU if available unless the user overrides it
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _get_batch_size(self, n_samples):
        """
        Determine an appropriate batch size based on dataset size and user settings.

        If ``batch_size`` is ``"auto"``, the trainer uses:
            - full-batch (``None``) for small datasets (< 2000 samples)
            - mini-batch (128) for larger datasets

        Returning ``None`` signals that the caller should bypass DataLoader
        construction and train on the entire dataset in a single batch.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.

        Returns
        -------
        int or None
            Batch size to use, or ``None`` for full-batch training.
        """
        if self.batch_size == "auto":
            return None if n_samples < 2000 else 128
        return self.batch_size

    def _l1_penalty(self, model):
        """
        Compute the L1 penalty across all model parameters.

        This is used to implement optional sparsity regularisation by adding
        ``l1_lambda * _l1_penalty(model)`` to the loss during training.

        Parameters
        ----------
        model : nn.Module
            Model whose parameters will be penalised.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the sum of absolute values of all
            parameters.
        """
        return sum(p.abs().sum() for p in model.parameters())

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        Train a model.

        Subclasses must implement this method to define the optimisation loop.
        The method should accept a model and training data, perform gradient
        updates, and return the trained model along with any diagnostics
        (e.g., loss curves, validation metrics).

        Returns
        -------
        Any
            Typically the trained model and optional training diagnostics.
        """
        pass


class PoissonTrainer(BaseTrainer):
    """
    Trainer for single-cell Poisson neural network models.

    This class implements a standard training loop for Poisson regression models,
    including device management, batching, optimisation, and early stopping. It is
    designed for models inheriting from ``BasePoissonModel`` and provides a unified
    interface for fitting single-cell architectures such as MLPs, CNN/RNN
    extractors, and transfer-learning variants.

    Training is performed using Poisson negative log-likelihood loss, with L2
    regularisation handled through optimizer weight decay and optional L1
    regularisation applied manually. Early stopping monitors validation loss and
    automatically restores the best-performing model state.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimiser.
    epochs : int, optional
        Maximum number of training epochs.
    weight_decay : float, optional
        L2 regularisation strength passed to the optimiser.
    l1_lambda : float, optional
        Coefficient for optional L1 regularisation applied to model parameters.
    batch_size : int or "auto", optional
        Batch size for training. If ``"auto"``, the trainer selects full-batch
        training for small datasets and mini-batch training for larger ones.
    patience : int, optional
        Number of epochs without validation improvement before early stopping.

    Attributes
    ----------
    device : torch.device
        Device on which training is performed, inherited from ``BaseTrainer``.
    patience : int
        Early stopping patience threshold.

    Methods
    -------
    train(model, X_train, y_train, X_val, y_val)
        Fit the model using training data and monitor validation loss for early
        stopping. Returns the best model state and loss curves.

    Notes
    -----
    This trainer is intended for single-cell models. Multi-cell or shared-extractor
    architectures should use a corresponding multi-cell trainer subclass.
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
        """
        Fit a Poisson model on training data, using a validation set for early stopping.

        This method performs epoch-based optimisation with optional mini-batching.
        After each epoch, the model is evaluated on the validation set, and the best
        model state (lowest validation loss) is tracked. Training stops early if the
        validation loss does not improve for ``patience`` consecutive epochs.

        Parameters
        ----------
        model : nn.Module
            Poisson model implementing ``forward`` and ``preprocess``.
        X_train, y_train : array-like
            Training features and Poisson targets.
        X_val, y_val : array-like
            Validation features and targets used to monitor early stopping.

        Returns
        -------
        tuple
            ``(best_model, train_losses, val_losses)`` where:
                - ``best_model`` is the model restored to its best validation state
                - ``train_losses`` is a list of per-epoch training losses
                - ``val_losses`` is a list of per-epoch validation losses

        Notes
        -----
        The loss function is ``PoissonNLLLoss`` with ``log_input=False`` so the model
        is expected to output positive rate predictions directly (e.g., via Softplus).
        L1 regularisation is applied manually if ``l1_lambda > 0``.
        """
        # move model parameters to target device (CPU/GPU)
        model.to(self.device)

        # convert inputs to tensors and transfer to chosen device
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        # determine minibatch size
        batch_size = self._get_batch_size(len(X_train))
        if batch_size is None:
            loader = [(X_train, y_train)]
        else:
            ds = TensorDataset(X_train, y_train)
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

        # loss + optimiser
        criterion = nn.PoissonNLLLoss(log_input=False)
        # Adam optimizer with optional L2 weight decay for regularisation
        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # mixed precision scaler
        scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

        best_val = float("inf")
        best_state = None
        patience_counter = 0

        train_losses = []
        val_losses = []

        # main epoch loop: each iteration performs training on the entire
        # training set (possibly split into minibatches) and then evaluates on
        # the validation set
        for _ in range(self.epochs):
            model.train()
            epoch_loss = 0.0

            for xb, yb in loader:
                optimiser.zero_grad()

                # autocast for mixed precision
                with torch.amp.autocast(
                    device_type="cuda", enabled=(self.device.type == "cuda")
                ):
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    if self.l1_lambda > 0:
                        loss = loss + self.l1_lambda * self._l1_penalty(model)

                # scaled backward + step
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)
            train_losses.append(epoch_loss)

            # validation (no AMP needed)
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()

            val_losses.append(val_loss)

            # early stopping
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
    """
    Trainer for multi-cell transfer-learning Poisson models.

    This class implements a training loop for architectures that share parameters
    across neurons, such as shared-extractor or shared-representation models. During
    each epoch, the model is evaluated on every cell's dataset in turn, and the
    resulting gradients accumulate before a single optimiser step is taken. This
    ensures that shared parameters are updated using information from all cells.

    Training uses Poisson negative log-likelihood loss, with L2 regularisation
    handled through optimiser weight decay and optional L1 penalties applied
    manually. Early stopping monitors the total loss summed across all cells and
    restores the best-performing model state.

    Parameters
    ----------
    lr : float, optional
        Learning rate for the optimiser.
    epochs : int, optional
        Maximum number of training epochs.
    weight_decay : float, optional
        L2 regularisation strength passed to the optimiser.
    l1_lambda : float, optional
        Coefficient for optional L1 regularisation applied to model parameters.
    batch_size : int or "auto", optional
        Batch size for per-cell training. If ``"auto"``, the trainer selects
        full-batch training for small cell-specific datasets and mini-batch
        training for larger ones.
    patience : int, optional
        Number of epochs without improvement in total loss before early stopping.

    Attributes
    ----------
    device : torch.device
        Device on which training is performed, inherited from ``BaseTrainer``.
    patience : int
        Early stopping patience threshold.

    Methods
    -------
    train(model, X_cells, Y_cells)
        Fit the model using per-cell datasets, accumulating gradients across cells
        before each optimiser step. Returns the best model state and loss history.

    Notes
    -----
    This trainer is intended for models whose ``forward`` method accepts a cell
    index (e.g., ``model(x, cell_idx)``). It supports any architecture that shares
    parameters across neurons, including shared-extractor, shared-representation,
    and hybrid transfer-learning models.
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
        """
        Train a transfer-learning model using per-cell datasets.

        This method performs epoch-based optimisation where, during each epoch, the
        model is evaluated on every cell's dataset in turn. Gradients from all cells
        accumulate before a single optimiser step is taken, ensuring that shared
        parameters are updated using information from the entire population. The total
        loss summed across all cells is used for early stopping, and the best model
        state is restored at the end of training.

        Parameters
        ----------
        model : nn.Module
            Transfer-learning model whose ``forward`` method accepts inputs of the form
            ``model(X, cell_index)``.
        X_cells : list of array-like
            List of feature arrays, one per cell.
        Y_cells : list of array-like
            List of target arrays corresponding to ``X_cells``.

        Returns
        -------
        tuple
            ``(best_model, train_losses)`` where:
                - ``best_model`` is the model restored to its best validation state
                according to total loss across all cells.
                - ``train_losses`` is a list of per-epoch total losses (summed across
                all cells and batches).

        Notes
        -----
        Batch size is determined independently for each cell's dataset using the
        trainer's batch-size heuristic. L1 regularisation is applied manually if
        ``l1_lambda > 0``. This method is intended for models that share parameters
        across cells and require gradient accumulation across multiple datasets.
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

        # mixed precision scaler
        scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

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

            # loop over each cell's dataset
            for ci, (Xc, Yc) in enumerate(zip(X_cells, Y_cells)):

                batch_size = self._get_batch_size(len(Xc))
                if batch_size is None:
                    # full batch for small cell-specific dataset
                    loader = [(Xc, Yc)]
                else:
                    ds = TensorDataset(Xc, Yc)
                    loader = DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                    )

                # iterate through minibatches for this cell
                for xb, yb in loader:

                    # autocast for mixed precision
                    with torch.amp.autocast(
                        device_type="cuda", enabled=(self.device.type == "cuda")
                    ):
                        preds = model(xb, ci)
                        loss = criterion(preds, yb)
                        if self.l1_lambda > 0:
                            loss = loss + self.l1_lambda * self._l1_penalty(model)

                    # accumulate loss across cells
                    total_loss += loss

            # scaled backward + step (after all cells)
            scaler.scale(total_loss).backward()
            scaler.step(optimiser)
            scaler.update()

            train_losses.append(total_loss.item())

            # early stopping
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
