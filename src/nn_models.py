import torch
import torch.nn as nn
from src.model_utils import fit_model_per_cell
import numpy as np


class PoissonNN:
    """
    Simple Poisson Neural Network
    """

    def __init__(
        self, n_features, hidden_sizes=[32, 16], lr=1e-3, epochs=100, weight_decay=1e-4
    ):
        """
        Instatiate Poisson Neural Network.

        :param n_features: Number of covaraites/features (input dimension)
        :param hidden_sizes: List of hidden layer sizes
        :param lr: Learning rate for adam optimiser
        :param epochs: How many passes through the training data
        :param weight_decay: L2 regularisation strength
        """
        self.n_features = n_features
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.model = self._build_model()

    def _build_model(self):
        """
        Build Poisson neural network model using Linear layers and reLU activation function.

        :return: Sequential Neural Network model
        """
        layers = []
        in_dim = self.n_features

        for h in self.hidden_sizes:
            layers.append(
                nn.Linear(in_dim, h)
            )  # fully connected layer, input dim -> out dim
            layers.append(
                nn.ReLU()
            )  # non linearity to allow for complex relationships between covariates and firing rate
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # output layer
        layers.append(nn.Softplus())  # ensures positive firing rate

        return nn.Sequential(*layers)  # chain all layers together

    def fit(self, X, y):
        """
        Train Poisson Neural Network model on given data.

        :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
        :param y: Array of shape (n_time_bins,) containing the target values (spike counts)

        :return: Model object (self)
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        criterion = nn.PoissonNLLLoss(log_input=False)
        optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # track best model state based on training loss to avoid overfitting
        best_loss = float("inf")
        best_state = None

        for _ in range(self.epochs):
            self.model.train()  # training mode
            optimiser.zero_grad()  # clear previous gradients
            y_pred = self.model(X).squeeze(
                -1
            )  # get predicted firing rates (forward pass), remove last dimension
            loss = criterion(y_pred, y)  # compute Poisson negative log likelihood loss
            loss.backward()  # compute gradients of loss w.r.t. model parameters
            optimiser.step()  # update parameters using gradients

            # if this epoch has the lowest training loss so far, save the model state
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = self.model.state_dict()

        # after training, load the best model state to ensure we have the best performing model
        self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        """
        Predict target values (spike counts) based on given input features (covariates)

        :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)

        :return: Predicted spike counts
        """
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()  # evaluation mode
        with torch.no_grad():  # no gradients needed for prediction
            # forward pass to get predicted firing rates, remove last dimension and convert to numpy array
            y_pred = self.model(X).squeeze(-1).numpy()
        return y_pred


def fit_poisson_nn(
    X, Y, cell_ids, train_frac=0.7, val_frac=0.15, scaler=None, **kwargs
):
    """
    Fit a Poisson Neural Network model for each cell.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the target values (spike counts)
    :param cell_ids: Array of all cell IDs
    :param train_frac: Fraction of samples to use for training (default 0.7)
    :param val_frac: Fraction of samples to use for validation (default 0.15)
    :param scaler: Scaler for input features
    :param kwargs: Additional keyword arguments to be passed to the PoissonNN constructor

    :return: Dictionary containing fitted models, coefficients, and performance metrics for each cell
    """
    # number of features is the number of rows in X
    # which is the number of covariates
    n_features = X.shape[0]

    return fit_model_per_cell(
        X,
        Y,
        cell_ids,
        model_class=lambda **kw: PoissonNN(n_features, **kw),
        model_kwargs=kwargs,
        train_frac=train_frac,
        val_frac=val_frac,
        scaler=scaler,
    )
