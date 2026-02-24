import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePoissonModel(nn.Module, ABC):
    """
    Abstract base class for all Poisson neural models.
    Ensures consistent API across MLP, CNN, RNN, and TL models.
    """

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")

    def to(self, device):
        """Move model to CPU or GPU and remember the device."""
        self.device = device
        return super().to(device)

    @abstractmethod
    def forward(self, X, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        """
        Default predict method for all Poisson models.
        Converts numpy → torch → numpy and ensures positivity.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_t, *args, **kwargs).cpu().numpy()

        return preds.clip(1e-8, None)


class PoissonNN(BasePoissonModel):
    """
    Standard feedforward Poisson neural network.
    """

    def __init__(self, n_features, hidden_sizes=[32, 16]):
        super().__init__()

        layers = []
        in_dim = n_features

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # ensures positive firing rate

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(-1)


class SharedHiddenPoissonNN(BasePoissonModel):
    """
    Shared hidden layers across cells, separate output heads per cell.
    """

    def __init__(self, n_features, hidden_sizes, n_cells):
        super().__init__()

        # Shared feature extractor
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # Per-cell output heads
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, 1), nn.Softplus()) for _ in range(n_cells)]
        )

    def forward(self, X, cell_idx):
        h = self.feature_extractor(X)
        return self.heads[cell_idx](h).squeeze(-1)
