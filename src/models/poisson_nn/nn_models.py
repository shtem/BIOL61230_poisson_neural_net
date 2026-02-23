import torch
import torch.nn as nn


class PoissonNN(nn.Module):
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
        """
        X: (batch_size, n_features)
        Returns: (batch_size,)
        """
        return self.network(X).squeeze(-1)

    def predict(self, X):
        """
        X: numpy array (n_samples, n_features)
        Returns: numpy array (n_samples,)
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self.forward(X_t).cpu().numpy()


class SharedHiddenPoissonNN(nn.Module):
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
        """
        X: (batch_size, n_features)
        cell_idx: int
        """
        h = self.feature_extractor(X)
        return self.heads[cell_idx](h).squeeze(-1)

    def predict(self, X, cell_idx):
        """
        X: numpy array (n_samples, n_features)
        cell_idx: int
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return self.forward(X_t, cell_idx).cpu().numpy()
