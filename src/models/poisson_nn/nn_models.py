import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BasePoissonModel(nn.Module, ABC):
    """Minimal base class for Poisson neural nets.

    All concrete models inherit from ``nn.Module`` and this ABC to guarantee a
    common ``predict`` method, device tracking, and a consistent ``forward``
    signature.  This base is also used for type hints when writing training
    utilities that should accept any Poisson model.
    """

    def __init__(self):
        super().__init__()
        # record the device the model is currently residing on
        self.device = torch.device("cpu")

    def to(self, device):
        """Override ``nn.Module.to`` to capture the device for later use.

        This ensures that predictions are sent to the same device without
        requiring the caller to remember to move inputs explicitly.
        """
        self.device = device
        return super().to(device)

    @abstractmethod
    def forward(self, X, *args, **kwargs):
        """Concrete subclasses must implement standard forward pass."""
        pass

    def predict(self, X, *args, **kwargs):
        """Run a forward pass on ``X`` and return numpy predictions.

        This convenience wraps the model call with type conversion and
        ``torch.no_grad``.  Outputs are clipped to be strictly positive since
        Poisson rates must be non-negative.
        """
        # convert numpy input to tensor and place on current device
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_t, *args, **kwargs).cpu().numpy()

        return preds.clip(1e-8, None)


class PoissonNN(BasePoissonModel):
    """Fully-connected feedforward network for Poisson rate prediction.

    The architecture is defined by ``hidden_sizes``: a list of integers
    specifying the number of units in each hidden layer.  Each hidden layer is
    followed by a ``ReLU`` nonlinearity.  The final output layer projects to a
    single unit and applies ``Softplus`` to guarantee a positive output rate.
    """

    def __init__(self, n_features, hidden_sizes=[32, 16]):
        super().__init__()

        layers = []
        in_dim = n_features

        # build each hidden layer sequentially
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))  # linear transform
            layers.append(nn.ReLU())  # nonlinearity
            in_dim = h

        # final output layer mapping to single rate value per sample
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Softplus())  # ensures positive firing rate

        self.network = nn.Sequential(*layers)

    def forward(self, X):
        # forward through sequential network, squeeze to remove last dim
        return self.network(X).squeeze(-1)


class SharedHiddenPoissonNN(BasePoissonModel):
    """Multi-cell network with shared hidden representation and individual heads.

    This architecture is useful when the same input features are used to predict
    rates for multiple cells; a common feature extractor learns a shared
    embedding and then each cell has its own output head producing a rate.
    """

    def __init__(self, n_features, hidden_sizes, n_cells):
        super().__init__()

        # build shared hidden layers (feature extractor)
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)

        # create a separate small network for each cell that maps shared features
        # to a single output rate; Softplus again enforces positivity
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, 1), nn.Softplus()) for _ in range(n_cells)]
        )

    def forward(self, X, cell_idx):
        # pass input through shared layers then select correct head
        h = self.feature_extractor(X)
        return self.heads[cell_idx](h).squeeze(-1)
