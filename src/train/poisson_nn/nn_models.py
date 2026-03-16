import torch
import torch.nn as nn
from abc import ABC, abstractmethod

# ------------------------------------------------------------------
# Base classes and utilities


class BasePoissonModel(nn.Module, ABC):
    """Minimal base class for Poisson neural nets.

    All concrete models inherit from ``nn.Module`` and this ABC to guarantee a
    common ``predict`` method, device tracking, and a consistent ``forward``
    signature.  This base is also used for type hints when writing training
    utilities that should accept any Poisson model.
    """

    def __init__(self, input_type="flat"):
        super().__init__()
        self.device = torch.device("cpu")
        self.input_type = input_type  # "flat", "sequence", "image"

    def to(self, device):
        """Override ``nn.Module.to`` to capture the device for later use.

        This ensures that predictions are sent to the same device without
        requiring the caller to remember to move inputs explicitly.
        """
        self.device = device
        return super().to(device)

    def preprocess(self, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)

        if self.input_type == "flat":
            return X_t

        if self.input_type == "sequence":
            # reshape (batch, features) → (batch, seq_len=1, features)
            return X_t.unsqueeze(1)

        if self.input_type == "image":
            # reshape (batch, features) → (batch, channels=1, H, W)
            H = int(X_t.shape[1] ** 0.5)
            return X_t.view(-1, 1, H, H)

        return X_t

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
        X_t = self.preprocess(X)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_t, *args, **kwargs).cpu().numpy()
        return preds.clip(1e-8, None)


class BaseExtractor(nn.Module):
    """
    Standardised interface for CNN/RNN/MLP extractors.
    Must define:
      - self.out_dim
      - forward(X)
      - input_type ("flat", "sequence")
    """

    def __init__(self, input_type="flat"):
        super().__init__()
        self.input_type = input_type
        self.out_dim = None  # set by subclasses

    def preprocess(self, X):
        X = torch.tensor(
            X, dtype=torch.float32, device=X.device if torch.is_tensor(X) else "cpu"
        )
        if self.input_type == "sequence":
            return X.unsqueeze(1)  # (batch, seq=1, features)
        return X


# ------------------------------------------------------------------
# Concrete extractor implementations


class CNNExtractor(BaseExtractor):
    def __init__(self, n_features, channels=16, kernel=3, num_layers=2):
        super().__init__(input_type="sequence")

        layers = []
        in_channels = 1

        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, channels, kernel, padding=kernel // 2))
            layers.append(nn.ReLU())
            in_channels = channels

        self.conv = nn.Sequential(*layers)
        self.out_dim = channels  # after global pooling

    def forward(self, X):
        X = self.preprocess(X)  # (batch, 1, features)
        h = self.conv(X)  # (batch, channels, features)
        return h.mean(dim=2)  # global average pooling → (batch, channels)


class RNNExtractor(BaseExtractor):
    def __init__(self, n_features, hidden_dim=32, num_layers=1):
        super().__init__(input_type="sequence")

        self.rnn = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.out_dim = hidden_dim

    def forward(self, X):
        X = self.preprocess(X)  # (batch, seq=1, features)
        out, _ = self.rnn(X)
        return out[:, -1, :]  # last timestep → (batch, hidden_dim)


# ------------------------------------------------------------------
# Concrete Poisson model implementations


class PoissonNN(BasePoissonModel):
    """Fully-connected feedforward network for Poisson rate prediction.

    The architecture is defined by ``hidden_sizes``: a list of integers
    specifying the number of units in each hidden layer.  Each hidden layer is
    followed by a ``ReLU`` nonlinearity.  The final output layer projects to a
    single unit and applies ``Softplus`` to guarantee a positive output rate.
    """

    def __init__(self, n_features, hidden_sizes=[32, 16], extractor=None):
        super().__init__()

        # build each hidden layer sequentially or use provided extractor
        if extractor is not None:
            self.extractor = extractor
            in_dim = extractor.out_dim
        else:
            layers = []
            in_dim = n_features
            for h in hidden_sizes:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            self.extractor = nn.Sequential(*layers)

        # final output layer mapping to single rate value per sample
        self.head = nn.Sequential(
            nn.Linear(in_dim, 1), nn.Softplus()  # ensures positive firing rate
        )

    def forward(self, X):
        # forward through sequential network, squeeze to remove last dim
        h = self.extractor(X)
        return self.head(h).squeeze(-1)


class SharedHiddenPoissonNN(BasePoissonModel):
    """Multi-cell network with shared hidden representation and individual heads.

    This architecture is useful when the same input features are used to predict
    rates for multiple cells; a common feature extractor learns a shared
    embedding and then each cell has its own output head producing a rate.
    """

    def __init__(self, n_features, hidden_sizes, n_cells, shared_extractor=None):
        super().__init__()
        self.n_cells = n_cells

        # Shared low-level feature extractor (custom or original)
        if shared_extractor is not None:
            self.feature_extractor = shared_extractor
            out_dim = shared_extractor.out_dim
        else:
            # Original MLP extractor
            layers = []
            in_dim = n_features
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.feature_extractor = nn.Sequential(*layers)
            out_dim = hidden_sizes[-1]

        # create a separate small network for each cell that maps shared features
        # to a single output rate; Softplus again enforces positivity
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(out_dim, 1), nn.Softplus())
                for _ in range(n_cells)
            ]
        )

    def forward(self, X, cell_idx):
        # pass input through shared layers then select correct head
        h = self.feature_extractor(X)
        return self.heads[cell_idx](h).squeeze(-1)


class SharedNonlinearHeadsPoissonNN(BasePoissonModel):
    """Multi-cell network with a shallow shared feature extractor and nonlinear per-cell heads.

    This architecture shares only low-level representations across cells, while each cell
    receives its own multi-layer nonlinear head. It is useful when cells respond to similar
    basic features but require distinct nonlinear transformations to capture their tuning.
    """

    def __init__(
        self, n_features, shared_sizes, head_sizes, n_cells, shared_extractor=None
    ):
        super().__init__()
        self.n_cells = n_cells

        # Shared low-level feature extractor (custom or original)
        if shared_extractor is not None:
            self.shared = shared_extractor
            shared_dim = shared_extractor.out_dim
        else:
            layers = []
            in_dim = n_features
            for h in shared_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            self.shared = nn.Sequential(*layers)
            shared_dim = shared_sizes[-1]

        # Per-cell nonlinear heads
        self.heads = nn.ModuleList()
        for _ in range(n_cells):
            layers = []
            in_dim = shared_dim
            for h in head_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Softplus())
            self.heads.append(nn.Sequential(*layers))

    def forward(self, X, cell_idx):
        h = self.shared(X)
        return self.heads[cell_idx](h).squeeze(-1)


class SharedFirstLayerPoissonNN(BasePoissonModel):
    """Multi-cell network that shares only the first layer, with deeper layers unique to each cell.

    This architecture provides minimal sharing: a common first transformation captures basic
    structure across cells, while the remaining layers are cell-specific. It is suited to
    populations where cells exhibit limited shared structure and benefit from flexible,
    independent heads.
    """

    def __init__(
        self, n_features, shared_dim, head_sizes, n_cells, shared_extractor=None
    ):
        super().__init__()
        self.n_cells = n_cells

        # Shared low-level feature extractor (custom or original)
        if shared_extractor is not None:
            self.shared = shared_extractor
            shared_dim = shared_extractor.out_dim
        else:
            self.shared = nn.Sequential(nn.Linear(n_features, shared_dim), nn.ReLU())

        # Per-cell MLP heads
        self.heads = nn.ModuleList()
        for _ in range(n_cells):
            layers = []
            in_dim = shared_dim
            for h in head_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Softplus())
            self.heads.append(nn.Sequential(*layers))

    def forward(self, X, cell_idx):
        h = self.shared(X)
        return self.heads[cell_idx](h).squeeze(-1)
