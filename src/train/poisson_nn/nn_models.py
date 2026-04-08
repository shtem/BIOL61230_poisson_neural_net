import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.train.utils import _to_tensor

# ------------------------------------------------------------------
# Base classes and utilities


class BasePoissonModel(nn.Module, ABC):
    """
    Abstract base class for Poisson neural network models.

    This class provides a unified interface for all Poisson models used in the
    codebase, including per-cell MLPs, CNN/RNN extractors, and multi-cell
    transfer-learning architectures. It standardises preprocessing, device
    handling, and prediction, ensuring that heterogeneous model types can be
    trained and evaluated through a common API.

    The ``input_type`` attribute determines how raw inputs are reshaped before
    being passed through the model. Flat models operate directly on feature
    vectors; sequence models receive inputs as ``(batch, features, 1)``; and
    image models assume a square spatial layout and reshape accordingly.

    The class also overrides ``nn.Module.to`` to track the active device,
    ensuring that preprocessing and prediction automatically move inputs to the
    correct device without requiring manual handling by the caller.

    Parameters
    ----------
    input_type : str, optional
        One of ``"flat"``, ``"sequence"``, or ``"image"``, controlling how
        inputs are reshaped in ``preprocess``.

    Attributes
    ----------
    device : torch.device
        The device (CPU or GPU) to which the model and inputs are moved.
    input_type : str
        Declares the expected input format for preprocessing.

    Methods
    -------
    preprocess(X)
        Converts numpy arrays to tensors, moves data to the correct device,
        and reshapes inputs according to ``input_type``.
    forward(X, *args, **kwargs)
        Abstract method defining the model's forward computation.
    predict(X, *args, **kwargs)
        Convenience wrapper for inference that handles preprocessing, disables
        gradient tracking, and returns strictly positive Poisson rate estimates.

    Notes
    -----
    Subclasses must implement ``forward`` and may override ``preprocess`` for
    custom input handling (e.g., CNN/RNN extractors). The ``predict`` method is
    intended for evaluation and should not be used during training.
    """

    def __init__(self, input_type="flat"):
        super().__init__()
        self.device = torch.device("cpu")
        self.input_type = input_type

    def to(self, device):
        """
        Move the model to a given device and record the active device.

        This overrides ``nn.Module.to`` so that ``preprocess`` automatically
        moves inputs to the correct device without requiring manual handling.

        Parameters
        ----------
        device : str or torch.device
            Target device for model parameters and inputs.

        Returns
        -------
        nn.Module
            The model on the specified device.
        """
        self.device = device
        return super().to(device)

    def preprocess(self, X):
        """
        Convert input data to a float32 tensor on the correct device and reshape
        it according to ``input_type``.

        This method handles:
            - numpy → tensor conversion
            - cloning/detaching user-provided tensors
            - device placement
            - FakeTensor/ProxyTensor detection for torchview tracing
            - reshaping for flat, sequence, or image models

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input data to be preprocessed.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor ready for the model's forward pass.
        """
        # Detect FakeTensor / ProxyTensor used by torchview tracing
        is_fake = isinstance(X, torch.Tensor) and type(X).__name__ in (
            "FakeTensor",
            "ProxyTensor",
        )

        if is_fake:
            X_t = X
        else:
            # Convert numpy → tensor and move to correct device
            X_t = _to_tensor(X, self.device)

        # Reshape based on declared input type
        if self.input_type == "flat":
            return X_t

        if self.input_type == "sequence":
            return X_t.unsqueeze(1)

        if self.input_type == "image":
            H = int(X_t.shape[1] ** 0.5)
            return X_t.view(-1, 1, H, H)

        return X_t

    @abstractmethod
    def forward(self, X, *args, **kwargs):
        """
        Define the model's forward computation.

        Subclasses must implement this method to specify how preprocessed inputs
        are transformed into Poisson rate predictions.

        Parameters
        ----------
        X : torch.Tensor
            Preprocessed input tensor.

        Returns
        -------
        torch.Tensor
            Model output before conversion to numpy.
        """
        pass

    def predict(self, X, *args, **kwargs):
        """
        Run a forward pass in evaluation mode and return numpy predictions.

        This method:
            - preprocesses inputs
            - disables gradient tracking
            - moves outputs to CPU
            - converts to numpy
            - clips predictions to be strictly positive (Poisson constraint)

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input data for prediction.

        Returns
        -------
        numpy.ndarray
            Positive Poisson rate predictions.
        """
        X_t = self.preprocess(X)
        self.eval()
        with torch.no_grad():
            preds = self.forward(X_t, *args, **kwargs).cpu().numpy()
        return preds.clip(1e-8, None)


class BaseExtractor(nn.Module):
    """
    Minimal base class for feature extractors used in Poisson neural networks.

    Concrete CNN, RNN, and MLP extractors inherit from this class to guarantee a
    consistent interface and preprocessing behaviour.  Each extractor declares
    an ``input_type`` (``"flat"``, ``"sequence"``, or ``"image"``) and must set
    ``self.out_dim`` to the dimensionality of the feature embedding it produces.

    The base class provides a standard ``preprocess`` method that converts
    numpy arrays to tensors, moves inputs to the correct device, and reshapes
    them according to the declared ``input_type``.  Sequence extractors receive
    inputs as ``(batch, features, 1)`` to support temporal or convolutional
    processing, while flat extractors operate directly on ``(batch, features)``.

    This class is used by higher-level Poisson models to ensure that arbitrary
    extractors can be plugged into shared or per-cell architectures without
    requiring special-case handling.

    Attributes
    ----------
    input_type : str
        One of ``"flat"``, ``"sequence"``, or ``"image"``, determining how
        inputs are reshaped before being passed to the extractor.
    out_dim : int
        Dimensionality of the output embedding produced by the extractor.
        Must be set by subclasses.

    Notes
    -----
    Subclasses must implement a ``forward`` method that accepts a preprocessed
    tensor and returns an embedding of shape ``(batch, out_dim)``.
    """

    def __init__(self, input_type="flat"):
        super().__init__()
        self.input_type = input_type
        self.out_dim = None  # set by subclasses

    def preprocess(self, X):
        """
        Convert input data to a float32 tensor on the correct device and reshape it
        according to the extractor's declared ``input_type``.

        This method handles:
            - numpy → tensor conversion
            - cloning/detaching user-provided tensors
            - device placement
            - reshaping for sequence extractors (``(batch, features, 1)``)

        Sequence extractors treat the feature vector as a temporal or ordered
        sequence, enabling CNN/RNN modules to operate over the feature dimension.

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input data to be preprocessed.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor ready for the extractor's forward pass.
        """
        # Determine device from tensor or default to CPU
        device = X.device if torch.is_tensor(X) else "cpu"

        # Convert numpy → tensor or clone existing tensor
        X = _to_tensor(X, device)

        # Sequence extractors expect (batch, features, 1)
        if self.input_type == "sequence":
            # Treat features as a sequence of length n_features with 1 channel
            return X.unsqueeze(-1)  # (batch, features, 1)
        return X


# ------------------------------------------------------------------
# Concrete extractor implementations


class CNNExtractor(BaseExtractor):
    """
    One-dimensional convolutional feature extractor for Poisson neural networks.

    This extractor treats the input feature vector as a short sequence of length
    ``n_features`` with a single channel, enabling convolutional filters to learn
    local patterns or interactions between neighbouring features. The depth and
    width of the convolutional stack are controlled by ``hidden_channels``, which
    may be either:

    - an integer (e.g. ``32``), producing a single Conv1d block, or
    - a sequence of integers (e.g. ``(64, 32)``), producing a multi-layer stack
    where each entry specifies the number of output channels for that layer.

    Each convolutional block consists of:

        Conv1d → ReLU → BatchNorm1d → Dropout

    After the convolutional stack, global average pooling collapses the sequence
    dimension, yielding a fixed-dimensional embedding. A small adapter MLP is then
    applied to stabilise optimisation and improve transfer-learning performance.

    The final output is a shared embedding of dimension ``hidden_channels[-1]``,
    suitable for use in multi-cell Poisson models or as input to per-cell heads.

    Parameters
    ----------
    n_features : int
        Number of input features before sequence reshaping.
    hidden_channels : int or sequence of int
        Channel sizes for each convolutional block. If an integer is provided,
        a single block is constructed. If a list/tuple is provided, one block is
        created per entry.
    kernel : int, optional
        Width of the convolutional kernel. Default is 3.
    dropout : float, optional
        Dropout probability applied after each convolutional block. Default is 0.1.

    Attributes
    ----------
    out_dim : int
        Dimensionality of the output embedding (equal to the last entry of
        ``hidden_channels``).
    conv : nn.Sequential
        Stack of convolutional processing layers.
    adapter : nn.Sequential
        Small MLP applied after global pooling to refine the embedding.

    Notes
    -----
    Inputs are reshaped by ``BaseExtractor`` to ``(batch, features, 1)`` and then
    transposed to ``(batch, 1, features)`` before convolution.
    """

    def __init__(self, n_features, hidden_channels, kernel=3, dropout=0.1):
        super().__init__(input_type="sequence")

        # Allow int or tuple/list
        if isinstance(hidden_channels, int):
            hidden_channels = (hidden_channels,)
        self.hidden_channels = tuple(hidden_channels)

        layers = []
        in_channels = 1

        for out_channels in self.hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.conv = nn.Sequential(*layers)

        # Output dimension is the last channel count
        self.out_dim = self.hidden_channels[-1]

        # Optional adapter MLP (keeps things stable for TL)
        self.adapter = nn.Sequential(
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )

    def forward(self, X):
        X = self.preprocess(X)  # (batch, features, 1)
        X = X.transpose(1, 2)  # (batch, 1, features)
        h = self.conv(X)  # (batch, C, features)
        h = h.mean(dim=2)  # global average pooling → (batch, C)
        h = self.adapter(h)  # (batch, C)
        return h


class RNNExtractor(BaseExtractor):
    """
    Gated recurrent feature extractor for Poisson neural networks.

    This extractor interprets the input feature vector as a sequence of length
    ``n_features`` with a single feature channel and processes it using a stack of
    GRU (Gated Recurrent Unit) layers. The depth and width of the recurrent stack
    are controlled by ``hidden_sizes``, which may be either:

    - an integer (e.g. ``32``), producing a single GRU layer with hidden size 32, or
    - a sequence of integers (e.g. ``(64, 32)``), producing a multi-layer GRU stack
      where each entry specifies the hidden size of one GRU layer.

    Each GRU layer receives the full sequence and outputs a sequence of hidden
    states. The output of one GRU is fed into the next, allowing the model to learn
    hierarchical or multi-scale temporal/structural patterns across the feature
    dimension. A ``LayerNorm`` layer is applied to the final GRU output to improve
    optimisation stability, particularly in transfer-learning settings.

    The final embedding is the last hidden state of the final GRU layer, yielding a
    fixed-dimensional representation suitable for downstream Poisson prediction or
    as the shared representation in multi-cell models.

    Parameters
    ----------
    n_features : int
        Number of input features before sequence reshaping.
    hidden_sizes : int or sequence of int
        Hidden sizes for each GRU layer. If an integer is provided, a single GRU
        layer is constructed. If a list/tuple is provided, one GRU layer is created
        per entry.
    dropout : float, optional
        Dropout probability applied between GRU layers. Default is 0.1.

    Attributes
    ----------
    out_dim : int
        Dimensionality of the output embedding (equal to the last entry of
        ``hidden_sizes``).
    rnns : nn.ModuleList
        Stack of GRU layers, each processing the full input sequence.
    norm : nn.LayerNorm
        Normalisation applied to the final GRU output sequence.

    Notes
    -----
    Inputs are reshaped by ``BaseExtractor`` to ``(batch, features, 1)``, so each
    GRU receives a sequence of length ``n_features`` with ``input_size = 1`` for the
    first layer and ``input_size = hidden_sizes[i]`` for subsequent layers.
    The final embedding is ``out[:, -1, :]`` from the last GRU layer.
    """

    def __init__(self, n_features, hidden_sizes, dropout=0.1):
        super().__init__(input_type="sequence")

        # Allow int or tuple/list
        if isinstance(hidden_sizes, int):
            hidden_sizes = (hidden_sizes,)
        self.hidden_sizes = tuple(hidden_sizes)

        self.rnns = nn.ModuleList()
        input_dim = 1  # first GRU sees (batch, seq_len, 1)

        for h in self.hidden_sizes:
            self.rnns.append(
                nn.GRU(
                    input_size=input_dim,
                    hidden_size=h,
                    num_layers=1,
                    batch_first=True,
                    dropout=dropout,
                )
            )
            input_dim = h  # next GRU receives previous hidden size

        self.out_dim = self.hidden_sizes[-1]
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, X):
        X = self.preprocess(X)  # (batch, features, 1)

        out = X
        for gru in self.rnns:
            out, _ = gru(out)  # (batch, seq_len, hidden)

        out = self.norm(out)
        return out[:, -1, :]  # final hidden state


# ------------------------------------------------------------------
# Concrete Poisson model implementations


class PoissonNN(BasePoissonModel):
    """
    Fully connected feedforward network for single-cell Poisson rate prediction.

    This model implements a standard multilayer perceptron (MLP) for predicting
    Poisson firing rates from a feature vector. The architecture consists of a
    configurable stack of linear layers with ReLU nonlinearities, followed by a
    final ``Softplus`` output unit to ensure strictly positive rate estimates.
    Alternatively, a custom feature extractor (e.g., CNN or RNN) may be supplied
    via the ``extractor`` argument, in which case the MLP layers are skipped and
    the model uses the extractor's embedding as input to the output head.

    This class serves as the baseline single-cell model and is also used as the
    per-cell component in transfer-learning architectures.

    Parameters
    ----------
    n_features : int
        Number of input features before preprocessing.
    hidden_sizes : list of int, optional
        Sizes of the hidden layers in the MLP. Ignored if ``extractor`` is
        provided.
    extractor : BaseExtractor or nn.Module, optional
        Custom feature extractor whose output embedding is fed into the final
        Poisson head.

    Attributes
    ----------
    extractor : nn.Module
        Either the constructed MLP or the user-provided extractor.
    head : nn.Sequential
        Final linear layer followed by ``Softplus`` to produce positive rates.

    Notes
    -----
    Inputs are preprocessed by ``BasePoissonModel`` according to ``input_type``.
    The output is a vector of shape ``(batch,)`` containing Poisson rate
    predictions for a single neuron.
    """

    def __init__(self, n_features, hidden_sizes, extractor=None):
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
        X = self.preprocess(X)
        h = self.extractor(X)
        return self.head(h).squeeze(-1)


class DeepSharedShallowHeadPoissonNN(BasePoissonModel):
    """
    Multi-cell Poisson model with a deep shared hidden representation and
    shallow cell-specific output heads.

    This architecture is designed for population modelling where multiple cells
    are recorded from the same stimulus or feature set. A shared feature
    extractor learns a common embedding across all cells, capturing population-
    level structure. Each cell then receives its own lightweight output head
    (a linear layer followed by ``Softplus``) that maps the shared embedding to
    a Poisson firing rate.

    This model corresponds to the classical “shared representation + linear
    readout” transfer-learning setup and provides a strong baseline for
    evaluating how much structure can be shared across neurons.

    Parameters
    ----------
    n_features : int
        Number of input features before preprocessing.
    hidden_sizes : list of int
        Sizes of the hidden layers in the shared MLP extractor. Ignored if
        ``shared_extractor`` is provided.
    n_cells : int
        Number of neurons to model.
    shared_extractor : BaseExtractor or nn.Module, optional
        Custom shared feature extractor. If provided, it replaces the MLP
        defined by ``hidden_sizes``.

    Attributes
    ----------
    feature_extractor : nn.Module
        Shared module that produces a population-level embedding.
    heads : nn.ModuleList
        List of per-cell output heads, each mapping the shared embedding to a
        single Poisson rate.

    Notes
    -----
    This model enforces strong sharing: all nonlinear transformations occur in
    the shared extractor, while per-cell variability is captured only through
    the final linear readout. More flexible variants include
    ``DeepSharedDeepHeadPoissonNN`` and ``ShallowSharedDeepHeadPoissonNN``.
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


class DeepSharedDeepHeadPoissonNN(BasePoissonModel):
    """
    Multi-cell Poisson model with a deep shared feature extractor and
    deep cell-specific nonlinear heads.

    This architecture shares only the low-level representation across cells,
    allowing the model to capture population-level structure while still giving
    each neuron a flexible, multi-layer nonlinear head. This design is well
    suited to neural populations where cells respond to similar basic stimulus
    features but require distinct nonlinear transformations to express their
    individual tuning curves.

    The shared module may be a user-provided extractor (e.g., CNN or RNN) or a
    small MLP defined by ``shared_sizes``. Each cell receives its own MLP head
    defined by ``head_sizes``, ending in a ``Softplus`` unit to ensure positive
    Poisson rate predictions.

    Parameters
    ----------
    n_features : int
        Number of input features before preprocessing.
    shared_sizes : list of int
        Hidden layer sizes for the shared low-level MLP extractor. Ignored if
        ``shared_extractor`` is provided.
    head_sizes : list of int
        Hidden layer sizes for each cell-specific nonlinear head.
    n_cells : int
        Number of neurons to model.
    shared_extractor : BaseExtractor or nn.Module, optional
        Custom shared extractor. If provided, it replaces the MLP defined by
        ``shared_sizes``.

    Attributes
    ----------
    shared : nn.Module
        Shared low-level feature extractor producing a population-level
        embedding.
    heads : nn.ModuleList
        List of per-cell nonlinear MLP heads mapping the shared embedding to a
        Poisson rate.

    Notes
    -----
    This model provides a more flexible alternative to
    ``DeepSharedShallowHeadPoissonNN`` by allowing each cell to learn its own nonlinear
    transformation on top of the shared representation. It is often a strong
    choice when cells share coarse structure but differ in tuning complexity.
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


class ShallowSharedDeepHeadPoissonNN(BasePoissonModel):
    """
    Multi-cell Poisson model that shares only the first layer, with deeper
    layers unique to each cell.

    This architecture implements minimal sharing: a single shared linear
    transformation captures basic structure across the population, while each
    neuron receives its own deeper MLP head to model cell-specific nonlinear
    tuning. This design is appropriate when cells exhibit limited shared
    structure or when over-sharing harms performance.

    The shared component may be a user-provided extractor or a simple
    ``Linear → ReLU`` block. Each cell's head is a multi-layer MLP defined by
    ``head_sizes``, ending in a ``Softplus`` output to ensure positive Poisson
    rates.

    Parameters
    ----------
    n_features : int
        Number of input features before preprocessing.
    shared_dim : int
        Dimensionality of the shared first-layer embedding.
    head_sizes : list of int
        Hidden layer sizes for each cell-specific MLP head.
    n_cells : int
        Number of neurons to model.
    shared_extractor : BaseExtractor or nn.Module, optional
        Custom shared extractor. If provided, it replaces the default first
        layer and determines ``shared_dim`` automatically.

    Attributes
    ----------
    shared : nn.Module
        Shared first-layer transformation applied to all cells.
    heads : nn.ModuleList
        List of per-cell MLP heads mapping the shared embedding to a Poisson
        rate.

    Notes
    -----
    This model provides the least restrictive form of sharing among the
    transfer-learning architectures. It is particularly useful when cells share
    only coarse structure and benefit from highly flexible, independent
    nonlinear heads.
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
