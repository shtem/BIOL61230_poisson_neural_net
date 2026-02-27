from pathlib import Path
import pickle
import matplotlib.pyplot as plt


def save_model(model, model_name: str, base_dir: Path = Path("data/models")):
    """Persist a model object to disk.

    The model is serialized with :mod:`pickle` and stored under
    ``{base_dir}/{model_name}.pkl``.  Directories are created if necessary.

    Parameters
    ----------
    model : any
        Object to pickle (typically a fitted estimator).
    model_name : str
        Identifier to use for the filename.
    base_dir : Path, optional
        Root directory for model files (default ``data/models``).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(model_name: str, base_dir: Path = Path("data/models")):
    """Load a pickled model previously saved with :func:`save_model`.

    Parameters
    ----------
    model_name : str
        Identifier used when the model was saved; this is appended with
        ``.pkl`` to form the filename.
    base_dir : Path, optional
        Directory where models are stored (default ``data/models``).

    Returns
    -------
    any
        The unpickled object (typically a fitted estimator) that was stored
        under the given name.
    """
    path = Path(base_dir) / f"{model_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def save_plot(
    fig: plt.Figure,
    model_name: str,
    filename: str,
    base_dir: Path = Path("data/results"),
):
    """Save a Matplotlib figure to a results subdirectory.

    Parameters
    ----------
    fig : plt.Figure
        Figure to write.
    model_name : str
        Subfolder within ``base_dir`` where the figure will be stored.
    filename : str
        Name of the file (including extension, e.g. ``"train.png"``).
    base_dir : Path, optional
        Root results directory (default ``data/results``).

    Returns
    -------
    Path
        Full path to the saved file.
    """
    out_dir = Path(base_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    return path
