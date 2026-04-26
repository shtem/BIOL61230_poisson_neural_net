import os
import mat73
import numpy as np
from scipy import io
from sklearn.impute import SimpleImputer


def load_data(path, filenames=None):
    """
    Load and concatenate multiple MATLAB data files.

    Each file is expected to contain arrays ``X``, ``y``, ``cell_ids`` and
    ``rec_id``.  To avoid identifier collisions across files we offset the
    cell and recording IDs before concatenation.

    Parameters
    ----------
    path : str
        Directory path containing the .mat files.
    filenames : list of str
        Names of files (not full paths) to load.

    Returns
    -------
    tuple
        ``(X_all, Y_all, cell_ids_all, rec_ids_all)`` where ``X_all`` has shape
        ``(n_features, total_time)`` and the others are 1-d arrays of length
        ``total_time``.
    """
    # -----------------------------
    # Case 1: Real data (single file)
    # -----------------------------
    if filenames is None:
        data = io.loadmat(path)  # real data uses scipy.io.loadmat

        X = data["X"].astype(float)  # shape (n_bins, n_features)
        Y = data["y"].squeeze().astype(float)
        cell_ids = data["cell_ids"].squeeze()
        rec_ids = data["rec_ids"].squeeze()

        # Transpose X to (n_features, n_bins)
        if X.shape[0] > X.shape[1]:  # real data is (bins, features)
            X = X.T

        # Impute nan values in X with column means (features are columns after transpose)
        imp = SimpleImputer(strategy="mean")
        X = imp.fit_transform(X.T).T
        return X, Y, cell_ids, rec_ids

    # -----------------------------------
    # Case 2: Simulated data (multiple files)
    # -----------------------------------
    X_list = []
    Y_list = []
    cell_ids_list = []
    rec_ids_list = []

    cell_offset = 0
    rec_offset = 0

    for fname in filenames:
        full_path = os.path.join(path, fname)

        # Simulated data uses mat73 (HDF5-based MATLAB v7.3)
        data = mat73.loadmat(full_path)

        X = data["X"].astype(float)  # shape (n_features, n_bins)
        Y = data["y"].squeeze().astype(float)
        cell_ids = data["cell_ids"].squeeze()
        rec_ids = data["rec_id"].squeeze()

        # Offset IDs to avoid collisions across files
        cell_ids = cell_ids + cell_offset
        rec_ids = rec_ids + rec_offset

        # Append
        X_list.append(X)
        Y_list.append(Y)
        cell_ids_list.append(cell_ids)
        rec_ids_list.append(rec_ids)

        # Update offsets
        cell_offset = cell_ids.max() + 1
        rec_offset = rec_ids.max() + 1

    # Concatenate along time axis
    X_all = np.concatenate(X_list, axis=1)
    Y_all = np.concatenate(Y_list)
    cell_ids_all = np.concatenate(cell_ids_list)
    rec_ids_all = np.concatenate(rec_ids_list)

    imp = SimpleImputer(strategy="mean")
    X_all = imp.fit_transform(X_all.T).T

    return X_all, Y_all, cell_ids_all, rec_ids_all


def save_data(filepath, filename, results):
    """
    Persist a dictionary of results to a MATLAB .mat file.

    Parameters
    ----------
    filepath : str
        Directory path in which to save the file.
    filename : str
        Filename (should end in .mat).
    results : dict
        Mapping of variable names to arrays or scalars.
    """
    name = filepath + "/" + filename
    io.savemat(name, results)


def get_cell_slice(cell_idx, cell_ids):
    """
    Return indices of samples belonging to a given cell.

    Parameters
    ----------
    cell_idx : int
        Cell identifier to match.
    cell_ids : ndarray
        Array of cell IDs for each sample.

    Returns
    -------
    ndarray
        Integer array of indices where ``cell_ids`` equals ``cell_idx``.
    """
    idx = np.where(cell_ids == cell_idx)[0]
    return idx


def get_trial_slice(cell_idx, trial_idx, cell_ids, trials_per_cell=120):
    """
    Compute a slice covering a particular trial for a given cell.

    Trials are assumed to be contiguous blocks of equal length within each
    cell's time series.  This helper first locates the indices for the cell,
    then subdivides them into ``trials_per_cell`` equal bins, returning the
    slice corresponding to ``trial_idx``.

    Parameters
    ----------
    cell_idx : int
        Cell identifier.
    trial_idx : int
        Zero-based trial index within the cell.
    cell_ids : ndarray
        Array of cell IDs for each sample.
    trials_per_cell : int, optional
        Number of trials assumed per cell (default 120).

    Returns
    -------
    slice
        Python slice object that can index into arrays along the time axis.
    """
    idx = get_cell_slice(cell_idx, cell_ids)
    bins_per_cell = len(idx)
    bins_per_trial = bins_per_cell // trials_per_cell

    start = idx[0] + trial_idx * bins_per_trial
    end = start + bins_per_trial
    return slice(start, end)


def split_cell_data(cell_ids, train_frac=0.7, val_frac=0.15, use_val=True):
    """
    Generate index splits for each cell into train/val/test sets.

    For simplicity the split is non-random: early time bins go to train,
    middle to validation, and late to test.  The fractions ``train_frac`` and
    ``val_frac`` are relative to each cell's total bins.  When ``use_val`` is
    False the validation portion is omitted and its bins become part of the
    training set.

    Parameters
    ----------
    cell_ids : ndarray
        Array mapping each sample to a cell.
    train_frac : float
        Fraction of bins per cell allocated to training.
    val_frac : float
        Fraction allocated to validation (ignored if ``use_val`` is False).
    use_val : bool
        Whether to create a separate validation split.

    Returns
    -------
    dict
        Mapping cell ID to dictionary with keys ``train_idx``, ``val_idx`` and
        ``test_idx`` containing index arrays.
    """
    unique_cells = np.unique(cell_ids)
    splits = {}

    for cell in unique_cells:
        idx = get_cell_slice(cell, cell_ids)
        n = len(idx)

        n_train = int(train_frac * n)
        n_val = int(val_frac * n) if use_val else 0

        # temporarily just do a simple split (not randomised)
        # early trail bins in train, middle in val, late in test
        # total 3000 bins per cell, 120 trails, 25 bins per trial
        # following a 70/15/15 split at the trial level, we get:
        # train = first 84 trials (2100 bins), val = next 18 trials (450 bins),
        # test = last 18 trials (450 bins)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val] if use_val else np.array([], dtype=int)
        test_idx = idx[n_train + n_val :]

        splits[cell] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

    return splits


def prepare_cellwise_datasets(
    X, Y, cell_ids, train_frac=0.7, val_frac=0.15, use_val=True
):
    """
    Convert global arrays into per-cell train/val/test dictionaries.

    This function first obtains index splits via :func:`split_cell_data` and
    then slices the feature matrix ``X`` and response vector ``Y`` accordingly.
    The returned dictionaries map each cell ID to its respective arrays, and
    feature matrices are transposed to shape ``(bins, features)`` to match the
    expectations of sklearn-style estimators.

    Parameters
    ----------
    X : ndarray, shape (n_features, total_bins)
    Y : ndarray, shape (total_bins,)
    cell_ids : ndarray, shape (total_bins,)
    train_frac : float
    val_frac : float
    use_val : bool

    Returns
    -------
    tuple
        ``(X_train, Y_train, X_val, Y_val, X_test, Y_test)`` where each element
        is a dict keyed by cell ID. Validation dicts contain ``None`` if
        ``use_val`` is False.
    """
    splits = split_cell_data(cell_ids, train_frac, val_frac, use_val)

    X_train = {}
    X_val = {}
    X_test = {}
    Y_train = {}
    Y_val = {}
    Y_test = {}

    for cell, s in splits.items():
        train_idx = s["train_idx"]
        val_idx = s["val_idx"]
        test_idx = s["test_idx"]

        X_train[cell] = X[:, train_idx].T
        Y_train[cell] = Y[train_idx]

        X_val[cell] = X[:, val_idx].T if use_val else None
        Y_val[cell] = Y[val_idx] if use_val else None

        X_test[cell] = X[:, test_idx].T
        Y_test[cell] = Y[test_idx]

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def flatten_cellwise_data(X_dict, Y_dict):
    """
    Convert per-cell dicts into flattened arrays suitable for grid search.

    The reverse of :func:`prepare_cellwise_datasets`; it stitches each cell's
    data back into a global feature matrix and target vector while also
    producing a cell ID array of matching length.

    Parameters
    ----------
    X_dict : dict
        Mapping cell -> feature array (bins, features).
    Y_dict : dict
        Mapping cell -> target vector (bins,).

    Returns
    -------
    tuple
        ``(X_flat, Y_flat, cell_ids_flat)`` where ``X_flat`` has shape
        ``(features, total_bins)``.
    """
    X_list = []
    Y_list = []
    cell_id_list = []

    for cell, Xc in X_dict.items():
        yc = Y_dict[cell]
        n = len(yc)

        # Xc is already (time, features) from prepare_cellwise_datasets
        X_list.append(Xc)
        Y_list.append(yc)
        cell_id_list.append(np.full(n, cell))

    X_flat = np.concatenate(X_list, axis=0).T  # (features, time)
    Y_flat = np.concatenate(Y_list)
    cell_ids_flat = np.concatenate(cell_id_list)

    return X_flat, Y_flat, cell_ids_flat
