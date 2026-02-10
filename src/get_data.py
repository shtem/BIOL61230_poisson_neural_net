import mat73
import numpy as np
from scipy import io


def load_data(filepath, filenames):
    """
    Load data from multiple .mat files and concatenate them into a single dataset.
    Creates X (features), Y (responses), cell_ids, and rec_ids arrays.

    :param filepath: Path to the directory containing .mat files
    :param filenames: List of .mat file names to be loaded

    :return: Tuple of (X_all, Y_all, cell_ids_all, rec_ids_all)
    """
    X_list = []
    Y_list = []
    cell_ids_list = []
    rec_ids_list = []

    cell_offset = 0
    rec_offset = 0

    for fname in filenames:
        data = mat73.loadmat(filepath + "/" + fname)

        X = data["X"]
        Y = data["y"]
        cell_ids = data["cell_ids"]
        rec_ids = data["rec_id"]

        # Offset IDs so they remain unique across subjects
        cell_ids = cell_ids + cell_offset
        rec_ids = rec_ids + rec_offset

        # Append to lists
        X_list.append(X)
        Y_list.append(Y)
        cell_ids_list.append(cell_ids)
        rec_ids_list.append(rec_ids)

        # Update offsets
        cell_offset = cell_ids.max() + 1
        rec_offset = rec_ids.max() + 1

    # Concatenate across subjects
    X_all = np.concatenate(X_list, axis=1)  # concatenate time dimension
    Y_all = np.concatenate(Y_list, axis=0)
    cell_ids_all = np.concatenate(cell_ids_list, axis=0)
    rec_ids_all = np.concatenate(rec_ids_list, axis=0)

    return X_all, Y_all, cell_ids_all, rec_ids_all


def save_data(filepath, filename, results):
    """
    Save results to a .mat file.

    :param filepath: Path to the directory where the file will be saved
    :param filename: Name of the .mat file to be saved
    :param results: Dictionary of results to be saved
    """
    name = filepath + "/" + filename
    io.savemat(name, results)


def get_cell_slice(cell_idx, cell_ids):
    """
    Get the slice of indices corresponding to a specific cell ID.

    :param cell_idx: The cell ID to retrieve indices for
    :param cell_ids: Array of all cell IDs

    :return: Array of indices corresponding to the specified cell ID
    """
    idx = np.where(cell_ids == cell_idx)[0]
    return idx


def get_trial_slice(cell_idx, trial_idx, cell_ids, trials_per_cell=120):
    """
    Get the slice of indices corresponding to a specific trial within a specific cell.

    :param cell_idx: The cell ID to retrieve trial indices for
    :param trial_idx: The trial index within the specified cell
    :param cell_ids: Array of all cell IDs
    :param trials_per_cell: Number of trials per cell (default is 120)

    :return: Slice object corresponding to the specified trial within the specified cell
    """
    idx = get_cell_slice(cell_idx, cell_ids)
    bins_per_cell = len(idx)
    bins_per_trial = bins_per_cell // trials_per_cell

    start = idx[0] + trial_idx * bins_per_trial
    end = start + bins_per_trial
    return slice(start, end)


def split_cell_data(cell_ids, train_frac=0.7, val_frac=0.15):
    """
    Get train/val/test splits for each cell based on the provided fractions.
    Assumes that data is ordered by trial within each cell.

    :param cell_ids: Array of all cell IDs
    :param train_frac: Fraction of data to be used for training
    :param val_frac: Fraction of data to be used for validation

    :return: Dictionary mapping each cell ID to its train/val/test indices
    """
    unique_cells = np.unique(cell_ids)
    splits = {}

    for cell in unique_cells:
        idx = np.where(cell_ids == cell)[0]

        n = len(idx)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        # temporarily just do a simple split (not randomized)
        # early trail bins in train, middle in val, late in test
        # total 3000 bins per cell, 120 trails, 25 bins per trial
        # following a 70/15/15 spliyt at the trial level, we get:
        # train = first 84 trials (2100 bins), val = next 18 trials (450 bins),
        # test = last 18 trials (450 bins)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        splits[cell] = {
            "train_idx": train_idx,
            "val_idx": val_idx,
            "test_idx": test_idx,
        }

    return splits
