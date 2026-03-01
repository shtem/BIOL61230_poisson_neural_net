import inspect
from pathlib import Path
from typing import Callable, Optional

from src.visualisation import (
    plot_training_curves,
    plot_ytrue_vs_ypred,
)

# import functions from the new modules so callers of `train.utils` don't break
from src.train.training import summarise_model_results
from src.train.io import save_model, save_plot


def run_experiment(
    model_name: str,
    fit_fn: Callable,
    fit_kwargs: dict,
    X,
    Y,
    cell_ids,
    scaler=None,
    plot: bool = True,
    save_models: bool = True,
    base_models_dir: Optional[Path] = None,
    base_results_dir: Optional[Path] = None,
):
    """Unified training/evaluation routine for notebook experiments.

    This helper centralises the sequence that was previously duplicated in the
    notebook:

    1. call ``fit_fn`` to train the model
    2. summarise the per-cell results
    3. optionally generate a couple of diagnostic plots
    4. save the fitted object and figures to disk

    Models are stored under ``data/models`` by default and plots are written to
    ``data/results/{model_name}`` unless alternate directories are specified.  When
    a fitted model is a PyTorch ``nn.Module`` an additional architecture diagram
    will be produced (using :func:`src.visualisation.plot_nn_architecture`) and
    saved alongside the other figures.

    Parameters
    ----------
    model_name : str
        Identifier used for plot subdirectory and model filename.
    fit_fn : callable
        One of the existing ``fit_*`` wrappers (GLM, XGBoost, NN, etc.).
    fit_kwargs : dict
        Keyword arguments forwarded to ``fit_fn``.
    X, Y, cell_ids
        Data arrays passed through to ``fit_fn``.
    scaler : callable or None, optional
        Feature scaler factory forwarded to ``fit_fn``.
    plot : bool, optional
        If True, create summary plots for the first cell in the results.
    save_models : bool, optional
        If True, persist the returned model object(s) using
        :func:`src.train.io.save_model`.
    base_models_dir : Path or None, optional
        Root directory for model files (default ``data/models``).
    base_results_dir : Path or None, optional
        Root directory for plots (default ``data/results``).

    Returns
    -------
    dict
        The dictionary returned by ``fit_fn``.  Callers can inspect
        ``["results"]`` and ``["best_params"]`` as before.
    """

    # Only pass scaler if the fit function accepts it
    sig = inspect.signature(fit_fn)
    if "scaler" in sig.parameters:
        res = fit_fn(
            X,
            Y,
            cell_ids,
            scaler=scaler,
            **fit_kwargs,
        )
    else:
        res = fit_fn(
            X,
            Y,
            cell_ids,
            **fit_kwargs,
        )

    if plot:
        summarise_model_results(res["results"], model_name=model_name)
        first = sorted(res["results"].keys())[0]
        fig1 = plot_training_curves(
            train_losses=res["results"][first].get("train_losses"),
            val_losses=res["results"][first].get("val_losses"),
        )
        fig2 = plot_ytrue_vs_ypred(
            y_true=res["results"][first]["y_test"],
            y_pred=res["results"][first]["y_pred_test"],
            title=f"{model_name} — Test set",
        )
    else:
        fig1 = fig2 = None

    # if this is a PyTorch model, also generate an architecture diagram
    if plot:
        try:
            import torch
            from src.visualisation import plot_nn_architecture

            if isinstance(res["results"][first]["model"], torch.nn.Module):
                br = Path(base_results_dir or "data/results")
                try:
                    arch_path = plot_nn_architecture(
                        res["results"][first]["model"], model_name, base_dir=br
                    )
                    if arch_path is not None:
                        print(f"Architecture diagram saved to {arch_path}")
                except Exception as exc:
                    # render failure (likely graphviz missing); warn but continue
                    print("Warning: architecture plot failed:", exc)
        except ImportError:
            # either torch or visualization helper missing; ignore
            pass

    if save_models:
        bm = Path(base_models_dir or "data/models")
        save_model(res, model_name, base_dir=bm)  # store entire result dict

    if plot and (fig1 is not None or fig2 is not None):
        br = Path(base_results_dir or "data/results")
        if fig1 is not None:
            save_plot(fig1, model_name, "train_losses.png", base_dir=br)
        if fig2 is not None:
            save_plot(fig2, model_name, "ytrue_vs_ypred.png", base_dir=br)

    return res
