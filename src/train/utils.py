def summarise_model_results(results, model_name="Model"):
    """Print a human‑readable summary of per‑cell evaluation metrics.

    Parameters
    ----------
    results : dict
        Output from :func:`fit_model_per_cell` (or any function returning the
        same structure).  Keys are cell ids and values are dictionaries
        containing metric information under the keys ``"train"``,
        ``"val"``, and ``"test"``.
    model_name : str, optional
        Label to display in the printed headings (default ``"Model"``).

    Returns
    -------
    None
        This function only prints to standard output; it does not return a
        value.
    """
    print(f"\n===== {model_name} Summary =====")

    for cell, info in results.items():
        print(f"\n--- Cell {cell} ---")

        train = info.get("train")
        if train is not None:
            print(f"Train pseudo-R²:       {train['pseudo_r2']:.4f}")
            print(f"Train log-likelihood:  {train['log_likelihood']:.2f}")
            print(f"Train deviance:        {train['deviance']:.2f}")
        else:
            print("Train metrics:         (not available)")

        val = info.get("val")
        if val is not None:
            print(f"Val pseudo-R²:         {val['pseudo_r2']:.4f}")
            print(f"Val log-likelihood:    {val['log_likelihood']:.2f}")
            print(f"Val deviance:          {val['deviance']:.2f}")
        else:
            print("Val metrics:           (not available)")

        test = info.get("test")
        if test is None:
            print("Warning: test metrics missing for this cell")
        else:
            print(f"Test pseudo-R²:        {test['pseudo_r2']:.4f}")
            print(f"Test log-likelihood:   {test['log_likelihood']:.2f}")
            print(f"Test deviance:         {test['deviance']:.2f}")

    print("\n===== End of Summary =====\n")
