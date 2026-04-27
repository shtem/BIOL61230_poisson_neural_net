import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def summarise_and_test(df, base_results_dir, batch_index):
    """
    Compute summary statistics and Wilcoxon tests from a results DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The df_test_perf_gs DataFrame produced earlier in the notebook.
        Must have a 'cell' column and one column per model.
    base_results_dir : str or Path
        Directory where CSV outputs will be saved.
    batch_index : int
        The batch number (used for filenames and printed headers).

    Returns
    -------
    pd.DataFrame
        Summary statistics table (also saved to CSV).
    """
    base_results_dir = Path(base_results_dir)
    base_results_dir.mkdir(parents=True, exist_ok=True)

    model_cols = [c for c in df.columns if c != "cell"]

    # ── 1. Summary statistics ───────────────────────────────────────────────
    rows = []
    for m in model_cols:
        arr = df[m].values
        rows.append(
            {
                "model": m,
                "n_cells": len(arr),
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "iqr_lo": float(np.percentile(arr, 25)),
                "iqr_hi": float(np.percentile(arr, 75)),
                "pct_above_chance": float(100 * np.mean(arr > 0)),
                "n_above_chance": int(np.sum(arr > 0)),
            }
        )
    summary = pd.DataFrame(rows).set_index("model")

    print(f"\n{'='*80}")
    print(f"BATCH {batch_index} — SUMMARY STATISTICS ({len(df)} cells)")
    print(f"{'='*80}")
    header = f"{'Model':<42} {'Median':>7} {'Mean':>7} {'IQR lo':>7} {'IQR hi':>7} {'%>0':>6} {'N>0':>5}"
    print(header)
    print("-" * 82)
    for m, row in summary.iterrows():
        print(
            f"{m:<42} "
            f"{row['median']:>7.4f} "
            f"{row['mean']:>7.4f} "
            f"{row['iqr_lo']:>7.4f} "
            f"{row['iqr_hi']:>7.4f} "
            f"{row['pct_above_chance']:>5.0f}% "
            f"{int(row['n_above_chance']):>5d}"
        )

    # ── 2. Wilcoxon: every model vs GLM ────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"WILCOXON SIGNED-RANK TESTS vs GLM (one-sided: model > GLM)")
    print(f"{'='*80}")
    print(f"{'Model':<42} {'statistic':>10} {'p-value':>10} {'sig':>8}")
    print("-" * 72)

    glm_arr = df["GLM"].values
    wilcoxon_glm = []

    for m in model_cols:
        if m == "GLM":
            wilcoxon_glm.append(
                {"model": m, "statistic": np.nan, "p_value": np.nan, "significant": ""}
            )
            continue
        arr = df[m].values
        try:
            stat, p = stats.wilcoxon(arr, glm_arr, alternative="greater")
        except ValueError:
            # identical arrays or zero differences
            stat, p = np.nan, 1.0

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{m:<42} {stat:>10.1f} {p:>10.4f} {sig:>8}")
        wilcoxon_glm.append(
            {"model": m, "statistic": stat, "p_value": p, "significant": sig}
        )

    df_wglm = pd.DataFrame(wilcoxon_glm).set_index("model")

    # ── 3. Wilcoxon: TL models vs NN-PerCell-MLP ───────────────────────────
    tl_cols = [c for c in model_cols if "TL" in c]

    if "NN-PerCell-MLP" in model_cols and tl_cols:
        mlp_arr = df["NN-PerCell-MLP"].values
        print(f"\n{'='*80}")
        print(f"WILCOXON SIGNED-RANK TESTS vs NN-PerCell-MLP (one-sided: TL > MLP)")
        print(f"{'='*80}")
        print(f"{'Model':<42} {'statistic':>10} {'p-value':>10} {'sig':>8}")
        print("-" * 72)

        wilcoxon_mlp = []
        for m in tl_cols:
            arr = df[m].values
            try:
                stat, p = stats.wilcoxon(arr, mlp_arr, alternative="greater")
            except ValueError:
                stat, p = np.nan, 1.0

            sig = (
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            )
            print(f"{m:<42} {stat:>10.1f} {p:>10.4f} {sig:>8}")
            wilcoxon_mlp.append(
                {"model": m, "statistic": stat, "p_value": p, "significant": sig}
            )

        df_wmlp = pd.DataFrame(wilcoxon_mlp).set_index("model")
    else:
        df_wmlp = pd.DataFrame()

    # ── 4. Ranking by median ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"MODEL RANKING BY MEDIAN PSEUDO-R² (batch {batch_index})")
    print(f"{'='*80}")
    ranked = summary["median"].sort_values(ascending=False)
    for rank, (m, med) in enumerate(ranked.items(), 1):
        pct = summary.loc[m, "pct_above_chance"]
        print(f"  {rank:>2}. {m:<42} {med:.4f}  ({pct:.0f}% above chance)")

    # ── 5. Save to CSV ──────────────────────────────────────────────────────
    summary_path = base_results_dir / f"batch_{batch_index}_summary_stats.csv"
    summary.to_csv(summary_path)
    print(f"\nSummary statistics saved to {summary_path}")

    wglm_path = base_results_dir / f"batch_{batch_index}_wilcoxon_vs_glm.csv"
    df_wglm.to_csv(wglm_path)
    print(f"Wilcoxon (vs GLM) results saved to {wglm_path}")

    if not df_wmlp.empty:
        wmlp_path = base_results_dir / f"batch_{batch_index}_wilcoxon_vs_mlp.csv"
        df_wmlp.to_csv(wmlp_path)
        print(f"Wilcoxon (vs MLP) results saved to {wmlp_path}")

    return summary
