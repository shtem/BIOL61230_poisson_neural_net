"""
Cross-Batch Aggregation and Final Reporting
============================================
It loads the CSV files saved by batch_statistical_analysis.py
and produces combined statistics across all completed batches.
"""

# ── imports ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
import pickle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── config ─────────────────────────────────────────────────────────────────
# Set this to the parent directory containing your batch_0, batch_1, etc. folders
BASE_REAL_DIR = Path("resources/models/real")
BASE_RESULTS_DIR = Path("resources/results/real")

# Which batches to include
COMPLETED_BATCHES = [0, 1, 2, 3]

# ── load per-batch result tables ───────────────────────────────────────────


def load_batch_results(batch_index, base_models_dir):
    """
    Load the cached results dict for a completed batch and extract
    per-cell pseudo-R² values for all models into a flat DataFrame.

    Parameters
    ----------
    batch_index : int
        The batch number (used for filenames and printed headers).
    base_models_dir : Path
        Directory where batch folders are located (e.g. BASE_REAL_DIR).

    Returns
    -------
    pd.DataFrame or None
        Columns: 'cell', then one column per model.
        Returns None if the batch directory doesn't exist.
    """
    batch_dir = base_models_dir / f"batch_{batch_index}"
    if not batch_dir.exists():
        print(f"  [WARN] batch_{batch_index} directory not found, skipping.")
        return None

    # Collect all .pkl model files in this batch
    pkl_files = sorted(batch_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"  [WARN] No pkl files in batch_{batch_index}, skipping.")
        return None

    batch_rows = {}  # cell_id -> {model_name: r2}

    for pkl_path in pkl_files:
        model_name = pkl_path.stem  # filename without .pkl
        try:
            with open(pkl_path, "rb") as f:
                res = pickle.load(f)
        except Exception as e:
            print(f"  [WARN] Could not load {pkl_path.name}: {e}")
            continue

        # res is the dict returned by run_experiment: {'results': {...}, ...}
        results = res.get("results", {})
        for cell_id, cell_data in results.items():
            test_metrics = cell_data.get("test")
            if test_metrics is None:
                continue
            r2 = test_metrics.get("pseudo_r2", np.nan)
            if cell_id not in batch_rows:
                batch_rows[cell_id] = {"cell": cell_id}
            batch_rows[cell_id][model_name] = r2

    if not batch_rows:
        return None

    df = pd.DataFrame(list(batch_rows.values()))
    df = df.sort_values("cell").reset_index(drop=True)
    return df


# ── load all batches ────────────────────────────────────────────────────────
print(f"Loading batches: {COMPLETED_BATCHES}")
all_dfs = []

for b in COMPLETED_BATCHES:
    print(f"  Loading batch {b}...")
    df = load_batch_results(b, BASE_REAL_DIR)
    if df is not None:
        all_dfs.append(df)
        print(f"    {len(df)} cells, {len(df.columns)-1} models")

if not all_dfs:
    raise RuntimeError("No batch data could be loaded. Check BASE_REAL_DIR.")

# Concatenate all batches
df_combined = pd.concat(all_dfs, ignore_index=True)
n_cells = len(df_combined)
model_cols = [c for c in df_combined.columns if c != "cell"]

print(f"\nCombined dataset: {n_cells} cells, {len(model_cols)} models")
print(f"Cell IDs: {df_combined['cell'].min()} – {df_combined['cell'].max()}")


# ── combined summary statistics ─────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"COMBINED STATISTICS — {n_cells} CELLS ACROSS {len(COMPLETED_BATCHES)} BATCHES")
print(f"{'='*85}")

summary_rows = []
for m in model_cols:
    arr = df_combined[m].dropna().values
    summary_rows.append(
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

summary = pd.DataFrame(summary_rows).set_index("model")

print(
    f"\n{'Model':<42} {'Median':>7} {'Mean':>7} {'IQR lo':>7} {'IQR hi':>7} {'%>0':>6} {'N>0':>5}"
)
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


# ── Wilcoxon vs GLM ─────────────────────────────────────────────────────────
print(f"\n{'='*85}")
print("WILCOXON SIGNED-RANK TESTS vs GLM (one-sided: model > GLM)")
print(f"{'='*85}")
print(f"{'Model':<42} {'W statistic':>12} {'p-value':>10} {'sig':>6}")
print("-" * 72)

glm_arr = df_combined["GLM"].dropna().values
wilcoxon_glm_rows = []

for m in model_cols:
    if m == "GLM":
        continue
    arr = df_combined[m].dropna().values
    # align on common cells
    common = df_combined[["GLM", m]].dropna()
    g = common["GLM"].values
    a = common[m].values
    try:
        stat, p = stats.wilcoxon(a, g, alternative="greater")
    except ValueError:
        stat, p = np.nan, 1.0

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{m:<42} {stat:>12.1f} {p:>10.4f} {sig:>6}")
    wilcoxon_glm_rows.append({"model": m, "W": stat, "p_vs_glm": p, "sig_vs_glm": sig})

df_wglm = pd.DataFrame(wilcoxon_glm_rows).set_index("model")


# ── Wilcoxon: TL models vs NN-PerCell-MLP ───────────────────────────────────
tl_cols = [c for c in model_cols if "TL" in c]

if "NN-PerCell-MLP" in model_cols and tl_cols:
    print(f"\n{'='*85}")
    print("WILCOXON SIGNED-RANK TESTS vs NN-PerCell-MLP (one-sided: TL > MLP)")
    print(f"{'='*85}")
    print(f"{'Model':<42} {'W statistic':>12} {'p-value':>10} {'sig':>6}")
    print("-" * 72)

    wilcoxon_mlp_rows = []
    for m in tl_cols:
        common = df_combined[["NN-PerCell-MLP", m]].dropna()
        mlp = common["NN-PerCell-MLP"].values
        arr = common[m].values
        try:
            stat, p = stats.wilcoxon(arr, mlp, alternative="greater")
        except ValueError:
            stat, p = np.nan, 1.0

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{m:<42} {stat:>12.1f} {p:>10.4f} {sig:>6}")
        wilcoxon_mlp_rows.append(
            {"model": m, "W": stat, "p_vs_mlp": p, "sig_vs_mlp": sig}
        )

    df_wmlp = pd.DataFrame(wilcoxon_mlp_rows).set_index("model")


# ── Per-batch consistency check ─────────────────────────────────────────────
print(f"\n{'='*85}")
print("BATCH-BY-BATCH CONSISTENCY CHECK")
print(f"{'='*85}")

key_models = [
    "GLM",
    "XGBoost",
    "NN-PerCell-MLP",
    "NN-DeepSharedDeepHead-TL-MLP",
    "NN-DeepSharedShallowHead-TL-MLP",
    "NN-ShallowSharedDeepHead-TL-MLP",
]

# Only include models that exist in the data
key_models = [m for m in key_models if m in model_cols]

print(f"\n{'Model':<42}", end="")
for b in COMPLETED_BATCHES:
    print(f"  Batch {b} Med", end="")
print()
print("-" * (42 + 14 * len(COMPLETED_BATCHES)))

for m in key_models:
    print(f"{m:<42}", end="")
    for b_idx, df_b in zip(COMPLETED_BATCHES, all_dfs):
        if m in df_b.columns:
            med = np.median(df_b[m].dropna().values)
            print(f"  {med:>11.4f}", end="")
        else:
            print(f"  {'N/A':>11}", end="")
    print()


# ── ranking by combined median ───────────────────────────────────────────────
print(f"\n{'='*85}")
print(f"FINAL MODEL RANKING BY COMBINED MEDIAN ({n_cells} cells)")
print(f"{'='*85}")
ranked = summary["median"].sort_values(ascending=False)
for rank, (m, med) in enumerate(ranked.items(), 1):
    pct = summary.loc[m, "pct_above_chance"]
    n = summary.loc[m, "n_above_chance"]
    print(
        f"  {rank:>2}. {m:<42} {med:.4f}  ({pct:.0f}% above chance, {n}/{n_cells} cells)"
    )


# ── save combined CSVs ────────────────────────────────────────────────────
out_dir = BASE_RESULTS_DIR / "combined"
out_dir.mkdir(parents=True, exist_ok=True)

batches_str = "_".join(str(b) for b in COMPLETED_BATCHES)
summary.to_csv(out_dir / f"combined_batches_{batches_str}_summary.csv")
df_combined.to_csv(out_dir / f"combined_batches_{batches_str}_all_r2.csv", index=False)
df_wglm.to_csv(out_dir / f"combined_batches_{batches_str}_wilcoxon_glm.csv")
if "df_wmlp" in dir():
    df_wmlp.to_csv(out_dir / f"combined_batches_{batches_str}_wilcoxon_mlp.csv")

print(f"\nAll combined results saved to {out_dir}")
print(f"  - combined_batches_{batches_str}_summary.csv")
print(f"  - combined_batches_{batches_str}_all_r2.csv")
print(f"  - combined_batches_{batches_str}_wilcoxon_glm.csv")
print(f"  - combined_batches_{batches_str}_wilcoxon_mlp.csv")
