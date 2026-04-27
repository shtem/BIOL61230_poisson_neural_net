import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from src.get_data import get_cell_slice


# -----------------------------
# Feature extraction
# -----------------------------
def extract_corr_features(X, Y, cell_ids):
    """
    Compute per-cell correlation between covariates and spike counts.

    This simple feature extractor treats each covariate (row of ``X``) as a
    predictor and computes the Pearson correlation with the corresponding
    cell's spike count time series.  The resulting feature vector for each cell
    has length equal to ``n_features`` and can be used as input to clustering
    algorithms.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_time_bins)
        Covariate time series.
    Y : ndarray, shape (n_time_bins,)
        Spike count observations across time.
    cell_ids : ndarray, shape (n_time_bins,)
        Identifier specifying which cell produced each observation.

    Returns
    -------
    ndarray, shape (n_cells, n_features)
        Correlation-based feature matrix. Rows correspond to unique cells in
        the order returned by ``np.unique(cell_ids)``.
    """
    features = []
    for cell in np.unique(cell_ids):
        idx = get_cell_slice(cell, cell_ids)
        y_cell = Y[idx]
        x_cell = X[:, idx]

        # compute correlation between each feature and the cell's response
        corr = [np.corrcoef(x_cell[i], y_cell)[0, 1] for i in range(X.shape[0])]
        features.append(corr)

    return np.array(features)


def extract_glm_features(X, Y, cell_ids):
    """
    Generate features by fitting a Poisson GLM per cell and using weights.

    For each cell, the design matrix (features across time) is transposed to
    shape ``(n_time_bins_cell, n_features)`` and a Poisson regression is fit.
    The fitted coefficients are then returned as the feature vector for that
    cell.  This method can capture more nuanced relationships than simple
    correlations.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_time_bins)
    Y : ndarray, shape (n_time_bins,)
    cell_ids : ndarray, shape (n_time_bins,)

    Returns
    -------
    ndarray, shape (n_cells, n_features)
        GLM coefficient matrix.
    """
    features = []
    for cell in np.unique(cell_ids):
        idx = get_cell_slice(cell, cell_ids)
        y_cell = Y[idx]
        x_cell = X[:, idx].T  # transpose for sklearn's convention

        glm = PoissonRegressor(alpha=0.0, max_iter=1000)
        glm.fit(x_cell, y_cell)

        features.append(glm.coef_)
    return np.array(features)


# -----------------------------
# Feature normalisation
# -----------------------------
def scale_features(features):
    """
    Standardise feature matrix to zero mean and unit variance across cells.

    This global scaling ensures that each feature contributes equally to
    distance-based clustering algorithms.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
        Raw feature matrix to be standardised.

    Returns
    -------
    ndarray, shape (n_cells, n_features)
        Standardised features with sero mean and unit variance.
    """
    return StandardScaler().fit_transform(features)


def zscore_features_within_subject(features, cell_ids, rec_ids):
    """
    Normalise features separately for each recording/subject.

    This is helpful when recordings come from multiple subjects and one wishes
    to remove subject-specific offsets or scale differences before clustering.
    The operation is performed per unique recording ID, z-scoring only those
    cells belonging to that recording.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
        Feature matrix to be normalised.
    cell_ids : ndarray, shape (n_cells,)
        Unique identifiers for each cell.
    rec_ids : ndarray, shape (n_cells,)
        Recording/subject ID corresponding to each cell.

    Returns
    -------
    ndarray, shape (n_cells, n_features)
        Z-scored features within each subject.
    """
    unique_subjects = np.unique(rec_ids)
    features_z = features.copy()

    for subj in unique_subjects:
        subj_cells = np.unique(cell_ids[rec_ids == subj])
        idx = [np.where(np.unique(cell_ids) == c)[0][0] for c in subj_cells]

        features_z[idx] = (features[idx] - features[idx].mean(axis=0)) / features[
            idx
        ].std(axis=0)

    return features_z


# -----------------------------
# Clustering algorithms
# -----------------------------
def kmeans_cluster(features, n_clusters=2):
    """
    Perform K-means clustering on cell feature vectors.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    n_clusters : int
        Desired number of clusters.

    Returns
    -------
    ndarray, shape (n_cells,)
        Integer cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(features)


def hierarchical_cluster(features, method="ward", max_clusters=None, show_plot=True):
    """
    Apply hierarchical clustering and optionally plot dendrogram.

    The linkage matrix ``Z`` is computed from the feature matrix.  If
    ``show_plot`` is True a matplotlib figure containing the dendrogram is
    returned.  ``max_clusters`` can be specified to extract flat cluster labels
    at a particular cut height.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    method : str
        Agglomeration method used by ``scipy.cluster.hierarchy.linkage``.
    max_clusters : int or None
        If provided, generate flat cluster labels with at most this many
        clusters.
    show_plot : bool
        Whether to render and return a dendrogram figure.

    Returns
    -------
    labels : ndarray or None
        Cluster labels if ``max_clusters`` is not None, else None.
    fig : matplotlib.figure.Figure or None
        Dendrogram figure if ``show_plot`` is True, otherwise None.
    """
    Z = linkage(features, method=method)

    fig = None
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Cell index")
        ax.set_ylabel("Distance")
        ax.grid(alpha=0.3)

    labels = None
    if max_clusters is not None:
        labels = fcluster(Z, max_clusters, criterion="maxclust")

    return labels, fig


# -----------------------------
# Visualisation
# -----------------------------
def plot_clusters(features, labels, title):
    """
    Visualise clusters after projecting onto first two principal components.

    PCA is a quick way to reduce dimensionality for plotting.  Points are
    coloured by cluster label.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
        Feature matrix to project.
    labels : ndarray, shape (n_cells,)
        Cluster labels for colouring.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scatter plot.
    """
    pca = PCA(n_components=2)
    proj = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="viridis", s=80)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.3)

    return fig


def plot_umap(features, labels, title):
    """
    Use UMAP to embed features in 2D for cluster visualisation.

    UMAP often preserves local structure better than PCA and can reveal
    nonlinear relationships between clusters.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
        Feature matrix to embed.
    labels : ndarray, shape (n_cells,)
        Cluster labels for colouring.
    title : str
        Plot title.

    Returns
    -------
    tuple
        ``(fig, embedding)`` where ``fig`` is the Matplotlib figure and
        ``embedding`` is an ndarray of shape (n_cells, 2) containing coordinates.
    """
    reducer = umap.UMAP(
        n_neighbors=5,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )

    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="viridis", s=80)

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.3)

    return fig, embedding


# -----------------------------
# Cluster summaries and reporting
# -----------------------------
def summarise_clusters(features, labels):
    """
    Compute centroid (mean feature vector) for each cluster.

    This summary can be used to understand the typical tuning profile of each
    group of cells.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    labels : ndarray, shape (n_cells,)

    Returns
    -------
    dict
        Mapping from cluster label to mean feature vector.
    """
    summaries = {}
    for c in np.unique(labels):
        summaries[c] = features[labels == c].mean(axis=0)
    return summaries


def plot_cluster_tuning(features, labels, title_prefix="Cluster", covariate_names=None):
    """
    Visualise average feature responses for each cluster.

    Bar plots are generated showing the mean feature value across cells within
    each cluster, which is useful when features represent tuning to covariates.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    labels : ndarray, shape (n_cells,)
    title_prefix : str
        Prefix for plot title.
    covariate_names : list of str or None
        Names for each feature. If None, generic ``Cov 0, Cov 1...`` labels are used.

    Returns
    -------
    list
        List of Matplotlib figure objects, one per cluster.
    """
    n_features = features.shape[1]
    clusters = np.unique(labels)
    tick_labels = (
        covariate_names
        if covariate_names is not None
        else [f"Cov {i}" for i in range(n_features)]
    )
    figs = []

    for c in clusters:
        mean_vec = features[labels == c].mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(n_features), mean_vec)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean feature value")
        ax.set_title(f"{title_prefix} {c} tuning profile")
        ax.grid(alpha=0.3)
        fig.tight_layout()

        figs.append(fig)

    return figs


def suggest_labels(cluster_summaries, covariate_names=None):
    """
    Generate textual descriptions of each cluster's dominant feature.

    Identifies the covariate with maximum absolute mean weight and reports
    whether the tuning is positive or negative.  This is a simple heuristic for
    assigning functional interpretations to clusters.

    Parameters
    ----------
    cluster_summaries : dict
        Mapping cluster label to mean feature vector.
    covariate_names : list of str or None
        Names for each feature. If None, generic ``Covariate N`` labels are used.

    Returns
    -------
    dict
        Mapping cluster label to suggested textual label.
    """
    labels = {}
    for c, vec in cluster_summaries.items():
        strongest = np.argmax(np.abs(vec))
        direction = "positive" if vec[strongest] > 0 else "negative"
        name = (
            covariate_names[strongest]
            if covariate_names is not None
            else f"Covariate {strongest}"
        )
        labels[c] = f"Strongly tuned to {name} ({direction})"
    return labels


def evaluate_clustering(features, labels):
    """
    Compute silhouette score as an internal measure of cluster separation.

    The silhouette score ranges from -1 to +1, with higher values indicating
    that samples are well-matched to their own cluster and poorly matched to
    neighboring clusters.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    labels : ndarray, shape (n_cells,)

    Returns
    -------
    float
        Silhouette score.
    """
    return silhouette_score(features, labels)


def map_cells_to_clusters(labels, cell_ids):
    """
    Create a dictionary mapping each unique cell ID to its cluster label.

    Useful for later lookup or exporting membership information.

    Parameters
    ----------
    labels : ndarray, shape (n_cells,)
    cell_ids : ndarray, shape (n_cells,)

    Returns
    -------
    dict
        Mapping cell ID -> cluster label.
    """
    unique_cells = np.unique(cell_ids)
    return {cell: labels[i] for i, cell in enumerate(unique_cells)}


def print_cluster_membership(labels, cell_ids):
    """
    Display cells belonging to each cluster and return the mapping.

    Parameters are same as :func:`map_cells_to_clusters`.  The printed output is
    grouped by cluster label for easy human inspection.

    Parameters
    ----------
    labels : ndarray, shape (n_cells,)
    cell_ids : ndarray, shape (n_cells,)

    Returns
    -------
    dict
        Mapping cell ID -> cluster label.
    """
    mapping = map_cells_to_clusters(labels, cell_ids)
    clusters = {}
    for cell, cl in mapping.items():
        clusters.setdefault(cl, []).append(int(cell))

    for cl, cells in clusters.items():
        print(f"Cluster {cl}: cells {cells}")

    return mapping


def cluster_report(
    features, labels, cell_ids, title_prefix="Cluster", covariate_names=None
):
    """
    Produce a full textual and graphical summary of clustering results.

    This helper prints a report to the console and returns a dictionary
    containing various artefacts such as membership mappings, cluster
    centroids, suggested labels, and generated figures for later use.

    Parameters
    ----------
    features : ndarray, shape (n_cells, n_features)
    labels : ndarray, shape (n_cells,)
    cell_ids : ndarray, shape (n_cells,)
    title_prefix : str
        Prefix for report titles.
    covariate_names : list of str or None
        Names for each feature used in tuning profile plots and suggested labels.
        If None, generic ``Cov N`` labels are used.

    Returns
    -------
    dict
        Contains keys 'membership', 'summaries', 'suggested_labels',
        'tuning_figures', 'pca_figure', 'umap_figure', 'umap_embedding', and
        'silhouette_score'.
    """
    print(f"\n{title_prefix} report:")

    score = evaluate_clustering(features, labels)
    mapping = print_cluster_membership(labels, cell_ids)
    summaries = summarise_clusters(features, labels)
    tuning_figs = plot_cluster_tuning(
        features, labels, title_prefix, covariate_names=covariate_names
    )
    suggested = suggest_labels(summaries, covariate_names=covariate_names)

    print(f"Silhouette score: {score:.3f}")

    print("\nSuggested functional labels:")
    for c, label in suggested.items():
        print(f"Cluster {c}: {label}")

    pca_fig = plot_clusters(features, labels, f"{title_prefix} Clusters")
    umap_fig, embedding = plot_umap(features, labels, f"{title_prefix} UMAP")

    return {
        "membership": mapping,
        "summaries": summaries,
        "suggested_labels": suggested,
        "tuning_figures": tuning_figs,
        "pca_figure": pca_fig,
        "umap_figure": umap_fig,
        "umap_embedding": embedding,
        "silhouette_score": score,
    }


FEATURE_EXTRACTORS = {
    "correlation": extract_corr_features,
    "glm": extract_glm_features,
}

CLUSTERING_METHODS = {
    "kmeans": kmeans_cluster,
    "hierarchical": hierarchical_cluster,
}


def run_clustering(
    X,
    Y,
    cell_ids,
    rec_ids,
    feature_key,
    cluster_key,
    n_clusters=None,
    zscore_within_subject=False,
    title_prefix="Clustering",
    show_report=True,
):
    """
    Execute end-to-end cell clustering workflow and optionally report.

    This convenience function looks up the requested feature extractor and
    clustering algorithm from the registries defined above, normalises the
    extracted features, runs the clustering, and generates a detailed report if
    requested.  It is designed to be a single-entry point for exploratory
    analyses.

    Parameters
    ----------
    X : ndarray, shape (n_features, n_time_bins)
        Covariate matrix.
    Y : ndarray, shape (n_time_bins,)
        Spike counts.
    cell_ids : ndarray, shape (n_time_bins,)
        Cell identifier per sample.
    rec_ids : ndarray, shape (n_cells,)
        Recording/subject ID per cell (for z-scoring).
    feature_key : str
        Key selecting feature extraction method.
    cluster_key : str
        Key selecting clustering algorithm.
    n_clusters : int or None
        Number of clusters for k-means or max clusters for hierarchical.
    zscore_within_subject : bool
        If True, z-score features within each recording.
    title_prefix : str
        Prefix used in plot/report titles.
    show_report : bool
        Whether to call ``cluster_report`` and return its output.

    Returns
    -------
    dict
        Contains 'features', 'scaled_features', 'labels', and 'report' (if
        requested).
    """

    # 1. Look up functions from registries
    if feature_key not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature extractor '{feature_key}'")

    if cluster_key not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method '{cluster_key}'")

    feature_extractor = FEATURE_EXTRACTORS[feature_key]
    clustering_method = CLUSTERING_METHODS[cluster_key]

    # 2. Extract features
    features = feature_extractor(X, Y, cell_ids)
    if np.isnan(features).any():
        imp = SimpleImputer(strategy="mean")
        features = imp.fit_transform(features)

    # 3. Normalise features
    if zscore_within_subject:
        scaled = zscore_features_within_subject(features, cell_ids, rec_ids)
    else:
        scaled = scale_features(features)

    # 4. Clustering
    if cluster_key == "kmeans":
        if n_clusters is None:
            raise ValueError("KMeans requires n_clusters.")
        labels = clustering_method(scaled, n_clusters=n_clusters)

    elif cluster_key == "hierarchical":
        labels, _ = clustering_method(scaled, max_clusters=n_clusters, show_plot=False)

    # 5. Optional reporting
    report = None
    if show_report:
        report = cluster_report(
            features,
            labels,
            cell_ids,
            title_prefix=f"{title_prefix} ({feature_key}, {cluster_key})",
        )

    return {
        "features": features,
        "scaled_features": scaled,
        "labels": labels,
        "report": report,
    }
