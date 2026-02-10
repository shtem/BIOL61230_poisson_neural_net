import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap


# -----------------------------
# Feature extraction
# -----------------------------
def extract_corr_features(X, Y, cell_ids):
    """
    Extract correlation features for each cell by computing
    the correlation between each covariate and the cell's spike counts.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the spike counts for each time bin
    :param cell_ids: Array of all cell IDs

    :return: Array of shape (n_cells, n_features) containing the correlation features for each cell
    """
    features = []
    for cell in np.unique(cell_ids):
        idx = np.where(cell_ids == cell)[0]
        y_cell = Y[idx]
        x_cell = X[:, idx]

        corr = [np.corrcoef(x_cell[i], y_cell)[0, 1] for i in range(X.shape[0])]
        features.append(corr)

    return np.array(features)


def extract_glm_features(X, Y, cell_ids):
    """
    Extract GLM weight features for each cell by fitting a Poisson GLM to each cell's data.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the spike counts for each time bin
    :param cell_ids: Array of all cell IDs

    :return: Array of shape (n_cells, n_features) containing the GLM weight features for each cell
    """
    features = []
    for cell in np.unique(cell_ids):
        idx = np.where(cell_ids == cell)[0]
        y_cell = Y[idx]
        x_cell = X[:, idx].T

        glm = PoissonRegressor(alpha=0.0, max_iter=1000)
        glm.fit(x_cell, y_cell)

        features.append(glm.coef_)
    return np.array(features)


# -----------------------------
# Feature normalisation
# -----------------------------
def scale_features(features):
    """
    Standardise features globally.

    :param features: Array of shape (n_cells, n_features) containing the features to be standardised

    :return: Array of shape (n_cells, n_features) containing the standardised features
    """
    return StandardScaler().fit_transform(features)


def zscore_features_within_subject(features, cell_ids, rec_ids):
    """
    Z-score features within each subject to control for inter-subject variability.

    :param features: Array of shape (n_cells, n_features) containing the features to be z-scored within each subject
    :param cell_ids: Array of shape (n_cells,) containing the cell IDs for each cell
    :param rec_ids: Array of shape (n_cells,) containing the recording IDs for each cell

    :return: Array of shape (n_cells, n_features) containing the z-scored features
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
    Cluster cells based on their features using K-means clustering.

    :param features: Array of shape (n_cells, n_features) containing the features to be clustered
    :param n_clusters: Number of clusters to form

    :return: Array of shape (n_cells,) containing the cluster labels for each cell
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(features)


def hierarchical_cluster(features, method="ward", max_clusters=None, show_plot=True):
    """
    Cluster cells based on their features using hierarchical clustering.

    :param features: Array of shape (n_cells, n_features) containing the features to be clustered
    :param method: Method for hierarchical clustering (e.g., "ward", "complete", "average")
    :param max_clusters: Maximum number of clusters to form
    :param show_plot: Whether to show the dendrogram plot

    :return: Array of shape (n_cells,) containing the cluster labels for each cell, and Matplotlib figure object if show_plot is True
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
    Plot clusters in 2D using PCA for dimensionality reduction.

    :param features: Array of shape (n_cells, n_features) containing the features to be plotted
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param title: Title for the plot

    :return: Matplotlib figure object
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
    Plot clusters in 2D using UMAP for dimensionality reduction.

    :param features: Array of shape (n_cells, n_features) containing the features to be plotted
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param title: Title for the plot

    :return: Matplotlib figure object and UMAP embedding array of shape (n_cells, 2)
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
    Summarise clusters by computing the mean feature vector for each cluster.

    :param features: Array of shape (n_cells, n_features) containing the features to be summarised
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell

    :return: Dictionary mapping cluster labels to their mean feature vectors
    """
    summaries = {}
    for c in np.unique(labels):
        summaries[c] = features[labels == c].mean(axis=0)
    return summaries


def plot_cluster_tuning(features, labels, title_prefix="Cluster"):
    """
    Plot tuning profiles for each cluster.

    :param features: Array of shape (n_cells, n_features) containing the features to be plotted
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param title_prefix: Prefix for the plot titles

    :return: List of Matplotlib figure objects for each cluster's tuning profile
    """
    n_features = features.shape[1]
    clusters = np.unique(labels)
    figs = []

    for c in clusters:
        mean_vec = features[labels == c].mean(axis=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(n_features), mean_vec)
        ax.set_xticks(range(n_features))
        ax.set_xticklabels([f"Cov {i}" for i in range(n_features)])
        ax.set_ylabel("Mean feature value")
        ax.set_title(f"{title_prefix} {c} tuning profile")
        ax.grid(alpha=0.3)

        figs.append(fig)

    return figs


def suggest_labels(cluster_summaries):
    """
    Suggest functional labels for each cluster based on the most strongly tuned covariate.

    :param cluster_summaries: Dictionary mapping cluster labels to their mean feature vectors

    :return: Dictionary mapping cluster labels to suggested functional labels
    """
    labels = {}
    for c, vec in cluster_summaries.items():
        strongest = np.argmax(np.abs(vec))
        direction = "positive" if vec[strongest] > 0 else "negative"
        labels[c] = f"Strongly tuned to Covariate {strongest} ({direction})"
    return labels


def evaluate_clustering(features, labels):
    """
    Evaluate clustering quality using the silhouette score.

    :param features: Array of shape (n_cells, n_features) containing the features to be evaluated
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell

    :return: Silhouette score for the clustering
    """
    return silhouette_score(features, labels)


def map_cells_to_clusters(labels, cell_ids):
    """
    Map cell IDs to their corresponding cluster labels.

    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param cell_ids: Array of shape (n_cells,) containing the unique cell IDs

    :return: Dictionary mapping cell IDs to their cluster labels
    """
    unique_cells = np.unique(cell_ids)
    return {cell: labels[i] for i, cell in enumerate(unique_cells)}


def print_cluster_membership(labels, cell_ids):
    """
    Print the membership of each cluster.

    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param cell_ids: Array of shape (n_cells,) containing the unique cell IDs

    :return: Dictionary mapping cell IDs to their cluster labels
    """
    mapping = map_cells_to_clusters(labels, cell_ids)
    clusters = {}
    for cell, cl in mapping.items():
        clusters.setdefault(cl, []).append(int(cell))

    for cl, cells in clusters.items():
        print(f"Cluster {cl}: cells {cells}")

    return mapping


def cluster_report(features, labels, cell_ids, title_prefix="Cluster"):
    """
    Generate a comprehensive report for clustering results.

    :param features: Array of shape (n_cells, n_features) containing the features to be evaluated
    :param labels: Array of shape (n_cells,) containing the cluster labels for each cell
    :param cell_ids: Array of shape (n_cells,) containing the unique cell IDs
    :param title_prefix: String prefix for plot titles and report sections

    :return: Dictionary containing cluster membership, summaries, suggested labels, and figures for tuning profiles, PCA, and UMAP
    """
    print(f"\n{title_prefix} report:")

    score = evaluate_clustering(features, labels)
    mapping = print_cluster_membership(labels, cell_ids)
    summaries = summarise_clusters(features, labels)
    tuning_figs = plot_cluster_tuning(features, labels, title_prefix)
    suggested = suggest_labels(summaries)

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
    Run the full clustering pipeline: feature extraction, normalisation, clustering, and reporting.

    :param X: Array of shape (n_features, n_time_bins) containing the input features (covariates)
    :param Y: Array of shape (n_time_bins,) containing the spike counts for each time bin
    :param cell_ids: Array of all cell IDs
    :param rec_ids: Array of all recording IDs corresponding to each cell
    :param feature_key: String key for the feature extractor to use (e.g., "correlation" or "glm")
    :param cluster_key: String key for the clustering method to use (e.g., "kmeans" or "hierarchical")
    :param n_clusters: Number of clusters to form (required for KMeans, optional for hierarchical clustering)
    :param zscore_within_subject: Whether to z-score features within each subject to control for inter-subject variability
    :param title_prefix: Prefix for plot titles and report sections (default is "Clustering")
    :param show_report: Whether to generate and show the clustering report (default is True)

    :return: Dictionary containing clustering results and report
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
