"""
Analysis functions for finding similar structures and performing clustering.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)

def find_nearest_neighbors_tsne(target_mp_id: str,
                               df_plot: pd.DataFrame,
                               k: int = 5) -> pd.DataFrame:
    """
    Find k nearest neighbors in t-SNE space for a target material.

    Args:
        target_mp_id: Material ID to find neighbors for
        df_plot: DataFrame with 'mp_id', 'x', 'y' columns
        k: Number of neighbors to return

    Returns:
        DataFrame with neighbor information

    Raises:
        ValueError: If target_mp_id not found in df_plot
    """
    if target_mp_id not in df_plot['mp_id'].values:
        raise ValueError(f"Target material '{target_mp_id}' not found in dataset")

    # Get target point coordinates
    target_row = df_plot[df_plot['mp_id'] == target_mp_id]
    target_point = target_row[['x', 'y']].values[0]

    # Calculate Euclidean distances to all other points
    df_plot_copy = df_plot.copy()
    df_plot_copy['tsne_distance'] = df_plot_copy.apply(
        lambda row: np.linalg.norm([row['x'] - target_point[0],
                                   row['y'] - target_point[1]]),
        axis=1
    )

    # Get k nearest neighbors (excluding the target itself)
    neighbors = (df_plot_copy[df_plot_copy['mp_id'] != target_mp_id]
                .nsmallest(k, 'tsne_distance'))

    # Select relevant columns
    result_columns = ['mp_id', 'tsne_distance']
    if 'cluster' in neighbors.columns:
        result_columns.append('cluster')

    neighbors_result = neighbors[result_columns].copy()
    neighbors_result = neighbors_result.reset_index(drop=True)

    logger.info(f"Found {len(neighbors_result)} nearest neighbors for {target_mp_id}")
    return neighbors_result

def perform_clustering(embeddings: np.ndarray,
                      n_clusters: Optional[int] = None,
                      method: str = 'kmeans',
                      random_state: int = 42) -> np.ndarray:
    """
    Perform clustering on embeddings.

    Args:
        embeddings: Embeddings array
        n_clusters: Number of clusters (if None, will try to determine optimal)
        method: Clustering method ('kmeans')
        random_state: Random seed

    Returns:
        Cluster labels array
    """
    if method == 'kmeans':
        if n_clusters is None:
            # Try to find optimal number of clusters
            n_clusters = find_optimal_clusters_kmeans(embeddings, max_clusters=10)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        logger.info(f"K-means clustering completed with {n_clusters} clusters")
        return labels

    else:
        raise ValueError(f"Unsupported clustering method: {method}")

def find_optimal_clusters_kmeans(embeddings: np.ndarray,
                                max_clusters: int = 10) -> int:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Embeddings array
        max_clusters: Maximum number of clusters to try

    Returns:
        Optimal number of clusters
    """
    if len(embeddings) < max_clusters:
        max_clusters = len(embeddings) - 1

    silhouette_scores = []

    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)

    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we start from 2
    best_score = max(silhouette_scores)

    logger.info(f"Optimal clusters: {optimal_clusters} (silhouette score: {best_score:.3f})")
    return optimal_clusters

def analyze_cluster_composition(df_plot: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the composition of clusters.

    Args:
        df_plot: DataFrame with cluster labels

    Returns:
        DataFrame with cluster statistics
    """
    if 'cluster' not in df_plot.columns:
        raise ValueError("DataFrame must contain 'cluster' column")

    cluster_stats = df_plot.groupby('cluster').agg({
        'mp_id': 'count',
        'x': ['mean', 'std'],
        'y': ['mean', 'std']
    }).round(3)

    cluster_stats.columns = ['count', 'x_mean', 'x_std', 'y_mean', 'y_std']
    cluster_stats = cluster_stats.reset_index()

    logger.info(f"Cluster analysis completed for {len(cluster_stats)} clusters")
    return cluster_stats

def get_cluster_members(cluster_id: int, df_plot: pd.DataFrame) -> List[str]:
    """
    Get all material IDs belonging to a specific cluster.

    Args:
        cluster_id: Cluster identifier
        df_plot: DataFrame with cluster labels

    Returns:
        List of material IDs in the cluster
    """
    if 'cluster' not in df_plot.columns:
        raise ValueError("DataFrame must contain 'cluster' column")

    members = df_plot[df_plot['cluster'] == cluster_id]['mp_id'].tolist()
    logger.info(f"Cluster {cluster_id} has {len(members)} members")
    return members
