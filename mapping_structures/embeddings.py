"""
Graph embedding functions for converting polyhedron graphs to numerical representations.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

def edge_type_histogram_embedding(graph, bins: int = 10) -> np.ndarray:
    """
    Convert a polyhedron connectivity graph to a histogram-based embedding.

    Args:
        graph: NetworkX graph with edge types ('face-tri', 'face-quad', 'edge', 'point')
        bins: Number of bins for histograms

    Returns:
        Normalized histogram embedding as numpy array
    """
    # Initialize degree dictionaries for each edge type
    face_tri_deg = {node: 0 for node in graph.nodes}
    face_quad_deg = {node: 0 for node in graph.nodes}
    edge_deg = {node: 0 for node in graph.nodes}
    point_deg = {node: 0 for node in graph.nodes}

    # Count connections by edge type
    for u, v, data in graph.edges(data=True):
        edge_type = data.get("type")

        if edge_type == "face-tri":
            face_tri_deg[u] += 1
            face_tri_deg[v] += 1
        elif edge_type == "face-quad":
            face_quad_deg[u] += 1
            face_quad_deg[v] += 1
        elif edge_type == "edge":
            edge_deg[u] += 1
            edge_deg[v] += 1
        elif edge_type == "point":
            point_deg[u] += 1
            point_deg[v] += 1

    # Compute histograms for each edge type
    tri_values = list(face_tri_deg.values())
    quad_values = list(face_quad_deg.values())
    edge_values = list(edge_deg.values())
    point_values = list(point_deg.values())

    tri_hist, _ = np.histogram(tri_values, bins=bins, range=(0, bins))
    quad_hist, _ = np.histogram(quad_values, bins=bins, range=(0, bins))
    edge_hist, _ = np.histogram(edge_values, bins=bins, range=(0, bins))
    point_hist, _ = np.histogram(point_values, bins=bins, range=(0, bins))

    # Normalize histograms (avoid division by zero)
    def normalize_histogram(hist: np.ndarray) -> np.ndarray:
        total = hist.sum()
        return hist / total if total > 0 else np.zeros(bins)

    tri_feat = normalize_histogram(tri_hist)
    quad_feat = normalize_histogram(quad_hist)
    edge_feat = normalize_histogram(edge_hist)
    point_feat = normalize_histogram(point_hist)

    # Concatenate all features
    embedding = np.concatenate([tri_feat, quad_feat, edge_feat, point_feat])

    return embedding

def compute_graph_embeddings(graphs_dict: Dict[str, Any], bins: int = 10) -> Tuple[np.ndarray, List[str]]:
    """
    Compute embeddings for a collection of graphs.

    Args:
        graphs_dict: Dictionary mapping material IDs to graph objects
        bins: Number of bins for histogram embeddings

    Returns:
        Tuple of (embeddings_array, valid_material_ids)
    """
    embeddings = []
    valid_mp_ids = []

    total_graphs = len(graphs_dict)
    logger.info(f"Computing embeddings for {total_graphs} graphs...")

    for i, (mp_id, graph) in enumerate(graphs_dict.items()):
        if i % 100 == 0:
            logger.info(f"Processing graph {i+1}/{total_graphs}")

        try:
            embedding = edge_type_histogram_embedding(graph, bins=bins)

            # Check for NaN values
            if not np.isnan(embedding).any():
                embeddings.append(embedding)
                valid_mp_ids.append(mp_id)
            else:
                logger.warning(f"NaN values in embedding for {mp_id}")

        except Exception as e:
            logger.error(f"Error computing embedding for {mp_id}: {e}")

    embeddings_array = np.array(embeddings)
    logger.info(f"Successfully computed {len(valid_mp_ids)} embeddings "
               f"(shape: {embeddings_array.shape})")

    return embeddings_array, valid_mp_ids

def reduce_dimensionality_tsne(embeddings: np.ndarray,
                               n_components: int = 2,
                               perplexity: int = 10,
                               random_state: int = 42) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using t-SNE.

    Args:
        embeddings: High-dimensional embeddings array
        n_components: Target dimensionality
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility

    Returns:
        Reduced dimensionality embeddings
    """
    logger.info(f"Running t-SNE on {embeddings.shape[0]} samples, "
               f"reducing to {n_components}D (perplexity={perplexity})")

    tsne = TSNE(n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                verbose=1)

    reduced_embeddings = tsne.fit_transform(embeddings)

    logger.info("t-SNE completed")
    return reduced_embeddings
