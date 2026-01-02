"""
Data loading utilities for the polymorphism analysis pipeline.
"""

import pickle
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

def load_graph_data(graphs_file: str) -> Dict[str, Any]:
    """
    Load pre-computed polyhedron graphs from pickle file.

    Args:
        graphs_file: Path to the pickle file containing graphs

    Returns:
        Dictionary mapping material IDs to graph objects

    Raises:
        FileNotFoundError: If graphs file doesn't exist
        pickle.UnpicklingError: If file is corrupted
    """
    graphs_path = Path(graphs_file)

    if not graphs_path.exists():
        raise FileNotFoundError(f"Graphs file not found: {graphs_path}")

    try:
        with open(graphs_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} graphs from {graphs_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading graphs from {graphs_path}: {e}")
        raise

def load_dataset(dataset_file: str, polymorphs_only: bool = True) -> pd.DataFrame:
    """
    Load the materials dataset.

    Args:
        dataset_file: Path to the CSV dataset file
        polymorphs_only: If True, filter to structures with polymorphs > 0

    Returns:
        Filtered DataFrame

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty or malformed
    """
    dataset_path = Path(dataset_file)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset with {len(df)} entries from {dataset_path}")

        if polymorphs_only:
            original_count = len(df)
            df = df[df['polymorphs'] == 1]
            logger.info(f"Filtered to {len(df)} polymorph structures (from {original_count})")

        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {dataset_path}: {e}")
        raise

def filter_graphs_by_mp_ids(graph_data: Dict[str, Any], mp_ids: List[str]) -> Dict[str, Any]:
    """
    Filter graph data to only include specified material IDs.

    Args:
        graph_data: Dictionary of all graphs
        mp_ids: List of material IDs to keep

    Returns:
        Filtered dictionary containing only requested graphs
    """
    filtered_graphs = {}
    missing_ids = []

    for mp_id in mp_ids:
        if mp_id in graph_data:
            filtered_graphs[mp_id] = graph_data[mp_id]
        else:
            missing_ids.append(mp_id)

    if missing_ids:
        logger.warning(f"Missing graphs for {len(missing_ids)} material IDs: {missing_ids[:5]}...")

    logger.info(f"Filtered to {len(filtered_graphs)} graphs from {len(mp_ids)} requested IDs")
    return filtered_graphs

def filter_degenerate_graphs(graph_data: Dict[str, Any], min_nodes: int = 3) -> Tuple[Dict[str, Any], List[str]]:
    """
    Filter out graphs with too few nodes or no edges (degenerate structures).
    
    Args:
        graph_data: Dictionary of graphs
        min_nodes: Minimum number of nodes required (default: 3)
        
    Returns:
        Tuple of (filtered_graphs_dict, list_of_removed_ids)
    """
    filtered_graphs = {}
    removed_ids = []
    
    for mat_id, G in graph_data.items():
        # Check if graph is valid
        if G is None:
            removed_ids.append(mat_id)
            continue
            
        num_nodes = G.number_of_nodes()
        
        # Filter out graphs with too few nodes
        if num_nodes < min_nodes:
            removed_ids.append(mat_id)
            continue
            
        # Keep this graph
        filtered_graphs[mat_id] = G
    
    if removed_ids:
        logger.info(f"Filtered out {len(removed_ids)} degenerate graphs (nodes < {min_nodes})")
    
    return filtered_graphs, removed_ids

def prepare_analysis_data(dataset_file: str, graphs_file: str, structure_type: str = None, include_icsd: bool = False):
    """
    Prepare data for analysis by loading and filtering datasets.

    Args:
        dataset_file: Path to dataset CSV
        graphs_file: Path to graphs pickle file
        structure_type: Type of structure to filter for (None = all structures)
        include_icsd: Whether to include ICSD structures

    Returns:
        Tuple of (filtered_graphs_dict, material_ids_list)
    """
    from filters import filter_dataframe_by_structure

    # Load MP data
    df = load_dataset(dataset_file)
    graph_data = load_graph_data(graphs_file)

    # Filter by structure type if specified
    if structure_type:
        df_filtered = filter_dataframe_by_structure(df, structure_type)
        logger.info(f"Filtered to {len(df_filtered)} {structure_type} structures")
    else:
        df_filtered = df
        logger.info(f"Using all {len(df_filtered)} structures")

    # Get material IDs
    mp_ids = df_filtered['mp_id'].tolist()

    # Filter MP graphs
    filtered_graphs = filter_graphs_by_mp_ids(graph_data, mp_ids)
    
    # Remove degenerate MP graphs (require at least 5 nodes for meaningful topology)
    filtered_graphs, removed_mp = filter_degenerate_graphs(filtered_graphs, min_nodes=5)

    # Add ICSD structures if requested
    if include_icsd:
        try:
            # ICSD file is in the same directory as this script
            script_dir = Path(__file__).parent
            icsd_graphs_file = script_dir / "icsd_polyhedron_graphs_full.pkl"
            if icsd_graphs_file.exists():
                icsd_graph_data = load_graph_data(str(icsd_graphs_file))
                logger.info(f"Loaded {len(icsd_graph_data)} ICSD graphs")

                # Filter degenerate ICSD graphs first (require at least 5 nodes)
                icsd_graph_data_filtered, removed_icsd = filter_degenerate_graphs(icsd_graph_data, min_nodes=5)
                
                # Add ICSD graphs with prefixed keys to avoid conflicts
                for icsd_key, icsd_graph in icsd_graph_data_filtered.items():
                    prefixed_key = f"icsd/{icsd_key}"
                    filtered_graphs[prefixed_key] = icsd_graph

                # Add ICSD IDs to material_ids list
                icsd_ids = [f"icsd/{key}" for key in icsd_graph_data_filtered.keys()]
                mp_ids.extend(icsd_ids)

                logger.info(f"Added {len(icsd_ids)} ICSD structures to analysis (removed {len(removed_icsd)} degenerate)")
            else:
                logger.warning(f"ICSD graphs file not found: {icsd_graphs_file}")
        except Exception as e:
            logger.warning(f"Could not load ICSD data: {e}")

    return filtered_graphs, mp_ids
