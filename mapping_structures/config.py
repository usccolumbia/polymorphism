"""
Configuration constants for the polymorphism analysis pipeline.
"""

# Data paths
DATA_DIR = "dataset"
GRAPHS_FILE = f"{DATA_DIR}/polyhedron_graphs3.pkl"
DATASET_FILE = f"{DATA_DIR}/dataset.csv"

# Embedding parameters
EMBEDDING_CONFIG = {
    'histogram_bins': 10,
    'tsne_components': 2,
    'tsne_perplexity': 10,
    'tsne_random_state': 42,
    'max_neighbors': 5
}

# Plotting parameters
PLOT_CONFIG = {
    'width': 800,
    'height': 600,
    'marker_size': 10,
    'font_size': 16,
    'axis_font_size': 18,
    'dpi': 300
}

# Structure type filters - stoichiometry ratios
# Format: (element_count, sorted_atom_ratios)
STRUCTURE_TYPES = {
    'spinel': (3, [1, 2, 4]),
    'chalcopyrite': (3, [1, 1, 2]),
    'pyrochlore': (3, [2, 2, 7]),
    'scheelite': (3, [1, 1, 4]),
    'olivine': (3, [1, 2, 4])
}
