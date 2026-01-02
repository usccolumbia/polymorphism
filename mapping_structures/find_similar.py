#!/usr/bin/env python3
"""
Polymorphism Structure Similarity Finder

A command-line tool to find the top 5 most similar crystal structures
based on graph-based topological analysis.

Usage:
    python find_similar.py <structure_id> [structure_type]

Example:
    python find_similar.py mp-1238791 olivine
    python find_similar.py cifs/mp-1000001.cif spinel
"""

import sys
import argparse
import logging
import shutil
import pickle
import warnings
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

# Suppress pymatgen warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, GRAPHS_FILE, DATASET_FILE, EMBEDDING_CONFIG
from data_loader import prepare_analysis_data
from embeddings import compute_graph_embeddings, reduce_dimensionality_tsne
from visualization import create_plotting_dataframe
from analysis import find_nearest_neighbors_tsne, perform_clustering

# Set up logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StructureSimilarityFinder:
    """Main class for finding similar crystal structures"""

    def __init__(self):
        """
        Initialize the similarity finder for all structures.
        """
        self.is_loaded = False
        self.df_plot = None
        self.valid_mp_ids = None

        logger.info(f"Initializing similarity finder for all structures")

    def load_data(self, include_icsd: bool = False) -> bool:
        """
        Load and prepare the analysis data.

        Args:
            include_icsd: Whether to include ICSD structures in the analysis

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create cache filename based on ICSD inclusion (all structures, v2 = filtered)
            icsd_suffix = "_with_icsd" if include_icsd else ""
            cache_file = Path(DATA_DIR) / f"tsne_cache_all_structures{icsd_suffix}_v2_filtered.pkl"

            # Check if cached data exists
            if cache_file.exists():
                logger.info(f"Loading cached t-SNE data from {cache_file}")
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)

                    self.df_plot = cached_data['df_plot']
                    self.valid_mp_ids = cached_data['material_ids']

                    logger.info(f"Successfully loaded cached data for {len(self.valid_mp_ids)} structures")
                    self.is_loaded = True
                    return True

                except Exception as e:
                    logger.warning(f"Failed to load cached data: {e}. Recomputing...")

            # Load data - all structures without filtering
            structure_graphs, material_ids = prepare_analysis_data(
                dataset_file=DATASET_FILE,
                graphs_file=GRAPHS_FILE,
                structure_type=None,  # Load all structures
                include_icsd=include_icsd
            )

            # Compute embeddings
            embeddings, self.valid_mp_ids = compute_graph_embeddings(
                graphs_dict=structure_graphs,
                bins=EMBEDDING_CONFIG['histogram_bins']
            )

            # Reduce dimensionality
            logger.info(f"Computing t-SNE for {len(embeddings)} structures...")
            embeddings_2d = reduce_dimensionality_tsne(
                embeddings=embeddings,
                n_components=EMBEDDING_CONFIG['tsne_components'],
                perplexity=EMBEDDING_CONFIG['tsne_perplexity'],
                random_state=EMBEDDING_CONFIG['tsne_random_state']
            )

            # Perform clustering
            cluster_labels = perform_clustering(embeddings=embeddings, method='kmeans')

            # Create plotting DataFrame
            self.df_plot = create_plotting_dataframe(
                material_ids=self.valid_mp_ids,
                embeddings_2d=embeddings_2d,
                cluster_labels=cluster_labels
            )

            # Cache the results
            logger.info(f"Saving computed t-SNE data to {cache_file}")
            cached_data = {
                'df_plot': self.df_plot,
                'material_ids': self.valid_mp_ids,
                'include_icsd': include_icsd,
                'timestamp': pd.Timestamp.now()
            }

            with open(cache_file, 'wb') as f:
                pickle.dump(cached_data, f)

            self.is_loaded = True
            logger.info(f"Successfully loaded data for {len(self.valid_mp_ids)} structures")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def find_similar_structures(self, target_structure: str, top_k: int = 5) -> List[Tuple[str, float, int]]:
        """
        Find the top-k most similar structures to the target.

        Args:
            target_structure: Material ID or CIF filename to find similarities for
            top_k: Number of similar structures to return

        Returns:
            List of tuples: (material_id, distance, cluster)
        """
        if not self.is_loaded:
            logger.error("Data not loaded. Call load_data() first.")
            return []

        try:
            # Find nearest neighbors
            similar_df = find_nearest_neighbors_tsne(
                target_mp_id=target_structure,
                df_plot=self.df_plot,
                k=top_k
            )

            # Convert to list of tuples
            results = []
            for _, row in similar_df.iterrows():
                results.append((
                    row['mp_id'],
                    round(row['tsne_distance'], 4),
                    int(row['cluster']) if 'cluster' in row else -1
                ))

            return results

        except ValueError as e:
            logger.error(f"Error finding similar structures: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

    def get_available_structures(self, limit: int = 10) -> List[str]:
        """
        Get a list of available structure IDs.

        Args:
            limit: Maximum number to return

        Returns:
            List of material IDs
        """
        if not self.is_loaded:
            return []

        return self.valid_mp_ids[:limit]

    def get_structure_info(self, structure_id: str) -> Optional[dict]:
        """
        Get information about a specific structure.

        Args:
            structure_id: Material ID to look up

        Returns:
            Dictionary with structure information or None if not found
        """
        if not self.is_loaded or self.df_plot is None:
            return None

        row = self.df_plot[self.df_plot['mp_id'] == structure_id]
        if len(row) == 0:
            return None

        return {
            'mp_id': structure_id,
            'x_coordinate': round(row['x'].iloc[0], 4),
            'y_coordinate': round(row['y'].iloc[0], 4),
            'cluster': int(row['cluster'].iloc[0]) if 'cluster' in row.columns else -1
        }

    def plot_similar_structures(self, target_structure: str, similar_structures: List[Tuple[str, float, int]],
                               save_plots: bool = True, output_dir: str = None) -> None:
        """
        Create polyhedral plots for the target structure and its similar structures.

        Args:
            target_structure: The query structure ID
            similar_structures: List of (structure_id, distance, cluster) tuples
            save_plots: Whether to save the plots to files
            output_dir: Directory to save the plot (if None, saves to current directory)
        """
        from view_polyhedra import create_multi_structure_polyhedra_plot

        # Prepare structure IDs and titles
        structure_ids = [target_structure] + [struct_id for struct_id, _, _ in similar_structures]

        titles = ["Query Structure"]
        for i, (struct_id, distance, cluster) in enumerate(similar_structures, 1):
            titles.append(f"Similar #{i}: {struct_id} (dist: {distance:.4f}, cluster: {cluster})")
        try:
            # Create the multi-panel plot - only save HTML, not PNG to avoid hanging
            fig = create_multi_structure_polyhedra_plot(
                structure_ids=structure_ids,
                titles=titles,
                save_html=save_plots,
                save_png=False,  # Disable PNG to prevent hanging
                output_dir=output_dir
            )

            # Don't show the plot interactively as it causes the program to hang
            # fig.show()

            logger.info(f"Successfully created polyhedral plots for {len(structure_ids)} structures")
            if save_plots and output_dir:
                print(f"✅ Plot saved to {output_dir}/polyhedra_comparison_*.html")
            elif save_plots:
                print(f"✅ Plots saved to polyhedra_comparison_*.html file")

        except Exception as e:
            logger.error(f"Error creating polyhedral plots: {e}")
            print(f"❌ Error creating plots: {e}")
            print("   Make sure the CIF files exist in the dataset/cifs/ directory")

    def copy_similar_cifs(self, target_structure: str, similar_structures: List[Tuple[str, float, int]],
                         output_dir: str = 'similar_structures') -> str:
        """
        Copy CIF files of similar structures to a new folder.

        Args:
            target_structure: The query structure ID
            similar_structures: List of (structure_id, distance, cluster) tuples
            output_dir: Base directory for output (default: 'similar_structures')

        Returns:
            Path to the created folder
        """
        from config import DATA_DIR

        # Create folder name from target structure ID
        # Clean up the ID for folder name (replace slashes and special chars)
        folder_name = target_structure.replace('/', '_').replace('.cif', '')
        if folder_name.startswith('cifs_'):
            folder_name = folder_name[5:]  # Remove 'cifs_' prefix

        # Create output directory
        output_path = Path(output_dir) / folder_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine CIF paths for MP and ICSD
        mp_cif_base = Path(r"C:\Users\moons\OneDrive\Documents\spring25\mp cifs\drive-download-20250114T195716Z-001\cifs\cifs")
        icsd_cif_base = Path(__file__).parent.parent / "icsd" / "CIF"

        copied_files = []
        missing_files = []

        # Copy query structure
        try:
            if target_structure.startswith('icsd/'):
                icsd_id = target_structure.split('/')[1]
                source = icsd_cif_base / f"{icsd_id}.cif"
            elif target_structure.startswith('cifs/'):
                mp_id = target_structure.split('/')[1].replace('.cif', '')
                source = mp_cif_base / f"{mp_id}.cif"
            else:
                # Assume it's an MP ID without prefix
                source = mp_cif_base / f"{target_structure.replace('.cif', '')}.cif"

            if source.exists():
                dest = output_path / f"query_{source.name}"
                shutil.copy2(source, dest)
                copied_files.append(f"query_{source.name}")
                logger.info(f"Copied query structure: {source.name}")
            else:
                logger.warning(f"Query structure CIF not found: {source}")
        except Exception as e:
            logger.error(f"Error copying query structure: {e}")

        # Copy similar structures
        for i, (struct_id, distance, cluster) in enumerate(similar_structures, 1):
            try:
                if struct_id.startswith('icsd/'):
                    icsd_id = struct_id.split('/')[1]
                    source = icsd_cif_base / f"{icsd_id}.cif"
                elif struct_id.startswith('cifs/'):
                    mp_id = struct_id.split('/')[1].replace('.cif', '')
                    source = mp_cif_base / f"{mp_id}.cif"
                else:
                    # Assume it's an MP ID
                    source = mp_cif_base / f"{struct_id.replace('.cif', '')}.cif"

                if source.exists():
                    # Add ranking and distance info to filename
                    dest_name = f"rank{i:02d}_dist{distance:.4f}_{source.name}"
                    dest = output_path / dest_name
                    shutil.copy2(source, dest)
                    copied_files.append(dest_name)
                    logger.info(f"Copied similar structure {i}: {source.name}")
                else:
                    missing_files.append(struct_id)
                    logger.warning(f"CIF file not found: {source}")

            except Exception as e:
                logger.error(f"Error copying {struct_id}: {e}")
                missing_files.append(struct_id)

        # Print summary
        print(f"\n📁 Copied {len(copied_files)} CIF files to: {output_path.absolute()}")
        if missing_files:
            print(f"⚠️  Could not find {len(missing_files)} CIF files: {missing_files[:3]}...")

        return str(output_path.absolute())

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Find similar crystal structures using topological analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python find_similar.py mp-1238791 olivine
  python find_similar.py cifs/mp-1000001.cif spinel --top-k 10
  python find_similar.py --list olivine
  python find_similar.py mp-1238791 olivine --plot
        """
    )

    parser.add_argument(
        'structure_id',
        nargs='?',
        help='Material ID or CIF filename to find similarities for'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of similar structures to return (default: 5)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available structures instead of finding similarities'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show information about the target structure'
    )

    parser.add_argument(
        '--include-icsd',
        action='store_true',
        help='Include ICSD structures in the similarity search'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate polyhedral plots for the similar structures'
    )

    parser.add_argument(
        '--copy-cifs',
        action='store_true',
        help='Copy similar structure CIF files to a new folder'
    )

    args = parser.parse_args()

    # Initialize finder
    finder = StructureSimilarityFinder()

    # Load data
    print(f"Loading structure data (all structures, no filtering)...")
    if not finder.load_data(include_icsd=args.include_icsd):
        print("❌ Failed to load data. Check your data files and try again.")
        sys.exit(1)

    total_structures = len(finder.valid_mp_ids)
    mp_count = sum(1 for id in finder.valid_mp_ids if not str(id).startswith('icsd/'))
    icsd_count = total_structures - mp_count

    print(f"✅ Loaded {total_structures} structures")
    if args.include_icsd:
        print(f"   MP: {mp_count} structures")
        print(f"   ICSD: {icsd_count} structures")

    # Handle different modes
    if args.list:
        # List available structures
        print(f"\n📋 Available structures (first 20):")
        structures = finder.get_available_structures(limit=20)
        for i, structure in enumerate(structures, 1):
            print(f"  {i:2d}. {structure}")

    elif args.structure_id:
        # Show structure info if requested
        if args.info:
            info = finder.get_structure_info(args.structure_id)
            if info:
                print(f"\n📊 Structure Information for {args.structure_id}:")
                print(f"  Cluster: {info['cluster']}")
                print(f"  t-SNE coordinates: ({info['x_coordinate']}, {info['y_coordinate']})")
            else:
                print(f"❌ Structure {args.structure_id} not found in dataset")
                sys.exit(1)

        # Find similar structures
        print(f"\n🔍 Finding top {args.top_k} similar structures to {args.structure_id}...")

        similar = finder.find_similar_structures(args.structure_id, args.top_k)

        if not similar:
            print(f"❌ Could not find similar structures for {args.structure_id}")
            print("   Make sure the structure ID is correct and exists in the dataset.")
            sys.exit(1)

        print(f"\n🎯 Top {len(similar)} most similar structures:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Structure ID':<15} {'Distance':<10} {'Cluster':<8}")
        print("-" * 70)

        for i, (mp_id, distance, cluster) in enumerate(similar, 1):
            print(f"{i:<4} {mp_id:<15} {distance:<10.4f} {cluster:<8}")

        # Copy CIF files if requested and get output directory
        output_folder = None
        if args.copy_cifs:
            print(f"\n📋 Copying CIF files...")
            output_folder = finder.copy_similar_cifs(args.structure_id, similar)

        # Generate polyhedral plots if requested
        if args.plot:
            print(f"\n🎨 Generating polyhedral plots...")
            finder.plot_similar_structures(args.structure_id, similar, save_plots=True, output_dir=output_folder)

    else:
        # No arguments provided - show help
        parser.print_help()

if __name__ == "__main__":
    main()
