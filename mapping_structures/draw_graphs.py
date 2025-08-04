from itertools import combinations
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def remove_overlapping_polyhedra(polyhedra_data, distance_threshold=1.5):
    """
    Removes overlapping polyhedra based on center-to-center distance and angle distortion.

    Args:
        polyhedra_data (list of dict): List of polyhedra info, each containing 'center', 'angle_distortion', etc.
        distance_threshold (float): Distance below which two centers are considered overlapping (default: 1.5 Å).

    Returns:
        cleaned_data (list): Filtered list of polyhedra with overlaps removed.
        removed_indices (set): Set of indices that were removed.
    """
    import numpy as np

    to_remove = set()

    for i in range(len(polyhedra_data)):
        for j in range(i + 1, len(polyhedra_data)):
            ci = np.array(polyhedra_data[i]['center'])
            cj = np.array(polyhedra_data[j]['center'])
            dist = np.linalg.norm(ci - cj)

            if dist < distance_threshold:
                ai = polyhedra_data[i]['angle_distortion']
                aj = polyhedra_data[j]['angle_distortion']

                # Handle None cases conservatively
                if ai is None and aj is None:
                    to_remove.add(j)  # arbitrarily remove one
                elif ai is None:
                    to_remove.add(i)
                elif aj is None:
                    to_remove.add(j)
                else:
                    # Remove the one with higher distortion
                    if ai < aj:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)

    cleaned_data = [p for idx, p in enumerate(polyhedra_data) if idx not in to_remove]
    return cleaned_data, to_remove


def get_polyhedron_sharing_pairs_verbose(z):
    """
    Return:
        - face_tri_pairs: dict of (i, j) → list of shared triangle faces (length-3)
        - face_quad_pairs: dict of (i, j) → list of shared quad faces (length-4)
        - edge_pairs: dict of (i, j) → shared 2-atom edge list
        - point_pairs: dict of (i, j) → shared 1-atom vertex list
    """
    face_tri_pairs = defaultdict(list)
    face_quad_pairs = defaultdict(list)
    face_map = defaultdict(list)

    for i, poly in enumerate(z):
        for face in poly['faces']:
            key = tuple(sorted(face))
            face_map[key].append(i)

    for face, polys in face_map.items():
        if len(polys) >= 2:
            for i1, i2 in combinations(sorted(polys), 2):
                if len(face) == 3:
                    face_tri_pairs[(i1, i2)].append(face)
                elif len(face) == 4:
                    face_quad_pairs[(i1, i2)].append(face)

    edge_pairs = {}
    point_pairs = {}

    # Combine all face-sharing pairs to skip them in edge/pnt evaluation
    all_face_pairs = set(face_tri_pairs) | set(face_quad_pairs)

    for i1, i2 in combinations(range(len(z)), 2):
        if (i1, i2) in all_face_pairs:
            continue

        vi1 = set(z[i1]['vertex_indices'])
        vi2 = set(z[i2]['vertex_indices'])
        shared = sorted(vi1 & vi2)

        if len(shared) == 2:
            edge_pairs[(i1, i2)] = shared
        elif len(shared) == 1:
            point_pairs[(i1, i2)] = shared

    return face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs



def draw_polyhedron_sharing_graph(face_tri_pairs, face_quad_pairs, edge_pairs, point_pairs):
    G = nx.MultiGraph()

    # Collect all node indices from all pair types
    all_keys = (
        list(face_tri_pairs.keys()) +
        list(face_quad_pairs.keys()) +
        list(edge_pairs.keys()) +
        list(point_pairs.keys())
    )
    node_ids = set(i for pair in all_keys for i in pair)
    G.add_nodes_from(sorted(node_ids))

    # Add triangle face-sharing edges (blue, dashed)
    for (i1, i2), faces in face_tri_pairs.items():
        G.add_edge(i1, i2, type='face-tri', faces=faces)

    # Add quad face-sharing edges (blue, solid)
    for (i1, i2), faces in face_quad_pairs.items():
        G.add_edge(i1, i2, type='face-quad', faces=faces)

    # Add edge-sharing edges (red, dashdot)
    for (i1, i2), shared_vertices in edge_pairs.items():
        G.add_edge(i1, i2, type='edge', shared_vertices=shared_vertices)

    # Add point-sharing edges (gray, dotted)
    for (i1, i2), shared_vertex in point_pairs.items():
        G.add_edge(i1, i2, type='point', shared_vertex=shared_vertex)

    return G


