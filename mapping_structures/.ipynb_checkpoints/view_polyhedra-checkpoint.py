import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from collections import Counter

def count_and_visualize_polyhedra_interactive(cif_filename, save_html=True, save_png=True, open_browser=True):
    """
    Counts and visualizes the polyhedra in a crystal structure file using interactive Plotly.
    Also calculates angle distortion for each polyhedron.
    """
    # Load the structure
    path="../dataset/"
    
    structure = Structure.from_file(path+cif_filename)
    # structure = structure.get_primitive_structure()
    # structure.make_supercell((1,1,1))
    # Use CrystalNN to determine coordination environments
    # nn_analyzer = CrystalNN(distance_cutoffs=(0, 1.0), search_cutoff=1.0)
    nn_analyzer = CrystalNN()
    # Collect coordination numbers, polyhedra vertices, and angle distortions
    coordination_numbers = []
    polyhedra_data = []
    
    cation_sites = []
    ignore = ["O", "S", "N", "F", "Cl", "Br", "I", "H"]  # ignored these as center atoms
    for site_idx in range(len(structure)):
        site = structure[site_idx]
        if site.specie.symbol not in ignore:
            cation_sites.append(site_idx)
    
    # Analyze each cation site
    for site_idx in cation_sites:
        nn_info = nn_analyzer.get_nn_info(structure, site_idx)
        cn = len(nn_info)
        coordination_numbers.append(cn)
        
        center = structure[site_idx].coords
        vertices = [info['site'].coords for info in nn_info]
        vertex_elements = [info['site'].specie.symbol for info in nn_info]
        # other_centers = [structure[i].coords for i in cation_sites if i != site_idx]

        # filtered_vertices = []
        # for info in nn_info:
        #     dist_to_others = [np.linalg.norm(info['site'].coords - oc) for oc in other_centers]
        #     # Only keep if it's not extremely close to another cation center
        #     if all(d > 1.5 for d in dist_to_others):  # Threshold may be tuned (e.g., 1.5 Å)
        #         filtered_vertices.append(info['site'].coords)
        
        # vertices = filtered_vertices

        vertex_indices = [info['site_index'] for info in nn_info]
        polyhedron_type = get_polyhedron_type(cn)
    
        # === NEW: angle distortion ===
        if cn >= 3:
            angles = []
            for i in range(cn):
                for j in range(i + 1, cn):
                    v1 = vertices[i] - center
                    v2 = vertices[j] - center
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    if angle < 150:
                        angles.append(angle)
            angles = np.array(angles)
            if cn == 4:
                ideal_angle = 109.47
            elif cn == 6:
                ideal_angle = 90.0
            elif cn == 8:
                ideal_angle = 70.5
            else:
                ideal_angle = np.mean(angles)
            angle_distortion = np.sqrt(np.mean((angles - ideal_angle) ** 2))
        else:
            angle_distortion = None
    
        # === NEW: Face triplets (by index) ===
        face_indices = []
        if len(vertices) >= 4:
            try:
                hull = ConvexHull(vertices)
                for simplex in hull.simplices:
                    face = sorted([vertex_indices[i] for i in simplex])
                    face_indices.append(face)
            except:
                pass  # hull may fail on degenerate configs
    
        # Store info
        polyhedra_data.append({
            'center': center,
            'vertices': vertices,
            'vertex_indices': vertex_indices,
            'faces': face_indices,  # <<< new
            'cn': cn,
            'type': polyhedron_type,
            'element': structure[site_idx].specie.symbol,
            'angle_distortion': angle_distortion,
            'center_index': site_idx,
            've_elements': vertex_elements
        })

    
    # Count polyhedron types
    cn_counts = Counter(coordination_numbers)
    polyhedra_counts = {get_polyhedron_type(cn): count for cn, count in cn_counts.items()}
    
    # Create interactive visualization
    fig = create_interactive_polyhedra_plot(structure, polyhedra_data, cif_filename)
    
    # Save plots (same as before)
    # [Your save_html and save_png part stays same]

    if open_browser:
        fig.show()
    
    return polyhedra_counts, cn_counts, polyhedra_data


    


def get_polyhedron_type(coordination_number):
    """
    Get a polyhedron type name based on coordination number.
    """
    polyhedron_types = {
        1: "Mono",
        2: "Linear",
        3: "Triangular",
        4: "Tetrahedral",
        5: "Trigonal bipyramidal",
        6: "Octahedral",
        7: "Pentagonal bipyramidal",
        8: "Cubic/Square antiprismatic",
        9: "Tricapped trigonal prismatic",
        10: "Bicapped square antiprismatic",
        11: "Pentagonal antiprismatic",
        12: "Icosahedral/Cuboctahedral"
    }
    return polyhedron_types.get(coordination_number, f"CN-{coordination_number}")


def create_interactive_polyhedra_plot(structure, polyhedra_data, filename):
    """
    Create an interactive 3D plot with Plotly.
    """
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    lattice = structure.lattice
    fig = add_unit_cell_to_plot(fig, lattice)
    
    unique_types = list(set(p['type'] for p in polyhedra_data))
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in plt_colors(len(unique_types))]
    color_map = {t: colors[i] for i, t in enumerate(unique_types)}
    
    added_to_legend = set()
    
    for polyhedron in polyhedra_data:
        center = polyhedron['center']
        vertices = polyhedron['vertices']
        poly_type = polyhedron['type']
        element = polyhedron['element']
        color = color_map[poly_type]
        
        legend_key = f"{element} ({poly_type})"
        show_in_legend = legend_key not in added_to_legend
        if show_in_legend:
            added_to_legend.add(legend_key)
        
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers', marker=dict(size=8, color='black'),
            name=legend_key, showlegend=show_in_legend, hoverinfo='name',
        ))
        
        x_vertices = [v[0] for v in vertices]
        y_vertices = [v[1] for v in vertices]
        z_vertices = [v[2] for v in vertices]
        
        for v, elem in zip(vertices, polyhedron['ve_elements']):
            fig.add_trace(go.Scatter3d(
                x=[v[0]], y=[v[1]], z=[v[2]],
                mode='markers',
                marker=dict(size=6, color='red'),  # optionally color-code by element
                name=elem,
                showlegend=False,
                hovertext=elem,
                hoverinfo='text'
            ))

        
        
        if len(vertices) >= 4:
            try:
                hull = ConvexHull(vertices)
                faces = [simplex.tolist() for simplex in hull.simplices]
                i, j, k = zip(*faces)
                
                fig.add_trace(go.Mesh3d(
                    x=x_vertices, y=y_vertices, z=z_vertices,
                    i=i, j=j, k=k, color=color, opacity=0.4,
                    name=legend_key, showlegend=False, hoverinfo='name',
                ))
                for simplex in hull.simplices:
                    edge_indices = [(simplex[0], simplex[1]),
                                    (simplex[1], simplex[2]),
                                    (simplex[2], simplex[0])]
                    for idx1, idx2 in edge_indices:
                        fig.add_trace(go.Scatter3d(
                            x=[vertices[idx1][0], vertices[idx2][0]],
                            y=[vertices[idx1][1], vertices[idx2][1]],
                            z=[vertices[idx1][2], vertices[idx2][2]],
                            mode='lines',
                            line=dict(color='black', width=2),
                            showlegend=False,
                            hoverinfo='skip',
                        ))
            except:
                pass
    
        # Add formula as title with subscript and large font
    formula = structure.composition.reduced_formula
    formula_with_sub = ''.join([f"<sub>{c}</sub>" if c.isdigit() else c for c in formula])

    fig.update_layout(
        title=dict(
            text=f"<b>{formula_with_sub}</b>",
            x=0.5,
            y=0.98,
            font=dict(size=32),
            xanchor="center",
            yanchor="top"
        ),
        scene=dict(
            xaxis_title='a (Å)', 
            yaxis_title='b (Å)', 
            zaxis_title='c (Å)',
            xaxis=dict(title_font=dict(size=28), tickfont=dict(size=18), tickmode="linear", showline=True, zeroline=False, dtick=2), 
            yaxis=dict(title_font=dict(size=28), tickfont=dict(size=18), tickmode="linear", showline=True, zeroline=False, dtick=2),
            zaxis=dict(title_font=dict(size=28), tickfont=dict(size=18), tickmode="auto", showline=True, zeroline=False),
            aspectmode='cube',
        ),
        legend_title="Polyhedra Types",
        margin=dict(l=0, r=0, b=0, t=80),  # Leave top margin for title
        autosize=True,
        scene_camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.5, y=1.5, z=1.5))
    )



    
    return fig
import numpy as np
import matplotlib.pyplot as plt



def plt_colors(n):
    """
    Generate `n` distinct colors using matplotlib colormap.
    """
    cmap = plt.get_cmap("tab10")  # You can change this to "viridis", "plasma", etc.
    return [cmap(i / n)[:3] for i in range(n)]


def add_unit_cell_to_plot(fig, lattice):
    """
    Add unit cell edges to the Plotly figure.
    """
    a, b, c = lattice.matrix
    corners = [np.array([0, 0, 0]), a, b, c, a+b, a+c, b+c, a+b+c]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    
    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[corners[start][0], corners[end][0]],
            y=[corners[start][1], corners[end][1]],
            z=[corners[start][2], corners[end][2]],
            mode='lines', line=dict(color='black', width=2), showlegend=False,
        ))
    return fig
