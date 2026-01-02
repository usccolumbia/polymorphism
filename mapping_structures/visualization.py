"""
Visualization functions for plotting embeddings and analysis results.
"""

import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

def create_embedding_plotly(df_plot: pd.DataFrame,
                           x_col: str = 'x',
                           y_col: str = 'y',
                           hover_col: str = 'mp_id',
                           color_col: Optional[str] = None,
                           title: Optional[str] = None,
                           width: int = 800,
                           height: int = 600,
                           marker_size: int = 10,
                           font_size: int = 16,
                           axis_font_size: int = 18) -> px.scatter:
    """
    Create an interactive Plotly scatter plot of embeddings.

    Args:
        df_plot: DataFrame with embedding coordinates
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        hover_col: Column to show on hover
        color_col: Optional column for coloring points
        title: Plot title
        width: Plot width
        height: Plot height
        marker_size: Size of markers
        font_size: Base font size
        axis_font_size: Axis label font size

    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name=hover_col,
        labels={x_col: f'{x_col.upper()}', y_col: f'{y_col.upper()}'},
        width=width,
        height=height
    )

    # Update marker styling
    fig.update_traces(marker=dict(size=marker_size, line=dict(width=1, color='black')))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title_font=dict(size=axis_font_size),
        yaxis_title_font=dict(size=axis_font_size),
        font=dict(size=font_size)
    )

    return fig

def create_embedding_matplotlib(df_plot: pd.DataFrame,
                               x_col: str = 'x',
                               y_col: str = 'y',
                               figsize: tuple = (8, 6),
                               dpi: int = 300,
                               marker_size: int = 30,
                               alpha: float = 0.8,
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a static matplotlib scatter plot of embeddings.

    Args:
        df_plot: DataFrame with embedding coordinates
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        figsize: Figure size tuple
        dpi: Resolution for saved image
        marker_size: Size of markers
        alpha: Transparency
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.scatter(
        df_plot[x_col],
        df_plot[y_col],
        s=marker_size,
        edgecolors='black',
        facecolors='white',
        linewidth=0.5,
        alpha=alpha
    )

    ax.set_xlabel(f"{x_col.upper()}", fontsize=18)
    ax.set_ylabel(f"{y_col.upper()}", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")

    return fig

def save_plotly_figure(fig: px.scatter, filename: str, scale: int = 3, **kwargs):
    """
    Save a Plotly figure as a high-resolution image.

    Args:
        fig: Plotly figure object
        filename: Output filename
        scale: Image scale factor
        **kwargs: Additional arguments for write_image
    """
    try:
        import plotly.io as pio
        fig.write_image(filename, scale=scale, **kwargs)
        logger.info(f"Saved Plotly figure to {filename}")
    except ImportError:
        logger.error("plotly.io not available for image export")
    except Exception as e:
        logger.error(f"Error saving Plotly figure: {e}")

def create_plotting_dataframe(material_ids: list,
                             embeddings_2d: np.ndarray,
                             cluster_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Create a DataFrame suitable for plotting embeddings.

    Args:
        material_ids: List of material IDs
        embeddings_2d: 2D embeddings array
        cluster_labels: Optional cluster labels

    Returns:
        DataFrame for plotting
    """
    data = {
        'mp_id': material_ids,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1]
    }

    if cluster_labels is not None:
        data['cluster'] = cluster_labels

    return pd.DataFrame(data)
