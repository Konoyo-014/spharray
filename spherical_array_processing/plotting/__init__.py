"""Library module.

Usage:
    from spherical_array_processing.plotting import <symbol>
"""

from .style import apply_matlab_like_style, figure_style_context
from .spatial_helpers import plot_directional_map_from_grid, plot_mic_array

__all__ = [
    "apply_matlab_like_style",
    "figure_style_context",
    "plot_directional_map_from_grid",
    "plot_mic_array",
]
