"""Plotting and visualisation utilities.

Examples
--------
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> from spharray.plotting import (
...     apply_matlab_like_style, figure_style_context, plot_mic_array,
... )
"""

from .spatial_helpers import plot_directional_map_from_grid, plot_mic_array
from .style import apply_matlab_like_style, figure_style_context

__all__ = [
    "apply_matlab_like_style",
    "figure_style_context",
    "plot_directional_map_from_grid",
    "plot_mic_array",
]
