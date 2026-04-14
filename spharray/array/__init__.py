"""Spatial sampling grids, array simulation, and spatial aliasing utilities.

Examples
--------
>>> from spharray.array import fibonacci_grid, equiangle_sampling
>>> from spharray.array import simulate_plane_wave_array_response
>>> from spharray.array import spatial_aliasing_frequency, max_sh_order
"""

from .sampling import (
    equiangle_sampling,
    fibonacci_grid,
    get_tdesign_fallback,
    max_sh_order,
    spatial_aliasing_frequency,
)
from .simulation import simulate_plane_wave_array_response

__all__ = [
    "equiangle_sampling",
    "fibonacci_grid",
    "get_tdesign_fallback",
    "max_sh_order",
    "simulate_plane_wave_array_response",
    "spatial_aliasing_frequency",
]
