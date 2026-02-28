"""Library module.

Usage:
    from spherical_array_processing.array import <symbol>
"""

from .sampling import equiangle_sampling, fibonacci_grid, get_tdesign_fallback
from .simulation import simulate_plane_wave_array_response

__all__ = [
    "equiangle_sampling",
    "fibonacci_grid",
    "get_tdesign_fallback",
    "simulate_plane_wave_array_response",
]

