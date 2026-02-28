"""Library module.

Usage:
    from spherical_array_processing.array.sampling import <symbol>
"""

from __future__ import annotations

import numpy as np

from ..types import SphericalGrid


def fibonacci_grid(n_points: int) -> SphericalGrid:
    """Usage:
        Run fibonacci grid.
    
    Args:
        n_points: int.
    
    Returns:
        SphericalGrid.
    """
    if n_points < 1:
        raise ValueError("n_points must be positive")
    i = np.arange(n_points)
    golden = (1 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    colat = np.arccos(np.clip(z, -1.0, 1.0))
    az = (2 * np.pi * i / golden) % (2 * np.pi)
    w = np.full(n_points, 4 * np.pi / n_points)
    return SphericalGrid(azimuth=az, angle2=colat, weights=w, convention="az_colat")


def get_tdesign_fallback(order: int, n_points: int | None = None) -> SphericalGrid:
    """Usage:
        Run get tdesign fallback.
    
    Args:
        order: int.
        n_points: int | None, default=None.
    
    Returns:
        SphericalGrid.
    """
    if n_points is None:
        n_points = max(2 * (order + 1) ** 2, 32)
    return fibonacci_grid(n_points)


def equiangle_sampling(order: int) -> SphericalGrid:
    """Usage:
        Run equiangle sampling.
    
    Args:
        order: int.
    
    Returns:
        SphericalGrid.
    """
    n_theta = 2 * (order + 1)
    n_phi = 2 * n_theta
    colat = np.linspace(0.0, np.pi, n_theta, endpoint=True)
    az = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    aa, tt = np.meshgrid(az, colat, indexing="xy")
    w = np.sin(tt).reshape(-1)
    w = (4 * np.pi) * (w / w.sum())
    return SphericalGrid(azimuth=aa.reshape(-1), angle2=tt.reshape(-1), weights=w, convention="az_colat")

