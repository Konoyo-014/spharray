"""Library module.

Usage:
    from spherical_array_processing.sh.transforms import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import SphericalGrid


def direct_sht(
    samples: ArrayLike,
    y_matrix: ArrayLike,
    grid: SphericalGrid | None = None,
    weights: ArrayLike | None = None,
) -> NDArray[np.complex128]:
    """Direct weighted spherical harmonic transform via quadrature/LS.

    `samples` shape may be (..., n_points), `y_matrix` shape (n_points, n_coeffs).
    """
    x = np.asarray(samples)
    y = np.asarray(y_matrix)
    if y.ndim != 2:
        raise ValueError("y_matrix must be 2D")
    n_points = y.shape[0]
    if x.shape[-1] != n_points:
        raise ValueError("samples last axis must equal number of grid points")
    if weights is None and grid is not None:
        weights = grid.weights
    if weights is None:
        w = np.ones(n_points, dtype=float) / n_points
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n_points:
            raise ValueError("weights length mismatch")
    yw = y.conj() * w[:, None]
    return np.tensordot(x, yw, axes=([-1], [0]))

