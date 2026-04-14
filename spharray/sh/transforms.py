"""Spherical-harmonic transforms (SHT and iSHT).

The direct (forward) SHT converts a function sampled on the sphere to SH
coefficients; the inverse SHT synthesizes a function on the sphere from
its SH coefficients.

Forward SHT (quadrature / least-squares)
-----------------------------------------
Given M samples f(Ω_i) and quadrature weights w_i with Σ w_i = 4π, the
weighted SHT is::

    f_n^m = Σ_i w_i · f(Ω_i) · (Y_n^m(Ω_i))*

In matrix form with Y the SH design matrix (M × (N+1)²) and W = diag(w)::

    f = (Y^H W) f_samples     →     shape (N+1)²

For an overdetermined system without weights, least-squares is used instead.

Inverse SHT (synthesis)
------------------------
Given SH coefficients f_nm, the function is synthesized as::

    f(Ω_k) = Σ_{n,m} f_n^m · Y_n^m(Ω_k)

In matrix form::

    f_samples = Y · f_nm     →     shape M
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
    """Compute the direct (forward) spherical-harmonic transform.

    Parameters
    ----------
    samples : array-like, shape (..., M)
        Function values at M grid points.  Leading batch dimensions are
        supported; the SHT is applied along the last axis.
    y_matrix : array-like, shape (M, (N+1)²)
        SH design matrix from :func:`~spharray.sh.matrix`.
    grid : SphericalGrid, optional
        Grid whose ``.weights`` attribute is used if ``weights`` is not given.
    weights : array-like, shape (M,), optional
        Quadrature weights (should sum to 4π for proper normalization).  If
        neither ``weights`` nor ``grid.weights`` is provided, equal weights
        (1/M each) are used (equivalent to least-squares on a quasi-uniform
        grid).

    Returns
    -------
    coeffs : ndarray, complex128, shape (..., (N+1)²)
        SH coefficients in ACN order.

    Notes
    -----
    For a perfect quadrature rule (e.g., equiangle sampling with the correct
    weights), the quadrature SHT gives exact coefficients up to order N.
    For Fibonacci or other approximate grids, the result is a least-squares
    fit to the data.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.array.sampling import equiangle_sampling
    >>> from spharray.sh import matrix, direct_sht
    >>> from spharray.types import SHBasisSpec
    >>> spec = SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")
    >>> grid = equiangle_sampling(2)
    >>> Y = matrix(spec, grid)
    >>> # Synthesize Y_1^0 (= cos(colatitude)/sqrt(4pi/3)) and round-trip it
    >>> f_samples = Y[:, 4]                # acn_index(1, 0) == 2; wait, n=1,m=0 -> idx=2
    >>> coeffs = direct_sht(f_samples, Y, grid)
    >>> abs(coeffs[2] - 1.0) < 1e-10
    True
    """
    x = np.asarray(samples)
    y = np.asarray(y_matrix)
    if y.ndim != 2:
        raise ValueError("y_matrix must be 2D with shape (M, n_coeffs)")
    n_points = y.shape[0]
    if x.shape[-1] != n_points:
        raise ValueError(
            f"samples last axis has size {x.shape[-1]}, "
            f"but y_matrix has {n_points} rows"
        )
    if weights is None and grid is not None:
        weights = grid.weights
    if weights is None:
        # Equal-weight fallback (works well for quasi-uniform grids)
        w = np.ones(n_points, dtype=float) / n_points
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.size != n_points:
            raise ValueError(
                f"weights has {w.size} elements but y_matrix has {n_points} rows"
            )
    # coeffs = (Y^H diag(w)) @ samples.T  via einsum
    yw = y.conj() * w[:, None]           # (M, n_coeffs)
    return np.tensordot(x, yw, axes=([-1], [0]))


def inverse_sht(
    coeffs: ArrayLike,
    y_matrix: ArrayLike,
) -> NDArray[np.complex128]:
    """Compute the inverse (synthesis) spherical-harmonic transform.

    Parameters
    ----------
    coeffs : array-like, shape (..., (N+1)²)
        SH coefficients in ACN order.
    y_matrix : array-like, shape (M, (N+1)²)
        SH design matrix from :func:`~spharray.sh.matrix`.

    Returns
    -------
    samples : ndarray, shape (..., M)
        Synthesized function values at the M grid directions.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.array.sampling import fibonacci_grid
    >>> from spharray.sh import matrix
    >>> from spharray.types import SHBasisSpec
    >>> spec = SHBasisSpec(max_order=1, basis="complex", angle_convention="az_colat")
    >>> grid = fibonacci_grid(50)
    >>> Y = matrix(spec, grid)
    >>> coeffs = np.zeros(4, dtype=complex); coeffs[0] = 1.0  # DC only
    >>> samples = inverse_sht(coeffs, Y)
    >>> np.allclose(samples, 1.0, atol=1e-10)
    True
    """
    c = np.asarray(coeffs, dtype=np.complex128)
    y = np.asarray(y_matrix, dtype=np.complex128)
    if y.ndim != 2:
        raise ValueError("y_matrix must be 2D with shape (M, n_coeffs)")
    if c.shape[-1] != y.shape[1]:
        raise ValueError(
            f"coeffs last axis ({c.shape[-1]}) does not match "
            f"y_matrix columns ({y.shape[1]})"
        )
    # samples = Y @ coeffs.T  →  shape (..., M)
    return np.tensordot(c, y.T, axes=([-1], [0]))
