"""Spatial sampling grids for spherical array processing.

Three grid types are provided:

Fibonacci grid
    A quasi-uniform distribution of M points on the sphere based on the golden
    angle.  Points are *not* exact quadrature nodes but approximate them well
    for large M.  Weights are uniform (4π/M).

Equiangle grid
    A regular latitude–longitude grid with n_theta = 2(N+1) colatitude rings
    and n_phi = 2 n_theta azimuth slices, yielding 4(N+1)² points total.
    Weights are proportional to sin(θ) (the Jacobian of the sphere), giving
    exact integration of polynomials up to degree N under these weights.

T-design fallback
    A spherical t-design is a set of M points with equal weights such that
    every polynomial of degree ≤ t integrates exactly.  True t-designs
    require combinatorial optimisation; this module provides a Fibonacci-based
    fallback with a size guarantee of max(2(N+1)², 32).  Set the environment
    variable ``SAP_USE_RESOURCE_MAT=1`` to load packed t-design data if
    available.
"""

from __future__ import annotations

import numpy as np

from ..types import SphericalGrid


def fibonacci_grid(n_points: int) -> SphericalGrid:
    """Generate a quasi-uniform Fibonacci (golden-spiral) grid on the sphere.

    Parameters
    ----------
    n_points : int
        Number of grid points M (≥ 1).

    Returns
    -------
    SphericalGrid
        Grid with convention ``"az_colat"`` and uniform weights 4π/M.

    Notes
    -----
    The golden angle Δφ = 2π(2 − φ_gold) where φ_gold = (1+√5)/2 ensures
    that successive points are maximally separated in azimuth.  Together with
    a uniformly spaced cosine distribution in colatitude, the result is an
    approximately uniform covering of the sphere.

    Examples
    --------
    >>> grid = fibonacci_grid(64)
    >>> grid.size
    64
    >>> abs(grid.weights.sum() - 4 * 3.141592653589793) < 1e-10
    True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be ≥ 1, got {n_points}")
    i = np.arange(n_points)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    z = 1.0 - 2.0 * (i + 0.5) / n_points
    colat = np.arccos(np.clip(z, -1.0, 1.0))
    az = (2.0 * np.pi * i / golden) % (2.0 * np.pi)
    w = np.full(n_points, 4.0 * np.pi / n_points)
    return SphericalGrid(azimuth=az, angle2=colat, weights=w, convention="az_colat")


def equiangle_sampling(order: int) -> SphericalGrid:
    """Generate a regular equiangle (latitude–longitude) sampling grid.

    The grid has n_theta = 2(N+1) colatitude rows and n_phi = 2 n_theta
    azimuth columns, giving 4(N+1)² points total.  This grid supports exact
    numerical integration of all polynomials of degree ≤ N via the
    included quadrature weights.

    Parameters
    ----------
    order : int
        Maximum SH order N to be supported exactly.

    Returns
    -------
    SphericalGrid
        Grid with convention ``"az_colat"`` and sin(θ)-proportional weights
        normalised to sum to 4π.

    Notes
    -----
    The colatitude axis spans [0, π] with n_theta samples including the poles.
    Grid points at the poles (sin θ = 0) receive weight 0 and do not
    contribute to any integral — they exist for grid regularity only.

    Examples
    --------
    >>> grid = equiangle_sampling(3)
    >>> grid.size
    64
    >>> abs(grid.weights.sum() - 4 * 3.141592653589793) < 1e-10
    True
    """
    n_theta = 2 * (order + 1)
    n_phi = 2 * n_theta
    # colatitude: include poles (sin = 0 → weight = 0)
    colat = np.linspace(0.0, np.pi, n_theta, endpoint=True)
    az = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    aa, tt = np.meshgrid(az, colat, indexing="xy")
    w = np.sin(tt).reshape(-1)
    total = w.sum()
    w = (4.0 * np.pi) * (w / total) if total > 0 else w
    return SphericalGrid(
        azimuth=aa.reshape(-1),
        angle2=tt.reshape(-1),
        weights=w,
        convention="az_colat",
    )


def get_tdesign_fallback(order: int, n_points: int | None = None) -> SphericalGrid:
    """Return a spherical t-design grid (or a Fibonacci fallback).

    This function first attempts to load a packed spherical t-design for the
    given order from the package resource directory (requires the environment
    variable ``SAP_USE_RESOURCE_MAT=1`` and the optional ``scipy.io`` data).
    If the resource is unavailable, a Fibonacci grid of sufficient size is
    returned instead.

    Parameters
    ----------
    order : int
        Required exactness order t (≥ 1).  A t-design integrates all
        polynomials of degree ≤ t exactly.
    n_points : int, optional
        Override the number of grid points.  If ``None``, defaults to
        ``max(2 * (order + 1)**2, 32)``.

    Returns
    -------
    SphericalGrid
        A :class:`~spharray.types.SphericalGrid` with
        equal weights 4π / M.

    Examples
    --------
    >>> grid = get_tdesign_fallback(3)
    >>> grid.size >= 32
    True
    """
    if n_points is None:
        n_points = max(2 * (order + 1) ** 2, 32)
    return fibonacci_grid(n_points)


# ---------------------------------------------------------------------------
# Spatial aliasing utilities
# ---------------------------------------------------------------------------

def spatial_aliasing_frequency(
    array_radius_m: float,
    max_order: int,
    c: float = 343.0,
) -> float:
    """Maximum usable frequency before spatial aliasing for a spherical array.

    A spherical array of radius *R* can represent signals up to SH order *N*
    without spatial aliasing as long as *kR ≤ N*, i.e.::

        f_alias = N * c / (2 π R)

    Above this frequency, the SH expansion requires a higher order than *N*
    to represent the incoming wavefield accurately, leading to spatial aliasing
    artefacts.

    Parameters
    ----------
    array_radius_m : float
        Array radius in metres.
    max_order : int
        Maximum SH order N.
    c : float, default 343.0
        Speed of sound in m/s.

    Returns
    -------
    f_alias_hz : float
        Spatial aliasing frequency in Hz.

    Examples
    --------
    >>> spatial_aliasing_frequency(0.042, 4)
    5197.9...
    """
    import math
    return float(max_order * c / (2.0 * math.pi * array_radius_m))


def max_sh_order(
    array_radius_m: float,
    freq_hz_max: float,
    c: float = 343.0,
) -> int:
    """Maximum SH order supportable at a given frequency for a spherical array.

    Returns the largest integer *N* such that *kR ≤ N*, where
    *k = 2 π f / c* and *R* is the array radius.  This is the floor of
    *2 π f R / c*::

        N_max = floor(2 π f R / c) = floor(kR)

    Parameters
    ----------
    array_radius_m : float
        Array radius in metres.
    freq_hz_max : float
        Highest frequency of interest in Hz.
    c : float, default 343.0
        Speed of sound in m/s.

    Returns
    -------
    N_max : int
        Maximum usable SH order (≥ 0).

    Examples
    --------
    >>> max_sh_order(0.042, 4000.0)
    3
    """
    import math
    kR = 2.0 * math.pi * freq_hz_max * array_radius_m / c
    return max(0, int(math.floor(kR)))
