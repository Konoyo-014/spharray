"""Spherical-harmonic basis matrices and coefficient conversions.

This module generates the spherical-harmonic (SH) design matrix Y of shape
(M, (N+1)²), where M is the number of grid points and (N+1)² is the number
of SH coefficients for an N-th order expansion.  Both complex and real (tesseral)
SH are supported, with three normalization conventions.

Mathematical background
-----------------------
Complex SH  Y_n^m(θ, φ)
    Defined for degree n ≥ 0 and order m = −n, …, n.  With the
    Condon–Shortley phase convention (used by SciPy)::

        Y_n^m(θ, φ) = √((2n+1)/(4π) · (n−|m|)!/(n+|m|)!) ·
                      P_n^|m|(cos θ) · exp(imφ)   for m ≥ 0,
        Y_n^{-m}   = (−1)^m · (Y_n^m)*            for m < 0.

Real (tesseral) SH  Y_n^m_r(θ, φ)
    ::
        Y_n^m_r = √2 · (−1)^m · Im(Y_n^{|m|})   for m < 0,
        Y_n^0_r = Y_n^0                            for m = 0,
        Y_n^m_r = √2 · (−1)^m · Re(Y_n^m)        for m > 0.

Channel ordering: ACN
    ::
        index(n, m) = n(n+1) + m.

References
----------
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Chapter 2.
* Zotter, F. (2009). *Analysis and Synthesis of Sound-Radiation with
  Spherical Arrays*.  Doctoral thesis, IEM Graz.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # SciPy >= 1.15 new API
    from scipy.special import sph_harm_y as _modern_sph_harm_y
except ImportError:
    _modern_sph_harm_y = None

try:  # SciPy <= 1.14 legacy API
    from scipy.special import sph_harm as _legacy_sph_harm
except ImportError:
    _legacy_sph_harm = None

from ..coords import azel_to_az_colat
from ..types import SHBasisSpec, SphericalGrid


# ---------------------------------------------------------------------------
# SciPy API shim
# ---------------------------------------------------------------------------

def _eval_sph_harm(
    m: int,
    n: int,
    azimuth: NDArray[np.float64],
    colatitude: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """Evaluate complex Y_n^m at the given (azimuth, colatitude) directions.

    Handles both the legacy ``sph_harm(m, n, phi, theta)`` API (SciPy ≤ 1.14)
    and the modern ``sph_harm_y(n, m, theta, phi)`` API (SciPy ≥ 1.15).
    Both use the Condon–Shortley phase convention.
    """
    if _modern_sph_harm_y is not None:
        return np.asarray(
            _modern_sph_harm_y(n, m, colatitude, azimuth), dtype=np.complex128
        )
    if _legacy_sph_harm is not None:
        return np.asarray(
            _legacy_sph_harm(m, n, azimuth, colatitude), dtype=np.complex128
        )
    raise ImportError(
        "Neither scipy.special.sph_harm nor scipy.special.sph_harm_y "
        "is available.  Install SciPy >= 1.11."
    )


# ---------------------------------------------------------------------------
# ACN index
# ---------------------------------------------------------------------------

def acn_index(n: int, m: int) -> int:
    """Return the ACN (Ambisonic Channel Number) index for degree n, order m.

    Parameters
    ----------
    n : int
        SH degree (≥ 0).
    m : int
        SH order (|m| ≤ n).

    Returns
    -------
    int
        ACN index = n(n+1) + m.

    Examples
    --------
    >>> acn_index(0, 0)
    0
    >>> acn_index(1, -1)
    1
    >>> acn_index(2, 0)
    6
    """
    return n * (n + 1) + m


# ---------------------------------------------------------------------------
# Normalization scale
# ---------------------------------------------------------------------------

def _norm_scale(n: int, normalization: str) -> float:
    """Return the scale factor to convert from orthonormal to the target norm.

    Parameters
    ----------
    n : int
        SH degree.
    normalization : {"orthonormal", "sn3d", "n3d"}

    Returns
    -------
    float
        Scale factor s such that Y_target = s · Y_orthonormal.
    """
    if normalization == "orthonormal":
        return 1.0
    if normalization == "sn3d":
        # ∫|Y_sn3d|² dΩ = 4π/(2n+1)  →  scale = √(4π/(2n+1))
        return float(np.sqrt(4.0 * np.pi / (2 * n + 1)))
    if normalization == "n3d":
        # ∫|Y_n3d|² dΩ = 4π  →  scale = √(4π)
        return float(np.sqrt(4.0 * np.pi))
    raise ValueError(f"unsupported normalization: {normalization!r}")


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _grid_to_az_colat(
    grid: SphericalGrid,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (azimuth, colatitude) arrays in radians from any grid convention."""
    if grid.convention == "az_colat":
        return grid.azimuth, grid.angle2
    return azel_to_az_colat(grid.azimuth, grid.angle2)


# ---------------------------------------------------------------------------
# Complex SH matrix
# ---------------------------------------------------------------------------

def complex_matrix(
    spec: SHBasisSpec,
    grid: SphericalGrid,
) -> NDArray[np.complex128]:
    """Compute the complex SH design matrix Y of shape (M, (N+1)²).

    Each row contains the complex SH values at one grid direction.  Entry
    Y[i, acn_index(n, m)] = Y_n^m(θ_i, φ_i) · scale(n, normalization).

    Parameters
    ----------
    spec : SHBasisSpec
        Basis specification (order, normalization, angle convention).
    grid : SphericalGrid
        M directions on the unit sphere.

    Returns
    -------
    Y : ndarray, complex128, shape (M, (N+1)²)
        Complex SH matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.array.sampling import equiangle_sampling
    >>> spec = SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")
    >>> grid = equiangle_sampling(2)
    >>> Y = complex_matrix(spec, grid)
    >>> Y.shape[1]
    9
    """
    if spec.angle_convention not in {"az_el", "az_colat"}:
        raise ValueError(f"unsupported angle convention: {spec.angle_convention!r}")
    az, colat = _grid_to_az_colat(grid)
    y = np.zeros((grid.size, spec.n_coeffs), dtype=np.complex128)
    for n in range(spec.max_order + 1):
        s = _norm_scale(n, spec.normalization)
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            y[:, idx] = _eval_sph_harm(m, n, az, colat) * s
    return y


# ---------------------------------------------------------------------------
# Real SH matrix
# ---------------------------------------------------------------------------

def real_matrix(spec: SHBasisSpec, grid: SphericalGrid) -> NDArray[np.float64]:
    """Compute the real (tesseral) SH design matrix Y of shape (M, (N+1)²).

    Real SH are obtained from the complex SH via the unitary transform::

        Y_n^{m<0}_r = √2 · (−1)^m · Im(Y_n^{|m|}_c)
        Y_n^0_r     = Y_n^0_c   (already real)
        Y_n^{m>0}_r = √2 · (−1)^m · Re(Y_n^m_c)

    This transform preserves orthonormality: ∫ Y_n^m_r · Y_{n'}^{m'}_r dΩ = δ_{nn'} δ_{mm'}.

    Parameters
    ----------
    spec : SHBasisSpec
        Basis specification.  ``spec.basis`` is ignored (always "real").
    grid : SphericalGrid
        M directions on the unit sphere.

    Returns
    -------
    Y : ndarray, float64, shape (M, (N+1)²)
        Real SH matrix.
    """
    yc = complex_matrix(
        SHBasisSpec(
            max_order=spec.max_order,
            basis="complex",
            normalization=spec.normalization,
            angle_convention=spec.angle_convention,
        ),
        grid,
    )
    yr = np.zeros_like(yc.real)
    for n in range(spec.max_order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            if m < 0:
                yr[:, idx] = np.sqrt(2.0) * ((-1) ** m) * yc[:, acn_index(n, -m)].imag
            elif m == 0:
                yr[:, idx] = yc[:, idx].real
            else:  # m > 0
                yr[:, idx] = np.sqrt(2.0) * ((-1) ** m) * yc[:, idx].real
    return yr


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def matrix(
    spec: SHBasisSpec,
    grid: SphericalGrid,
) -> NDArray[np.complex128] | NDArray[np.float64]:
    """Compute the SH design matrix for the given basis specification.

    Dispatches to :func:`complex_matrix` or :func:`real_matrix` based on
    ``spec.basis``.

    Parameters
    ----------
    spec : SHBasisSpec
        Basis specification.
    grid : SphericalGrid
        M directions on the unit sphere.

    Returns
    -------
    Y : ndarray, shape (M, (N+1)²)
        Complex (``dtype=complex128``) or real (``dtype=float64``) SH matrix.
    """
    if spec.basis == "complex":
        return complex_matrix(spec, grid)
    if spec.basis == "real":
        return real_matrix(spec, grid)
    raise ValueError(f"unsupported basis: {spec.basis!r}")


# ---------------------------------------------------------------------------
# Per-order replication
# ---------------------------------------------------------------------------

def replicate_per_order(values: ArrayLike) -> NDArray[np.float64]:
    """Expand per-order values to a full ACN coefficient vector.

    Given a vector ``[v_0, v_1, …, v_N]`` of N+1 values (one per SH degree),
    this function returns a vector of (N+1)² values where degree n is
    repeated (2n+1) times.  This is useful for applying per-order filters
    or weighting to SH coefficient vectors.

    Parameters
    ----------
    values : array-like, shape (N+1,)
        One value per SH degree 0, 1, …, N.

    Returns
    -------
    ndarray, shape ((N+1)²,)
        Expanded vector in ACN order.

    Examples
    --------
    >>> replicate_per_order([1.0, 2.0])
    array([1., 2., 2., 2.])
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    out: list[float] = []
    for n, v in enumerate(vals):
        out.extend([v] * (2 * n + 1))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Coefficient-space conversions
# ---------------------------------------------------------------------------

def complex_to_real_coeffs(
    coeffs: ArrayLike,
    max_order: int,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Convert complex SH coefficients to real (tesseral) SH coefficients.

    The conversion is the unitary transform::

        r_{n,-|m|} = √2 · (−1)^m · Im(c_{n,|m|})   (m < 0, using |m| index)
        r_{n,0}   = Re(c_{n,0})
        r_{n,m}   = √2 · (−1)^m · Re(c_{n,m})       (m > 0)

    Parameters
    ----------
    coeffs : array-like
        Complex SH coefficients.  The axis ``axis`` must have length (N+1)².
    max_order : int
        Maximum SH order N.
    axis : int, default -1
        Axis along which the coefficients are stored.

    Returns
    -------
    ndarray, float64
        Real SH coefficients, same shape as ``coeffs``.
    """
    c = np.asarray(coeffs, dtype=np.complex128)
    c = np.moveaxis(c, axis, -1)
    if c.shape[-1] != (max_order + 1) ** 2:
        raise ValueError(
            f"last axis size {c.shape[-1]} does not match max_order={max_order} "
            f"(expected {(max_order + 1) ** 2})"
        )
    r = np.zeros(c.shape, dtype=float)
    for n in range(max_order + 1):
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            if m < 0:
                r[..., idx] = np.sqrt(2.0) * ((-1) ** m) * c[..., acn_index(n, -m)].imag
            elif m == 0:
                r[..., idx] = c[..., idx].real
            else:
                r[..., idx] = np.sqrt(2.0) * ((-1) ** m) * c[..., idx].real
    return np.moveaxis(r, -1, axis)


def real_to_complex_coeffs(
    coeffs: ArrayLike,
    max_order: int,
    axis: int = -1,
) -> NDArray[np.complex128]:
    """Inverse of :func:`complex_to_real_coeffs`.

    Convert real (tesseral) SH coefficients back to complex SH coefficients,
    assuming the real SH were obtained from complex SH with the standard
    Condon–Shortley phase and the transform in :func:`complex_to_real_coeffs`.

    Parameters
    ----------
    coeffs : array-like
        Real SH coefficients.  The axis ``axis`` must have length (N+1)².
    max_order : int
        Maximum SH order N.
    axis : int, default -1
        Axis along which the coefficients are stored.

    Returns
    -------
    ndarray, complex128
        Complex SH coefficients, same shape as ``coeffs``.
    """
    r = np.asarray(coeffs, dtype=float)
    r = np.moveaxis(r, axis, -1)
    if r.shape[-1] != (max_order + 1) ** 2:
        raise ValueError(
            f"last axis size {r.shape[-1]} does not match max_order={max_order} "
            f"(expected {(max_order + 1) ** 2})"
        )
    c = np.zeros(r.shape, dtype=np.complex128)
    for n in range(max_order + 1):
        c[..., acn_index(n, 0)] = r[..., acn_index(n, 0)]
        for m in range(1, n + 1):
            rp = r[..., acn_index(n, m)]    # real coeff at positive m
            rn = r[..., acn_index(n, -m)]   # real coeff at negative m
            # Inverse: c_m = (-1)^m (rp + i rn) / √2,  c_{-m} = (rp - i rn) / √2
            c[..., acn_index(n, m)]  = ((-1) ** m) / np.sqrt(2.0) * (rp + 1j * rn)
            c[..., acn_index(n, -m)] = (1.0 / np.sqrt(2.0)) * (rp - 1j * rn)
    return np.moveaxis(c, -1, axis)
