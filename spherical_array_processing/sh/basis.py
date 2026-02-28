"""Library module.

Usage:
    from spherical_array_processing.sh.basis import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:  # SciPy API compatibility across versions
    from scipy.special import sph_harm as _scipy_sph_harm
except Exception:  # pragma: no cover
    from scipy.special import sph_harm_y as _scipy_sph_harm  # type: ignore

from ..coords import azel_to_az_colat
from ..types import SHBasisSpec, SphericalGrid


def acn_index(n: int, m: int) -> int:
    """Usage:
        Run acn index.
    
    Args:
        n: int.
        m: int.
    
    Returns:
        int.
    """
    return n * (n + 1) + m


def _norm_scale(n: int, normalization: str) -> float:
    """Usage:
        Run norm scale.
    
    Args:
        n: int.
        normalization: str.
    
    Returns:
        float.
    """
    if normalization == "orthonormal":
        return 1.0
    if normalization == "sn3d":
        return np.sqrt(4.0 * np.pi / (2 * n + 1))
    if normalization == "n3d":
        return np.sqrt(4.0 * np.pi)
    raise ValueError(f"unsupported normalization: {normalization}")


def _grid_to_az_colat(grid: SphericalGrid) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Usage:
        Run grid to az colat.
    
    Args:
        grid: SphericalGrid.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]].
    """
    if grid.convention == "az_colat":
        return grid.azimuth, grid.angle2
    return azel_to_az_colat(grid.azimuth, grid.angle2)


def complex_matrix(
    spec: SHBasisSpec,
    grid: SphericalGrid,
) -> NDArray[np.complex128]:
    """Usage:
        Run complex matrix.
    
    Args:
        spec: SHBasisSpec.
        grid: SphericalGrid.
    
    Returns:
        NDArray[np.complex128].
    """
    if spec.angle_convention not in {"az_el", "az_colat"}:
        raise ValueError(f"unsupported angle convention: {spec.angle_convention}")
    az, colat = _grid_to_az_colat(grid)
    y = np.zeros((grid.size, spec.n_coeffs), dtype=np.complex128)
    for n in range(spec.max_order + 1):
        s = _norm_scale(n, spec.normalization)
        for m in range(-n, n + 1):
            idx = acn_index(n, m)
            y[:, idx] = _scipy_sph_harm(m, n, az, colat) * s
    return y


def real_matrix(spec: SHBasisSpec, grid: SphericalGrid) -> NDArray[np.float64]:
    """Usage:
        Run real matrix.
    
    Args:
        spec: SHBasisSpec.
        grid: SphericalGrid.
    
    Returns:
        NDArray[np.float64].
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
            else:
                yr[:, idx] = np.sqrt(2.0) * ((-1) ** m) * yc[:, idx].real
    return yr


def matrix(spec: SHBasisSpec, grid: SphericalGrid) -> NDArray[np.complex128] | NDArray[np.float64]:
    """Usage:
        Run matrix.
    
    Args:
        spec: SHBasisSpec.
        grid: SphericalGrid.
    
    Returns:
        NDArray[np.complex128] | NDArray[np.float64].
    """
    if spec.basis == "complex":
        return complex_matrix(spec, grid)
    if spec.basis == "real":
        return real_matrix(spec, grid)
    raise ValueError(f"unsupported basis: {spec.basis}")


def replicate_per_order(values: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run replicate per order.
    
    Args:
        values: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    out = []
    for n, v in enumerate(vals):
        out.extend([v] * (2 * n + 1))
    return np.asarray(out, dtype=float)


def complex_to_real_coeffs(coeffs: ArrayLike, max_order: int, axis: int = -1) -> NDArray[np.float64]:
    """Usage:
        Run complex to real coeffs.
    
    Args:
        coeffs: ArrayLike.
        max_order: int.
        axis: int, default=-1.
    
    Returns:
        NDArray[np.float64].
    """
    c = np.asarray(coeffs, dtype=np.complex128)
    c = np.moveaxis(c, axis, -1)
    if c.shape[-1] != (max_order + 1) ** 2:
        raise ValueError("last axis does not match max_order")
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


def real_to_complex_coeffs(coeffs: ArrayLike, max_order: int, axis: int = -1) -> NDArray[np.complex128]:
    """Inverse of `complex_to_real_coeffs` under the real-field symmetry assumption."""
    r = np.asarray(coeffs, dtype=float)
    r = np.moveaxis(r, axis, -1)
    if r.shape[-1] != (max_order + 1) ** 2:
        raise ValueError("last axis does not match max_order")
    c = np.zeros(r.shape, dtype=np.complex128)
    for n in range(max_order + 1):
        c[..., acn_index(n, 0)] = r[..., acn_index(n, 0)]
        for m in range(1, n + 1):
            rp = r[..., acn_index(n, m)]
            rn = r[..., acn_index(n, -m)]
            cpos = ((-1) ** m) / np.sqrt(2.0) * (rp + 1j * rn)
            cneg = (1.0 / np.sqrt(2.0)) * (rp - 1j * rn)
            c[..., acn_index(n, m)] = cpos
            c[..., acn_index(n, -m)] = cneg
    return np.moveaxis(c, -1, axis)
