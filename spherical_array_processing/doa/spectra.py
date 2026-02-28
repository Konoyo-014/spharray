"""Library module.

Usage:
    from spherical_array_processing.doa.spectra import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SpatialSpectrumResult, SphericalGrid


def peak_pick_spectrum(spectrum: ArrayLike, n_peaks: int) -> np.ndarray:
    """Usage:
        Run peak pick spectrum.
    
    Args:
        spectrum: ArrayLike.
        n_peaks: int.
    
    Returns:
        np.ndarray.
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    n = max(1, min(n_peaks, s.size))
    idx = np.argpartition(s, -n)[-n:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return idx.astype(int)


def spatial_spectrum_from_map(spectrum: ArrayLike, grid: SphericalGrid, n_peaks: int, metadata: dict | None = None) -> SpatialSpectrumResult:
    """Usage:
        Run spatial spectrum from map.
    
    Args:
        spectrum: ArrayLike.
        grid: SphericalGrid.
        n_peaks: int.
        metadata: dict | None, default=None.
    
    Returns:
        SpatialSpectrumResult.
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    idx = peak_pick_spectrum(s, n_peaks)
    dirs = np.stack([grid.azimuth[idx], grid.elevation[idx]], axis=1)
    return SpatialSpectrumResult(
        spectrum=s,
        grid=grid,
        peak_indices=idx.astype(np.int64),
        peak_dirs_rad=dirs.astype(float),
        metadata={} if metadata is None else dict(metadata),
    )


def pwd_spectrum(cov: ArrayLike, grid: SphericalGrid, basis: SHBasisSpec, n_peaks: int = 1) -> SpatialSpectrumResult:
    """Usage:
        Run pwd spectrum.
    
    Args:
        cov: ArrayLike.
        grid: SphericalGrid.
        basis: SHBasisSpec.
        n_peaks: int, default=1.
    
    Returns:
        SpatialSpectrumResult.
    """
    r = np.asarray(cov, dtype=np.complex128)
    y = np.asarray(sh_matrix(basis, grid))
    if y.ndim != 2:
        raise ValueError("SH matrix must be 2D")
    if y.shape[1] != r.shape[0]:
        raise ValueError("basis/grid and covariance size mismatch")
    p = np.real(np.einsum("gi,ij,gj->g", np.conj(y), r, y))
    return spatial_spectrum_from_map(p, grid, n_peaks=n_peaks, metadata={"method": "pwd"})


def music_spectrum(cov: ArrayLike, grid: SphericalGrid, basis: SHBasisSpec, n_sources: int, n_peaks: int | None = None) -> SpatialSpectrumResult:
    """Usage:
        Run music spectrum.
    
    Args:
        cov: ArrayLike.
        grid: SphericalGrid.
        basis: SHBasisSpec.
        n_sources: int.
        n_peaks: int | None, default=None.
    
    Returns:
        SpatialSpectrumResult.
    """
    r = np.asarray(cov, dtype=np.complex128)
    evals, evecs = np.linalg.eigh(r)
    order = np.argsort(evals.real)
    evecs = evecs[:, order]
    n_sources = int(n_sources)
    if n_sources < 1 or n_sources >= r.shape[0]:
        raise ValueError("n_sources must be in [1, n_channels-1]")
    en = evecs[:, : r.shape[0] - n_sources]
    proj = en @ en.conj().T
    y = np.asarray(sh_matrix(basis, grid))
    denom = np.real(np.einsum("gi,ij,gj->g", np.conj(y), proj, y))
    spec = 1.0 / np.maximum(denom, 1e-15)
    return spatial_spectrum_from_map(spec, grid, n_peaks=n_peaks or n_sources, metadata={"method": "music"})

