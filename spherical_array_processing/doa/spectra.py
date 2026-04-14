"""Spatial spectrum estimators for direction-of-arrival (DOA) estimation.

All estimators produce a :class:`~spherical_array_processing.types.SpatialSpectrumResult`
containing the full spatial spectrum over a search grid and the estimated
peak directions.

PWD (Plane-Wave Decomposition)
-------------------------------
The beamformer-based PWD spatial spectrum is::

    P_PWD(Ω) = y(Ω)^H R y(Ω)

where y(Ω) ∈ ℂ^Q is the SH steering vector for direction Ω and R ∈ ℂ^{Q×Q}
is the SH-domain covariance matrix.  Peaks in P_PWD correspond to dominant
source directions.

MUSIC (Multiple Signal Classification)
---------------------------------------
MUSIC projects the steering vector onto the noise subspace spanned by the
Q − K smallest eigenvectors of R (where K = n_sources)::

    P_MUSIC(Ω) = 1 / (y(Ω)^H E_N E_N^H y(Ω))

Peaks of P_MUSIC are sharper than PWD, especially for closely spaced sources.

References
----------
* Schmidt, R. O. (1986). "Multiple emitter location and signal parameter
  estimation". *IEEE Trans. Antennas Propag.*, 34(3), 276–280.
* Rafaely, B. (2005). "Analysis and design of spherical microphone arrays".
  *IEEE Trans. Speech Audio Process.*, 13(1), 135–143.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from ..sh import matrix as sh_matrix
from ..types import SHBasisSpec, SpatialSpectrumResult, SphericalGrid


def peak_pick_spectrum(spectrum: ArrayLike, n_peaks: int) -> np.ndarray:
    """Find the indices of the top-N peaks in a spatial spectrum.

    Parameters
    ----------
    spectrum : array-like, shape (G,)
        Spatial spectrum values at G grid points.
    n_peaks : int
        Number of peaks to return.

    Returns
    -------
    indices : ndarray, int64, shape (n_peaks,)
        Grid indices sorted by decreasing spectrum value.

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array([0.1, 0.9, 0.3, 0.7])
    >>> peak_pick_spectrum(s, 2)
    array([1, 3])
    """
    s = np.asarray(spectrum, dtype=float).reshape(-1)
    n = max(1, min(n_peaks, s.size))
    idx = np.argpartition(s, -n)[-n:]
    idx = idx[np.argsort(s[idx])[::-1]]
    return idx.astype(np.int64)


def spatial_spectrum_from_map(
    spectrum: ArrayLike,
    grid: SphericalGrid,
    n_peaks: int,
    metadata: dict | None = None,
) -> SpatialSpectrumResult:
    """Wrap a raw spectrum array in a :class:`SpatialSpectrumResult`.

    Parameters
    ----------
    spectrum : array-like, shape (G,)
        Spatial spectrum values at G grid points.
    grid : SphericalGrid
        The search grid; must have ``size == G``.
    n_peaks : int
        Number of peaks to detect and return.
    metadata : dict, optional
        Extra metadata to attach (e.g., ``{"method": "music"}``).

    Returns
    -------
    SpatialSpectrumResult
        Populated result with peak indices and directions in ``az_el`` convention.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.array.sampling import fibonacci_grid
    >>> grid = fibonacci_grid(100)
    >>> spec = np.zeros(100); spec[42] = 1.0
    >>> result = spatial_spectrum_from_map(spec, grid, n_peaks=1)
    >>> result.peak_indices[0]
    42
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


def pwd_spectrum(
    cov: ArrayLike,
    grid: SphericalGrid,
    basis: SHBasisSpec,
    n_peaks: int = 1,
) -> SpatialSpectrumResult:
    """Compute the PWD (plane-wave decomposition) spatial spectrum.

    P_PWD(Ω_k) = y(Ω_k)^H R y(Ω_k)

    Parameters
    ----------
    cov : array-like, shape (Q, Q)
        Hermitian SH-domain covariance matrix.
    grid : SphericalGrid
        Search grid with G directions.
    basis : SHBasisSpec
        SH basis matching the covariance matrix.
    n_peaks : int, default 1
        Number of source peaks to detect.

    Returns
    -------
    SpatialSpectrumResult
        Spectrum and detected peak directions.

    Notes
    -----
    The PWD spectrum is equivalent to applying a matched-filter (data-independent)
    beamformer to the estimated covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.array.sampling import fibonacci_grid
    >>> from spherical_array_processing.types import SHBasisSpec
    >>> spec  = SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
    >>> grid  = fibonacci_grid(200)
    >>> Q     = spec.n_coeffs
    >>> R     = np.eye(Q, dtype=complex)  # diffuse field → flat spectrum
    >>> result = pwd_spectrum(R, grid, spec)
    >>> result.spectrum.shape == (200,)
    True
    """
    r = np.asarray(cov, dtype=np.complex128)
    y = np.asarray(sh_matrix(basis, grid), dtype=np.complex128)
    if y.ndim != 2:
        raise ValueError("SH matrix must be 2-D")
    if y.shape[1] != r.shape[0]:
        raise ValueError(
            f"SH matrix has {y.shape[1]} columns but covariance has "
            f"{r.shape[0]} rows; check that basis matches cov"
        )
    p = np.real(np.einsum("gi,ij,gj->g", np.conj(y), r, y))
    return spatial_spectrum_from_map(p, grid, n_peaks=n_peaks, metadata={"method": "pwd"})


def music_spectrum(
    cov: ArrayLike,
    grid: SphericalGrid,
    basis: SHBasisSpec,
    n_sources: int,
    n_peaks: int | None = None,
) -> SpatialSpectrumResult:
    """Compute the MUSIC spatial spectrum.

    P_MUSIC(Ω_k) = 1 / (y(Ω_k)^H E_N E_N^H y(Ω_k))

    where E_N contains the Q − n_sources smallest eigenvectors of R (noise
    subspace).

    Parameters
    ----------
    cov : array-like, shape (Q, Q)
        Hermitian SH-domain covariance matrix.
    grid : SphericalGrid
        Search grid with G directions.
    basis : SHBasisSpec
        SH basis matching the covariance matrix.
    n_sources : int
        Assumed number of coherent sources (≥ 1 and < Q).
    n_peaks : int, optional
        Number of peaks to detect.  Defaults to ``n_sources``.

    Returns
    -------
    SpatialSpectrumResult
        Spectrum and detected peak directions.

    Notes
    -----
    MUSIC requires that the signal subspace and noise subspace are well
    separated — i.e., that the covariance has been estimated from enough
    snapshots and that the sources are incoherent.  For coherent sources,
    spatial smoothing should be applied before calling this function.

    Examples
    --------
    >>> import numpy as np
    >>> from spherical_array_processing.array.sampling import fibonacci_grid
    >>> from spherical_array_processing.types import SHBasisSpec
    >>> spec = SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")
    >>> grid = fibonacci_grid(200)
    >>> Q    = spec.n_coeffs
    >>> # Build rank-1 covariance (single source)
    >>> y0   = np.zeros(Q, dtype=complex); y0[0] = 1.0
    >>> R    = np.outer(y0, y0.conj()) + 0.01 * np.eye(Q)
    >>> result = music_spectrum(R, grid, spec, n_sources=1)
    >>> result.spectrum.shape == (200,)
    True
    """
    r = np.asarray(cov, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square 2-D matrix")
    q = r.shape[0]
    n_sources = int(n_sources)
    if n_sources < 1 or n_sources >= q:
        raise ValueError(f"n_sources must be in [1, Q-1] where Q = {q}")

    # Eigen-decomposition (Hermitian → eigvalsh for numerical stability)
    evals, evecs = np.linalg.eigh(r)   # ascending order
    # Noise subspace: Q - n_sources smallest eigenvectors
    en = evecs[:, : q - n_sources]     # shape (Q, Q - n_sources)
    proj = en @ en.conj().T            # noise projection matrix (Q, Q)

    y = np.asarray(sh_matrix(basis, grid), dtype=np.complex128)
    if y.shape[1] != q:
        raise ValueError(
            f"SH matrix has {y.shape[1]} columns but covariance has {q} rows"
        )
    denom = np.real(np.einsum("gi,ij,gj->g", np.conj(y), proj, y))
    spec = 1.0 / np.maximum(denom, 1e-15)
    return spatial_spectrum_from_map(
        spec, grid,
        n_peaks=n_peaks if n_peaks is not None else n_sources,
        metadata={"method": "music"},
    )
