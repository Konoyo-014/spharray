"""Core data structures for spherical array processing.

All physical angles are in **radians** throughout the library unless explicitly
noted (e.g., in plotting helpers that accept degrees).

Angle conventions
-----------------
``az_el``
    Azimuth φ ∈ [0, 2π) measured counter-clockwise from the +x axis in the
    horizontal plane; elevation θ_e ∈ [-π/2, π/2] measured upward from the
    horizontal plane.  Common in acoustics / audio.

``az_colat``
    Azimuth φ ∈ [0, 2π) as above; colatitude θ ∈ [0, π] measured downward
    from the +z pole.  Physics / mathematics convention required by most
    spherical-harmonic formulas.

Normalization conventions for spherical harmonics
--------------------------------------------------
``orthonormal``
    ∫_S² |Y_n^m(Ω)|² dΩ = 1.  Convention used by
    ``scipy.special.sph_harm`` / ``sph_harm_y`` and as the internal
    default in this library.

``sn3d``
    Schmidt semi-normalized (AmbiX / ATK standard).
    ∫_S² |Y_n^m|² dΩ = 4π / (2n+1).
    Relation to orthonormal: Y_sn3d = Y_orth · √(4π/(2n+1)).

``n3d``
    Full N3D normalization.
    ∫_S² |Y_n^m|² dΩ = 4π.
    Relation to orthonormal: Y_n3d = Y_orth · √(4π).

Channel ordering
----------------
Only ACN (Ambisonic Channel Numbering) is supported::

    index = n(n + 1) + m,   n = 0, 1, …, N,   m = -n, …, n.

This yields (N+1)² channels for an N-th order expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BasisKind = Literal["complex", "real"]
NormalizationKind = Literal["orthonormal", "n3d", "sn3d"]
AngleConvention = Literal["az_el", "az_colat"]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_1d_float(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class SHBasisSpec:
    """Specification for a spherical-harmonic basis.

    Parameters
    ----------
    max_order : int
        Maximum SH order N.  The basis contains (N+1)² coefficients.
    basis : {"complex", "real"}, default "complex"
        Whether to use complex or real (tesseral) spherical harmonics.
    normalization : {"orthonormal", "n3d", "sn3d"}, default "orthonormal"
        Normalization convention (see module docstring).
    angle_convention : {"az_el", "az_colat"}, default "az_el"
        Angle convention for the input grid (see module docstring).
    channel_order : {"acn"}, default "acn"
        Channel ordering.  Only ACN is currently supported.

    Examples
    --------
    >>> spec = SHBasisSpec(max_order=3, basis="real", normalization="sn3d")
    >>> spec.n_coeffs
    16
    """

    max_order: int
    basis: BasisKind = "complex"
    normalization: NormalizationKind = "orthonormal"
    angle_convention: AngleConvention = "az_el"
    channel_order: Literal["acn"] = "acn"

    @property
    def n_coeffs(self) -> int:
        """Total number of SH coefficients: (max_order + 1)²."""
        return (self.max_order + 1) ** 2


@dataclass
class SphericalGrid:
    """A set of directions on the unit sphere, optionally with quadrature weights.

    Parameters
    ----------
    azimuth : array-like, shape (M,)
        Azimuth angles in radians.
    angle2 : array-like, shape (M,)
        Second angle in radians.  Interpretation depends on ``convention``:
        elevation for ``"az_el"``, colatitude for ``"az_colat"``.
    weights : array-like, shape (M,), optional
        Quadrature weights.  Should satisfy ``sum(weights) ≈ 4π`` for
        proper spherical integration.  ``None`` means uniform (unweighted).
    convention : {"az_el", "az_colat"}, default "az_el"
        Angle convention (see module docstring).

    Examples
    --------
    >>> import numpy as np
    >>> grid = SphericalGrid(
    ...     azimuth=np.array([0.0, np.pi]),
    ...     angle2=np.array([0.0, 0.0]),
    ...     convention="az_el",
    ... )
    >>> grid.size
    2
    """

    azimuth: NDArray[np.float64]
    angle2: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    convention: AngleConvention = "az_el"
    _xyz_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.azimuth = _to_1d_float(self.azimuth)
        self.angle2 = _to_1d_float(self.angle2)
        if self.azimuth.shape != self.angle2.shape:
            raise ValueError("azimuth and angle2 must have the same shape")
        if self.weights is not None:
            self.weights = _to_1d_float(self.weights)
            if self.weights.shape != self.azimuth.shape:
                raise ValueError("weights shape must match grid size")

    @property
    def size(self) -> int:
        """Number of grid points M."""
        return int(self.azimuth.size)

    @property
    def elevation(self) -> NDArray[np.float64]:
        """Elevation in radians ∈ [-π/2, π/2], derived from the stored angle."""
        if self.convention == "az_el":
            return self.angle2
        return (np.pi / 2.0) - self.angle2

    @property
    def colatitude(self) -> NDArray[np.float64]:
        """Colatitude in radians ∈ [0, π], derived from the stored angle."""
        if self.convention == "az_colat":
            return self.angle2
        return (np.pi / 2.0) - self.angle2


@dataclass
class ArrayGeometry:
    """Physical geometry of a spherical microphone array.

    Parameters
    ----------
    radius_m : float
        Array radius in metres.
    sensor_grid : SphericalGrid
        Directions to the sensors on the sphere surface.
    array_type : {"open", "rigid", "cardioid"}, default "rigid"
        Sphere / baffle type, which determines the radial transfer functions.

        * ``"open"``     – sensors on an open sphere (no rigid baffle).
        * ``"rigid"``    – rigid sphere (Neumann boundary condition);
          pressure is doubled at the sphere surface compared to free field.
        * ``"cardioid"`` – cardioid-response sensors on an open sphere;
          equivalent to a combination of pressure and radial velocity
          sensors.
    sensor_kind : {"pressure", "directional"}, default "pressure"
        Sensor capsule response type.
    metadata : dict, optional
        Arbitrary user-defined metadata (array name, manufacturer, etc.).

    Examples
    --------
    >>> from spharray.array.sampling import fibonacci_grid
    >>> geom = ArrayGeometry(
    ...     radius_m=0.042,
    ...     sensor_grid=fibonacci_grid(32),
    ...     array_type="rigid",
    ... )
    >>> geom.n_sensors
    32
    """

    radius_m: float
    sensor_grid: SphericalGrid
    array_type: Literal["open", "rigid", "cardioid"] = "rigid"
    sensor_kind: Literal["pressure", "directional"] = "pressure"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        """Number of microphones / sensors M."""
        return self.sensor_grid.size


@dataclass
class SHSignalFrame:
    """A block of SH-domain signals in the frequency domain.

    Parameters
    ----------
    data : ndarray, shape (n_coeffs, n_freqs) or (n_coeffs, n_freqs, n_frames)
        Complex SH-domain signals.
    freqs_hz : ndarray, shape (n_freqs,)
        Frequency axis in Hz.
    basis : SHBasisSpec
        SH basis specification used to encode this frame.
    """

    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64]
    basis: SHBasisSpec


@dataclass
class SHCovariance:
    """Spatial covariance matrix (or stack) in the SH domain.

    Parameters
    ----------
    data : ndarray, shape (n_coeffs, n_coeffs) or (n_freqs, n_coeffs, n_coeffs)
        Hermitian positive-semidefinite covariance matrix (or per-frequency
        stack).
    freqs_hz : ndarray, shape (n_freqs,), optional
        Frequency axis in Hz.  ``None`` for a single wideband matrix.
    basis : SHBasisSpec
        SH basis specification.
    """

    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64] | None
    basis: SHBasisSpec


@dataclass
class SpatialSpectrumResult:
    """Result of a spatial-spectrum DOA estimator.

    Parameters
    ----------
    spectrum : ndarray, shape (n_grid,)
        Spatial spectrum values at each grid point (real, non-negative).
    grid : SphericalGrid
        The search grid used to compute the spectrum.
    peak_indices : ndarray, shape (n_peaks,)
        Grid indices of detected peaks, sorted by decreasing spectrum value.
    peak_dirs_rad : ndarray, shape (n_peaks, 2)
        Directions of peaks as ``[azimuth, elevation]`` in radians
        (always ``az_el`` convention regardless of the grid's stored
        convention).
    metadata : dict, optional
        Method name (``"method"`` key) and any extra diagnostics.
    """

    spectrum: NDArray[np.float64]
    grid: SphericalGrid
    peak_indices: NDArray[np.int64]
    peak_dirs_rad: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureStyleConfig:
    """Matplotlib style configuration for publication-quality figures.

    Parameters
    ----------
    dpi : int, default 150
        Output resolution in dots per inch.
    figsize : (float, float), default (8.0, 6.0)
        Figure width and height in inches.
    font_family : str, default "DejaVu Sans"
        Matplotlib font family name.
    font_size : float, default 12.0
        Base font size in points.
    line_width : float, default 1.8
        Default line width for curves.
    colormap : str, default "viridis"
        Default colormap name.
    image_ssim_threshold : float, default 0.95
        Minimum SSIM score required for image regression tests to pass.
    max_rel_curve_error : float, default 0.01
        Maximum allowed relative root-mean-square error for curve comparisons.
    """

    dpi: int = 150
    figsize: tuple[float, float] = (8.0, 6.0)
    font_family: str = "DejaVu Sans"
    font_size: float = 12.0
    line_width: float = 1.8
    colormap: str = "viridis"
    image_ssim_threshold: float = 0.95
    max_rel_curve_error: float = 1e-2
