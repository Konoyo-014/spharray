"""Library module.

Usage:
    from spherical_array_processing.types import <symbol>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


BasisKind = Literal["complex", "real"]
NormalizationKind = Literal["orthonormal", "n3d", "sn3d"]
AngleConvention = Literal["az_el", "az_colat"]


def _to_1d_float(x: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run to 1d float.
    
    Args:
        x: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


@dataclass
class SHBasisSpec:
    """Usage:
        Instantiate `SHBasisSpec` to work with SHBasisSpec.
    """
    max_order: int
    basis: BasisKind = "complex"
    normalization: NormalizationKind = "orthonormal"
    angle_convention: AngleConvention = "az_el"
    channel_order: Literal["acn"] = "acn"

    @property
    def n_coeffs(self) -> int:
        """Usage:
            Run n coeffs.
        
        Returns:
            int.
        """
        return (self.max_order + 1) ** 2


@dataclass
class SphericalGrid:
    """Usage:
        Instantiate `SphericalGrid` to work with SphericalGrid.
    """
    azimuth: NDArray[np.float64]
    angle2: NDArray[np.float64]
    weights: NDArray[np.float64] | None = None
    convention: AngleConvention = "az_el"
    _xyz_cache: NDArray[np.float64] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Usage:
            Run post init.
        
        Returns:
            None.
        """
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
        """Usage:
            Run size.
        
        Returns:
            int.
        """
        return self.azimuth.size

    @property
    def elevation(self) -> NDArray[np.float64]:
        """Usage:
            Run elevation.
        
        Returns:
            NDArray[np.float64].
        """
        if self.convention == "az_el":
            return self.angle2
        return (np.pi / 2.0) - self.angle2

    @property
    def colatitude(self) -> NDArray[np.float64]:
        """Usage:
            Run colatitude.
        
        Returns:
            NDArray[np.float64].
        """
        if self.convention == "az_colat":
            return self.angle2
        return (np.pi / 2.0) - self.angle2


@dataclass
class ArrayGeometry:
    """Usage:
        Instantiate `ArrayGeometry` to work with ArrayGeometry.
    """
    radius_m: float
    sensor_grid: SphericalGrid
    array_type: Literal["open", "rigid", "cardioid"] = "rigid"
    sensor_kind: Literal["pressure", "directional"] = "pressure"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        """Usage:
            Run n sensors.
        
        Returns:
            int.
        """
        return self.sensor_grid.size


@dataclass
class SHSignalFrame:
    """Usage:
        Instantiate `SHSignalFrame` to work with SHSignalFrame.
    """
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64]
    basis: SHBasisSpec


@dataclass
class SHCovariance:
    """Usage:
        Instantiate `SHCovariance` to work with SHCovariance.
    """
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64] | None
    basis: SHBasisSpec


@dataclass
class SpatialSpectrumResult:
    """Usage:
        Instantiate `SpatialSpectrumResult` to work with SpatialSpectrumResult.
    """
    spectrum: NDArray[np.float64]
    grid: SphericalGrid
    peak_indices: NDArray[np.int64]
    peak_dirs_rad: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureStyleConfig:
    """Usage:
        Instantiate `FigureStyleConfig` to work with FigureStyleConfig.
    """
    dpi: int = 150
    figsize: tuple[float, float] = (8.0, 6.0)
    font_family: str = "DejaVu Sans"
    font_size: float = 12.0
    line_width: float = 1.8
    colormap: str = "viridis"
    image_ssim_threshold: float = 0.95
    max_rel_curve_error: float = 1e-2
