from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


BasisKind = Literal["complex", "real"]
NormalizationKind = Literal["orthonormal", "n3d", "sn3d"]
AngleConvention = Literal["az_el", "az_colat"]


def _to_1d_float(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


@dataclass
class SHBasisSpec:
    max_order: int
    basis: BasisKind = "complex"
    normalization: NormalizationKind = "orthonormal"
    angle_convention: AngleConvention = "az_el"
    channel_order: Literal["acn"] = "acn"

    @property
    def n_coeffs(self) -> int:
        return (self.max_order + 1) ** 2


@dataclass
class SphericalGrid:
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
        return self.azimuth.size

    @property
    def elevation(self) -> NDArray[np.float64]:
        if self.convention == "az_el":
            return self.angle2
        return (np.pi / 2.0) - self.angle2

    @property
    def colatitude(self) -> NDArray[np.float64]:
        if self.convention == "az_colat":
            return self.angle2
        return (np.pi / 2.0) - self.angle2


@dataclass
class ArrayGeometry:
    radius_m: float
    sensor_grid: SphericalGrid
    array_type: Literal["open", "rigid", "cardioid"] = "rigid"
    sensor_kind: Literal["pressure", "directional"] = "pressure"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return self.sensor_grid.size


@dataclass
class SHSignalFrame:
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64]
    basis: SHBasisSpec


@dataclass
class SHCovariance:
    data: NDArray[np.complex128]
    freqs_hz: NDArray[np.float64] | None
    basis: SHBasisSpec


@dataclass
class SpatialSpectrumResult:
    spectrum: NDArray[np.float64]
    grid: SphericalGrid
    peak_indices: NDArray[np.int64]
    peak_dirs_rad: NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureReproConfig:
    dpi: int = 150
    figsize: tuple[float, float] = (8.0, 6.0)
    font_family: str = "DejaVu Sans"
    font_size: float = 12.0
    line_width: float = 1.8
    colormap: str = "viridis"
    image_ssim_threshold: float = 0.95
    max_rel_curve_error: float = 1e-2
