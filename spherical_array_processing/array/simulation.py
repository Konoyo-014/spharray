"""Library module.

Usage:
    from spherical_array_processing.array.simulation import <symbol>
"""

from __future__ import annotations

import numpy as np

from ..coords import unit_sph_to_cart
from ..types import ArrayGeometry, SphericalGrid


def simulate_plane_wave_array_response(
    fft_len: int,
    fs: float,
    geometry: ArrayGeometry,
    source_grid: SphericalGrid,
    c: float = 343.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-field plane-wave transfer functions.

    Returns `(freqs_hz, H)` where `H.shape == (n_bins, n_sensors, n_sources)`.
    """
    n_bins = fft_len // 2 + 1
    freqs = np.arange(n_bins, dtype=float) * fs / fft_len
    k = 2.0 * np.pi * freqs / c
    mic_xyz = unit_sph_to_cart(
        geometry.sensor_grid.azimuth,
        geometry.sensor_grid.angle2,
        convention=geometry.sensor_grid.convention,
    ) * geometry.radius_m
    src_u = unit_sph_to_cart(source_grid.azimuth, source_grid.angle2, convention=source_grid.convention)
    # Plane wave phase at sensors: exp(-j k r·u)
    proj = mic_xyz @ src_u.T  # [M, S]
    h = np.exp(-1j * k[:, None, None] * proj[None, :, :])
    return freqs, h

