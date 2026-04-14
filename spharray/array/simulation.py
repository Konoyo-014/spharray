"""Free-field plane-wave array simulation.

This module simulates the frequency-domain response of a rigid / open
spherical microphone array to one or more simultaneous plane-wave sources.

Plane-wave model
----------------
In the free-field (no sphere), the pressure at position r due to a plane
wave arriving from direction Ω_s with unit amplitude is::

    p(r, f) = exp(−i k r · û_s)

where k = 2π f / c, r is the sensor position vector and û_s is the unit
vector pointing *toward* the source.  The exponent computes the time delay
τ = (r · û_s) / c between the array phase centre and the sensor.

Note: this model does *not* include the scattering effects of a rigid sphere.
For rigid-sphere array modelling in the SH domain, combine this simulation
with the modal equalisation provided by :mod:`spharray.acoustics`.
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
    """Compute free-field plane-wave transfer functions for a spherical array.

    Parameters
    ----------
    fft_len : int
        FFT length.  The frequency axis has ``fft_len // 2 + 1`` bins
        (one-sided, DC to Nyquist).
    fs : float
        Sample rate in Hz.
    geometry : ArrayGeometry
        Array geometry (radius, sensor positions, array type).
    source_grid : SphericalGrid
        Directions of the S incoming plane waves.
    c : float, default 343.0
        Speed of sound in m/s.

    Returns
    -------
    freqs_hz : ndarray, float64, shape (n_bins,)
        Frequency axis in Hz (0, fs/fft_len, …, fs/2).
    H : ndarray, complex128, shape (n_bins, n_sensors, n_sources)
        Transfer functions H[f, m, s] from source s to sensor m at frequency f.

    Notes
    -----
    The model is a free-field (anechoic) delay model.  It does *not* include
    the scattering of a rigid sphere.  For a physically accurate simulation of
    a rigid-sphere array, the modal equalisation filters from
    :mod:`spharray.acoustics` should be applied.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.array.sampling import fibonacci_grid
    >>> from spharray.types import ArrayGeometry
    >>> geom = ArrayGeometry(radius_m=0.05, sensor_grid=fibonacci_grid(16))
    >>> src  = fibonacci_grid(4)
    >>> freqs, H = simulate_plane_wave_array_response(512, 16000.0, geom, src)
    >>> H.shape
    (257, 16, 4)
    """
    n_bins = fft_len // 2 + 1
    freqs_hz = np.arange(n_bins, dtype=float) * (fs / fft_len)
    k = 2.0 * np.pi * freqs_hz / c          # shape (n_bins,)

    # Sensor positions in Cartesian coordinates (M, 3)
    mic_xyz = unit_sph_to_cart(
        geometry.sensor_grid.azimuth,
        geometry.sensor_grid.angle2,
        convention=geometry.sensor_grid.convention,
    ) * geometry.radius_m

    # Source direction unit vectors (S, 3)
    src_u = unit_sph_to_cart(
        source_grid.azimuth,
        source_grid.angle2,
        convention=source_grid.convention,
    )

    # Projection: proj[m, s] = r_m · û_s  (path-length delay in metres)
    proj = mic_xyz @ src_u.T              # shape (M, S)

    # Transfer function: H[f, m, s] = exp(-i k[f] proj[m, s])
    H = np.exp(-1j * k[:, None, None] * proj[None, :, :])
    return freqs_hz, H
