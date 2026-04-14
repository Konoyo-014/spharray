"""Acoustic radial functions for spherical array processing.

The key function for most users is :func:`bn_matrix`, which returns the
frequency-dependent radial equalisation matrix for a given array type and SH
order.

:func:`equalize_modal_coeffs` applies regularised modal inversion to convert
raw SH-domain pressures to plane-wave SH amplitudes suitable for beamforming
and DOA estimation.

Examples
--------
>>> import numpy as np
>>> from spharray.acoustics import bn_matrix, sph_modal_coeffs
>>> kR = np.linspace(0.1, 3.0, 20)          # wavenumber × radius
>>> B = bn_matrix(3, kR, sphere="rigid")    # (20, 16) -- ACN expanded
>>> Bn = sph_modal_coeffs(3, kR)            # (20,  4) -- per order
"""

from .radial import (
    besselhs,
    besselhsd,
    besseljs,
    besseljsd,
    bn_matrix,
    equalize_modal_coeffs,
    plane_wave_radial_bn,
    sph_modal_coeffs,
)

__all__ = [
    "besselhs",
    "besselhsd",
    "besseljs",
    "besseljsd",
    "bn_matrix",
    "equalize_modal_coeffs",
    "plane_wave_radial_bn",
    "sph_modal_coeffs",
]
