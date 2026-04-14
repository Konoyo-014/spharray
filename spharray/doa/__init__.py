"""Direction-of-arrival (DOA) spatial spectrum estimators and covariance tools.

Examples
--------
>>> import numpy as np
>>> from spharray.array.sampling import fibonacci_grid
>>> from spharray.doa import pwd_spectrum, music_spectrum
>>> from spharray.types import SHBasisSpec
>>> spec   = SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
>>> grid   = fibonacci_grid(200)
>>> Q      = spec.n_coeffs
>>> R      = np.eye(Q, dtype=complex)
>>> result = pwd_spectrum(R, grid, spec)
>>> result.spectrum.shape == (200,)
True
"""

from .covariance import (
    diagonal_loading,
    estimate_sh_cov,
    forward_backward_cov,
)
from .spectra import (
    music_spectrum,
    peak_pick_spectrum,
    pwd_spectrum,
    spatial_spectrum_from_map,
)

__all__ = [
    # Spectrum estimators
    "music_spectrum",
    "peak_pick_spectrum",
    "pwd_spectrum",
    "spatial_spectrum_from_map",
    # Covariance utilities
    "diagonal_loading",
    "estimate_sh_cov",
    "forward_backward_cov",
]
