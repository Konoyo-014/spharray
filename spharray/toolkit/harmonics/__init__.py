"""Library module.

Usage:
    from spharray.toolkit.harmonics import <symbol>
"""

from .._resource_paths import provider_resource_root


HARMONICS_RESOURCE_DIR = provider_resource_root("harmonics")

from .math import (
    bn,
    bn_mat,
    c2s,
    chebyshev_coefficients,
    derivative_ph,
    derivative_th,
    equiangle_sampling,
    gaussian_sampling,
    legendre_coefficients,
    platonic_solid,
    s2c,
    sh2,
    uniform_sampling,
    wigner_d_matrix,
)

__all__ = [
    "HARMONICS_RESOURCE_DIR",
    "bn",
    "bn_mat",
    "c2s",
    "chebyshev_coefficients",
    "derivative_ph",
    "derivative_th",
    "equiangle_sampling",
    "gaussian_sampling",
    "legendre_coefficients",
    "platonic_solid",
    "s2c",
    "sh2",
    "uniform_sampling",
    "wigner_d_matrix",
]
