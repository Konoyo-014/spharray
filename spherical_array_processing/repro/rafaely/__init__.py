from .._reference_paths import provider_reference_root


RAFAELY_SOURCE_ROOT = provider_reference_root("rafaely")
RAFAELY_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)

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
    "RAFAELY_SOURCE_ROOT",
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
