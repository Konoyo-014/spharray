"""Spherical-harmonic basis matrices and transforms.

The main entry point for most users is :func:`matrix`, which computes the
SH design matrix for any combination of basis type, normalization, and grid.

Examples
--------
>>> from spharray import SHBasisSpec
>>> from spharray.array.sampling import fibonacci_grid
>>> from spharray.sh import matrix, direct_sht, inverse_sht
>>> import numpy as np
>>> spec  = SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
>>> grid  = fibonacci_grid(100)
>>> Y     = matrix(spec, grid)         # (100, 16)
>>> f     = Y[:, 0]                    # DC component = constant on sphere
>>> nm    = direct_sht(f, Y, grid)     # forward SHT
>>> f_rec = inverse_sht(nm, Y)         # synthesis
"""

from .basis import (
    acn_index,
    complex_matrix,
    complex_to_real_coeffs,
    matrix,
    real_matrix,
    real_to_complex_coeffs,
    replicate_per_order,
)
from .transforms import direct_sht, inverse_sht

__all__ = [
    "acn_index",
    "complex_matrix",
    "complex_to_real_coeffs",
    "direct_sht",
    "inverse_sht",
    "matrix",
    "real_matrix",
    "real_to_complex_coeffs",
    "replicate_per_order",
]
