"""spherical-array-processing
============================

A self-contained Python toolkit for spherical array processing, including:

* Spherical-harmonic (SH) basis matrices and transforms
* Radial (Bessel / Hankel) functions for spherical array modelling
* Spatial sampling grids (Fibonacci, equiangle, t-design fallback)
* Free-field plane-wave array simulation
* Fixed and adaptive SH-domain beamforming
* Direction-of-arrival (DOA) spatial spectrum estimators
* Diffuseness estimation from first-order ambisonics / SH covariance
* Diffuse-field coherence models
* Plotting utilities

Quick start
-----------
::

    import numpy as np
    import spherical_array_processing as sap

    # Build a 3rd-order complex SH basis on a Fibonacci grid
    spec = sap.SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
    grid = sap.array.fibonacci_grid(100)
    Y    = sap.sh.matrix(spec, grid)     # (100, 16)

    # Forward and inverse SHT round-trip
    f       = np.random.randn(100)
    coeffs  = sap.sh.direct_sht(f, Y, grid)
    f_rec   = sap.sh.inverse_sht(coeffs, Y).real

    # Maximum directivity beamformer
    b = sap.beamforming.beam_weights_hypercardioid(3)

    # DOA estimation (plane-wave decomposition)
    R      = np.eye(16, dtype=complex)   # toy diffuse covariance
    result = sap.doa.pwd_spectrum(R, grid, spec)
    print("Peak direction:", result.peak_dirs_rad[0])

Submodule layout
----------------
``sap.sh``           — SH basis matrices, transforms, coefficient conversions.
``sap.array``        — Spatial sampling grids, array simulation, aliasing utilities.
``sap.acoustics``    — Bessel / Hankel functions, modal coefficients, equalization.
``sap.beamforming``  — Fixed (cardioid, hypercardioid, …), adaptive (MVDR, LCMV), steering.
``sap.doa``          — PWD / MUSIC spectra, covariance estimation, forward-backward averaging.
``sap.diffuseness``  — IE, TV, SV, CMD diffuseness estimators.
``sap.coherence``    — Diffuse-field coherence models.
``sap.plotting``     — Figure styling and spatial visualisation.
``sap.coords``       — Coordinate transforms, angular distance, great-circle utilities.
``sap.regression``   — MATLAB / Octave regression tooling (optional).
"""

from __future__ import annotations

# ── Submodule exports ────────────────────────────────────────────────────────
from . import (
    acoustics,
    array,
    beamforming,
    coherence,
    coords,
    diffuseness,
    doa,
    plotting,
    sh,
)

# ── Top-level type re-exports (most commonly used) ───────────────────────────
from .types import (
    ArrayGeometry,
    FigureStyleConfig,
    SHBasisSpec,
    SHCovariance,
    SHSignalFrame,
    SpatialSpectrumResult,
    SphericalGrid,
)

__version__ = "0.3.0"

__all__ = [
    # submodules
    "acoustics",
    "array",
    "beamforming",
    "coherence",
    "coords",
    "diffuseness",
    "doa",
    "plotting",
    "sh",
    # types
    "ArrayGeometry",
    "FigureStyleConfig",
    "SHBasisSpec",
    "SHCovariance",
    "SHSignalFrame",
    "SpatialSpectrumResult",
    "SphericalGrid",
    # version
    "__version__",
]
