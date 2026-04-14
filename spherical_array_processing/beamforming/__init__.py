"""Fixed and adaptive SH-domain beamformers, and beam steering utilities.

Fixed beamformers
-----------------
:func:`beam_weights_cardioid`      — widest main lobe, rear null.
:func:`beam_weights_hypercardioid` — maximum directivity index.
:func:`beam_weights_supercardioid` — maximum front-to-back energy ratio.
:func:`beam_weights_maxev`         — energy-vector / perceptual optimised.
:func:`axisymmetric_pattern`       — evaluate a beam pattern at given angles.

Adaptive beamformers
--------------------
:func:`mvdr_weights`               — Capon / MVDR.
:func:`lcmv_weights`               — Frost / LCMV (multi-constraint).

Steering utilities
------------------
:func:`steer_sh_weights`           — SH weight vector for a steered axisymmetric beam.
:func:`beamform_sh`                — apply SH weights to produce a beamformed signal.

Examples
--------
>>> from spherical_array_processing.beamforming import (
...     beam_weights_hypercardioid, axisymmetric_pattern,
... )
>>> import numpy as np
>>> b = beam_weights_hypercardioid(3)
>>> pattern_front = axisymmetric_pattern(np.array([0.0]), b)
>>> abs(pattern_front[0] - 1.0) < 1e-6
True
"""

from .adaptive import lcmv_weights, mvdr_weights
from .fixed import (
    axisymmetric_pattern,
    beam_weights_cardioid,
    beam_weights_hypercardioid,
    beam_weights_maxev,
    beam_weights_supercardioid,
)
from .steer import beamform_sh, steer_sh_weights

__all__ = [
    # Fixed
    "axisymmetric_pattern",
    "beam_weights_cardioid",
    "beam_weights_hypercardioid",
    "beam_weights_maxev",
    "beam_weights_supercardioid",
    # Adaptive
    "lcmv_weights",
    "mvdr_weights",
    # Steering
    "beamform_sh",
    "steer_sh_weights",
]
