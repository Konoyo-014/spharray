"""Library module.

Usage:
    from spherical_array_processing.coherence.diffuse import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def diffuse_coherence_matrix_omni(sensor_xyz: ArrayLike, freqs_hz: ArrayLike, c: float = 343.0) -> np.ndarray:
    """Usage:
        Run diffuse coherence matrix omni.
    
    Args:
        sensor_xyz: ArrayLike.
        freqs_hz: ArrayLike.
        c: float, default=343.0.
    
    Returns:
        np.ndarray.
    """
    xyz = np.asarray(sensor_xyz, dtype=float)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("sensor_xyz must be [M,3]")
    d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    k = 2 * np.pi * f / c
    out = np.ones((f.size, xyz.shape[0], xyz.shape[0]), dtype=np.complex128)
    for i, kk in enumerate(k):
        x = kk * d
        out[i] = np.sinc(x / np.pi)  # sinc(x) = sin(pi x)/(pi x)
    return out


def diffuse_coherence_from_weights(w_a: ArrayLike, w_b: ArrayLike) -> complex:
    """Usage:
        Run diffuse coherence from weights.
    
    Args:
        w_a: ArrayLike.
        w_b: ArrayLike.
    
    Returns:
        complex.
    """
    a = np.asarray(w_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(w_b, dtype=np.complex128).reshape(-1)
    if a.size != b.size:
        raise ValueError("weight vectors must have same length")
    na = np.vdot(a, a).real
    nb = np.vdot(b, b).real
    if na <= 0 or nb <= 0:
        return 0.0 + 0.0j
    return np.vdot(a, b) / np.sqrt(na * nb)

