"""Library module.

Usage:
    from spherical_array_processing.beamforming.adaptive import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mvdr_weights(cov: ArrayLike, steering: ArrayLike, diagonal_loading: float = 1e-8) -> np.ndarray:
    """Usage:
        Run mvdr weights.
    
    Args:
        cov: ArrayLike.
        steering: ArrayLike.
        diagonal_loading: float, default=1e-08.
    
    Returns:
        np.ndarray.
    """
    r = np.asarray(cov, dtype=np.complex128)
    d = np.asarray(steering, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be square 2D matrix")
    if d.ndim == 1:
        d = d[:, None]
    if d.shape[0] != r.shape[0]:
        raise ValueError("steering length mismatch")
    rl = r + diagonal_loading * np.trace(r).real / max(r.shape[0], 1) * np.eye(r.shape[0], dtype=r.dtype)
    x = np.linalg.solve(rl, d)
    denom = np.sum(np.conj(d) * x, axis=0, keepdims=True)
    w = x / denom
    return w[:, 0] if w.shape[1] == 1 else w


def lcmv_weights(
    cov: ArrayLike,
    constraint_matrix: ArrayLike,
    response: ArrayLike,
    diagonal_loading: float = 1e-8,
) -> np.ndarray:
    """Usage:
        Run lcmv weights.
    
    Args:
        cov: ArrayLike.
        constraint_matrix: ArrayLike.
        response: ArrayLike.
        diagonal_loading: float, default=1e-08.
    
    Returns:
        np.ndarray.
    """
    r = np.asarray(cov, dtype=np.complex128)
    c = np.asarray(constraint_matrix, dtype=np.complex128)
    f = np.asarray(response, dtype=np.complex128).reshape(-1, 1)
    if c.ndim != 2:
        raise ValueError("constraint_matrix must be 2D")
    if c.shape[0] != r.shape[0]:
        raise ValueError("constraint rows must match cov size")
    if c.shape[1] != f.shape[0]:
        raise ValueError("response length mismatch")
    rl = r + diagonal_loading * np.trace(r).real / max(r.shape[0], 1) * np.eye(r.shape[0], dtype=r.dtype)
    rinv_c = np.linalg.solve(rl, c)
    gram = c.conj().T @ rinv_c
    lam = np.linalg.solve(gram, f)
    w = rinv_c @ lam
    return w[:, 0]

