from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import eval_legendre


def beam_weights_cardioid(order: int) -> np.ndarray:
    """Axisymmetric spherical coefficients for (1+cos(theta))^N normalized to unit front gain."""
    x = np.linspace(-1, 1, 4097)
    p = ((1 + x) / 2) ** order
    return _legendre_fit_axisymmetric(x, p, order)


def beam_weights_hypercardioid(order: int) -> np.ndarray:
    """Maximum DI axisymmetric weights (Harmonics/Spatial-style hypercardioid)."""
    return np.array([(2 * n + 1) / (order + 1) ** 2 for n in range(order + 1)], dtype=float)


_SUPERCARDIOID_TABLE = {
    1: np.array([0.36602540378443865, 0.6339745962155614]),
    2: np.array([0.203739490, 0.460607985, 0.335652525]),
    3: np.array([0.133966, 0.339896, 0.356913, 0.169225]),
    4: np.array([0.0982, 0.2552, 0.3179, 0.2301, 0.0986]),
}


def beam_weights_supercardioid(order: int) -> np.ndarray:
    if order in _SUPERCARDIOID_TABLE:
        return _SUPERCARDIOID_TABLE[order].copy()
    # Fallback: numerically optimize front/back ratio surrogate under unit front gain.
    return _design_supercardioid_fallback(order)


def beam_weights_maxev(order: int) -> np.ndarray:
    # Practical approximation via energy-vector preserving taper.
    n = np.arange(order + 1, dtype=float)
    w = (2 * n + 1) * np.cos(np.pi * n / (2 * (order + 1))) ** 2
    return w / np.sum(w)


def axisymmetric_pattern(theta: ArrayLike, b_n: ArrayLike) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    b = np.asarray(b_n, dtype=float).reshape(-1)
    x = np.cos(theta)
    out = np.zeros_like(theta, dtype=float)
    for n, bn in enumerate(b):
        out = out + bn * ((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x)
    return out


def _legendre_fit_axisymmetric(x: np.ndarray, target: np.ndarray, order: int) -> np.ndarray:
    a = np.stack([((2 * n + 1) / (4 * np.pi)) * eval_legendre(n, x) for n in range(order + 1)], axis=1)
    b, *_ = np.linalg.lstsq(a, target, rcond=None)
    front_gain = sum(b[n] * ((2 * n + 1) / (4 * np.pi)) for n in range(order + 1))
    if front_gain != 0:
        b = b / front_gain
    return b


def _design_supercardioid_fallback(order: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, 8193)
    target = np.exp(4.0 * (x - 1.0))  # narrow forward lobe surrogate
    b = _legendre_fit_axisymmetric(x, target, order)
    back = abs(axisymmetric_pattern(np.array([math.pi]), b)[0])
    if back > 0:
        b = b * (1.0 + 0.1 * order)
    return b / np.sum(np.abs(b))

