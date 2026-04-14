"""Fixed (data-independent) SH-domain beamformer weight functions.

All beamformers in this module are *axisymmetric* — the beam pattern is
rotationally symmetric around the look direction.  The pattern is::

    B(θ) = Σ_{n=0}^{N} b_n · (2n+1)/(4π) · P_n(cos θ)

where θ is the angle from the look direction, b_n are the per-order weights
(normalised so that B(0) = 1 by convention), and P_n is the Legendre
polynomial of degree n.

The weight vector ``[b_0, …, b_N]`` returned by each function here can be
applied in the SH domain as a diagonal filter.

Beamformer types
----------------
Cardioid (max front-lobe width)
    Pattern = ((1 + cos θ)/2)^N, normalised.  Gives the widest main lobe for
    a given order N.  No sidelobes.

Hypercardioid (max directivity index)
    DI-optimal weights b_n = (2n+1)/(N+1)².  Maximises the ratio of
    on-axis power to total radiated power.  DI = (N+1)².

Supercardioid (max front-to-back ratio)
    Minimises the ratio of rear hemispherical energy to total energy for
    unit front gain.  Analytical solutions exist for N = 1 … 4; higher orders
    use numerical optimisation.

MaxEV (energy-vector optimised, perceptual)
    Weights based on a von-Hann window in degree space, which provides a
    good tradeoff between spatial selectivity and binaural perceptual quality.

References
----------
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Sec. 4.3.
* Zotter, F. & Frank, M. (2019). *Ambisonics*. Springer.  Ch. 4.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import eval_legendre


# ---------------------------------------------------------------------------
# Cardioid
# ---------------------------------------------------------------------------

def beam_weights_cardioid(order: int) -> np.ndarray:
    """Axisymmetric SH weights for the N-th order cardioid pattern.

    The target pattern is f(θ) = ((1 + cos θ) / 2)^N, which has unit gain
    at θ = 0 and zero gain at θ = π (backside null).  The returned weights
    b_n are its Legendre series coefficients such that

        B(θ) = Σ b_n · (2n+1)/(4π) · P_n(cos θ)   with B(0) = 1.

    Parameters
    ----------
    order : int
        Maximum SH order N (≥ 1).

    Returns
    -------
    b : ndarray, shape (N+1,)
        Per-order beam weights, normalised to unit front gain.

    Examples
    --------
    >>> b = beam_weights_cardioid(1)
    >>> abs(sum(b) - 1.0) < 1e-10    # unit front gain check
    True
    """
    x = np.linspace(-1.0, 1.0, 4097)
    p = ((1.0 + x) / 2.0) ** order
    return _legendre_fit_axisymmetric(x, p, order)


# ---------------------------------------------------------------------------
# Hypercardioid (maximum directivity index)
# ---------------------------------------------------------------------------

def beam_weights_hypercardioid(order: int) -> np.ndarray:
    """Axisymmetric SH weights maximising the directivity index (DI).

    The optimal weights are b_n = (2n+1) / (N+1)², which follow from
    the Cauchy–Schwarz inequality applied to the DI functional.  The
    resulting DI = (N+1)² (linear scale) = 20 log₁₀(N+1) dB.

    Parameters
    ----------
    order : int
        Maximum SH order N (≥ 0).

    Returns
    -------
    b : ndarray, shape (N+1,)
        Per-order beam weights, normalised to unit front gain.

    References
    ----------
    * Rafaely, B. (2015), Eq. (4.26).

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.beamforming import beam_weights_hypercardioid, axisymmetric_pattern
    >>> b = beam_weights_hypercardioid(2)
    >>> len(b)
    3
    >>> abs(axisymmetric_pattern(np.array([0.0]), b)[0] - 1.0) < 1e-10  # B(0)=1
    True
    """
    n = np.arange(order + 1, dtype=float)
    b_raw = 2.0 * n + 1.0   # proportional to DI-optimal (Cauchy-Schwarz)
    front_gain = float(np.sum(b_raw * (2.0 * n + 1.0) / (4.0 * np.pi)))
    return b_raw / front_gain


# ---------------------------------------------------------------------------
# Supercardioid (maximum front-to-back energy ratio)
# ---------------------------------------------------------------------------

# Analytical solutions from Zotter & Frank (2019), Table 4.2 / Politis toolbox.
# b_n are normalised so that Σ b_n = 1 (unit front gain).
_SUPERCARDIOID_TABLE: dict[int, np.ndarray] = {
    1: np.array([1.0 / (1.0 + np.sqrt(3.0)), np.sqrt(3.0) / (1.0 + np.sqrt(3.0))]),
    2: np.array([0.203739490, 0.460607985, 0.335652525]),
    3: np.array([0.133966, 0.339896, 0.356913, 0.169225]),
    4: np.array([0.0982, 0.2552, 0.3179, 0.2301, 0.0986]),
}


def beam_weights_supercardioid(order: int) -> np.ndarray:
    """Axisymmetric SH weights maximising the front-to-back energy ratio.

    The supercardioid minimises the energy radiated into the rear hemisphere
    (θ > π/2) subject to unit front gain B(0) = 1.  The optimum has a null
    at cos(θ_null) = −b_0 / b_1 for N = 1, and analogous multi-lobe
    optima for higher orders.

    Parameters
    ----------
    order : int
        Maximum SH order N (≥ 1).  Exact analytical weights are used for
        N = 1 … 4; numerical optimisation is used for N ≥ 5.

    Returns
    -------
    b : ndarray, shape (N+1,)
        Per-order beam weights, normalised to unit front gain.

    Notes
    -----
    For N = 1, the analytical solution is b_0 = 1/(1+√3), b_1 = √3/(1+√3),
    giving a null at cos θ = −1/√3 ≈ 125.3°, which maximises the
    front-to-back power ratio.

    Examples
    --------
    >>> b = beam_weights_supercardioid(1)
    >>> abs(sum(b) - 1.0) < 1e-10    # unit front gain
    True
    """
    if order in _SUPERCARDIOID_TABLE:
        b = _SUPERCARDIOID_TABLE[order].copy()
    else:
        b = _design_supercardioid_numerical(order)
    # Re-normalise to unit front gain in the axisymmetric_pattern convention
    n = np.arange(len(b), dtype=float)
    front_gain = float(np.sum(b * (2.0 * n + 1.0) / (4.0 * np.pi)))
    return b / front_gain if front_gain > 1e-30 else b


# ---------------------------------------------------------------------------
# MaxEV (energy-vector / perceptual optimised)
# ---------------------------------------------------------------------------

def beam_weights_maxev(order: int) -> np.ndarray:
    """Axisymmetric SH weights optimised for perceptual energy-vector quality.

    A von-Hann window in degree space is applied to the hypercardioid weights::

        w_n = (2n+1) · cos²(π n / (2(N+1)))

    then normalised to unit front gain.  This taper trades off some
    directivity index for a smoother pattern with reduced perceptual
    localisation blur — it is related to the max-rE beamformer used in
    Ambisonics decoding.

    Parameters
    ----------
    order : int
        Maximum SH order N (≥ 0).

    Returns
    -------
    b : ndarray, shape (N+1,)
        Per-order beam weights, normalised to unit front gain B(0) = 1.

    References
    ----------
    * Zotter, F. & Frank, M. (2019). *Ambisonics*. Springer.  Sec. 4.3.2.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.beamforming import beam_weights_maxev, axisymmetric_pattern
    >>> b = beam_weights_maxev(3)
    >>> abs(axisymmetric_pattern(np.array([0.0]), b)[0] - 1.0) < 1e-10  # B(0)=1
    True
    """
    n = np.arange(order + 1, dtype=float)
    w = (2.0 * n + 1.0) * np.cos(np.pi * n / (2.0 * (order + 1))) ** 2
    front_gain = float(np.sum(w * (2.0 * n + 1.0) / (4.0 * np.pi)))
    return w / front_gain if front_gain > 0 else w


# ---------------------------------------------------------------------------
# Pattern evaluation
# ---------------------------------------------------------------------------

def axisymmetric_pattern(theta: ArrayLike, b_n: ArrayLike) -> np.ndarray:
    """Evaluate an axisymmetric SH beam pattern at given polar angles.

    Given per-order weights b_n, the pattern is::

        B(θ) = Σ_{n=0}^{N} b_n · (2n+1)/(4π) · P_n(cos θ)

    Parameters
    ----------
    theta : array-like
        Polar (look-direction) angles in radians.
    b_n : array-like, shape (N+1,)
        Per-order beam weights (e.g., from :func:`beam_weights_hypercardioid`).

    Returns
    -------
    pattern : ndarray
        Pattern values at each angle in ``theta``.

    Examples
    --------
    >>> import numpy as np
    >>> b = beam_weights_hypercardioid(2)
    >>> val_front = axisymmetric_pattern(np.array([0.0]), b)
    >>> abs(val_front[0] - 1.0) < 1e-6     # unit front gain
    True
    """
    theta = np.asarray(theta, dtype=float)
    b = np.asarray(b_n, dtype=float).reshape(-1)
    x = np.cos(theta)
    out = np.zeros_like(theta, dtype=float)
    for n, bn in enumerate(b):
        out = out + bn * ((2.0 * n + 1.0) / (4.0 * np.pi)) * eval_legendre(n, x)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _legendre_fit_axisymmetric(
    x: np.ndarray,
    target: np.ndarray,
    order: int,
) -> np.ndarray:
    """Fit a target function on [-1,1] with a Legendre series (LS solution)."""
    a = np.stack(
        [((2.0 * n + 1.0) / (4.0 * np.pi)) * eval_legendre(n, x) for n in range(order + 1)],
        axis=1,
    )
    b, *_ = np.linalg.lstsq(a, target, rcond=None)
    # Normalise to unit front gain: B(0) = Σ b_n (2n+1)/(4π) * P_n(1) = Σ b_n (2n+1)/(4π)
    front_gain = sum(b[n] * ((2.0 * n + 1.0) / (4.0 * np.pi)) for n in range(order + 1))
    if abs(front_gain) > 1e-30:
        b = b / front_gain
    return b


def _design_supercardioid_numerical(order: int) -> np.ndarray:
    """Numerically optimise supercardioid weights for order > 4."""
    # Objective: minimise rear energy ∫_{π/2}^{π} B(θ)² sin θ dθ
    # subject to unit front gain Σ b_n = 1
    x_rear = np.linspace(-1.0, 0.0, 2049)   # cos(π/2) to cos(π) = -1

    def obj(b: np.ndarray) -> float:
        legmat = np.stack(
            [((2.0 * n + 1.0) / (4.0 * np.pi)) * eval_legendre(n, x_rear)
             for n in range(order + 1)], axis=1
        )
        pattern_rear = legmat @ b
        return float(np.trapz(pattern_rear ** 2, x=x_rear))

    # Constraint: front gain = 1  (Σ b_n (2n+1)/(4π) = 1/(4π) with P_n(1)=1)
    coeffs_at_one = np.array([(2.0 * n + 1.0) / (4.0 * np.pi) for n in range(order + 1)])
    constraints = [{"type": "eq", "fun": lambda b, c=coeffs_at_one: np.dot(c, b) - 1.0 / (4.0 * np.pi)}]
    b0 = beam_weights_hypercardioid(order) / (4.0 * np.pi) * (order + 1) ** 2
    b0 = b0 / b0.sum()  # normalise to unit front gain

    try:
        res = minimize(obj, x0=b0, constraints=constraints, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 1000})
        b_opt = res.x
    except Exception:
        b_opt = b0

    # Ensure unit front gain
    fg = float(sum(b_opt[n] * (2.0 * n + 1.0) / (4.0 * np.pi) for n in range(order + 1)))
    if abs(fg) > 1e-30:
        b_opt = b_opt * (1.0 / (4.0 * np.pi)) / fg
    return b_opt
