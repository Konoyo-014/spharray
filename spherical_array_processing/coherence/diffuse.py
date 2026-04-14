"""Diffuse-field spatial coherence models.

An isotropic diffuse sound field has the property that sound energy arrives
uniformly from all directions with uncorrelated phases.  The resulting
inter-microphone coherence depends only on the sensor separation.

Omnidirectional sensors
-----------------------
For two omnidirectional microphones separated by distance d in an isotropic
diffuse field (Jacobsen & Roisin, 2000)::

    Γ(f) = sin(k d) / (k d) = sinc(k d / π)

where k = 2π f / c and sinc(x) = sin(π x) / (π x) (NumPy convention).

This expression is real and bounded in [-0.22, 1].  It is 1 at d = 0,
oscillates, and decays toward 0 for large kd.

References
----------
* Jacobsen, F. & Roisin, T. (2000). "The coherence of reverberant sound
  fields". *Journal of the Acoustical Society of America*, 108(1), 204–210.
* Habets, E. A. P. & Gannot, S. (2007). "Generating sensor signals in
  isotropic noise fields". *JASA*, 122(6), 3464–3470.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def diffuse_coherence_matrix_omni(
    sensor_xyz: ArrayLike,
    freqs_hz: ArrayLike,
    c: float = 343.0,
) -> np.ndarray:
    """Compute the inter-sensor coherence matrix for an isotropic diffuse field.

    Parameters
    ----------
    sensor_xyz : array-like, shape (M, 3)
        Sensor positions in metres (Cartesian).
    freqs_hz : array-like, shape (K,)
        Frequencies in Hz.
    c : float, default 343.0
        Speed of sound in m/s.

    Returns
    -------
    Gamma : ndarray, complex128, shape (K, M, M)
        Coherence matrix at each frequency.  Diagonal entries are 1.
        Off-diagonal entry (k, m, n) = sin(k_k · d_{mn}) / (k_k · d_{mn}),
        where d_{mn} = ‖r_m − r_n‖ and k_k = 2π f_k / c.

    Notes
    -----
    The result is real-valued (but returned as complex128 for compatibility
    with downstream Hermitian operations).

    Examples
    --------
    >>> import numpy as np
    >>> xyz = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=float)
    >>> G = diffuse_coherence_matrix_omni(xyz, np.array([1000.0]))
    >>> G.shape
    (1, 2, 2)
    >>> abs(G[0, 0, 0] - 1.0) < 1e-12   # auto-coherence = 1
    True
    """
    xyz = np.asarray(sensor_xyz, dtype=float)
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            f"sensor_xyz must have shape (M, 3), got {xyz.shape}"
        )
    M = xyz.shape[0]
    # Pairwise distances d_{mn}  — shape (M, M)
    diff = xyz[:, None, :] - xyz[None, :, :]   # (M, M, 3)
    d = np.linalg.norm(diff, axis=-1)           # (M, M)

    # Wavenumbers  k[f]  — shape (K,)
    k = 2.0 * np.pi * f / c                     # (K,)

    # Coherence: sinc(k d / π) = sin(k d) / (k d)
    # NumPy's sinc(x) = sin(π x) / (π x), so sinc(kd/π) = sin(kd)/(kd).
    kd = k[:, None, None] * d[None, :, :]       # (K, M, M)  — vectorised
    # At d = 0 (same sensor), kd = 0 and sinc(0) = 1 by definition.
    gamma = np.sinc(kd / np.pi)                  # (K, M, M), real

    return gamma.astype(np.complex128)


def diffuse_coherence_from_weights(
    w_a: ArrayLike,
    w_b: ArrayLike,
) -> complex:
    """Compute the diffuse-field coherence between two steered beams.

    Given beamformer weight vectors w_a and w_b in the SH domain, the
    coherence of a diffuse field between the two beams is::

        γ = w_a^H w_b / √(‖w_a‖² ‖w_b‖²)

    This exploits the fact that in an isotropic diffuse field, the SH-domain
    covariance is proportional to the identity matrix, so the coherence
    simplifies to the normalised inner product of the weight vectors.

    Parameters
    ----------
    w_a : array-like, shape (Q,)
        SH weight vector of the first beam.
    w_b : array-like, shape (Q,)
        SH weight vector of the second beam.

    Returns
    -------
    complex
        Diffuse-field coherence ∈ [−1, 1] (or complex for asymmetric weights).

    Examples
    --------
    >>> import numpy as np
    >>> w = np.array([1.0, 0.0, 0.0, 0.0])
    >>> abs(diffuse_coherence_from_weights(w, w) - 1.0) < 1e-12
    True
    """
    a = np.asarray(w_a, dtype=np.complex128).reshape(-1)
    b = np.asarray(w_b, dtype=np.complex128).reshape(-1)
    if a.size != b.size:
        raise ValueError(
            f"weight vectors must have the same length ({a.size} ≠ {b.size})"
        )
    na = float(np.vdot(a, a).real)
    nb = float(np.vdot(b, b).real)
    if na <= 0.0 or nb <= 0.0:
        return 0.0 + 0.0j
    return complex(np.vdot(a, b) / np.sqrt(na * nb))
