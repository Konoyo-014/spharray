"""SH-domain beamformer steering utilities.

Steers an axisymmetric beam pattern to an arbitrary look direction by
constructing the SH-domain weight vector.

Theory
------
An axisymmetric beam with per-order weights ``b_n`` (e.g. from
:func:`beam_weights_hypercardioid`) points along the +z axis by default.
To steer it toward direction Ω_0 = (az_0, colat_0) we use the SH addition
theorem::

    w_{nm} = b_n · Y_n^m(Ω_0)

Applying these weights in the SH domain produces (with P_nm = Y_nm^*(Ω_s))::

    y(Ω) = Σ_{nm} w_{nm} · Y_n^m(Ω) = Σ_n b_n · B_n(Ω, Ω_0)

where B_n is proportional to the Legendre polynomial P_n(cos θ), θ being
the angle between Ω and Ω_0 — exactly the axisymmetric pattern.

References
----------
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Sec. 4.1.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..types import SHBasisSpec, SphericalGrid
from ..sh import matrix as _sh_matrix


def steer_sh_weights(
    b_n: ArrayLike,
    look_azimuth: float,
    look_angle2: float,
    basis: SHBasisSpec,
) -> NDArray[np.complex128]:
    """Build the SH-domain weight vector for a steered axisymmetric beam.

    Given per-order weights *b_n* (from any fixed beamformer design function)
    and a look direction Ω_0, returns the complex weight vector *w* such that

        y = a @ w        (signal model: a_nm = Y_nm^*(Ω_s))

    produces the beamformed output with the axisymmetric pattern *b_n* steered
    toward Ω_0, where *a* is the row vector of SH signal coefficients.

    Parameters
    ----------
    b_n : array-like, shape (N+1,)
        Per-order beam weights (e.g. from
        :func:`~spharray.beamforming.beam_weights_hypercardioid`).
        Must have length ``basis.max_order + 1``.
    look_azimuth : float
        Azimuth of the look direction in radians.
    look_angle2 : float
        Elevation or colatitude of the look direction in radians,
        as determined by ``basis.angle_convention``.
    basis : SHBasisSpec
        SH basis specification (order, type, angle convention) matching the
        SH signals the weights will be applied to.

    Returns
    -------
    w : ndarray, complex128, shape ((N+1)²,)
        SH-domain weight vector.  Apply as ``y = sh_signal @ w`` or
        equivalently ``y = np.dot(sh_signal, w)``  (signal model: SH signals
        carry the conjugated basis ``Y_nm^*(Ω_s)``, as returned by
        :func:`~spharray.sh.matrix`).

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.types import SHBasisSpec
    >>> from spharray.beamforming import beam_weights_hypercardioid
    >>> basis = SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")
    >>> b  = beam_weights_hypercardioid(2)
    >>> w  = steer_sh_weights(b, look_azimuth=0.0, look_angle2=0.0, basis=basis)
    >>> w.shape
    (9,)
    """
    b_n = np.asarray(b_n, dtype=float).reshape(-1)
    N = len(b_n) - 1
    if N != basis.max_order:
        raise ValueError(
            f"len(b_n)={len(b_n)} implies max_order={N}, "
            f"but basis.max_order={basis.max_order}"
        )

    # Evaluate SH at the look direction — shape (1, Q)
    look = SphericalGrid(
        azimuth=np.array([look_azimuth]),
        angle2=np.array([look_angle2]),
        weights=np.array([4.0 * np.pi]),
        convention=basis.angle_convention,
    )
    Y_look = _sh_matrix(basis, look)    # (1, Q)
    y_look = Y_look[0]                  # (Q,)

    # Expand b_n from per-order to per-coefficient (ACN)
    b_expanded = np.zeros(basis.n_coeffs, dtype=float)
    cursor = 0
    for n in range(N + 1):
        count = 2 * n + 1
        b_expanded[cursor: cursor + count] = b_n[n]
        cursor += count

    # w_nm = b_n * Y_nm(Ω_0)   (Rafaely 2015, Sec. 4.1 — no conjugate here)
    return (b_expanded * y_look).astype(np.complex128)


def beamform_sh(
    sh_signals: ArrayLike,
    weights: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply SH-domain weights to SH signals to produce a beamformed output.

    Parameters
    ----------
    sh_signals : array-like, shape (..., Q)
        SH-domain signals.  The last axis is the SH channel index (ACN).
        Leading dimensions are treated as batch (e.g. frequency bins, time
        frames).
    weights : array-like, shape (Q,)
        Complex SH weight vector (e.g. from :func:`steer_sh_weights` or
        :func:`~spharray.beamforming.mvdr_weights`).

    Returns
    -------
    y : ndarray, complex128, shape (...)
        Beamformed output with the last axis contracted.

    Examples
    --------
    >>> import numpy as np
    >>> sh = np.random.randn(64, 9) + 0j   # 64 freq bins, Q=9
    >>> w  = np.ones(9, dtype=complex)
    >>> y  = beamform_sh(sh, w)
    >>> y.shape
    (64,)
    """
    sig = np.asarray(sh_signals, dtype=np.complex128)
    w   = np.asarray(weights,    dtype=np.complex128).reshape(-1)
    return sig @ w
