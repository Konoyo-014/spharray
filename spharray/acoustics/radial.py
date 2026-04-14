"""Spherical Bessel / Hankel functions and plane-wave radial coefficients.

These functions implement the radial part of the plane-wave expansion in
spherical coordinates.  They are needed to model the pressure field on a
spherical microphone array and to design SH-domain equalisation filters.

Plane-wave expansion on a sphere of radius a
--------------------------------------------
A plane wave arriving from direction Ω_s = (θ_s, φ_s) at a microphone
located at position r = (r, θ, φ) can be expanded as::

    p(r, Ω_s, k) = 4π Σ_{n,m} i^n · b_n(kr, ka) · Y_n^m*(Ω_s) · Y_n^m(θ, φ)

where k = ω/c is the wavenumber, and the radial coefficient b_n depends on
the sphere type:

Open sphere (sensors in free field)
    b_n(kr) = j_n(kr)

Rigid sphere (Neumann boundary condition)
    b_n(kr, ka) = j_n(kr) − [j_n′(ka) / h_n^{(2)′}(ka)] · h_n^{(2)}(kr)

    where j_n is the spherical Bessel function of the first kind, h_n^{(2)}
    is the spherical Hankel function of the second kind (= j_n − i y_n), and
    primes denote derivatives with respect to the argument.  Note that
    h_n^{(2)}(x) = conj(h_n^{(1)}(x)) for real x.

    At the sphere surface (kr = ka) this simplifies using the Wronskian::
        b_n(ka, ka) = −i / (ka)² · 1 / h_n^{(2)′}(ka)

Cardioid sensors on open sphere
    b_n(kr) = j_n(kr) − i j_n′(kr)

    This models a combination of the omnidirectional and radial-gradient
    responses, giving a cardioid pattern.

The full coefficient including the 4π i^n prefactor is returned by
:func:`plane_wave_radial_bn`.

References
----------
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Eq. (3.21).
* Williams, E. G. (1999). *Fourier Acoustics*. Academic Press.  Ch. 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import spherical_jn, spherical_yn

from ..sh.basis import replicate_per_order


# ---------------------------------------------------------------------------
# Spherical Bessel / Hankel wrappers
# ---------------------------------------------------------------------------

def _fa(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def besseljs(n: int, x: ArrayLike) -> NDArray[np.float64]:
    """Spherical Bessel function of the first kind j_n(x).

    Parameters
    ----------
    n : int
        Order (≥ 0).
    x : array-like
        Real argument(s).

    Returns
    -------
    ndarray, float64
        j_n(x).
    """
    return spherical_jn(n, _fa(x))


def besseljsd(n: int, x: ArrayLike) -> NDArray[np.float64]:
    """Derivative of the spherical Bessel function of the first kind, j_n′(x).

    Parameters
    ----------
    n : int
        Order (≥ 0).
    x : array-like
        Real argument(s).

    Returns
    -------
    ndarray, float64
        d/dx j_n(x).
    """
    return spherical_jn(n, _fa(x), derivative=True)


def besselhs(n: int, x: ArrayLike) -> NDArray[np.complex128]:
    """Spherical Hankel function of the first kind h_n^{(1)}(x) = j_n(x) + i y_n(x).

    Parameters
    ----------
    n : int
        Order (≥ 0).
    x : array-like
        Real argument(s) (positive).

    Returns
    -------
    ndarray, complex128
        h_n^{(1)}(x).

    Notes
    -----
    NaN / Inf values appear at x = 0 (singular for n ≥ 1) and are left
    as-is so that callers can handle them appropriately.
    """
    x = _fa(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    return np.asarray(y, dtype=np.complex128)


def besselhsd(n: int, x: ArrayLike) -> NDArray[np.complex128]:
    """Derivative of h_n^{(1)}(x) with respect to x.

    Parameters
    ----------
    n : int
        Order (≥ 0).
    x : array-like
        Real argument(s) (positive).

    Returns
    -------
    ndarray, complex128
        d/dx h_n^{(1)}(x).
    """
    x = _fa(x)
    with np.errstate(all="ignore"):
        y = (
            spherical_jn(n, x, derivative=True)
            + 1j * spherical_yn(n, x, derivative=True)
        )
    return np.asarray(y, dtype=np.complex128)


# ---------------------------------------------------------------------------
# Plane-wave radial coefficient
# ---------------------------------------------------------------------------

def plane_wave_radial_bn(
    n: int,
    kr: ArrayLike,
    ka: ArrayLike | None = None,
    sphere: int | str = "rigid",
) -> NDArray[np.complex128]:
    """Compute the n-th order plane-wave radial coefficient b_n(kr [, ka]).

    The coefficient includes the 4π i^n prefactor of the plane-wave expansion::

        b_n = 4π · i^n · [radial factor]

    Parameters
    ----------
    n : int
        SH degree (≥ 0).
    kr : array-like
        Wave-number times evaluation radius (kr = ω r / c).
    ka : array-like, optional
        Wave-number times sphere radius (ka = ω a / c).  Required only for
        ``"rigid"`` sphere; defaults to ``kr`` if omitted.
    sphere : {0, "open", 1, "rigid", 2, "cardioid"}, default "rigid"
        Sphere type.

    Returns
    -------
    ndarray, complex128
        b_n(kr [, ka]), same shape as ``kr``.

    Notes
    -----
    For the **rigid sphere** the formula involves dividing by
    h_n^{(2)′}(ka).  This function is zero at certain wavenumbers (internal
    resonances of the sphere), producing Inf / NaN at those points.  These
    singularities are inherent to the model and not numerical errors; callers
    may apply regularisation (e.g., Tikhonov) before inverting.

    Examples
    --------
    >>> import numpy as np
    >>> # At ka → 0 the rigid b_0 → 4π (pressure doubling + 1/sqrt(4π) norm)
    >>> b0 = plane_wave_radial_bn(0, kr=np.array([0.01]), ka=np.array([0.01]))
    >>> abs(abs(b0[0]) - 4 * np.pi) < 0.01
    True
    """
    kr = _fa(kr)
    if ka is None:
        ka = kr.copy()
    ka = _fa(ka)

    kind = sphere
    if isinstance(kind, str):
        kind = {"open": 0, "rigid": 1, "cardioid": 2}.get(kind)
        if kind is None:
            raise ValueError(f"unsupported sphere type: {sphere!r}")

    j = 1j
    prefactor = 4.0 * np.pi * (j ** n)

    if kind == 0:  # open sphere
        return prefactor * besseljs(n, kr)

    if kind == 1:  # rigid sphere — Neumann boundary condition
        # b_n = j_n(kr) - [j_n'(ka) / h_n^(2)'(ka)] · h_n^(2)(kr)
        # h_n^(2) = conj(h_n^(1)) for real argument
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = besseljsd(n, ka) / np.conj(besselhsd(n, ka))
        return prefactor * (besseljs(n, kr) - ratio * np.conj(besselhs(n, kr)))

    if kind == 2:  # cardioid sensors on open sphere
        # b_n = j_n(kr) - i j_n'(kr)
        return prefactor * (besseljs(n, kr) - j * besseljsd(n, kr))

    raise ValueError(f"unsupported sphere kind value: {sphere!r}")


# ---------------------------------------------------------------------------
# Radial coefficient matrix
# ---------------------------------------------------------------------------

def bn_matrix(
    max_order: int,
    kr: ArrayLike,
    ka: ArrayLike | None = None,
    sphere: int | str = "rigid",
    repeat_per_order: bool = True,
) -> NDArray[np.complex128]:
    """Build the radial coefficient matrix B_n for orders 0 … N.

    Parameters
    ----------
    max_order : int
        Maximum SH order N.
    kr : array-like, shape (K,)
        Wavenumber × evaluation radius for each of K frequency bins.
    ka : array-like, shape (K,), optional
        Wavenumber × sphere radius.  Defaults to ``kr``.
    sphere : {0, "open", 1, "rigid", 2, "cardioid"}, default "rigid"
        Sphere type passed to :func:`plane_wave_radial_bn`.
    repeat_per_order : bool, default True
        If ``True`` (default), repeat each order's coefficient (2n+1) times
        to match ACN ordering, yielding shape (K, (N+1)²).
        If ``False``, return shape (K, N+1) with one coefficient per order.

    Returns
    -------
    B : ndarray, complex128, shape (K, (N+1)²) or (K, N+1)
        Radial coefficient matrix.

    Examples
    --------
    >>> import numpy as np
    >>> B = bn_matrix(2, kr=np.array([0.5, 1.0]), repeat_per_order=False)
    >>> B.shape
    (2, 3)
    """
    kr_arr = _fa(kr).reshape(-1)
    rows: list[NDArray[np.complex128]] = []
    for n in range(max_order + 1):
        rows.append(plane_wave_radial_bn(n, kr_arr, ka=ka, sphere=sphere))
    b = np.stack(rows, axis=-1)  # shape (K, N+1)

    if not repeat_per_order:
        return b

    # Expand to ACN order: degree n appears (2n+1) times
    n_coeffs = (max_order + 1) ** 2
    out = np.zeros((kr_arr.size, n_coeffs), dtype=np.complex128)
    cursor = 0
    for n in range(max_order + 1):
        count = 2 * n + 1
        out[:, cursor: cursor + count] = b[:, [n]]
        cursor += count
    return out


def sph_modal_coeffs(
    max_order: int,
    kR: ArrayLike,
    array_type: str = "rigid",
) -> NDArray[np.complex128]:
    """Return the per-order modal coefficients b_n(kR) for a spherical array.

    This is a convenience wrapper around :func:`bn_matrix` that sets
    kr = ka = kR (evaluation at the sphere surface itself) and returns
    the (K, N+1) shape (one column per order, not ACN-expanded).

    Parameters
    ----------
    max_order : int
        Maximum SH order N.
    kR : array-like, shape (K,)
        Wavenumber × sphere radius at K frequency bins.
    array_type : {"open", "rigid", "cardioid"}, default "rigid"
        Sphere type.

    Returns
    -------
    B : ndarray, complex128, shape (K, N+1)
        Per-order modal coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> B = sph_modal_coeffs(2, kR=np.array([0.5, 1.0]))
    >>> B.shape
    (2, 3)
    """
    return bn_matrix(max_order, kR, ka=kR, sphere=array_type, repeat_per_order=False)


# ---------------------------------------------------------------------------
# Modal equalisation (radial filter inversion)
# ---------------------------------------------------------------------------

def equalize_modal_coeffs(
    sh_signals: ArrayLike,
    bn: ArrayLike,
    reg_param: float = 1e-4,
    reg_type: str = "tikhonov",
) -> NDArray[np.complex128]:
    """Apply regularised modal equalisation to SH-domain signals.

    A spherical array records the pressure field whose SH coefficients are
    related to the true plane-wave SH amplitudes A_nm by::

        P_nm(f) = b_n(kR) · A_nm(f)

    This function inverts the modal response b_n to recover A_nm::

        A_nm(f) = P_nm(f) / b_n(kR)   (regularised)

    Two regularisation strategies are supported:

    ``"tikhonov"``
        Wiener-type regularisation::

            w_n = b_n* / (|b_n|² + β · max_n|b_n|²)

        where β = ``reg_param``.  This gracefully handles internal resonances
        of rigid spheres where |b_n| → 0.

    ``"softlimit"``
        Soft-limiter (clamp the magnitude before inversion)::

            b_n_reg = b_n · max(|b_n|, δ) / |b_n|   with δ = β · max_n|b_n|

        This zeroes out components where the array sensitivity is below the
        soft threshold.

    Parameters
    ----------
    sh_signals : array-like, shape (..., K, Q)
        SH-domain signals.  The second-to-last axis is frequency (K bins),
        and the last axis is SH channel (Q = (N+1)²).  Leading dimensions
        (if any) are batch dimensions (e.g., sources, time frames).
    bn : array-like, shape (K, Q) or (K, N+1)
        Radial coefficients from :func:`bn_matrix` or :func:`sph_modal_coeffs`.
        If shape (K, N+1), each order is broadcast to the (2n+1) corresponding
        ACN channels automatically.
    reg_param : float, default 1e-4
        Regularisation parameter β (relative to the maximum modal coefficient
        magnitude per frequency bin).  Values in [1e-6, 1e-2] are typical.
    reg_type : {"tikhonov", "softlimit"}, default "tikhonov"
        Regularisation strategy.

    Returns
    -------
    sh_eq : ndarray, complex128, shape (..., K, Q)
        Equalised SH-domain signals.

    Notes
    -----
    The returned signals can be used directly as plane-wave SH amplitudes for
    beamforming, DOA estimation, and other SH-domain processing.

    Examples
    --------
    >>> import numpy as np
    >>> from spharray.acoustics import sph_modal_coeffs, bn_matrix
    >>> N, K = 3, 64
    >>> kR = np.linspace(0.1, 4.0, K)
    >>> bn = bn_matrix(N, kR)                  # (K, 16)
    >>> sh_in = np.random.randn(K, 16) + 0j
    >>> sh_out = equalize_modal_coeffs(sh_in, bn)
    >>> sh_out.shape
    (64, 16)
    """
    sig = np.asarray(sh_signals, dtype=np.complex128)
    bn_arr = np.asarray(bn, dtype=np.complex128)

    # If bn has shape (K, N+1), expand to (K, (N+1)²) via replicate_per_order
    if bn_arr.ndim == 2:
        K, nc = bn_arr.shape
        Q = sig.shape[-1]
        if nc != Q:
            # nc is N+1, Q is (N+1)^2 — expand
            N = nc - 1
            bn_exp = np.zeros((K, (N + 1) ** 2), dtype=np.complex128)
            cursor = 0
            for n in range(N + 1):
                count = 2 * n + 1
                bn_exp[:, cursor: cursor + count] = bn_arr[:, [n]]
                cursor += count
            bn_arr = bn_exp

    # Per-frequency maximum magnitude for normalising reg_param
    mag = np.abs(bn_arr)                    # (K, Q)
    mag_max = mag.max(axis=-1, keepdims=True).clip(min=1e-30)   # (K, 1)

    if reg_type == "tikhonov":
        # w_n = b_n* / (|b_n|^2 + beta * max|b_n|^2)
        denom = mag ** 2 + reg_param * mag_max ** 2
        inv_bn = np.conj(bn_arr) / denom        # (K, Q)
    elif reg_type == "softlimit":
        # clamp |b_n| to max(|b_n|, delta)
        delta = reg_param * mag_max             # (K, 1)
        mag_clamped = np.maximum(mag, delta)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_bn = np.where(mag > 0, mag_clamped / mag * (1.0 / bn_arr), 0.0)
    else:
        raise ValueError(f"unsupported reg_type: {reg_type!r}")

    # Apply: A_nm = P_nm / b_n  ↔  element-wise multiply by inv_bn
    # sig shape (..., K, Q), inv_bn shape (K, Q)
    return sig * inv_bn
