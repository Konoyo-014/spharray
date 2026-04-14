"""SH-domain covariance matrix estimation and pre-processing.

Provides tools for:

* Estimating the SH-domain spatial covariance from signal snapshots
* Forward-backward averaging to improve covariance conditioning
* Diagonal loading for regularisation

These are standard pre-processing steps before DOA estimation with
:func:`~spharray.doa.pwd_spectrum` or
:func:`~spharray.doa.music_spectrum`.

References
----------
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Sec. 6.1.
* Stoica, P. & Moses, R. (2005). *Spectral Analysis of Signals*.
  Prentice Hall.  Ch. 6.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_sh_cov(
    sh_snapshots: ArrayLike,
) -> NDArray[np.complex128]:
    """Estimate the SH-domain spatial covariance matrix from snapshots.

    Given *L* snapshot vectors a_l ∈ ℂ^Q, the sample covariance is::

        R = (1/L) Σ_{l=1}^{L} a_l a_l^H

    Parameters
    ----------
    sh_snapshots : array-like, shape (L, Q) or (Q, L)
        SH-domain snapshots.  If the first axis is larger than the second,
        the matrix is transposed automatically so that each *row* is one
        snapshot.

    Returns
    -------
    R : ndarray, complex128, shape (Q, Q)
        Hermitian sample covariance matrix.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> snaps = rng.standard_normal((100, 9)) + 1j * rng.standard_normal((100, 9))
    >>> R = estimate_sh_cov(snaps)
    >>> R.shape
    (9, 9)
    >>> np.allclose(R, R.conj().T)   # Hermitian
    True
    """
    A = np.asarray(sh_snapshots, dtype=np.complex128)
    if A.ndim != 2:
        raise ValueError(f"sh_snapshots must be 2-D, got shape {A.shape}")
    # Ensure rows are snapshots
    L, Q = A.shape
    if L < Q:
        A = A.T          # now (L, Q) even if user passed (Q, L)
        L, Q = A.shape
    R = (A.conj().T @ A) / L
    return R


def forward_backward_cov(
    R: ArrayLike,
) -> NDArray[np.complex128]:
    """Apply forward-backward averaging to a spatial covariance matrix.

    Forward-backward averaging produces a matrix that is both Hermitian and
    *persymmetric* (symmetric with respect to the anti-diagonal), which
    improves the detection of coherent sources::

        R_fb = (R + P R* P) / 2

    where P is the anti-diagonal exchange matrix (P_{ij} = 1 if i+j = Q-1,
    else 0).

    Parameters
    ----------
    R : array-like, shape (Q, Q)
        Hermitian spatial covariance matrix.

    Returns
    -------
    R_fb : ndarray, complex128, shape (Q, Q)
        Forward-backward averaged covariance matrix (Hermitian and real on
        the diagonal).

    Notes
    -----
    Forward-backward averaging effectively doubles the number of snapshots
    and partially decorrelates coherent sources, improving MUSIC and MVDR
    performance in multipath environments.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.array([[2+0j, 1+1j], [1-1j, 2+0j]])
    >>> R_fb = forward_backward_cov(R)
    >>> np.allclose(R_fb, R_fb.conj().T)   # still Hermitian
    True
    """
    R = np.asarray(R, dtype=np.complex128)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square 2-D, got shape {R.shape}")
    Q = R.shape[0]
    P = np.eye(Q, dtype=float)[::-1]   # exchange / flip matrix
    R_back = P @ R.conj() @ P
    return (R + R_back) / 2.0


def diagonal_loading(
    R: ArrayLike,
    load: float | None = None,
    relative: bool = True,
) -> NDArray[np.complex128]:
    """Apply diagonal loading (Tikhonov regularisation) to a covariance matrix.

    Returns ``R + δ I`` where::

        δ = load · trace(R) / Q   if relative is True  (default)
        δ = load                  if relative is False

    Diagonal loading improves numerical conditioning and reduces the
    sensitivity of adaptive beamformers (MVDR / LCMV) to covariance
    estimation errors and model mismatch.

    Parameters
    ----------
    R : array-like, shape (Q, Q)
        Covariance matrix to regularise.
    load : float, optional
        Loading factor.  Default: 1e-4 (relative).
    relative : bool, default True
        If ``True``, the load is interpreted as a fraction of
        ``trace(R) / Q``.  If ``False``, it is an absolute value.

    Returns
    -------
    R_loaded : ndarray, complex128, shape (Q, Q)
        Diagonally loaded covariance.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.eye(4, dtype=complex)
    >>> R_loaded = diagonal_loading(R, load=0.1)
    >>> float(R_loaded[0, 0].real)
    1.1
    """
    R = np.asarray(R, dtype=np.complex128)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"R must be square 2-D, got shape {R.shape}")
    Q = R.shape[0]
    if load is None:
        load = 1e-4
    delta = (float(load) * float(np.trace(R).real) / Q) if relative else float(load)
    return R + delta * np.eye(Q, dtype=np.complex128)
