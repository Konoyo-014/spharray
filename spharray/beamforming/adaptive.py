"""Data-dependent (adaptive) SH-domain beamformers.

Both beamformers work on the SH-domain covariance matrix R ∈ ℂ^{Q×Q}
where Q = (N+1)² is the number of SH coefficients.

MVDR (Minimum Variance Distortionless Response / Capon)
--------------------------------------------------------
Minimise the output variance w^H R w subject to w^H d = 1 (distortionless
response toward the look direction with steering vector d).  Solution::

    w_MVDR = (R + ε I)^{-1} d / (d^H (R + ε I)^{-1} d)

The diagonal loading ε = δ · trace(R) / Q regularises the inversion.

LCMV (Linearly Constrained Minimum Variance / Frost)
-----------------------------------------------------
Generalise MVDR to multiple simultaneous linear constraints::

    min w^H R w    subject to   C^H w = f

where C ∈ ℂ^{Q×K} is the constraint matrix and f ∈ ℂ^K the desired
response vector.  Solution::

    w_LCMV = (R + ε I)^{-1} C [(C^H (R + ε I)^{-1} C)^{-1}] f

References
----------
* Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley.  Ch. 6.
* Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*.
  Springer.  Sec. 4.4.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mvdr_weights(
    cov: ArrayLike,
    steering: ArrayLike,
    diagonal_loading: float = 1e-8,
) -> np.ndarray:
    """Compute MVDR (Capon) beamformer weights.

    Parameters
    ----------
    cov : array-like, shape (Q, Q)
        Hermitian spatial covariance matrix in the SH domain.
    steering : array-like, shape (Q,) or (Q, K)
        SH steering vector(s) toward the look direction(s).
        If shape is (Q, K), K independent MVDR problems are solved.
    diagonal_loading : float, default 1e-8
        Regularisation factor δ.  The loaded matrix is
        R_loaded = R + δ · trace(R)/Q · I.  Set to 0 to disable.

    Returns
    -------
    w : ndarray, complex128, shape (Q,) or (Q, K)
        MVDR weight vector(s) satisfying w^H d = 1.

    Notes
    -----
    The unit response guarantee w^H d = 1 holds exactly (up to floating-point
    precision) regardless of the diagonal loading level.

    Examples
    --------
    >>> import numpy as np
    >>> Q = 4
    >>> R = np.eye(Q, dtype=complex)
    >>> d = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    >>> w = mvdr_weights(R, d)
    >>> abs(np.conj(w) @ d - 1.0) < 1e-12    # unit response
    True
    """
    r = np.asarray(cov, dtype=np.complex128)
    d = np.asarray(steering, dtype=np.complex128)
    if r.ndim != 2 or r.shape[0] != r.shape[1]:
        raise ValueError("cov must be a square 2-D matrix")
    if d.ndim == 1:
        d = d[:, None]
    if d.shape[0] != r.shape[0]:
        raise ValueError(
            f"steering has {d.shape[0]} rows but cov has shape {r.shape}"
        )
    q = r.shape[0]
    load = diagonal_loading * r.trace().real / max(q, 1)
    rl = r + load * np.eye(q, dtype=r.dtype)
    x = np.linalg.solve(rl, d)                        # R^{-1} d
    denom = np.sum(np.conj(d) * x, axis=0, keepdims=True)  # d^H R^{-1} d
    w = x / denom
    return w[:, 0] if w.shape[1] == 1 else w


def lcmv_weights(
    cov: ArrayLike,
    constraint_matrix: ArrayLike,
    response: ArrayLike,
    diagonal_loading: float = 1e-8,
) -> np.ndarray:
    """Compute LCMV (Frost) beamformer weights.

    Parameters
    ----------
    cov : array-like, shape (Q, Q)
        Hermitian spatial covariance matrix in the SH domain.
    constraint_matrix : array-like, shape (Q, K)
        Constraint matrix C — each column is a steering vector for one
        constrained direction.
    response : array-like, shape (K,) or (K, 1)
        Desired complex response for each constraint, e.g.,
        ``[1, 0]`` for look direction and one null.
    diagonal_loading : float, default 1e-8
        Regularisation factor (same convention as :func:`mvdr_weights`).

    Returns
    -------
    w : ndarray, complex128, shape (Q,)
        LCMV weight vector satisfying C^H w = f.

    Examples
    --------
    >>> import numpy as np
    >>> Q = 4
    >>> R = np.eye(Q, dtype=complex)
    >>> d_look  = np.array([[1], [0], [0], [0]], dtype=complex)
    >>> d_null  = np.array([[0], [1], [0], [0]], dtype=complex)
    >>> C = np.hstack([d_look, d_null])   # (4, 2)
    >>> f = np.array([1.0, 0.0])          # pass look, null the other
    >>> w = lcmv_weights(R, C, f)
    >>> np.allclose(C.conj().T @ w, f, atol=1e-10)
    True
    """
    r = np.asarray(cov, dtype=np.complex128)
    c = np.asarray(constraint_matrix, dtype=np.complex128)
    f = np.asarray(response, dtype=np.complex128).reshape(-1, 1)
    if c.ndim != 2:
        raise ValueError("constraint_matrix must be 2-D")
    if c.shape[0] != r.shape[0]:
        raise ValueError(
            f"constraint_matrix has {c.shape[0]} rows but cov has shape {r.shape}"
        )
    if c.shape[1] != f.shape[0]:
        raise ValueError(
            f"constraint_matrix has {c.shape[1]} columns but response has "
            f"{f.shape[0]} elements"
        )
    q = r.shape[0]
    load = diagonal_loading * r.trace().real / max(q, 1)
    rl = r + load * np.eye(q, dtype=r.dtype)
    rinv_c = np.linalg.solve(rl, c)        # R^{-1} C,  shape (Q, K)
    gram = c.conj().T @ rinv_c             # C^H R^{-1} C,  shape (K, K)
    lam = np.linalg.solve(gram, f)         # Lagrange multipliers
    w = rinv_c @ lam                        # shape (Q, 1)
    return w[:, 0]
