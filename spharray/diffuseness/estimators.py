"""Diffuseness estimators for first-order ambisonics (FOA) and SH signals.

Diffuseness Ψ ∈ [0, 1] quantifies the degree to which a sound field is
spatially diffuse (Ψ = 1) versus dominated by coherent plane waves (Ψ = 0).

Estimators implemented
----------------------
IE (Intensity–Energy ratio)
    Ψ_IE = 1 − |E[I]| / E

    where I = ρ₀ c Re(W* V) is the instantaneous acoustic intensity,
    E = ρ₀ c² |W|² / 2 is the potential energy density (up to a constant
    shared by both), and E[·] is a time / frequency average.

    This estimator is robust and fast but requires the omnidirectional W
    and three velocity components (X, Y, Z) of a first-order ambisonic
    signal.

    Reference: Ahonen & Pulkki (2009), "Diffuseness estimation using
    temporal variation of intensity vectors", EUSIPCO.

TV (Temporal Variance of intensity direction)
    Ψ_TV = 1 − ‖E[Î]‖ / E[‖Î‖]

    where Î = I / ‖I‖ is the unit-norm intensity direction (instantaneous
    DOA estimate).  A diffuse field produces randomly oriented intensity
    vectors, so E[Î] ≈ 0 → Ψ_TV ≈ 1.  A coherent source gives
    E[Î] ≈ constant unit vector → Ψ_TV ≈ 0.

    Reference: Sälli & Pulkki (2018), "Parametric time-frequency
    representation of spatial sound in virtual reality".

SV (Spatial Variance of unit DOA vectors)
    Ψ_SV = 1 − ‖E[î]‖

    where î = I / ‖I‖ are the unit-norm intensity vectors.  Equivalent to
    TV but without normalising by the mean magnitude.  Faster to compute but
    slightly more sensitive to energy fluctuations.

CMD (Covariance Matrix Decomposition)
    Operates on the SH-domain covariance matrix R.  A pure diffuse field in
    an N-th order SH domain has R = σ² I ((N+1)² × (N+1)²), so all
    eigenvalues are equal.  Diffuseness is estimated from the spread of
    eigenvalues relative to their mean.

    Reference: Politis, A. et al. (2015), "Direction of arrival and
    diffuseness estimation above the spatial Nyquist frequency for
    spherical microphone arrays", ICASSP.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Intensity extraction (FOA helper)
# ---------------------------------------------------------------------------

def intensity_vectors_from_foa(foa: ArrayLike) -> np.ndarray:
    """Compute instantaneous acoustic intensity-like vectors from FOA signals.

    For a first-order ambisonic signal [W, X, Y, Z] (pressure and velocity
    channels in ACN/SN3D convention), the unnormalised instantaneous
    intensity is proportional to::

        I(t) = Re(W*(t) · [X(t), Y(t), Z(t)])

    Parameters
    ----------
    foa : array-like, shape (..., 4) or (..., T, 4)
        Complex FOA signal with channels [W, X, Y, Z] in the last axis.
        Any leading batch dimensions are preserved.

    Returns
    -------
    I : ndarray, float64, shape (..., 3)
        Instantaneous intensity vectors (unnormalised, in arbitrary units).
        The three columns correspond to x, y, z Cartesian components.

    Examples
    --------
    >>> import numpy as np
    >>> T = 100
    >>> foa = np.zeros((T, 4), dtype=complex)
    >>> foa[:, 0] = 1.0   # pure pressure → zero intensity
    >>> I = intensity_vectors_from_foa(foa)
    >>> np.allclose(I, 0.0)
    True
    """
    a = np.asarray(foa, dtype=np.complex128)
    if a.shape[-1] < 4:
        raise ValueError(
            f"foa must have at least 4 channels [W, X, Y, Z] in the last axis, "
            f"got {a.shape[-1]}"
        )
    w = a[..., 0]          # omnidirectional / pressure channel
    v = a[..., 1:4]        # velocity channels X, Y, Z
    return np.real(np.conj(w)[..., None] * v)


# ---------------------------------------------------------------------------
# IE diffuseness (covariance matrix input)
# ---------------------------------------------------------------------------

def diffuseness_ie(pv_cov: ArrayLike) -> float:
    """Estimate diffuseness from a pressure–velocity covariance matrix (IE method).

    Parameters
    ----------
    pv_cov : array-like, shape (4, 4) or larger
        Covariance matrix of [W, X, Y, Z] channels.  The upper-left 4×4
        block is used.  Should be Hermitian.

    Returns
    -------
    Psi : float
        Diffuseness estimate ∈ [0, 1].

    Notes
    -----
    The formula is::

        Ψ_IE = 1 − |I_avg| / E_avg

    where E_avg = trace(C) / 2 and I_avg = Re(C[1:4, 0])
    (time-averaged intensity from the cross-covariance).

    Examples
    --------
    >>> import numpy as np
    >>> # Diffuse field: R ∝ I → zero mean intensity
    >>> C = np.eye(4, dtype=complex)
    >>> abs(diffuseness_ie(C) - 1.0) < 1e-10
    True
    """
    c = np.asarray(pv_cov, dtype=np.complex128)
    if c.shape[0] < 4 or c.shape[1] < 4:
        raise ValueError("pv_cov must be at least 4×4")
    ia = np.real(c[1:4, 0])        # mean intensity vector
    ia_norm = float(np.linalg.norm(ia))
    e = float(np.real(np.trace(c))) / 2.0   # mean energy (up to ρ₀c²/2)
    if e <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - ia_norm / e, 0.0, 1.0))


# ---------------------------------------------------------------------------
# TV diffuseness (instantaneous intensity vectors input)
# ---------------------------------------------------------------------------

def diffuseness_tv(i_vecs: ArrayLike) -> float:
    """Estimate diffuseness from instantaneous intensity vectors (TV method).

    Parameters
    ----------
    i_vecs : array-like, shape (T, 3)
        Instantaneous acoustic intensity vectors at T time frames.
        Obtain from :func:`intensity_vectors_from_foa`.

    Returns
    -------
    Psi : float
        Diffuseness estimate ∈ [0, 1].

    Notes
    -----
    The formula is::

        Ψ_TV = 1 − ‖mean(Î)‖ / mean(‖Î‖)

    where Î_t = I_t / ‖I_t‖ (unit-normalised intensity direction at frame t).
    This is purely based on the spread of DOA estimates across frames, not
    on magnitude.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # Uniformly random directions → fully diffuse
    >>> i_random = rng.standard_normal((2000, 3))
    >>> Psi = diffuseness_tv(i_random)
    >>> Psi > 0.95    # should be close to 1 for truly random directions
    True
    """
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError(
            f"i_vecs must have shape (T, 3), got {i.shape}"
        )
    norms = np.linalg.norm(i, axis=1)        # (T,)
    active = norms > 1e-12
    if not active.any():
        return 1.0
    # Unit-normalised intensity directions
    i_hat = i[active] / norms[active, None]  # (T_active, 3)
    mean_i_hat = np.mean(i_hat, axis=0)      # (3,)
    norm_mean = float(np.linalg.norm(mean_i_hat))
    return float(np.clip(1.0 - norm_mean, 0.0, 1.0))


# ---------------------------------------------------------------------------
# SV diffuseness
# ---------------------------------------------------------------------------

def diffuseness_sv(i_vecs: ArrayLike) -> float:
    """Estimate diffuseness from instantaneous intensity vectors (SV method).

    Parameters
    ----------
    i_vecs : array-like, shape (T, 3)
        Instantaneous acoustic intensity vectors at T time frames.

    Returns
    -------
    Psi : float
        Diffuseness estimate ∈ [0, 1].

    Notes
    -----
    The formula is::

        Ψ_SV = 1 − ‖mean(Î)‖

    where Î_t = I_t / ‖I_t‖.  Similar to TV but normalises each vector
    individually before averaging; equivalent to measuring the resultant
    vector length of uniformly-weighted unit DOA vectors.

    Examples
    --------
    >>> import numpy as np
    >>> # Single coherent source → Psi close to 0
    >>> i_coherent = np.ones((50, 3))   # all pointing same direction
    >>> diffuseness_sv(i_coherent) < 0.05
    True
    """
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError(
            f"i_vecs must have shape (T, 3), got {i.shape}"
        )
    norms = np.linalg.norm(i, axis=1)
    active = norms > 1e-12
    if not active.any():
        return 1.0
    i_hat = i[active] / norms[active, None]
    mean_doa = np.mean(i_hat, axis=0)
    return float(np.clip(1.0 - np.linalg.norm(mean_doa), 0.0, 1.0))


# ---------------------------------------------------------------------------
# CMD diffuseness (SH covariance matrix input)
# ---------------------------------------------------------------------------

def diffuseness_cmd(sh_cov: ArrayLike) -> tuple[float, np.ndarray]:
    """Estimate diffuseness from the SH-domain covariance matrix (CMD method).

    Parameters
    ----------
    sh_cov : array-like, shape ((N+1)², (N+1)²)
        Hermitian SH-domain covariance matrix for an N-th order expansion.

    Returns
    -------
    Psi : float
        Global diffuseness estimate ∈ [0, 1] using the full covariance matrix.
    Psi_per_order : ndarray, shape (N,)
        Per-order diffuseness estimates (orders 1, 2, …, N).

    Notes
    -----
    A pure diffuse field at order N has R = σ² I, so all (N+1)² eigenvalues
    are equal.  The CMD diffuseness measures the eigenvalue spread::

        g(R) = (1/λ_mean) Σ |λ_i − λ_mean|

    and normalises by the maximum possible spread for (N+1)² eigenvalues::

        g_max = 2 ((N+1)² − 1)

    Diffuseness is then Ψ_CMD = 1 − g(R) / g_max.

    Examples
    --------
    >>> import numpy as np
    >>> N = 2
    >>> C = np.eye((N + 1) ** 2, dtype=complex)   # ideal diffuse covariance
    >>> Psi, Psi_per_order = diffuseness_cmd(C)
    >>> abs(Psi - 1.0) < 1e-10
    True
    """
    c = np.asarray(sh_cov, dtype=np.complex128)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("sh_cov must be a square 2-D matrix")
    n_sh = c.shape[0]
    order = int(round(np.sqrt(n_sh) - 1))
    if (order + 1) ** 2 != n_sh:
        raise ValueError(
            f"sh_cov size {n_sh} does not correspond to any integer SH order "
            f"((N+1)² must be an integer)"
        )

    def _cmd_from_cov(cov: np.ndarray, n: int) -> float:
        """CMD diffuseness for a single covariance block of order n."""
        eigvals = np.real(np.linalg.eigvalsh(cov))   # hermitian → eigvalsh
        n_eigs = (n + 1) ** 2
        mean_ev = eigvals.sum() / n_eigs
        if abs(mean_ev) <= 1e-12:
            return 1.0
        g_max = 2.0 * (n_eigs - 1)
        g = np.sum(np.abs(eigvals - mean_ev)) / mean_ev
        return float(np.clip(1.0 - g / max(g_max, 1e-12), 0.0, 1.0))

    psi_global = _cmd_from_cov(c, order)
    psi_per_order = np.zeros(order, dtype=float)
    for n in range(1, order + 1):
        c_n = c[: (n + 1) ** 2, : (n + 1) ** 2]
        psi_per_order[n - 1] = _cmd_from_cov(c_n, n)
    return psi_global, psi_per_order
