from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def intensity_vectors_from_foa(foa: ArrayLike) -> np.ndarray:
    """Compute instantaneous intensity-like vectors from FOA [T,4] as [W,X,Y,Z]."""
    a = np.asarray(foa, dtype=np.complex128)
    if a.shape[-1] < 4:
        raise ValueError("FOA array must have at least 4 channels [W,X,Y,Z]")
    w = a[..., 0]
    v = a[..., 1:4]
    return np.real(np.conj(w)[..., None] * v)


def diffuseness_ie(pv_cov: ArrayLike) -> float:
    c = np.asarray(pv_cov, dtype=np.complex128)
    if c.shape[0] < 4 or c.shape[1] < 4:
        raise ValueError("pv_cov must be at least 4x4")
    ia = np.real(c[1:4, 0])
    ia_norm = np.linalg.norm(ia)
    e = np.real(np.trace(c)) / 2.0
    if e <= 1e-12:
        return 1.0
    return float(np.clip(1.0 - ia_norm / e, 0.0, 1.0))


def diffuseness_tv(i_vecs: ArrayLike) -> float:
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError("i_vecs must be [T,3]")
    norm_i = np.linalg.norm(i, axis=1)
    mean_norm_i = float(np.mean(norm_i))
    if mean_norm_i <= 1e-12:
        return 1.0
    norm_mean_i = float(np.linalg.norm(np.mean(i, axis=0)))
    val = 1.0 - norm_mean_i / mean_norm_i
    return float(np.sqrt(np.clip(val, 0.0, 1.0)))


def diffuseness_sv(i_vecs: ArrayLike) -> float:
    i = np.asarray(i_vecs, dtype=float)
    if i.ndim != 2 or i.shape[1] != 3:
        raise ValueError("i_vecs must be [T,3]")
    mags = np.linalg.norm(i, axis=1)
    if np.all(mags <= 1e-12):
        return 1.0
    doa = i / np.maximum(mags[:, None], 1e-12)
    mean_doa = np.mean(doa, axis=0)
    return float(np.clip(1.0 - np.linalg.norm(mean_doa), 0.0, 1.0))


def diffuseness_cmd(sh_cov: ArrayLike) -> tuple[float, np.ndarray]:
    c = np.asarray(sh_cov, dtype=np.complex128)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("sh_cov must be square")
    n_sh = c.shape[0]
    order = int(round(np.sqrt(n_sh) - 1))
    if (order + 1) ** 2 != n_sh:
        raise ValueError("sh_cov size does not correspond to SH order")

    def _cmd_from_cov(cov: np.ndarray, n: int) -> float:
        eigvals = np.real(np.linalg.eigvals(cov))
        mean_ev = np.sum(eigvals) / ((n + 1) ** 2)
        if abs(mean_ev) <= 1e-12:
            return 1.0
        g0 = 2 * (((n + 1) ** 2) - 1)
        g = (1.0 / mean_ev) * np.sum(np.abs(eigvals - mean_ev))
        return float(np.clip(1.0 - g / np.maximum(g0, 1e-12), 0.0, 1.0))

    diff = _cmd_from_cov(c, order)
    diff_ord = np.zeros(order, dtype=float)
    for n in range(1, order):
        c_n = c[: (n + 1) ** 2, : (n + 1) ** 2]
        diff_ord[n - 1] = _cmd_from_cov(c_n, n)
    if order >= 1:
        diff_ord[order - 1] = diff
    return diff, diff_ord
