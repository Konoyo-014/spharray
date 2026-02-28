"""Library module.

Usage:
    from spherical_array_processing.acoustics.radial import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import spherical_jn, spherical_yn

from ..sh.basis import replicate_per_order


def _a(x: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run a.
    
    Args:
        x: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    return np.asarray(x, dtype=float)


def besseljs(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run besseljs.
    
    Args:
        n: int | ArrayLike.
        x: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    return spherical_jn(n, _a(x))


def besseljsd(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run besseljsd.
    
    Args:
        n: int | ArrayLike.
        x: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    return spherical_jn(n, _a(x), derivative=True)


def besselhs(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Usage:
        Run besselhs.
    
    Args:
        n: int | ArrayLike.
        x: ArrayLike.
    
    Returns:
        NDArray[np.complex128].
    """
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x) + 1j * spherical_yn(n, x)
    return np.asarray(y, dtype=np.complex128)


def besselhsd(n: int | ArrayLike, x: ArrayLike) -> NDArray[np.complex128]:
    """Usage:
        Run besselhsd.
    
    Args:
        n: int | ArrayLike.
        x: ArrayLike.
    
    Returns:
        NDArray[np.complex128].
    """
    x = _a(x)
    with np.errstate(all="ignore"):
        y = spherical_jn(n, x, derivative=True) + 1j * spherical_yn(n, x, derivative=True)
    return np.asarray(y, dtype=np.complex128)


def plane_wave_radial_bn(n: int, kr: ArrayLike, ka: ArrayLike | None = None, sphere: int | str = 1) -> NDArray[np.complex128]:
    """Harmonics-style Bn radial function.

    `sphere`: 0/'open', 1/'rigid', 2/'cardioid'
    """
    kr = _a(kr)
    if ka is None:
        ka = kr
    ka = _a(ka)
    kind = sphere
    if isinstance(kind, str):
        kind = {"open": 0, "rigid": 1, "cardioid": 2}[kind]
    j = 1j
    if kind == 0:
        return 4 * np.pi * (j ** n) * besseljs(n, kr)
    if kind == 1:
        return 4 * np.pi * (j ** n) * (
            besseljs(n, kr) - (besseljsd(n, ka) / np.conj(besselhsd(n, ka))) * np.conj(besselhs(n, kr))
        )
    if kind == 2:
        return 4 * np.pi * (j ** n) * (besseljs(n, kr) - j * besseljsd(n, kr))
    raise ValueError(f"unsupported sphere kind: {sphere}")


def bn_matrix(max_order: int, kr: ArrayLike, ka: ArrayLike | None = None, sphere: int | str = 1, repeat_per_order: bool = True) -> NDArray[np.complex128]:
    """Usage:
        Run bn matrix.
    
    Args:
        max_order: int.
        kr: ArrayLike.
        ka: ArrayLike | None, default=None.
        sphere: int | str, default=1.
        repeat_per_order: bool, default=True.
    
    Returns:
        NDArray[np.complex128].
    """
    kr_arr = _a(kr).reshape(-1)
    rows: list[NDArray[np.complex128]] = []
    for n in range(max_order + 1):
        rows.append(plane_wave_radial_bn(n, kr_arr, ka=ka, sphere=sphere))
    b = np.stack(rows, axis=-1)  # [K, N+1]
    if not repeat_per_order:
        return b
    rep = replicate_per_order(np.arange(max_order + 1))
    out = np.zeros((kr_arr.size, rep.size), dtype=np.complex128)
    cursor = 0
    for n in range(max_order + 1):
        count = 2 * n + 1
        out[:, cursor : cursor + count] = b[:, [n]]
        cursor += count
    return out


def sph_modal_coeffs(max_order: int, kR: ArrayLike, array_type: str = "rigid") -> NDArray[np.complex128]:
    """Usage:
        Run sph modal coeffs.
    
    Args:
        max_order: int.
        kR: ArrayLike.
        array_type: str, default='rigid'.
    
    Returns:
        NDArray[np.complex128].
    """
    sphere = {"open": 0, "rigid": 1, "cardioid": 2}.get(array_type, 1)
    return bn_matrix(max_order, kR, ka=kR, sphere=sphere, repeat_per_order=False)
