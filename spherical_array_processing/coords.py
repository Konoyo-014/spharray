"""Library module.

Usage:
    from spherical_array_processing.coords import <symbol>
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _a(x: ArrayLike) -> NDArray[np.float64]:
    """Usage:
        Run a.
    
    Args:
        x: ArrayLike.
    
    Returns:
        NDArray[np.float64].
    """
    return np.asarray(x, dtype=float)


def sph_to_cart(
    azimuth: ArrayLike,
    angle2: ArrayLike,
    radius: ArrayLike | float = 1.0,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Usage:
        Run sph to cart.
    
    Args:
        azimuth: ArrayLike.
        angle2: ArrayLike.
        radius: ArrayLike | float, default=1.0.
        convention: str, default='az_el'.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]].
    """
    az = _a(azimuth)
    a2 = _a(angle2)
    r = _a(radius)
    if convention == "az_el":
        el = a2
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
    if convention == "az_colat":
        th = a2
        x = r * np.sin(th) * np.cos(az)
        y = r * np.sin(th) * np.sin(az)
        z = r * np.cos(th)
        return x, y, z
    raise ValueError(f"unsupported convention: {convention}")


def cart_to_sph(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Usage:
        Run cart to sph.
    
    Args:
        x: ArrayLike.
        y: ArrayLike.
        z: ArrayLike.
        convention: str, default='az_el'.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]].
    """
    x = _a(x)
    y = _a(y)
    z = _a(z)
    r = np.sqrt(x * x + y * y + z * z)
    az = np.arctan2(y, x)
    with np.errstate(invalid="ignore", divide="ignore"):
        if convention == "az_el":
            el = np.arcsin(np.where(r == 0, 0.0, z / r))
            return az, el, r
        if convention == "az_colat":
            th = np.arccos(np.clip(np.where(r == 0, 1.0, z / r), -1.0, 1.0))
            return az, th, r
    raise ValueError(f"unsupported convention: {convention}")


def azel_to_az_colat(azimuth: ArrayLike, elevation: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Usage:
        Run azel to az colat.
    
    Args:
        azimuth: ArrayLike.
        elevation: ArrayLike.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]].
    """
    az = _a(azimuth)
    el = _a(elevation)
    return az, (np.pi / 2.0) - el


def az_colat_to_azel(azimuth: ArrayLike, colatitude: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Usage:
        Run az colat to azel.
    
    Args:
        azimuth: ArrayLike.
        colatitude: ArrayLike.
    
    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]].
    """
    az = _a(azimuth)
    th = _a(colatitude)
    return az, (np.pi / 2.0) - th


def unit_sph_to_cart(azimuth: ArrayLike, angle2: ArrayLike, convention: str = "az_el") -> NDArray[np.float64]:
    """Usage:
        Run unit sph to cart.
    
    Args:
        azimuth: ArrayLike.
        angle2: ArrayLike.
        convention: str, default='az_el'.
    
    Returns:
        NDArray[np.float64].
    """
    x, y, z = sph_to_cart(azimuth, angle2, radius=1.0, convention=convention)
    return np.stack([x, y, z], axis=-1)

