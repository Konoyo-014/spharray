"""Coordinate-system transformations for spherical array processing.

All angles are in **radians**.

Supported conventions
---------------------
``az_el`` (azimuth–elevation)
    * Azimuth φ ∈ [0, 2π) — counter-clockwise from the +x axis.
    * Elevation θ_e ∈ [−π/2, π/2] — upward from the horizontal plane.
    * Cartesian: x = r cos θ_e cos φ,  y = r cos θ_e sin φ,  z = r sin θ_e.

``az_colat`` (azimuth–colatitude)
    * Azimuth φ ∈ [0, 2π) — same as above.
    * Colatitude θ ∈ [0, π] — downward from the +z pole (θ = π/2 − θ_e).
    * Cartesian: x = r sin θ cos φ,  y = r sin θ sin φ,  z = r cos θ.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _a(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=float)


def sph_to_cart(
    azimuth: ArrayLike,
    angle2: ArrayLike,
    radius: ArrayLike | float = 1.0,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    azimuth : array-like
        Azimuth angle(s) in radians.
    angle2 : array-like
        Elevation (``"az_el"``) or colatitude (``"az_colat"``) in radians.
    radius : array-like or float, default 1.0
        Radial distance(s).  Broadcast-compatible with ``azimuth``.
    convention : {"az_el", "az_colat"}, default "az_el"
        Angle convention for ``angle2``.

    Returns
    -------
    x, y, z : ndarray
        Cartesian coordinates with the same shape as the broadcasted inputs.

    Examples
    --------
    >>> import numpy as np
    >>> x, y, z = sph_to_cart(0.0, np.pi / 2, convention="az_colat")
    >>> np.allclose([x, y, z], [1.0, 0.0, 0.0])
    True
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
    raise ValueError(f"unsupported convention: {convention!r}")


def cart_to_sph(
    x: ArrayLike,
    y: ArrayLike,
    z: ArrayLike,
    convention: str = "az_el",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x, y, z : array-like
        Cartesian coordinates.
    convention : {"az_el", "az_colat"}, default "az_el"
        Output angle convention.

    Returns
    -------
    azimuth : ndarray
        Azimuth in radians ∈ (-π, π].
    angle2 : ndarray
        Elevation (``"az_el"``) or colatitude (``"az_colat"``) in radians.
    radius : ndarray
        Radial distance r = √(x² + y² + z²).

    Notes
    -----
    At the origin (r = 0) the azimuth is set to 0 and the elevation / colatitude
    defaults to 0 / (π/2) respectively.

    Examples
    --------
    >>> import numpy as np
    >>> az, el, r = cart_to_sph(1.0, 0.0, 0.0, convention="az_el")
    >>> np.allclose([az, el, r], [0.0, 0.0, 1.0])
    True
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
    raise ValueError(f"unsupported convention: {convention!r}")


def azel_to_az_colat(
    azimuth: ArrayLike,
    elevation: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert from azimuth–elevation to azimuth–colatitude.

    Parameters
    ----------
    azimuth : array-like
        Azimuth in radians (unchanged).
    elevation : array-like
        Elevation in radians ∈ [-π/2, π/2].

    Returns
    -------
    azimuth : ndarray
        Same as input.
    colatitude : ndarray
        Colatitude = π/2 − elevation, in radians ∈ [0, π].
    """
    az = _a(azimuth)
    el = _a(elevation)
    return az, (np.pi / 2.0) - el


def az_colat_to_azel(
    azimuth: ArrayLike,
    colatitude: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert from azimuth–colatitude to azimuth–elevation.

    Parameters
    ----------
    azimuth : array-like
        Azimuth in radians (unchanged).
    colatitude : array-like
        Colatitude in radians ∈ [0, π].

    Returns
    -------
    azimuth : ndarray
        Same as input.
    elevation : ndarray
        Elevation = π/2 − colatitude, in radians ∈ [-π/2, π/2].
    """
    az = _a(azimuth)
    th = _a(colatitude)
    return az, (np.pi / 2.0) - th



def angular_distance(
    azimuth1,
    angle2_1,
    azimuth2,
    angle2_2,
    convention="az_el",
):
    """Great-circle angular distance between two directions on the unit sphere.

    Uses the numerically stable haversine formula.

    Parameters
    ----------
    azimuth1, azimuth2 : array-like
        Azimuth angles in radians.
    angle2_1, angle2_2 : array-like
        Elevation (az_el) or colatitude (az_colat) in radians.
    convention : {"az_el", "az_colat"}, default "az_el"
        Interpretation of the angle2_* arguments.

    Returns
    -------
    dist : ndarray, float64
        Angular separation in radians, in [0, pi].

    Examples
    --------
    >>> import numpy as np
    >>> angular_distance(0.0, 0.0, np.pi, 0.0, convention="az_el")
    array(3.14159265)
    """
    import numpy as _np
    az1 = _np.asarray(azimuth1, dtype=float)
    az2 = _np.asarray(azimuth2, dtype=float)
    if convention == "az_el":
        lat1 = _np.asarray(angle2_1, dtype=float)
        lat2 = _np.asarray(angle2_2, dtype=float)
    elif convention == "az_colat":
        lat1 = _np.pi / 2.0 - _np.asarray(angle2_1, dtype=float)
        lat2 = _np.pi / 2.0 - _np.asarray(angle2_2, dtype=float)
    else:
        raise ValueError(f"unsupported convention: {convention!r}")
    dlat = lat2 - lat1
    dlon = az2 - az1
    a = (_np.sin(dlat / 2.0) ** 2
         + _np.cos(lat1) * _np.cos(lat2) * _np.sin(dlon / 2.0) ** 2)
    return 2.0 * _np.arcsin(_np.sqrt(_np.clip(a, 0.0, 1.0)))


def angular_distance_deg(
    azimuth1,
    angle2_1,
    azimuth2,
    angle2_2,
    convention="az_el",
):
    """Great-circle angular distance in **degrees**.

    Convenience wrapper around angular_distance.

    Examples
    --------
    >>> angular_distance_deg(0.0, 0.0, 0.0, 0.0)
    array(0.)
    """
    import numpy as _np
    return _np.degrees(
        angular_distance(azimuth1, angle2_1, azimuth2, angle2_2, convention)
    )


def unit_sph_to_cart(
    azimuth: ArrayLike,
    angle2: ArrayLike,
    convention: str = "az_el",
) -> NDArray[np.float64]:
    """Convert unit-sphere directions to Cartesian unit vectors.

    Parameters
    ----------
    azimuth : array-like, shape (M,)
        Azimuth angles in radians.
    angle2 : array-like, shape (M,)
        Elevation or colatitude in radians.
    convention : {"az_el", "az_colat"}, default "az_el"
        Angle convention.

    Returns
    -------
    xyz : ndarray, shape (M, 3)
        Unit vectors as rows ``[x, y, z]``.

    Examples
    --------
    >>> import numpy as np
    >>> xyz = unit_sph_to_cart(0.0, 0.0, convention="az_el")
    >>> np.allclose(xyz, [[1.0, 0.0, 0.0]])
    True
    """
    x, y, z = sph_to_cart(azimuth, angle2, radius=1.0, convention=convention)
    return np.stack([x, y, z], axis=-1)
