"""Spatial visualisation helpers for spherical array processing.

Provides convenient plotting routines for:
- 3-D microphone array geometry
- 2-D directional map / heatmap on the sphere
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

from ..coords import sph_to_cart


def plot_mic_array(
    mic_dirs_deg: ArrayLike,
    radius_m: float,
    ax: "plt.Axes | None" = None,
    **scatter_kwargs,
) -> Axes:
    """Plot a 3-D spherical microphone array geometry.

    Parameters
    ----------
    mic_dirs_deg : array-like, shape (M, 2)
        Sensor directions in degrees as ``[azimuth_deg, elevation_deg]``.
        Uses the ``az_el`` convention.
    radius_m : float
        Array radius in metres.
    ax : mpl 3D Axes, optional
        Existing axes to draw into.  If ``None``, a new figure is created.
    **scatter_kwargs
        Extra keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    Axes
        The 3-D axes containing the plot.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from spherical_array_processing.plotting import plot_mic_array
    >>> dirs = np.column_stack([
    ...     np.linspace(0, 360, 8, endpoint=False),
    ...     np.zeros(8),
    ... ])
    >>> ax = plot_mic_array(dirs, radius_m=0.05)
    """
    dirs = np.asarray(mic_dirs_deg, dtype=float)
    if dirs.ndim != 2 or dirs.shape[1] != 2:
        raise ValueError(
            f"mic_dirs_deg must have shape (M, 2) in [az_deg, el_deg], "
            f"got {dirs.shape}"
        )
    az = np.deg2rad(dirs[:, 0])
    el = np.deg2rad(dirs[:, 1])
    x, y, z = sph_to_cart(az, el, radius_m, convention="az_el")
    if ax is None:
        fig: Figure = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    kw = {"s": 40, "zorder": 5}
    kw.update(scatter_kwargs)
    ax.scatter(x, y, z, **kw)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.25 * radius_m
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Array geometry  (r = {radius_m*100:.1f} cm,  M = {len(az)})")
    return ax


def plot_directional_map_from_grid(
    fgrid: ArrayLike,
    azi_res_deg: float,
    polar_res_deg: float,
    ax: "plt.Axes | None" = None,
    polar_or_elev: str = "elev",
    zeroed_or_centered: bool = True,
    colorbar: bool = True,
    title: str | None = None,
    **imshow_kwargs,
) -> Axes:
    """Plot a directional function on a 2-D azimuth–elevation map.

    Parameters
    ----------
    fgrid : array-like, shape (n_polar × n_azi,)
        Spatial function values on a regular polar × azimuth grid.
        Values are expected in row-major order: polar (elevation / colatitude)
        index varies first, then azimuth.
    azi_res_deg : float
        Azimuth grid resolution in degrees.
    polar_res_deg : float
        Polar (elevation or colatitude) resolution in degrees.
    ax : Axes, optional
        Existing axes.  If ``None``, a new figure is created.
    polar_or_elev : {"elev", "polar"}, default "elev"
        Whether the vertical axis represents elevation [-90°, 90°] or
        colatitude [0°, 180°].
    zeroed_or_centered : bool, default True
        If ``True``, azimuth axis is centered at 0 (front).  If ``False``,
        azimuth runs from 0° to 360°.
    colorbar : bool, default True
        Whether to add a colourbar.
    title : str, optional
        Plot title.
    **imshow_kwargs
        Extra keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    Axes
        The 2-D axes containing the plot.
    """
    vals = np.asarray(fgrid, dtype=float).reshape(-1)
    n_azi = int(round(360.0 / azi_res_deg)) + 1
    n_pol = int(round(180.0 / polar_res_deg)) + 1
    if vals.size != n_azi * n_pol:
        # Graceful fallback for unexpected grid sizes
        side = int(np.sqrt(vals.size))
        img = vals[: side * side].reshape(side, side)
        if ax is None:
            _, ax = plt.subplots()
        ax.imshow(img, origin="lower", aspect="auto")
        return ax
    img = vals.reshape(n_pol, n_azi)
    if zeroed_or_centered:
        img = np.roll(img, n_azi // 2, axis=1)
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    extent_y = (-90, 90) if polar_or_elev == "elev" else (0, 180)
    extent_x = (-180, 180) if zeroed_or_centered else (0, 360)
    kw = {"origin": "lower", "aspect": "auto", "cmap": "viridis",
          "extent": [extent_x[0], extent_x[1], extent_y[0], extent_y[1]]}
    kw.update(imshow_kwargs)
    im = ax.imshow(img, **kw)
    if colorbar:
        plt.colorbar(im, ax=ax, pad=0.02)
    ax.set_xlabel("Azimuth (°)")
    y_label = "Elevation (°)" if polar_or_elev == "elev" else "Colatitude (°)"
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    return ax
