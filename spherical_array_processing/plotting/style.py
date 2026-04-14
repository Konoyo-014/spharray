"""Matplotlib style utilities for publication-quality figures.

Provides a MATLAB-like default style and a context manager for temporary
style changes without permanently modifying global rcParams.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import matplotlib as mpl

from ..types import FigureStyleConfig


def apply_matlab_like_style(config: FigureStyleConfig | None = None) -> None:
    """Apply a clean, MATLAB-like matplotlib style globally.

    Parameters
    ----------
    config : FigureStyleConfig, optional
        Style configuration.  Uses defaults if ``None``.

    Notes
    -----
    This modifies global ``mpl.rcParams``.  To apply the style only within
    a code block, use :func:`figure_style_context` instead.

    Examples
    --------
    >>> from spherical_array_processing.plotting import apply_matlab_like_style
    >>> apply_matlab_like_style()
    """
    cfg = config or FigureStyleConfig()
    mpl.rcParams.update(
        {
            "figure.dpi": cfg.dpi,
            "figure.figsize": list(cfg.figsize),
            "font.family": cfg.font_family,
            "font.size": cfg.font_size,
            "axes.titlesize": cfg.font_size + 1,
            "axes.labelsize": cfg.font_size,
            "xtick.labelsize": cfg.font_size - 1,
            "ytick.labelsize": cfg.font_size - 1,
            "legend.fontsize": cfg.font_size - 1,
            "lines.linewidth": cfg.line_width,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": cfg.dpi,
            "savefig.bbox": "tight",
        }
    )


@contextmanager
def figure_style_context(
    config: FigureStyleConfig | None = None,
) -> Generator[None, None, None]:
    """Context manager: apply MATLAB-like style, restore on exit.

    Parameters
    ----------
    config : FigureStyleConfig, optional
        Style configuration.  Uses defaults if ``None``.

    Yields
    ------
    None

    Examples
    --------
    >>> from spherical_array_processing.plotting import figure_style_context
    >>> with figure_style_context():
    ...     pass  # create figures here
    """
    old = mpl.rcParams.copy()
    apply_matlab_like_style(config)
    try:
        yield
    finally:
        mpl.rcParams.update(old)
