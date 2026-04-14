from __future__ import annotations

import numpy as np

from spharray import SHBasisSpec
from spharray.array.sampling import fibonacci_grid
from spharray.sh import acn_index, matrix
from spharray.toolkit import spatial as tk_spatial

try:
    from scipy.special import sph_harm_y as _sph_harm_y

    def _ref_sph_harm(m: int, n: int, azimuth: np.ndarray, colatitude: np.ndarray) -> np.ndarray:
        return np.asarray(_sph_harm_y(n, m, colatitude, azimuth), dtype=np.complex128)

except Exception:  # pragma: no cover
    from scipy.special import sph_harm as _legacy_sph_harm

    def _ref_sph_harm(m: int, n: int, azimuth: np.ndarray, colatitude: np.ndarray) -> np.ndarray:
        return np.asarray(_legacy_sph_harm(m, n, azimuth, colatitude), dtype=np.complex128)


def _reference_complex_sh_matrix(order: int, azimuth: np.ndarray, colatitude: np.ndarray) -> np.ndarray:
    y = np.zeros((azimuth.size, (order + 1) ** 2), dtype=np.complex128)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            y[:, acn_index(n, m)] = _ref_sph_harm(m, n, azimuth, colatitude)
    return y


def test_complex_sh_matrix_matches_scipy_reference() -> None:
    order = 4
    grid = fibonacci_grid(1024)
    y_pkg = np.asarray(matrix(SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat"), grid))
    y_ref = _reference_complex_sh_matrix(order, grid.azimuth, grid.colatitude)
    rel_err = np.linalg.norm(y_pkg - y_ref) / np.maximum(np.linalg.norm(y_ref), 1e-12)
    assert rel_err < 1e-12


def test_toolkit_getsh_matches_scipy_reference() -> None:
    order = 4
    grid = fibonacci_grid(1024)
    dirs_incl = np.column_stack([grid.azimuth, grid.colatitude])
    y_pkg = np.asarray(tk_spatial.getSH(order, dirs_incl, basisType="complex"))
    y_ref = _reference_complex_sh_matrix(order, grid.azimuth, grid.colatitude)
    rel_err = np.linalg.norm(y_pkg - y_ref) / np.maximum(np.linalg.norm(y_ref), 1e-12)
    assert rel_err < 1e-12
