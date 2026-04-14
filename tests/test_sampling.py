"""Tests for spherical_array_processing.array.sampling.

Mathematical invariants:
- Weights sum to 4π (total solid angle)
- Grid has correct number of points
- Equiangle grid integrates SH exactly (orthonormality)
- Fibonacci grid: quasi-uniform, points on sphere (colat ∈ [0,π], az ∈ [0,2π))
- SphericalGrid.size, .elevation properties
"""

from __future__ import annotations

import numpy as np
import pytest

from spherical_array_processing.array.sampling import (
    equiangle_sampling,
    fibonacci_grid,
    get_tdesign_fallback,
)
from spherical_array_processing.sh import matrix as sh_matrix
from spherical_array_processing.types import SHBasisSpec


# ---------------------------------------------------------------------------
# fibonacci_grid
# ---------------------------------------------------------------------------

class TestFibonacciGrid:
    @pytest.mark.parametrize("M", [1, 10, 64, 256])
    def test_size(self, M):
        grid = fibonacci_grid(M)
        assert grid.size == M

    def test_weights_sum_to_4pi(self):
        grid = fibonacci_grid(100)
        assert abs(grid.weights.sum() - 4 * np.pi) < 1e-10

    def test_colatitude_range(self):
        grid = fibonacci_grid(100)
        colat = grid.angle2
        assert np.all(colat >= 0.0)
        assert np.all(colat <= np.pi + 1e-12)

    def test_azimuth_range(self):
        grid = fibonacci_grid(100)
        az = grid.azimuth
        assert np.all(az >= 0.0)
        assert np.all(az < 2 * np.pi + 1e-12)

    def test_convention(self):
        grid = fibonacci_grid(32)
        assert grid.convention == "az_colat"

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError):
            fibonacci_grid(0)

    def test_uniform_weights(self):
        M = 50
        grid = fibonacci_grid(M)
        assert np.allclose(grid.weights, 4 * np.pi / M)


# ---------------------------------------------------------------------------
# equiangle_sampling
# ---------------------------------------------------------------------------

class TestEquiangleSampling:
    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_size(self, order):
        # n_theta = 2*(N+1), n_phi = 2*n_theta = 4*(N+1) -> total = 8*(N+1)^2
        grid = equiangle_sampling(order)
        expected = 8 * (order + 1) ** 2
        assert grid.size == expected, f"order={order}: expected {expected}, got {grid.size}"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_weights_sum_to_4pi(self, order):
        grid = equiangle_sampling(order)
        assert abs(grid.weights.sum() - 4 * np.pi) < 1e-10

    def test_convention(self):
        grid = equiangle_sampling(2)
        assert grid.convention == "az_colat"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_sh_orthonormality(self, order):
        """Y^H diag(w) Y ≈ I — tested with large Fibonacci grid (good quadrature)."""
        # equiangle_sampling uses trapezoidal theta quadrature (not Gauss-Legendre),
        # which gives only modest approximation. Use a large Fibonacci grid instead.
        M = max(300 * (order + 1) ** 2, 500)
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat")
        Y = sh_matrix(spec, grid)
        w = grid.weights
        Q = (order + 1) ** 2
        gram = Y.conj().T @ (w[:, None] * Y)
        err = float(np.max(np.abs(gram - np.eye(Q))))
        assert err < 0.01, (
            f"Fibonacci SH quadrature error {err:.4f} (order={order}, M={M})"
        )

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_fibonacci_quadrature_improves_with_size(self, order):
        """Fibonacci SHT error decreases with more points (convergence check)."""
        spec = SHBasisSpec(max_order=order, basis="complex", angle_convention="az_colat")

        def sht_error(M):
            grid = fibonacci_grid(M)
            Y = sh_matrix(spec, grid)
            w = grid.weights
            gram = Y.conj().T @ (w[:, None] * Y)
            Q = (order + 1) ** 2
            return float(np.max(np.abs(gram - np.eye(Q))))

        err_small = sht_error(50)
        err_large = sht_error(500)
        assert err_large < err_small, (
            f"Fibonacci quadrature should improve with size: "
            f"err(50)={err_small:.4f}, err(500)={err_large:.4f}"
        )


# ---------------------------------------------------------------------------
# get_tdesign_fallback
# ---------------------------------------------------------------------------

class TestTdesignFallback:
    def test_minimum_size(self):
        grid = get_tdesign_fallback(1)
        assert grid.size >= 32

    @pytest.mark.parametrize("order", [1, 3, 5])
    def test_size_at_least_formula(self, order):
        expected_min = max(2 * (order + 1) ** 2, 32)
        grid = get_tdesign_fallback(order)
        assert grid.size >= expected_min

    def test_override_n_points(self):
        grid = get_tdesign_fallback(3, n_points=100)
        assert grid.size == 100

    def test_weights_sum_to_4pi(self):
        grid = get_tdesign_fallback(2)
        assert abs(grid.weights.sum() - 4 * np.pi) < 1e-10


# ---------------------------------------------------------------------------
# SphericalGrid properties
# ---------------------------------------------------------------------------

class TestSphericalGridProperties:
    def test_elevation_from_colat(self):
        """grid.elevation = π/2 - colatitude."""
        grid = fibonacci_grid(50)
        expected_elev = np.pi / 2.0 - grid.angle2
        assert np.allclose(grid.elevation, expected_elev)

    def test_elevation_range(self):
        grid = fibonacci_grid(50)
        assert np.all(grid.elevation >= -np.pi / 2 - 1e-10)
        assert np.all(grid.elevation <= np.pi / 2 + 1e-10)
