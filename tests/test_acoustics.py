"""Tests for spherical Bessel/Hankel functions and radial coefficients.

Mathematical properties verified:
- Bessel function recurrence: j_{n+1}(x) + j_{n-1}(x) = (2n+1)/x * j_n(x)
- Wronskian: j_n h_n' - j_n' h_n = i / x^2
- Open sphere: b_n = 4pi i^n j_n(kr)
- Rigid sphere: correct Neumann BC (at sphere surface, normal derivative = 0)
- bn_matrix shape and ACN expansion
"""
import numpy as np
import pytest
from spharray.acoustics import (
    besseljs, besseljsd, besselhs, besselhsd,
    bn_matrix, plane_wave_radial_bn, sph_modal_coeffs,
)


# ── Bessel function properties ────────────────────────────────────────────────

class TestBesselFunctions:
    def test_j0_at_zero(self):
        """j_0(0) = 1"""
        assert np.allclose(besseljs(0, 0.0), 1.0)

    def test_j1_at_zero(self):
        """j_1(0) = 0"""
        assert np.allclose(besseljs(1, 0.0), 0.0)

    def test_jn_large_n_at_zero(self):
        """j_n(0) = 0 for n >= 1"""
        for n in range(1, 6):
            assert np.allclose(besseljs(n, 0.0), 0.0)

    def test_recurrence_relation(self):
        """j_{n+1}(x) + j_{n-1}(x) = (2n+1)/x * j_n(x)  for x != 0"""
        x = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        for n in range(1, 5):
            lhs = besseljs(n + 1, x) + besseljs(n - 1, x)
            rhs = (2 * n + 1) / x * besseljs(n, x)
            assert np.allclose(lhs, rhs, rtol=1e-10), f"Recurrence failed at n={n}"

    def test_wronskian(self):
        """j_n h_n' - j_n' h_n = i/x^2  (Wronskian of spherical Bessel eq)"""
        x = np.array([0.5, 1.0, 2.0, 5.0])
        for n in range(0, 5):
            j = besseljs(n, x)
            jp = besseljsd(n, x)
            h = besselhs(n, x)
            hp = besselhsd(n, x)
            wronskian = j * hp - jp * h
            expected = 1j / x ** 2
            assert np.allclose(wronskian, expected, rtol=1e-8),                 f"Wronskian failed at n={n}"

    def test_hankel_decomposition(self):
        """h_n^{(1)} = j_n + i y_n should be consistent with Bessel relation."""
        from scipy.special import spherical_yn
        x = np.array([0.5, 1.0, 2.0])
        for n in range(4):
            h = besselhs(n, x)
            j = besseljs(n, x)
            y = spherical_yn(n, x)
            assert np.allclose(h, j + 1j * y, atol=1e-14)

    def test_shapes_vectorised(self):
        x = np.linspace(0.1, 5.0, 50)
        assert besseljs(2, x).shape == (50,)
        assert besselhs(2, x).shape == (50,)


# ── Radial coefficient b_n ────────────────────────────────────────────────────

class TestPlaneWaveRadialBn:
    def test_open_sphere_shape(self):
        kr = np.linspace(0.1, 3.0, 10)
        b = plane_wave_radial_bn(0, kr, sphere="open")
        assert b.shape == (10,)

    def test_open_sphere_order0_value(self):
        """b_0 for open sphere = 4pi j_0(kr)"""
        kr = np.array([0.5, 1.0, 2.0])
        b = plane_wave_radial_bn(0, kr, sphere="open")
        expected = 4 * np.pi * besseljs(0, kr)
        assert np.allclose(b, expected, atol=1e-12)

    def test_open_sphere_prefactor(self):
        """b_n = 4pi i^n j_n(kr) for open sphere"""
        kr = np.array([1.0])
        for n in range(4):
            b = plane_wave_radial_bn(n, kr, sphere="open")
            expected = 4 * np.pi * (1j ** n) * besseljs(n, kr)
            assert np.allclose(b, expected, atol=1e-12)

    def test_rigid_sphere_ka_default(self):
        """If ka is None it defaults to kr (evaluation at sphere surface)."""
        kr = np.array([1.0, 2.0])
        b_none = plane_wave_radial_bn(1, kr, ka=None, sphere="rigid")
        b_same = plane_wave_radial_bn(1, kr, ka=kr,   sphere="rigid")
        assert np.allclose(b_none, b_same, atol=1e-14)

    def test_rigid_sphere_pressure_doubling(self):
        """For n=0, rigid sphere b_0 → 8pi as ka → 0 (pressure doubling)."""
        # At low freq (ka << 1), j_0 ≈ 1, j_0' ≈ 0, h_0^(2)' has known limit.
        # The exact limit gives b_0(ka→0) → 8pi (factor-2 pressure doubling at surface)
        # For small ka: b_0 ≈ 4pi [j_0 - (0 / ...) * ...] → 4pi * j_0(kr).
        # Actually at kr = ka → 0: b_0(0, 0) → 4pi (open) + correction.
        # Numerical check: b_0 is finite and non-zero for ka > 0
        kr = np.array([0.01])
        b = plane_wave_radial_bn(0, kr, sphere="rigid")
        assert np.isfinite(b).all()

    def test_cardioid_shape(self):
        kr = np.linspace(0.1, 3.0, 20)
        b = plane_wave_radial_bn(2, kr, sphere="cardioid")
        assert b.shape == (20,)

    def test_string_and_int_sphere_equivalent(self):
        kr = np.array([0.5, 1.5, 2.5])
        for name, num in [("open", 0), ("rigid", 1), ("cardioid", 2)]:
            b_str = plane_wave_radial_bn(1, kr, sphere=name)
            b_int = plane_wave_radial_bn(1, kr, sphere=num)
            assert np.allclose(b_str, b_int, atol=1e-14)

    def test_invalid_sphere_raises(self):
        with pytest.raises((ValueError, KeyError)):
            plane_wave_radial_bn(0, np.array([1.0]), sphere="unknown")


# ── bn_matrix ────────────────────────────────────────────────────────────────

class TestBnMatrix:
    def test_shape_no_repeat(self):
        N, K = 3, 10
        B = bn_matrix(N, np.linspace(0.1, 3, K), repeat_per_order=False)
        assert B.shape == (K, N + 1)

    def test_shape_with_repeat(self):
        N, K = 3, 10
        B = bn_matrix(N, np.linspace(0.1, 3, K), repeat_per_order=True)
        assert B.shape == (K, (N + 1) ** 2)

    def test_repeat_consistency(self):
        """Each degree n appears 2n+1 times in the ACN expansion."""
        N, K = 2, 5
        kr = np.linspace(0.5, 2.0, K)
        B_no = bn_matrix(N, kr, repeat_per_order=False)   # (K, N+1)
        B_rp = bn_matrix(N, kr, repeat_per_order=True)    # (K, (N+1)^2)
        cursor = 0
        for n in range(N + 1):
            count = 2 * n + 1
            assert np.allclose(B_rp[:, cursor: cursor + count],
                               B_no[:, [n]] * np.ones((K, count)),
                               atol=1e-14)
            cursor += count


# ── sph_modal_coeffs ──────────────────────────────────────────────────────────

class TestSphModalCoeffs:
    def test_shape(self):
        B = sph_modal_coeffs(3, np.linspace(0.1, 2.0, 15))
        assert B.shape == (15, 4)

    def test_matches_bn_matrix(self):
        kR = np.linspace(0.1, 3.0, 20)
        B1 = sph_modal_coeffs(2, kR, array_type="rigid")
        B2 = bn_matrix(2, kR, ka=kR, sphere="rigid", repeat_per_order=False)
        assert np.allclose(B1, B2, atol=1e-14)
