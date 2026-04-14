"""Tests for coordinate-system transformations."""
import numpy as np
import pytest
from spherical_array_processing.coords import (
    azel_to_az_colat, az_colat_to_azel,
    cart_to_sph, sph_to_cart, unit_sph_to_cart,
)


class TestSphToCart:
    def test_east_direction_azel(self):
        x, y, z = sph_to_cart(0.0, 0.0, 1.0, convention="az_el")
        assert np.allclose([x, y, z], [1.0, 0.0, 0.0])

    def test_north_direction_azel(self):
        x, y, z = sph_to_cart(np.pi / 2, 0.0, 1.0, convention="az_el")
        assert np.allclose([x, y, z], [0.0, 1.0, 0.0], atol=1e-14)

    def test_zenith_azel(self):
        x, y, z = sph_to_cart(0.0, np.pi / 2, 1.0, convention="az_el")
        assert np.allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-14)

    def test_equator_azcolat(self):
        # colatitude = π/2 → equator
        x, y, z = sph_to_cart(0.0, np.pi / 2, 1.0, convention="az_colat")
        assert np.allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-14)

    def test_north_pole_azcolat(self):
        x, y, z = sph_to_cart(0.0, 0.0, 1.0, convention="az_colat")
        assert np.allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-14)

    def test_radius_scaling(self):
        x, y, z = sph_to_cart(0.0, 0.0, 3.7, convention="az_el")
        assert np.allclose([x, y, z], [3.7, 0.0, 0.0], atol=1e-14)

    def test_vector_input(self):
        az = np.array([0.0, np.pi / 2])
        el = np.zeros(2)
        x, y, z = sph_to_cart(az, el, 1.0, convention="az_el")
        assert np.allclose(x, [1.0, 0.0], atol=1e-14)
        assert np.allclose(y, [0.0, 1.0], atol=1e-14)

    def test_unsupported_convention(self):
        with pytest.raises(ValueError):
            sph_to_cart(0.0, 0.0, 1.0, convention="xyz")


class TestCartToSph:
    def test_east_azel(self):
        az, el, r = cart_to_sph(1.0, 0.0, 0.0, convention="az_el")
        assert np.allclose([az, el, r], [0.0, 0.0, 1.0], atol=1e-14)

    def test_zenith_azel(self):
        az, el, r = cart_to_sph(0.0, 0.0, 1.0, convention="az_el")
        assert np.allclose([el, r], [np.pi / 2, 1.0], atol=1e-14)

    def test_north_pole_azcolat(self):
        az, th, r = cart_to_sph(0.0, 0.0, 1.0, convention="az_colat")
        assert np.allclose([th, r], [0.0, 1.0], atol=1e-14)

    def test_origin(self):
        az, el, r = cart_to_sph(0.0, 0.0, 0.0, convention="az_el")
        assert r == 0.0

    def test_unsupported_convention(self):
        with pytest.raises(ValueError):
            cart_to_sph(1.0, 0.0, 0.0, convention="rtz")


class TestRoundTrip:
    """Spherical ↔ Cartesian roundtrip for random points."""

    def test_azel_roundtrip(self, n=200):
        rng = np.random.default_rng(42)
        az = rng.uniform(0, 2 * np.pi, n)
        el = rng.uniform(-np.pi / 2, np.pi / 2, n)
        r = rng.uniform(0.5, 5.0, n)
        x, y, z = sph_to_cart(az, el, r, convention="az_el")
        az2, el2, r2 = cart_to_sph(x, y, z, convention="az_el")
        # Azimuth is only defined up to 2π
        assert np.allclose(np.mod(az, 2 * np.pi), np.mod(az2, 2 * np.pi), atol=1e-12)
        assert np.allclose(el, el2, atol=1e-12)
        assert np.allclose(r, r2, atol=1e-12)

    def test_azcolat_roundtrip(self, n=200):
        rng = np.random.default_rng(7)
        az = rng.uniform(0, 2 * np.pi, n)
        th = rng.uniform(0, np.pi, n)
        r = rng.uniform(0.5, 5.0, n)
        x, y, z = sph_to_cart(az, th, r, convention="az_colat")
        az2, th2, r2 = cart_to_sph(x, y, z, convention="az_colat")
        assert np.allclose(np.mod(az, 2 * np.pi), np.mod(az2, 2 * np.pi), atol=1e-12)
        assert np.allclose(th, th2, atol=1e-12)
        assert np.allclose(r, r2, atol=1e-12)


class TestConventionConversions:
    def test_azel_to_azcolat(self):
        az, th = azel_to_az_colat(np.array([0.3]), np.array([0.5]))
        assert np.allclose(th, np.pi / 2 - 0.5, atol=1e-14)
        assert np.allclose(az, 0.3, atol=1e-14)

    def test_inverse_roundtrip(self):
        rng = np.random.default_rng(1)
        az0 = rng.uniform(0, 2 * np.pi, 50)
        el0 = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        az1, th1 = azel_to_az_colat(az0, el0)
        az2, el2 = az_colat_to_azel(az1, th1)
        assert np.allclose(az0, az2, atol=1e-14)
        assert np.allclose(el0, el2, atol=1e-14)


class TestUnitSphToCart:
    def test_shape(self):
        n = 10
        az = np.linspace(0, 2 * np.pi, n, endpoint=False)
        el = np.zeros(n)
        xyz = unit_sph_to_cart(az, el, convention="az_el")
        assert xyz.shape == (n, 3)

    def test_unit_norm(self):
        rng = np.random.default_rng(3)
        az = rng.uniform(0, 2 * np.pi, 50)
        el = rng.uniform(-np.pi / 2, np.pi / 2, 50)
        xyz = unit_sph_to_cart(az, el, convention="az_el")
        norms = np.linalg.norm(xyz, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-14)
