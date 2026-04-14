"""Tests for spherical-harmonic basis and transforms.

Mathematical properties verified:
- Orthonormality: Y^H diag(w) Y ≈ I (complex/real SH, large Fibonacci grid)
- SH addition theorem: Sigma_m Y_n^m(a) Y_n^m*(b) = (2n+1)/(4pi) P_n(cos gamma)
- ACN index correctness
- Roundtrip SHT: iSHT(SHT(f)) ≈ f for bandlimited signals
- real→complex→real coefficient roundtrip (exact)
- complex→real→complex roundtrip for conjugate-symmetric inputs (exact)

Note on equiangle quadrature:
  equiangle_sampling uses trapezoidal theta quadrature (NOT Gauss-Legendre),
  so exact SH orthonormality is not guaranteed.  Orthonormality tests use a
  large Fibonacci grid where the approximation error is < 1%.
"""
import numpy as np
import pytest
from scipy.special import eval_legendre

from spharray.sh import (
    acn_index, complex_matrix, direct_sht, inverse_sht,
    matrix, real_matrix,
    complex_to_real_coeffs, real_to_complex_coeffs,
    replicate_per_order,
)
from spharray.sh.basis import acn_index as acn
from spharray.array.sampling import equiangle_sampling, fibonacci_grid
from spharray.types import SHBasisSpec, SphericalGrid


# ── ACN index ────────────────────────────────────────────────────────────────

class TestACNIndex:
    def test_order0(self):
        assert acn_index(0, 0) == 0

    def test_order1(self):
        assert acn_index(1, -1) == 1
        assert acn_index(1,  0) == 2
        assert acn_index(1,  1) == 3

    def test_order2(self):
        assert acn_index(2, -2) == 4
        assert acn_index(2,  0) == 6
        assert acn_index(2,  2) == 8

    def test_formula(self):
        for n in range(6):
            for m in range(-n, n + 1):
                assert acn_index(n, m) == n * (n + 1) + m


# ── Orthonormality ────────────────────────────────────────────────────────────

class TestOrthonormality:
    """Y^H diag(w) Y ≈ I.

    Uses a large Fibonacci grid (M >> (N+1)^2) where the quadrature
    approximation error is < 1%.  equiangle_sampling is NOT used here because
    its trapezoidal theta rule does not give exact quadrature.
    """

    @pytest.mark.parametrize("N", [1, 2, 3])
    def test_complex_sh_orthonormal(self, N):
        M = max(300 * (N + 1) ** 2, 500)
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="complex",
                           normalization="orthonormal", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        gram = Y.conj().T @ (grid.weights[:, None] * Y)   # Y^H W Y
        assert Y.shape == (M, (N + 1) ** 2)
        err = float(np.max(np.abs(gram - np.eye((N + 1) ** 2))))
        assert err < 0.01, f"Orthonormality error {err:.4f} for order {N}"

    @pytest.mark.parametrize("N", [1, 2])
    def test_real_sh_orthonormal(self, N):
        M = max(300 * (N + 1) ** 2, 500)
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="real",
                           normalization="orthonormal", angle_convention="az_colat")
        Y = real_matrix(spec, grid)
        gram = Y.T @ (grid.weights[:, None] * Y)
        err = float(np.max(np.abs(gram - np.eye((N + 1) ** 2))))
        assert err < 0.01, f"Real SH orthonormality error {err:.4f} for order {N}"

    def test_sn3d_normalization(self):
        """SN3D diagonal should be 4π/(2n+1) per degree."""
        N = 2
        M = 3000
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="complex",
                           normalization="sn3d", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        gram = Y.conj().T @ (grid.weights[:, None] * Y)
        expected_diag = np.array([4 * np.pi / (2 * n + 1)
                                   for n in range(N + 1)
                                   for _ in range(2 * n + 1)])
        assert np.allclose(np.diag(gram).real, expected_diag, rtol=0.01)

    def test_n3d_normalization(self):
        """N3D diagonal should be 4π for all coefficients."""
        N = 2
        M = 3000
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="complex",
                           normalization="n3d", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        gram = Y.conj().T @ (grid.weights[:, None] * Y)
        expected_diag = np.full((N + 1) ** 2, 4 * np.pi)
        assert np.allclose(np.diag(gram).real, expected_diag, rtol=0.01)


# ── Addition theorem ──────────────────────────────────────────────────────────

class TestAdditionTheorem:
    """Verify the SH addition theorem:
    sum_m Y_n^m(a) * Y_n^m*(b) = (2n+1)/(4pi) * P_n(cos_gamma)
    """

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_addition_theorem(self, n):
        rng = np.random.default_rng(n)
        az_a, th_a = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
        az_b, th_b = rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)

        spec = SHBasisSpec(max_order=n, basis="complex",
                           normalization="orthonormal", angle_convention="az_colat")
        grid_a = SphericalGrid(azimuth=np.array([az_a]), angle2=np.array([th_a]),
                               convention="az_colat")
        grid_b = SphericalGrid(azimuth=np.array([az_b]), angle2=np.array([th_b]),
                               convention="az_colat")

        Y_a = complex_matrix(spec, grid_a)[0]
        Y_b = complex_matrix(spec, grid_b)[0]

        idxs = [acn_index(n, m) for m in range(-n, n + 1)]
        lhs = float(np.sum(Y_a[idxs] * np.conj(Y_b[idxs])).real)

        xa = np.sin(th_a) * np.cos(az_a)
        ya = np.sin(th_a) * np.sin(az_a)
        za = np.cos(th_a)
        xb = np.sin(th_b) * np.cos(az_b)
        yb = np.sin(th_b) * np.sin(az_b)
        zb = np.cos(th_b)
        cos_gamma = np.clip(xa * xb + ya * yb + za * zb, -1.0, 1.0)
        rhs = float((2 * n + 1) / (4 * np.pi) * eval_legendre(n, cos_gamma))

        assert abs(lhs - rhs) < 1e-10, f"Addition theorem failed for n={n}"


# ── SHT roundtrip ─────────────────────────────────────────────────────────────

class TestSHTRoundtrip:
    def test_bandlimited_function_fibonacci(self):
        """Bandlimited function: SHT→iSHT roundtrip with large Fibonacci grid."""
        N = 3
        M = 3000
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        rng = np.random.default_rng(0)
        f_coeffs = rng.standard_normal(spec.n_coeffs) + 1j * rng.standard_normal(spec.n_coeffs)
        f_samples = Y @ f_coeffs
        f_coeffs_rec = direct_sht(f_samples, Y, grid)
        f_rec = inverse_sht(f_coeffs_rec, Y)
        rel_err = float(np.linalg.norm(f_samples - f_rec) / np.linalg.norm(f_samples))
        assert rel_err < 0.01, f"SHT roundtrip rel error {rel_err:.4f}"

    def test_direct_sht_single_coeff_large_grid(self):
        """direct_sht approximately recovers a single SH coefficient."""
        N = 2
        M = 3000
        grid = fibonacci_grid(M)
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y = complex_matrix(spec, grid)
        idx = acn_index(2, 1)
        f_samples = Y[:, idx]
        coeffs = direct_sht(f_samples, Y, grid)
        assert abs(coeffs[idx]) > 0.99, (
            f"Expected |coeff[{idx}]| > 0.99, got {abs(coeffs[idx]):.4f}"
        )
        mask = np.ones(spec.n_coeffs, dtype=bool)
        mask[idx] = False
        assert np.max(np.abs(coeffs[mask])) < 0.01

    def test_inverse_sht_shape(self):
        N = 2
        spec = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        grid = fibonacci_grid(50)
        Y = matrix(spec, grid)
        coeffs = np.zeros(spec.n_coeffs, dtype=complex)
        coeffs[0] = 1.0
        f = inverse_sht(coeffs, Y)
        assert f.shape == (50,)


# ── Coefficient conversions ───────────────────────────────────────────────────

class TestCoeffConversions:
    def _make_conjugate_symmetric(self, N, rng):
        """Create complex SH coefficients for a real-valued function.

        For a real function f(Ω): c_{n,-m} = (-1)^m conj(c_{n,m}).
        The complex→real→complex roundtrip is exact only for such inputs.
        """
        Q = (N + 1) ** 2
        c = np.zeros(Q, dtype=complex)
        for n in range(N + 1):
            c[acn(n, 0)] = rng.standard_normal()  # m=0: must be real
        for n in range(N + 1):
            for m in range(1, n + 1):
                val = rng.standard_normal() + 1j * rng.standard_normal()
                c[acn(n, m)] = val
                c[acn(n, -m)] = (-1) ** m * np.conj(val)
        return c

    def test_real_to_complex_roundtrip(self):
        """real→complex→real is always a valid roundtrip."""
        N = 3
        rng = np.random.default_rng(99)
        r = rng.standard_normal((N + 1) ** 2)
        c = real_to_complex_coeffs(r, N)
        r_rec = complex_to_real_coeffs(c, N)
        assert np.allclose(r, r_rec, atol=1e-12)

    def test_complex_to_real_roundtrip_conjugate_symmetric(self):
        """complex→real→complex roundtrip for conjugate-symmetric inputs."""
        N = 3
        rng = np.random.default_rng(42)
        c = self._make_conjugate_symmetric(N, rng)
        r = complex_to_real_coeffs(c, N)
        c_rec = real_to_complex_coeffs(r, N)
        assert np.allclose(c, c_rec, atol=1e-12), (
            f"max err = {np.max(np.abs(c - c_rec)):.2e}"
        )

    def test_real_coeffs_are_real(self):
        N = 2
        rng = np.random.default_rng(5)
        c = rng.standard_normal((N + 1) ** 2) + 1j * rng.standard_normal((N + 1) ** 2)
        r = complex_to_real_coeffs(c, N)
        assert np.isrealobj(r) or np.allclose(r.imag, 0)

    def test_batch_dimension(self):
        """Batch processing: real→complex→real across batches."""
        N = 2
        rng = np.random.default_rng(11)
        r = rng.standard_normal((5, (N + 1) ** 2))
        c = real_to_complex_coeffs(r, N, axis=-1)
        r_rec = complex_to_real_coeffs(c, N, axis=-1)
        assert r_rec.shape == r.shape
        assert np.allclose(r, r_rec, atol=1e-12)

    def test_mismatched_axis_raises(self):
        with pytest.raises(ValueError):
            complex_to_real_coeffs(np.zeros(10, dtype=complex), max_order=3)


# ── Replicate per order ───────────────────────────────────────────────────────

class TestReplicatePerOrder:
    def test_order1(self):
        result = replicate_per_order([1.0, 2.0])
        assert np.allclose(result, [1.0, 2.0, 2.0, 2.0])

    def test_order0(self):
        result = replicate_per_order([5.0])
        assert np.allclose(result, [5.0])

    def test_length(self):
        N = 4
        result = replicate_per_order(np.arange(N + 1, dtype=float))
        assert len(result) == (N + 1) ** 2


# ── Matrix dispatch ────────────────────────────────────────────────────────────

class TestMatrixDispatch:
    def test_complex_spec(self):
        spec = SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")
        grid = fibonacci_grid(20)
        Y = matrix(spec, grid)
        assert np.iscomplexobj(Y)
        assert Y.shape == (20, 9)

    def test_real_spec(self):
        spec = SHBasisSpec(max_order=2, basis="real", angle_convention="az_colat")
        grid = fibonacci_grid(20)
        Y = matrix(spec, grid)
        assert np.isrealobj(Y)
        assert Y.shape == (20, 9)

    def test_unsupported_basis_raises(self):
        spec = SHBasisSpec(max_order=1, basis="complex", angle_convention="az_colat")
        spec.basis = "unknown"       # type: ignore
        grid = fibonacci_grid(5)
        with pytest.raises(ValueError):
            matrix(spec, grid)


# ── Convention compatibility ───────────────────────────────────────────────────

class TestConventionCompatibility:
    def test_azel_vs_azcolat_same_result(self):
        """az_el and az_colat grids at same physical locations give same Y."""
        N = 2
        rng = np.random.default_rng(77)
        az = rng.uniform(0, 2 * np.pi, 10)
        el = rng.uniform(-np.pi / 2, np.pi / 2, 10)
        grid_el = SphericalGrid(azimuth=az, angle2=el, convention="az_el")
        grid_co = SphericalGrid(azimuth=az, angle2=np.pi / 2 - el, convention="az_colat")
        spec_el = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_el")
        spec_co = SHBasisSpec(max_order=N, basis="complex", angle_convention="az_colat")
        Y_el = complex_matrix(spec_el, grid_el)
        Y_co = complex_matrix(spec_co, grid_co)
        assert np.allclose(Y_el, Y_co, atol=1e-12)
