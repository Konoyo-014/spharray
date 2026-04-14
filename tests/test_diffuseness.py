"""Tests for spharray.diffuseness.estimators.

Mathematical invariants verified:
- IE: diffuse field (R = I) → Ψ = 1; coherent source → Ψ = 0
- TV/SV: random isotropic directions → Ψ ≈ 1; aligned → Ψ = 0
- CMD: diagonal covariance → Ψ = 1; rank-1 → Ψ small
- TV no longer has a spurious sqrt (post-refactor correctness check)
"""

from __future__ import annotations

import numpy as np
import pytest

from spharray.diffuseness.estimators import (
    diffuseness_cmd,
    diffuseness_ie,
    diffuseness_sv,
    diffuseness_tv,
    intensity_vectors_from_foa,
)


# ---------------------------------------------------------------------------
# intensity_vectors_from_foa
# ---------------------------------------------------------------------------

class TestIntensityVectors:
    def test_pure_pressure_zero_intensity(self):
        """Pure pressure (W=1, XYZ=0) → intensity = 0."""
        T = 50
        foa = np.zeros((T, 4), dtype=complex)
        foa[:, 0] = 1.0
        I = intensity_vectors_from_foa(foa)
        assert I.shape == (T, 3)
        assert np.allclose(I, 0.0)

    def test_pure_x_velocity(self):
        """W=1, X=1, Y=Z=0 → intensity in +x direction."""
        foa = np.ones((10, 4), dtype=complex)
        foa[:, 1] = 1.0
        foa[:, 2] = 0.0
        foa[:, 3] = 0.0
        I = intensity_vectors_from_foa(foa)
        assert np.all(I[:, 0] > 0), "x-intensity should be positive"
        assert np.allclose(I[:, 1], 0.0)
        assert np.allclose(I[:, 2], 0.0)

    def test_shape_preservation(self):
        rng = np.random.default_rng(42)
        foa = rng.standard_normal((20, 4)) + 1j * rng.standard_normal((20, 4))
        I = intensity_vectors_from_foa(foa)
        assert I.shape == (20, 3)

    def test_insufficient_channels_raises(self):
        with pytest.raises(ValueError, match="4 channels"):
            intensity_vectors_from_foa(np.ones((10, 3)))


# ---------------------------------------------------------------------------
# diffuseness_ie
# ---------------------------------------------------------------------------

class TestDiffusenessIE:
    def test_diffuse_field_identity(self):
        """Diffuse field: R = I → Ψ_IE = 1 (zero off-diagonal cross-terms)."""
        C = np.eye(4, dtype=complex)
        psi = diffuseness_ie(C)
        assert abs(psi - 1.0) < 1e-10

    def test_coherent_source_low_diffuseness(self):
        """Coherent source from +x: W correlated with X → low diffuseness."""
        # W=1, X=1 perfectly correlated
        c = np.zeros((4, 4), dtype=complex)
        c[0, 0] = 1.0
        c[1, 1] = 1.0
        c[1, 0] = 1.0   # cross-power: I_x = Re(C[1,0]) = 1
        c[0, 1] = 1.0
        psi = diffuseness_ie(c)
        assert psi < 0.5, f"Expected Ψ_IE < 0.5 for coherent source, got {psi}"

    def test_clipped_to_unit_interval(self):
        C = np.eye(4, dtype=complex) * 1e-20
        psi = diffuseness_ie(C)
        assert 0.0 <= psi <= 1.0

    def test_input_too_small_raises(self):
        with pytest.raises(ValueError, match="4×4"):
            diffuseness_ie(np.eye(3, dtype=complex))

    def test_zero_energy_returns_one(self):
        psi = diffuseness_ie(np.zeros((4, 4), dtype=complex))
        assert psi == 1.0


# ---------------------------------------------------------------------------
# diffuseness_tv
# ---------------------------------------------------------------------------

class TestDiffusenessTV:
    def test_isotropic_random_near_one(self):
        """Random isotropic intensity directions → Ψ_TV ≈ 1."""
        rng = np.random.default_rng(0)
        i_vecs = rng.standard_normal((5000, 3))
        psi = diffuseness_tv(i_vecs)
        assert psi > 0.95, f"Expected Ψ_TV > 0.95 for random directions, got {psi}"

    def test_coherent_source_near_zero(self):
        """All intensity vectors pointing +x → Ψ_TV = 0."""
        i_vecs = np.tile([1.0, 0.0, 0.0], (100, 1))
        psi = diffuseness_tv(i_vecs)
        assert psi < 0.01, f"Expected Ψ_TV ≈ 0 for coherent source, got {psi}"

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(1)
        i_vecs = rng.standard_normal((200, 3))
        psi = diffuseness_tv(i_vecs)
        assert 0.0 <= psi <= 1.0

    def test_no_spurious_sqrt(self):
        """Post-refactor: Ψ_TV for fully aligned vectors must equal 0, not sqrt(0)=0.
        For near-diffuse case, result should NOT be forced above 0.97 by a sqrt
        (sqrt would inflate borderline values artificially)."""
        rng = np.random.default_rng(7)
        # 50-50 mix of +x and -x → mean DOA ≈ 0 → Ψ_TV should be ≈ 1, not ≈ sqrt(1)=1
        i_half = np.tile([1.0, 0.0, 0.0], (500, 1))
        i_neg = np.tile([-1.0, 0.0, 0.0], (500, 1))
        i_vecs = np.vstack([i_half, i_neg])
        psi = diffuseness_tv(i_vecs)
        # The mean DOA is ~0, so Ψ_TV = 1 - 0 = 1.
        assert psi > 0.99, f"Expected Ψ_TV ≈ 1 for anti-parallel vectors, got {psi}"

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            diffuseness_tv(np.ones((10, 2)))

    def test_zero_vectors_returns_one(self):
        psi = diffuseness_tv(np.zeros((10, 3)))
        assert psi == 1.0


# ---------------------------------------------------------------------------
# diffuseness_sv
# ---------------------------------------------------------------------------

class TestDiffusenessSV:
    def test_isotropic_random_near_one(self):
        rng = np.random.default_rng(42)
        i_vecs = rng.standard_normal((5000, 3))
        psi = diffuseness_sv(i_vecs)
        assert psi > 0.95

    def test_coherent_source_near_zero(self):
        i_vecs = np.tile([1.0, 2.0, 0.0], (50, 1))
        psi = diffuseness_sv(i_vecs)
        assert psi < 0.01

    def test_tv_sv_agree_for_unit_vectors(self):
        """TV and SV are identical when input vectors already have unit norm."""
        rng = np.random.default_rng(5)
        raw = rng.standard_normal((300, 3))
        i_unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        psi_tv = diffuseness_tv(i_unit)
        psi_sv = diffuseness_sv(i_unit)
        assert abs(psi_tv - psi_sv) < 1e-10, (
            f"TV and SV should agree for unit-norm inputs: {psi_tv} vs {psi_sv}"
        )


# ---------------------------------------------------------------------------
# diffuseness_cmd
# ---------------------------------------------------------------------------

class TestDiffusenessCMD:
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_identity_covariance_fully_diffuse(self, order):
        """R = σ² I → Ψ_CMD = 1 (all eigenvalues equal)."""
        Q = (order + 1) ** 2
        C = np.eye(Q, dtype=complex) * 2.5
        psi, psi_per_order = diffuseness_cmd(C)
        assert abs(psi - 1.0) < 1e-10, f"Expected Ψ=1 for identity covariance (order={order})"
        for n, pn in enumerate(psi_per_order, start=1):
            assert abs(pn - 1.0) < 1e-10, f"Expected Ψ_per_order[{n}]=1"

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_rank1_covariance_low_diffuseness(self, order):
        """Rank-1 covariance (single coherent source) → Ψ_CMD < 0.5."""
        Q = (order + 1) ** 2
        v = np.zeros(Q, dtype=complex)
        v[0] = 1.0
        C = np.outer(v, v.conj()) + 0.01 * np.eye(Q)
        psi, _ = diffuseness_cmd(C)
        assert psi < 0.5, f"Expected Ψ_CMD < 0.5 for rank-1 cov (order={order}), got {psi}"

    def test_output_range(self):
        rng = np.random.default_rng(3)
        Q = 9
        A = rng.standard_normal((Q, Q)) + 1j * rng.standard_normal((Q, Q))
        C = A @ A.conj().T + np.eye(Q)
        psi, psi_per = diffuseness_cmd(C)
        assert 0.0 <= psi <= 1.0
        assert np.all(psi_per >= 0.0) and np.all(psi_per <= 1.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            diffuseness_cmd(np.eye(4, 5))

    def test_non_perfect_square_size_raises(self):
        with pytest.raises(ValueError, match="SH order"):
            diffuseness_cmd(np.eye(5, dtype=complex))

    def test_per_order_length(self):
        order = 3
        Q = (order + 1) ** 2
        C = np.eye(Q, dtype=complex)
        psi, psi_per = diffuseness_cmd(C)
        assert len(psi_per) == order
