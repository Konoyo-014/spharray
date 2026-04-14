"""Tests for spharray.coherence.diffuse.

Mathematical invariants:
- Auto-coherence (same sensor) = 1 at all frequencies
- sinc formula: Γ(f) = sin(kd)/(kd) — verified numerically
- Matrix is Hermitian-symmetric
- Vectorized result matches loop-based reference
- diffuse_coherence_from_weights: normalised inner product
"""

from __future__ import annotations

import numpy as np
import pytest

from spharray.coherence.diffuse import (
    diffuse_coherence_from_weights,
    diffuse_coherence_matrix_omni,
)


# ---------------------------------------------------------------------------
# diffuse_coherence_matrix_omni
# ---------------------------------------------------------------------------

class TestDiffuseCoherenceMatrixOmni:
    def setup_method(self):
        self.xyz = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],   # 10 cm apart along x
            [0.0, 0.2, 0.0],   # 20 cm apart along y
        ], dtype=float)
        self.freqs = np.array([0.0, 500.0, 1000.0, 2000.0, 4000.0])

    def test_shape(self):
        G = diffuse_coherence_matrix_omni(self.xyz, self.freqs)
        assert G.shape == (5, 3, 3)

    def test_auto_coherence_is_one(self):
        """Γ(m, m) = 1 for all frequencies."""
        G = diffuse_coherence_matrix_omni(self.xyz, self.freqs)
        diag = np.array([[G[k, m, m] for m in range(3)] for k in range(5)])
        assert np.allclose(diag, 1.0, atol=1e-12)

    def test_symmetry(self):
        """Γ(m, n) = Γ(n, m)* — actually real and symmetric."""
        G = diffuse_coherence_matrix_omni(self.xyz, self.freqs)
        for k in range(len(self.freqs)):
            assert np.allclose(G[k], G[k].T.conj(), atol=1e-12), (
                f"Coherence matrix not Hermitian at freq index {k}"
            )

    def test_sinc_formula_manual(self):
        """Verify off-diagonal entry against sinc formula for a 2-sensor case."""
        c = 343.0
        f = np.array([1000.0])
        xyz = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1]])  # 10 cm apart
        d = 0.1
        k = 2.0 * np.pi * f[0] / c
        expected = np.sinc(k * d / np.pi)  # np.sinc(x) = sin(πx)/(πx)
        G = diffuse_coherence_matrix_omni(xyz, f)
        assert abs(G[0, 0, 1].real - expected) < 1e-12

    def test_vectorized_matches_loop(self):
        """Vectorized result must exactly match a reference loop."""
        c = 343.0
        xyz = self.xyz
        freqs = self.freqs
        M = xyz.shape[0]
        K = len(freqs)
        G_ref = np.zeros((K, M, M), dtype=complex)
        for ki, f in enumerate(freqs):
            k = 2.0 * np.pi * f / c
            for mi in range(M):
                for mj in range(M):
                    d = np.linalg.norm(xyz[mi] - xyz[mj])
                    kd = k * d
                    G_ref[ki, mi, mj] = np.sinc(kd / np.pi) if kd > 0 else 1.0
        G = diffuse_coherence_matrix_omni(xyz, freqs)
        assert np.allclose(G, G_ref, atol=1e-12), "Vectorized result differs from loop"

    def test_dc_frequency_all_ones(self):
        """At f=0 (k=0), kd=0 for all sensor pairs → Γ = 1 everywhere."""
        G = diffuse_coherence_matrix_omni(self.xyz, np.array([0.0]))
        assert np.allclose(G[0], 1.0, atol=1e-12)

    def test_wrong_xyz_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            diffuse_coherence_matrix_omni(np.ones((3,)), np.array([1000.0]))

    def test_single_sensor(self):
        """Single sensor → 1×1 matrix = [[1]]."""
        xyz = np.array([[0.0, 0.0, 0.0]])
        G = diffuse_coherence_matrix_omni(xyz, np.array([1000.0]))
        assert G.shape == (1, 1, 1)
        assert abs(G[0, 0, 0] - 1.0) < 1e-12

    def test_dtype_complex128(self):
        G = diffuse_coherence_matrix_omni(self.xyz, self.freqs)
        assert G.dtype == np.complex128


# ---------------------------------------------------------------------------
# diffuse_coherence_from_weights
# ---------------------------------------------------------------------------

class TestDiffuseCoherenceFromWeights:
    def test_unit_self_coherence(self):
        """Coherence of a vector with itself = 1."""
        w = np.array([1.0, 0.5, 0.25, 0.0], dtype=complex)
        gamma = diffuse_coherence_from_weights(w, w)
        assert abs(gamma - 1.0) < 1e-12

    def test_orthogonal_vectors_zero_coherence(self):
        """Orthogonal weight vectors → coherence = 0."""
        w_a = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        w_b = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
        gamma = diffuse_coherence_from_weights(w_a, w_b)
        assert abs(gamma) < 1e-12

    def test_normalisation(self):
        """Coherence is normalised: |γ| ≤ 1."""
        rng = np.random.default_rng(99)
        w_a = rng.standard_normal(9) + 1j * rng.standard_normal(9)
        w_b = rng.standard_normal(9) + 1j * rng.standard_normal(9)
        gamma = diffuse_coherence_from_weights(w_a, w_b)
        assert abs(gamma) <= 1.0 + 1e-12

    def test_zero_vector_returns_zero(self):
        w_a = np.zeros(4, dtype=complex)
        w_b = np.ones(4, dtype=complex)
        gamma = diffuse_coherence_from_weights(w_a, w_b)
        assert gamma == 0.0 + 0.0j

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            diffuse_coherence_from_weights(np.ones(4), np.ones(5))

    def test_real_symmetric_for_real_weights(self):
        """For real weight vectors, γ = γ* (real coherence)."""
        w_a = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
        w_b = np.array([0.25, 0.75, 0.0, 0.0], dtype=float)
        gamma = diffuse_coherence_from_weights(w_a, w_b)
        assert abs(gamma.imag) < 1e-12
