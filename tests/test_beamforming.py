"""Tests for fixed and adaptive beamformers.

Mathematical properties verified:
- All fixed beamformers: unit front gain Sigma b_n = 1
- Hypercardioid weights analytical formula (2n+1)/(N+1)^2
- MVDR: unit response w^H d = 1
- LCMV: constraint satisfaction C^H w = f
- Pattern evaluation at theta = 0 matches unit gain
"""
import numpy as np
import pytest
from spherical_array_processing.beamforming import (
    axisymmetric_pattern,
    beam_weights_cardioid,
    beam_weights_hypercardioid,
    beam_weights_maxev,
    beam_weights_supercardioid,
    lcmv_weights,
    mvdr_weights,
)


# ── Fixed beamformers ─────────────────────────────────────────────────────────

class TestFixedBeamformers:
    """All fixed beamformers should have unit front gain: B(0) = 1."""

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_hypercardioid_unit_front_gain(self, N):
        b = beam_weights_hypercardioid(N)
        assert len(b) == N + 1
        p = axisymmetric_pattern(np.array([0.0]), b)
        assert abs(p[0] - 1.0) < 1e-10, f"Hypercardioid N={N}: front gain = {p[0]}"

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_cardioid_unit_front_gain(self, N):
        b = beam_weights_cardioid(N)
        p = axisymmetric_pattern(np.array([0.0]), b)
        assert abs(p[0] - 1.0) < 1e-6, f"Cardioid N={N}: front gain = {p[0]}"

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_supercardioid_unit_front_gain(self, N):
        b = beam_weights_supercardioid(N)
        assert len(b) == N + 1
        p = axisymmetric_pattern(np.array([0.0]), b)
        assert abs(p[0] - 1.0) < 1e-6, f"Supercardioid N={N}: front gain = {p[0]}"

    @pytest.mark.parametrize("N", [1, 2, 3, 4])
    def test_maxev_unit_front_gain(self, N):
        b = beam_weights_maxev(N)
        assert len(b) == N + 1
        p = axisymmetric_pattern(np.array([0.0]), b)
        assert abs(p[0] - 1.0) < 1e-10, f"MaxEV N={N}: front gain = {p[0]}"

    def test_hypercardioid_formula(self):
        """Weights are proportional to (2n+1): b_n / b_0 = (2n+1) for all n."""
        for N in range(1, 6):
            b = beam_weights_hypercardioid(N)
            # Check ratios are preserved: b_n / b_0 = (2n+1)
            for n in range(1, N + 1):
                ratio = b[n] / b[0]
                expected = (2 * n + 1) / 1.0   # b_n/b_0 = (2n+1)/(1)
                assert abs(ratio - expected) < 1e-10, (
                    f"N={N}, n={n}: b[n]/b[0] = {ratio}, expected {expected}"
                )

    def test_cardioid_rear_null(self):
        """Cardioid should have zero (or near-zero) gain at theta = pi."""
        for N in [1, 2, 3]:
            b = beam_weights_cardioid(N)
            p = axisymmetric_pattern(np.array([np.pi]), b)
            assert abs(p[0]) < 1e-4, f"Cardioid N={N} rear gain = {p[0]}"

    def test_supercardioid_order1_analytical(self):
        """For N=1, the optimal ratio b_1/b_0 = sqrt(3) is preserved after normalization."""
        b = beam_weights_supercardioid(1)
        # The key analytical property: b_1/b_0 = sqrt(3)
        assert abs(b[1] / b[0] - np.sqrt(3.0)) < 1e-8, (
            f"Supercardioid N=1: b_1/b_0 = {b[1]/b[0]}, expected {np.sqrt(3.0)}"
        )

    def test_hypercardioid_di_maximum(self):
        """Hypercardioid should have the highest directivity index among common patterns."""
        N = 3
        b_hyper = beam_weights_hypercardioid(N)
        b_card  = beam_weights_cardioid(N)
        # DI = (Sigma b_n)^2 / (Sigma b_n^2 / (2n+1)) * (N+1)^2
        # For hypercardioid, DI = (N+1)^2 exactly
        n = np.arange(N + 1, dtype=float)
        di_hyper = (np.sum(b_hyper) ** 2) / np.sum(b_hyper ** 2 / (2 * n + 1))
        di_card  = (np.sum(b_card) ** 2) / np.sum(b_card ** 2 / (2 * n + 1))
        assert di_hyper >= di_card - 1e-10


class TestAxisymmetricPattern:
    def test_shape(self):
        b = beam_weights_hypercardioid(2)
        theta = np.linspace(0, np.pi, 100)
        p = axisymmetric_pattern(theta, b)
        assert p.shape == (100,)

    def test_scalar_input(self):
        b = beam_weights_hypercardioid(1)
        p = axisymmetric_pattern(0.0, b)
        assert p.shape == ()


# ── Adaptive beamformers ──────────────────────────────────────────────────────

class TestMVDR:
    def test_unit_response_identity_cov(self):
        """MVDR weight satisfies w^H d = 1 for identity covariance."""
        Q = 9
        R = np.eye(Q, dtype=complex)
        d = np.zeros(Q, dtype=complex); d[0] = 1.0
        w = mvdr_weights(R, d)
        assert abs(np.conj(w) @ d - 1.0) < 1e-12

    def test_unit_response_random_cov(self):
        """Unit response must hold for any positive-definite R."""
        rng = np.random.default_rng(42)
        Q = 4
        A = rng.standard_normal((Q, Q)) + 1j * rng.standard_normal((Q, Q))
        R = A @ A.conj().T + 0.1 * np.eye(Q)
        d = rng.standard_normal(Q) + 1j * rng.standard_normal(Q)
        w = mvdr_weights(R, d, diagonal_loading=0.0)
        assert abs(np.conj(w) @ d - 1.0) < 1e-10

    def test_output_shape_single_steering(self):
        Q = 9
        R = np.eye(Q, dtype=complex)
        d = np.ones(Q, dtype=complex) / np.sqrt(Q)
        w = mvdr_weights(R, d)
        assert w.shape == (Q,)

    def test_output_shape_multiple_steering(self):
        Q = 9
        K = 3
        R = np.eye(Q, dtype=complex)
        D = np.eye(Q, dtype=complex)[:, :K]    # Q × K
        W = mvdr_weights(R, D)
        assert W.shape == (Q, K)

    def test_invalid_cov_raises(self):
        with pytest.raises(ValueError):
            mvdr_weights(np.eye(3), np.ones(4))

    def test_non_square_cov_raises(self):
        with pytest.raises(ValueError):
            mvdr_weights(np.ones((3, 4)), np.ones(3))


class TestLCMV:
    def test_constraint_satisfaction(self):
        """LCMV satisfies C^H w = f exactly."""
        rng = np.random.default_rng(7)
        Q, K = 9, 2
        A = rng.standard_normal((Q, Q)) + 1j * rng.standard_normal((Q, Q))
        R = A @ A.conj().T + np.eye(Q)
        C = rng.standard_normal((Q, K)) + 1j * rng.standard_normal((Q, K))
        f = np.array([1.0 + 0j, 0.0 + 0j])
        w = lcmv_weights(R, C, f, diagonal_loading=0.0)
        assert w.shape == (Q,)
        assert np.allclose(C.conj().T @ w, f, atol=1e-10)

    def test_lcmv_reduces_to_mvdr(self):
        """With single constraint and unit response, LCMV == MVDR."""
        rng = np.random.default_rng(11)
        Q = 9
        A = rng.standard_normal((Q, Q)) + 1j * rng.standard_normal((Q, Q))
        R = A @ A.conj().T + np.eye(Q)
        d = rng.standard_normal(Q) + 1j * rng.standard_normal(Q)
        w_mvdr = mvdr_weights(R, d, diagonal_loading=0.0)
        w_lcmv = lcmv_weights(R, d[:, None], np.array([1.0 + 0j]),
                               diagonal_loading=0.0)
        assert np.allclose(w_mvdr, w_lcmv, atol=1e-10)

    def test_invalid_inputs(self):
        R = np.eye(4, dtype=complex)
        C = np.ones((4, 2), dtype=complex)
        with pytest.raises(ValueError):
            lcmv_weights(R, C, np.array([1.0, 2.0, 3.0]))   # f length mismatch
