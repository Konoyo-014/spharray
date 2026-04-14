"""Tests for spharray.doa.spectra.

Mathematical invariants:
- PWD: rank-1 covariance from a known direction → peak at that direction
- MUSIC: single source → pseudospectrum peak at source direction
- PWD flat spectrum for identity (diffuse) covariance
- Peak-pick: top-N ordering is monotone decreasing
- SpatialSpectrumResult structure
"""

from __future__ import annotations

import numpy as np
import pytest

from spharray.array.sampling import fibonacci_grid
from spharray.doa.spectra import (
    music_spectrum,
    peak_pick_spectrum,
    pwd_spectrum,
    spatial_spectrum_from_map,
)
from spharray.sh import matrix as sh_matrix
from spharray.types import SHBasisSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def order2_basis():
    return SHBasisSpec(max_order=2, basis="complex", angle_convention="az_colat")


@pytest.fixture
def search_grid():
    return fibonacci_grid(300)


# ---------------------------------------------------------------------------
# peak_pick_spectrum
# ---------------------------------------------------------------------------

class TestPeakPickSpectrum:
    def test_single_peak(self):
        s = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        idx = peak_pick_spectrum(s, 1)
        assert idx[0] == 1

    def test_top2_decreasing_order(self):
        s = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        idx = peak_pick_spectrum(s, 2)
        assert idx[0] == 1   # highest
        assert idx[1] == 3   # second highest

    def test_clamp_n_peaks(self):
        s = np.arange(5, dtype=float)
        idx = peak_pick_spectrum(s, 100)   # ask for more than available
        assert len(idx) == 5

    def test_dtype_int64(self):
        s = np.ones(10)
        idx = peak_pick_spectrum(s, 3)
        assert idx.dtype == np.int64


# ---------------------------------------------------------------------------
# spatial_spectrum_from_map
# ---------------------------------------------------------------------------

class TestSpatialSpectrumFromMap:
    def test_known_peak(self, search_grid):
        spec = np.zeros(search_grid.size)
        spec[42] = 1.0
        result = spatial_spectrum_from_map(spec, search_grid, n_peaks=1)
        assert result.peak_indices[0] == 42

    def test_peak_dirs_shape(self, search_grid):
        spec = np.zeros(search_grid.size)
        spec[10] = 1.0
        spec[20] = 0.5
        result = spatial_spectrum_from_map(spec, search_grid, n_peaks=2)
        assert result.peak_dirs_rad.shape == (2, 2)

    def test_metadata_attached(self, search_grid):
        spec = np.ones(search_grid.size)
        result = spatial_spectrum_from_map(spec, search_grid, 1, metadata={"method": "test"})
        assert result.metadata["method"] == "test"


# ---------------------------------------------------------------------------
# pwd_spectrum
# ---------------------------------------------------------------------------

class TestPWDSpectrum:
    def test_identity_cov_flat_spectrum(self, order2_basis, search_grid):
        """Diffuse-field covariance (R = I) → flat PWD spectrum."""
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        result = pwd_spectrum(R, search_grid, order2_basis, n_peaks=1)
        s = result.spectrum
        # Flatness: std/mean should be small
        rel_std = s.std() / s.mean()
        assert rel_std < 0.05, f"Expected flat spectrum but rel_std={rel_std:.4f}"

    def test_rank1_peak_location(self, order2_basis, search_grid):
        """Single-source covariance → PWD peak at source direction."""
        # Pick a specific grid point as 'source'
        src_idx = 42
        Y = sh_matrix(order2_basis, search_grid)   # (G, Q)
        y_src = Y[src_idx, :]                       # SH vector at source
        R = np.outer(y_src, y_src.conj())           # rank-1 cov
        result = pwd_spectrum(R, search_grid, order2_basis, n_peaks=1)
        found_idx = result.peak_indices[0]
        # Peak should be at or very near source direction
        assert found_idx == src_idx, (
            f"PWD peak at grid index {found_idx}, expected {src_idx}"
        )

    def test_output_shape(self, order2_basis, search_grid):
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        result = pwd_spectrum(R, search_grid, order2_basis)
        assert result.spectrum.shape == (search_grid.size,)

    def test_spectrum_nonnegative(self, order2_basis, search_grid):
        """PWD spectrum is always ≥ 0 (y^H R y is real non-negative for PSD R)."""
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        result = pwd_spectrum(R, search_grid, order2_basis)
        assert np.all(result.spectrum >= -1e-12)

    def test_shape_mismatch_raises(self, order2_basis, search_grid):
        """Covariance shape inconsistent with basis → ValueError."""
        R = np.eye(4, dtype=complex)   # wrong size for order-2 basis (Q=9)
        with pytest.raises(ValueError):
            pwd_spectrum(R, search_grid, order2_basis)


# ---------------------------------------------------------------------------
# music_spectrum
# ---------------------------------------------------------------------------

class TestMUSICSpectrum:
    def test_rank1_cov_peak_at_source(self, order2_basis, search_grid):
        """Single source rank-1 covariance → MUSIC peak at source direction."""
        src_idx = 77
        Y = sh_matrix(order2_basis, search_grid)
        y_src = Y[src_idx, :]
        Q = order2_basis.n_coeffs
        R = np.outer(y_src, y_src.conj()) + 0.01 * np.eye(Q)
        result = music_spectrum(R, search_grid, order2_basis, n_sources=1)
        found_idx = result.peak_indices[0]
        assert found_idx == src_idx, (
            f"MUSIC peak at {found_idx}, expected {src_idx}"
        )

    def test_output_shape(self, order2_basis, search_grid):
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        result = music_spectrum(R, search_grid, order2_basis, n_sources=1)
        assert result.spectrum.shape == (search_grid.size,)

    def test_n_sources_zero_raises(self, order2_basis, search_grid):
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        with pytest.raises(ValueError, match="n_sources"):
            music_spectrum(R, search_grid, order2_basis, n_sources=0)

    def test_n_sources_too_large_raises(self, order2_basis, search_grid):
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        with pytest.raises(ValueError, match="n_sources"):
            music_spectrum(R, search_grid, order2_basis, n_sources=Q)

    def test_non_square_cov_raises(self, order2_basis, search_grid):
        with pytest.raises(ValueError):
            music_spectrum(np.eye(4, 5, dtype=complex), search_grid, order2_basis, n_sources=1)

    def test_music_sharper_than_pwd(self, order2_basis, search_grid):
        """MUSIC pseudospectrum has larger peak-to-mean ratio than PWD."""
        src_idx = 120
        Y = sh_matrix(order2_basis, search_grid)
        y_src = Y[src_idx, :]
        Q = order2_basis.n_coeffs
        noise = 0.01
        R = np.outer(y_src, y_src.conj()) + noise * np.eye(Q)
        pwd_res = pwd_spectrum(R, search_grid, order2_basis, n_peaks=1)
        music_res = music_spectrum(R, search_grid, order2_basis, n_sources=1)

        def peak_to_mean(s):
            return s.max() / s.mean()

        assert peak_to_mean(music_res.spectrum) >= peak_to_mean(pwd_res.spectrum), (
            "MUSIC should have sharper peak than PWD"
        )

    def test_n_peaks_defaults_to_n_sources(self, order2_basis, search_grid):
        Q = order2_basis.n_coeffs
        R = np.eye(Q, dtype=complex)
        result = music_spectrum(R, search_grid, order2_basis, n_sources=2)
        assert len(result.peak_indices) == 2
