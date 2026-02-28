"""Test module.

Usage:
    pytest -q tests/core/test_beamforming_doa.py
"""

import numpy as np

from spherical_array_processing.array.sampling import get_tdesign_fallback
from spherical_array_processing.beamforming import mvdr_weights
from spherical_array_processing.doa import music_spectrum, pwd_spectrum
from spherical_array_processing.types import SHBasisSpec


def test_mvdr_unit_response():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    r = np.eye(4, dtype=complex)
    d = np.array([1, 0, 0, 0], dtype=complex)
    w = mvdr_weights(r, d)
    assert np.allclose(np.vdot(w, d), 1.0)


def test_pwd_and_music_return_peaks():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    basis = SHBasisSpec(max_order=1, basis="real")
    grid = get_tdesign_fallback(order=2, n_points=50)
    r = np.eye(basis.n_coeffs, dtype=complex)
    pwd = pwd_spectrum(r, grid, basis, n_peaks=3)
    mus = music_spectrum(r + 0.01 * np.eye(basis.n_coeffs), grid, basis, n_sources=1)
    assert pwd.peak_indices.size == 3
    assert mus.peak_indices.size == 1

