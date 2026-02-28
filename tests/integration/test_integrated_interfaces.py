"""Test module.

Usage:
    pytest -q tests/integration/test_integrated_interfaces.py
"""

import numpy as np

from spherical_array_processing.toolkit import array_response_simulator as ars
from spherical_array_processing.toolkit import sht
from spherical_array_processing.toolkit.sht import functions as sht_functions
from spherical_array_processing.toolkit.spatial import SPATIAL_RESOURCE_DIR
from spherical_array_processing.toolkit.harmonics import HARMONICS_RESOURCE_DIR


def test_toolkit_resource_dirs_are_resolvable():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    assert SPATIAL_RESOURCE_DIR.exists()
    assert SPATIAL_RESOURCE_DIR.is_dir()
    assert HARMONICS_RESOURCE_DIR.exists()
    assert HARMONICS_RESOURCE_DIR.is_dir()


def test_toolkit_exports_have_expected_surface():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    assert "getFliegeNodes" in sht.__all__
    assert "getTdesign" in sht.__all__
    assert "simulateSphArray" in ars.__all__
    assert "sphModalCoeffs" in ars.__all__


def test_sht_smoke_roundtrip_and_geometry_helpers():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    dirs = sht.grid2dirs(90, 90, POLAR_OR_ELEV=1, ZEROED_OR_CENTERED=1)
    f = np.linspace(0.0, 1.0, dirs.shape[0])
    fn, _ = sht.directSHT(1, f, dirs, "real")
    recon = sht.inverseSHT(fn, dirs, "real")
    grid = sht.Fdirs2grid(f, 90, 90)

    _, t_dirs = sht.getTdesign(4)
    w = sht.getVoronoiWeights(t_dirs)

    assert fn.shape == (4, 1)
    assert recon.shape == (dirs.shape[0], 1)
    assert grid.shape == (3, 4)
    assert np.isclose(np.sum(w), 4 * np.pi, atol=5e-2)

    vecs, dirs_f, weights = sht.getFliegeNodes(2)
    assert vecs.shape[1] == 3
    assert dirs_f.shape[1] == 2
    assert weights.ndim == 1
    assert np.isclose(np.sum(weights), 4 * np.pi, atol=5e-2)


def test_array_response_simulator_smoke():
    """Usage:
        Run this test case.
    
    Returns:
        value.
    """
    x = np.array([0.0, 0.3, 1.2])
    j0 = ars.sph_besselj(0, x)
    coeffs = ars.sphModalCoeffs(3, np.linspace(0.1, 2.0, 8), "open")

    mic_dirs = np.array([[0.0, 0.0], [np.pi / 2, 0.0]], dtype=float)
    src_dirs = np.array([[0.0, 0.0]], dtype=float)
    h_t, h_f = ars.simulateSphArray(32, mic_dirs, src_dirs, "open", 0.042, 2, 16000)

    u_doa = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    r_mic = np.array([[0.02, 0.0, 0.0], [0.0, 0.02, 0.0]], dtype=float)
    irs, tfs = ars.getArrayResponse(u_doa, r_mic, None, None, 32, fs=16000)

    assert np.isclose(j0[0], 1.0)
    assert coeffs.shape == (8, 4)
    assert h_t.shape == (32, 2, 1)
    assert h_f.shape == (17, 2, 1)
    assert irs.shape == (32, 2, 2)
    assert tfs.shape == (17, 2, 2)


def test_get_fliege_nodes_uses_deterministic_fallback_when_resource_missing(tmp_path, monkeypatch):
    """Usage:
        Run this test case.
    
    Args:
        tmp_path: value.
        monkeypatch: value.
    
    Returns:
        value.
    """
    monkeypatch.setattr(sht_functions, "_FLIEGE_MAT", tmp_path / "missing_fliege.mat")
    v1, d1, w1 = sht.getFliegeNodes(5)
    v2, d2, w2 = sht.getFliegeNodes(5)
    assert np.allclose(v1, v2)
    assert np.allclose(d1, d2)
    assert np.allclose(w1, w2)
