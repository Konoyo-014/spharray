import numpy as np

from spherical_array_processing.acoustics import bn_matrix, plane_wave_radial_bn


def test_bn_shapes():
    kr = np.linspace(0.1, 3.0, 10)
    b0 = plane_wave_radial_bn(0, kr, sphere="open")
    assert b0.shape == (10,)
    bm = bn_matrix(3, kr, sphere="rigid", repeat_per_order=False)
    assert bm.shape == (10, 4)
    br = bn_matrix(3, kr, sphere="rigid", repeat_per_order=True)
    assert br.shape == (10, 16)

