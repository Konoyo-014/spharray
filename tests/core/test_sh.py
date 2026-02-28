import numpy as np

from spherical_array_processing.array.sampling import get_tdesign_fallback
from spherical_array_processing.sh import (
    complex_to_real_coeffs,
    matrix,
    real_to_complex_coeffs,
    replicate_per_order,
)
from spherical_array_processing.types import SHBasisSpec


def test_replicate_per_order():
    x = replicate_per_order([1, 2, 3])
    assert x.tolist() == [1, 2, 2, 2, 3, 3, 3, 3, 3]


def test_real_complex_coeff_roundtrip():
    n = 3
    rng = np.random.default_rng(0)
    # Build complex SH coefficients corresponding to a real-valued function:
    # c_{n,-m} = (-1)^m conj(c_{n,m}), c_{n,0} real
    c = np.zeros((2, (n + 1) ** 2), dtype=np.complex128)
    from spherical_array_processing.sh import acn_index

    for nn in range(n + 1):
        c[:, acn_index(nn, 0)] = rng.normal(size=2)
        for m in range(1, nn + 1):
            cp = rng.normal(size=2) + 1j * rng.normal(size=2)
            c[:, acn_index(nn, m)] = cp
            c[:, acn_index(nn, -m)] = ((-1) ** m) * np.conj(cp)
    r = complex_to_real_coeffs(c, n, axis=-1)
    c2 = real_to_complex_coeffs(r, n, axis=-1)
    assert np.allclose(c, c2, atol=1e-10)


def test_real_sh_matrix_shape_and_finite():
    grid = get_tdesign_fallback(order=3, n_points=64)
    y = matrix(SHBasisSpec(max_order=3, basis="real"), grid)
    assert y.shape == (64, 16)
    assert np.isfinite(y).all()
