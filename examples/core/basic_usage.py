from __future__ import annotations

import numpy as np

from spherical_array_processing.array.sampling import fibonacci_grid
from spherical_array_processing.sh import SHBasisSpec, matrix


def main() -> None:
    grid = fibonacci_grid(64)
    spec = SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
    y = matrix(spec, grid)
    print("grid_size:", grid.size)
    print("sh_matrix_shape:", y.shape)
    print("first_row_energy:", float(np.sum(np.abs(y[0]) ** 2)))


if __name__ == "__main__":
    main()

