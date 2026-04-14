# spherical-array-processing

[![CI](https://github.com/Konoyo-014/spherical-array-processing/actions/workflows/ci.yml/badge.svg)](https://github.com/Konoyo-014/spherical-array-processing/actions/workflows/ci.yml)

A self-contained Python toolkit for spherical microphone array processing. This
repository is the intended open-source root; migration workspaces, local virtual
environments, generated reports, and large reference assets should stay outside
this directory.

## Features

- **Spherical harmonics**: complex and real (tesseral) SH basis matrices, ACN/SN3D normalisation, forward and inverse SHTs, coefficient conversions
- **Spatial sampling**: Fibonacci, equiangle, and t-design grids with integration weights
- **Array simulation**: free-field plane-wave transfer functions
- **Fixed beamforming**: cardioid, hypercardioid (max DI), supercardioid (max F/B ratio), MaxEV (perceptual) weight vectors and pattern evaluation
- **Adaptive beamforming**: MVDR and LCMV SH-domain beamformers
- **DOA estimation**: plane-wave decomposition (PWD) and MUSIC spatial spectra with peak picking
- **Diffuseness estimation**: IE, TV, SV, and CMD estimators from FOA/SH covariance
- **Diffuse-field coherence**: sinc-based models for omnidirectional sensor arrays
- **Acoustics**: spherical Bessel/Hankel functions, modal coefficients for open/rigid arrays
- **Plotting**: 3-D array geometry, 2-D spatial maps, MATLAB-like figure style

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

Python >= 3.11 is required. Core dependencies: `numpy`, `scipy`, `matplotlib`.

Optional extras:

| Extra  | Adds                                         |
|--------|----------------------------------------------|
| `audio`| `soundfile` for WAV I/O                      |
| `image`| `scikit-image` for image processing utilities|
| `dev`  | `pytest`, `build`                            |
| `notebook` | `jupyterlab`, `ipykernel`                |

## Quick start

```python
import numpy as np
import spherical_array_processing as sap

# ── Spatial sampling ──────────────────────────────────────────────────────────
grid = sap.array.fibonacci_grid(100)   # SphericalGrid, 100 directions
print(f"weights sum: {grid.weights.sum():.4f}")  # ≈ 4π

# ── SH basis and transforms ───────────────────────────────────────────────────
spec = sap.SHBasisSpec(max_order=3, basis="complex", angle_convention="az_colat")
Y  = sap.sh.matrix(spec, grid)          # (100, 16) complex SH matrix
f  = np.random.randn(grid.size)
nm = sap.sh.direct_sht(f, Y, grid)     # (16,) SH coefficients
f_rec = sap.sh.inverse_sht(nm, Y).real # reconstruct

# ── Fixed beamforming ─────────────────────────────────────────────────────────
b       = sap.beamforming.beam_weights_hypercardioid(3)   # max-DI weights, N=3
thetas  = np.linspace(0, np.pi, 181)
pattern = sap.beamforming.axisymmetric_pattern(thetas, b) # B(0) = 1

# ── Array simulation ──────────────────────────────────────────────────────────
sensor_grid = sap.array.fibonacci_grid(32)
geometry    = sap.ArrayGeometry(radius_m=0.042, sensor_grid=sensor_grid)
src_grid    = sap.array.fibonacci_grid(4)
freqs, H    = sap.array.simulate_plane_wave_array_response(
    fft_len=256, fs=16000.0, geometry=geometry, source_grid=src_grid
)
# H.shape = (129, 32, 4)  — (n_bins, n_mics, n_sources)

# ── DOA estimation ────────────────────────────────────────────────────────────
search = sap.array.fibonacci_grid(300)
Q      = spec.n_coeffs                        # (N+1)^2 = 16
R_eye  = np.eye(Q, dtype=complex)             # diffuse covariance → flat spectrum
result = sap.doa.pwd_spectrum(R_eye, search, spec, n_peaks=1)
print("Peak direction (az, el) rad:", result.peak_dirs_rad[0])

# ── MUSIC spectrum ────────────────────────────────────────────────────────────
y0  = sap.sh.matrix(spec, search)[42]         # steering vector at grid index 42
R1  = np.outer(y0, y0.conj()) + 0.01 * np.eye(Q)
res = sap.doa.music_spectrum(R1, search, spec, n_sources=1)
print("MUSIC peak index:", res.peak_indices[0])  # → 42

# ── Diffuseness ───────────────────────────────────────────────────────────────
frame = (np.random.randn(4, 1024) + 1j * np.random.randn(4, 1024))
psi   = sap.diffuseness.diffuseness_ie(frame)
print(f"IE diffuseness mean: {psi.mean():.3f}")
```

## New user path

Start with [docs/getting_started.md](docs/getting_started.md). It walks through
installation, the first runnable example, and the smallest useful SH workflow.
Read [docs/concepts.md](docs/concepts.md) when terms such as **ACN**,
**colatitude**, **modal coefficient**, **SHT**, or **unit front gain** are not
yet clear.

The most useful runnable tutorials are:

```bash
python examples/tutorials/01_sht_and_beamforming.py
python examples/tutorials/02_simulated_doa_pipeline.py
python examples/tutorials/03_modal_equalization_pipeline.py
```

Notebook users can open `examples/notebooks/getting_started.ipynb`.

## Module reference

| Module              | Key symbols                                                                                         |
|---------------------|-----------------------------------------------------------------------------------------------------|
| `sap.sh`            | `matrix`, `complex_matrix`, `real_matrix`, `direct_sht`, `inverse_sht`, `acn_index`, `complex_to_real_coeffs`, `real_to_complex_coeffs`, `replicate_per_order` |
| `sap.array`         | `fibonacci_grid`, `equiangle_sampling`, `get_tdesign_fallback`, `simulate_plane_wave_array_response`, `spatial_aliasing_frequency`, `max_sh_order` |
| `sap.acoustics`     | `besseljs`, `besseljsd`, `besselhs`, `besselhsd`, `plane_wave_radial_bn`, `bn_matrix`, `sph_modal_coeffs`, `equalize_modal_coeffs` |
| `sap.beamforming`   | `beam_weights_cardioid`, `beam_weights_hypercardioid`, `beam_weights_supercardioid`, `beam_weights_maxev`, `axisymmetric_pattern`, `mvdr_weights`, `lcmv_weights`, `steer_sh_weights`, `beamform_sh` |
| `sap.doa`           | `pwd_spectrum`, `music_spectrum`, `peak_pick_spectrum`, `spatial_spectrum_from_map`, `estimate_sh_cov`, `forward_backward_cov`, `diagonal_loading` |
| `sap.diffuseness`   | `diffuseness_ie`, `diffuseness_tv`, `diffuseness_sv`, `diffuseness_cmd`, `intensity_vectors_from_foa` |
| `sap.coherence`     | `diffuse_coherence_matrix_omni`, `diffuse_coherence_from_weights`                                   |
| `sap.coords`        | `cart_to_sph`, `sph_to_cart`, `az_colat_to_azel`, `azel_to_az_colat`, `angular_distance`, `angular_distance_deg` |
| `sap.plotting`      | `plot_mic_array`, `plot_directional_map_from_grid`, `apply_matlab_like_style`, `figure_style_context` |

## Beamformer normalisation

All beamformers satisfy `B(0) = 1` (unit front gain) in the `axisymmetric_pattern` convention:

```
B(θ) = Σ_{n=0}^{N} b_n · (2n+1)/(4π) · P_n(cos θ)
```

The weight vectors are normalised so that `Σ b_n · (2n+1)/(4π) = 1`.

| Beamformer    | DI (linear) | Notes                                     |
|---------------|-------------|-------------------------------------------|
| Cardioid      | varies      | Widest main lobe, no sidelobes            |
| Hypercardioid | `(N+1)²`   | Maximum directivity index                 |
| Supercardioid | varies      | Maximum front-to-back energy ratio        |
| MaxEV         | varies      | Von-Hann taper, good perceptual quality   |

## SH conventions

Channel ordering follows ACN (Ambisonic Channel Numbering):

```
index q = n² + n + m,   n = 0…N,   m = -n…n
```

Complex SH are orthonormal (SN3D normalisation scaled to 4π). Real SH use tesseral form. Converting `complex → real → complex` is lossless only for conjugate-symmetric inputs (i.e., inputs that correspond to real-valued spatial functions).

## Running tests

```bash
python -m pytest tests/ -q
# 248 passed
```

## License

MIT. See [LICENSE](LICENSE).

## References

- Rafaely, B. (2015). *Fundamentals of Spherical Array Processing*. Springer.
- Zotter, F. & Frank, M. (2019). *Ambisonics*. Springer.
- Schmidt, R. O. (1986). Multiple emitter location and signal parameter estimation. *IEEE Trans. Antennas Propag.*, 34(3), 276–280.
