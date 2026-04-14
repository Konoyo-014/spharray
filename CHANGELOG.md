# Changelog

All notable changes to `spharray` are documented here.

## [0.3.0] — 2026-04-14

### Bug fixes

- **SH beam steering convention** (`beamforming/steer.py`): `steer_sh_weights` was returning `w_nm = b_n · Y_nm*(Ω_0)` (conjugated) instead of the correct `w_nm = b_n · Y_nm(Ω_0)` (Rafaely 2015, Sec. 4.1). `beamform_sh` was applying `sig @ w.conj()` instead of `sig @ w`. The distortionless constraint `Σ_nm Y_nm*(Ω_s) · b_n · Y_nm(Ω_0) = 1` is now satisfied.

### New features

#### `coords` module

- **`angular_distance(az1, a2_1, az2, a2_2, convention)`**: Numerically stable great-circle distance using the haversine formula. Supports both `"az_el"` and `"az_colat"` conventions.
- **`angular_distance_deg(...)`**: Convenience wrapper returning degrees.

#### `array` module

- **`spatial_aliasing_frequency(array_radius_m, max_order, c=343.0)`**: Returns the maximum usable frequency `f = N·c / (2π R)` above which spatial aliasing occurs (kR > N condition).
- **`max_sh_order(array_radius_m, freq_hz_max, c=343.0)`**: Returns the maximum SH order supportable at a given frequency: `N_max = floor(2π f R / c)`.

#### `acoustics` module

- **`equalize_modal_coeffs(sh_signals, bn, reg_param=1e-4, reg_type="tikhonov")`**: Modal equalizer for rigid-sphere SHT. Inverts the radial filter `b_n(kR)` with Tikhonov regularization (`reg_type="tikhonov"`) or soft-limiting (`reg_type="softlimit"`). Accepts both per-order `bn` of shape `(K, N+1)` and ACN-expanded `bn` of shape `(K, Q)`.

#### `beamforming` module

- **`steer_sh_weights(b_n, look_azimuth, look_angle2, basis)`**: Builds the SH-domain weight vector for an axisymmetric beam steered to an arbitrary look direction using the SH addition theorem.
- **`beamform_sh(sh_signals, weights)`**: Applies SH weights to SH signals, contracting the last (channel) axis.

#### `doa` module

- **`estimate_sh_cov(sh_snapshots)`**: Computes the sample SH-domain covariance matrix `R = (1/L) A^H A` from a matrix of SH snapshots. Auto-transposes `(Q, L)` input.
- **`forward_backward_cov(R)`**: Forward-backward averaged covariance `R_fb = (R + P R* P) / 2` for improved robustness against coherent sources.
- **`diagonal_loading(R, load, relative=True)`**: Adds `δI` to the covariance, where `δ = load · trace(R) / Q` (relative mode) or `δ = load` (absolute mode).

### New files

- `examples/core/sh_transforms.py` — SH orthonormality check, direct/inverse SHT, coefficient roundtrip
- `examples/core/beamforming.py` — Beamformer pattern comparison, DI computation, MVDR demo
- `examples/core/doa_estimation.py` — PWD/MUSIC pipeline from rank-1 covariance and simulation
- `tests/test_new_features.py` — 42 tests covering all new v0.3.0 features

### Documentation

- `pyproject.toml`: added `keywords`, `classifiers`, and `[project.urls]` (Source, Bug Tracker, Changelog)
- Module-level docstrings updated to list new public symbols
- Added a beginner-oriented getting started guide, concepts guide, release
  checklist, runnable tutorial scripts, and a getting started notebook.
- Added source distribution manifest rules so docs, examples, notebooks, tests,
  and governance files are included in the sdist while local environments and
  generated artifacts stay excluded.
- Added `py.typed` and package-data metadata to make the `Typing :: Typed`
  classifier accurate.
- Expanded CI to test supported Python versions, run tutorial scripts, execute
  notebook code cells, build distributions, smoke-test the wheel, and verify the
  source distribution test suite.

## [0.2.0] — 2026-03-13

### Bug fixes

- **Beamforming normalisation** (`beamforming/fixed.py`): `beam_weights_hypercardioid`, `beam_weights_supercardioid`, and `beam_weights_maxev` were normalised to `Σ b_n = 1` (sum convention), but `axisymmetric_pattern` uses `B(θ) = Σ b_n (2n+1)/(4π) P_n(cos θ)`, so the correct unit-front-gain normalisation requires `Σ b_n (2n+1)/(4π) = 1`. All three functions now normalise by the actual front gain computed via the pattern formula. `beam_weights_cardioid` was already correct (via `_legendre_fit_axisymmetric`).

### New test files

- `tests/test_beamforming.py` — 25 tests covering all four beamformer types, `axisymmetric_pattern`, unit front gain, rear null, rear attenuation, DI ratio, and supercardioid analytical ratio
- `tests/test_sh.py` — 40+ tests for SH basis matrices, orthonormality (using large Fibonacci grids, 1% tolerance), forward/inverse SHT roundtrip, real/complex coefficient conversions
- `tests/test_sampling.py` — tests for `fibonacci_grid`, `equiangle_grid`: grid size formula (`8(N+1)²`), weight sum (≈ 4π), angle range and convention
- `tests/test_doa.py` — tests for `peak_pick_spectrum`, `spatial_spectrum_from_map`, `pwd_spectrum`, `music_spectrum`; rank-1 covariance → exact PWD/MUSIC peak; MUSIC sharpness vs PWD
- `tests/test_diffuseness.py` — tests for all four diffuseness estimators; identity covariance → Ψ = 1, rank-1 → Ψ small; anti-parallel intensity vectors → TV Ψ = 1
- `tests/test_coherence.py` — tests for `diffuse_coherence_matrix_omni` and `diffuse_coherence_from_weights`; DC frequency → all coherences = 1, sinc formula match, Hermitian symmetry
- `tests/test_integration.py` — end-to-end pipeline tests: simulation → SHT → DOA → beamforming; 200-sensor Fibonacci array for reliable N=2 DOA

### Documentation

- **`README.md`** fully rewritten with installation instructions, quick-start code, module reference table, beamformer normalisation notes, and SH convention details
- Module docstrings reviewed and updated to match actual implementation behaviour
- All `Examples` sections in docstrings verified to run correctly

### Internal

- Equiangle grid size formula documented: `n_theta = 2(N+1)`, `n_phi = 4(N+1)`, total `8(N+1)²` points
- SH quadrature behaviour documented: equiangle grid uses trapezoidal theta integration (not Gauss-Legendre); large Fibonacci grids (~200 × (N+1)²) give ~1% quadrature approximation
- `complex_to_real_coeffs` / `real_to_complex_coeffs` roundtrip behaviour documented: `real → complex → real` is always lossless; `complex → real → complex` requires conjugate-symmetric inputs

## [0.1.0] — initial release

Initial Python port from MATLAB spherical array processing toolbox.

- Core SH basis and transform functions
- Spatial sampling: Fibonacci, equiangle grids
- Plane-wave array simulation
- Fixed beamforming: cardioid, hypercardioid, supercardioid, MaxEV
- Adaptive beamforming: MVDR, LCMV
- DOA: PWD and MUSIC spatial spectra
- Diffuseness: IE, TV, SV, CMD estimators
- Diffuse-field coherence models
- Spherical acoustics: Bessel/Hankel, modal coefficients
- Coordinate system utilities
- Plotting helpers
