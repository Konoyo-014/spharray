# Spherical Array Processing (Pythonic Open Source Edition)

`spherical-array-processing` is a Python toolkit for spherical array processing, SH-domain transforms, beamforming, DOA spectra, diffuseness estimation, and reproducible reference-style workflows.

This repository is intentionally packaged as an independently installable open-source project. It does **not** include large third-party MATLAB source trees under `src/`.

## Python Requirement

This package targets **Python 3.11+**.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
pytest -q
```

## Packaging Check

```bash
python -m pip install -U build
python -m build
pip install --force-reinstall dist/*.whl
python -c "from spherical_array_processing.repro import sht; print(sht.getFliegeNodes(2)[0].shape)"
```

## Repro Layer Behavior Without External Assets

The `repro` modules are designed to remain callable even when external MATLAB reference assets are absent.

`repro.sht.getFliegeNodes` now uses deterministic fallback sampling when optional `.mat` references are unavailable.

`POLITIS_SOURCE_ROOT` and `RAFAELY_SOURCE_ROOT` point to package-resolvable reference directories (and can be overridden via environment variables).

## Optional Reference Root Overrides

You can point repro modules to external reference directories:

```bash
export SAP_REFERENCE_ROOT=/path/to/reference_roots
# or provider-specific
export SAP_POLITIS_REFERENCE_ROOT=/path/to/politis
export SAP_RAFAELY_REFERENCE_ROOT=/path/to/rafaely
export SAP_SHT_REFERENCE_ROOT=/path/to/sht
```

For `getTdesign`, loading optional MATLAB `t_designs_1_21.mat` is opt-in:

```bash
export SAP_USE_REFERENCE_MAT=1
```

Without this variable, deterministic internal fallback is used by default.

## License

MIT. See [LICENSE](LICENSE).

