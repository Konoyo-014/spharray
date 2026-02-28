# Spherical Array Processing (Pythonic Open Source Edition)

`spherical-array-processing` is a self-contained Python toolkit for spherical array processing, SH-domain transforms, beamforming, DOA spectra, and diffuseness estimation.

This repository is intentionally packaged as an independently installable open-source project with no required external source trees.

## Python Requirement

This package targets **Python 3.11+**.

## Quick Start

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,image]"
pytest -q
```

## Packaging Check

```bash
python -m pip install -U build
python -m build
pip install --force-reinstall dist/*.whl
python -c "from spherical_array_processing.toolkit import sht; print(sht.getFliegeNodes(2)[0].shape)"
```

## Repository Structure

The open-source repository now keeps a small layered structure:

`spherical_array_processing/` contains the installable library.

`tests/` is layered by purpose (`core/`, `integration/`, `plotting/`, `experimental/`).

`scripts/experimental/` contains optional FOA training/evaluation helper scripts.

## Self-Contained Toolkit Layer

The `toolkit` modules are designed to run directly from package code and packaged resources.

`toolkit.sht.getFliegeNodes` always provides deterministic fallback sampling when optional data files are unavailable.

For `getTdesign`, loading an optional packaged `.mat` lookup table is opt-in:

```bash
export SAP_USE_RESOURCE_MAT=1
```

Without this variable, deterministic internal fallback is used by default.

## License

MIT. See [LICENSE](LICENSE).
