# Contributing

Thanks for contributing.

## Development Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,image]"
```

## Local Validation

```bash
pytest -q
python -m pip install -U build
python -m build
```

## Pull Request Expectations

Keep changes focused and include tests for behavior changes.

Document API or behavior changes in `README.md` when applicable.

If a change affects `repro` fallback behavior, include a test that runs without external reference assets.

