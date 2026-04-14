# Contributing

Thanks for contributing to `spharray`.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,image]"
```

Install notebook dependencies only when editing notebook tutorials:

```bash
python -m pip install -e ".[notebook]"
```

## Local Validation

```bash
python -m pytest -q --tb=short
python examples/core/basic_usage.py
python examples/tutorials/01_sht_and_beamforming.py
python examples/tutorials/02_simulated_doa_pipeline.py
python examples/tutorials/03_modal_equalization_pipeline.py
python -m build
```

`python examples/core/basic_usage.py` should print `B(0°)=1.0000`. That value
is a useful smoke check because it verifies that the local checkout is imported
and fixed beamformer normalization is correct.

## Pull Request Expectations

Keep changes focused and include tests for behavior changes.

Document API or behavior changes in `README.md` and `CHANGELOG.md` when
applicable. If a change affects concepts that new users need to understand,
update `docs/getting_started.md` or `docs/concepts.md`.

If a change affects `toolkit` fallback behavior, include a test that runs without optional resource files.
