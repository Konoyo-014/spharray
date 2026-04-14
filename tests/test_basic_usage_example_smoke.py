from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_basic_usage_example_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "examples" / "core" / "basic_usage.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Fibonacci grid" in result.stdout
    assert "SH matrix shape" in result.stdout
    assert "Hypercardioid" in result.stdout
    assert "B(0°)=1.0000" in result.stdout


def test_tutorial_examples_smoke() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = sorted((repo_root / "examples" / "tutorials").glob("*.py"))
    scripts = [script for script in scripts if script.name != "_bootstrap.py"]

    assert scripts
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "===" in result.stdout
