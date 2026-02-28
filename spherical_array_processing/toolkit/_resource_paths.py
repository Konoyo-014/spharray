from __future__ import annotations

from pathlib import Path


def _package_resource_root() -> Path:
    return Path(__file__).resolve().parents[1] / "resources"


def provider_resource_root(provider: str) -> Path:
    """Return the package-internal resource directory for a provider namespace."""
    root = _package_resource_root() / provider
    root.mkdir(parents=True, exist_ok=True)
    return root
