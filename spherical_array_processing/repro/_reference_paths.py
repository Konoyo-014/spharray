from __future__ import annotations

import os
from pathlib import Path


def _package_reference_root() -> Path:
    return Path(__file__).resolve().parents[1] / "references"


def provider_reference_root(provider: str) -> Path:
    """Resolve provider reference directory with env override support.

    Priority:
    1) SAP_<PROVIDER>_REFERENCE_ROOT
    2) SAP_REFERENCE_ROOT/<provider>
    3) package-internal spherical_array_processing/references/<provider>
    """
    key = f"SAP_{provider.upper()}_REFERENCE_ROOT"
    specific = os.environ.get(key)
    if specific:
        return Path(specific).expanduser().resolve()

    generic = os.environ.get("SAP_REFERENCE_ROOT")
    if generic:
        return (Path(generic).expanduser().resolve() / provider)

    return _package_reference_root() / provider

