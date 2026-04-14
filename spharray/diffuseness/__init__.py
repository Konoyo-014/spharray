"""Diffuseness estimators for first-order ambisonic and SH-domain signals.

Examples
--------
>>> import numpy as np
>>> from spharray.diffuseness import (
...     intensity_vectors_from_foa, diffuseness_tv,
... )
>>> rng = np.random.default_rng(42)
>>> foa = rng.standard_normal((200, 4)) + 1j * rng.standard_normal((200, 4))
>>> I = intensity_vectors_from_foa(foa)
>>> Psi = diffuseness_tv(I)
>>> 0.0 <= Psi <= 1.0
True
"""

from .estimators import (
    diffuseness_cmd,
    diffuseness_ie,
    diffuseness_sv,
    diffuseness_tv,
    intensity_vectors_from_foa,
)

__all__ = [
    "diffuseness_cmd",
    "diffuseness_ie",
    "diffuseness_sv",
    "diffuseness_tv",
    "intensity_vectors_from_foa",
]
