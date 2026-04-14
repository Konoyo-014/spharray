"""Diffuse-field spatial coherence models.

Examples
--------
>>> import numpy as np
>>> from spharray.coherence import diffuse_coherence_matrix_omni
>>> xyz = np.random.randn(8, 3) * 0.05   # 8 sensors, ~5 cm spread
>>> G = diffuse_coherence_matrix_omni(xyz, np.linspace(200, 4000, 50))
>>> G.shape
(50, 8, 8)
"""

from .diffuse import diffuse_coherence_from_weights, diffuse_coherence_matrix_omni

__all__ = [
    "diffuse_coherence_from_weights",
    "diffuse_coherence_matrix_omni",
]
