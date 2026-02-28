"""Library module.

Usage:
    from spherical_array_processing.regression import <symbol>
"""

from .matlab import (
    detect_matlab,
    detect_octave,
    matlab_available,
    run_matlab_batch,
    run_octave_eval,
)
from .status import CaseStatus

__all__ = [
    "CaseStatus",
    "detect_matlab",
    "detect_octave",
    "matlab_available",
    "run_matlab_batch",
    "run_octave_eval",
]
