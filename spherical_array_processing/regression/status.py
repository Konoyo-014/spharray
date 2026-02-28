"""Library module.

Usage:
    from spherical_array_processing.regression.status import <symbol>
"""

from __future__ import annotations

from typing import Literal


CaseStatus = Literal["pass", "fail", "skip_dependency", "expected_difference", "error_matlab", "error_python"]

