"""
Validation Package - Statistical testing and validation.

This package provides modules for:
- Statistical hypothesis tests
- Walk-forward validation
- Bootstrap confidence intervals
- Multiple testing corrections
"""

from .statistical_tests import StatisticalTests
from .walk_forward import WalkForwardValidator
from .bootstrap import BootstrapValidator

__all__ = [
    "StatisticalTests",
    "WalkForwardValidator", 
    "BootstrapValidator"
]
