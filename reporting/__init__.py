"""
Reporting Package - Visualization and report generation.

This package provides modules for:
- Equity curve visualization
- Drawdown charts
- Edge performance reports
- HTML report generation
"""

from .visualizer import EdgeVisualizer
from .html_report import HTMLReportGenerator

__all__ = [
    "EdgeVisualizer",
    "HTMLReportGenerator"
]
