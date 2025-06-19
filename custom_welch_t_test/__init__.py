"""
Custom Welch's T-Test Library

A Python library for performing Welch's 2-sided t-test on dataframe data.
"""

from .core import welch_t_test
from .exceptions import WelchTTestError, InvalidDataError, InvalidColumnError

__version__ = "1.0.0"
__author__ = "Custom Library"

__all__ = ["welch_t_test", "WelchTTestError", "InvalidDataError", "InvalidColumnError"]