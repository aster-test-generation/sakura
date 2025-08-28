"""Evaluation utilities for nltest.

This package provides reusable grading utilities that can be used across
both nl2test and test2nl flows without duplication.
"""

from .test_grader import TestGrader

__all__ = ["TestGrader"]
