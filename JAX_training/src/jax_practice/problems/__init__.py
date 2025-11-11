"""Problem registry."""

from __future__ import annotations

from typing import Dict

from .base import Problem
from .transpose import problem as transpose_problem

PROBLEMS: Dict[str, Problem] = {
    transpose_problem.slug: transpose_problem,
}

__all__ = ["Problem", "PROBLEMS"]
