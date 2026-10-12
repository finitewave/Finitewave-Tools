"""Sparse systems and solvers"""

from .boundary_conditions import make_system
from .magnetic_laplacian import magnetic_laplacian
from .solvers import linear_solver, smallest_eigenvector

__all__ = [
    "make_system",
    "magnetic_laplacian",
    "linear_solver",
    "smallest_eigenvector",
]
