import numpy as np
from scipy.sparse.linalg import gmres
import pyamg


def make_system(A, b=None, x0=None, dirichlet_conditions=None):
    """
    Assemble the system Ax = b with Dirichlet boundary conditions applied.

    Parameters
    ----------
    A : scipy.sparse matrix
        The stiffness matrix A of the system.
    b : array_like, optional
        The assembled load vector. Defaults to zero (Laplace equation).
        For FEM, supply the assembled load, not raw nodal source values.
    x0 : array_like, optional
        Initial guess. Defaults to zero and is not modified.
    dirichlet_conditions : list of tuples, optional
        Each tuple contains an array of node indices and their prescribed
        value or values. Defaults to no prescribed nodes.
    
    Returns
    -------
    A_reduced : scipy.sparse matrix
        The reduced stiffness matrix after applying Dirichlet conditions.
    b_reduced : numpy.ndarray
        The reduced load vector after applying Dirichlet conditions.
    x0 : numpy.ndarray
        The solution vector with Dirichlet values applied.
    interior_indices : numpy.ndarray
        The indices of the interior nodes (not on the Dirichlet boundary).

    Notes
    -----
    The x0 vector is modified in place to include the Dirichlet values. And
    ``A_reduced.shape[0] == b_reduced.shape[0] != x0.shape[0]``.

    """
    if dirichlet_conditions is None:
        dirichlet_conditions = []

    if b is None and len(dirichlet_conditions) == 0:
        raise ValueError("Either ``b`` or ``dirichlet_conditions`` must be provided.")

    if x0 is None:
        x0 = np.zeros(A.shape[0], dtype=float)

    if b is None:
        b = np.zeros(A.shape[0], dtype=float)
    else:
        b = np.asarray(b, dtype=float)

    dirichlet_indices = _set_dirichlet_values(x0, dirichlet_conditions)
    interior_indices = _find_interior_indices(A.shape[0], dirichlet_indices)
    A_reduced, b_reduced = _build_reduced_system(A, b, x0, interior_indices, 
                                                 dirichlet_indices)
    
    return A_reduced, b_reduced, x0, interior_indices


def linear_solver(A, b, x0=None, **kwargs):
    """
    Solve the linear system Ax = b using GMRES with an AMG preconditioner.

    Parameters
    ----------
    A : scipy.sparse matrix
        The stiffness matrix of the system.
    b : numpy.ndarray
        The right-hand side vector of the system.
    x0 : numpy.ndarray, optional
        The initial guess for the solution. Defaults to zero.
    **kwargs
        Additional keyword arguments to pass to the GMRES solver.

    Returns
    -------
    x : numpy.ndarray
        The solution vector.
    """
    ml = pyamg.smoothed_aggregation_solver(A)
    M_preconditioner = ml.aspreconditioner()

    x, info = gmres(A, b, M=M_preconditioner, x0=x0, **kwargs)
    if info != 0:
        raise RuntimeError(f"GMRES did not converge successfully (info={info}).")
    return x


def _find_interior_indices(size, boundary_indices):
    """
    Find the indices of the interior nodes (not on the Dirichlet boundary).

    Parameters
    ----------
    size : int
        The total number of nodes in the system.
    boundary_indices : numpy.ndarray
        The indices of the nodes with Dirichlet boundary conditions.
    
    Returns
    -------
    interior_indices : numpy.ndarray
        The indices of the interior nodes (not on the Dirichlet boundary).
    """
    mask = np.ones(size, dtype=bool)
    mask[boundary_indices] = False
    return np.flatnonzero(mask)


def _build_reduced_system(A, b, x, interior_indices, boundary_indices):
    """
    Build the reduced system of equations after applying Dirichlet boundary conditions.

    Parameters
    ----------
    A : scipy.sparse matrix
        The stiffness matrix of the system.
    b : numpy.ndarray
        The right-hand side vector of the system.
    x : numpy.ndarray
        The solution vector with Dirichlet values applied.
    interior_indices : numpy.ndarray
        The indices of the interior nodes (not on the Dirichlet boundary).
    boundary_indices : numpy.ndarray
        The indices of the nodes with Dirichlet boundary conditions.

    Returns
    -------
    A_reduced : scipy.sparse matrix
        The reduced stiffness matrix after applying Dirichlet conditions.
    b_reduced : numpy.ndarray
        The reduced right-hand side vector after applying Dirichlet conditions.
    """
    A_reduced = A[interior_indices][:, interior_indices]
    A_boundary = A[interior_indices][:, boundary_indices]

    b_reduced = b[interior_indices] - A_boundary @ x[boundary_indices]

    return A_reduced, b_reduced

def _set_dirichlet_values(x, dirichlet_conditions):
    """
    Apply Dirichlet boundary conditions to the solution vector.
    
    Parameters
    ----------
    x : numpy.ndarray
        The solution vector to which the boundary conditions will be applied.
    dirichlet_conditions : list of tuples
        A list of tuples where each tuple contains an index and the corresponding Dirichlet value.

    Returns
    -------
    dirichlet_indices : numpy.ndarray
        The indices of the nodes with Dirichlet boundary conditions.
    """
    if len(dirichlet_conditions) == 0:
        return np.array([], dtype=int)

    dirichlet_indices = np.concatenate([index for index, _ in dirichlet_conditions], dtype=int)

    if not _has_unique_indices(dirichlet_indices):
        raise ValueError("Duplicate Dirichlet boundary indices found.")

    for index, value in dirichlet_conditions:
        x[index] = value

    return dirichlet_indices


@staticmethod
def _has_unique_indices(indices):
    return np.unique(indices).size == indices.size
