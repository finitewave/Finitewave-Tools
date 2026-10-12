import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres
import pyamg


def linear_solver(A, b, x0=None, solver=gmres, **kwargs):
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
    solver : callable, optional
        The linear solver to use. Defaults to `scipy.sparse.linalg.gmres`.
    **kwargs
        Additional keyword arguments to pass to the GMRES solver.

    Returns
    -------
    x : numpy.ndarray
        The solution vector.
    """
    ml = pyamg.smoothed_aggregation_solver(A)
    M_preconditioner = ml.aspreconditioner()

    x, info = solver(A, b, M=M_preconditioner, x0=x0, **kwargs)
    if info != 0:
        raise RuntimeError(f"Linear solver did not converge successfully (info={info}).")
    return x


def smallest_eigenvector(L, M):
    """Compute the eigenvector corresponding to the smallest eigenvalue of
    the generalized eigenproblem.

    Parameters
    ----------
    L : scipy.sparse matrix, shape (N, N)
        Hermitian magnetic Laplacian stiffness matrix.
    M : scipy.sparse matrix, shape (N, N)
        Positive-definite mass matrix for the generalized eigenproblem.

    Returns
    -------
    eigenvector : ndarray, shape (N,)
        The eigenvector corresponding to the smallest eigenvalue of the
        generalized eigenproblem L * psi = lambda * M * psi.
    """
    scale = np.max(np.abs(L.diagonal()) / M.diagonal())
    shift = -1e-6 * scale

    _, eigenvectors = sparse.linalg.eigsh(L, M=M, k=1, sigma=shift, which="LM")

    return eigenvectors[:, 0]
