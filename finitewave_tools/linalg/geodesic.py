
import numpy as np
from .linalg import make_system, linear_solver


def geodesic_distance(K, M, G, elems, elem_sizes, source_ids, t=100, atol=1e-8, 
                      maxiter=1000):
    """Estimate nodal distances from a set of source nodes.

    Parameters
    ----------
    K : scipy.sparse matrix, shape (N, N)
        Positive-semidefinite stiffness matrix for the negative Laplacian.
    M : scipy.sparse matrix, shape (N, N)
        Finite-element mass matrix, consistent with K.
    G : ndarray, shape (E, D, P)
        Element gradient operators mapping P local nodal values to a
        D-component spatial gradient on each of E elements.
    elems : ndarray of int, shape (E, P)
        Global node indices for each element, in the local ordering of G.
    elem_sizes : ndarray, shape (E,)
        Element integration measures, such as triangle areas or tetrahedral
        volumes, consistent with the stiffness and gradient operators.
    source_ids : ndarray of int, shape (S,)
        Nonempty set of distinct source node indices. Distances are fixed
        to zero at these nodes. Each connected component to be solved
        should contain a source.
    t : float, optional
        Positive diffusion time in squared mesh-length units. Defaults to
        100; it is not automatically scaled to the mesh resolution.
    atol : float, optional
        Absolute GMRES tolerance for the heat solve. Defaults to 1e-8.
    maxiter : int, optional
        GMRES iteration limit for the heat solve. Defaults to 1000.

    Returns
    -------
    distance : ndarray, shape (N,)
        Approximate distance at every node, in mesh-length units, with zero
        values at the sources. Values are not clipped to be nonnegative.

    References
    ----------
    Crane, K., Weischedel, C., and Wardetzky, M. (2013).
    Geodesics in Heat: A New Approach to Computing Distance Based on Heat Flow.
    Paper: https://www.cs.cmu.edu/~kmcrane/Projects/HeatMethod/paperTOG.pdf
    """
    direction = distance_direction(K, M, G, elems, source_ids, t=t, atol=atol, maxiter=maxiter)
    rhs = assemble_poisson_rhs(K, G, elems, elem_sizes, direction)

    K_reduced, rhs_reduced, x0, interior_indices = make_system(
        K, b=rhs, dirichlet_conditions=[(source_ids, 0.0)]
    )
    x0_reduced = x0[interior_indices]
    x0[interior_indices] = linear_solver(K_reduced, rhs_reduced, x0=x0_reduced, 
                                        atol=atol, maxiter=maxiter)
    
    return x0


def assemble_poisson_rhs(K, G, elems, elem_sizes, direction):
    """Assemble the weak Poisson load from an elementwise direction field.

    Parameters
    ----------
    K : scipy.sparse matrix, shape (N, N)
        Global stiffness matrix. Only its first dimension is used here
        to determine the number of nodes.
    G : ndarray, shape (E, D, P)
        Element basis-function gradients, with D spatial components and
        P local nodes per element.
    elems : array-like of int, shape (E, P)
        Global node indices in the local ordering used by G.
    elem_sizes : ndarray, shape (E,)
        Element integration measures, such as areas or volumes.
    direction : ndarray, shape (E, D)
        Vector field on the elements, normally obtained from
        distance_direction.

    Returns
    -------
    rhs : ndarray, shape (N,)
        Assembled nodal load. Contributions from shared nodes are summed.

    Notes
    -----
    Each element contributes elem_sizes[e] * G[e].T @ direction[e].
    """
    local_rhs = elem_sizes[..., None] * (G.transpose(0, 2, 1) @ 
                                         direction[..., None]).squeeze(-1)
    rhs = np.zeros(K.shape[0], dtype=float)
    np.add.at(rhs, np.asarray(elems).ravel(), local_rhs.ravel())
    return rhs


def distance_direction(K, M, G, elems, source_ids, t=100, atol=1e-8, maxiter=1000):
    """Compute elementwise directions pointing away from heat sources.

    Parameters
    ----------
    K : scipy.sparse matrix, shape (N, N)
        Positive-semidefinite stiffness matrix for the negative Laplacian.
    M : scipy.sparse matrix, shape (N, N)
        Finite-element mass matrix, consistent with K.
    G : ndarray, shape (E, D, P)
        Operators mapping P local nodal values to D-component gradients
        on E elements.
    elems : ndarray of int, shape (E, P)
        Global node indices in the local ordering used by G.
    source_ids : ndarray of int, shape (S,)
        Source node indices receiving unit entries in the load vector.
    t : float, optional
        Positive diffusion time in squared mesh-length units. Defaults
        to 100 without automatic mesh-dependent scaling.
    atol : float, optional
        Absolute GMRES tolerance. Defaults to 1e-8.
    maxiter : int, optional
        GMRES iteration limit. Defaults to 1000.

    Returns
    -------
    direction : ndarray, shape (E, D)
        Negative heat gradients normalized to unit length per element.
        Directions are undefined where the heat gradient is zero.


    Notes
    -----
    Solves (M + t*K) u = b, where b is zero except for unit source entries,
    then evaluates -grad(u) / norm(grad(u)). The source vector is used
    directly as a load; it is not multiplied by M. No zero-gradient guard
    is applied, so zero gradients produce invalid values on division.
    """
    A = M + t * K
    b = np.zeros(A.shape[0])
    b[source_ids] = 1.0

    u = linear_solver(A, b, atol=atol, maxiter=maxiter)

    grads = - (G @ u[elems][..., None]).squeeze(-1)
    direction = grads / np.linalg.norm(grads, axis=1)[:, None]
    return direction
