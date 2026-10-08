
import numpy as np
from scipy import sparse
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
    direction = geodesic_direction(K, M, G, elems, source_ids, t=t, atol=atol, maxiter=maxiter)
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
        geodesic_direction.

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


def geodesic_direction(K, M, G, elems, source_ids, t=100, atol=1e-8, maxiter=1000):
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


def assemble_magnetic_laplacian(K, G, elems, areas, circ, geo_dist, epsilon=1e-3):
    """Assemble a complex magnetic Laplacian on a triangular mesh.

    Parameters
    ----------
    K : scipy.sparse matrix, shape (N, N)
        Real stiffness matrix for the negative Laplacian.
    G : ndarray, shape (E, D, 3)
        Basis-function gradients for E triangles in D spatial dimensions.
    elems : ndarray of int, shape (E, 3)
        Global node indices in the local ordering used by G.
    areas : ndarray, shape (E,)
        Triangle areas.
    circ : ndarray, shape (E, D)
        Elementwise circumferential direction vectors.
    geo_dist : ndarray, shape (N,)
        Nodal geodesic distances used to scale the vector potential.
    epsilon : float, optional
        Positive lower bound for the mean distance on each triangle.
        Defaults to 1e-3, in the same units as geo_dist.

    Returns
    -------
    L : scipy.sparse matrix, shape (N, N)
        Complex stiffness matrix including the vector-potential terms.

    Notes
    -----
    The vector potential is circ divided by the element's mean geodesic
    distance, clamped below by epsilon. Linear triangle basis functions
    are used to assemble its squared-magnitude and imaginary coupling terms.
    """
    r_elem = geo_dist[elems].mean(axis=1)
    phase_grad = circ / np.maximum(r_elem[:, None], epsilon)
    basis_grad = G.transpose(0, 2, 1)

    M_local = (areas[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3)))

    phase_grad_sq = np.einsum("ed,ed->e", phase_grad, phase_grad)
    V_local = phase_grad_sq[:, None, None] * M_local

    q = np.einsum("eid,ed->ei", basis_grad, phase_grad)

    H_local = (areas[:, None, None] / 3 * (q[:, None, :] - q[:, :, None]))

    N = K.shape[0]

    rows = np.broadcast_to(elems[:, :, None], (len(elems), 3, 3)).ravel()
    cols = np.broadcast_to(elems[:, None, :], (len(elems), 3, 3)).ravel()

    extra = sparse.coo_matrix(((V_local + 1j * H_local).ravel(), (rows, cols)), 
                              shape=(N, N)).tocsr()

    L = K.astype(complex) + extra
    return L


def solve_magnetic_laplacian(L, M, ref_point_id):
    """Extract a reference-aligned phase from a magnetic eigenmode.

    Parameters
    ----------
    L : scipy.sparse matrix, shape (N, N)
        Hermitian magnetic Laplacian stiffness matrix.
    M : scipy.sparse matrix, shape (N, N)
        Positive-definite mass matrix for the generalized eigenproblem.
    ref_point_id : int
        Node index used to align the eigenvector's phase to zero.
        Its eigenvector entry must be nonzero for meaningful alignment.

    Returns
    -------
    theta : ndarray, shape (N,)
        Nodal phase angles in radians, in the interval [-pi, pi], aligned
        to the reference node. Phase is undefined at zero-magnitude entries.

    Notes
    -----
    Solves L psi = lambda M psi for the eigenvalue nearest a small negative
    shift, scaled by the largest absolute stiffness-to-mass diagonal ratio.
    For a positive-semidefinite L, this targets the lowest eigenmode.
    """
    scale = np.max(np.abs(L.diagonal()) / M.diagonal())
    shift = -1e-6 * scale

    _, eigenvectors = sparse.linalg.eigsh(
        L,
        M=M,
        k=1,
        sigma=shift,
        which="LM",
    )

    psi = eigenvectors[:, 0]
    psi *= np.exp(-1j * np.angle(psi[ref_point_id]))
    theta = np.angle(psi)
    return theta


def geodesic_coords(K, M, G, elems, elem_sizes, normals, source_inds, 
                    reference_ind, t=100, atol=1e-8, maxiter=1000):
    """Compute geodesic distances and polar phase for a given source and reference point.

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
    elem_sizes : ndarray, shape (E,)
        Element integration measures, such as areas or volumes.
    normals : ndarray, shape (E, D)
        Unit normal vectors for each element, consistent with the ordering of G.
    source_inds : int or array of int, shape (S,)
        Source node indices receiving a unit entry in the load vector.
    reference_ind : int
        Reference node index for phase alignment. The phase is rotated so
        that its value at this node is zero.
    t : float, optional
        Positive diffusion time in squared mesh-length units. Defaults to 100.
    atol : float, optional
        Absolute GMRES tolerance. Defaults to 1e-8.
    maxiter : int, optional
        GMRES iteration limit. Defaults to 1000.

    Returns
    -------
    distance : ndarray, shape (N,)
        Approximate geodesic distance at every node from the sources.
    theta : ndarray, shape (N,)
        Polar phase at every node aligned to be zero at the reference point.

    Notes
    -----
    The geodesic distance is computed using the heat method. The polar phase
    is computed by solving a magnetic Laplacian eigenproblem with a vector
    potential derived from the geodesic distance gradient. The phase is then
    aligned to be zero at the specified reference point.

    References
    ----------
    Banduc, T., Pezzuto, S., and Costabal, F.S. (2026).
    LogMap: Geodesic Polar Coordinates Parameterization with the Magnetic Laplacian.
    Paper: https://doi.org/10.48550/arXiv.2609.10503
    """
    radial_vec = geodesic_direction(K, M, G, elems, source_inds)
    circ_vec = np.cross(radial_vec, normals)
    circ_vec /= np.linalg.norm(circ_vec, keepdims=True, axis=1)
    geod_dist = geodesic_distance(K, M, G, elems, elem_sizes, source_inds,
                                  t=t, atol=atol, maxiter=maxiter)
    L = assemble_magnetic_laplacian(K, G, elems, elem_sizes, circ_vec, geod_dist)
    geod_phase = solve_magnetic_laplacian(L, M, reference_ind)
    return geod_dist, geod_phase
    
