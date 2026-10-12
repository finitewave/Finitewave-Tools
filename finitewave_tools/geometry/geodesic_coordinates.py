import numpy as np
from finitewave_tools.linalg.boundary_conditions import make_system
from finitewave_tools.linalg.magnetic_laplacian import magnetic_laplacian
from finitewave_tools.linalg.solvers import (
    linear_solver,
    smallest_eigenvector)


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


def geodesic_coords(K, M, G, elems, elem_sizes, normals, source_inds, 
                    reference_ind, num_dist_bins=100, t=100, atol=1e-8, maxiter=1000):
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
    num_dist_bins : int, optional
        Number of bins to use for histogramming the distances when estimating
        the angular scale. Defaults to 100.
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

    # radial_dist = geod_dist[elems].mean(axis=1)
    # phase_grad = circ_vec / np.maximum(radial_dist[:, None], epsilon)
    dist_scale = angular_scale(G, elems, elem_sizes, geod_dist, num_dist_bins)
    phase_grad = circ_vec / dist_scale[:, None]
    L = magnetic_laplacian(K, G, elems, elem_sizes, phase_grad)

    psi = smallest_eigenvector(L, M)
    theta_ref = np.angle(psi[reference_ind])
    psi *= np.exp(-1j * theta_ref)
    theta = np.angle(psi)
    return geod_dist, theta


def angular_scale(G, elems, elem_sizes, geod_dist, num_bins=50):
    """Estimate the angular scale of geodesic coordinates.

    Parameters
    ----------
    G : ndarray, shape (E, D, P)
        Operators mapping P local nodal values to D-component gradients
        on E elements.
    elems : ndarray of int, shape (E, P)
        Global node indices in the local ordering used by G.
    elem_sizes : ndarray, shape (E,)
        Element integration measures, such as areas or volumes.
    geod_dist : ndarray, shape (N,)
        Approximate geodesic distance at every node from the sources.
    num_bins : int, optional
        Number of bins to use for histogramming the distances. Defaults to 50.

    Returns
    -------
    angular_scales : ndarray, shape (E,)
        Estimated angular scale for each element, in radians. The scale is
        interpolated from the mean angular scales computed in for each distance bin.
    """
    element_dist = geod_dist[elems].mean(axis=1)
    grad = (G @ geod_dist[elems][..., None]).squeeze(-1)
    grad_norm = np.linalg.norm(grad, axis=1)

    distance_bins = np.linspace(0, np.max(element_dist), num=num_bins)
    bin_centers = 0.5 * (distance_bins[:-1] + distance_bins[1:])

    weighted_areas, _ = np.histogram(
        element_dist,
        bins=distance_bins,
        weights=elem_sizes * grad_norm,
    )

    mean_lengths = weighted_areas / np.diff(distance_bins)
    mean_angular_scales = mean_lengths / (2 * np.pi)

    angular_scales = np.interp(
        element_dist,
        bin_centers,
        mean_angular_scales,
        left=mean_angular_scales[0],
        right=mean_angular_scales[-1]
    )
    return angular_scales
