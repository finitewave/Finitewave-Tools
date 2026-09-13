import numpy as np
import pyamg
from scipy.sparse.linalg import gmres
from .linalg import make_system, linear_solver


def geodesic_distance(K, M, G, elems, elem_sizes, source_ids, t=100, atol=1e-8, 
                      maxiter=1000):
    direction = distance_direction(K, M, G, elems, source_ids, t=t, atol=atol, maxiter=maxiter)
    rhs = assemble_poisson_rhs(K, G, elems, elem_sizes, direction)

    K_reduced, rhs_reduced, x0, interior_indices = make_system(
        K, b=rhs, dirichlet_conditions=[(source_ids, 0.0)]
    )
    x0_reduced = x0[interior_indices]
    x0[interior_indices] = linear_solver(K_reduced, rhs_reduced, x0=x0_reduced, 
                                        atol=1e-8, maxiter=1000)
    
    return x0


def assemble_poisson_rhs(K, G, elems, elem_sizes, direction):
    local_rhs = elem_sizes[..., None] * (G.transpose(0, 2, 1) @ 
                                         direction[..., None]).squeeze(-1)
    rhs = np.zeros(K.shape[0], dtype=float)
    np.add.at(rhs, np.asarray(elems).ravel(), local_rhs.ravel())
    return rhs


def distance_direction(K, M, G, elems, source_ids, t=100, atol=1e-8, maxiter=1000):
    A = M + t * K
    b = np.zeros(A.shape[0])
    b[source_ids] = 1.0

    u = linear_solver(A, b, atol=atol, maxiter=maxiter)

    grads = - (G @ u[elems][..., None]).squeeze(-1)
    direction = grads / np.linalg.norm(grads, axis=1)[:, None]
    return direction
