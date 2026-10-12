import numpy as np
from scipy import sparse


def magnetic_laplacian(K, G, elems, areas, vector_potential):
    """Assemble a magnetic Laplacian stiffness matrix on a triangular mesh.

    Parameters
    ----------
    K : scipy.sparse matrix, shape (N, N)
        Real, symmetric stiffness matrix for the negative Laplacian,
        assembled consistently with G and areas.
    G : ndarray, shape (E, D, 3)
        Gradients of the three linear basis functions on each of E
        triangles, expressed in D spatial dimensions. The final axis uses
        the same local node ordering as elems.
    elems : ndarray of int, shape (E, 3)
        Zero-based global node indices for each triangle.
    areas : ndarray, shape (E,)
        Positive triangle areas, consistent with the geometry used for G.
    vector_potential : ndarray, shape (E, D)
        Real vector potential A, treated as constant on each triangle.
        Components use the same spatial coordinate frame as G and have
        inverse-length units when the phase is dimensionless.

    Returns
    -------
    L : scipy.sparse matrix, shape (N, N)
        Complex magnetic stiffness matrix K + V + 1j*H. For real inputs
        and symmetric K, L is Hermitian. A zero vector potential recovers
        K, represented with a complex dtype.
    """
    basis_grad = G.transpose(0, 2, 1)

    M_local = (areas[:, None, None] / 12 * (np.ones((3, 3)) + np.eye(3)))

    vector_potential_sq = np.einsum("ed,ed->e", vector_potential, vector_potential)
    V_local = vector_potential_sq[:, None, None] * M_local

    q = np.einsum("eid,ed->ei", basis_grad, vector_potential)

    H_local = (areas[:, None, None] / 3 * (q[:, None, :] - q[:, :, None]))

    rows = np.broadcast_to(elems[:, :, None], (len(elems), 3, 3)).ravel()
    cols = np.broadcast_to(elems[:, None, :], (len(elems), 3, 3)).ravel()

    extra = sparse.coo_matrix(((V_local + 1j * H_local).ravel(), (rows, cols)), 
                              shape=K.shape).tocsr()

    L = K.astype(complex) + extra
    return L
