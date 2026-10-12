import numpy as np
from finitewave_tools.linalg.boundary_conditions import make_system
from finitewave_tools.linalg.solvers import linear_solver
from .vectors import normalize_vectors


def laplace_coords(K, zero_ids, base_ids):
    """Compute Laplace coordinates between two boundaries
    
    Parameters
    ----------
    K : sparse matrix
        Stiffness matrix of the mesh.
    zero_ids : array-like
        Indices of the nodes where the Laplace coordinate is 0.
    base_ids : array-like
        Indices of the nodes where the Laplace coordinate is 1.

    Returns
    -------
    laplace_coords : ndarray
        The computed Laplace coordinates.
    """
    boundaries = [(zero_ids, 0), (base_ids, 1)]

    K_reduced, b_reduced, x0, inds = make_system(K, dirichlet_conditions=boundaries)

    laplace_coords = x0
    laplace_coords[inds] = linear_solver(K_reduced, b_reduced, x0=x0[inds])
    return laplace_coords


def laplace_direction(G, laplace_coords, out_mask=None):
    """Compute the gradient of Laplace coordinates
    
    Parameters
    ----------
    G : list of sparse matrices
        Gradient matrices for each spatial dimension.
    laplace_coords : ndarray
        The computed Laplace coordinates.
    out_mask : array-like, optional
        A boolean mask to filter the output. If provided, only the gradients
        corresponding to True values in the mask will be returned.

    Returns
    -------
    direction : ndarray
        The normalized gradient vectors of the Laplace coordinates.
    """
    if G[0].shape[0] != laplace_coords.shape[0]:
        raise ValueError("Mismatch between gradient matrix and Laplace coordinates dimensions.")

    if out_mask is not None and len(out_mask) != laplace_coords.shape[0]:
        raise ValueError("Mismatch between output mask and Laplace coordinates dimensions.")

    grad = np.stack([g @ laplace_coords for g in G], axis=-1)

    if out_mask is not None:
        grad = grad[out_mask]

    direction = normalize_vectors(grad)

    return direction


def coord_system(e0, e1):
    """
    Build a local coordinate system based on two vectors.
    
    Parameters:
    -----------
    e0 : array-like
        The first vector which direction will be preserved.
    e1 : array-like
        The second vector which will be corrected to be orthogonal to e0.

    Returns:
    --------
    e0, e1, e2 : array
        The orthonormal basis vectors forming the local coordinate system.
    """
    e0 = normalize_vectors(e0)
    e1 = correct_orthogonality(e0, e1, normalize=True)
    e2 = np.cross(e0, e1)
    e2 = normalize_vectors(e2)
    return e0, e1, e2


def correct_orthogonality(main_vector, secondary_vector, normalize=True):
    """
    Correct the secondary vector to be orthogonal to the main vector.
    
    Parameters:
    -----------
    main_vector : array-like
        The primary vector to which the secondary vector should be orthogonal.
    secondary_vector : array-like
        The vector to be corrected.

    Returns:
    --------
    corrected_vector : array
        The corrected secondary vector that is orthogonal to the main vector.
    """

    corrected_vector = (
        secondary_vector - 
        main_vector * np.sum(secondary_vector * main_vector, axis=1, keepdims=True)
    )

    if normalize:
        corrected_vector = normalize_vectors(corrected_vector)

    return corrected_vector