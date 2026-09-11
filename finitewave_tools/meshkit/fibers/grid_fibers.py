

from numba import np
from scipy import spatial
from finitewave_tools.meshkit.points.point_sampler import select_random_points



def compute_gradient(mesh, normalize=False, mask=None):
    dx, dy, dz = np.gradient(mesh, edge_order=2)
    gradient = np.array([dx, dy, dz])
    if mask is not None:
        gradient = gradient[:, mask].T
    if normalize:
        gradient = normalize_vectors(gradient)
    return gradient


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length vector encountered during normalization.")
    return vectors / norms


def sample_from_layers(layers, coords, radius, n_layers=None):
    """
    Sample points from each layer with a minimum distance constraint.
    
    Parameters:
    -----------
    layers : array-like
        An array indicating the layer index for each point in `coords`.
    coords : array-like
        An array of coordinates from which to sample points.
    radius : float
        The minimum distance between sampled points.
    n_layers : int, optional
        The number of layers to sample from. If None, it will be inferred from `layers`.

    Returns:
    --------
    sampled_indices : array
        Indices of the sampled points in the original `coords` array.
    """
    if n_layers is None:
        n_layers = np.max(layers) + 1

    full_indices = np.arange(coords.shape[0])
    sampled_indices = []

    for i in range(n_layers):
        layer_mask = layers == i
        layer_coords = coords[layer_mask]

        if layer_coords.size > 0:
            layer_indices = select_random_points(layer_coords, distance=radius)
            layer_indices = full_indices[layer_mask][layer_indices]
            sampled_indices.append(layer_indices)

    return np.concatenate(sampled_indices)


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


def build_coord_system(e0, e1):
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
    e1 = correct_orthogonality(e0, e1)
    e2 = np.cross(e0, e1)
    e2 = normalize_vectors(e2)
    return e0, e1, e2
