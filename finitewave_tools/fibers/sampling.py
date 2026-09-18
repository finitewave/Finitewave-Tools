import numpy as np
from finitewave_tools.meshkit.points.point_sampler import select_random_points


def sample_from_layers(layers, coords, radius, n_layers=None):
    """
    Sample points with a minimum distance constraint within each layer.

    Points from different layers may be closer than radius.
    
    Parameters:
    -----------
    layers : array-like
        Integer array of shape (N,) with zero-based layer indices.
    coords : array-like
        Array of shape (N, 3) containing point coordinates.
    radius : float
        The minimum distance between sampled points.
    n_layers : int, optional
        The number of layers to sample from. If None, it will be inferred from `layers`.

    Returns:
    --------
    sampled_indices : array
        Indices of the sampled points in the original `coords` array.
        An empty integer array is returned when no points are sampled.
    """
    layers = np.asarray(layers)
    coords = np.asarray(coords)

    if layers.size == 0:
        return np.empty(0, dtype=np.intp)

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

    if not sampled_indices:
        return np.empty(0, dtype=np.intp)

    return np.concatenate(sampled_indices)
