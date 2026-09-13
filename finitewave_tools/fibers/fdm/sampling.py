
import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersPoints import vtkPoissonDiskSampler


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


def select_random_points(points, distance, seed=12345):
    """
    Select random points from a given set of points using Poisson disk sampling.

    Parameters
    ----------
    points : numpy.ndarray
        The input points from which to select random points.
    distance : float
        The minimum distance between selected points.
    seed : int, optional
        The random seed for reproducibility. Default is 12345.

    Returns
    -------
    numpy.ndarray
        The indices of the selected random points in the original array.
    """
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(points.shape[0])
    shuffled_coords = points[shuffled_indices]
    poly = pv.PolyData(shuffled_coords)
    poly["original_indices"] = shuffled_indices

    # Assign the random sequence to the sampler
    points_sampler = vtkPoissonDiskSampler()
    points_sampler.SetRadius(distance)
    points_sampler.SetInputData(poly)
    # points_sampler.SetLocator()
    points_sampler.Update()
    out = pv.wrap(points_sampler.GetOutput())
    return np.array(out["original_indices"])