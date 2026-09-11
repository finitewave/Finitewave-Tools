import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersPoints import vtkPoissonDiskSampler


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