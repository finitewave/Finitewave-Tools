import numpy as np

from finitewave_tools.fibers.vectors import normalize_vectors


def compute_gradient(mesh, normalize=False, mask=None, spacing=1.0):
    """Compute the gradient of a three-dimensional scalar field.

    Parameters
    ----------
    mesh : array-like, shape (nx, ny, nz)
        Scalar field with at least three cells along each axis, as required
        by second-order boundary differences.
    normalize : bool, optional
        Normalize each output vector to unit length. Zero-length vectors
        raise ValueError.
    mask : ndarray of bool, shape (nx, ny, nz), optional
        Select cells before normalization.
    spacing : float or sequence of three floats, optional
        Physical cell spacing, either uniform or per axis. Defaults to 1.

    Returns
    -------
    ndarray
        Gradient components in array-axis order, with shape (nx, ny, nz, 3)
        or (N, 3) when a mask selects N cells.
    """
    mesh = np.asarray(mesh)
    if mesh.ndim != 3 or any(size < 3 for size in mesh.shape):
        raise ValueError("mesh must be three-dimensional with at least three cells per axis.")
    spacing = np.asarray(spacing, dtype=float)
    if spacing.ndim == 0:
        spacing = np.repeat(spacing, 3)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing) & (spacing > 0)):
        raise ValueError("spacing must be a positive finite scalar or three positive finite values.")

    gradient = np.stack(np.gradient(mesh, *spacing, edge_order=2), axis=-1)

    if mask is not None:
        gradient = gradient[mask]

    if normalize:
        gradient = normalize_vectors(gradient)

    return gradient
