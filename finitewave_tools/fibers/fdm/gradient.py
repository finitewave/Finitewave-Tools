
import numpy as np


def compute_gradient(mesh, normalize=False, mask=None):
    dx, dy, dz = np.gradient(mesh, edge_order=2)
    gradient = np.moveaxis(np.array([dx, dy, dz]), 0, -1)

    if mask is not None:
        gradient = gradient[mask]

    if normalize:
        gradient = normalize_vectors(gradient)

    return gradient


def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length vector encountered during normalization.")
    return vectors / norms
