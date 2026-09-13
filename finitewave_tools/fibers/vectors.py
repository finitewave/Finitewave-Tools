import numpy as np


def normalize_vectors(vectors):
    """Normalize vectors along the last axis of an array with shape (..., 3).

    Return an array of the same shape. Raise ValueError for zero-length vectors.

    Parameters
    ----------
    vectors : array-like, shape (..., 3)
        Input vectors to be normalized.

    Returns
    -------
    ndarray
        Normalized vectors with the same shape as the input.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-length vector encountered during normalization.")
    return vectors / norms
