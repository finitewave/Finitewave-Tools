import numpy as np
from finitewave_tools.fibers.fdm.gradient import normalize_vectors


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


# def correct_transmural_direction(el, et):
#     et_corr = et - np.sum(et * el, axis=1, keepdims=True) * el
#     et_corr = normalize_vectors(et_corr)
#     return et_corr