
import numpy as np
from finitewave_tools.fibers.fdm.gradient import normalize_vectors
 

def rule_based_fibers(el, et, ec, alpha, beta):
    fibers = (np.cos(alpha[:, None]) * np.cos(beta[:, None]) * ec +
              np.sin(alpha[:, None]) * np.cos(beta[:, None]) * el +
              np.sin(beta[:, None]) * et)
    return normalize_vectors(fibers)


def fiber_angles(distance_map, alpha_endo=-60, alpha_epi=60):
    if distance_map.min() < 0 or distance_map.max() > 1:
        raise ValueError("Distance map must be normalized between 0 and 1.")

    alpha = distance_map * (alpha_epi - alpha_endo) + alpha_endo
    return alpha