
import numpy as np
from finitewave_tools.fibers.vectors import normalize_vectors
 

def rule_based_fibers(el, et, ec, alpha, beta):
    """Construct unit fiber directions from a local basis and two angles.

    Parameters
    ----------
    el : ndarray, shape (N, 3)
        Local longitudinal unit vectors.
    et : ndarray, shape (N, 3)
        Local transmural unit vectors.
    ec : ndarray, shape (N, 3)
        Local circumferential unit vectors. The three input directions are
        expected to form an orthonormal basis at each point.
    alpha : ndarray, shape (N,)
        In-plane angles in radians, measured from ec toward el.
    beta : ndarray, shape (N,)
        Out-of-plane tilt angles in radians, measured toward et.

    Returns
    -------
    fibers : ndarray, shape (N, 3)
        Unit fiber vectors in the same coordinate frame as the input basis.

    Notes
    -----
    The direction before normalization is
    cos(beta) * (cos(alpha) * ec + sin(alpha) * el) + sin(beta) * et.
    With both angles zero, fibers follow ec; positive alpha rotates toward
    el, and positive beta tilts toward et.

    Bayer et al. [1]_ describe a related rule-based myocardial orientation
    method. This function only assembles directions from a supplied basis;
    it does not implement the full Laplace-Dirichlet algorithm. Here beta
    tilts the fiber itself, whereas the paper defines beta for the transverse
    fiber direction relative to the transmural axis.

    References
    ----------
    .. [1] Bayer, J. D., Blake, R. C., Plank, G., and Trayanova, N. A.
       (2012). A Novel Rule-Based Algorithm for Assigning Myocardial Fiber
       Orientation to Computational Heart Models. Annals of Biomedical
       Engineering, 40, 2243-2254.
       https://doi.org/10.1007/s10439-012-0593-5


    Examples
    --------
    >>> el = np.array([[1, 0, 0], [0, 1, 0]])
    >>> et = np.array([[0, 1, 0], [0, 0, 1]])
    >>> ec = np.array([[0, 0, 1], [1, 0, 0]])
    >>> alpha = np.array([0, np.pi/2])
    >>> beta = np.array([0, 0])
    >>> fibers = rule_based_fibers(el, et, ec, alpha, beta)
    >>> fibers
    array([[0., 0., 1.],
           [0., 1., 0.]])
    """
    fibers = (np.cos(alpha[:, None]) * np.cos(beta[:, None]) * ec +
              np.sin(alpha[:, None]) * np.cos(beta[:, None]) * el +
              np.sin(beta[:, None]) * et)
    return normalize_vectors(fibers)


def fiber_angles(distance_map, alpha_endo=-60, alpha_epi=60):
    """Linearly interpolate fiber angles across the myocardial wall.

    Parameters
    ----------
    distance_map : ndarray
        Nonempty normalized transmural coordinates: 0 at the endocardium
        and 1 at the epicardium.
    alpha_endo : float, optional
        Endocardial angle. Defaults to -60 degrees.
    alpha_epi : float, optional
        Epicardial angle, in the same units as alpha_endo. Defaults to
        60 degrees.

    Returns
    -------
    alpha : ndarray
        Interpolated angles with the same shape as distance_map and the
        same units as the endpoint angles. No unit conversion is performed.

    Notes
    -----
    Convert the default degree-valued output with np.deg2rad before passing
    it to rule_based_fibers, which expects radians and one angle per point.
    The anatomical meaning of angle signs depends on the supplied basis
    directions; the defaults specify this implementation's endpoint values.

    Examples
    --------
    >>> angles = fiber_angles(np.array([0.0, 0.5, 1.0]))
    >>> angles
    array([-60.,   0.,  60.])
    >>> alpha = np.deg2rad(angles)
    """
    if distance_map.min() < 0 or distance_map.max() > 1:
        raise ValueError("Distance map must be normalized between 0 and 1.")

    if abs(alpha_endo) > 90 or abs(alpha_epi) > 90:
        raise ValueError("Fiber angles should be within [-90, 90] degrees.")

    alpha = distance_map * (alpha_epi - alpha_endo) + alpha_endo
    alpha = np.radians(alpha)
    return alpha
