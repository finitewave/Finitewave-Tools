import numpy as np
import pyvista as pv


class PyVistaSurfaceGrid:
    """Class to hold a pyvista surface grid and its associated data.

    Attributes:
    ------------
    grid : pv.PolyData
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __new__(cls, coords, elems):
        """Build a PolyData from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.
        """
        faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
        poly = pv.PolyData(coords, faces)
        poly["indices"] = np.arange(elems.shape[0])
        return poly
