import pyvista as pv
import numpy as np


class PyVistaSurfaceGrid(pv.PolyData):
    """Class to hold a pyvista surface grid and its associated data.

    Attributes:
    ------------
    grid : pv.PolyData
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, coords, elems):
        """Build a PolyData from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.
        """
        faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
        self.indices = np.arange(elems.shape[0])
        super().__init__(coords, faces)


class PyVistaTetraGrid(pv.UnstructuredGrid):
    """Class to hold a pyvista tetrahedral grid and its associated data.

    Attributes:
    ------------
    grid : pv.UnstructuredGrid
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, coords, elems, as_surface=False):
        """Build a Unstructured Grid from coords and elems.

        Parameters:
        ------------
        coords : np.array
            Coordinates of the mesh nodes.
        elems : np.array
            Elements of the mesh.

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        """
        faces = np.hstack([[elems.shape[1], *elem] for elem in elems])
        cell_types = np.full(elems.shape[0], pv.VTK_TETRA)
        grid = pv.UnstructuredGrid(faces, cell_types, coords)
        
        self.as_surface = as_surface
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.indices = grid.cell_data['idx']   
        super().__init__(grid)

    def __setitem__(self, name, value):
        if self.as_surface:
            self.cell_data[name] = value[*self.indices, ...]
        else:
            super().__setitem__(name, value)


class PyVistaMeshGrid(pv.UnstructuredGrid):
    """Class to hold a pyvista grid and its associated data.

    Attributes:
    ------------
    grid : pv.UnstructuredGrid
        Masked grid with cells where mesh > 0.
    indices : tuple of np.array
        Indices of the non-empty cells in the original mesh.
    """
    def __init__(self, mesh, as_surface=False):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).

        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        """
        self._mesh = mesh
        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        grid.cell_data['idx'] = np.arange(mesh.size)

        # Threshold the mesh to remove empty cells
        grid = grid.threshold(0.5)
        self.n_full_cells = grid.n_cells
        
        self.as_surface = as_surface
    
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")

        self.indices = np.unravel_index(grid.cell_data['idx'], mesh.shape, order='F')
        super().__init__(grid)
    
    def __setitem__(self, key, value):
        if value.shape[:3] == self._mesh.shape:
            self.cell_data[key] = value[*self.indices, ...]
        elif not self.as_surface and value.shape[0] == self.n_full_cells:
            value_full = np.zeros(self._mesh.shape + value.shape[1:], dtype=value.dtype)
            value_full[self._mesh > 0.5] = value
            self.cell_data[key] = value_full[*self.indices, ...]
        else:
            super().__setitem__(key, value)
                    
    def add_masked_scalar(self, scalars, mask, name='Scalars'):
        """Add a scalar field to the mesh. The scalars assumed to be
        values where mask is True.

        Parameters
        ----------
        scalars : np.array
            Flat scalar field.
        mask : np.array
            Boolean mask where scalars are defined.
        name : str, optional
            Name of the scalar. Default is 'Scalars'.

        Returns
        -------
        grid : pv.UnstructuredGrid
            Grid with the scalar field added as active cell scalars.
        """
        scalars_mesh = np.zeros_like(mask, dtype=scalars.dtype)
        scalars_mesh[mask] = scalars
        self.cell_data[name] = scalars_mesh[self.indices]
