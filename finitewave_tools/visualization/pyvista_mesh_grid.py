import pyvista as pv
import numpy as np


class PyVistaMeshGrid(pv.UnstructuredGrid):
    """Class to hold a pyvista grid and its associated data.

    Use `from_mesh` to construct grid from a 3D mesh with empty space.

    Attributes:
    ------------
    mesh : np.array
        3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0), and fibrosis (elems = 2).
    raw_grid : pv.UnstructuredGrid
        Unstructured grid with all cells, including empty ones.
    as_surface : bool
        If True, build a surface mesh. Default is False.
    threshold : float
        Threshold value for the mesh. Default is 0.5.
    n_nonzero_cells : int
        Number of non-empty cells in the original mesh.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mesh = None
        self.raw_grid = None
        self.as_surface = False
        self.threshold = 0.5
        self.n_nonzero_cells = None
    
    @classmethod
    def from_mesh(cls, mesh, dr=1, as_surface=False, threshold=0.5,
                  dx=None, dy=None, dz=None):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Parameters:
        ------------
        mesh : np.array
            3D mesh with cardiomyocytes (elems = 1), empty space (elems = 0),
            and fibrosis (elems = 2).
        dr : float, optional
            Spacing in the mesh. Default is 1.
        as_surface : bool, optional
            If True, build a surface mesh. Default is False.
        threshold : float, optional
            Threshold value for the mesh. Default is 0.5.
        dx : float, optional
            Spacing in the x-direction. Default is None.
        dy : float, optional
            Spacing in the y-direction. Default is None.
        dz : float, optional
            Spacing in the z-direction. Default is None.
        """

        mesh = np.atleast_3d(mesh)
        grid = cls._build_grid(mesh, dr=dr, dx=dx, dy=dy, dz=dz)
        grid = cls._apply_threshold(grid, mesh, threshold)
        n_nonzero_cells = grid.n_cells
        
        if as_surface:
            grid = grid.extract_surface(algorithm="geometry")
 
        instance = cls(grid)
        instance.mesh = mesh
        instance.raw_grid = grid
        instance.as_surface = as_surface
        instance.threshold = threshold
        instance.n_nonzero_cells = n_nonzero_cells

        return instance
    
    @staticmethod
    def _build_grid(mesh, dr=1, dx=None, dy=None, dz=None):
        if dx is None or dy is None or dz is None:
            dx = dy = dz = dr

        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (dx, dy, dz)
        grid.origin = - np.array([dx, dy, dz]) * 0.5
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        c_inds = np.arange(mesh.size).reshape(mesh.shape)
        grid.cell_data['source_cell_ids'] = c_inds.flatten(order='F')
        return grid
    
    @staticmethod
    def _apply_threshold(grid, mesh, threshold=0.5):
        grid = grid.threshold(threshold, scalars='mesh', invert=False)
        c_source_cell_ids = - np.ones(mesh.shape, dtype=int)
        c_source_cell_ids[mesh > threshold] = np.arange(grid.n_cells)
        non_zero_inds = np.unravel_index(grid.cell_data['source_cell_ids'], mesh.shape, order='C')
        grid.cell_data['nonzero_inds'] = c_source_cell_ids[*non_zero_inds]
        return grid
    
    def __setitem__(self, key, value):
        if self.mesh is None or "nonzero_inds" not in self.cell_data:
            super().__setitem__(key, value)
            return

        value = np.asarray(value)

        if value.shape[:3] == self.mesh.shape:
            inds = np.unravel_index(self.cell_data['source_cell_ids'], self.mesh.shape, order='C')
            self.cell_data[key] = value[*inds, ...]
        elif value.shape[0] == self.mesh.size:
            self.cell_data[key] = value[self.cell_data['source_cell_ids'], ...]
        elif value.shape[0] == self.n_nonzero_cells:
            self.cell_data[key] = value[self.cell_data['nonzero_inds'], ...]
        else:
            super().__setitem__(key, value)

    @property
    def mesh_indices(self):
        """Return the indices of the non-empty cells in the original mesh."""
        if self.mesh is None or "source_cell_ids" not in self.cell_data:
            raise ValueError("Mesh is not set or 'source_cell_ids' not in cell_data.")

        indices = np.unique(self.cell_data['source_cell_ids'])
        return indices

    @property
    def mesh_coords(self):
        """Return the coordinates of the non-empty cells in the original mesh."""
        if self.mesh is None or "source_cell_ids" not in self.cell_data:
            raise ValueError("Mesh is not set or 'source_cell_ids' not in cell_data.")

        indices = self.mesh_indices
        coords = np.array(np.unravel_index(indices, self.mesh.shape)).T
        return coords
