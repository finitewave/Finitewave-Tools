"""Build element grids with scalar assignment in original element order."""

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkGenericCell


class PyVistaElemGrid(pv.UnstructuredGrid):
    """A grid whose item assignment accepts one value per original element.

    Construct from coordinates and connectivity with ``from_elements``.
    Use ``point_data`` for node values and ``cell_data`` for output-cell values.
    Surface extraction preserves the original element IDs for scalar mapping.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_original_points = None
        self.n_original_cells = None
        self.as_surface = False

    @classmethod
    def from_elements(cls, coords, elems, cell_type, as_surface=False):
        """Build a homogeneous or mixed mesh from element connectivity.

        Parameters
        ----------
        coords : array_like, shape (n_points, 3)
            Finite node coordinates.
        elems : sequence of sequences of int
            Zero-based node indices for each element, in VTK node order.
            Rows may have different lengths for mixed meshes.
        cell_type : str, int, pyvista.CellType, or sequence of these
            One type for all elements, or one per element. Strings are
            case-insensitive PyVista enum names, such as ``'triangle'``,
            ``'quad'``, ``'tetra'``, ``'hexahedron'``, or ``'quadratic_tetra'``.
            Types are explicit because node count alone is ambiguous.
        as_surface : bool, optional
            Extract boundary cells before constructing the result. The result
            remains an UnstructuredGrid containing the extracted surface cells.

        Returns
        -------
        PyVistaElemGrid
            Grid with ``source_cell_id`` cell data mapping each output cell to
            its original element. This array is reserved for internal mapping.

        Notes
        -----
        ``grid[name] = values`` always expects the original element count,
        including elements that are absent from an extracted surface. Scalar
        and multi-component arrays are supported. Assign node data explicitly
        through ``grid.point_data[name]``.
        """
        coords = cls._validate_coords(coords)
        rows = [np.asarray(row) for row in elems]
        if not rows:
            raise ValueError('elems must contain at least one element.')
        types = cls._expand_cell_types(cell_type, len(rows))
        cells, cell_types = cls._build_connectivity(rows, types, len(coords))
        grid = cls._build_grid(coords, cells, cell_types, as_surface)

        instance = cls(grid)
        instance.n_original_cells = len(rows)
        instance.n_original_points = len(coords)
        instance.as_surface = as_surface
        return instance

    @staticmethod
    def _validate_coords(coords):
        """Convert coordinates to an array and validate their shape and values."""
        coords = np.asarray(coords, dtype=np.result_type(coords, np.float32))
        if (coords.ndim != 2 or coords.shape[1] != 3
                or coords.dtype.kind not in 'iuf' or not np.isfinite(coords).all()):
            raise ValueError('coords must be a finite numeric array of shape (n_points, 3).')
        return coords

    @staticmethod
    def _expand_cell_types(cell_type, n_elements):
        """Expand a shared type or check that each element has a type."""
        if isinstance(cell_type, (str, int, np.integer)):
            types = [cell_type] * n_elements
        else:
            types = list(cell_type)
        if len(types) != n_elements:
            raise ValueError('Provide one cell type per element.')
        return types

    @staticmethod
    def _parse_cell_type(kind):
        """Resolve a name or VTK ID to a supported PyVista cell type."""
        try:
            if isinstance(kind, str):
                kind = pv.CellType[kind.upper()]
            elif isinstance(kind, (bool, np.bool_)) or not isinstance(kind, (int, np.integer)):
                raise ValueError('Cell types must be names or integer VTK IDs.')
            kind = pv.CellType(kind)
        except (KeyError, ValueError) as exc:
            raise ValueError(f'Unknown cell type: {kind!r}') from exc
        if kind in (pv.CellType.POLYHEDRON, pv.CellType.EMPTY_CELL):
            raise ValueError(f'{kind.name} is not supported by element-node connectivity.')
        return kind

    @staticmethod
    def _validate_element(row, kind, n_points, probe):
        """Check node indices and the node count required by the cell type."""
        if row.ndim != 1 or row.dtype.kind not in 'iu' or row.size == 0:
            raise ValueError('Each element must be a nonempty 1D sequence of integer node indices.')
        if np.any(row < 0) or np.any(row >= n_points):
            raise ValueError('Element node indices must be within coords.')
        probe.SetCellType(int(kind))
        expected = probe.GetNumberOfPoints()
        if expected and len(row) != expected:
            raise ValueError(f'{kind.name} requires {expected} nodes; got {len(row)}.')
        minimum_sizes = {'POLY_VERTEX': 1, 'POLY_LINE': 2,
                         'TRIANGLE_STRIP': 3, 'POLYGON': 3,
                         'CONVEX_POINT_SET': 4}
        if len(row) < minimum_sizes.get(kind.name, 1):
            raise ValueError(f'Too few nodes for {kind.name}.')

    @classmethod
    def _build_connectivity(cls, rows, types, n_points):
        """Validate elements and pack connectivity and type arrays for VTK."""
        cells = []
        cell_types = []
        probe = vtkGenericCell()
        for row, kind in zip(rows, types):
            kind = cls._parse_cell_type(kind)
            cls._validate_element(row, kind, n_points, probe)
            cells.extend([len(row), *row])
            cell_types.append(int(kind))
        return np.asarray(cells, dtype=np.int64), np.asarray(cell_types, dtype=np.uint8)

    @staticmethod
    def _build_grid(coords, cells, cell_types, as_surface):
        """Create the grid and preserve element IDs through surface extraction."""
        grid = pv.UnstructuredGrid(cells, cell_types, coords)
        grid.cell_data['source_cell_id'] = np.arange(len(cell_types), dtype=np.int64)
        grid.point_data['source_point_id'] = np.arange(len(coords), dtype=np.int64)
        if as_surface:
            grid = grid.extract_surface(algorithm='geometry')
        return grid

    def __setitem__(self, name, values):
        """Map original element scalars or vectors onto output cells."""
        if len(values) == self.n_cells or len(values) == self.n_points:
            super().__setitem__(name, values)
            return

        if name == 'source_cell_id' or name == 'source_point_id':
            raise ValueError('source_cell_id and source_point_id are reserved for element mapping.')

        values = np.asarray(values)

        if len(values) == self.n_original_cells:
            self.cell_data[name] = values[self.cell_data['source_cell_id']]
        elif len(values) == self.n_original_points:
            self.point_data[name] = values[self.point_data['source_point_id']]
        else:
            raise ValueError(f'Expected {self.n_original_cells} or {self.n_original_points} values; got {len(values)}.')

    @property
    def cell_indices(self):
        """Unique indices of the original elements corresponding to the existing cells."""
        return np.unique(self.cell_data['source_cell_id'])
