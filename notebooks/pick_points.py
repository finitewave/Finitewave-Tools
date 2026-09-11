"""Display a mesh and print the ID of each point clicked in PyVista."""

from pathlib import Path

import numpy as np
import pyvista as pv
import finitewave as fw


path_mesh = Path("./data/")
mesh = np.load(path_mesh / "lv_mesh.npy")
grid = fw.PyVistaMeshGrid.from_mesh(mesh)

def print_point_id(point):
    if point is None:
        return

    cell_id = grid.find_closest_cell(point)
    mesh_id = grid["mesh_inds"][cell_id]
    coords = np.unravel_index(mesh_id, grid.mesh.shape)
    print(f"Mesh ID: {mesh_id}, coordinates: {coords}", flush=True)


plotter = pv.Plotter()
plotter.add_mesh(grid, show_edges=False, pickable=True, color="lightgray", opacity=0.5)
plotter.enable_point_picking(
    callback=print_point_id,
    left_clicking=True,
    show_message="Left-click a mesh point to print its ID",
    show_point=True,
    point_size=12,
)
plotter.show()
