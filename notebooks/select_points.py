"""Right-click a mesh vertex to display and print its zero-based point ID."""

import argparse
from importlib.resources import path
from pathlib import Path

import pyvista as pv
import numpy as np


def main():
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--mesh", type=Path,
    #                     default=Path("/Users/arstanbekokenov/Projects/Finitewave-Tools/Tutorials") / "data/atrial_mesh.vtk")
    # args = parser.parse_args()
    path = Path("/Users/arstanbekokenov/Projects/Finitewave-Tools/Tutorials/data/atrial")
    points = np.load(path / "points.npy")
    elems = np.load(path / "elems.npy")
    mv = np.load(path / "mv_ids.npy")
    lpv = np.load(path / "lpv_ids.npy")
    rpv = np.load(path / "rpv_ids.npy")

    faces = np.hstack([np.full((elems.shape[0], 1), 3), elems]).astype(np.int32)
    mesh = pv.PolyData(points, faces)

    pv.set_plot_theme("document")
    pl = pv.Plotter()
    pl.add_mesh(mesh, color="lightgrey", show_edges=True)
    pl.add_points(points[mv], color="red", point_size=20, render_points_as_spheres=True, label="MV")
    pl.add_points(points[lpv], color="blue", point_size=20, render_points_as_spheres=True, label="LPV")
    pl.add_points(points[rpv], color="green", point_size=20, render_points_as_spheres=True, label="RPV")
    pl.add_text("Right-click: select a vertex | Left-drag: rotate", font_size=11)

    def show_id(point, picker):
        point_id = picker.GetPointId()
        if point_id < 0:
            return
        print(f"Point ID: {point_id}", flush=True)
        pl.add_text(f"Point ID: {point_id}", name="selected_id",
                    position="lower_left", font_size=14)
        pl.add_point_labels(
            mesh.points[point_id:point_id + 1], [str(point_id)],
            name="selected_point", point_color="red", point_size=12,
            always_visible=True, pickable=False, reset_camera=False,
        )

    pl.enable_point_picking(callback=show_id, picker="point", use_picker=True,
                            show_point=False, show_message=False)
    pl.camera_position = "yz"
    pl.show()


if __name__ == "__main__":
    main()
