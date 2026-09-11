
from numba import np
from scipy import ndimage


def extract_outer_surface(mesh):
    # check if mesh is touched by the boundary of the volume
    if (np.any([mesh[i, :, :].any() for i in [0, -1]])
        or np.any([mesh[:, i, :].any() for i in [0, -1]])
        or np.any([mesh[:, :, i].any() for i in [0, -1]])):
        raise ValueError("Mesh touches the boundary of the volume. "
                         "Cannot dilate mesh to create outer surface.")

    dialated_mesh = ndimage.binary_dilation(mesh > 0)
    outer_mesh = dialated_mesh & (mesh == 0)
    return outer_mesh


def extract_endo_epi_surface(outer_mesh, axis=2):
    outer_mesh = outer_mesh.copy()
    axis_max = outer_mesh.shape[axis]

    idx = [slice(None)] * outer_mesh.ndim
    idx[axis] = axis_max
    outer_mesh[tuple(idx)] = False

    surface_labeled, n_surfaces = ndimage.label(outer_mesh, structure=np.ones((3, 3, 3)))
    if n_surfaces != 2:
        raise ValueError(f"Expected 2 surfaces, but found {n_surfaces}.")

    endo_surface = surface_labeled == 1
    epi_surface = surface_labeled == 2

    return endo_surface, epi_surface
