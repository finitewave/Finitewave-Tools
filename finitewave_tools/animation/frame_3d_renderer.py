
"""Render scalar fields on a PyVista mesh as off-screen RGB frames."""

import pyvista as pv
from finitewave_tools.animation.frame_2d_renderer import Frame2DRenderer


class Frame3DRenderer(Frame2DRenderer):
    """Render scalar-frame sequences on a shared 3D dataset.

    Select arrays or a directory using the inherited ``set_frames`` method,
    then call ``configure_grid``. Input frames should be one-dimensional
    arrays containing one scalar per mesh point or cell. The inherited
    ``render_all_frames`` method can capture the entire sequence.

    Rendering does not display a window or an inline Jupyter plot. Call
    ``finalize`` when finished to release the rendering resources.
    """

    def __init__(self):
        """Initialize an empty frame source and an unconfigured scene."""
        super().__init__()
        self.grid = None
        self.plotter = None
        self.actor = None

    def render_frame(self, frame_index, scalar_name="Scalars", camera_position=None, 
                     clim=(0, 1), cmap="viridis", nan_color="black", callback=None):
        """Update the mesh scalars and capture one RGB frame.

        Parameters
        ----------
        frame_index : int
            Zero-based index in the source selected with ``set_frames``.
        scalar_name : str, optional
            Mesh array name under which to store the loaded values. Existing
            values are replaced and the mapper selects this array name.
        camera_position : str or sequence, optional
            PyVista camera preset, or camera position, focal point, and view-up
            vector. Applied on every call, including when the default is used.
        clim : tuple of float, optional
            Lower and upper limits for the mapper's scalar range.
        cmap : str or matplotlib.colors.Colormap, optional
            Colormap used to construct a new lookup table for this frame.
        nan_color : color_like, optional
            Lookup-table color for NaN values and values outside its range.
        callback : callable, optional
            A function to be called after each frame is rendered.
            The function should accept two arguments: the renderer instance and the frame index.
        **kwargs : dict
            Accepted for interface compatibility; currently ignored.

        Returns
        -------
        numpy.ndarray
            RGB image of shape ``(height, width, 3)`` with dtype ``uint8``.
            Any screenshot alpha channel is discarded.

        Raises
        ------
        RuntimeError
            If the plotter is unconfigured or no frames are selected.
        IndexError
            If ``frame_index`` is invalid.

        Notes
        -----
        Updates the dataset in place. PyVista infers point or cell association
        from the input length. The mapper's association and scalar visibility
        are established during configuration; selecting an array name here
        does not change them. Options revert to their defaults on subsequent
        calls unless explicitly supplied again.
        """

        if self.plotter is None:
            raise RuntimeError("Configure the grid before rendering.")

        frame = self._load_frame(frame_index)

        self.grid[scalar_name] = frame

        scalar_map_mode = ("cell_field" if scalar_name in self.grid.cell_data else "point_field")
        
        self.actor.mapper.scalar_map_mode = scalar_map_mode
        self.actor.mapper.SelectColorArray(scalar_name)
        self.actor.mapper.color_mode = "map"
        self.actor.mapper.scalar_visibility = True

        lut = self.actor.mapper.lookup_table
        lut.apply_cmap(cmap)
        lut.scalar_range = clim
        lut.nan_color = nan_color
        lut.above_range_color = nan_color
        lut.below_range_color = nan_color
        self.actor.mapper.scalar_range = clim

        for bar in self.plotter.scalar_bars.values():
            bar.SetLookupTable(lut)
            bar.SetTitle(scalar_name)

        if camera_position is not None:
            self.plotter.camera_position = camera_position

        if callback is not None:
            callback(self, frame_index)

        self.plotter.render()

        img = self.plotter.screenshot(None) 
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img

    def configure_grid(self, grid, window_size=(800, 800), **kwargs):
        """Create an off-screen scene for a PyVista dataset.

        Parameters
        ----------
        grid : pyvista.DataSet
            Mesh receiving frame scalars. Retained by reference and modified
            in place during rendering.
        window_size : tuple of int, optional
            Output dimensions in pixels, ordered as ``(width, height)``.
        **kwargs : dict
            Forwarded to ``pyvista.Plotter.add_mesh``, including ``scalars``,
            point/cell ``preference``, and ``show_scalar_bar``. Select an
            existing scalar array to establish the intended scalar mapping.

        Notes
        -----
        Adds orientation axes and initializes the scene for screenshots.
        Both notebook display and on-screen rendering are disabled.
        Call ``finalize`` before reconfiguring an existing scene: this method
        replaces the plotter without closing the previous one.
        """
        self.grid = grid
        self.width, self.height = window_size
        # Off-screen rendering alone still allows inline display in Jupyter.
        self.plotter = pv.Plotter(
            notebook=False, off_screen=True, window_size=window_size
        )
        self.actor = self.plotter.add_mesh(self.grid, **kwargs)
        self.plotter.add_axes(line_width=5)
        self.plotter.show(auto_close=False)

    def finalize(self):
        """Close the configured plotter and release rendering resources.

        Call after successful configuration. Configure a new scene before
        attempting to render again.
        """
        self.plotter.close()
