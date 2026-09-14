"""Render scalar fields as RGB animation frames."""

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import natsort
import numpy as np


class Frame2DRenderer:
    """Render dense 2D fields or packed values on a configured tissue grid.

    Call ``set_frames`` and ``configure_grid`` before rendering. Packed frames
    contain one value per positive mesh entry, in NumPy boolean-index order.
    Dense frames have the mesh shape. With restoration enabled, nonpositive
    mesh entries are masked in either representation.
    """

    def __init__(self):
        self.frames = None
        self.frame_paths = None
        self.total_frames = 0
        self.width = None
        self.height = None
        self._grid_shape = None

    def set_frames(self, path=None, frames=None):
        """Select exactly one source: a directory of .npy files or a sequence.
        
        Parameters
        ----------
        path : str, optional
            Directory containing .npy files for each frame. Files are sorted
            naturally by name. Provide this or ``frames``, but not both.
        frames : sequence of ndarray, optional
            Sequence of 2D arrays representing frames. Provide this or ``path``,
            but not both. All frames must have the same shape.
        """
        if (path is None) == (frames is None):
            raise ValueError("Provide exactly one of path or frames.")

        if frames is not None:
            if len(frames) == 0:
                raise ValueError("Frames must not be empty.")
            self.frames = frames
            self.frame_paths = None
            self.total_frames = len(frames)
        else:
            paths, total_frames = self.collect_files(path)
            self.frame_paths = paths
            self.frames = None
            self.total_frames = total_frames

    def configure_grid(self, mesh, upscale_factor=1):
        """Set the spatial shape and optional tissue mask; scale by repetition.
        
        Parameters
        ----------
        mesh : array_like
            2D array representing the spatial shape of the grid.
        upscale_factor : int, optional
            Factor by which to scale the grid.
        """
        mesh = np.asarray(mesh)
        if mesh.ndim != 2 or 0 in mesh.shape:
            raise ValueError("Mesh must be a nonempty 2D array.")

        self._validate_scale(upscale_factor)
        self._grid_shape = mesh.shape
        self.width = mesh.shape[1] * upscale_factor
        self.height = mesh.shape[0] * upscale_factor

    def collect_files(self, path):
        """Return naturally sorted .npy paths and their count."""
        files = natsort.natsorted(p for p in Path(path).glob("*.npy") if p.is_file())
        if not files:
            raise ValueError(f"No .npy files found in {Path(path)}")
        return files, len(files)

    def render_frame(self, frame_index, clim=(0, 1), cmap="viridis",
                     nan_color="black", output_mask=None, nan_mask=None,
                     upscale_factor=1):
        """Render a single frame as an RGB array.

        Parameters
        ----------
        frame_index : int
            Index of the frame to render.
        clim : tuple of float, optional
            Color limits for the colormap. Must be two finite, increasing values.
        cmap : str, optional
            Name of a Matplotlib colormap to use.
        nan_color : str or tuple, optional
            Color for nonfinite values, masked cells, and values outside ``clim``.

        Returns
        -------
        ndarray
            RGB array of shape (height, width, 3) with dtype uint8.
        """
        if self._grid_shape is None:
            raise RuntimeError("Configure the grid before rendering.")

        frame = self._load_frame(frame_index)

        if nan_mask is not None:
            frame = self._mask_frame(frame, nan_mask)

        if output_mask is not None:
            frame = self._restore_frame(frame, output_mask)

        if upscale_factor > 1:
            frame = self._upscale_frame(frame, upscale_factor)

        if frame.shape != (self.height, self.width):
            raise ValueError("Frame shape must match the configured grid.")

        return self._to_rgb(frame, clim, self.setup_cmap(cmap, nan_color))

    def render_all_frames(self, **kwargs):
        """Render all selected frames into a list of RGB arrays.
        
        Returns
        -------
        list of ndarray
            List of RGB arrays, each of shape (height, width, 3) with dtype uint8.
        """
        if self.frames is None and self.frame_paths is None:
            raise RuntimeError("Set frames before rendering.")

        return [self.render_frame(i, **kwargs) for i in range(self.total_frames)]

    def setup_cmap(self, cmap, nan_color="black"):
        """Configure a private colormap without changing a caller's colormap."""
        return plt.get_cmap(cmap).with_extremes(bad=nan_color)

    def _load_frame(self, frame_index):
        if self.frames is None and self.frame_paths is None:
            raise RuntimeError("Set frames before rendering.")
        
        if (isinstance(frame_index, (bool, np.bool_))
                or not isinstance(frame_index, Integral)
                or not 0 <= frame_index < self.total_frames):
            raise IndexError("Frame index is out of range.")
        
        if self.frames is not None:
            return np.asarray(self.frames[frame_index])

        return np.load(self.frame_paths[frame_index])

    def _validate_scale(self, upscale_factor):
        if (isinstance(upscale_factor, (bool, np.bool_))
                or not isinstance(upscale_factor, Integral)
                or upscale_factor < 1):
            raise ValueError("upscale_factor must be a positive integer.")

    def _restore_frame(self, frame, output_mask):

        """Restore a packed frame to its full grid shape using a boolean mask.

        Parameters
        ----------
        frame : ndarray
            1D array representing the packed frame.
        output_mask : ndarray
            2D boolean array indicating which entries to restore.
            True values indicate positions to fill from the packed frame.
        
        Returns
        -------
        ndarray
            2D array of the same shape as output_mask, with values from frame
            at positions where output_mask is True, and NaN elsewhere.
        """

        frame = np.asarray(frame)

        output_mask = np.asarray(output_mask, dtype=bool)

        if output_mask.ndim != 2:
            raise ValueError("Output mask must be 2D.")

        if frame.size == output_mask.size:
            return frame.reshape(output_mask.shape)

        if frame.size != np.count_nonzero(output_mask):
            raise ValueError("Frame size must match the number of positive " + 
                             "entries in the output mask.")

        output = np.full(output_mask.shape, np.nan,
                         dtype=np.result_type(frame.dtype, np.float32))
        output[output_mask] = frame
        return output

    def _mask_frame(self, frame, nan_mask):
        """Apply a mask to a frame, setting masked entries to NaN.
        
        Parameters
        ----------
        frame : ndarray
            An array representing the frame.
        nan_mask : ndarray or None
            A boolean array indicating which entries to mask. If None, no masking is applied.
        Returns
        -------
        ndarray
            An array with the specified entries set to NaN.
        """
        frame = np.asarray(frame, dtype=np.result_type(frame.dtype, np.float32))

        if nan_mask is not None:
            return frame
        
        nan_mask = np.asarray(nan_mask, dtype=bool)

        if frame.shape != nan_mask.shape:
            raise ValueError("Frame must be the same shape as the mask.")

        frame[nan_mask] = np.nan
        return frame

    def _upscale_frame(self, frame, upscale_factor):
        self._validate_scale(upscale_factor)
        if upscale_factor > 1:
            frame = np.repeat(frame, upscale_factor, axis=0)
            frame = np.repeat(frame, upscale_factor, axis=1)

        return frame

    def _to_rgb(self, frame, clim, cmap):
        limits = np.asarray(clim, dtype=float)
        if (limits.shape != (2,) or not np.all(np.isfinite(limits))
                or limits[0] >= limits[1]):
            raise ValueError("clim must contain two finite, increasing values.")
        frame = np.asarray(frame, dtype=float)
        mask = ~np.isfinite(frame) | (frame < limits[0]) | (frame > limits[1])
        normalized = (frame - limits[0]) / (limits[1] - limits[0])
        normalized[mask] = np.nan
        return cmap(normalized, bytes=True)[:, :, :3].copy()

    def finalize(self):
        """Satisfy the animation writer interface; no resources need closing."""
        pass
