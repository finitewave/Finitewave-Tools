"""Export RGB frame sequences as MP4 videos or animated GIF files."""

from pathlib import Path
from tqdm.auto import tqdm
import av


class AnimationBuilder:
    """Encode frames supplied by a configured 2D or 3D renderer.

    Assign a renderer to ``frame_renderer`` after selecting its frame source
    and configuring its grid. The renderer must expose ``width``, ``height``,
    ``total_frames``, ``render_frame(index, **kwargs)``, and ``finalize()``.
    Each rendered frame must be an RGB ``uint8`` array with shape
    ``(height, width, 3)``.

    MP4 export uses PyAV with the ``libx264`` encoder. GIF export additionally
    requires ImageIO. Frames are rendered and submitted one at a time.

    Notes
    -----
    Successful MP4 export finalizes the renderer; configure its scene again
    before reusing a renderer that owns rendering resources. GIF export leaves
    the renderer open, so the caller is responsible for finalizing it.
    """

    def __init__(self):
        """Initialize a writer with no assigned frame renderer."""
        self.frame_renderer = None

    def write(self, path=".", animation_name="frames", prog_bar=False, fps=24, **kwargs):
        """Write all selected frames to an H.264-encoded MP4 file.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Existing output directory. Missing directories are not created.
        animation_name : str, optional
            Output filename. Its suffix, if present, is replaced with ``.mp4``.
        prog_bar : bool, optional
            Whether to display a tqdm progress bar while rendering frames.
        fps : int, optional
            Positive number of frames per second, passed to the encoder.
        **kwargs : dict
            Forwarded to ``frame_renderer.render_frame`` for every frame,
            for example ``clim``, ``cmap``, or renderer-specific camera options.
            These arguments are not passed to the encoder.

        Returns
        -------
        None
            Writes the file and finalizes the renderer after successful export.

        Notes
        -----
        Uses the renderer's configured dimensions and ``yuv444p`` pixel format.
        Frame rate, renderer readiness, and frame shapes are not validated here.
        Rendering and encoding errors propagate to the caller; cleanup is only
        performed on the successful path in the current implementation.
        """
        
        path = Path(path, animation_name)

        container = av.open(str(path.with_suffix(".mp4")), mode="w")

        stream = container.add_stream("libx264", rate=fps)
        stream.width = self.frame_renderer.width
        stream.height = self.frame_renderer.height
        stream.pix_fmt = "yuv444p"

        for frame_i in tqdm(range(self.frame_renderer.total_frames),
                            desc="Building animation", disable=not prog_bar):
            # Grab the frame as a NumPy RGB array
            img = self.frame_renderer.render_frame(frame_i, **kwargs)

            # Convert to PyAV frame and mux
            frame = av.VideoFrame.from_ndarray(img, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        # Cleanup
        for packet in stream.encode():
            container.mux(packet)

        container.close()
        self.frame_renderer.finalize()

    def write_gif(self, path=".", animation_name="frames", prog_bar=False, fps=24, **kwargs):
        """Write all selected frames to a looping animated GIF.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Existing output directory. Missing directories are not created.
        animation_name : str, optional
            Output filename. Its suffix, if present, is replaced with ``.gif``.
        prog_bar : bool, optional
            Whether to display a tqdm progress bar while rendering frames.
        fps : int, optional
            Requested frame rate, passed to the ImageIO writer. Timing support
            and precision depend on the selected backend and GIF format.
        **kwargs : dict
            Forwarded to ``frame_renderer.render_frame`` for every frame.
            These arguments are not passed to the GIF writer.

        Returns
        -------
        None
            Writes the GIF file. The renderer remains open.

        Notes
        -----
        Requests indefinite looping with ``loop=0``. The ImageIO writer is
        closed on exit, including when rendering fails. Call the renderer's
        ``finalize`` method when it is no longer needed.
        """
        import imageio
        path = Path(path, animation_name).with_suffix(".gif")
        total_frames = self.frame_renderer.total_frames
        with imageio.get_writer(str(path), mode='I', fps=fps, plugin="pyav", loop=0) as writer:
            for i in tqdm(range(total_frames), desc="Building animation", disable=not prog_bar):
                img = self.frame_renderer.render_frame(i, **kwargs)
                writer.append_data(img)
