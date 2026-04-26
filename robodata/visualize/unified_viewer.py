"""Unified dataset visualizer using Robodata's intermediate format.

This viewer uses the FormatRegistry to read any supported format through
the Episode/Frame intermediate representation, providing a single unified
visualization experience across all formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robodata.core.exceptions import MissingDependencyError
from robodata.core.models import Episode, Frame
from robodata.formats.registry import FormatRegistry


def _check_matplotlib() -> tuple[Any, Any, Any]:
    """Check and import matplotlib dependencies."""
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
    except ImportError:
        raise MissingDependencyError(
            dependency="matplotlib",
            feature="Dataset visualization",
            install_hint="pip install matplotlib",
        )
    return plt, Slider, Button


class UnifiedBackend:
    """Backend that uses FormatRegistry to read any supported format.

    Instead of format-specific backends, this uses the intermediate
    Episode/Frame representation that all readers produce.
    """

    def __init__(self, dataset_path: Path, max_episodes: int = 50):
        self.dataset_path = Path(dataset_path)

        # Detect format and get reader
        self.format_name = FormatRegistry.detect_format(dataset_path)
        self.reader = FormatRegistry.get_reader(self.format_name)

        # Inspect dataset for metadata
        self.info = self.reader.inspect(dataset_path)

        # Pre-load episodes (cache frames for random access)
        self._episodes: list[list[Frame]] = []
        self._camera_keys: list[str] = []
        self._numeric_keys: list[str] = []
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        # Cache images as they're loaded (replay same episode is fast)
        self._image_cache: dict[tuple[int, int, str], np.ndarray] = {}

        # Load episodes up to limit
        for i, episode in enumerate(self.reader.read_episodes(dataset_path)):
            if i >= max_episodes:
                break

            # Materialize frames for random access
            frames = list(episode.frames())
            self._episodes.append(frames)

            # Detect features from first episode's first frame
            if i == 0 and frames:
                self._detect_features(frames[0], episode)

    def _detect_features(self, frame: Frame, episode: Episode) -> None:
        """Detect camera and numeric features from a frame."""
        # Cameras from frame images
        for cam_name, lazy_img in frame.images.items():
            self._camera_keys.append(cam_name)
            self._image_shapes[cam_name] = lazy_img.shape

        # Also check episode.cameras for additional info
        for cam_name, cam_info in episode.cameras.items():
            if cam_name not in self._camera_keys:
                self._camera_keys.append(cam_name)
                self._image_shapes[cam_name] = cam_info.shape

        # Numeric features: action and state
        if frame.action is not None:
            self._numeric_keys.append("action")
        if frame.state is not None:
            self._numeric_keys.append("state")

        # Also add from DatasetInfo
        if self.info.action_schema:
            if "action" not in self._numeric_keys:
                self._numeric_keys.append("action")
        for obs_name in self.info.observation_schema:
            if obs_name not in self._numeric_keys and obs_name not in self._camera_keys:
                self._numeric_keys.append(obs_name)

    def get_num_episodes(self) -> int:
        return len(self._episodes)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self._episodes):
            return 0
        return len(self._episodes[episode_idx])

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        # Try cache first
        cache_key = (episode_idx, frame_idx, camera_key)
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]

        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]
        if frame_idx >= len(frames):
            return None

        frame = frames[frame_idx]
        if camera_key not in frame.images:
            return None

        # Load the lazy image
        img = frame.images[camera_key].load()

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Cache for replay
        self._image_cache[cache_key] = img
        return img

    def get_frame_data(self, episode_idx: int, frame_idx: int, feature_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]
        if frame_idx >= len(frames):
            return None

        frame = frames[frame_idx]

        if feature_key == "action":
            return frame.action
        elif feature_key == "state":
            return frame.state

        return None

    def get_episode_data(self, episode_idx: int, feature_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]

        data = []
        for frame in frames:
            if feature_key == "action" and frame.action is not None:
                data.append(frame.action)
            elif feature_key == "state" and frame.state is not None:
                data.append(frame.state)

        if data:
            return np.array(data)
        return None

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_numeric_keys(self) -> list[str]:
        # Prioritize action and state
        priority = ["action", "state"]
        result = [k for k in priority if k in self._numeric_keys]
        result.extend([k for k in self._numeric_keys if k not in priority])
        return result

    def get_image_shape(self, camera_key: str) -> tuple[int, int, int]:
        return self._image_shapes.get(camera_key, (480, 640, 3))

    def get_fps(self) -> float:
        return self.info.inferred_fps or 10.0

    def get_name(self) -> str:
        return f"{self.dataset_path.name} ({self.format_name})"


class UnifiedViewer:
    """Interactive viewer using the unified backend.

    Supports any format registered in FormatRegistry through the
    intermediate Episode/Frame representation.

    Example:
        >>> viewer = UnifiedViewer("path/to/dataset")
        >>> viewer.show()

        # Comparison mode
        >>> viewer = UnifiedViewer("original", "converted")
        >>> viewer.show()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        compare_path: str | Path | None = None,
        max_episodes: int = 50,
    ):
        self.plt, self.Slider, self.Button = _check_matplotlib()

        # Create backends
        self.backends: list[UnifiedBackend] = [
            UnifiedBackend(Path(dataset_path), max_episodes)
        ]
        if compare_path:
            self.backends.append(UnifiedBackend(Path(compare_path), max_episodes))

        self.comparison_mode = len(self.backends) > 1

        # State
        self.current_episode = 0
        self.current_frame = 0
        self.playing = False

        # UI elements (set during setup)
        self.fig = None
        self.image_displays: list[list[tuple[Any, str]]] = []  # [(ax, cam_key), ...]
        self.plot_displays: list[list[tuple[Any, list, Any, str]]] = []  # [(ax, lines, marker, key), ...]
        self.frame_slider = None
        self.episode_slider = None

    def show(self) -> None:
        """Display the viewer window."""
        self._setup_figure()
        self._update_display()
        self.plt.show()

    def _setup_figure(self) -> None:
        """Set up the matplotlib figure and controls."""
        # Determine layout
        num_backends = len(self.backends)

        # Get camera keys for each backend (they may differ between formats)
        cameras_per_backend = [backend.get_camera_keys() for backend in self.backends]
        max_cameras = max(len(cams) for cams in cameras_per_backend)

        # Only include numeric keys that have actual data
        numeric_keys = []
        for key in self.backends[0].get_numeric_keys()[:3]:
            data = self.backends[0].get_episode_data(0, key)
            if data is not None and data.size > 0:
                numeric_keys.append(key)

        num_plots = len(numeric_keys)

        # Calculate figure size
        if self.comparison_mode:
            fig_width = 12
            n_cols = 2
        else:
            fig_width = 8
            n_cols = 1

        n_image_rows = max(1, max_cameras)
        n_plot_rows = num_plots if num_plots > 0 else 0
        fig_height = 3 * n_image_rows + (1.5 * n_plot_rows if n_plot_rows > 0 else 0) + 1

        self.fig = self.plt.figure(figsize=(fig_width, fig_height))

        # Create grid
        total_rows = n_image_rows + n_plot_rows if n_plot_rows > 0 else n_image_rows
        height_ratios = [3] * n_image_rows + ([1.5] * n_plot_rows if n_plot_rows > 0 else []) + [0.3]
        gs = self.fig.add_gridspec(
            total_rows + 1, n_cols,
            height_ratios=height_ratios,
            hspace=0.4, wspace=0.15
        )

        # Image displays - each backend uses its own camera keys
        self.image_displays = [[] for _ in range(num_backends)]
        for b_idx, backend in enumerate(self.backends):
            col = b_idx if self.comparison_mode else 0
            backend_cameras = cameras_per_backend[b_idx]
            for row in range(n_image_rows):
                ax = self.fig.add_subplot(gs[row, col])
                if row < len(backend_cameras):
                    cam_key = backend_cameras[row]
                    # Shorter title - just camera name
                    short_cam = cam_key.split('/')[-1].replace('_image', '')
                    ax.set_title(short_cam, fontsize=8, pad=2)
                    self.image_displays[b_idx].append((ax, cam_key))
                else:
                    ax.axis('off')
                    self.image_displays[b_idx].append((ax, None))
                ax.axis('off')

        # Plot displays - only if we have data
        self.plot_displays = [[] for _ in range(num_backends)]
        if n_plot_rows > 0:
            for b_idx, backend in enumerate(self.backends):
                col = b_idx if self.comparison_mode else 0
                for row, feat_key in enumerate(numeric_keys):
                    ax = self.fig.add_subplot(gs[n_image_rows + row, col])
                    ax.set_title(feat_key, fontsize=7, pad=1)
                    ax.tick_params(axis='both', labelsize=6)

                    # Initial plot
                    ep_data = backend.get_episode_data(self.current_episode, feat_key)
                    lines = []
                    if ep_data is not None:
                        if len(ep_data.shape) > 2:
                            ep_data = ep_data.reshape(ep_data.shape[0], -1)

                        if len(ep_data.shape) > 1:
                            for dim in range(min(ep_data.shape[1], 6)):
                                line, = ax.plot(ep_data[:, dim], alpha=0.6, linewidth=0.8)
                                lines.append(line)
                        else:
                            line, = ax.plot(ep_data, linewidth=0.8)
                            lines.append(line)

                    # Frame marker
                    marker = ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    self.plot_displays[b_idx].append((ax, lines, marker, feat_key))

        # Controls
        control_ax = self.fig.add_subplot(gs[-1, :])
        control_ax.axis('off')

        # Episode slider
        ep_ax = self.fig.add_axes([0.15, 0.06, 0.3, 0.02])
        max_ep = self.backends[0].get_num_episodes() - 1
        self.episode_slider = self.Slider(
            ep_ax, 'Episode', 0, max(1, max_ep),
            valinit=0, valstep=1
        )
        self.episode_slider.on_changed(self._on_episode_change)

        # Frame slider
        frame_ax = self.fig.add_axes([0.55, 0.06, 0.3, 0.02])
        max_frame = self.backends[0].get_episode_length(0) - 1
        self.frame_slider = self.Slider(
            frame_ax, 'Frame', 0, max(1, max_frame),
            valinit=0, valstep=1
        )
        self.frame_slider.on_changed(self._on_frame_change)

        # Play/Pause button
        play_ax = self.fig.add_axes([0.45, 0.01, 0.1, 0.03])
        self.play_button = self.Button(play_ax, 'Play')
        self.play_button.on_clicked(self._on_play_click)

    def _update_display(self) -> None:
        """Update all displays for current episode/frame."""
        for b_idx, backend in enumerate(self.backends):
            # Update images
            for idx, (ax, cam_key) in enumerate(self.image_displays[b_idx]):
                if cam_key is None:
                    continue
                img = backend.get_frame_image(self.current_episode, self.current_frame, cam_key)
                if img is not None:
                    # Update existing image if possible, else create new
                    if ax.images:
                        ax.images[0].set_data(img)
                    else:
                        ax.imshow(img)
                        ax.axis('off')

            # Update plot markers
            if b_idx < len(self.plot_displays):
                for ax, lines, marker, feat_key in self.plot_displays[b_idx]:
                    marker.set_xdata([self.current_frame, self.current_frame])

        # Use blit-friendly update if playing, full draw otherwise
        if self.playing:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def _on_episode_change(self, val: float) -> None:
        """Handle episode slider change."""
        self.current_episode = int(val)
        self.current_frame = 0

        # Update frame slider range
        max_frame = self.backends[0].get_episode_length(self.current_episode) - 1
        self.frame_slider.valmax = max(1, max_frame)
        self.frame_slider.set_val(0)

        # Update plots
        self._update_plots()
        self._update_display()

    def _on_frame_change(self, val: float) -> None:
        """Handle frame slider change."""
        self.current_frame = int(val)
        self._update_display()

    def _on_play_click(self, event: Any) -> None:
        """Handle play/pause button click."""
        self.playing = not self.playing
        self.play_button.label.set_text('Pause' if self.playing else 'Play')

        if self.playing:
            self._animate()

    def _animate(self) -> None:
        """Animate playback."""
        if not self.playing:
            return

        try:
            max_frame = self.backends[0].get_episode_length(self.current_episode) - 1
            if self.current_frame < max_frame:
                self.current_frame += 1
                self.frame_slider.set_val(self.current_frame)
                self._update_display()  # Ensure display updates even if slider callback doesn't fire
            else:
                self.playing = False
                self.play_button.label.set_text('Play')
                return

            # Schedule next frame
            fps = self.backends[0].get_fps()
            interval = 1000 / fps  # ms
            try:
                # Try Tkinter backend
                self.fig.canvas.get_tk_widget().after(int(interval), self._animate)
            except AttributeError:
                # Fallback for other backends - use timer
                timer = self.fig.canvas.new_timer(interval=int(interval))
                timer.add_callback(lambda: self._animate())
                timer.single_shot = True
                timer.start()
        except Exception as e:
            # Stop animation on error but don't crash
            print(f"Animation error: {e}")
            self.playing = False
            self.play_button.label.set_text('Play')

    def _update_plots(self) -> None:
        """Update plot data for current episode."""
        for b_idx, backend in enumerate(self.backends):
            if b_idx >= len(self.plot_displays):
                continue

            for ax, lines, marker, feat_key in self.plot_displays[b_idx]:
                ep_data = backend.get_episode_data(self.current_episode, feat_key)
                if ep_data is None:
                    continue

                if len(ep_data.shape) > 2:
                    ep_data = ep_data.reshape(ep_data.shape[0], -1)

                if len(ep_data.shape) > 1:
                    for dim, line in enumerate(lines):
                        if dim < ep_data.shape[1]:
                            line.set_data(range(len(ep_data)), ep_data[:, dim])
                elif lines:
                    lines[0].set_data(range(len(ep_data)), ep_data)

                ax.relim()
                ax.autoscale_view()


def unified_visualize(
    dataset_path: str | Path,
    compare_path: str | Path | None = None,
    max_episodes: int = 50,
) -> None:
    """Visualize a dataset using the unified viewer.

    Args:
        dataset_path: Path to dataset (any supported format).
        compare_path: Optional path to second dataset for comparison.
        max_episodes: Maximum episodes to load (default 50).
    """
    viewer = UnifiedViewer(dataset_path, compare_path, max_episodes)
    viewer.show()
