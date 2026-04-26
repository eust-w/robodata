"""Fast OpenCV-based dataset visualizer.

Uses OpenCV for video display (much faster than matplotlib) with
action/state plots rendered as composited images.

Controls:
    Space: Play/Pause
    Left/Right: Previous/Next frame
    Up/Down: Previous/Next episode
    Q/Escape: Quit
    +/-: Increase/Decrease playback speed
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from robodata.core.exceptions import MissingDependencyError
from robodata.core.models import Episode, Frame
from robodata.formats.registry import FormatRegistry


def _check_opencv():
    """Check if OpenCV is available."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise MissingDependencyError(
            dependency="opencv-python",
            feature="Fast OpenCV visualization",
            install_hint="pip install opencv-python",
        )


def _render_plot_to_array(
    data: np.ndarray,
    current_frame: int,
    width: int,
    height: int,
    title: str = "",
) -> np.ndarray:
    """Render a plot to a numpy array using matplotlib Agg backend.

    Args:
        data: 2D array of shape (num_frames, num_dims) or 1D array
        current_frame: Current frame index for marker
        width: Output image width
        height: Output image height
        title: Plot title

    Returns:
        RGB numpy array of shape (height, width, 3)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create figure with exact pixel dimensions
    dpi = 100
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)

    # Plot data
    if len(data.shape) == 1:
        ax.plot(data, linewidth=1, alpha=0.8)
    else:
        # Limit to 6 dimensions for readability
        for dim in range(min(data.shape[1], 6)):
            ax.plot(data[:, dim], linewidth=1, alpha=0.7, label=f'd{dim}')

    # Current frame marker
    ax.axvline(x=current_frame, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

    # Styling
    ax.set_xlim(0, len(data))
    ax.set_title(title, fontsize=9, pad=2)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)

    # Remove margins
    fig.tight_layout(pad=0.3)

    # Render to numpy array
    fig.canvas.draw()
    # Get RGBA buffer and convert to RGB
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()  # Drop alpha channel

    plt.close(fig)
    return img


class CVBackend:
    """Backend that loads data for OpenCV viewer."""

    def __init__(self, dataset_path: Path, max_episodes: int = 50):
        self.dataset_path = Path(dataset_path)

        # Detect format and get reader
        self.format_name = FormatRegistry.detect_format(dataset_path)
        self.reader = FormatRegistry.get_reader(self.format_name)

        # Inspect dataset for metadata
        self.info = self.reader.inspect(dataset_path)

        # Storage
        self._episodes: list[list[Frame]] = []
        self._camera_keys: list[str] = []
        self._episode_actions: list[np.ndarray | None] = []
        self._episode_states: list[np.ndarray | None] = []

        print(f"Loading episodes from {self.format_name} dataset...")

        # Load episodes
        for i, episode in enumerate(self.reader.read_episodes(dataset_path)):
            if i >= max_episodes:
                break

            if (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1} episodes...")

            # Materialize frames
            frames = list(episode.frames())
            self._episodes.append(frames)

            # Detect cameras from first episode
            if i == 0 and frames:
                for cam_name in frames[0].images.keys():
                    self._camera_keys.append(cam_name)

            # Pre-extract action/state arrays for plotting
            actions = []
            states = []
            for frame in frames:
                if frame.action is not None:
                    actions.append(frame.action)
                if frame.state is not None:
                    states.append(frame.state)

            self._episode_actions.append(np.array(actions) if actions else None)
            self._episode_states.append(np.array(states) if states else None)

        print(f"Loaded {len(self._episodes)} episodes")

    def get_num_episodes(self) -> int:
        return len(self._episodes)

    def get_episode_length(self, episode_idx: int) -> int:
        if episode_idx >= len(self._episodes):
            return 0
        return len(self._episodes[episode_idx])

    def get_frame_image(self, episode_idx: int, frame_idx: int, camera_key: str) -> np.ndarray | None:
        if episode_idx >= len(self._episodes):
            return None
        frames = self._episodes[episode_idx]
        if frame_idx >= len(frames):
            return None

        frame = frames[frame_idx]
        if camera_key not in frame.images:
            return None

        img = frame.images[camera_key].load()

        # Convert to uint8 if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def get_episode_actions(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_actions):
            return None
        return self._episode_actions[episode_idx]

    def get_episode_states(self, episode_idx: int) -> np.ndarray | None:
        if episode_idx >= len(self._episode_states):
            return None
        return self._episode_states[episode_idx]

    def get_camera_keys(self) -> list[str]:
        return self._camera_keys

    def get_fps(self) -> float:
        return self.info.inferred_fps or 30.0

    def get_name(self) -> str:
        return f"{self.dataset_path.name} ({self.format_name})"


class CVViewer:
    """Fast OpenCV-based dataset viewer.

    Uses OpenCV for video display with matplotlib-rendered plots
    composited into a single window.

    Example:
        >>> viewer = CVViewer("path/to/dataset")
        >>> viewer.show()
    """

    def __init__(
        self,
        dataset_path: str | Path,
        max_episodes: int = 50,
        plot_height: int = 120,
    ):
        self.cv2 = _check_opencv()

        # Load data
        self.backend = CVBackend(Path(dataset_path), max_episodes)
        self.plot_height = plot_height

        # State
        self.current_episode = 0
        self.current_frame = 0
        self.playing = False
        self.speed_multiplier = 1.0

        # Cache for plot backgrounds (re-render only on episode change)
        self._action_plot_bg: np.ndarray | None = None
        self._state_plot_bg: np.ndarray | None = None
        self._plot_width: int = 0

    def _render_plots(self, width: int) -> None:
        """Pre-render plot backgrounds for current episode."""
        self._plot_width = width

        actions = self.backend.get_episode_actions(self.current_episode)
        states = self.backend.get_episode_states(self.current_episode)

        if actions is not None and actions.size > 0:
            self._action_plot_bg = _render_plot_to_array(
                actions, 0, width, self.plot_height, "Actions"
            )
        else:
            self._action_plot_bg = None

        if states is not None and states.size > 0:
            self._state_plot_bg = _render_plot_to_array(
                states, 0, width, self.plot_height, "States"
            )
        else:
            self._state_plot_bg = None

    def _get_plot_with_marker(self, plot_bg: np.ndarray | None, data: np.ndarray | None) -> np.ndarray | None:
        """Get plot image with current frame marker."""
        if plot_bg is None or data is None:
            return None

        # Copy background
        plot = plot_bg.copy()

        # Calculate marker x position
        num_frames = len(data)
        if num_frames <= 1:
            return plot

        # Map frame index to pixel x coordinate (approximate)
        # Plot area is roughly 85% of width, starting at ~12%
        plot_left = int(self._plot_width * 0.12)
        plot_right = int(self._plot_width * 0.97)
        plot_range = plot_right - plot_left

        x = plot_left + int((self.current_frame / (num_frames - 1)) * plot_range)
        x = max(plot_left, min(plot_right, x))

        # Draw vertical red line
        self.cv2.line(plot, (x, 0), (x, self.plot_height), (0, 0, 255), 2)

        return plot

    def _compose_frame(self) -> np.ndarray:
        """Compose the full display frame."""
        # Get camera image(s)
        camera_keys = self.backend.get_camera_keys()
        images = []

        for cam_key in camera_keys:
            img = self.backend.get_frame_image(self.current_episode, self.current_frame, cam_key)
            if img is not None:
                # Convert RGB to BGR for OpenCV
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = self.cv2.cvtColor(img, self.cv2.COLOR_RGB2BGR)
                images.append(img)

        if not images:
            # Create placeholder
            images = [np.zeros((480, 640, 3), dtype=np.uint8)]

        # Stack camera images horizontally if multiple
        if len(images) > 1:
            # Resize to same height
            max_h = max(img.shape[0] for img in images)
            resized = []
            for img in images:
                if img.shape[0] != max_h:
                    scale = max_h / img.shape[0]
                    new_w = int(img.shape[1] * scale)
                    img = self.cv2.resize(img, (new_w, max_h))
                resized.append(img)
            camera_frame = np.hstack(resized)
        else:
            camera_frame = images[0]

        frame_width = camera_frame.shape[1]

        # Render plots if needed (on first frame or episode change)
        if self._plot_width != frame_width:
            self._render_plots(frame_width)

        # Get plots with markers
        components = [camera_frame]

        actions = self.backend.get_episode_actions(self.current_episode)
        action_plot = self._get_plot_with_marker(self._action_plot_bg, actions)
        if action_plot is not None:
            # Convert RGB to BGR
            action_plot = self.cv2.cvtColor(action_plot, self.cv2.COLOR_RGB2BGR)
            # Resize to match width
            if action_plot.shape[1] != frame_width:
                action_plot = self.cv2.resize(action_plot, (frame_width, self.plot_height))
            components.append(action_plot)

        states = self.backend.get_episode_states(self.current_episode)
        state_plot = self._get_plot_with_marker(self._state_plot_bg, states)
        if state_plot is not None:
            state_plot = self.cv2.cvtColor(state_plot, self.cv2.COLOR_RGB2BGR)
            if state_plot.shape[1] != frame_width:
                state_plot = self.cv2.resize(state_plot, (frame_width, self.plot_height))
            components.append(state_plot)

        # Stack vertically
        composed = np.vstack(components)

        # Add info overlay
        info_text = f"Episode {self.current_episode + 1}/{self.backend.get_num_episodes()}  "
        info_text += f"Frame {self.current_frame + 1}/{self.backend.get_episode_length(self.current_episode)}  "
        info_text += f"Speed: {self.speed_multiplier:.1f}x  "
        info_text += "[PLAYING]" if self.playing else "[PAUSED]"

        self.cv2.putText(
            composed, info_text, (10, 25),
            self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        self.cv2.putText(
            composed, info_text, (10, 25),
            self.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )

        # Controls hint
        controls = "Space:Play/Pause  Arrows:Navigate  +/-:Speed  Q:Quit"
        self.cv2.putText(
            composed, controls, (10, composed.shape[0] - 10),
            self.cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1
        )

        return composed

    def show(self) -> None:
        """Display the viewer window."""
        window_name = f"Robodata Viewer - {self.backend.get_name()}"
        self.cv2.namedWindow(window_name, self.cv2.WINDOW_NORMAL)

        fps = self.backend.get_fps()
        frame_time = 1.0 / fps

        last_frame_time = time.time()

        while True:
            # Compose and display frame
            frame = self._compose_frame()
            self.cv2.imshow(window_name, frame)

            # Handle playback timing
            if self.playing:
                current_time = time.time()
                elapsed = current_time - last_frame_time
                adjusted_frame_time = frame_time / self.speed_multiplier

                if elapsed >= adjusted_frame_time:
                    last_frame_time = current_time
                    max_frame = self.backend.get_episode_length(self.current_episode) - 1
                    if self.current_frame < max_frame:
                        self.current_frame += 1
                    else:
                        self.playing = False

            # Handle input (short wait for responsive controls)
            key = self.cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or Escape
                break
            elif key == ord(' '):  # Space - toggle play
                self.playing = not self.playing
                last_frame_time = time.time()
            elif key == 83 or key == ord('d'):  # Right arrow or D
                max_frame = self.backend.get_episode_length(self.current_episode) - 1
                self.current_frame = min(self.current_frame + 1, max_frame)
                self.playing = False
            elif key == 81 or key == ord('a'):  # Left arrow or A
                self.current_frame = max(self.current_frame - 1, 0)
                self.playing = False
            elif key == 82 or key == ord('w'):  # Up arrow or W
                if self.current_episode > 0:
                    self.current_episode -= 1
                    self.current_frame = 0
                    self._render_plots(self._plot_width or 640)
                self.playing = False
            elif key == 84 or key == ord('s'):  # Down arrow or S
                if self.current_episode < self.backend.get_num_episodes() - 1:
                    self.current_episode += 1
                    self.current_frame = 0
                    self._render_plots(self._plot_width or 640)
                self.playing = False
            elif key == ord('+') or key == ord('='):
                self.speed_multiplier = min(self.speed_multiplier * 1.5, 8.0)
            elif key == ord('-') or key == ord('_'):
                self.speed_multiplier = max(self.speed_multiplier / 1.5, 0.25)

        self.cv2.destroyAllWindows()


def cv_visualize(
    dataset_path: str | Path,
    max_episodes: int = 50,
) -> None:
    """Visualize a dataset using the fast OpenCV viewer.

    Args:
        dataset_path: Path to dataset (any supported format).
        max_episodes: Maximum episodes to load (default 50).
    """
    viewer = CVViewer(dataset_path, max_episodes)
    viewer.show()
