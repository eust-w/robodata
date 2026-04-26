"""RoboDM format reader for Robodata.

Reads .vla trajectory files from Berkeley's RoboDM format.
See: https://github.com/BerkeleyAutomation/robodm

Structure:
    dataset/
    ├── trajectory_0.vla
    ├── trajectory_1.vla
    └── ...

Or single file:
    trajectory.vla

Each .vla file contains one trajectory with hierarchical keys:
    - camera/rgb, camera/depth (images)
    - robot/joint_positions, robot/joint_velocities (state)
    - action/gripper_action, action/joint_positions (actions)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robodata.core.exceptions import (
    EpisodeNotFoundError,
    InspectionError,
    MissingDependencyError,
)
from robodata.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyImage,
)
from robodata.formats.common import numpy_dtype_to_dtype
from robodata.formats.registry import FormatRegistry

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_robodm() -> None:
    """Check if robodm is available."""
    try:
        import robodm  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="robodm",
            feature="RoboDM format support",
            install_hint="pip install robodm or git clone https://github.com/BerkeleyAutomation/robodm && pip install -e robodm",
        )


def _find_vla_files(path: Path) -> list[Path]:
    """Find all .vla files in a directory or return single file."""
    if path.is_file() and path.suffix == ".vla":
        return [path]
    elif path.is_dir():
        vla_files = sorted(path.glob("*.vla"))
        # Also check subdirectories one level deep
        if not vla_files:
            vla_files = sorted(path.glob("*/*.vla"))
        return vla_files
    return []


def _parse_robodm_keys(data: dict[str, Any]) -> dict[str, str]:
    """Parse RoboDM hierarchical keys into categories.

    Returns mapping of key -> category (camera, state, action, other).
    """
    categories = {}
    for key in data.keys():
        key_lower = key.lower()
        if any(cam in key_lower for cam in ["camera", "image", "rgb", "depth", "video"]):
            categories[key] = "camera"
        elif any(act in key_lower for act in ["action", "command", "target"]):
            categories[key] = "action"
        elif any(state in key_lower for state in ["robot", "joint", "qpos", "qvel", "state", "eef", "gripper"]):
            categories[key] = "state"
        else:
            categories[key] = "other"
    return categories


@FormatRegistry.register_reader("robodm")
class RoboDMReader:
    """Reader for RoboDM .vla format.

    RoboDM stores trajectories in .vla files with efficient video compression.
    Each file contains one trajectory (episode) with hierarchical key structure.

    Example:
        >>> reader = RoboDMReader()
        >>> for episode in reader.read_episodes(Path("./dataset")):
        ...     for frame in episode.frames():
        ...         print(frame.state)
    """

    @property
    def format_name(self) -> str:
        return "robodm"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check if path contains .vla files."""
        path = Path(path)
        if path.is_file():
            return path.suffix == ".vla"
        elif path.is_dir():
            return len(_find_vla_files(path)) > 0
        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """RoboDM doesn't have explicit versions yet."""
        return None

    def inspect(self, path: Path) -> DatasetInfo:
        """Inspect RoboDM dataset structure."""
        _check_robodm()
        import robodm
        import numpy as np

        path = Path(path)
        vla_files = _find_vla_files(path)

        if not vla_files:
            raise InspectionError(f"No .vla files found in {path}")

        # Inspect first trajectory for schema
        trajectory = robodm.Trajectory(path=str(vla_files[0]), mode="r")
        data = trajectory.load()

        # RoboDM sometimes adds a batch dimension - squeeze it out
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim > 1 and data[key].shape[0] == 1:
                data[key] = data[key].squeeze(axis=0)

        key_categories = _parse_robodm_keys(data)

        # Extract schema info
        cameras: dict[str, CameraInfo] = {}
        observation_schema: dict[str, FieldSchema] = {}
        action_schema: FieldSchema | None = None
        action_shape: tuple[int, ...] = ()
        num_frames = 0

        for key, values in data.items():
            if not isinstance(values, (list, np.ndarray)) or len(values) == 0:
                continue

            sample = values[0] if isinstance(values, list) else values[0]
            num_frames = max(num_frames, len(values))
            category = key_categories.get(key, "other")

            if category == "camera":
                # Image data
                if hasattr(sample, "shape") and len(sample.shape) >= 2:
                    h, w = sample.shape[:2]
                    c = sample.shape[2] if len(sample.shape) > 2 else 1
                    cam_name = key.replace("/", "_").replace("camera_", "").replace("image_", "")
                    cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=h,
                        width=w,
                        channels=c,
                        encoding="rgb" if c == 3 else ("depth" if c == 1 else "rgb"),
                    )
            elif category == "state":
                if hasattr(sample, "shape"):
                    observation_schema[key] = FieldSchema(
                        name=key,
                        shape=tuple(sample.shape),
                        dtype=numpy_dtype_to_dtype(sample.dtype),
                    )
            elif category == "action":
                if hasattr(sample, "shape"):
                    # Accumulate action shape (we'll combine if multiple action keys)
                    action_shape = tuple(sample.shape)
                    action_schema = FieldSchema(
                        name=key,
                        shape=action_shape,
                        dtype=numpy_dtype_to_dtype(sample.dtype),
                    )

        return DatasetInfo(
            path=path,
            format="robodm",
            format_version=None,
            num_episodes=len(vla_files),
            total_frames=num_frames * len(vla_files),  # Estimate
            cameras=cameras,
            observation_schema=observation_schema,
            action_schema=action_schema,
            metadata={
                "vla_files": [str(f) for f in vla_files],
                "keys": list(data.keys()),
            },
        )

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Iterate over all trajectories as Episodes."""
        _check_robodm()

        path = Path(path)
        vla_files = _find_vla_files(path)

        for idx, vla_path in enumerate(vla_files):
            yield self._read_vla_as_episode(vla_path, idx)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID."""
        _check_robodm()

        path = Path(path)
        vla_files = _find_vla_files(path)

        # Try to find by index or filename
        try:
            idx = int(episode_id)
            if 0 <= idx < len(vla_files):
                return self._read_vla_as_episode(vla_files[idx], idx)
        except ValueError:
            # Try matching by filename
            for idx, vla_path in enumerate(vla_files):
                if vla_path.stem == episode_id or str(vla_path) == episode_id:
                    return self._read_vla_as_episode(vla_path, idx)

        raise EpisodeNotFoundError(episode_id, path)

    def _read_vla_as_episode(self, vla_path: Path, episode_idx: int) -> Episode:
        """Convert a single .vla file to an Episode."""
        import robodm
        import numpy as np

        trajectory = robodm.Trajectory(path=str(vla_path), mode="r")
        data = trajectory.load()

        # RoboDM sometimes adds a batch dimension - squeeze it out
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim > 1 and data[key].shape[0] == 1:
                data[key] = data[key].squeeze(axis=0)

        key_categories = _parse_robodm_keys(data)

        # Identify cameras, state, and action keys
        camera_keys: dict[str, str] = {}  # cam_name -> original_key
        state_keys: list[str] = []
        action_keys: list[str] = []

        cameras: dict[str, CameraInfo] = {}

        for key, values in data.items():
            if not isinstance(values, (list, np.ndarray)) or len(values) == 0:
                continue

            sample = values[0] if isinstance(values, list) else values[0]
            category = key_categories.get(key, "other")

            if category == "camera" and hasattr(sample, "shape") and len(sample.shape) >= 2:
                cam_name = key.replace("/", "_").replace("camera_", "").replace("image_", "")
                camera_keys[cam_name] = key
                h, w = sample.shape[:2]
                c = sample.shape[2] if len(sample.shape) > 2 else 1
                cameras[cam_name] = CameraInfo(
                    name=cam_name,
                    height=h,
                    width=w,
                    channels=c,
                )
            elif category == "state":
                state_keys.append(key)
            elif category == "action":
                action_keys.append(key)

        # Determine number of frames
        num_frames = 0
        for key in list(camera_keys.values()) + state_keys + action_keys:
            if key in data:
                num_frames = max(num_frames, len(data[key]))

        def create_frame_loader():
            """Create lazy frame iterator."""
            for frame_idx in range(num_frames):
                # Build images dict with lazy loading
                images: dict[str, LazyImage] = {}
                for cam_name, orig_key in camera_keys.items():
                    cam_data = data[orig_key]
                    if frame_idx < len(cam_data):
                        img = cam_data[frame_idx]
                        h, w = img.shape[:2]
                        c = img.shape[2] if len(img.shape) > 2 else 1

                        # Capture by value for closure
                        def make_loader(image_data):
                            return lambda: image_data

                        images[cam_name] = LazyImage(
                            loader=make_loader(img),
                            height=h,
                            width=w,
                            channels=c,
                        )

                # Concatenate state
                state = None
                if state_keys:
                    state_parts = []
                    for key in state_keys:
                        if key in data and frame_idx < len(data[key]):
                            val = data[key][frame_idx]
                            state_parts.append(np.atleast_1d(val).flatten())
                    if state_parts:
                        state = np.concatenate(state_parts)

                # Concatenate action
                action = None
                if action_keys:
                    action_parts = []
                    for key in action_keys:
                        if key in data and frame_idx < len(data[key]):
                            val = data[key][frame_idx]
                            action_parts.append(np.atleast_1d(val).flatten())
                    if action_parts:
                        action = np.concatenate(action_parts)

                yield Frame(
                    index=frame_idx,
                    timestamp=float(frame_idx) / 30.0,  # Assume 30fps default
                    images=images,
                    state=state,
                    action=action,
                    is_first=(frame_idx == 0),
                    is_last=(frame_idx == num_frames - 1),
                )

        # Calculate state/action dims
        state_dim = None
        action_dim = None
        if state_keys and num_frames > 0:
            sample_state = []
            for key in state_keys:
                if key in data and len(data[key]) > 0:
                    sample_state.append(np.atleast_1d(data[key][0]).flatten())
            if sample_state:
                state_dim = sum(len(s) for s in sample_state)

        if action_keys and num_frames > 0:
            sample_action = []
            for key in action_keys:
                if key in data and len(data[key]) > 0:
                    sample_action.append(np.atleast_1d(data[key][0]).flatten())
            if sample_action:
                action_dim = sum(len(a) for a in sample_action)

        return Episode(
            episode_id=str(episode_idx),
            metadata={
                "source_file": str(vla_path),
                "original_keys": list(data.keys()),
            },
            cameras=cameras,
            state_dim=state_dim,
            action_dim=action_dim,
            fps=30.0,  # Default, could be extracted from metadata if available
            _frame_loader=create_frame_loader,
        )
