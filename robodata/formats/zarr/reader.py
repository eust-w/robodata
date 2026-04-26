"""Zarr format reader for Robodata.

Reads Zarr-based robotics datasets, commonly used by UMI and similar projects.
Zarr stores data as chunked, compressed arrays with episode boundaries in metadata.
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
from robodata.core.typing_utils import ensure_numpy, ArrayLike
from robodata.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyImage,
)
from robodata.formats.registry import FormatRegistry
from robodata.formats.common import numpy_dtype_to_dtype, extract_camera_name_from_key

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_zarr() -> None:
    """Check if zarr is available."""
    try:
        import zarr  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="zarr",
            feature="Zarr format support",
            install_hint="pip install robodata[zarr]",
        )


@FormatRegistry.register_reader("zarr")
class ZarrReader:
    """Reader for Zarr-based robotics datasets.

    Zarr datasets typically have a structure like:
        /data/
            camera0_rgb         (frames, H, W, C)
            robot0_eef_pos      (frames, 3)
            robot0_gripper_width (frames, 1)
            action              (frames, action_dim)
        /meta/
            episode_ends        [end_idx_0, end_idx_1, ...]

    This reader supports:
    - Auto-detection via .zarr directories or .zarr.zip files
    - Lazy episode and frame loading
    - Schema inference from array shapes and dtypes
    """

    @property
    def format_name(self) -> str:
        """Return format identifier."""
        return "zarr"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for Zarr markers: .zarr directory, .zarr.zip file, or .zarray files.

        Args:
            path: Path to potential Zarr dataset.

        Returns:
            True if Zarr markers found.
        """
        if not path.exists():
            return False

        # Check for .zarr.zip file
        if path.is_file() and (path.name.endswith(".zarr.zip") or path.name.endswith(".zarr")):
            return True

        # Check for .zarr directory
        if path.is_dir():
            # Look for .zarray or .zgroup files (zarr v2)
            if (path / ".zarray").exists() or (path / ".zgroup").exists():
                return True
            # Look for zarr.json (zarr v3)
            if (path / "zarr.json").exists():
                return True
            # Check if path ends with .zarr
            if path.suffix == ".zarr":
                return True
            # Check for nested structure with data/ and meta/
            if (path / "data").exists() or (path / "meta").exists():
                # Verify it has zarr structure inside
                data_dir = path / "data"
                if data_dir.exists():
                    for subdir in data_dir.iterdir():
                        if subdir.is_dir() and (subdir / ".zarray").exists():
                            return True
            # Check subdirectories for .zarr
            for subpath in path.iterdir():
                if subpath.is_dir() and subpath.suffix == ".zarr":
                    return True
                if subpath.name.endswith(".zarr.zip"):
                    return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect Zarr format version.

        Args:
            path: Path to dataset.

        Returns:
            Version string or None.
        """
        _check_zarr()
        import zarr

        try:
            root = zarr.open(str(path), mode="r")
            # Check for version in attrs
            if hasattr(root, "attrs"):
                version = root.attrs.get("version", root.attrs.get("format_version"))
                if version:
                    return str(version)
        except Exception:
            pass

        return None

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze Zarr dataset structure.

        Args:
            path: Path to Zarr dataset.

        Returns:
            DatasetInfo with schema and metadata.

        Raises:
            InspectionError: If inspection fails.
        """
        _check_zarr()
        import zarr

        path = Path(path)
        info = DatasetInfo(path=path, format="zarr")
        info.format_version = self.detect_version(path)

        try:
            root = zarr.open(str(path), mode="r")
        except Exception as e:
            raise InspectionError(path, f"Failed to open Zarr store: {e}")

        # Load metadata from attrs
        self._load_attrs(root, info)

        # Analyze data arrays
        self._analyze_data_group(root, info)

        # Load episode boundaries
        self._load_episode_ends(root, info)

        return info

    def _load_attrs(self, root: Any, info: DatasetInfo) -> None:
        """Load metadata from Zarr attributes."""
        if not hasattr(root, "attrs"):
            return

        attrs = dict(root.attrs)

        # Common metadata fields
        info.inferred_fps = attrs.get("fps", attrs.get("frame_rate"))
        info.inferred_robot_type = attrs.get("robot_type", attrs.get("robot"))

        if "total_episodes" in attrs:
            info.num_episodes = int(attrs["total_episodes"])
        if "total_frames" in attrs or "total_steps" in attrs:
            total = attrs.get("total_frames", attrs.get("total_steps", 0))
            info.total_frames = int(total) if total is not None else 0

    def _analyze_data_group(self, root: Any, info: DatasetInfo) -> None:
        """Analyze arrays in the data group."""
        import zarr

        # Find data group - could be root, root/data, or nested
        dataset_group = None
        if "data" in root:
            dataset_group = root["data"]
        elif hasattr(root, "keys") and any(
            key in root for key in ["camera0_rgb", "action", "robot0_eef_pos"]
        ):
            dataset_group = root
        else:
            # Try to find data arrays at root level
            dataset_group = root

        if dataset_group is None:
            return

        # Iterate through arrays
        for key in dataset_group.keys():
            try:
                array_data = dataset_group[key]
                if not isinstance(array_data, zarr.Array):
                    continue

                shape = array_data.shape
                dtype = array_data.dtype

                # Detect cameras (4D arrays with last dim 1, 3, or 4)
                if self._is_image_array(key, shape, dtype):
                    cam_name = extract_camera_name_from_key(key)
                    # Shape is (frames, H, W, C)
                    info.cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=shape[1] if len(shape) > 1 else 480,
                        width=shape[2] if len(shape) > 2 else 640,
                        channels=shape[3] if len(shape) > 3 else 3,
                    )
                elif "action" in key.lower():
                    # Action array
                    info.action_schema = FieldSchema(
                        name=key,
                        shape=shape[1:] if len(shape) > 1 else (),
                        dtype=numpy_dtype_to_dtype(dtype),
                    )
                elif not self._is_metadata_key(key):
                    # Regular observation
                    info.observation_schema[key] = FieldSchema(
                        name=key,
                        shape=shape[1:] if len(shape) > 1 else (),
                        dtype=numpy_dtype_to_dtype(dtype),
                    )

                # Update total frames from array length
                if len(shape) > 0 and info.total_frames == 0:
                    info.total_frames = shape[0]

            except Exception:
                continue

    def _is_image_array(self, key: str, shape: tuple, dtype: Any) -> bool:
        """Check if array represents image data."""
        np = ensure_numpy()

        key_lower = key.lower()

        # Check by name
        if any(x in key_lower for x in ["image", "rgb", "camera", "img", "frame", "video"]):
            return True

        # Check by shape and dtype (4D with last dim in [1, 3, 4] and uint8)
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            if np.dtype(dtype) == np.uint8:
                return True

        return False

    def _is_metadata_key(self, key: str) -> bool:
        """Check if key is metadata rather than observation."""
        key_lower = key.lower()
        return key_lower in [
            "episode_ends",
            "episode_end",
            "episode_idx",
            "episode_index",
            "timestamp",
            "timestamps",
        ]

    def _load_episode_ends(self, root: Any, info: DatasetInfo) -> None:
        """Load episode boundary information."""
        np = ensure_numpy()

        # Check meta group first
        episode_ends = None
        if "meta" in root and "episode_ends" in root["meta"]:
            episode_ends = np.array(root["meta"]["episode_ends"])
        elif "episode_ends" in root:
            episode_ends = np.array(root["episode_ends"])

        if episode_ends is not None:
            info.num_episodes = len(episode_ends)
            total = root.attrs.get("total_frames", root.attrs.get("total_steps", 0))
            info.total_frames = int(total) if total is not None else episode_ends[-1]

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over Zarr episodes.

        Args:
            path: Path to Zarr dataset.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_zarr()
        np = ensure_numpy()
        import zarr

        path = Path(path)
        root = zarr.open(str(path), mode="r")

        # Get episode boundaries
        episode_ends = self._get_episode_ends(root)
        if episode_ends is None:
            # No episode metadata - treat entire dataset as one episode
            total_frames = self._get_total_frames(root)
            episode_ends = np.array([total_frames])

        # Get data group
        dataset_group = root["data"] if "data" in root else root

        # Iterate through episodes
        start_idx = 0
        for episode_idx, end_idx in enumerate(episode_ends):
            end_idx = int(end_idx)
            episode_id = str(episode_idx)

            # Create lazy frame loader for this episode
            def make_frame_loader(data: Any, start: int, end: int) -> Iterator[Frame]:
                for frame_idx in range(end - start):
                    abs_idx = start + frame_idx
                    yield self._load_frame(data, abs_idx, frame_idx)

            yield Episode(
                episode_id=episode_id,
                _frame_loader=lambda d=dataset_group, s=start_idx, e=end_idx: make_frame_loader(  # type: ignore[misc]
                    d, s, e
                ),
            )

            start_idx = end_idx

    def _get_episode_ends(self, root: Any) -> NDArray[Any] | None:
        """Get episode end indices."""
        np = ensure_numpy()

        if "meta" in root and "episode_ends" in root["meta"]:
            return np.array(root["meta"]["episode_ends"])
        elif "episode_ends" in root:
            return np.array(root["episode_ends"])
        return None

    def _get_total_frames(self, root: Any) -> int:
        """Get total number of frames from data arrays."""
        dataset_group = root["data"] if "data" in root else root
        for key in dataset_group.keys():
            array_data = dataset_group[key]
            if hasattr(array_data, "shape") and len(array_data.shape) > 0:
                return int(array_data.shape[0])
        return 0

    def _load_frame(self, dataset_group: Any, abs_idx: int, frame_idx: int) -> Frame:
        """Load a single frame from the data group."""
        np = ensure_numpy()

        images: dict[str, LazyImage] = {}
        state = None
        action = None

        for key in dataset_group.keys():
            array_data = dataset_group[key]
            if not hasattr(array_data, "shape"):
                continue

            shape = array_data.shape
            dtype = array_data.dtype

            if self._is_image_array(key, shape, dtype):
                cam_name = extract_camera_name_from_key(key)

                def make_loader(a: Any = array_data, idx: int = abs_idx) -> NDArray[Any]:
                    return np.array(a[idx])

                images[cam_name] = LazyImage(
                    loader=make_loader,
                    height=shape[1] if len(shape) > 1 else 480,
                    width=shape[2] if len(shape) > 2 else 640,
                    channels=shape[3] if len(shape) > 3 else 3,
                )
            elif "action" in key.lower():
                action = np.array(array_data[abs_idx])
            elif not self._is_metadata_key(key):
                # State/observation data
                if state is None:
                    state = {}
                state[key] = np.array(array_data[abs_idx])

        return Frame(
            index=frame_idx,
            timestamp=None,
            images=images,
            state=state,
            action=action,
            is_first=frame_idx == 0,
        )

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to dataset.
            episode_id: Episode identifier (0-indexed).

        Returns:
            The requested Episode.

        Raises:
            EpisodeNotFoundError: If episode not found.
        """
        for episode in self.read_episodes(path):
            if episode.episode_id == episode_id:
                return episode

        raise EpisodeNotFoundError(episode_id, path)
