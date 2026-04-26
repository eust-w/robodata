"""HDF5 format reader for Robodata.

Reads HDF5-based robotics datasets, commonly used by:
- robomimic (https://robomimic.github.io/)
- ACT/ALOHA (https://github.com/tonyzhaozh/act)
- Mobile ALOHA

HDF5 datasets typically have two layouts:

1. Robomimic style (single HDF5 with multiple demos):
    /data/
        demo_0/
            actions         (T, action_dim)
            obs/
                agentview_image (T, H, W, C)
                robot0_eef_pos  (T, 3)
                ...
            rewards         (T,)
            dones           (T,)
        demo_1/
            ...
    /mask/
        train             episode indices for train split
        valid             episode indices for valid split

2. ACT/ALOHA style (one HDF5 per episode):
    /action             (T, action_dim)
    /observations/
        images/
            cam_high        (T, H, W, C)
            cam_left_wrist  (T, H, W, C)
        qpos                (T, state_dim)
        qvel                (T, state_dim)
"""

from __future__ import annotations

import json
import re
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
from robodata.formats.registry import FormatRegistry
from robodata.formats.common import numpy_dtype_to_dtype, extract_camera_name_from_key

if TYPE_CHECKING:
    import h5py
    from numpy.typing import NDArray


def _check_h5py() -> None:
    """Check if h5py is available."""
    try:
        import h5py  # noqa: F401
    except ImportError:
        raise MissingDependencyError(
            dependency="h5py",
            feature="HDF5 format support",
            install_hint="pip install h5py",
        )


@FormatRegistry.register_reader("hdf5")
class HDF5Reader:
    """Reader for HDF5-based robotics datasets.

    Supports both robomimic-style (single file, multiple demos) and
    ACT/ALOHA-style (one file per episode) datasets.
    """

    @property
    def format_name(self) -> str:
        """Return format identifier."""
        return "hdf5"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check if path is an HDF5 dataset.

        Detects:
        - Single .hdf5 or .h5 file
        - Directory containing multiple .hdf5/.h5 files (episode-per-file)

        Args:
            path: Path to potential HDF5 dataset.

        Returns:
            True if HDF5 markers found.
        """
        if not path.exists():
            return False

        # Single HDF5 file
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in (".hdf5", ".h5"):
                return True
            return False

        # Directory - check for HDF5 files
        if path.is_dir():
            # Look for .hdf5 or .h5 files
            hdf5_files = list(path.glob("*.hdf5")) + list(path.glob("*.h5"))
            if hdf5_files:
                return True

            # Check subdirectories (e.g., data/ or episodes/)
            for subdir in ["data", "episodes", "demos"]:
                subpath = path / subdir
                if subpath.exists():
                    hdf5_files = list(subpath.glob("*.hdf5")) + list(subpath.glob("*.h5"))
                    if hdf5_files:
                        return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect HDF5 dataset version from attributes.

        Args:
            path: Path to dataset.

        Returns:
            Version string or None.
        """
        _check_h5py()
        import h5py

        try:
            hdf5_path = cls._get_main_file(path)
            if hdf5_path is None:
                return None

            with h5py.File(hdf5_path, "r") as f:
                # Check common version attributes
                for attr in ["version", "format_version", "robomimic_version"]:
                    if attr in f.attrs:
                        return str(f.attrs[attr])

                # Check data group attrs
                if "data" in f and hasattr(f["data"], "attrs"):
                    for attr in ["env_args"]:
                        if attr in f["data"].attrs:
                            return "robomimic"

        except Exception:
            pass

        return None

    @classmethod
    def _get_main_file(cls, path: Path) -> Path | None:
        """Get the main HDF5 file from path."""
        if path.is_file():
            return path

        if path.is_dir():
            # Try to find a single main file
            hdf5_files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
            if hdf5_files:
                return hdf5_files[0]

            # Check subdirectories
            for subdir in ["data", "episodes", "demos"]:
                subpath = path / subdir
                if subpath.exists():
                    hdf5_files = sorted(subpath.glob("*.hdf5")) + sorted(subpath.glob("*.h5"))
                    if hdf5_files:
                        return hdf5_files[0]

        return None

    @classmethod
    def _get_all_files(cls, path: Path) -> list[Path]:
        """Get all HDF5 files for episode-per-file datasets."""
        if path.is_file():
            return [path]

        hdf5_files: list[Path] = []
        if path.is_dir():
            hdf5_files = sorted(path.glob("*.hdf5")) + sorted(path.glob("*.h5"))
            if not hdf5_files:
                for subdir in ["data", "episodes", "demos"]:
                    subpath = path / subdir
                    if subpath.exists():
                        hdf5_files = sorted(subpath.glob("*.hdf5")) + sorted(subpath.glob("*.h5"))
                        if hdf5_files:
                            break

        return hdf5_files

    @classmethod
    def _detect_layout(cls, file_handle: h5py.File) -> str:
        """Detect the HDF5 layout style.

        Returns:
            'robomimic' for single-file multi-demo layout
            'aloha' for episode-per-file layout
        """
        # Robomimic style has /data/demo_X groups
        if "data" in file_handle:
            dataset_group = file_handle["data"]
            demo_groups = [key for key in dataset_group.keys() if key.startswith("demo_")]
            if demo_groups:
                return "robomimic"

        # ALOHA style has /action or /observations at root
        if "action" in file_handle or "observations" in file_handle:
            return "aloha"

        # Check for qpos/qvel at root (some ALOHA variants)
        if "qpos" in file_handle or "images" in file_handle:
            return "aloha"

        # Default to robomimic if data group exists
        if "data" in file_handle:
            return "robomimic"

        return "unknown"

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze HDF5 dataset structure.

        Args:
            path: Path to HDF5 dataset.

        Returns:
            DatasetInfo with schema and metadata.

        Raises:
            InspectionError: If inspection fails.
        """
        _check_h5py()
        import h5py

        path = Path(path)
        info = DatasetInfo(path=path, format="hdf5")
        info.format_version = self.detect_version(path)

        hdf5_files = self._get_all_files(path)
        if not hdf5_files:
            raise InspectionError(path, "No HDF5 files found")

        main_file = hdf5_files[0]

        try:
            with h5py.File(main_file, "r") as f:
                layout = self._detect_layout(f)
                info.format_version = layout

                if layout == "robomimic":
                    self._inspect_robomimic(file_handle, info)
                else:
                    self._inspect_aloha(file_handle, info, len(hdf5_files))

        except Exception as e:
            raise InspectionError(path, f"Failed to read HDF5 file: {e}")

        return info

    def _inspect_robomimic(self, file_handle: h5py.File, info: DatasetInfo) -> None:
        """Inspect robomimic-style HDF5 file."""
        import h5py

        # Get episode count from data group
        if "data" not in file_handle:
            return

        dataset_group = file_handle["data"]
        demo_keys = sorted(
            [key for key in dataset_group.keys() if key.startswith("demo_")],
            key=lambda demo_name: int(demo_name.split("_")[1]),
        )
        info.num_episodes = len(demo_keys)

        # Load metadata from attrs
        if "env_args" in dataset_group.attrs:
            try:
                env_args = json.loads(dataset_group.attrs["env_args"])
                info.inferred_fps = env_args.get("env_kwargs", {}).get("control_freq")
                robots = env_args.get("env_kwargs", {}).get("robots", [])
                if robots:
                    info.inferred_robot_type = robots[0] if isinstance(robots, list) else robots
            except (json.JSONDecodeError, KeyError):
                pass

        if "total" in dataset_group.attrs:
            info.total_frames = int(dataset_group.attrs["total"])

        # Analyze first demo structure
        if demo_keys:
            demo = dataset_group[demo_keys[0]]
            num_samples = demo.attrs.get("num_samples", 0)

            # Get action schema
            if "actions" in demo:
                actions = demo["actions"]
                info.action_schema = FieldSchema(
                    name="actions",
                    shape=actions.shape[1:] if len(actions.shape) > 1 else (),
                    dtype=numpy_dtype_to_dtype(actions.dtype),
                )

            # Analyze observations
            if "obs" in demo:
                self._analyze_obs_group(demo["obs"], info)

            # Calculate total frames
            if info.total_frames == 0:
                total = 0
                for demo_key in demo_keys:
                    demo_data = dataset_group[demo_key]
                    if "actions" in demo_data:
                        total += demo_data["actions"].shape[0]
                info.total_frames = total

    def _inspect_aloha(self, file_handle: h5py.File, info: DatasetInfo, num_files: int) -> None:
        """Inspect ALOHA-style HDF5 file."""
        info.num_episodes = num_files

        # Get action schema
        if "action" in file_handle:
            action = file_handle["action"]
            info.action_schema = FieldSchema(
                name="action",
                shape=action.shape[1:] if len(action.shape) > 1 else (),
                dtype=numpy_dtype_to_dtype(action.dtype),
            )
            info.total_frames = action.shape[0]

        # Analyze observations
        if "observations" in file_handle:
            obs_group = file_handle["observations"]

            # Check for images group
            if "images" in obs_group:
                self._analyze_images_group(obs_group["images"], info)

            # Check for qpos/qvel
            for key in ["qpos", "qvel"]:
                if key in obs_group:
                    array_data = obs_group[key]
                    info.observation_schema[key] = FieldSchema(
                        name=key,
                        shape=array_data.shape[1:] if len(array_data.shape) > 1 else (),
                        dtype=numpy_dtype_to_dtype(array_data.dtype),
                    )

        # ALOHA variant with images at root
        if "images" in file_handle:
            self._analyze_images_group(file_handle["images"], info)

        # Direct qpos/qvel at root
        for key in ["qpos", "qvel"]:
            if key in file_handle:
                array_data = file_handle[key]
                info.observation_schema[key] = FieldSchema(
                    name=key,
                    shape=array_data.shape[1:] if len(array_data.shape) > 1 else (),
                    dtype=numpy_dtype_to_dtype(array_data.dtype),
                )
                if info.total_frames == 0:
                    info.total_frames = array_data.shape[0]

    def _analyze_obs_group(self, obs_group: h5py.Group, info: DatasetInfo) -> None:
        """Analyze observation group for robomimic datasets."""
        import h5py

        for key in obs_group.keys():
            item = obs_group[key]
            if isinstance(item, h5py.Dataset):
                arr = item
                if self._is_image_dataset(key, arr):
                    cam_name = extract_camera_name_from_key(key)
                    info.cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=arr.shape[1] if len(arr.shape) > 1 else 84,
                        width=arr.shape[2] if len(arr.shape) > 2 else 84,
                        channels=arr.shape[3] if len(arr.shape) > 3 else 3,
                    )
                else:
                    info.observation_schema[key] = FieldSchema(
                        name=key,
                        shape=arr.shape[1:] if len(arr.shape) > 1 else (),
                        dtype=numpy_dtype_to_dtype(arr.dtype),
                    )

    def _analyze_images_group(self, images_group: h5py.Group, info: DatasetInfo) -> None:
        """Analyze images group for ALOHA datasets."""
        import h5py

        for key in images_group.keys():
            item = images_group[key]
            if isinstance(item, h5py.Dataset):
                arr = item
                info.cameras[key] = CameraInfo(
                    name=key,
                    height=arr.shape[1] if len(arr.shape) > 1 else 480,
                    width=arr.shape[2] if len(arr.shape) > 2 else 640,
                    channels=arr.shape[3] if len(arr.shape) > 3 else 3,
                )

    def _is_image_dataset(self, key: str, arr: h5py.Dataset) -> bool:
        """Check if dataset contains image data."""
        import numpy as np

        key_lower = key.lower()

        # Check by name
        if any(x in key_lower for x in ["image", "rgb", "camera", "img", "frame"]):
            return True

        # Check by shape (T, H, W, C) with C in [1, 3, 4] and uint8
        shape = arr.shape
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            if np.dtype(arr.dtype) == np.uint8:
                return True

        return False

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over HDF5 episodes.

        Args:
            path: Path to HDF5 dataset.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_h5py()
        import h5py

        path = Path(path)
        hdf5_files = self._get_all_files(path)

        if not hdf5_files:
            return

        # Check layout from first file
        with h5py.File(hdf5_files[0], "r") as f:
            layout = self._detect_layout(f)

        if layout == "robomimic":
            yield from self._read_robomimic_episodes(hdf5_files[0])
        else:
            yield from self._read_aloha_episodes(hdf5_files)

    def _read_robomimic_episodes(self, file_path: Path) -> Iterator[Episode]:
        """Read episodes from robomimic-style HDF5 file."""
        import h5py

        with h5py.File(file_path, "r") as f:
            if "data" not in f:
                return

            data_group = f["data"]
            demo_keys = sorted(
                [k for k in data_group.keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[1]),
            )

            for ep_idx, demo_key in enumerate(demo_keys):
                demo = data_group[demo_key]

                def make_frame_loader(
                    fp: Path = file_path, dk: str = demo_key
                ) -> Iterator[Frame]:
                    yield from self._load_robomimic_frames(fp, dk)

                yield Episode(
                    episode_id=str(ep_idx),
                    _frame_loader=make_frame_loader,
                )

    def _load_robomimic_frames(self, file_path: Path, demo_key: str) -> Iterator[Frame]:
        """Load frames from a robomimic demo group."""
        import h5py
        import numpy as np

        with h5py.File(file_path, "r") as f:
            demo = f["data"][demo_key]
            num_frames = demo["actions"].shape[0]

            for frame_idx in range(num_frames):
                images: dict[str, LazyImage] = {}
                state = None
                action = None

                # Load action
                if "actions" in demo:
                    action = np.array(demo["actions"][frame_idx])

                # Load observations
                if "obs" in demo:
                    obs = demo["obs"]
                    for key in obs.keys():
                        arr = obs[key]
                        if self._is_image_dataset(key, arr):
                            cam_name = extract_camera_name_from_key(key)
                            # Create lazy loader with captured values
                            img_data = np.array(arr[frame_idx])
                            images[cam_name] = LazyImage(
                                loader=lambda d=img_data: d,
                                height=arr.shape[1],
                                width=arr.shape[2],
                                channels=arr.shape[3] if len(arr.shape) > 3 else 3,
                            )
                        elif key in ["robot0_joint_pos", "robot0_eef_pos"]:
                            if state is None:
                                state = np.array(arr[frame_idx])
                            else:
                                state = np.concatenate([state, np.array(arr[frame_idx])])

                yield Frame(
                    index=frame_idx,
                    images=images,
                    state=state,
                    action=action,
                    is_first=frame_idx == 0,
                )

    def _read_aloha_episodes(self, file_paths: list[Path]) -> Iterator[Episode]:
        """Read episodes from ALOHA-style HDF5 files."""
        # Sort files by episode number if possible
        def extract_episode_num(p: Path) -> int:
            match = re.search(r"(\d+)", p.stem)
            return int(match.group(1)) if match else 0

        sorted_files = sorted(file_paths, key=extract_episode_num)

        for ep_idx, file_path in enumerate(sorted_files):

            def make_frame_loader(fp: Path = file_path) -> Iterator[Frame]:
                yield from self._load_aloha_frames(fp)

            yield Episode(
                episode_id=str(ep_idx),
                _frame_loader=make_frame_loader,
            )

    def _load_aloha_frames(self, file_path: Path) -> Iterator[Frame]:
        """Load frames from an ALOHA-style HDF5 file."""
        import h5py
        import numpy as np

        with h5py.File(file_path, "r") as f:
            # Determine number of frames
            num_frames = 0
            if "action" in f:
                num_frames = f["action"].shape[0]
            elif "qpos" in f:
                num_frames = f["qpos"].shape[0]
            elif "observations" in f and "qpos" in f["observations"]:
                num_frames = f["observations"]["qpos"].shape[0]

            if num_frames == 0:
                return

            for frame_idx in range(num_frames):
                images: dict[str, LazyImage] = {}
                state = None
                action = None

                # Load action
                if "action" in f:
                    action = np.array(f["action"][frame_idx])

                # Load state (qpos/qvel)
                if "qpos" in f:
                    state = np.array(f["qpos"][frame_idx])
                elif "observations" in f and "qpos" in f["observations"]:
                    state = np.array(f["observations"]["qpos"][frame_idx])

                # Load images
                images_group = None
                if "images" in f:
                    images_group = f["images"]
                elif "observations" in f and "images" in f["observations"]:
                    images_group = f["observations"]["images"]

                if images_group:
                    for cam_name in images_group.keys():
                        arr = images_group[cam_name]
                        img_data = np.array(arr[frame_idx])
                        images[cam_name] = LazyImage(
                            loader=lambda d=img_data: d,
                            height=arr.shape[1],
                            width=arr.shape[2],
                            channels=arr.shape[3] if len(arr.shape) > 3 else 3,
                        )

                yield Frame(
                    index=frame_idx,
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
