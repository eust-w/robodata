"""RLDS format reader for Robodata.

Reads RLDS (TensorFlow-based) datasets, commonly used for Open-X robotics datasets.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Suppress TensorFlow logging BEFORE any TF import can happen
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # FATAL only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Suppress oneDNN messages
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # Google logging: ERROR+

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

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _check_tensorflow() -> None:
    """Check if TensorFlow is available and suppress verbose logging."""
    # Suppress absl logging (used by TensorFlow internally)
    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
    except ImportError:
        raise MissingDependencyError(
            dependency="tensorflow",
            feature="RLDS format support",
            install_hint="pip install robodata[rlds]",
        )


def _tf_dtype_to_dtype(tf_dtype: Any) -> Dtype:
    """Convert TensorFlow dtype to Robodata Dtype."""
    dtype_str = str(tf_dtype).lower()
    if "float32" in dtype_str:
        return Dtype.FLOAT32
    elif "float64" in dtype_str:
        return Dtype.FLOAT64
    elif "int32" in dtype_str:
        return Dtype.INT32
    elif "int64" in dtype_str:
        return Dtype.INT64
    elif "uint8" in dtype_str:
        return Dtype.UINT8
    elif "bool" in dtype_str:
        return Dtype.BOOL
    elif "string" in dtype_str:
        return Dtype.STRING
    else:
        return Dtype.FLOAT32  # Default fallback


def _shape_to_tuple(shape: Any) -> tuple[int, ...]:
    """Convert TensorFlow shape to tuple, replacing None with -1."""
    return tuple(dim if dim is not None else -1 for dim in shape)


@FormatRegistry.register_reader("rlds")
class RLDSReader:
    """Reader for RLDS (TensorFlow-based) datasets.

    RLDS datasets are typically stored as TFRecord files with a specific
    episode/step structure. This reader supports:
    - Auto-detection via tfrecord files or dataset_info.json
    - Lazy episode and frame loading
    - Schema inference from TensorFlow feature specs
    """

    @property
    def format_name(self) -> str:
        """Return format identifier."""
        return "rlds"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for RLDS markers: tfrecord files or dataset_info.json.

        Args:
            path: Path to potential RLDS dataset.

        Returns:
            True if RLDS markers found.
        """
        if not path.exists():
            return False

        if path.is_dir():
            # Check for tfrecord files
            if any(path.glob("*.tfrecord*")):
                return True
            # Check for dataset_info.json (TFDS format)
            if (path / "dataset_info.json").exists():
                return True
            # Check for features.json (alternative RLDS format)
            if (path / "features.json").exists():
                return True
            # Check subdirectories (TFDS often has version subdirs)
            for subdir in path.iterdir():
                if subdir.is_dir():
                    if any(subdir.glob("*.tfrecord*")):
                        return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect specific version if applicable.

        Args:
            path: Path to dataset.

        Returns:
            Version string or None.
        """
        # Check for version in dataset_info.json
        info_path = path / "dataset_info.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                return info.get("version", None)
            except (json.JSONDecodeError, KeyError):
                pass

        # Check for version directory pattern (e.g., "1.0.0")
        if path.is_dir():
            for subdir in path.iterdir():
                if subdir.is_dir() and subdir.name[0].isdigit():
                    return subdir.name

        return None

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze RLDS dataset structure.

        Args:
            path: Path to RLDS dataset.

        Returns:
            DatasetInfo with schema and metadata.

        Raises:
            InspectionError: If inspection fails.
        """
        _check_tensorflow()

        path = Path(path)
        info = DatasetInfo(path=path, format="rlds")
        info.format_version = self.detect_version(path)

        # Find tfrecord files
        tfrecord_files = self._find_tfrecord_files(path)
        if not tfrecord_files:
            raise InspectionError(path, "No TFRecord files found")

        # Try to load dataset_info.json for metadata
        self._load_dataset_info_json(path, info)

        # Sample first record to understand schema
        try:
            self._analyze_schema_from_records(tfrecord_files, info)
        except Exception as e:
            raise InspectionError(path, f"Failed to analyze schema: {e}")

        return info

    def _find_tfrecord_files(self, path: Path) -> list[Path]:
        """Find all TFRecord files in the dataset."""
        files = list(path.glob("*.tfrecord*"))

        # Check subdirectories (version dirs, train/test splits)
        for subdir in path.iterdir():
            if subdir.is_dir():
                files.extend(subdir.glob("*.tfrecord*"))

        return sorted(files)

    def _load_dataset_info_json(self, path: Path, info: DatasetInfo) -> None:
        """Load metadata from dataset_info.json if present."""
        info_path = path / "dataset_info.json"
        if not info_path.exists():
            # Check subdirectories
            for subdir in path.iterdir():
                if subdir.is_dir():
                    candidate = subdir / "dataset_info.json"
                    if candidate.exists():
                        info_path = candidate
                        break

        if info_path.exists():
            try:
                with open(info_path) as f:
                    data = json.load(f)

                # Extract counts from shardLengths
                if "splits" in data:
                    for split in data["splits"]:
                        if split.get("name") == "train":
                            shard_lengths = split.get("shardLengths", [])
                            if shard_lengths:
                                # Sum up episodes across all shards
                                info.num_episodes = sum(int(x) for x in shard_lengths)

                # Extract feature info
                if "features" in data:
                    self._parse_feature_spec(data["features"], info)

            except (json.JSONDecodeError, KeyError):
                pass

    def _parse_feature_spec(self, features: dict, info: DatasetInfo) -> None:
        """Parse TFDS feature specification."""
        # TFDS features are nested dicts describing the schema
        if "steps" in features:
            steps = features["steps"]
            if "observation" in steps:
                obs = steps["observation"]
                for key, spec in obs.items():
                    if self._is_image_feature(key, spec):
                        shape = spec.get("shape", [480, 640, 3])
                        info.cameras[key] = CameraInfo(
                            name=key,
                            height=shape[0] if len(shape) > 0 else 480,
                            width=shape[1] if len(shape) > 1 else 640,
                            channels=shape[2] if len(shape) > 2 else 3,
                        )
                    else:
                        shape = tuple(spec.get("shape", []))
                        dtype_str = spec.get("dtype", "float32")
                        info.observation_schema[key] = FieldSchema(
                            name=key,
                            shape=shape,
                            dtype=self._str_to_dtype(dtype_str),
                        )

            if "action" in steps:
                action_spec = steps["action"]
                shape = tuple(action_spec.get("shape", []))
                dtype_str = action_spec.get("dtype", "float32")
                info.action_schema = FieldSchema(
                    name="action",
                    shape=shape,
                    dtype=self._str_to_dtype(dtype_str),
                )

            if "reward" in steps:
                info.has_rewards = True

            if "language_instruction" in features or "language_instruction" in steps:
                info.has_language = True

    def _is_image_feature(self, key: str, spec: dict) -> bool:
        """Check if a feature spec represents an image."""
        key_lower = key.lower()
        if any(x in key_lower for x in ["image", "rgb", "camera", "img"]):
            return True
        shape = spec.get("shape", [])
        if len(shape) == 3 and shape[-1] in [1, 3, 4]:
            return True
        return False

    def _str_to_dtype(self, dtype_str: str) -> Dtype:
        """Convert string dtype to Robodata Dtype."""
        dtype_str = dtype_str.lower()
        mapping = {
            "float32": Dtype.FLOAT32,
            "float64": Dtype.FLOAT64,
            "int32": Dtype.INT32,
            "int64": Dtype.INT64,
            "uint8": Dtype.UINT8,
            "bool": Dtype.BOOL,
            "string": Dtype.STRING,
        }
        return mapping.get(dtype_str, Dtype.FLOAT32)

    def _analyze_schema_from_records(self, tfrecord_files: list[Path], info: DatasetInfo) -> None:
        """Analyze schema by reading actual TFRecord data."""
        import tensorflow as tf

        # Create dataset from tfrecord files, trying different compression types
        ds = None
        for compression in ["", "GZIP", "ZLIB"]:
            try:
                test_ds = tf.data.TFRecordDataset(
                    [str(f) for f in tfrecord_files[:1]],
                    compression_type=compression,
                )
                # Test by reading first record
                for _ in test_ds.take(1):
                    pass
                ds = test_ds
                break
            except Exception:
                continue

        if ds is None:
            ds = tf.data.TFRecordDataset([str(f) for f in tfrecord_files[:1]])

        episode_count = 0

        for raw_record in ds.take(5):
            # Parse as Example to understand structure
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            episode_count += 1

            # Analyze features
            for key, feature in example.features.feature.items():
                # Determine type from which field is populated
                if feature.HasField("bytes_list"):
                    if self._looks_like_image(key):
                        if key not in info.cameras:
                            # We'll get dimensions when we decode
                            info.cameras[key] = CameraInfo(
                                name=key, height=480, width=640, channels=3
                            )
                elif feature.HasField("float_list"):
                    values = list(feature.float_list.value)
                    if key not in info.observation_schema and "action" not in key.lower():
                        info.observation_schema[key] = FieldSchema(
                            name=key,
                            shape=(len(values),) if values else (),
                            dtype=Dtype.FLOAT32,
                        )
                    if "action" in key.lower() and info.action_schema is None:
                        info.action_schema = FieldSchema(
                            name=key,
                            shape=(len(values),) if values else (),
                            dtype=Dtype.FLOAT32,
                        )
                elif feature.HasField("int64_list"):
                    if key not in info.observation_schema:
                        values = list(feature.int64_list.value)
                        info.observation_schema[key] = FieldSchema(
                            name=key,
                            shape=(len(values),) if values else (),
                            dtype=Dtype.INT64,
                        )

        info.num_episodes = max(info.num_episodes, episode_count)

        # Check for language and timestamps
        if any("language" in k.lower() for k in info.observation_schema):
            info.has_language = True
        if any("time" in k.lower() for k in info.observation_schema):
            info.has_timestamps = True

    def _looks_like_image(self, key: str) -> bool:
        """Heuristic to detect image fields."""
        key_lower = key.lower()
        return any(x in key_lower for x in ["image", "rgb", "camera", "img", "frame"])

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over RLDS episodes.

        Args:
            path: Path to RLDS dataset.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_tensorflow()

        path = Path(path)
        tfrecord_files = self._find_tfrecord_files(path)

        # Try to use tensorflow_datasets if available for proper RLDS parsing
        try:
            yield from self._read_with_tfds(path)
        except Exception:
            # Fall back to raw TFRecord parsing
            yield from self._read_raw_tfrecords(tfrecord_files)

    def _resolve_tfds_path(self, path: Path) -> tuple[Path, str]:
        """Resolve TFDS path to (data_dir, dataset_name).

        TFDS datasets can have various structures:
        - data_dir/dataset_name/version/ (simple dataset)
        - data_dir/dataset_name/config_name/version/ (dataset with config)

        Returns:
            Tuple of (data_dir, dataset_name) where dataset_name may include config.
        """
        import re

        # Check if path ends with a version directory (x.y.z format)
        version_pattern = re.compile(r'^\d+\.\d+\.\d+$')

        if version_pattern.match(path.name):
            # Path ends with version, go up to find dataset structure
            config_or_dataset = path.parent  # e.g., can_ph_image
            potential_parent = config_or_dataset.parent  # e.g., robomimic_ph

            # Check if this is a nested dataset (dataset/config structure)
            if (potential_parent.parent / potential_parent.name).is_dir():
                # Structure: data_dir/dataset/config/version
                data_dir = potential_parent.parent
                dataset_name = f"{potential_parent.name}/{config_or_dataset.name}"
                return data_dir, dataset_name
            else:
                # Structure: data_dir/dataset/version
                data_dir = config_or_dataset.parent
                dataset_name = config_or_dataset.name
                return data_dir, dataset_name

        # Check if dataset_info.json is in path itself
        if (path / "dataset_info.json").exists():
            # Path is the version directory, parent is config or dataset
            parent = path.parent
            grandparent = parent.parent

            # Check if grandparent has the parent as a subdir (nested structure)
            if (grandparent / parent.name).is_dir():
                data_dir = grandparent.parent
                dataset_name = f"{grandparent.name}/{parent.name}"
                return data_dir, dataset_name
            else:
                data_dir = grandparent
                dataset_name = parent.name
                return data_dir, dataset_name

        # Check if dataset_info.json is in parent (path is dataset dir without version)
        if (path.parent / "dataset_info.json").exists():
            data_dir = path.parent.parent
            dataset_name = f"{path.parent.name}/{path.name}"
            return data_dir, dataset_name

        # Fallback: treat path as dataset directory
        return path.parent, path.name

    def _read_with_tfds(self, path: Path) -> Iterator[Episode]:
        """Read using tensorflow_datasets for proper RLDS structure."""
        import tensorflow_datasets as tfds

        # First try builder_from_directory (most reliable for local datasets)
        try:
            builder = tfds.builder_from_directory(str(path))
            ds = builder.as_dataset(split="train")

            for i, episode_data in enumerate(ds):
                yield self._parse_tfds_episode(episode_data, str(i))
            return
        except Exception:
            pass  # Fall through to try tfds.load

        # Resolve path to TFDS data_dir and dataset_name
        data_dir, dataset_name = self._resolve_tfds_path(path)

        ds = tfds.load(dataset_name, data_dir=str(data_dir))

        if isinstance(ds, dict):
            ds = ds.get("train", list(ds.values())[0])

        for i, episode_data in enumerate(ds):
            yield self._parse_tfds_episode(episode_data, str(i))

    def _parse_tfds_episode(self, episode_data: dict, episode_id: str) -> Episode:
        """Convert TFDS episode dict to Robodata Episode."""

        steps = episode_data.get("steps", episode_data)

        # Extract metadata
        language = None
        if "language_instruction" in episode_data:
            lang_tensor = episode_data["language_instruction"]
            if hasattr(lang_tensor, "numpy"):
                language = lang_tensor.numpy().decode("utf-8")

        # Create lazy frame loader
        def load_frames() -> Iterator[Frame]:
            for i, step in enumerate(steps):
                obs = step.get("observation", step)

                # Create lazy image loaders
                images: dict[str, LazyImage] = {}
                state = None

                # Handle case where obs is a dict (standard RLDS) vs a tensor (some datasets)
                if isinstance(obs, dict):
                    for key in obs:
                        if self._looks_like_image(key):
                            img_tensor = obs[key]

                            def make_loader(t: Any = img_tensor) -> NDArray[Any]:
                                return t.numpy()

                            shape = img_tensor.shape
                            images[key] = LazyImage(
                                loader=make_loader,
                                height=int(shape[0]) if len(shape) > 0 else 480,
                                width=int(shape[1]) if len(shape) > 1 else 640,
                                channels=int(shape[2]) if len(shape) > 2 else 3,
                            )

                    # Get state/proprio from dict
                    for state_key in ["state", "proprio", "proprioception", "robot_state"]:
                        if state_key in obs:
                            state = obs[state_key].numpy()
                            break
                else:
                    # obs is a tensor itself (e.g., d4rl datasets)
                    if hasattr(obs, "numpy"):
                        state = obs.numpy()

                # Get action
                action = None
                if "action" in step:
                    action = step["action"].numpy()

                yield Frame(
                    index=i,
                    images=images,
                    state=state,
                    action=action,
                    reward=float(step.get("reward", 0.0)),
                    is_first=step.get("is_first", i == 0),
                    is_last=step.get("is_last", False),
                    is_terminal=step.get("is_terminal", False),
                )

        return Episode(
            episode_id=episode_id,
            language_instruction=language,
            _frame_loader=load_frames,
        )

    def _read_raw_tfrecords(self, tfrecord_files: list[Path]) -> Iterator[Episode]:
        """Read episodes from raw TFRecord files."""
        import tensorflow as tf

        # Try different compression types
        ds = None
        for compression in ["", "GZIP", "ZLIB"]:
            try:
                test_ds = tf.data.TFRecordDataset(
                    [str(f) for f in tfrecord_files],
                    compression_type=compression,
                )
                # Test by reading first record
                for _ in test_ds.take(1):
                    pass
                # If we got here, compression is correct
                ds = test_ds
                break
            except Exception:
                continue

        if ds is None:
            # Default to no compression
            ds = tf.data.TFRecordDataset([str(f) for f in tfrecord_files])

        for i, raw_record in enumerate(ds):
            yield self._parse_raw_episode(raw_record, str(i))

    def _parse_raw_episode(self, raw_record: Any, episode_id: str) -> Episode:
        """Parse a raw TFRecord into an Episode."""
        import tensorflow as tf

        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Determine number of steps from a sequence field
        num_steps = 1
        for key, feature in features.items():
            if key.startswith("steps/") and feature.HasField("int64_list"):
                num_steps = len(feature.int64_list.value)
                break
            elif key.startswith("steps/") and feature.HasField("bytes_list"):
                num_steps = len(feature.bytes_list.value)
                break

        # Detect action/state dimensions
        action_dim = None
        state_dim = None
        for key, feature in features.items():
            if feature.HasField("float_list"):
                n_values = len(feature.float_list.value)
                if "action" in key.lower() and n_values > 0:
                    action_dim = n_values // num_steps if num_steps > 0 else n_values
                elif ("state" in key.lower() or "observation" in key.lower()) and "image" not in key.lower():
                    state_dim = n_values // num_steps if num_steps > 0 else n_values

        def load_frames() -> Iterator[Frame]:
            import io
            import numpy as np
            from PIL import Image

            # Pre-extract all data
            image_data: dict[str, list] = {}
            action_data: NDArray[Any] | None = None
            state_data: NDArray[Any] | None = None

            for key, feature in features.items():
                if feature.HasField("bytes_list") and self._looks_like_image(key):
                    image_data[key] = list(feature.bytes_list.value)
                elif feature.HasField("float_list"):
                    values = np.array(feature.float_list.value, dtype=np.float32)
                    if "action" in key.lower():
                        action_data = values.reshape(num_steps, -1) if num_steps > 1 else values
                    elif ("state" in key.lower() or "observation" in key.lower()) and "image" not in key.lower():
                        state_data = values.reshape(num_steps, -1) if num_steps > 1 else values

            # Yield frames
            for step_idx in range(num_steps):
                images: dict[str, LazyImage] = {}

                for cam_key, img_list in image_data.items():
                    if step_idx < len(img_list):
                        def make_loader(data: bytes = img_list[step_idx]) -> NDArray[Any]:
                            try:
                                img = Image.open(io.BytesIO(data))
                                return np.array(img)
                            except Exception:
                                return np.frombuffer(data, dtype=np.uint8).reshape(480, 640, 3)

                        images[cam_key] = LazyImage(loader=make_loader, height=480, width=640, channels=3)

                action = action_data[step_idx] if action_data is not None and len(action_data.shape) > 1 else action_data
                state = state_data[step_idx] if state_data is not None and len(state_data.shape) > 1 else state_data

                yield Frame(
                    index=step_idx,
                    images=images,
                    state=state,
                    action=action,
                    is_first=(step_idx == 0),
                    is_last=(step_idx == num_steps - 1),
                )

        return Episode(episode_id=episode_id, _frame_loader=load_frames)

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to dataset.
            episode_id: Episode identifier.

        Returns:
            The requested Episode.

        Raises:
            EpisodeNotFoundError: If episode not found.
        """
        for episode in self.read_episodes(path):
            if episode.episode_id == episode_id:
                return episode

        raise EpisodeNotFoundError(episode_id, path)
