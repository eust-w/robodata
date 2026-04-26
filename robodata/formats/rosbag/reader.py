"""Rosbag format reader for Robodata.

Reads ROS1 (.bag) and ROS2 (MCAP, SQLite3) bag files using the rosbags library.
This is a pure Python implementation that doesn't require ROS to be installed.

Note: Rosbags are general-purpose and can contain any ROS data. This reader
focuses on extracting VLA-relevant data: camera images, robot state, and actions.
Non-VLA topics (LIDAR, TF, etc.) are noted but not included in observation schema.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

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
from robodata.formats.common import extract_camera_name_from_key

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Common image topic patterns in robotics
IMAGE_TOPIC_PATTERNS = [
    r".*/image_raw$",
    r".*/image$",
    r".*/rgb$",
    r".*/color/image.*",
    r".*/depth/image.*",
    r".*/camera.*/image.*",
    r".*/image_rect.*",
    r".*/compressed$",
]

# Common robotics message types for state
STATE_MSG_TYPES = [
    "sensor_msgs/msg/JointState",
    "sensor_msgs/JointState",
    "geometry_msgs/msg/PoseStamped",
    "geometry_msgs/PoseStamped",
    "geometry_msgs/msg/Pose",
    "geometry_msgs/Pose",
    "nav_msgs/msg/Odometry",
    "nav_msgs/Odometry",
    "sensor_msgs/msg/Imu",
    "sensor_msgs/Imu",
]

ACTION_TOPIC_PATTERNS = [
    r".*/command$",
    r".*/cmd$",
    r".*/action$",
    r".*/target.*",
    r".*/goal$",
    r".*/joint_command.*",
    r".*/arm_controller/command$",
    r".*/gripper.*",
]

# Topics to exclude from observation schema (non-VLA relevant)
EXCLUDED_MSG_TYPES = [
    "sensor_msgs/msg/PointCloud2",
    "sensor_msgs/PointCloud2",
    "sensor_msgs/msg/LaserScan",
    "sensor_msgs/LaserScan",
    "tf2_msgs/msg/TFMessage",
    "tf2_msgs/TFMessage",
    "sensor_msgs/msg/CameraInfo",
    "sensor_msgs/CameraInfo",
    "diagnostic_msgs/msg/DiagnosticArray",
    "rosgraph_msgs/msg/Clock",
]


def _is_image_topic(topic: str, msgtype: str) -> bool:
    """Check if a topic is an image topic."""
    # Check message type
    if "Image" in msgtype or "CompressedImage" in msgtype:
        return True
    # Check topic name patterns
    for pattern in IMAGE_TOPIC_PATTERNS:
        if re.match(pattern, topic):
            return True
    return False


def _is_state_topic(topic: str, msgtype: str) -> bool:
    """Check if a topic contains robot state."""
    return msgtype in STATE_MSG_TYPES


def _is_action_topic(topic: str, msgtype: str) -> bool:
    """Check if a topic contains action commands."""
    for pattern in ACTION_TOPIC_PATTERNS:
        if re.match(pattern, topic):
            return True
    return False


def _is_excluded_topic(topic: str, msgtype: str) -> bool:
    """Check if topic should be excluded from VLA observations."""
    return msgtype in EXCLUDED_MSG_TYPES


def _check_rosbags() -> None:
    """Check if rosbags is available."""
    try:
        import rosbags  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "rosbags is required for ROSBAG format support. Install with: pip install rosbags",
        ) from e


@FormatRegistry.register_reader("rosbag")
class RosbagReader:
    """Reader for ROS1 and ROS2 bag files.

    Supports:
    - ROS1 .bag files
    - ROS2 MCAP files (.mcap)
    - ROS2 SQLite3 files (.db3)
    - ROS2 bag directories (with metadata.yaml)

    Uses the rosbags library for pure Python reading without ROS dependencies.
    """

    @property
    def format_name(self) -> str:
        """Return format identifier."""
        return "rosbag"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for rosbag markers.

        Args:
            path: Path to potential rosbag.

        Returns:
            True if rosbag markers found.
        """
        if not path.exists():
            return False

        # ROS1 .bag file
        if path.is_file() and path.suffix == ".bag":
            return True

        # ROS2 MCAP file
        if path.is_file() and path.suffix == ".mcap":
            return True

        # ROS2 SQLite3 file
        if path.is_file() and path.suffix == ".db3":
            return True

        # ROS2 bag directory (contains metadata.yaml)
        if path.is_dir():
            if (path / "metadata.yaml").exists():
                return True
            # Check for .mcap or .db3 files inside
            if any(path.glob("*.mcap")) or any(path.glob("*.db3")):
                return True

        return False

    @classmethod
    def detect_version(cls, path: Path) -> str | None:
        """Detect ROS version and storage format.

        Args:
            path: Path to rosbag.

        Returns:
            Version string like "ros1", "ros2-mcap", "ros2-sqlite3".
        """
        path = Path(path)

        # ROS1 .bag file
        if path.is_file() and path.suffix == ".bag":
            return "ros1"

        # ROS2 MCAP
        if path.is_file() and path.suffix == ".mcap":
            return "ros2-mcap"

        # ROS2 SQLite3
        if path.is_file() and path.suffix == ".db3":
            return "ros2-sqlite3"

        # ROS2 bag directory
        if path.is_dir():
            if any(path.glob("*.mcap")):
                return "ros2-mcap"
            if any(path.glob("*.db3")):
                return "ros2-sqlite3"
            if (path / "metadata.yaml").exists():
                return "ros2"

        return None

    def inspect(self, path: Path) -> DatasetInfo:
        """Analyze rosbag structure.

        Args:
            path: Path to rosbag.

        Returns:
            DatasetInfo with topics, message types, and inferred schema.

        Raises:
            InspectionError: If inspection fails.
        """
        path = Path(path)
        info = DatasetInfo(path=path, format="rosbag")
        info.format_version = self.detect_version(path)

        # Try metadata.yaml first for ROS2 bags (faster, no rosbags dependency)
        metadata_path = None
        if path.is_dir() and (path / "metadata.yaml").exists():
            metadata_path = path / "metadata.yaml"

        if metadata_path:
            try:
                self._analyze_metadata_yaml(metadata_path, info)
                return info
            except Exception:
                pass  # Fall back to rosbags library

        # Use rosbags library for full analysis
        _check_rosbags()
        try:
            self._analyze_bag(path, info)
        except Exception as e:
            raise InspectionError(path, f"Failed to read rosbag: {e}")

        return info

    def _analyze_metadata_yaml(self, metadata_path: Path, info: DatasetInfo) -> None:
        """Fast inspection using metadata.yaml (ROS2 bags only)."""
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        bag_info = metadata.get("rosbag2_bagfile_information", {})

        # Refine format_version using storage identifier if available
        storage_id = bag_info.get("storage_identifier")
        if storage_id == "sqlite3":
            info.format_version = "ros2-sqlite3"
        elif storage_id == "mcap":
            info.format_version = "ros2-mcap"
        else:
            # Keep prior detect_version() result
            info.format_version = info.format_version or "ros2"

        # Duration and timing
        duration_ns = bag_info.get("duration", {}).get("nanoseconds", 0)
        info.num_episodes = 1

        # Count topics by type
        topics = bag_info.get("topics_with_message_count", [])
        total_image_msgs = 0

        for topic_info in topics:
            topic_meta = topic_info.get("topic_metadata", {})
            topic = topic_meta.get("name", "")
            msgtype = topic_meta.get("type", "")
            msgcount = topic_info.get("message_count", 0)

            # Detect image topics -> cameras
            if _is_image_topic(topic, msgtype):
                cam_name = extract_camera_name_from_key(topic)
                info.cameras[cam_name] = CameraInfo(
                    name=cam_name,
                    height=480,  # Default, would need actual bag to get resolution
                    width=640,
                    channels=3,
                )
                total_image_msgs = max(total_image_msgs, msgcount)

            # Detect state topics (only VLA-relevant)
            elif _is_state_topic(topic, msgtype):
                info.observation_schema[topic] = FieldSchema(
                    name=topic,
                    shape=(),
                    dtype=Dtype.FLOAT32,
                    description=msgtype,
                )

            # Detect action topics
            elif _is_action_topic(topic, msgtype):
                info.action_schema = FieldSchema(
                    name=topic,
                    shape=(),
                    dtype=Dtype.FLOAT32,
                )

            # Skip excluded topics (LIDAR, TF, CameraInfo, etc.)
            # These are not relevant for VLA training

        info.total_frames = total_image_msgs

        # Estimate FPS
        if duration_ns > 0 and total_image_msgs > 0:
            duration_sec = duration_ns / 1e9
            info.inferred_fps = round(total_image_msgs / duration_sec, 1)

    def _analyze_bag(self, path: Path, info: DatasetInfo) -> None:
        """Analyze bag contents using rosbags library."""
        from rosbags.highlevel import AnyReader

        with AnyReader([path]) as reader:
            # Analyze connections (topics)
            info.num_episodes = 1  # Rosbags are typically single episodes

            start_time = None
            end_time = None
            frame_count = 0

            for conn in reader.connections:
                topic = conn.topic
                msgtype = conn.msgtype
                msgcount = conn.msgcount

                # Track timestamps for duration/FPS estimation
                if hasattr(conn, "ext") and conn.ext is not None:
                    ext: Any = conn.ext
                    if hasattr(ext, "time_start"):
                        if start_time is None or ext.time_start < start_time:
                            start_time = ext.time_start
                    if hasattr(ext, "time_end"):
                        if end_time is None or ext.time_end > end_time:
                            end_time = ext.time_end

                # Detect image topics -> cameras
                if _is_image_topic(topic, msgtype):
                    cam_name = extract_camera_name_from_key(topic)
                    # We'll get resolution from first message
                    info.cameras[cam_name] = CameraInfo(
                        name=cam_name,
                        height=480,  # Default, will update from message
                        width=640,
                        channels=3,
                    )
                    frame_count = max(frame_count, msgcount)

                # Detect state topics (VLA-relevant)
                elif _is_state_topic(topic, msgtype):
                    info.observation_schema[topic] = FieldSchema(
                        name=topic,
                        shape=(),  # Will be determined from message
                        dtype=Dtype.FLOAT32,
                        description=msgtype,
                    )

                # Detect action topics
                elif _is_action_topic(topic, msgtype):
                    info.action_schema = FieldSchema(
                        name=topic,
                        shape=(),
                        dtype=Dtype.FLOAT32,
                    )

                # Skip non-VLA topics (LIDAR, TF, CameraInfo, etc.)
                elif _is_excluded_topic(topic, msgtype):
                    pass  # Not relevant for VLA training

            info.total_frames = frame_count

            # Estimate FPS from timestamps
            if start_time is not None and end_time is not None and frame_count > 0:
                duration_ns = end_time - start_time
                duration_sec = duration_ns / 1e9
                if duration_sec > 0:
                    info.inferred_fps = round(frame_count / duration_sec, 1)

            # Sample first image message to get resolution
            self._sample_image_resolution(reader, info)

    def _sample_image_resolution(self, reader: Any, info: DatasetInfo) -> None:
        """Sample first image message to determine resolution."""
        if not info.cameras:
            return

        # Find image connections
        image_connections = []
        for conn in reader.connections:
            if _is_image_topic(conn.topic, conn.msgtype):
                image_connections.append(conn)

        if not image_connections:
            return

        # Read first message from each image topic
        for conn in image_connections:
            cam_name = _extract_camera_name(conn.topic)
            if cam_name not in info.cameras:
                continue

            try:
                for connection, timestamp, rawdata in reader.messages(connections=[conn]):
                    msg = reader.deserialize(rawdata, connection.msgtype)

                    # sensor_msgs/Image
                    if hasattr(msg, "height") and hasattr(msg, "width"):
                        encoding = getattr(msg, "encoding", "rgb8")
                        info.cameras[cam_name] = CameraInfo(
                            name=cam_name,
                            height=msg.height,
                            width=msg.width,
                            channels=3
                            if "rgb" in encoding.lower() or "bgr" in encoding.lower()
                            else 1,
                        )
                    break  # Only need first message
            except Exception:
                pass  # Keep default resolution

    def read_episodes(self, path: Path) -> Iterator[Episode]:
        """Lazily iterate over rosbag as episodes.

        A rosbag is typically treated as a single episode.
        For multi-episode bags, segmentation can be done by time gaps.

        Args:
            path: Path to rosbag.

        Yields:
            Episode objects with lazy frame loading.
        """
        _check_rosbags()

        path = Path(path)

        # Create episode with lazy frame loader
        def load_frames() -> Iterator[Frame]:
            yield from self._load_frames_from_bag(path)

        yield Episode(
            episode_id="0",
            _frame_loader=load_frames,
        )

    def _load_frames_from_bag(self, path: Path) -> Iterator[Frame]:
        """Load frames from rosbag by synchronizing image timestamps."""
        import numpy as np
        from rosbags.highlevel import AnyReader

        with AnyReader([path]) as reader:
            # Find image connections
            image_connections = [
                conn for conn in reader.connections if _is_image_topic(conn.topic, conn.msgtype)
            ]

            if not image_connections:
                # No images - yield empty frames based on other data
                return

            # Use first image topic as reference for frame timing
            ref_conn = image_connections[0]

            frame_idx = 0
            for connection, timestamp, rawdata in reader.messages(connections=[ref_conn]):
                msg = reader.deserialize(rawdata, connection.msgtype)

                # Create lazy image loader
                images: dict[str, LazyImage] = {}
                cam_name = _extract_camera_name(connection.topic)

                if hasattr(msg, "data") and hasattr(msg, "height"):
                    height: int = msg.height
                    width: int = msg.width  # type: ignore[attr-defined]
                    encoding = getattr(msg, "encoding", "rgb8")

                    def make_loader(
                        data: bytes = bytes(msg.data),  # type: ignore[attr-defined]
                        h: int = height,
                        w: int = width,
                        enc: str = encoding,
                    ) -> NDArray[Any]:
                        arr = np.frombuffer(data, dtype=np.uint8)
                        if "rgb" in enc.lower() or "bgr" in enc.lower():
                            return arr.reshape(h, w, 3)
                        elif "mono" in enc.lower() or "8" in enc:
                            return arr.reshape(h, w)
                        else:
                            return arr.reshape(h, w, -1)

                    channels = 3 if "rgb" in encoding.lower() or "bgr" in encoding.lower() else 1
                    images[cam_name] = LazyImage(
                        loader=make_loader,
                        height=height,
                        width=width,
                        channels=channels,
                    )

                yield Frame(
                    index=frame_idx,
                    images=images,
                    timestamp=timestamp / 1e9,  # Convert ns to seconds
                    is_first=frame_idx == 0,
                )
                frame_idx += 1

    def read_episode(self, path: Path, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to rosbag.
            episode_id: Episode identifier (typically "0" for rosbags).

        Returns:
            The requested Episode.

        Raises:
            EpisodeNotFoundError: If episode not found.
        """
        for episode in self.read_episodes(path):
            if episode.episode_id == episode_id:
                return episode

        raise EpisodeNotFoundError(episode_id, path)
