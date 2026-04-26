"""Pytest configuration and fixtures for Robodata tests."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_data_dir() -> Path:
    """Return path to sample_data directory."""
    return Path(__file__).parent.parent / "sample_data"


@pytest.fixture
def temp_zarr_dataset(tmp_path: Path) -> Path:
    """Create a temporary Zarr dataset for testing."""
    zarr = pytest.importorskip("zarr")

    dataset_path = tmp_path / "test_dataset.zarr"
    root = zarr.open(str(dataset_path), mode="w")

    # Add metadata
    root.attrs["fps"] = 30
    root.attrs["robot_type"] = "test_robot"
    root.attrs["version"] = "1.0"

    # Create data group
    data = root.create_group("data")

    # 2 episodes with 50 frames each
    total_frames = 100

    # Camera data (frames, H, W, C)
    camera0_rgb = data.create_dataset(
        "camera0_rgb",
        shape=(total_frames, 32, 32, 3),
        dtype=np.uint8,
        chunks=(1, 32, 32, 3),
    )
    camera0_rgb[:] = np.random.randint(0, 255, (total_frames, 32, 32, 3), dtype=np.uint8)

    # Robot state
    robot_state = data.create_dataset(
        "robot_state",
        shape=(total_frames, 7),
        dtype=np.float32,
    )
    robot_state[:] = np.random.randn(total_frames, 7).astype(np.float32)

    # Action
    action = data.create_dataset(
        "action",
        shape=(total_frames, 7),
        dtype=np.float32,
    )
    action[:] = np.random.randn(total_frames, 7).astype(np.float32)

    # Episode boundaries
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=np.array([50, 100]))

    return dataset_path


@pytest.fixture
def temp_lerobot_v3_dataset(tmp_path: Path) -> Path:
    """Create a temporary LeRobot v3 dataset for testing."""
    import json

    dataset_path = tmp_path / "test_lerobot_v3"
    dataset_path.mkdir()

    # Create meta directory (v3 structure)
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir()

    # Create meta/info.json (v3 puts it in meta/)
    info = {
        "codebase_version": "v3.0",
        "fps": 30,
        "robot_type": "test_robot",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [7]},
            "action": {"dtype": "float32", "shape": [7]},
        },
        "total_episodes": 2,
        "total_frames": 100,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f)

    # Create data/train directory (v3 structure)
    data_dir = dataset_path / "data" / "train"
    data_dir.mkdir(parents=True)

    # Create meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        f.write('{"episode_index": 0, "length": 50}\n')
        f.write('{"episode_index": 1, "length": 50}\n')

    # Create meta/tasks.jsonl (v3-specific file)
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write('{"task_index": 0, "task": "pick up object"}\n')

    return dataset_path


@pytest.fixture
def temp_rosbag_metadata(tmp_path: Path) -> Path:
    """Create a temporary ROS2 bag directory with metadata.yaml for testing."""
    import yaml

    bag_path = tmp_path / "test_rosbag"
    bag_path.mkdir()

    metadata = {
        "rosbag2_bagfile_information": {
            "version": 5,
            "storage_identifier": "sqlite3",
            "duration": {"nanoseconds": 10000000000},  # 10 seconds
            "message_count": 600,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/camera/rgb/image_raw",
                        "type": "sensor_msgs/msg/Image",
                        "serialization_format": "cdr",
                    },
                    "message_count": 300,
                },
                {
                    "topic_metadata": {
                        "name": "/robot/joint_states",
                        "type": "sensor_msgs/msg/JointState",
                        "serialization_format": "cdr",
                    },
                    "message_count": 300,
                },
            ],
        }
    }

    with open(bag_path / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f)

    return bag_path
