"""GR00T format reader for Robodata.

GR00T (NVIDIA Isaac GR00T) uses LeRobot v2 format with additional
NVIDIA-specific metadata. This reader detects GR00T datasets and
delegates to the LeRobot reader for actual data access.

GR00T-specific markers:
- robot_type containing NVIDIA robot identifiers (GR1, SO100, etc.)
- annotation.human.* fields in features
- Motor naming convention (motor_0, motor_1, ...)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from robodata.core.models import DatasetInfo, Episode
from robodata.formats.registry import FormatRegistry

if TYPE_CHECKING:
    pass

# Known GR00T robot types from NVIDIA
GROOT_ROBOT_TYPES = {
    "GR1ArmsOnly",
    "GR1ArmsWaist",
    "GR1FullBody",
    "SO100DualArm",
    "SO100SingleArm",
    "BimanualPandaGripper",
    "BimanualPandaHand",
    "SinglePandaGripper",
    "SinglePandaHand",
}


@FormatRegistry.register_reader("groot")
class GR00TReader:
    """Reader for GR00T (NVIDIA Isaac) format.

    GR00T datasets use LeRobot v2 format with NVIDIA-specific extensions.
    This reader identifies GR00T datasets by checking for distinctive markers
    and delegates to the LeRobot reader for data access.

    Example:
        >>> reader = GR00TReader()
        >>> if reader.can_read(path):
        ...     info = reader.inspect(path)
        ...     for episode in reader.read_episodes(path):
        ...         process(episode)
    """

    def __init__(self):
        """Initialize GR00T reader with internal LeRobot reader."""
        # Import here to avoid circular imports
        from robodata.formats.lerobot_v3.reader import LeRobotV3Reader
        self._lerobot_reader = LeRobotV3Reader()

    @property
    def format_name(self) -> str:
        return "groot"

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check for GR00T-specific markers.

        GR00T is distinguished from standard LeRobot by:
        - robot_type matching known NVIDIA robot identifiers
        - annotation.human.* fields in features
        - Motor naming convention in state/action names

        Args:
            path: Path to potential GR00T dataset.

        Returns:
            True if GR00T markers found.
        """
        if not path.exists() or not path.is_dir():
            return False

        # Must have meta/info.json
        info_path = path / "meta" / "info.json"
        if not info_path.exists():
            return False

        try:
            with open(info_path) as f:
                info = json.load(f)

            # Check 1: robot_type matches known GR00T robots
            robot_type = info.get("robot_type", "")
            if robot_type in GROOT_ROBOT_TYPES:
                return True

            # Check 2: robot_type contains GR1 or SO100 patterns
            if any(pattern in robot_type for pattern in ["GR1", "SO100", "Panda"]):
                return True

            # Check 3: Has annotation.human.* fields (GR00T-specific)
            features = info.get("features", {})
            groot_fields = ["annotation.human.validity", "annotation.human.action.task_description"]
            if any(field in features for field in groot_fields):
                return True

            # Check 4: Motor naming convention in state/action
            for key in ["observation.state", "action"]:
                if key in features:
                    names = features[key].get("names", [])
                    if names and any(n.startswith("motor_") for n in names):
                        return True

            return False

        except (json.JSONDecodeError, KeyError, TypeError):
            return False

    def inspect(self, path: Path | str) -> DatasetInfo:
        """Inspect GR00T dataset metadata.

        Args:
            path: Path to GR00T dataset.

        Returns:
            DatasetInfo with GR00T-specific metadata.
        """
        path = Path(path)
        info = self._lerobot_reader.inspect(path)

        # Override format name to indicate GR00T
        info.format = "groot"

        # Read robot_type from info.json
        info_path = path / "meta" / "info.json"
        if info_path.exists():
            with open(info_path) as f:
                raw_info = json.load(f)
            # Store robot type in inferred_robot_type
            info.inferred_robot_type = raw_info.get("robot_type", info.inferred_robot_type)

        return info

    def read_episodes(self, path: Path | str) -> Iterator[Episode]:
        """Read episodes from GR00T dataset.

        Args:
            path: Path to GR00T dataset.

        Yields:
            Episode objects.
        """
        yield from self._lerobot_reader.read_episodes(path)

    def read_episode(self, path: Path | str, episode_id: str) -> Episode:
        """Read a specific episode by ID.

        Args:
            path: Path to GR00T dataset.
            episode_id: Episode identifier.

        Returns:
            Episode object.
        """
        return self._lerobot_reader.read_episode(path, episode_id)
