"""GR00T format support for Robodata.

GR00T (NVIDIA Isaac) uses LeRobot v2 format with additional metadata:
- robot_type: NVIDIA-specific robot identifiers (GR1ArmsOnly, SO100DualArm, etc.)
- Motor names in state/action features
- annotation.human.validity and annotation.human.action.task_description fields
- trajectory_id in episodes.jsonl

This reader detects GR00T-specific markers and uses the LeRobot reader internally.
"""

from robodata.formats.groot.reader import GR00TReader

__all__ = ["GR00TReader"]
