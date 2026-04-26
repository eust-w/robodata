"""Schema analyzer for Robodata.

Analyzes episode data to infer schema properties using heuristics
when explicit metadata is unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robodata.core.models import Frame


class SchemaAnalyzer:
    """Analyzes episode data to infer schema properties.

    Uses heuristics when explicit metadata is unavailable. This is particularly
    useful for datasets that don't include complete metadata about their structure.
    """

    def infer_gripper_index(
        self,
        frames: list[Frame],
        state_key: str = "state",
    ) -> int | None:
        """Infer gripper index from state vectors.

        Heuristic: gripper is often the last dimension and bounded [0, 1].

        Strategy:
        1. Check if last dimension stays in [0, 1] range
        2. Check if other dimensions have larger range
        3. Confirm pattern holds across frames

        Args:
            frames: List of frames to analyze.
            state_key: Key for state data (unused, state comes from Frame.state).

        Returns:
            Inferred gripper index or None if not detected.
        """
        import numpy as np

        if not frames or frames[0].state is None:
            return None

        # Collect state vectors
        states_list = [f.state for f in frames if f.state is not None]
        if len(states_list) < 10:
            return None

        states = np.array(states_list)
        if states.ndim != 2:
            return None

        num_dims = states.shape[1]
        if num_dims < 2:
            return None

        # Check last dimension - typical gripper position
        last_dim = states[:, -1]
        last_min, last_max = float(last_dim.min()), float(last_dim.max())

        # Gripper typically in [0, 1] or [-1, 1] range
        if last_min >= -0.1 and last_max <= 1.1:
            # Check that other dims have larger range (joint positions)
            other_dims = states[:, :-1]
            other_range = float(other_dims.max() - other_dims.min())
            last_range = last_max - last_min

            # Gripper should have smaller range than joint positions
            if other_range > 1.0 or last_range < other_range * 0.5:
                return num_dims - 1

        # Try checking each dimension for gripper-like behavior
        for i in range(num_dims):
            dim_values = states[:, i]
            dim_min, dim_max = float(dim_values.min()), float(dim_values.max())

            # Gripper characteristics: bounded, binary-ish transitions
            if dim_min >= -0.1 and dim_max <= 1.1:
                # Check for discrete transitions (gripper open/close)
                diffs = np.abs(np.diff(dim_values))
                if diffs.max() > 0.3 and diffs.mean() < 0.1:
                    return i

        return None

    def infer_fps(self, frames: list[Frame]) -> float | None:
        """Infer FPS from timestamp differences if available.

        Args:
            frames: List of frames with timestamps.

        Returns:
            Inferred FPS or None if cannot be determined.
        """
        import numpy as np

        timestamps = [f.timestamp for f in frames if f.timestamp is not None]
        if len(timestamps) < 2:
            return None

        diffs = np.diff(timestamps)

        # Filter out outliers (pauses, etc.)
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            return None

        # Filter to diffs within 2x of median
        valid_diffs = diffs[(diffs > median_diff * 0.5) & (diffs < median_diff * 2)]
        if len(valid_diffs) < 2:
            valid_diffs = diffs

        avg_diff = float(np.mean(valid_diffs))
        if avg_diff <= 0:
            return None

        fps = 1.0 / avg_diff

        # Round to common FPS values if close
        common_fps = [10, 15, 20, 24, 25, 30, 50, 60]
        for common in common_fps:
            if abs(fps - common) < 2:
                return float(common)

        return round(fps, 1)

    def infer_action_type(self, frames: list[Frame]) -> str:
        """Infer if actions are absolute positions, deltas, or velocities.

        Heuristic:
        - Deltas: small values, often centered around 0
        - Absolute: values similar to state range
        - Velocity: bounded range, can be negative

        Args:
            frames: List of frames to analyze.

        Returns:
            One of "absolute", "delta", "velocity", or "unknown".
        """
        import numpy as np

        if not frames or frames[0].action is None:
            return "unknown"

        actions_list = [f.action for f in frames if f.action is not None]
        if len(actions_list) < 5:
            return "unknown"

        actions = np.array(actions_list)
        states_list = [f.state for f in frames if f.state is not None]
        states = np.array(states_list) if states_list else None

        action_range = float(actions.max() - actions.min())
        action_mean = float(np.abs(actions).mean())
        action_std = float(actions.std())

        # Delta actions: small values centered around 0
        if action_mean < 0.1 and action_std < 0.1:
            return "delta"

        # Compare with state if available
        if states is not None and states.size > 0:
            state_range = float(states.max() - states.min())

            # Absolute: similar range to state
            if 0.3 < (action_range / max(state_range, 0.001)) < 3.0:
                return "absolute"

            # Delta: much smaller than state range
            if action_range < state_range * 0.3:
                return "delta"

        # Velocity: check for symmetric distribution around 0
        action_median = float(np.median(actions))
        if abs(action_median) < action_std * 0.5:
            return "velocity"

        return "unknown"

    def detect_camera_type(
        self,
        name: str,
        shape: tuple[int, int, int],
    ) -> str:
        """Infer camera type from name and shape.

        Args:
            name: Camera name.
            shape: Image shape (height, width, channels).

        Returns:
            Camera type: "wrist", "overhead", "front", "depth", or "rgb".
        """
        name_lower = name.lower()

        # Check name patterns
        if "wrist" in name_lower or "hand" in name_lower or "gripper" in name_lower:
            return "wrist"
        if "overhead" in name_lower or "top" in name_lower or "bird" in name_lower:
            return "overhead"
        if "front" in name_lower or "forward" in name_lower:
            return "front"
        if "side" in name_lower or "lateral" in name_lower:
            return "side"
        if "depth" in name_lower:
            return "depth"

        # Check shape for depth camera
        if len(shape) >= 3 and shape[2] == 1:
            return "depth"

        return "rgb"

    def infer_state_components(
        self,
        frames: list[Frame],
    ) -> dict[str, tuple[int, int]]:
        """Try to identify components in the state vector.

        Attempts to identify:
        - Joint positions (typically first N dims)
        - Joint velocities (if present, follows positions)
        - Gripper state (typically last dim, bounded [0, 1])
        - End-effector pose (6 or 7 dims if present)

        Args:
            frames: List of frames to analyze.

        Returns:
            Dictionary mapping component names to (start, end) index tuples.
        """
        import numpy as np

        if not frames or frames[0].state is None:
            return {}

        states_list = [f.state for f in frames if f.state is not None]
        if len(states_list) < 10:
            return {}

        states = np.array(states_list)
        if states.ndim != 2:
            return {}

        num_dims = states.shape[1]
        components: dict[str, tuple[int, int]] = {}

        # Detect gripper (usually last dim, bounded)
        gripper_idx = self.infer_gripper_index(frames)
        if gripper_idx is not None:
            components["gripper"] = (gripper_idx, gripper_idx + 1)

        # Common robot configurations
        gripper_end = gripper_idx if gripper_idx is not None else num_dims

        # Check for typical joint configurations
        if gripper_end == 7:  # 6 joints + gripper (e.g., UR5, Panda)
            components["joint_positions"] = (0, 6)
        elif gripper_end == 8:  # 7 joints + gripper (e.g., Franka)
            components["joint_positions"] = (0, 7)
        elif gripper_end >= 12:  # Might have velocities too
            half = gripper_end // 2
            components["joint_positions"] = (0, half)
            components["joint_velocities"] = (half, gripper_end)

        return components

    def detect_success_signal(self, frames: list[Frame]) -> bool | None:
        """Try to determine episode success from terminal reward or flags.

        Args:
            frames: List of frames to analyze.

        Returns:
            True if success detected, False if failure, None if unknown.
        """
        if not frames:
            return None

        # Check terminal frame
        terminal_frames = [f for f in frames if f.is_terminal or f.is_last]
        if terminal_frames:
            last = terminal_frames[-1]
            if last.reward is not None:
                if last.reward > 0.5:
                    return True
                elif last.reward < -0.5:
                    return False

        # Check cumulative reward
        rewards = [f.reward for f in frames if f.reward is not None]
        if rewards:
            total_reward = sum(rewards)
            if total_reward > 0.5:
                return True
            elif total_reward < -0.5:
                return False

        return None
