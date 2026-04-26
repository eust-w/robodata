"""Ray execution backend for Robodata.

This module is intentionally optional: it requires the `ray` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robodata.core.models import DatasetInfo


@dataclass
class RayExecutionOptions:
    address: str | None = None


def is_ray_available() -> bool:
    try:
        import ray  # noqa: F401

        return True
    except Exception:
        return False


def _init_ray(address: str | None) -> None:
    import ray

    if ray.is_initialized():
        return

    if address is None:
        ray.init(ignore_reinit_error=True)
        return

    if address == "auto":
        ray.init(address="auto", ignore_reinit_error=True)
        return

    ray.init(address=address, ignore_reinit_error=True)


def run_parallel_episode_tasks_ray(
    *,
    source: Path,
    output: Path,
    source_format: str,
    target_format: str,
    config_dict: dict[str, Any],
    dataset_info_dict: dict[str, Any],
    manifest_path: Path,
    total_episodes: int,
    options: RayExecutionOptions,
) -> tuple[dict[int, int], list[str]]:
    """Run per-episode conversion tasks on Ray.

    Returns:
        (episode_frame_counts, errors)
    """

    import ray

    from robodata.convert.converter import _process_episode_worker

    _init_ray(options.address)

    @ray.remote
    def _task(ep_idx: int) -> dict[str, Any]:
        ep_res = _process_episode_worker(
            str(source),
            str(output),
            ep_idx,
            source_format,
            target_format,
            config_dict,
            dataset_info_dict,
            str(manifest_path),
        )
        return {
            "episode_index": ep_res.episode_index,
            "success": ep_res.success,
            "num_frames": ep_res.num_frames,
            "error": ep_res.error,
        }

    refs = [_task.remote(i) for i in range(total_episodes)]

    episode_frame_counts: dict[int, int] = {}
    errors: list[str] = []

    for item in ray.get(refs):
        idx = int(item["episode_index"])
        if item.get("success"):
            episode_frame_counts[idx] = int(item.get("num_frames", 0) or 0)
        else:
            errors.append(f"Episode {idx}: {item.get('error')}")

    return episode_frame_counts, errors
