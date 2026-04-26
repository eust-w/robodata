"""Main converter facade for Robodata.

Provides the high-level conversion API that orchestrates format detection,
reading, and writing.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robodata.config.models import ConversionConfig
from robodata.core.cache import CacheContext, get_memory_manager
from robodata.core.error_handling import ErrorContext, handle_conversion_error
from robodata.core.exceptions import ConversionError, UnsupportedFormatError
from robodata.core.models import DatasetInfo, Episode
from robodata.core.manifest import DatasetManifest, get_default_manifest_path
from robodata.core.parallel import ParallelConfig, WorkerPool, get_optimal_config
from robodata.formats.registry import FormatRegistry


@dataclass
class ConversionResult:
    """Result of a conversion operation.

    Attributes:
        success: True if conversion completed successfully.
        source_format: Detected source format.
        target_format: Target format used.
        episodes_converted: Number of episodes successfully converted.
        episodes_failed: Number of episodes that failed.
        total_frames: Total frames written.
        output_path: Path to output dataset.
        errors: List of error messages for failed episodes.
    """

    success: bool
    source_format: str
    target_format: str
    episodes_converted: int = 0
    episodes_failed: int = 0
    total_frames: int = 0
    output_path: Path | None = None
    errors: list[str] = field(default_factory=list)


@dataclass
class _EpisodeTask:
    """Task descriptor for parallel episode processing."""

    episode_index: int
    num_frames: int | None = None


@dataclass
class _EpisodeResult:
    """Result from processing a single episode."""

    episode_index: int
    success: bool
    num_frames: int = 0
    error: str | None = None


def _process_episode_worker(
    source_path: str,
    output_path: str,
    episode_index: int,
    source_format: str,
    target_format: str,
    config_dict: dict[str, Any],
    dataset_info_dict: dict[str, Any],
    manifest_path: str | None = None,
) -> _EpisodeResult:
    """Worker function for processing a single episode in a separate process.

    This function is designed to be called from a ProcessPoolExecutor.
    It creates its own reader and writer instances to avoid pickling issues.

    Args:
        source_path: Path to source dataset.
        output_path: Path for output.
        episode_index: Index of the episode to process.
        source_format: Source format identifier.
        target_format: Target format identifier.
        config_dict: Serialized ConversionConfig.
        dataset_info_dict: Serialized dataset info for writer configuration.

    Returns:
        _EpisodeResult with success/failure information.
    """
    from pathlib import Path

    from robodata.config.models import ConversionConfig
    from robodata.core.manifest import DatasetManifest, get_default_manifest_path
    from robodata.core.models import CameraInfo, DatasetInfo
    from robodata.formats.registry import FormatRegistry

    try:
        source = Path(source_path)
        output = Path(output_path)

        # Reconstruct config from dict
        config = ConversionConfig.from_dict(config_dict)

        # Get reader and writer
        reader = FormatRegistry.get_reader(source_format)
        writer = FormatRegistry.get_writer(target_format)

        manifest: DatasetManifest | None = None
        manifest_file: Path | None = None
        if manifest_path:
            manifest_file = Path(manifest_path)
            manifest = DatasetManifest.load_or_create(
                output,
                source_path=source,
                source_format=source_format,
                target_format=target_format,
                manifest_path=manifest_file,
            )

        if config.resume and manifest is not None and manifest.is_success(episode_index):
            return _EpisodeResult(
                episode_index=episode_index,
                success=True,
                num_frames=manifest.get_success_frame_count(episode_index) or 0,
            )

        # Reconstruct minimal dataset_info for writer configuration
        dataset_info = DatasetInfo(
            path=source,
            format=source_format,
        )
        dataset_info.inferred_fps = dataset_info_dict.get("inferred_fps")
        dataset_info.inferred_robot_type = dataset_info_dict.get("inferred_robot_type")

        # Reconstruct cameras
        for cam_name, cam_data in dataset_info_dict.get("cameras", {}).items():
            dataset_info.cameras[cam_name] = CameraInfo(
                name=cam_name,
                height=cam_data["height"],
                width=cam_data["width"],
                channels=cam_data.get("channels", 3),
                encoding=cam_data.get("encoding", "rgb"),
            )

        # Apply config overrides
        if config.fps is not None:
            dataset_info.inferred_fps = config.fps
        if config.robot_type is not None:
            dataset_info.inferred_robot_type = config.robot_type

        # Configure writer
        if hasattr(writer, "config"):
            wconfig = writer.config
            if hasattr(wconfig, "fps") and dataset_info.inferred_fps:
                wconfig.fps = dataset_info.inferred_fps
            if hasattr(wconfig, "robot_type") and dataset_info.inferred_robot_type:
                wconfig.robot_type = dataset_info.inferred_robot_type
            if hasattr(wconfig, "camera_name_mapping") and config.camera_mapping:
                wconfig.camera_name_mapping = config.camera_mapping
            if hasattr(wconfig, "field_mapping") and config.field_mapping:
                wconfig.field_mapping = config.field_mapping
            if hasattr(wconfig, "action_field") and config.action_field:
                wconfig.action_field = config.action_field
            if hasattr(wconfig, "state_field") and config.state_field:
                wconfig.state_field = config.state_field

        # Read and write the specific episode
        num_frames = 0
        episode_id: str | None = None

        # Select shard for parallel/distributed safe writing if supported
        if hasattr(writer, "set_shard_id"):
            writer.set_shard_id(episode_index)  # type: ignore[attr-defined]

        if hasattr(reader, "read_episode_by_index"):
            episode = reader.read_episode_by_index(source, episode_index)  # type: ignore[attr-defined]
            episode_id = episode.episode_id
            frames = episode.load_frames()
            num_frames = len(frames)
            writer.write_episode(episode, output, episode_index=episode_index)
        else:
            episodes_iter = reader.read_episodes(source)
            for idx, episode in enumerate(episodes_iter):
                if idx == episode_index:
                    episode_id = episode.episode_id
                    # Load frames to get count (writer will iterate them anyway)
                    frames = episode.load_frames()
                    num_frames = len(frames)
                    writer.write_episode(episode, output, episode_index=episode_index)
                    break
                elif idx > episode_index:
                    break

        if manifest is not None and manifest_file is not None:
            if num_frames > 0:
                manifest.mark_success(episode_index, episode_id=episode_id, num_frames=num_frames)
            else:
                manifest.mark_success(episode_index, episode_id=episode_id, num_frames=None)
            manifest.save(output, manifest_path=manifest_file)

        return _EpisodeResult(
            episode_index=episode_index,
            success=True,
            num_frames=num_frames,
        )

    except Exception as e:
        try:
            if manifest_path:
                output = Path(output_path)
                manifest_file = Path(manifest_path)
                manifest = DatasetManifest.load_or_create(
                    output,
                    source_path=Path(source_path),
                    source_format=source_format,
                    target_format=target_format,
                    manifest_path=manifest_file,
                )
                manifest.mark_failed(episode_index, episode_id=None, error=e)
                manifest.save(output, manifest_path=manifest_file)
        except Exception:
            pass
        return _EpisodeResult(
            episode_index=episode_index,
            success=False,
            error=str(e),
        )


class Converter:
    """Main conversion facade.

    Orchestrates format detection, inspection, reading, and writing to
    convert datasets between formats.

    Example:
        >>> converter = Converter()
        >>> result = converter.convert(
        ...     source="./rlds_dataset",
        ...     output="./lerobot_output",
        ...     target_format="lerobot-v3",
        ... )
        >>> print(f"Converted {result.episodes_converted} episodes")
    """

    def __init__(self, config: ConversionConfig | None = None):
        """Initialize converter with configuration.

        Args:
            config: Conversion configuration. Uses defaults if None.
        """
        self.config = config or ConversionConfig()
        self.logger = logging.getLogger(__name__)

    def convert(
        self,
        source: str | Path,
        output: str | Path,
        target_format: str | None = None,
        source_format: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ConversionResult:
        """Convert a dataset from source to target format.

        Args:
            source: Path to source dataset.
            output: Path for output dataset.
            target_format: Target format (uses config default if None).
            source_format: Source format hint (auto-detect if None).
            progress_callback: Optional callback(stage, current, total) for progress.
                stage is one of: "inspect", "episode", "finalize"

        Returns:
            ConversionResult with success/failure information.

        Raises:
            UnsupportedFormatError: If source or target format not supported.
            ConversionError: If conversion fails and fail_on_error is True.
        """
        source = Path(source)
        output = Path(output)
        target_format = target_format or self.config.target_format

        result = ConversionResult(
            success=False,
            source_format="unknown",
            target_format=target_format,
            output_path=output,
        )

        # 1. Detect source format
        with ErrorContext("source format detection", self.logger, reraise=False):
            source_format = FormatRegistry.detect_format(source)
        
        if source_format is None:
            error_msg = "Unable to detect source format"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise ConversionError("unknown", target_format, error_msg)
            return result
            
        result.source_format = source_format

        # 2. Get reader and writer
        reader = None
        try:
            reader = FormatRegistry.get_reader(source_format)
        except UnsupportedFormatError as e:
            result.errors.append(str(e))
            if self.config.fail_on_error:
                raise  # Re-raise the original UnsupportedFormatError
            return result
            
        writer = None
        try:
            writer = FormatRegistry.get_writer(target_format)
        except UnsupportedFormatError as e:
            error_msg = f"No writer for format: {target_format}"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise  # Re-raise the original UnsupportedFormatError
            return result

        # 3. Inspect source dataset
        with ErrorContext("dataset inspection", self.logger):
            dataset_info = reader.inspect(source)

        # 4. Apply configuration overrides
        self._apply_config_overrides(dataset_info)
        
        # 5. Configure writer
        self._configure_writer(writer, dataset_info)

        # 6. Convert episodes with parallel processing if enabled
        if self.config.num_workers > 1 and hasattr(reader, 'read_episodes'):
            result = self._convert_episodes_parallel(
                reader, source, output, writer, dataset_info, 
                result, progress_callback
            )
        else:
            result = self._convert_episodes_sequential(
                reader, source, output, writer, dataset_info,
                result, progress_callback
            )

        # 7. Finalize output
        if progress_callback:
            progress_callback("finalize", 0, 1)

        try:
            # Update dataset_info with actual counts
            dataset_info.num_episodes = result.episodes_converted
            dataset_info.total_frames = result.total_frames

            writer.finalize(output, dataset_info)
        except Exception as e:
            error_msg = f"Finalization failed: {e}"
            result.errors.append(error_msg)
            if self.config.fail_on_error:
                raise handle_conversion_error(
                    e, source_format, target_format, 
                    "Finalizing output",
                    self.logger
                )
            return result

        if progress_callback:
            progress_callback("finalize", 1, 1)

        result.success = result.episodes_failed == 0
        return result

    def _convert_episodes_sequential(
        self,
        reader: Any,
        source: Path,
        output: Path,
        writer: Any,
        dataset_info: DatasetInfo,
        result: ConversionResult,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> ConversionResult:
        """Convert episodes sequentially."""
        if progress_callback:
            progress_callback("convert", 0, dataset_info.num_episodes)

        manifest = DatasetManifest()
        manifest_path = get_default_manifest_path(output)

        episode_idx = 0
        for episode in reader.read_episodes(source):
            try:
                # Load frames to get count
                frames = episode.load_frames()
                num_frames = len(frames)
                
                # Write episode
                writer.write_episode(episode, output, episode_index=episode_idx)
                
                # Update manifest
                manifest.mark_success(
                    episode_idx, 
                    episode_id=episode.episode_id, 
                    num_frames=num_frames
                )
                manifest.save(output, manifest_path=manifest_path)

            except Exception as e:
                error_msg = f"Episode {episode.episode_id}: {e}"
                result.errors.append(error_msg)

                manifest.mark_failed(
                    episode_idx, 
                    episode_id=episode.episode_id, 
                    error=e
                )
                manifest.save(output, manifest_path=manifest_path)

                if self.config.fail_on_error:
                    raise ConversionError(result.source_format, result.target_format, error_msg)

            episode_idx += 1
            
            if progress_callback:
                progress_callback("convert", episode_idx, dataset_info.num_episodes)

        result.episodes_converted = manifest.success_count()
        result.total_frames = manifest.total_frames_success()
        result.episodes_failed = max(0, episode_idx - result.episodes_converted)
        
        return result

    def _convert_episodes_parallel(
        self,
        reader: Any,
        source: Path,
        output: Path,
        writer: Any,
        dataset_info: DatasetInfo,
        result: ConversionResult,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> ConversionResult:
        """Convert episodes using parallel processing."""
        if progress_callback:
            progress_callback("convert", 0, dataset_info.num_episodes)

        # Get optimal parallel configuration
        config = get_optimal_config()
        config.num_workers = min(config.num_workers, self.config.num_workers)
        
        # Collect episodes first (needed for parallel processing)
        episodes = list(reader.read_episodes(source))
        
        def process_episode(args):
            """Process a single episode in parallel."""
            episode_idx, episode = args
            try:
                # Load frames to get count
                frames = episode.load_frames()
                num_frames = len(frames)
                
                # Write episode
                writer.write_episode(episode, output, episode_index=episode_idx)
                
                return {
                    "success": True,
                    "episode_idx": episode_idx,
                    "episode_id": episode.episode_id,
                    "num_frames": num_frames,
                    "error": None
                }
            except Exception as e:
                return {
                    "success": False,
                    "episode_idx": episode_idx,
                    "episode_id": episode.episode_id,
                    "num_frames": 0,
                    "error": str(e)
                }

        # Process episodes in parallel
        episode_args = list(enumerate(episodes))
        results = []
        
        with WorkerPool(config) as pool:
            results = pool.map(process_episode, episode_args)

        # Update manifest and result
        manifest = DatasetManifest()
        manifest_path = get_default_manifest_path(output)
        
        for episode_result in results:
            if episode_result["success"]:
                manifest.mark_success(
                    episode_result["episode_idx"],
                    episode_id=episode_result["episode_id"],
                    num_frames=episode_result["num_frames"]
                )
            else:
                error_msg = f"Episode {episode_result['episode_id']}: {episode_result['error']}"
                result.errors.append(error_msg)
                
                manifest.mark_failed(
                    episode_result["episode_idx"],
                    episode_id=episode_result["episode_id"],
                    error=Exception(episode_result["error"])
                )
        
        manifest.save(output, manifest_path=manifest_path)
        
        result.episodes_converted = manifest.success_count()
        result.total_frames = manifest.total_frames_success()
        result.episodes_failed = len([episode_result for episode_result in results if not episode_result["success"]])
        
        if progress_callback:
            progress_callback("convert", len(episodes), dataset_info.num_episodes)
        
        return result

    def _apply_config_overrides(self, dataset_info: DatasetInfo) -> None:
        """Apply configuration overrides to dataset info.

        Args:
            dataset_info: Dataset info to modify.
        """
        if self.config.fps is not None:
            dataset_info.inferred_fps = self.config.fps

        if self.config.robot_type is not None:
            dataset_info.inferred_robot_type = self.config.robot_type

    def _configure_writer(self, writer: Any, dataset_info: DatasetInfo) -> None:
        """Configure writer with format-specific options.

        Args:
            writer: Writer instance to configure.
            dataset_info: Dataset info for default values.
        """
        # Check if writer has a config attribute (like LeRobotV3Writer)
        if hasattr(writer, "config"):
            config = writer.config

            # Apply defaults from dataset_info
            if hasattr(config, "fps") and dataset_info.inferred_fps:
                config.fps = dataset_info.inferred_fps

            if hasattr(config, "robot_type") and dataset_info.inferred_robot_type:
                config.robot_type = dataset_info.inferred_robot_type

            if hasattr(config, "camera_name_mapping") and self.config.camera_mapping:
                config.camera_name_mapping = self.config.camera_mapping

            # Apply field mappings if writer supports them
            if hasattr(config, "field_mapping") and self.config.field_mapping:
                config.field_mapping = self.config.field_mapping

            if hasattr(config, "action_field") and self.config.action_field:
                config.action_field = self.config.action_field

            if hasattr(config, "state_field") and self.config.state_field:
                config.state_field = self.config.state_field

            # Apply format-specific config from conversion config
            for key, value in self.config.writer_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    def _convert_parallel(
        self,
        source: Path,
        output: Path,
        source_format: str,
        target_format: str,
        dataset_info: DatasetInfo,
        total_episodes: int,
        result: ConversionResult,
        manifest_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ConversionResult:
        """Convert episodes in parallel using ProcessPoolExecutor.

        Args:
            source: Path to source dataset.
            output: Path for output dataset.
            source_format: Source format identifier.
            target_format: Target format identifier.
            dataset_info: Dataset info from inspection.
            total_episodes: Total number of episodes.
            result: ConversionResult to update.
            progress_callback: Optional progress callback.

        Returns:
            Updated ConversionResult.
        """
        # Serialize config and dataset_info for workers
        config_dict = self.config.to_dict()

        # Create serializable dataset info dict
        dataset_info_dict: dict[str, Any] = {
            "inferred_fps": dataset_info.inferred_fps,
            "inferred_robot_type": dataset_info.inferred_robot_type,
            "cameras": {
                name: {
                    "height": cam.height,
                    "width": cam.width,
                    "channels": cam.channels,
                    "encoding": cam.encoding,
                }
                for name, cam in dataset_info.cameras.items()
            },
        }

        num_workers = min(self.config.num_workers, total_episodes)
        completed = 0

        # Use 'spawn' context for better compatibility across platforms
        ctx = mp.get_context("spawn")

        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            # Submit all episode tasks
            futures = {}
            for episode_idx in range(total_episodes):
                future = executor.submit(
                    _process_episode_worker,
                    str(source),
                    str(output),
                    episode_idx,
                    source_format,
                    target_format,
                    config_dict,
                    dataset_info_dict,
                    str(manifest_path),
                )
                futures[future] = episode_idx

            # Collect results as they complete
            # Track per-episode frame counts to avoid re-reading parquet files in finalize()
            episode_frame_counts: dict[int, int] = {}

            for future in as_completed(futures):
                episode_idx = futures[future]
                completed += 1

                if progress_callback:
                    progress_callback("episode", completed - 1, total_episodes)

                try:
                    ep_result = future.result()

                    if ep_result.success:
                        result.episodes_converted += 1
                        result.total_frames += ep_result.num_frames
                        episode_frame_counts[ep_result.episode_index] = ep_result.num_frames
                    else:
                        result.episodes_failed += 1
                        error_msg = f"Episode {episode_idx}: {ep_result.error}"
                        result.errors.append(error_msg)

                        if self.config.fail_on_error:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            raise ConversionError(
                                source_format, target_format, error_msg
                            )

                except Exception as e:
                    if isinstance(e, ConversionError):
                        raise
                    result.episodes_failed += 1
                    error_msg = f"Episode {episode_idx}: {e}"
                    result.errors.append(error_msg)

                    if self.config.fail_on_error:
                        raise ConversionError(source_format, target_format, error_msg)

        # Store episode frame counts in dataset_info for finalize()
        # This avoids re-reading parquet files to get lengths
        dataset_info.metadata["_parallel_episode_frame_counts"] = episode_frame_counts

        return result

    def _convert_ray(
        self,
        source: Path,
        output: Path,
        source_format: str,
        target_format: str,
        dataset_info: DatasetInfo,
        total_episodes: int,
        result: ConversionResult,
        manifest_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ConversionResult:
        """Convert episodes using Ray as a distributed executor."""
        try:
            from robodata.execution.ray import (
                RayExecutionOptions,
                is_ray_available,
                run_parallel_episode_tasks_ray,
            )
        except Exception as e:
            result.errors.append(f"Ray executor unavailable: {e}")
            if self.config.fail_on_error:
                raise ConversionError(source_format, target_format, str(e))
            return result

        if not is_ray_available():
            msg = "ray is not installed"
            result.errors.append(f"Ray executor unavailable: {msg}")
            if self.config.fail_on_error:
                raise ConversionError(source_format, target_format, msg)
            return result

        config_dict = self.config.to_dict()
        dataset_info_dict: dict[str, Any] = {
            "inferred_fps": dataset_info.inferred_fps,
            "inferred_robot_type": dataset_info.inferred_robot_type,
            "cameras": {
                name: {
                    "height": cam.height,
                    "width": cam.width,
                    "channels": cam.channels,
                    "encoding": cam.encoding,
                }
                for name, cam in dataset_info.cameras.items()
            },
        }

        opts = RayExecutionOptions(address=self.config.ray_address)
        episode_frame_counts, errors = run_parallel_episode_tasks_ray(
            source=source,
            output=output,
            source_format=source_format,
            target_format=target_format,
            config_dict=config_dict,
            dataset_info_dict=dataset_info_dict,
            manifest_path=manifest_path,
            total_episodes=total_episodes,
            options=opts,
        )

        result.errors.extend(errors)
        result.episodes_failed = len(errors)
        result.episodes_converted = total_episodes - result.episodes_failed
        result.total_frames = sum(episode_frame_counts.values())

        dataset_info.metadata["_parallel_episode_frame_counts"] = episode_frame_counts

        if progress_callback:
            progress_callback("episode", total_episodes, total_episodes)

        return result


def convert(
    source: str | Path,
    output: str | Path,
    target_format: str = "lerobot-v3",
    source_format: str | None = None,
    fps: float | None = None,
    robot_type: str | None = None,
    camera_mapping: dict[str, str] | None = None,
    fail_on_error: bool = False,
    **kwargs: Any,
) -> ConversionResult:
    """Convenience function to convert a dataset.

    Args:
        source: Path to source dataset.
        output: Path for output dataset.
        target_format: Target format (default: "lerobot-v3").
        source_format: Source format hint (auto-detect if None).
        fps: Override frames per second.
        robot_type: Override robot type.
        camera_mapping: Map source camera names to target names.
        fail_on_error: Stop on first error vs continue and log.
        **kwargs: Additional writer-specific configuration.

    Returns:
        ConversionResult with success/failure information.
    """
    config = ConversionConfig(
        target_format=target_format,
        fps=fps,
        robot_type=robot_type,
        camera_mapping=camera_mapping or {},
        fail_on_error=fail_on_error,
        writer_config=kwargs,
    )

    converter = Converter(config)
    return converter.convert(source, output, target_format, source_format)
