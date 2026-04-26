"""Strategy pattern for different data conversion approaches."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from robodata.core.models import DatasetInfo, Episode, Frame
from robodata.core.parallel import ParallelConfig, parallel_map
from robodata.formats.base import FormatReader, FormatWriter

logger = logging.getLogger(__name__)


class ConversionStrategy(Enum):
    """Available conversion strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    STREAMING = "streaming"
    BATCH = "batch"


@dataclass
class ConversionContext:
    """Context information for conversion operations."""
    
    source_path: Path
    output_path: Path
    source_format: str
    target_format: str
    dataset_info: DatasetInfo
    config: Dict[str, Any]
    parallel_config: Optional[ParallelConfig] = None
    
    @property
    def episode_count(self) -> int:
        """Get number of episodes."""
        return self.dataset_info.num_episodes
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames."""
        return self.dataset_info.total_frames or 0


@runtime_checkable
class ProgressCallback(Protocol):
    """Protocol for progress reporting."""
    
    def __call__(self, stage: str, current: int, total: int, message: str = "") -> None:
        """Report progress.
        
        Args:
            stage: Current conversion stage.
            current: Current progress count.
            total: Total count.
            message: Optional progress message.
        """
        ...


class ConversionStrategyBase(ABC):
    """Abstract base class for conversion strategies."""
    
    def __init__(self, name: str):
        """Initialize strategy.
        
        Args:
            name: Strategy name for logging.
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def execute(
        self,
        context: ConversionContext,
        reader: FormatReader,
        writer: FormatWriter,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute the conversion strategy.
        
        Args:
            context: Conversion context.
            reader: Format reader instance.
            writer: Format writer instance.
            progress_callback: Optional progress callback.
            
        Returns:
            Conversion result statistics.
        """
        pass
    
    def _log_start(self, context: ConversionContext) -> None:
        """Log strategy start.
        
        Args:
            context: Conversion context.
        """
        self.logger.info(
            f"Starting {self.name} conversion: "
            f"{context.source_format} → {context.target_format} "
            f"({context.episode_count} episodes, {context.total_frames} frames)"
        )
    
    def _log_complete(self, result: Dict[str, Any]) -> None:
        """Log strategy completion.
        
        Args:
            result: Conversion result.
        """
        self.logger.info(
            f"Completed {self.name} conversion: "
            f"{result.get('episodes_converted', 0)} episodes, "
            f"{result.get('total_frames', 0)} frames"
        )


class SequentialConversionStrategy(ConversionStrategyBase):
    """Sequential conversion strategy.
    
    Processes episodes one by one in a single thread.
    Suitable for small datasets or when memory is limited.
    """
    
    def __init__(self):
        super().__init__("sequential")
    
    def execute(
        self,
        context: ConversionContext,
        reader: FormatReader,
        writer: FormatWriter,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute sequential conversion."""
        self._log_start(context)
        
        episodes_converted = 0
        total_frames = 0
        failed_episodes = 0
        errors: List[str] = []
        
        try:
            for episode_idx, episode in enumerate(reader.read_episodes(context.source_path)):
                if progress_callback:
                    progress_callback("convert", episode_idx, context.episode_count)
                
                try:
                    # Write episode
                    writer.write_episode(episode, context.output_path, episode_idx)
                    episodes_converted += 1
                    
                    # Count frames
                    frame_count = len(list(episode.frames()))
                    total_frames += frame_count
                    
                except Exception as e:
                    failed_episodes += 1
                    error_msg = f"Episode {episode_idx}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    if context.config.get("fail_on_error", False):
                        raise
        
        finally:
            if progress_callback:
                progress_callback("finalize", 0, 1)
            
            # Finalize output
            try:
                writer.finalize(context.output_path, context.dataset_info)
            except Exception as e:
                errors.append(f"Finalization failed: {e}")
                self.logger.error(f"Finalization failed: {e}")
        
        result = {
            "strategy": self.name,
            "episodes_converted": episodes_converted,
            "total_frames": total_frames,
            "episodes_failed": failed_episodes,
            "errors": errors,
            "success": failed_episodes == 0,
        }
        
        self._log_complete(result)
        return result


class ParallelConversionStrategy(ConversionStrategyBase):
    """Parallel conversion strategy.
    
    Processes episodes in parallel using multiple processes/threads.
    Suitable for large datasets with sufficient memory.
    """
    
    def __init__(self, parallel_config: Optional[ParallelConfig] = None):
        super().__init__("parallel")
        self.parallel_config = parallel_config or ParallelConfig()
    
    def execute(
        self,
        context: ConversionContext,
        reader: FormatReader,
        writer: FormatWriter,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute parallel conversion."""
        self._log_start(context)
        
        # Prepare episodes for parallel processing
        episodes = list(reader.read_episodes(context.source_path))
        
        def process_episode(args: tuple[int, Episode]) -> Dict[str, Any]:
            """Process a single episode in parallel."""
            episode_idx, episode = args
            try:
                # Create writer instance for this process
                from robodata.formats.factory import get_writer
                local_writer = get_writer(context.target_format)
                
                # Write episode
                local_writer.write_episode(episode, context.output_path, episode_idx)
                
                frame_count = len(list(episode.frames()))
                
                return {
                    "episode_idx": episode_idx,
                    "success": True,
                    "frame_count": frame_count,
                    "error": None,
                }
            except Exception as e:
                return {
                    "episode_idx": episode_idx,
                    "success": False,
                    "frame_count": 0,
                    "error": str(e),
                }
        
        # Process episodes in parallel
        episode_args = list(enumerate(episodes))
        results = parallel_map(
            process_episode,
            episode_args,
            num_workers=self.parallel_config.num_workers,
            use_processes=self.parallel_config.use_processes,
            chunk_size=self.parallel_config.chunk_size,
            progress_callback=lambda current, total: (
                progress_callback("convert", current, total) if progress_callback else None
            ),
        )
        
        # Collect results
        episodes_converted = 0
        total_frames = 0
        failed_episodes = 0
        errors: List[str] = []
        
        for result in results:
            if result["success"]:
                episodes_converted += 1
                total_frames += result["frame_count"]
            else:
                failed_episodes += 1
                errors.append(f"Episode {result['episode_idx']}: {result['error']}")
                
                if context.config.get("fail_on_error", False):
                    raise Exception(f"Parallel conversion failed: {result['error']}")
        
        # Finalize output
        if progress_callback:
            progress_callback("finalize", 0, 1)
        
        try:
            writer.finalize(context.output_path, context.dataset_info)
        except Exception as e:
            errors.append(f"Finalization failed: {e}")
            self.logger.error(f"Finalization failed: {e}")
        
        result = {
            "strategy": self.name,
            "parallel_config": {
                "num_workers": self.parallel_config.num_workers,
                "use_processes": self.parallel_config.use_processes,
            },
            "episodes_converted": episodes_converted,
            "total_frames": total_frames,
            "episodes_failed": failed_episodes,
            "errors": errors,
            "success": failed_episodes == 0,
        }
        
        self._log_complete(result)
        return result


class StreamingConversionStrategy(ConversionStrategyBase):
    """Streaming conversion strategy.
    
    Processes data in a streaming fashion without loading entire episodes
    into memory. Suitable for very large datasets.
    """
    
    def __init__(self):
        super().__init__("streaming")
    
    def execute(
        self,
        context: ConversionContext,
        reader: FormatReader,
        writer: FormatWriter,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute streaming conversion."""
        self._log_start(context)
        
        episodes_converted = 0
        total_frames = 0
        failed_episodes = 0
        errors: List[str] = []
        
        # Stream episodes without loading all into memory
        for episode_idx, episode in enumerate(reader.read_episodes(context.source_path)):
            if progress_callback:
                progress_callback("convert", episode_idx, context.episode_count)
            
            try:
                # Process frames in streaming fashion
                frame_count = 0
                for frame in episode.frames():
                    frame_count += 1
                    # Process frame without storing all frames in memory
                
                # Write episode
                writer.write_episode(episode, context.output_path, episode_idx)
                episodes_converted += 1
                total_frames += frame_count
                
            except Exception as e:
                failed_episodes += 1
                error_msg = f"Episode {episode_idx}: {e}"
                errors.append(error_msg)
                self.logger.error(error_msg)
                
                if context.config.get("fail_on_error", False):
                    raise
        
        # Finalize output
        if progress_callback:
            progress_callback("finalize", 0, 1)
        
        try:
            writer.finalize(context.output_path, context.dataset_info)
        except Exception as e:
            errors.append(f"Finalization failed: {e}")
            self.logger.error(f"Finalization failed: {e}")
        
        result = {
            "strategy": self.name,
            "episodes_converted": episodes_converted,
            "total_frames": total_frames,
            "episodes_failed": failed_episodes,
            "errors": errors,
            "success": failed_episodes == 0,
        }
        
        self._log_complete(result)
        return result


class BatchConversionStrategy(ConversionStrategyBase):
    """Batch conversion strategy.
    
    Processes episodes in batches to balance memory usage and performance.
    """
    
    def __init__(self, batch_size: int = 10):
        super().__init__("batch")
        self.batch_size = batch_size
    
    def execute(
        self,
        context: ConversionContext,
        reader: FormatReader,
        writer: FormatWriter,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Execute batch conversion."""
        self._log_start(context)
        
        episodes_converted = 0
        total_frames = 0
        failed_episodes = 0
        errors: List[str] = []
        
        # Process episodes in batches
        episode_batch = []
        batch_idx = 0
        
        for episode_idx, episode in enumerate(reader.read_episodes(context.source_path)):
            episode_batch.append((episode_idx, episode))
            
            # Process batch when full or at end
            if len(episode_batch) >= self.batch_size or episode_idx == context.episode_count - 1:
                if progress_callback:
                    progress_callback("convert", episode_idx, context.episode_count)
                
                # Process current batch
                for batch_episode_idx, batch_episode in episode_batch:
                    try:
                        writer.write_episode(batch_episode, context.output_path, batch_episode_idx)
                        episodes_converted += 1
                        
                        frame_count = len(list(batch_episode.frames()))
                        total_frames += frame_count
                        
                    except Exception as e:
                        failed_episodes += 1
                        error_msg = f"Episode {batch_episode_idx}: {e}"
                        errors.append(error_msg)
                        self.logger.error(error_msg)
                        
                        if context.config.get("fail_on_error", False):
                            raise
                
                # Clear batch
                episode_batch.clear()
                batch_idx += 1
        
        # Finalize output
        if progress_callback:
            progress_callback("finalize", 0, 1)
        
        try:
            writer.finalize(context.output_path, context.dataset_info)
        except Exception as e:
            errors.append(f"Finalization failed: {e}")
            self.logger.error(f"Finalization failed: {e}")
        
        result = {
            "strategy": self.name,
            "batch_size": self.batch_size,
            "batches_processed": batch_idx,
            "episodes_converted": episodes_converted,
            "total_frames": total_frames,
            "episodes_failed": failed_episodes,
            "errors": errors,
            "success": failed_episodes == 0,
        }
        
        self._log_complete(result)
        return result


class ConversionStrategySelector:
    """Selects the appropriate conversion strategy based on context."""
    
    def __init__(self):
        """Initialize selector."""
        self.strategies = {
            ConversionStrategy.SEQUENTIAL: SequentialConversionStrategy(),
            ConversionStrategy.PARALLEL: ParallelConversionStrategy(),
            ConversionStrategy.STREAMING: StreamingConversionStrategy(),
            ConversionStrategy.BATCH: BatchConversionStrategy(),
        }
    
    def select_strategy(
        self,
        context: ConversionContext,
        strategy_hint: Optional[ConversionStrategy] = None,
    ) -> ConversionStrategyBase:
        """Select the best conversion strategy.
        
        Args:
            context: Conversion context.
            strategy_hint: Optional strategy hint from user.
            
        Returns:
            Selected conversion strategy.
        """
        if strategy_hint:
            return self.strategies[strategy_hint]
        
        # Auto-select based on dataset characteristics
        episode_count = context.episode_count
        total_frames = context.total_frames
        parallel_config = context.parallel_config
        
        # For very small datasets, use sequential
        if episode_count < 10 or total_frames < 1000:
            return self.strategies[ConversionStrategy.SEQUENTIAL]
        
        # For large datasets with parallel config, use parallel
        if episode_count > 100 and parallel_config and parallel_config.num_workers > 1:
            return self.strategies[ConversionStrategy.PARALLEL]
        
        # For very large datasets, use streaming
        if total_frames > 100000:  # 100k+ frames
            return self.strategies[ConversionStrategy.STREAMING]
        
        # Default to batch for medium datasets
        return self.strategies[ConversionStrategy.BATCH]
    
    def register_strategy(
        self,
        strategy_type: ConversionStrategy,
        strategy: ConversionStrategyBase,
    ) -> None:
        """Register a custom strategy.
        
        Args:
            strategy_type: Strategy type identifier.
            strategy: Strategy instance.
        """
        self.strategies[strategy_type] = strategy
    
    def get_available_strategies(self) -> List[ConversionStrategy]:
        """Get list of available strategies.
        
        Returns:
            Available strategy types.
        """
        return list(self.strategies.keys())
