"""Advanced streaming and distributed processing system for Robodata.

Provides efficient streaming capabilities for large datasets and
distributed processing across multiple nodes.
"""

from __future__ import annotations

import asyncio
import pickle
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import numpy as np

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame


class StreamingMode(Enum):
    """Streaming processing modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    ADAPTIVE = "adaptive"


class ChunkStrategy(Enum):
    """Data chunking strategies."""
    FIXED_SIZE = "fixed_size"
    TIME_BASED = "time_based"
    EPISODE_BASED = "episode_based"
    ADAPTIVE = "adaptive"


@dataclass
class StreamingConfig:
    """Configuration for streaming operations."""
    mode: StreamingMode = StreamingMode.SEQUENTIAL
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE
    chunk_size: int = 1000
    max_memory_mb: int = 1024
    buffer_size: int = 10
    num_workers: int = 1
    enable_prefetch: bool = True
    prefetch_depth: int = 2
    enable_compression: bool = False
    compression_level: int = 6


@dataclass
class DataChunk:
    """Chunk of data for streaming."""
    chunk_id: str
    data: List[Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def size_bytes(self) -> int:
        """Estimate size in bytes."""
        return len(pickle.dumps(self.data))


class StreamProcessor(ABC):
    """Abstract base class for stream processors."""
    
    @abstractmethod
    async def process_chunk(self, chunk: DataChunk) -> DataChunk:
        """Process a data chunk."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Processor name."""
        pass


class Chunker:
    """Data chunking utilities."""
    
    def __init__(self, strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE):
        self.strategy = strategy
        self.logger = get_logger(__name__)
    
    def chunk_episodes(
        self, 
        episodes: List[Episode], 
        chunk_size: int = 1000
    ) -> Iterator[DataChunk]:
        """Chunk episodes according to strategy."""
        if self.strategy == ChunkStrategy.FIXED_SIZE:
            yield from self._chunk_by_size(episodes, chunk_size)
        elif self.strategy == ChunkStrategy.EPISODE_BASED:
            yield from self._chunk_by_episode(episodes)
        elif self.strategy == ChunkStrategy.TIME_BASED:
            yield from self._chunk_by_time(episodes, chunk_size)
        elif self.strategy == ChunkStrategy.ADAPTIVE:
            yield from self._chunk_adaptive(episodes, chunk_size)
    
    def _chunk_by_size(self, episodes: List[Episode], chunk_size: int) -> Iterator[DataChunk]:
        """Chunk by fixed number of frames."""
        current_chunk = []
        chunk_id = 0
        
        for episode in episodes:
            for frame in episode.load_frames():
                current_chunk.append(frame)
                
                if len(current_chunk) >= chunk_size:
                    yield DataChunk(
                        chunk_id=f"chunk_{chunk_id}",
                        data=current_chunk.copy(),
                        metadata={"strategy": "fixed_size", "size": len(current_chunk)}
                    )
                    current_chunk.clear()
                    chunk_id += 1
        
        # Yield remaining frames
        if current_chunk:
            yield DataChunk(
                chunk_id=f"chunk_{chunk_id}",
                data=current_chunk,
                metadata={"strategy": "fixed_size", "size": len(current_chunk)}
            )
    
    def _chunk_by_episode(self, episodes: List[Episode]) -> Iterator[DataChunk]:
        """Chunk by episode boundaries."""
        for i, episode in enumerate(episodes):
            frames = list(episode.load_frames())
            yield DataChunk(
                chunk_id=f"episode_{episode.episode_id}",
                data=frames,
                metadata={
                    "strategy": "episode_based",
                    "episode_id": episode.episode_id,
                    "episode_index": i
                }
            )
    
    def _chunk_by_time(self, episodes: List[Episode], duration_seconds: float) -> Iterator[DataChunk]:
        """Chunk by time duration."""
        current_chunk = []
        chunk_start_time = None
        chunk_id = 0
        
        for episode in episodes:
            for frame in episode.load_frames():
                if chunk_start_time is None:
                    chunk_start_time = frame.timestamp
                
                current_chunk.append(frame)
                
                if frame.timestamp and chunk_start_time:
                    if frame.timestamp - chunk_start_time >= duration_seconds:
                        yield DataChunk(
                            chunk_id=f"time_chunk_{chunk_id}",
                            data=current_chunk.copy(),
                            metadata={
                                "strategy": "time_based",
                                "duration": frame.timestamp - chunk_start_time,
                                "start_time": chunk_start_time
                            }
                        )
                        current_chunk.clear()
                        chunk_start_time = frame.timestamp
                        chunk_id += 1
        
        if current_chunk:
            yield DataChunk(
                chunk_id=f"time_chunk_{chunk_id}",
                data=current_chunk,
                metadata={"strategy": "time_based", "remaining": True}
            )
    
    def _chunk_adaptive(self, episodes: List[Episode], target_size_mb: float) -> Iterator[DataChunk]:
        """Chunk by memory size (adaptive)."""
        current_chunk = []
        current_size = 0
        chunk_id = 0
        
        for episode in episodes:
            for frame in episode.load_frames():
                frame_size = len(pickle.dumps(frame))
                
                if current_size + frame_size > target_size_mb * 1024 * 1024 and current_chunk:
                    yield DataChunk(
                        chunk_id=f"adaptive_chunk_{chunk_id}",
                        data=current_chunk.copy(),
                        metadata={
                            "strategy": "adaptive",
                            "size_bytes": current_size,
                            "size_mb": current_size / (1024 * 1024)
                        }
                    )
                    current_chunk = [frame]
                    current_size = frame_size
                    chunk_id += 1
                else:
                    current_chunk.append(frame)
                    current_size += frame_size
        
        if current_chunk:
            yield DataChunk(
                chunk_id=f"adaptive_chunk_{chunk_id}",
                data=current_chunk,
                metadata={
                    "strategy": "adaptive",
                    "size_bytes": current_size,
                    "size_mb": current_size / (1024 * 1024)
                }
            )


class StreamBuffer:
    """Buffer for managing streaming data flow."""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.buffer: List[DataChunk] = []
        self.logger = get_logger(__name__)
        self._stats = {
            "chunks_received": 0,
            "chunks_sent": 0,
            "buffer_overflows": 0,
            "avg_buffer_size": 0.0
        }
    
    async def put(self, chunk: DataChunk) -> None:
        """Put chunk into buffer."""
        if len(self.buffer) >= self.max_size:
            self._stats["buffer_overflows"] += 1
            # Remove oldest chunk
            self.buffer.pop(0)
            self.logger.warning("Buffer overflow, dropping oldest chunk")
        
        self.buffer.append(chunk)
        self._stats["chunks_received"] += 1
    
    async def get(self) -> Optional[DataChunk]:
        """Get chunk from buffer."""
        if self.buffer:
            chunk = self.buffer.pop(0)
            self._stats["chunks_sent"] += 1
            return chunk
        return None
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.buffer) >= self.max_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        self._stats["avg_buffer_size"] = self._stats["chunks_received"] / max(1, self._stats["chunks_sent"])
        return self._stats.copy()


class StreamingEngine:
    """Main streaming engine for data processing."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.chunker = Chunker(config.chunk_strategy)
        self.buffer = StreamBuffer(config.buffer_size)
        self.processors: List[StreamProcessor] = []
        self._stats = {
            "chunks_processed": 0,
            "total_processing_time": 0.0,
            "avg_chunk_time": 0.0,
            "memory_peak": 0.0
        }
    
    def add_processor(self, processor: StreamProcessor) -> None:
        """Add a stream processor."""
        self.processors.append(processor)
        self.logger.info(f"Added stream processor: {processor.name}")
    
    async def process_stream(self, episodes: List[Episode]) -> AsyncIterator[DataChunk]:
        """Process episodes as a stream."""
        self.logger.info(f"Starting stream processing with {len(episodes)} episodes")
        
        # Start prefetch task if enabled
        prefetch_task = None
        if self.config.enable_prefetch:
            prefetch_task = asyncio.create_task(self._prefetch_chunks(episodes))
        
        try:
            # Process chunks
            async for chunk in self._process_chunks():
                yield chunk
        
        finally:
            if prefetch_task:
                prefetch_task.cancel()
                try:
                    await prefetch_task
                except asyncio.CancelledError:
                    pass
        
        self.logger.info("Stream processing completed")
    
    async def _prefetch_chunks(self, episodes: List[Episode]) -> None:
        """Prefetch chunks into buffer."""
        chunk_generator = self.chunker.chunk_episodes(episodes, self.config.chunk_size)
        
        async def put_chunk(chunk):
            await self.buffer.put(chunk)
        
        for chunk in chunk_generator:
            await put_chunk(chunk)
    
    async def _process_chunks(self) -> AsyncIterator[DataChunk]:
        """Process chunks from buffer."""
        while not self.buffer.is_empty() or self._should_continue_processing():
            chunk = await self.buffer.get()
            if chunk is None:
                await asyncio.sleep(0.01)  # Wait for chunks
                continue
            
            start_time = time.time()
            
            # Process through all processors
            processed_chunk = chunk
            for processor in self.processors:
                processed_chunk = await processor.process_chunk(processed_chunk)
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            yield processed_chunk
    
    def _should_continue_processing(self) -> bool:
        """Check if processing should continue."""
        # In a real implementation, this would check for completion conditions
        return True
    
    def _update_stats(self, processing_time: float) -> None:
        """Update processing statistics."""
        self._stats["chunks_processed"] += 1
        self._stats["total_processing_time"] += processing_time
        self._stats["avg_chunk_time"] = (
            self._stats["total_processing_time"] / self._stats["chunks_processed"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        return {
            "streaming_stats": self._stats.copy(),
            "buffer_stats": self.buffer.get_stats(),
            "config": {
                "mode": self.config.mode.value,
                "chunk_strategy": self.config.chunk_strategy.value,
                "chunk_size": self.config.chunk_size,
                "num_workers": self.config.num_workers
            }
        }


class DistributedStreamProcessor:
    """Distributed stream processing using multiple nodes."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.logger = get_logger(__name__)
        self.nodes: List[str] = []
        self.current_node = 0
    
    def add_node(self, node_address: str) -> None:
        """Add a processing node."""
        self.nodes.append(node_address)
        self.logger.info(f"Added processing node: {node_address}")
    
    async def distribute_chunk(self, chunk: DataChunk) -> DataChunk:
        """Distribute chunk to available node."""
        if not self.nodes:
            # Fallback to local processing
            return chunk
        
        # Simple round-robin distribution
        node = self.nodes[self.current_node]
        self.current_node = (self.current_node + 1) % len(self.nodes)
        
        # In a real implementation, this would send chunk to remote node
        self.logger.debug(f"Distributing chunk {chunk.chunk_id} to node {node}")
        
        # Simulate remote processing
        await asyncio.sleep(0.1)
        
        return chunk
    
    async def collect_results(self) -> List[DataChunk]:
        """Collect results from all nodes."""
        # In a real implementation, this would collect from remote nodes
        return []


class AdaptiveStreamProcessor(StreamProcessor):
    """Adaptive stream processor that adjusts based on load."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.processing_times: List[float] = []
        self.adaptation_threshold = 0.1  # 10% change threshold
    
    @property
    def name(self) -> str:
        return "adaptive_stream_processor"
    
    async def process_chunk(self, chunk: DataChunk) -> DataChunk:
        """Process chunk with adaptive strategy."""
        start_time = time.time()
        
        # Adaptive processing based on chunk size and load
        if len(chunk) > 1000:
            # Large chunk - use optimized processing
            processed_data = await self._process_large_chunk(chunk.data)
        else:
            # Small chunk - use standard processing
            processed_data = await self._process_standard_chunk(chunk.data)
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        # Keep only recent times for adaptation
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-50:]
        
        # Adapt processing strategy
        self._adapt_strategy()
        
        return DataChunk(
            chunk_id=chunk.chunk_id + "_adapted",
            data=processed_data,
            metadata={**chunk.metadata, "adaptive_processing": True}
        )
    
    async def _process_large_chunk(self, data: List[Any]) -> List[Any]:
        """Process large chunk with optimized method."""
        # Simulate optimized processing
        await asyncio.sleep(0.01)
        return data  # Placeholder for actual processing
    
    async def _process_standard_chunk(self, data: List[Any]) -> List[Any]:
        """Process standard chunk."""
        # Simulate standard processing
        await asyncio.sleep(0.05)
        return data  # Placeholder for actual processing
    
    def _adapt_strategy(self) -> None:
        """Adapt processing strategy based on performance."""
        if len(self.processing_times) < 10:
            return
        
        recent_avg = np.mean(self.processing_times[-10:])
        older_avg = np.mean(self.processing_times[-20:-10]) if len(self.processing_times) >= 20 else recent_avg
        
        if older_avg > 0:
            change_ratio = (recent_avg - older_avg) / older_avg
            
            if abs(change_ratio) > self.adaptation_threshold:
                if change_ratio > 0:
                    self.logger.warning("Processing time increasing, consider optimization")
                else:
                    self.logger.info("Processing time improving")


class StreamValidator(StreamProcessor):
    """Stream validation processor."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.validation_errors: List[str] = []
    
    @property
    def name(self) -> str:
        return "stream_validator"
    
    async def process_chunk(self, chunk: DataChunk) -> DataChunk:
        """Validate chunk data."""
        errors = []
        
        for i, item in enumerate(chunk.data):
            if hasattr(item, 'action') and item.action is not None:
                if np.any(np.isnan(item.action)):
                    errors.append(f"NaN in action at index {i}")
            
            if hasattr(item, 'state') and item.state is not None:
                if np.any(np.isnan(item.state)):
                    errors.append(f"NaN in state at index {i}")
        
        if errors:
            self.validation_errors.extend(errors)
            self.logger.warning(f"Validation errors in chunk {chunk.chunk_id}: {len(errors)}")
        
        return DataChunk(
            chunk_id=chunk.chunk_id + "_validated",
            data=chunk.data,
            metadata={
                **chunk.metadata,
                "validation_errors": len(errors),
                "validation_passed": len(errors) == 0
            }
        )


# Convenience functions
async def stream_process_episodes(
    episodes: List[Episode],
    config: Optional[StreamingConfig] = None,
    processors: Optional[List[StreamProcessor]] = None
) -> List[DataChunk]:
    """Stream process episodes."""
    if config is None:
        config = StreamingConfig()
    
    engine = StreamingEngine(config)
    
    if processors:
        for processor in processors:
            engine.add_processor(processor)
    
    results = []
    async for chunk in engine.process_stream(episodes):
        results.append(chunk)
    
    return results


def create_streaming_config(
    mode: StreamingMode = StreamingMode.SEQUENTIAL,
    chunk_strategy: ChunkStrategy = ChunkStrategy.FIXED_SIZE,
    chunk_size: int = 1000,
    num_workers: int = 1,
    **kwargs
) -> StreamingConfig:
    """Create streaming configuration."""
    return StreamingConfig(
        mode=mode,
        chunk_strategy=chunk_strategy,
        chunk_size=chunk_size,
        num_workers=num_workers,
        **kwargs
    )


class StreamMetrics:
    """Metrics collection for streaming operations."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = {
            "throughput": [],
            "latency": [],
            "memory_usage": [],
            "error_rates": [],
            "buffer_utilization": []
        }
    
    def record_throughput(self, chunks_per_second: float) -> None:
        """Record throughput metric."""
        self.metrics["throughput"].append(chunks_per_second)
    
    def record_latency(self, latency_ms: float) -> None:
        """Record latency metric."""
        self.metrics["latency"].append(latency_ms)
    
    def record_memory_usage(self, memory_mb: float) -> None:
        """Record memory usage metric."""
        self.metrics["memory_usage"].append(memory_mb)
    
    def record_error_rate(self, error_rate: float) -> None:
        """Record error rate metric."""
        self.metrics["error_rates"].append(error_rate)
    
    def record_buffer_utilization(self, utilization: float) -> None:
        """Record buffer utilization."""
        self.metrics["buffer_utilization"].append(utilization)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "std": np.std(values)
                }
            else:
                summary[metric_name] = {}
        
        return summary
