"""Intelligent data preprocessing and augmentation system for Robodata.

Provides automated data cleaning, enhancement, and preparation
for machine learning pipelines.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import interpolate
from scipy.signal import savgol_filter

from robodata.core.logging_system import get_logger
from robodata.core.models import DatasetInfo, Episode, Frame, LazyImage


class PreprocessingType(Enum):
    """Types of preprocessing operations."""
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    AUGMENTATION = "augmentation"
    FILTERING = "filtering"
    SAMPLING = "sampling"
    TRANSFORMATION = "transformation"


class AugmentationType(Enum):
    """Types of data augmentation."""
    NOISE_ADDITION = "noise_addition"
    TEMPORAL_WARPING = "temporal_warping"
    SPATIAL_TRANSFORMATION = "spatial_transformation"
    DROPOUT = "dropout"
    MIXUP = "mixup"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing operations."""
    enable_cleaning: bool = True
    enable_normalization: bool = True
    enable_augmentation: bool = False
    cleaning_config: Dict[str, Any] = field(default_factory=dict)
    normalization_config: Dict[str, Any] = field(default_factory=dict)
    augmentation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    """Result of preprocessing operation."""
    success: bool
    processed_episodes: List[Episode] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Preprocessor(ABC):
    """Abstract base class for preprocessors."""
    
    @abstractmethod
    def process(self, episodes: List[Episode], config: PreprocessingConfig) -> ProcessingResult:
        """Process episodes according to configuration."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Preprocessor name."""
        pass


class DataCleaner(Preprocessor):
    """Clean and fix data issues."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "data_cleaner"
    
    def process(self, episodes: List[Episode], config: PreprocessingConfig) -> ProcessingResult:
        """Clean data by fixing common issues."""
        result = ProcessingResult(success=True)
        cleaning_stats = {
            "nan_values_fixed": 0,
            "inf_values_fixed": 0,
            "duplicate_frames_removed": 0,
            "outliers_handled": 0,
            "timestamp_issues_fixed": 0,
        }
        
        cleaned_episodes = []
        
        for episode in episodes:
            try:
                cleaned_episode = self._clean_episode(episode, cleaning_stats)
                cleaned_episodes.append(cleaned_episode)
            except Exception as e:
                result.errors.append(f"Failed to clean episode {episode.episode_id}: {e}")
                self.logger.error(f"Cleaning failed for episode {episode.episode_id}: {e}")
        
        result.processed_episodes = cleaned_episodes
        result.statistics = cleaning_stats
        
        self.logger.info(f"Data cleaning complete: {len(cleaned_episodes)} episodes processed")
        return result
    
    def _clean_episode(self, episode: Episode, stats: Dict[str, int]) -> Episode:
        """Clean individual episode."""
        cleaned_frames = []
        last_timestamp = None
        
        for frame_idx, frame in enumerate(episode.load_frames()):
            cleaned_frame = self._clean_frame(frame, stats)
            
            # Fix timestamp issues
            if cleaned_frame.timestamp is not None:
                if last_timestamp is not None and cleaned_frame.timestamp <= last_timestamp:
                    # Fix backward or duplicate timestamps
                    cleaned_frame.timestamp = last_timestamp + 0.033  # Assume 30Hz
                    stats["timestamp_issues_fixed"] += 1
                last_timestamp = cleaned_frame.timestamp
            
            # Check for duplicate frames (based on action similarity)
            if cleaned_frames:
                prev_frame = cleaned_frames[-1]
                if self._frames_are_similar(cleaned_frame, prev_frame):
                    stats["duplicate_frames_removed"] += 1
                    continue  # Skip duplicate frame
            
            cleaned_frames.append(cleaned_frame)
        
        # Create cleaned episode
        return Episode(
            episode_id=episode.episode_id,
            metadata=episode.metadata,
            language_instruction=episode.language_instruction,
            frames=lambda: iter(cleaned_frames)
        )
    
    def _clean_frame(self, frame: Frame, stats: Dict[str, int]) -> Frame:
        """Clean individual frame."""
        # Clean action data
        if frame.action is not None:
            action_array = np.array(frame.action, dtype=np.float64)
            
            # Handle NaN values
            nan_mask = np.isnan(action_array)
            if np.any(nan_mask):
                # Replace with previous value or zeros
                action_array[nan_mask] = 0.0
                stats["nan_values_fixed"] += np.sum(nan_mask)
            
            # Handle infinite values
            inf_mask = np.isinf(action_array)
            if np.any(inf_mask):
                # Replace with reasonable bounds
                action_array[inf_mask] = np.sign(action_array[inf_mask]) * 10.0
                stats["inf_values_fixed"] += np.sum(inf_mask)
            
            # Handle outliers (simple clipping)
            if len(action_array) > 0:
                q1, q3 = np.percentile(action_array, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_mask = (action_array < lower_bound) | (action_array > upper_bound)
                if np.any(outlier_mask):
                    action_array[outlier_mask] = np.clip(action_array[outlier_mask], lower_bound, upper_bound)
                    stats["outliers_handled"] += np.sum(outlier_mask)
            
            frame.action = action_array
        
        # Clean state data
        if frame.state is not None:
            state_array = np.array(frame.state, dtype=np.float64)
            
            # Similar cleaning as actions
            nan_mask = np.isnan(state_array)
            if np.any(nan_mask):
                state_array[nan_mask] = 0.0
                stats["nan_values_fixed"] += np.sum(nan_mask)
            
            inf_mask = np.isinf(state_array)
            if np.any(inf_mask):
                state_array[inf_mask] = np.sign(state_array[inf_mask]) * 10.0
                stats["inf_values_fixed"] += np.sum(inf_mask)
            
            frame.state = state_array
        
        return frame
    
    def _frames_are_similar(self, frame1: Frame, frame2: Frame, threshold: float = 1e-6) -> bool:
        """Check if two frames are too similar (potential duplicates)."""
        if frame1.action is not None and frame2.action is not None:
            action_diff = np.linalg.norm(np.array(frame1.action) - np.array(frame2.action))
            if action_diff < threshold:
                return True
        
        if frame1.state is not None and frame2.state is not None:
            state_diff = np.linalg.norm(np.array(frame1.state) - np.array(frame2.state))
            if state_diff < threshold:
                return True
        
        return False


class DataNormalizer(Preprocessor):
    """Normalize and scale data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.action_stats = {}
        self.state_stats = {}
    
    @property
    def name(self) -> str:
        return "data_normalizer"
    
    def process(self, episodes: List[Episode], config: PreprocessingConfig) -> ProcessingResult:
        """Normalize data using various techniques."""
        result = ProcessingResult(success=True)
        
        # Compute normalization statistics
        self._compute_normalization_stats(episodes, config)
        
        normalization_stats = {
            "method": config.normalization_config.get("method", "standard"),
            "actions_normalized": len(self.action_stats),
            "states_normalized": len(self.state_stats),
        }
        
        normalized_episodes = []
        
        for episode in episodes:
            try:
                normalized_episode = self._normalize_episode(episode, config)
                normalized_episodes.append(normalized_episode)
            except Exception as e:
                result.errors.append(f"Failed to normalize episode {episode.episode_id}: {e}")
                self.logger.error(f"Normalization failed for episode {episode.episode_id}: {e}")
        
        result.processed_episodes = normalized_episodes
        result.statistics = normalization_stats
        
        self.logger.info(f"Data normalization complete: {len(normalized_episodes)} episodes processed")
        return result
    
    def _compute_normalization_stats(self, episodes: List[Episode], config: PreprocessingConfig):
        """Compute statistics for normalization."""
        method = config.normalization_config.get("method", "standard")
        
        # Collect all actions and states
        all_actions = []
        all_states = []
        
        for episode in episodes:
            for frame in episode.load_frames():
                if frame.action is not None:
                    all_actions.append(np.array(frame.action).flatten())
                if frame.state is not None:
                    all_states.append(np.array(frame.state).flatten())
        
        # Compute action statistics
        if all_actions:
            actions_matrix = np.array(all_actions)
            
            if method == "standard":
                self.action_stats = {
                    "mean": np.mean(actions_matrix, axis=0),
                    "std": np.std(actions_matrix, axis=0) + 1e-8,  # Avoid division by zero
                }
            elif method == "minmax":
                self.action_stats = {
                    "min": np.min(actions_matrix, axis=0),
                    "max": np.max(actions_matrix, axis=0),
                    "range": np.max(actions_matrix, axis=0) - np.min(actions_matrix, axis=0) + 1e-8,
                }
            elif method == "robust":
                self.action_stats = {
                    "median": np.median(actions_matrix, axis=0),
                    "mad": np.median(np.abs(actions_matrix - np.median(actions_matrix, axis=0)), axis=0) + 1e-8,
                }
        
        # Compute state statistics
        if all_states:
            states_matrix = np.array(all_states)
            
            if method == "standard":
                self.state_stats = {
                    "mean": np.mean(states_matrix, axis=0),
                    "std": np.std(states_matrix, axis=0) + 1e-8,
                }
            elif method == "minmax":
                self.state_stats = {
                    "min": np.min(states_matrix, axis=0),
                    "max": np.max(states_matrix, axis=0),
                    "range": np.max(states_matrix, axis=0) - np.min(states_matrix, axis=0) + 1e-8,
                }
            elif method == "robust":
                self.state_stats = {
                    "median": np.median(states_matrix, axis=0),
                    "mad": np.median(np.abs(states_matrix - np.median(states_matrix, axis=0)), axis=0) + 1e-8,
                }
    
    def _normalize_episode(self, episode: Episode, config: PreprocessingConfig) -> Episode:
        """Normalize individual episode."""
        method = config.normalization_config.get("method", "standard")
        
        normalized_frames = []
        
        for frame in episode.load_frames():
            normalized_frame = Frame(
                index=frame.index,
                timestamp=frame.timestamp,
                images=frame.images,
                is_first=frame.is_first,
                is_last=frame.is_last
            )
            
            # Normalize action
            if frame.action is not None and self.action_stats:
                action_array = np.array(frame.action, dtype=np.float64)
                normalized_frame.action = self._normalize_array(action_array, self.action_stats, method)
            
            # Normalize state
            if frame.state is not None and self.state_stats:
                state_array = np.array(frame.state, dtype=np.float64)
                normalized_frame.state = self._normalize_array(state_array, self.state_stats, method)
            
            normalized_frames.append(normalized_frame)
        
        return Episode(
            episode_id=episode.episode_id,
            metadata=episode.metadata,
            language_instruction=episode.language_instruction,
            frames=lambda: iter(normalized_frames)
        )
    
    def _normalize_array(self, array: np.ndarray, stats: Dict[str, np.ndarray], method: str) -> np.ndarray:
        """Normalize array using computed statistics."""
        if method == "standard":
            return (array - stats["mean"]) / stats["std"]
        elif method == "minmax":
            return (array - stats["min"]) / stats["range"]
        elif method == "robust":
            return (array - stats["median"]) / stats["mad"]
        else:
            return array  # No normalization


class DataAugmenter(Preprocessor):
    """Augment data to increase variety and improve training."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "data_augmenter"
    
    def process(self, episodes: List[Episode], config: PreprocessingConfig) -> ProcessingResult:
        """Augment data using various techniques."""
        result = ProcessingResult(success=True)
        
        augmentation_config = config.augmentation_config
        augmentation_factor = augmentation_config.get("factor", 2)
        augmentation_types = augmentation_config.get("types", [AugmentationType.NOISE_ADDITION])
        
        augmentation_stats = {
            "original_episodes": len(episodes),
            "augmented_episodes": 0,
            "augmentation_types": [at.value for at in augmentation_types],
            "augmentation_factor": augmentation_factor,
        }
        
        augmented_episodes = episodes.copy()  # Start with original episodes
        
        for episode in episodes:
            for i in range(augmentation_factor - 1):
                try:
                    augmented_episode = self._augment_episode(episode, augmentation_types, augmentation_config)
                    augmented_episode.episode_id = f"{episode.episode_id}_aug_{i}"
                    augmented_episodes.append(augmented_episode)
                    augmentation_stats["augmented_episodes"] += 1
                except Exception as e:
                    result.errors.append(f"Failed to augment episode {episode.episode_id}: {e}")
                    self.logger.error(f"Augmentation failed for episode {episode.episode_id}: {e}")
        
        result.processed_episodes = augmented_episodes
        result.statistics = augmentation_stats
        
        self.logger.info(f"Data augmentation complete: {len(augmented_episodes)} total episodes")
        return result
    
    def _augment_episode(self, episode: Episode, augmentation_types: List[AugmentationType], config: Dict[str, Any]) -> Episode:
        """Augment individual episode."""
        augmented_frames = []
        
        for frame in episode.load_frames():
            augmented_frame = self._augment_frame(frame, augmentation_types, config)
            augmented_frames.append(augmented_frame)
        
        return Episode(
            episode_id=episode.episode_id,
            metadata=episode.metadata,
            language_instruction=episode.language_instruction,
            frames=lambda: iter(augmented_frames)
        )
    
    def _augment_frame(self, frame: Frame, augmentation_types: List[AugmentationType], config: Dict[str, Any]) -> Frame:
        """Augment individual frame."""
        augmented_frame = Frame(
            index=frame.index,
            timestamp=frame.timestamp,
            images=frame.images.copy() if frame.images else {},
            is_first=frame.is_first,
            is_last=frame.is_last
        )
        
        for aug_type in augmentation_types:
            if aug_type == AugmentationType.NOISE_ADDITION:
                augmented_frame = self._add_noise(augmented_frame, config)
            elif aug_type == AugmentationType.TEMPORAL_WARPING:
                # This would need to be applied at episode level
                pass
            elif aug_type == AugmentationType.SPATIAL_TRANSFORMATION:
                augmented_frame = self._apply_spatial_transformation(augmented_frame, config)
            elif aug_type == AugmentationType.DROPOUT:
                augmented_frame = self._apply_dropout(augmented_frame, config)
        
        return augmented_frame
    
    def _add_noise(self, frame: Frame, config: Dict[str, Any]) -> Frame:
        """Add noise to actions and states."""
        noise_config = config.get("noise", {})
        action_noise_std = noise_config.get("action_std", 0.01)
        state_noise_std = noise_config.get("state_std", 0.01)
        
        if frame.action is not None:
            action_array = np.array(frame.action, dtype=np.float64)
            noise = np.random.normal(0, action_noise_std, action_array.shape)
            frame.action = action_array + noise
        
        if frame.state is not None:
            state_array = np.array(frame.state, dtype=np.float64)
            noise = np.random.normal(0, state_noise_std, state_array.shape)
            frame.state = state_array + noise
        
        return frame
    
    def _apply_spatial_transformation(self, frame: Frame, config: Dict[str, Any]) -> Frame:
        """Apply spatial transformations to images."""
        # This would implement image transformations
        # For now, just return the frame unchanged
        return frame
    
    def _apply_dropout(self, frame: Frame, config: Dict[str, Any]) -> Frame:
        """Apply dropout to actions and states."""
        dropout_config = config.get("dropout", {})
        action_dropout_rate = dropout_config.get("action_rate", 0.1)
        state_dropout_rate = dropout_config.get("state_rate", 0.1)
        
        if frame.action is not None:
            action_array = np.array(frame.action, dtype=np.float64)
            mask = np.random.random(action_array.shape) > action_dropout_rate
            action_array = action_array * mask
            frame.action = action_array
        
        if frame.state is not None:
            state_array = np.array(frame.state, dtype=np.float64)
            mask = np.random.random(state_array.shape) > state_dropout_rate
            state_array = state_array * mask
            frame.state = state_array
        
        return frame


class DataSampler(Preprocessor):
    """Sample and resample data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "data_sampler"
    
    def process(self, episodes: List[Episode], config: PreprocessingConfig) -> ProcessingResult:
        """Sample data according to configuration."""
        result = ProcessingResult(success=True)
        
        sampling_config = config.get("sampling_config", {})
        target_fps = sampling_config.get("target_fps")
        max_episodes = sampling_config.get("max_episodes")
        min_episode_length = sampling_config.get("min_episode_length")
        
        sampling_stats = {
            "original_episodes": len(episodes),
            "sampled_episodes": 0,
            "target_fps": target_fps,
            "frames_removed": 0,
        }
        
        # Filter episodes by length
        if min_episode_length:
            episodes = [ep for ep in episodes if len(list(ep.load_frames())) >= min_episode_length]
        
        # Limit number of episodes
        if max_episodes and len(episodes) > max_episodes:
            episodes = episodes[:max_episodes]
        
        sampled_episodes = []
        
        for episode in episodes:
            try:
                sampled_episode = self._sample_episode(episode, target_fps, sampling_stats)
                sampled_episodes.append(sampled_episode)
                sampling_stats["sampled_episodes"] += 1
            except Exception as e:
                result.errors.append(f"Failed to sample episode {episode.episode_id}: {e}")
                self.logger.error(f"Sampling failed for episode {episode.episode_id}: {e}")
        
        result.processed_episodes = sampled_episodes
        result.statistics = sampling_stats
        
        self.logger.info(f"Data sampling complete: {len(sampled_episodes)} episodes processed")
        return result
    
    def _sample_episode(self, episode: Episode, target_fps: Optional[float], stats: Dict[str, int]) -> Episode:
        """Sample individual episode."""
        frames = list(episode.load_frames())
        
        if target_fps is None:
            return episode  # No sampling needed
        
        # Calculate sampling interval
        if len(frames) < 2:
            return episode  # Cannot sample single frame
        
        timestamps = [f.timestamp for f in frames if f.timestamp is not None]
        if len(timestamps) < 2:
            return episode  # Cannot determine frame rate
        
        # Estimate current FPS
        duration = timestamps[-1] - timestamps[0]
        current_fps = (len(frames) - 1) / duration
        
        if current_fps <= target_fps:
            return episode  # No downsampling needed
        
        # Calculate sampling interval
        interval = max(1, int(current_fps / target_fps))
        
        # Sample frames
        sampled_frames = []
        for i in range(0, len(frames), interval):
            sampled_frames.append(frames[i])
        
        stats["frames_removed"] += len(frames) - len(sampled_frames)
        
        return Episode(
            episode_id=episode.episode_id,
            metadata=episode.metadata,
            language_instruction=episode.language_instruction,
            frames=lambda: iter(sampled_frames)
        )


class PreprocessingPipeline:
    """Main preprocessing pipeline that orchestrates all preprocessors."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.preprocessors: List[Preprocessor] = []
        self._setup_default_preprocessors()
    
    def _setup_default_preprocessors(self):
        """Setup default preprocessors."""
        self.preprocessors = [
            DataCleaner(),
            DataNormalizer(),
            DataSampler(),
        ]
    
    def add_preprocessor(self, preprocessor: Preprocessor):
        """Add a custom preprocessor."""
        self.preprocessors.append(preprocessor)
        self.logger.info(f"Added preprocessor: {preprocessor.name}")
    
    def remove_preprocessor(self, preprocessor_name: str):
        """Remove a preprocessor by name."""
        self.preprocessors = [p for p in self.preprocessors if p.name != preprocessor_name]
        self.logger.info(f"Removed preprocessor: {preprocessor_name}")
    
    def process(
        self,
        episodes: List[Episode],
        config: PreprocessingConfig
    ) -> ProcessingResult:
        """Process episodes through the pipeline."""
        self.logger.info(f"Starting preprocessing with {len(self.preprocessors)} preprocessors")
        
        result = ProcessingResult(success=True)
        current_episodes = episodes.copy()
        
        pipeline_stats = {
            "preprocessors_used": [],
            "episodes_at_each_stage": [len(current_episodes)],
            "total_processing_time": 0.0,
        }
        
        import time
        start_time = time.time()
        
        for preprocessor in self.preprocessors:
            self.logger.info(f"Running preprocessor: {preprocessor.name}")
            
            try:
                preprocessor_result = preprocessor.process(current_episodes, config)
                
                if not preprocessor_result.success:
                    result.errors.extend(preprocessor_result.errors)
                    self.logger.error(f"Preprocessor {preprocessor.name} failed")
                    continue
                
                current_episodes = preprocessor_result.processed_episodes
                pipeline_stats["preprocessors_used"].append(preprocessor.name)
                pipeline_stats["episodes_at_each_stage"].append(len(current_episodes))
                
                # Merge statistics
                result.statistics[preprocessor.name] = preprocessor_result.statistics
                
                self.logger.info(f"Preprocessor {preprocessor.name} complete: {len(current_episodes)} episodes")
                
            except Exception as e:
                error_msg = f"Preprocessor {preprocessor.name} crashed: {e}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
        
        end_time = time.time()
        pipeline_stats["total_processing_time"] = end_time - start_time
        
        result.processed_episodes = current_episodes
        result.metadata["pipeline_stats"] = pipeline_stats
        
        self.logger.info(f"Preprocessing complete: {len(current_episodes)} final episodes")
        return result


# Convenience functions
def preprocess_episodes(
    episodes: List[Episode],
    config: Optional[PreprocessingConfig] = None
) -> ProcessingResult:
    """Preprocess episodes with default configuration."""
    if config is None:
        config = PreprocessingConfig()
    
    pipeline = PreprocessingPipeline()
    return pipeline.process(episodes, config)


def clean_episodes(episodes: List[Episode]) -> ProcessingResult:
    """Clean episodes without other preprocessing."""
    config = PreprocessingConfig(enable_normalization=False, enable_augmentation=False)
    pipeline = PreprocessingPipeline()
    pipeline.preprocessors = [DataCleaner()]
    return pipeline.process(episodes, config)


def normalize_episodes(
    episodes: List[Episode],
    method: str = "standard"
) -> ProcessingResult:
    """Normalize episodes using specified method."""
    config = PreprocessingConfig(
        enable_cleaning=False,
        enable_augmentation=False,
        normalization_config={"method": method}
    )
    pipeline = PreprocessingPipeline()
    pipeline.preprocessors = [DataNormalizer()]
    return pipeline.process(episodes, config)


def augment_episodes(
    episodes: List[Episode],
    factor: int = 2,
    types: Optional[List[AugmentationType]] = None
) -> ProcessingResult:
    """Augment episodes to increase dataset size."""
    if types is None:
        types = [AugmentationType.NOISE_ADDITION]
    
    config = PreprocessingConfig(
        enable_cleaning=False,
        enable_normalization=False,
        enable_augmentation=True,
        augmentation_config={"factor": factor, "types": types}
    )
    pipeline = PreprocessingPipeline()
    pipeline.preprocessors = [DataAugmenter()]
    return pipeline.process(episodes, config)
